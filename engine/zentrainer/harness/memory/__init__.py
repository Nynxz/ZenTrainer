"""Gradient checkpointing and offload-aware recompute.

UPSTREAM SEAM: `patch`/`unpatch` (and the OffloadCheckpointFunction they use) are
private functions in comfy_extras/nodes_train.py. This is
the only module that reaches into them. The proposal is to move them to a public
`comfy/train/` module that the native node and ZenTrainer both import.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


def block_lists(model, depth: int = 1) -> list:
    """The repeated blocks to checkpoint: children of the outermost nn.ModuleLists
    (blocks, double_blocks, single_blocks, input_blocks, ...). depth=2 goes one list
    deeper (e.g. the transformer_blocks inside SD's input_blocks) for finer segments."""
    import torch.nn as nn

    found: list = []

    def walk(m, level: int) -> None:
        for child in m.children():
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                if level == depth:
                    found.extend(child)
                else:
                    for block in child:
                        walk(block, level + 1)
            elif not isinstance(child, nn.ModuleList):
                walk(child, level)

    walk(model, 1)
    return found


# memory.checkpoint_save "attention": a checkpointed block keeps attention's outputs (and its
# softmax statistics) instead of running attention's forward again in the backward; the rest is
# recomputed as before. Measured on an RTX 5090 at 1024 px, compiled, nothing kept un-checkpointed:
# Krea 2 1.08 -> 1.01 s a step for 1.9 GB more, MOSS 0.60 -> 0.51 for 1.0 GB (keeping 10 whole
# Krea 2 blocks bought the same 6% for 6 GB). The same losses: the step-one loss is the eager run's
# to the last digit uncompiled, and compiled within the rounding compiling itself changes.
_SAVE_OPS = ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
             "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu")


def _keep_attention():
    import torch
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    save = {getattr(torch.ops.aten, n).default for n in _SAVE_OPS if hasattr(torch.ops.aten, n)}

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in save else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


# A block whose attention is this sharp keeps nothing: kept attention saves the softmax statistics
# from the forward and pairs them with q and k recomputed in the backward, which compiled code
# rounds a little differently (other kernels): harmless on ordinary logits, but Krea 2's block 0
# (q and k norm weights up to 51.9, logits to ~31,000) turned that rounding into gradients of
# 1e8-1e17 and NaN, intermittently. The largest q-norm weight times the largest k-norm weight:
# Krea 2's block 0 2690, then 21; MOSS's sharpest blocks 55 and 42 (and those train as uncompiled).
SHARP_QK = 500.0


def _qk_peak(m) -> float:
    """The largest q-norm weight times the largest k-norm weight in block m (0 without them)."""
    peak = {"q": 0.0, "k": 0.0}
    for name, sub in m.named_modules():
        leaf = name.rsplit(".", 1)[-1].replace("_", "")
        side = "q" if leaf in ("qnorm", "normq", "querynorm") else "k" if leaf in ("knorm", "normk", "keynorm") else None
        if side is None:
            continue
        for p in sub.parameters(recurse=False):
            if p.ndim == 1:
                peak[side] = max(peak[side], float(p.detach().abs().max()))
    return peak["q"] * peak["k"]


def _unpatch(m, comfy_unpatch) -> None:
    """Undo checkpoint patching without leaving a plain bound class method in `m.__dict__`.
    ComfyUI's `unpatch` assigns `m.forward = m.org_forward` on the instance, which makes a later
    run's `compiled_blocks` skip the cached model's blocks."""
    comfy_unpatch(m)
    fwd = m.__dict__.get("forward")
    if getattr(fwd, "__func__", None) is type(m).forward and getattr(fwd, "__self__", None) is m:
        m.__dict__.pop("forward", None)


def _patch_checkpoint(m, save_attention: bool) -> None:
    """Checkpoint block `m` with or without `_keep_attention`. When `m` uses a shared compiled
    class method (kernels/compile.py), compile ONE shared wrapper around the checkpoint per class
    so AOTAutograd sees the checkpoint boundary (avoiding dead saved-activation writes in forward
    and fusing recompute into backward) and all identical blocks share one compiled graph."""
    import types

    import torch

    from ..kernels.compile import _ensure_grad

    org = m.forward
    m.org_forward = org  # what unpatch puts back
    org_func = getattr(org, "__func__", None)
    orig_cls_fwd = getattr(org_func, "_zt_orig_cls_fwd", None)
    if orig_cls_fwd is not None:
        attr = "_zt_ckpt_attn_compiled" if save_attention else "_zt_ckpt_none_compiled"
        wrapper = getattr(org_func, attr, None)
        if wrapper is None:
            if save_attention:
                def ckpt_cls_run(self, *args, _fn=orig_cls_fwd, **kwargs):
                    return torch.utils.checkpoint.checkpoint(_fn, self, *args, use_reentrant=False,
                                                             context_fn=_keep_attention, **kwargs)
            else:
                def ckpt_cls_run(self, *args, _fn=orig_cls_fwd, **kwargs):
                    return torch.utils.checkpoint.checkpoint(_fn, self, *args, use_reentrant=False, **kwargs)
            compiled = torch.compile(ckpt_cls_run, dynamic=True)

            def wrapper(self, *args, _fn=compiled, **kwargs):
                args, kwargs = _ensure_grad(self, args, kwargs)
                return _fn(self, *args, **kwargs)

            setattr(org_func, attr, wrapper)
        m.forward = types.MethodType(wrapper, m)
        return

    inner = getattr(org, "_torchdynamo_orig_callable", None)

    def checkpointed(fn):
        if save_attention:
            def run(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False,
                                                         context_fn=_keep_attention, **kwargs)
        else:
            def run(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return run

    m.forward = torch.compile(checkpointed(inner), dynamic=True) if inner is not None else checkpointed(org)


def _patch_saving(m) -> None:
    """A block checkpointed with _keep_attention (kept as its own entry point for tests)."""
    _patch_checkpoint(m, save_attention=True)


def _patch_offload_block(m) -> None:
    """Non-reentrant block checkpoint for offloaded models: skips guard_backward's weight clones
    during the forward pass (saved tensors are discarded) and caches streamed int8/packed weights
    across the block's recompute -> backward so they cross PCIe twice a step instead of three times."""
    import torch

    from .offload import recompute_scope, skip_guard

    org = m.forward

    def checkpointing_fwd(*args, **kwargs):
        recomputing = [False]

        def run(a, kw):
            if recomputing[0]:
                with recompute_scope():
                    return org(*a, **kw)
            recomputing[0] = True
            with skip_guard():
                return org(*a, **kw)

        return torch.utils.checkpoint.checkpoint(run, args, kwargs, use_reentrant=False)

    m.org_forward = org
    m.forward = checkpointing_fwd


@contextmanager
def gradient_checkpointing(diffusion_model, enabled: bool, depth: int, offloading: bool, skip: int = 0,
                           save: str = "none") -> Iterator[int]:
    """Recompute each block's activations in backward instead of keeping them. Yields
    how many blocks are checkpointed.

    ComfyUI's native helper (find_modules_at_depth) counts the model itself as depth 1,
    so "depth 1" there wraps the whole network as one segment: backward then rebuilds
    every activation at once and saves nothing. Checkpointing per block is what bounds
    memory to roughly one block's activations. `skip` leaves that many of the biggest blocks
    (the last first) as they are: keeping their activations, no recompute. `save` "attention": the
    checkpointed blocks keep attention's outputs (see _SAVE_OPS)."""
    import types

    import torch.nn as nn

    if not enabled:
        yield types.SimpleNamespace(blocks=0, input_grad=False, all=[], kept=0, allow_keep=False,
                                    saving_attention=False, set_save_attention=lambda on: False, keep=lambda n: 0)
        return
    from comfy_extras.nodes_train import patch as comfy_patch, unpatch

    can_save = hasattr(__import__("torch").utils.checkpoint, "create_selective_checkpoint_contexts")
    if save == "attention" and not can_save:
        # selective checkpointing is torch 2.4's: an older ComfyUI's torch checkpoints whole blocks
        logging.info("[ZenTrainer] checkpoints keep nothing, not attention: this torch has no selective checkpointing (2.4+)")
        save = "none"

    every = block_lists(diffusion_model, max(1, depth))
    if not every:  # no repeated blocks (unusual): fall back to the top-level children
        every = [c for c in diffusion_model.children()]
    # which blocks stay un-checkpointed first: main denoising blocks first (last of them first),
    # then other lists by parameter count. Putting main_blocks first prevents caption-only blocks
    # of equal parameter count (e.g. MOSS's 2 TextBlocks) from taking the probe slots and measuring
    # 0.00 GB activation cost.
    from ..kernels.compile import main_blocks

    main_ids = {id(b) for b in main_blocks(diffusion_model)}
    size = [sum(p.numel() for p in b.parameters()) for b in every]
    order = sorted(range(len(every) - 1, -1, -1), key=lambda i: (0 if id(every[i]) in main_ids else 1, -size[i]))
    state = types.SimpleNamespace(blocks=0, input_grad=False, all=every, kept=0, order=order,
                                  allow_keep=not offloading, saving_attention=save == "attention" and not offloading)
    sharp: set[int] = set()

    def patch_block(i: int, m) -> None:
        if state.saving_attention:
            if i not in sharp and (peak := _qk_peak(m)) > SHARP_QK:
                sharp.add(i)
                name = next((n for n, x in diffusion_model.named_modules() if x is m), f"block {i}")
                logging.info("[ZenTrainer] %s keeps nothing, not attention: its attention is too sharp for it "
                             "(q x k norm weights %.0f)", name, peak)
            if i not in sharp:
                _patch_saving(m)
            else:
                comfy_patch(m, offloading=False)
        elif offloading and not isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            _patch_offload_block(m)
        elif not offloading and getattr(getattr(m.forward, "__func__", None), "_zt_orig_cls_fwd", None) is not None:
            _patch_checkpoint(m, save_attention=False)
        else:
            comfy_patch(m, offloading=offloading)

    def keep(n: int) -> int:
        """The n blocks of `order` un-checkpointed, the rest checkpointed; between steps too
        (memory.grad_checkpoint_skip "auto" moves it as it measures). Blocked while weights are
        offloaded (kept blocks' weights would stream too); enabled automatically once _tighten()
        fits the whole model onto the GPU."""
        n = min(max(0, n), len(every))
        if state.blocks and (n == state.kept or (n > 0 and not state.allow_keep)):
            return state.kept
        kept = set(order[:n])
        for i, m in enumerate(every):
            if i not in kept and not hasattr(m, "org_forward"):
                patch_block(i, m)
            elif i in kept and hasattr(m, "org_forward"):
                _unpatch(m, unpatch)
        state.kept, state.blocks = n, len(every) - n
        return n

    def set_save_attention(on: bool) -> bool:
        """Toggle selective attention saving on currently checkpointed blocks between steps:
        turned on once _tighten() fits 100% of weights onto the GPU, and off first on OOM."""
        on = bool(on)
        if on and not can_save:
            return False
        if on == state.saving_attention:
            return False
        state.saving_attention = on
        for i, m in enumerate(every):
            if hasattr(m, "org_forward"):
                _unpatch(m, unpatch)
                patch_block(i, m)
        return True

    keep(skip)
    blocks = [m for m in every if hasattr(m, "org_forward")]
    logging.info("[ZenTrainer] gradient checkpointing on %d blocks (depth %d%s%s)", len(blocks), depth,
                 f", {state.kept} kept" if state.kept else "",
                 ", attention kept" if state.saving_attention else "")
    state.keep = keep
    state.set_save_attention = set_save_attention
    if offloading:
        # An offloaded layer's weight lives in a streaming buffer that the next layers reuse, so
        # autograd can't keep it for backward ("modified by an inplace operation" on step one).
        # Layers outside the checkpointed blocks (Krea 2's txtmlp, the final layer, the kept
        # last blocks) recompute too, then: torch's checkpoint, which unlike the offload one
        # also works when the input needs no grad. Any leaf with a weight: ComfyUI's quantized
        # Linear (MixedPrecisionOps) is a plain nn.Module, not an nn.Linear.
        import torch

        inside = {id(sub) for b in blocks for sub in b.modules()}
        loose = [m for m in diffusion_model.modules()
                 if id(m) not in inside and next(m.children(), None) is None
                 and isinstance(getattr(m, "weight", None), torch.Tensor)]
        for m in loose:
            comfy_patch(m, offloading=False)
        if loose:
            logging.info("[ZenTrainer] %d layers outside the checkpointed blocks recompute too (offloaded weights)", len(loose))
    # ComfyUI's per-layer offload checkpoint (a custom autograd Function, used when a patched
    # module is itself a Linear/Conv) only records when its input requires grad; block-level
    # checkpointing is torch's non-reentrant one, which doesn't. Without that need, the input
    # stays grad-free and the backward stops at the first trained layer.
    state.input_grad = offloading and any(isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in blocks)
    try:
        yield state
    finally:
        for m in diffusion_model.modules():
            _unpatch(m, unpatch)


# Attention backends with a backward pass. The rest ComfyUI registers are forward-only custom ops
# (flash, sage, sage3, comfy_kitchen_int8), and comfy_kitchen_int8 also refuses the fp32 q/k that
# autocast's qk-norms hand it beside a bf16 v ("q, k, and v must have the same dtype"). An allow-list,
# so a backend ComfyUI adds later isn't offered for training until it's known to train.
TRAINABLE_ATTENTION = ("pytorch", "xformers", "sub_quad", "split")


def attention_override(backend: str):
    """transformer_options entry routing every attention call to `backend`, so a server launched
    with --use-sage-attention (or --use-ck-attention) still trains. A backend that can't train (a
    preset or an older run) falls back to pytorch."""
    if not backend or backend == "default":
        return {}
    from comfy.ldm.modules.attention import get_attention_function

    import torch

    if backend not in TRAINABLE_ATTENTION:
        logging.warning("[ZenTrainer] attention %r has no backward pass: training with pytorch", backend)
        backend = "pytorch"
    fn = get_attention_function(backend)

    def override(_func, *args, **kwargs):
        # q/k leave the qk-norms in fp32 (the trainer keeps norm weights fp32) while v skips them
        # and stays bf16; SDPA needs one dtype (on ROCm it raises), so q/k go to v's, the model's
        q, k, v = args[:3] if len(args) >= 3 else (None, None, None)
        if isinstance(v, torch.Tensor) and isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) \
                and (q.dtype != v.dtype or k.dtype != v.dtype):
            args = (q.to(v.dtype), k.to(v.dtype), v, *args[3:])
        return fn(*args, **kwargs)

    return {"optimized_attention_override": override}
