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


@contextmanager
def gradient_checkpointing(diffusion_model, enabled: bool, depth: int, offloading: bool, skip: int = 0) -> Iterator[int]:
    """Recompute each block's activations in backward instead of keeping them. Yields
    how many blocks are checkpointed.

    ComfyUI's native helper (find_modules_at_depth) counts the model itself as depth 1,
    so "depth 1" there wraps the whole network as one segment: backward then rebuilds
    every activation at once and saves nothing. Checkpointing per block is what bounds
    memory to roughly one block's activations. `skip` leaves that many of the biggest blocks
    (the last first) as they are: keeping their activations, no recompute."""
    import types

    import torch.nn as nn

    if not enabled:
        yield types.SimpleNamespace(blocks=0, input_grad=False, all=[], kept=0, keep=lambda n: 0)
        return
    from comfy_extras.nodes_train import patch, unpatch

    every = block_lists(diffusion_model, max(1, depth))
    if not every:  # no repeated blocks (unusual): fall back to the top-level children
        every = [c for c in diffusion_model.children()]
    # which blocks stay un-checkpointed first: the biggest, the last of them first. By order alone
    # Krea 2's last blocks are its four small text-fusion ones, which run on the caption only
    size = [sum(p.numel() for p in b.parameters()) for b in every]
    order = sorted(range(len(every) - 1, -1, -1), key=lambda i: -size[i])
    state = types.SimpleNamespace(blocks=0, input_grad=False, all=every, kept=0)

    def keep(n: int) -> int:
        """The n blocks of `order` un-checkpointed, the rest checkpointed; between steps too
        (memory.grad_checkpoint_skip "auto" moves it as it measures). Fixed with offloading: kept
        blocks' weights would stream too, and the loose layers patched below assume the start's."""
        n = min(max(0, n), len(every))
        if state.blocks and (offloading or n == state.kept):
            return state.kept
        kept = set(order[:n])
        for i, m in enumerate(every):
            if i not in kept and not hasattr(m, "org_forward"):
                patch(m, offloading=offloading)
            elif i in kept and hasattr(m, "org_forward"):
                unpatch(m)
        state.kept, state.blocks = n, len(every) - n
        return n

    keep(skip)
    blocks = [m for m in every if hasattr(m, "org_forward")]
    logging.info("[ZenTrainer] gradient checkpointing on %d blocks (depth %d%s)", len(blocks), depth,
                 f", {state.kept} kept" if state.kept else "")
    state.keep = keep
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
            patch(m, offloading=False)
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
            unpatch(m)


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
