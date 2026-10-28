"""torch.compile on the model's repeated transformer block (memory.compile_blocks): the block is
compiled once and reused for every layer, so the elementwise work around the big matmuls (norms,
modulation, the SwiGLU, residual gates, the LoRA add) fuses into a few kernels instead of dozens.
OneTrainer's second-biggest speed-up. Compiles each block's own forward (dynamic shapes, so
TREAD's changing token counts and new buckets don't recompile every step); graph breaks at
ComfyUI's weight streaming just fall back to eager for that part. Undone when the run ends: the
model is cached between runs.
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch

# memory.compile_blocks "auto": compile when a run is at least this long. Compiling costs about 40 s
# up front (Krea 2, 1024 px, RTX 5090) and then saves 0.2 s a step, so it pays back after about
# 200 steps; at 512 px it saves far less, and the margin keeps a short run from losing.
AUTO_STEPS = 500


def wanted(setting: bool | str, steps: int) -> tuple[bool, str]:
    """Whether to compile, and why when it wasn't simply what was asked. Offloaded weights are
    fine: compile was kept off with them for a while after an illegal memory access (Krea 2, 16 GB
    card, ~2 GB streaming), but the cause was the offloaded weights' stream-ordered frees, in eager
    too, since fixed (memory/offload.py guard_backward). Compiled with weights streaming and
    moving: MiniMax H3 (0.7-6.4 GB streaming) and Krea 2 (8.5-9.4 GB, and a 32 GB card held to 16)
    with Automagic3 + EMA and samples, 10 of 10 runs clean. It pays when little streams (H3, 0.7
    GB: 32% a step) and is even when most does (Krea 2, 9.4 of 12.9 GB: the copies set the pace).
    Weights that move recompile fresh graphs (reset_if_moved). "auto" also needs
    Triton, which inductor generates its kernels with (often missing on Windows, where it also
    needs a C++ compiler)."""
    if setting != "auto":
        return bool(setting), ""
    import importlib.util

    if os.name == "nt" or importlib.util.find_spec("triton") is None:
        return False, "not compiling (needs Triton; turn it on in Memory to try)"
    if steps < AUTO_STEPS:
        return False, f"not compiling a run under {AUTO_STEPS} steps (the compile would cost more than it saves)"
    return True, f"compiling: {steps} steps repay the first steps' compile"


def main_blocks(diffusion_model) -> list:
    """The model's main repeated blocks: of its outermost block lists, the one with the most
    weights. The others (Krea 2's text fusion and refiner blocks) run on a hundred or so caption
    tokens, so compiling them gains little, and every new caption length recompiles them: often
    enough that a checkpointed block's recompute ran eager where its forward ran compiled
    ("Recomputed values ... have different metadata", on those blocks only). They run eager."""
    lists = []

    def walk(m):
        for c in m.children():
            if isinstance(c, torch.nn.ModuleList) and len(c) > 1:
                lists.append(c)
            elif not isinstance(c, torch.nn.ModuleList):
                walk(c)

    walk(diffusion_model)
    if not lists:
        return []
    return list(max(lists, key=lambda l: sum(p.numel() for p in l.parameters())))


def _swiglu_out_of_place(m):
    """Krea 2's SwiGLU, `down(silu(gate(x)).mul_(up(x)))`, without the in-place mul_: dynamo
    can't trace that mul_ here ("does not know how to trace the builtin Tensor.mul_"), so every
    block's graph broke at its MLP and the MLP ran eager. Same values."""
    import torch.nn.functional as F

    def forward(x):
        return m.down(F.silu(m.gate(x)) * m.up(x))

    return forward


@contextlib.contextmanager
def compiled_blocks(diffusion_model, enabled: bool):
    if not enabled:
        yield 0
        return
    import torch._dynamo

    import torch._functorch.config as functorch_config

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    # the on-disk autograd cache needs a stable hash of every tensor: int8 weights (QuantizedTensor)
    # have none, so each compile warned and skipped it. Off while these blocks are compiled
    cache_was = getattr(functorch_config, "enable_autograd_cache", None)
    if cache_was is not None:
        functorch_config.enable_autograd_cache = False
    blocks = main_blocks(diffusion_model)
    mlps = [sub for b in blocks for sub in b.modules()
            if type(sub).__name__ == "SwiGLU" and all(hasattr(sub, k) for k in ("gate", "up", "down"))
            and "forward" not in sub.__dict__]
    for sub in mlps:
        sub.forward = _swiglu_out_of_place(sub)
    # ComfyUI's layers read comfy_cast_weights, a class attribute until a low-VRAM round trip
    # (samples on a card the model doesn't fit beside) sets it on the instance and puts it back
    # as one. The graphs guard on its absence from the instance, so that round trip invalidated
    # every graph after the run's first sample (a cluster of slow steps, every graph again).
    # Set on the instance now, the guard is on its value, which the round trip restores.
    casts = [sub for b in blocks for sub in b.modules()
             if hasattr(sub, "comfy_cast_weights") and "comfy_cast_weights" not in sub.__dict__]
    for sub in casts:
        sub.comfy_cast_weights = sub.comfy_cast_weights
    done = []
    for m in blocks:
        if "forward" in m.__dict__:  # already wrapped by someone else: leave it
            continue
        m.forward = torch.compile(m.forward, dynamic=True)
        done.append(m)
    logging.info("[ZenTrainer] torch.compile on the %d main blocks (the first steps compile)", len(done))
    try:
        yield len(done)
    finally:
        for m in done + mlps:
            m.__dict__.pop("forward", None)
        for sub in casts:
            # only where it's still the class's value: a layer ComfyUI left offloaded (True on the
            # instance) keeps it, or the cached model would stop casting that layer's weights
            if sub.__dict__.get("comfy_cast_weights") == type(sub).comfy_cast_weights:
                sub.__dict__.pop("comfy_cast_weights", None)
        if cache_was is not None:
            functorch_config.enable_autograd_cache = cache_was


def residency(diffusion_model) -> tuple[bool, ...]:
    """Which weights are on the GPU, to compare: compiled graphs guard on each parameter's
    device, so a change means recompiles."""
    return tuple(p.is_cuda for p in diffusion_model.parameters())


def reset_if_moved(diffusion_model, before: tuple[bool, ...] | None, why: str = "", step: int = 0) -> tuple[bool, ...]:
    """Drop the compiled graphs when weights moved between CPU and GPU since `before` (the VRAM
    fit loading more, a back-off, a reload after samples). Left alone, the next steps recompile
    beside the old graphs, and a checkpointed block's recompute can run a different one than its
    forward did: "Recomputed values ... have different metadata" (3 of 3 runs at the first fit,
    0 of 2 with the weights left in place). Fresh graphs compile like the first steps do: the
    log line says when and why, since every graph compiles again (a few seconds a graph with a
    warm cache, far more without), which is what slow steps after it are."""
    now = residency(diffusion_model)
    if before is not None and now != before:
        import torch._dynamo

        torch._dynamo.reset()
        on = sum(1 for a, b in zip(before, now) if b and not a)
        off = sum(1 for a, b in zip(before, now) if a and not b)
        logging.info("[ZenTrainer] step %d%s: %d weights moved onto the GPU and %d off it, so the compiled blocks "
                     "recompile", step, f" ({why})" if why else "", on, off)
    return now
