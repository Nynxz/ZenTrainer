"""Fixes for ComfyUI model code that runs fine for sampling but can't train, applied at runtime
from here so the pinned ComfyUI checkout stays untouched: one module per model, after ComfyUI's
comfy/ldm/<model>/. Each patch only changes what runs while `comfy.model_management.in_training`
is set: sampling keeps ComfyUI's fused kernels. Each is the change worth sending upstream; drop
its module here once ComfyUI has it.

tests/comfy trains each patched model through a real ComfyUI and checks the patched model
computes what ComfyUI's code does.
"""

from __future__ import annotations

import logging

import torch

_applied: set[str] = set()
fused_qk_rope = False  # memory.fused_qk_rope, set by the trainer before each run (read at call time)


def apply(qk_rope: bool = False) -> None:
    global fused_qk_rope
    fused_qk_rope = qk_rope
    from . import flux3, krea2, minimax_h3, moss, qwen_image21

    for model in (minimax_h3, flux3, krea2, moss, qwen_image21):
        model.apply()
    if qk_rope:
        logging.info("[ZenTrainer] fused q/k norm + rope while training, for Krea 2, MOSS and MiniMax H3 (memory.fused_qk_rope)")


def prune_unused(patcher) -> int:
    """Drop weights the training forward never touches, so they don't take VRAM a run could
    keep the model in. MiniMax H3 int8 builds carry the sparse-attention (VSA) gate in every
    block, ~1.9 GB the dense forward never calls; None is ComfyUI's own "no gate" state.
    Returns the bytes freed."""
    import comfy.model_management as mm

    freed = 0
    for m in patcher.model.modules():
        gate = getattr(m, "to_gate_compress", None)
        if isinstance(gate, torch.nn.Module):
            freed += mm.module_size(gate)
            m.to_gate_compress = None
    if freed:
        patcher.size = 0  # recount
        logging.info("[ZenTrainer] dropped %.1f GB of weights training never uses (VSA gates)", freed / 1024 ** 3)
    return freed
