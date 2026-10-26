"""Qwen-Image 2.1 (comfy/ldm/qwen_image21/model.py):
- Prefix cache: the text and reference tokens' K/V are cached between sampling steps (one
  sampling run, one key). Training never resets it, so it kept a slot per pair and failed in
  `slots.remove` ("Boolean value of Tensor with more than one value is ambiguous", from comparing
  slots that hold tensors). Worse, a cached prefix would have been the LoRA of steps before, with
  no gradient through it. Off while training: every step computes its prefix.
- `_gated_residual` adds each block's branch into its input in place, which the norm before it
  saved for its backward and a checkpointed block recomputes from. Out of place in training.
"""

from __future__ import annotations

import logging

import torch

from .. import compat


def apply() -> None:
    if "qwen_image21" in compat._applied:
        return
    try:
        import comfy.ldm.qwen_image21.model as qi21
    except Exception:  # an older ComfyUI without Qwen-Image 2.1
        return
    import comfy.model_management as mm

    model = getattr(qi21, "QwenImage21Transformer2DModel", None)
    select = getattr(model, "select_prefix_cache", None)
    gated = getattr(qi21, "_gated_residual", None)
    if select is None or gated is None:
        return

    def select_prefix_cache(self, key, cache_bytes, device, options):
        if mm.in_training:
            return None, False  # recompute, as with the cache off
        return select(self, key, cache_bytes, device, options)

    def gated_residual(x, y, gate, prefix_len):
        if not mm.in_training:
            return gated(x, y, gate, prefix_len)
        g_prefix, g_target = gate
        if not prefix_len:
            return torch.addcmul(x, y, g_target)
        return torch.cat([torch.addcmul(x[:, :prefix_len], y[:, :prefix_len], g_prefix),
                          torch.addcmul(x[:, prefix_len:], y[:, prefix_len:], g_target)], dim=1)

    model.select_prefix_cache = select_prefix_cache
    qi21._gated_residual = gated_residual
    compat._applied.add("qwen_image21")
    logging.info("[ZenTrainer] Qwen-Image 2.1: prefix cache off and residuals out of place while training")
