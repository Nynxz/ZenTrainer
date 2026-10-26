"""Qwen-Image 2.1 (comfy/ldm/qwen_image21/model.py):
- Prefix cache: the text and reference tokens' K/V are cached between sampling steps (one
  sampling run, one key). Training never resets it, so it kept a slot per pair and failed in
  `slots.remove` ("Boolean value of Tensor with more than one value is ambiguous", from comparing
  slots that hold tensors). Worse, a cached prefix would have been the LoRA of steps before, with
  no gradient through it. Off while training: every step computes its prefix.
- `_gated_residual` adds each block's branch into its input in place, which the norm before it
  saved for its backward and a checkpointed block recomputes from. Out of place in training.
- The cache's `select` (PoseBranchCache, comfy/ldm/wan/model_animate2.py) moves the slot it found
  to the end with `slots.remove(s)`, which compares slots, and so their tensors, by value. With two
  slots, as sampling at CFG > 1 makes when the prompts' lengths differ (each pass its own key), that
  raises ("The size of tensor a (434187) must match the size of tensor b (425995)"). The slot is
  removed by identity instead, which is what remove meant. The one fix here that isn't for training
  only: it fails ComfyUI's own sampling the same way, and changes nothing else.
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
    _slots_by_identity()
    compat._applied.add("qwen_image21")
    logging.info("[ZenTrainer] Qwen-Image 2.1: prefix cache off and residuals out of place while training")


class _Slots(list):
    """A cache's slot list whose remove takes out that very slot, never comparing its tensors."""

    def remove(self, slot) -> None:
        for i, s in enumerate(self):
            if s is slot:
                del self[i]
                return
        raise ValueError("slot not in the cache")


def _slots_by_identity() -> None:
    try:
        from comfy.ldm.wan.model_animate2 import PoseBranchCache
    except Exception:
        return
    select = getattr(PoseBranchCache, "select", None)
    if select is None or getattr(select, "_zentrainer", False):
        return

    def by_identity(self, *args, **kwargs):
        if type(getattr(self, "slots", None)) is list:  # (free() and __init__ make plain lists)
            self.slots = _Slots(self.slots)
        return select(self, *args, **kwargs)

    by_identity._zentrainer = True
    PoseBranchCache.select = by_identity
