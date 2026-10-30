"""FLUX.1 / FLUX.2 (Dev, Klein) blocks (comfy/ldm/flux/layers.py):
- DoubleStreamBlock and SingleStreamBlock add their gated branches into the residual stream in place
  (img += ..., txt += ..., x += ...). The tensor they add into is the block's input, which the block
  before it (or its checkpoint) saved for backward: "one of the variables needed for gradient
  computation has been modified by an inplace operation ... output 0 of Add", at the first step. In
  training each block gets its own copy of the stream to add into; sampling is ComfyUI's as written.
"""

from __future__ import annotations

import torch

from .. import compat


def apply() -> None:
    if "flux" in compat._applied:
        return
    try:
        import comfy.ldm.flux.layers as layers
    except ModuleNotFoundError as e:  # an older ComfyUI without FLUX; anything else is raised
        if e.name not in ("comfy.ldm.flux", "comfy.ldm.flux.layers"):
            raise
        return
    import comfy.model_management as mm

    double, single = getattr(layers, "DoubleStreamBlock", None), getattr(layers, "SingleStreamBlock", None)
    if double is None or single is None:
        return
    double_forward, single_forward = double.forward, single.forward
    training = lambda: mm.in_training and torch.is_grad_enabled()

    # called by name (img=, txt=) or by position: both taken
    def double_block(self, img, txt, *args, **kwargs):
        if training():
            img, txt = img.clone(), txt.clone()
        return double_forward(self, img, txt, *args, **kwargs)

    def single_block(self, x, *args, **kwargs):
        if training():
            x = x.clone()
        return single_forward(self, x, *args, **kwargs)

    double.forward, single.forward = double_block, single_block
    compat._applied.add("flux")
