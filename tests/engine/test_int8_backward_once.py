"""The int8 ConvRot backward patch (harness/kernels/int8_backward.py) on a layer that isn't int8 (fp8,
nvfp4...): it reads the layer's saved tensors to tell, and ComfyUI's backward then gets those, not a
second read. A checkpointed layer's can be unpacked only once ("Unpack is being triggered for a
tensor that was already unpacked once"): FLUX.2 Klein's fp8 build failed at its first step."""

import sys
import types

import torch

from zentrainer.harness.kernels import int8_backward as ib


class OnceCtx:
    """ctx as torch.utils.checkpoint leaves it: its saved tensors read once, the second read raises."""

    def __init__(self, saved):
        self._saved, self._reads = saved, 0
        self.fp8_bwd, self.weight_requires_grad, self.compute_dtype = False, False, torch.float32

    @property
    def saved_tensors(self):
        self._reads += 1
        if self._reads > 1:
            raise RuntimeError("Unpack is being triggered for a tensor that was already unpacked once")
        return self._saved


def test_comfyuis_backward_gets_the_tensors_already_read(monkeypatch):
    seen = []

    class QuantLinearFunc:
        @staticmethod
        def backward(ctx, grad_output):  # ComfyUI's: reads the saved tensors and ctx's fields
            x, w = ctx.saved_tensors
            seen.append((x, w, ctx.compute_dtype))
            return grad_output @ w, None, None, None, None, None

    ops = types.ModuleType("comfy.ops")
    ops.QuantLinearFunc = QuantLinearFunc
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = True
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ops", ops), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(ib, "_patched", False)
    monkeypatch.setattr(ib, "_mode", "w8a8")
    ib._patch()
    x, w = torch.randn(2, 3), torch.randn(3, 4)  # a plain weight: not int8 ConvRot, so ComfyUI's backward
    g = QuantLinearFunc.backward(OnceCtx((x, w)), torch.randn(2, 3))[0]
    assert seen and seen[0][0] is x and seen[0][1] is w and seen[0][2] is torch.float32
    assert g.shape == (2, 4)
