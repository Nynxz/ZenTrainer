"""A faster backward for ComfyUI's int8 ConvRot layers while training (memory.int8_backward).

ComfyUI's QuantLinearFunc keeps the int8 forward fast but, in backward, dequantizes the whole
weight to bf16 for every frozen layer on every step just to get the input gradient. For a ConvRot
weight, W = (q * s) @ blockdiag(H^T) (int8 q, a scale per output row s, a normalised Hadamard H
over groups of input columns), so

    grad_in = grad_out @ W = ((grad_out * s) @ q) @ blockdiag(H^T)

Two ways to take the first product:

- "fused" (memory.int8_backward "fused"): a Triton kernel (kernels/w8a16_backward.py) that never
  builds the bf16 weight. Still 16-bit math: saves memory, not time.
- "w8a8" (memory.int8_backward "w8a8", the default): the gradient quantized to int8 per row too, and the product on
  int8 tensor cores (torch._int_mm), as OneTrainer does. Measured on H3's shapes at clip size
  (RTX 5090): 6.0 ms for the four GEMMs against 16.8 in bf16. cuBLAS wants the weight
  column-major for that, so each call takes a transposed copy (int8: a quarter of a bf16 one).
  Costs some gradient precision (an int8 step of each row's largest value), rounded
  stochastically so it stays unbiased.

The rotation after either is a small per-group matmul. Training only; anything else (other
layouts, an fp8 backward, a weight that trains, no Triton) and any error falls back to
ComfyUI's own path.
"""

from __future__ import annotations

import logging

import torch

import collections

_mode = ""  # "", "fused" or "w8a8"
stats: collections.Counter = collections.Counter()  # which path each backward took (see profile)
_patched = False
_warned = False
_flagged: list = []  # (layer, its _full_precision_mm before weight_only)


def enable(mode: str | bool) -> None:
    """Pick the backward ("" = ComfyUI's), patching ComfyUI's QuantLinearFunc once."""
    global _mode
    _mode = "fused" if mode is True else (mode or "")
    if _mode:
        _patch()


def _hadamard(size: int, device, dtype) -> torch.Tensor:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    return _build_hadamard(size, device=device, dtype=dtype)


def input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """grad_out @ dequantized ConvRot weight, without dequantizing it. grad_2d (M, out) bf16."""
    from .w8a16_backward import fused_w8a16_input_grad

    # fp32 out of the kernel and through the rotation, rounded once at the end like the eager path
    g = fused_w8a16_input_grad(grad_2d.to(torch.bfloat16), q, scale, out_dtype=torch.float32)  # (M, in), rotated
    if group:
        m, k = g.shape
        h = _hadamard(group, g.device, torch.float32)
        g = torch.matmul(g.view(m, k // group, group), h.T).view(m, k)
    return g.to(torch.bfloat16)


def w8a8_input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor | None:
    """grad_out @ dequantized ConvRot weight on int8 tensor cores: (grad_out * s) quantized per
    row, times q, rescaled, then rotated. None when the shapes don't suit _int_mm."""
    m, n = grad_2d.shape
    k = q.shape[1]
    if m <= 16 or n % 8 or k % 8:
        return None
    try:
        from .w8a8_backward import TRITON_AVAILABLE, int8_matmul, quantize_rows
    except Exception:
        TRITON_AVAILABLE = False
    if TRITON_AVAILABLE:  # two fused kernels (kernels/w8a8_backward.py), then the rotation in bf16
        seed = torch.randint(0, 2**31 - 1, (1,), device=grad_2d.device, dtype=torch.int32)  # no .item(): compile-safe
        gq, step = quantize_rows(grad_2d, scale, seed)
        g = int8_matmul(gq, q, step)
        if group:
            h = _hadamard(group, g.device, torch.bfloat16)
            g = torch.matmul(g.view(m, k // group, group), h.T).view(m, k)
        return g
    gs = grad_2d.float() * scale.reshape(1, -1).float()
    step = gs.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127.0
    # stochastic rounding: each value lands on a neighbouring int8 step with the odds that keep
    # its expectation exact, so the rounding error averages out over steps instead of biasing
    gq = gs.div_(step).add_(torch.rand_like(gs)).floor_().clamp_(-127, 127).to(torch.int8)
    del gs
    acc = torch._int_mm(gq, q.t().contiguous().t())  # (m, k) int32; the weight column-major
    g = acc.float().mul_(step)
    del acc
    if group:
        h = _hadamard(group, g.device, torch.float32)
        g = torch.matmul(g.view(m, k // group, group), h.T).view(m, k)
    return g.to(torch.bfloat16)


def _patch() -> None:
    global _patched
    if _patched:
        return
    try:
        import comfy.model_management as mm
        import comfy.ops as ops
    except Exception:
        return
    func = getattr(ops, "QuantLinearFunc", None)
    if func is None:
        return
    original = func.backward

    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        global _warned
        if not (_mode and mm.in_training and not ctx.fp8_bwd and not ctx.weight_requires_grad):
            stats["off: " + ("no mode" if not _mode else "not training" if not mm.in_training
                             else "fp8 backward" if ctx.fp8_bwd else "weight trains")] += 1
        if _mode and mm.in_training and not ctx.fp8_bwd and not ctx.weight_requires_grad:
            try:
                _input, weight = ctx.saved_tensors
                params = getattr(weight, "_params", None)
                why = ("not quantized" if params is None else "not convrot" if not getattr(params, "convrot", False)
                       else "no int8 data" if getattr(weight, "_qdata", None) is None or weight._qdata.dtype != torch.int8
                       else "cpu" if not grad_output.is_cuda else "")
                stats[why or _mode] += 1
                if params is not None and getattr(params, "convrot", False) and getattr(weight, "_qdata", None) is not None \
                        and weight._qdata.dtype == torch.int8 and grad_output.is_cuda:
                    grad_2d = grad_output.flatten(0, -2)
                    group = int(params.convrot_groupsize)
                    gi = w8a8_input_grad(grad_2d, weight._qdata, params.scale, group) if _mode == "w8a8" else None
                    if gi is None:
                        gi = input_grad(grad_2d, weight._qdata, params.scale, group)
                    gi = gi.to(ctx.compute_dtype)
                    if len(ctx.input_shape) > 2:
                        gi = gi.unflatten(0, ctx.input_shape[:-1])
                    gb = grad_2d.to(ctx.compute_dtype).sum(dim=0) if ctx.has_bias else None
                    return gi, None, gb, None, None, None
            except Exception:
                if not _warned:
                    _warned = True
                    logging.warning("[ZenTrainer] %s int8 backward failed; using ComfyUI's", _mode, exc_info=True)
        return original(ctx, grad_output)

    func.backward = staticmethod(backward)
    _patched = True
    logging.info("[ZenTrainer] int8 ConvRot backward: %s", _mode)


# ---- weight-only int8 layers (ComfyUI's int8_tensorwise: quantize_input False) --------------
# ComfyUI never routes these through QuantLinearFunc: a layer whose input needs a gradient runs
# F.linear on the QuantizedTensor under autograd, which decomposes to matmul, so the weight is
# dequantized and both the forward and the input-gradient matmul run in bf16 (profiled on H3 int8:
# every trained layer). Only grad-free layers hit the int8 kernel. While training, these layers
# get their own autograd Function instead: the int8 kernel forward, and the input gradient from
# the W8A8 kernels (mode "w8a8") or a dequantized matmul.


class _Int8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, m, original):
        # the layer's own forward without a gradient: ComfyUI's int8 path (a plain F.linear on the
        # QuantizedTensor decomposes to a dequantized bf16 matmul instead), weight streaming included
        with torch.no_grad():
            out = original(x.detach())
        ctx.m = m
        ctx.in_shape = x.shape
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        from comfy.ops import CastBiasWeightContext

        m = ctx.m
        g = grad_out.reshape(-1, grad_out.shape[-1])
        with CastBiasWeightContext(m, grad_out, offloadable=True, compute_dtype=grad_out.dtype, want_requant=False) as (w, _b):
            gi = None
            if type(w).__name__ == "QuantizedTensor":
                params = w._params
                group = int(getattr(params, "convrot_groupsize", 0) or 0) if getattr(params, "convrot", False) else 0
                if _mode == "w8a8" and g.is_cuda:
                    try:
                        gi = w8a8_input_grad(g, w._qdata, params.scale, group)
                    except Exception:
                        gi = None
                stats["int8 layer: w8a8" if gi is not None else "int8 layer: dequantized"] += 1
                if gi is None:
                    gi = torch.mm(g.to(torch.bfloat16), w.dequantize().to(torch.bfloat16))
            else:
                stats["int8 layer: plain weight"] += 1
                gi = torch.mm(g, w.to(g.dtype))
        ctx.m = None
        return gi.to(grad_out.dtype).reshape(ctx.in_shape), None, None


_installed: list = []


def _int8_layer(m) -> bool:
    w = getattr(m, "weight", None)
    return (getattr(m, "quant_format", None) == "int8_tensorwise" and getattr(m, "layout_type", None) is not None
            and type(w).__name__ == "QuantizedTensor" and hasattr(m, "forward_comfy_cast_weights"))


def install(model) -> int:
    """Give the model's weight-only int8 layers the training path above (undone by uninstall)."""
    import comfy.model_management as mm

    n = 0
    for m in model.modules():
        if not _int8_layer(m) or "forward" in m.__dict__:
            continue
        original = m.forward

        def forward(x, *args, _m=m, _orig=original, **kwargs):
            if not (mm.in_training and torch.is_grad_enabled() and x.requires_grad) or args or kwargs \
                    or len(_m.weight_function) or len(_m.bias_function):
                return _orig(x, *args, **kwargs)
            return _Int8Linear.apply(x, _m, _orig)

        m.forward = forward
        _installed.append(m)
        n += 1
    return n


def weight_only(model) -> int:
    """memory.int8_forward off: the model's int8 layers dequantize to bf16 for every matmul. ComfyUI
    otherwise quantizes the input and multiplies in int8 whenever a layer trains (its
    QuantLinearFunc), so without this "off" would change nothing. Undone by uninstall."""
    n = 0
    for m in model.modules():
        if _int8_layer(m):
            _flagged.append((m, m._full_precision_mm))
            m._full_precision_mm = True
            n += 1
    return n


def uninstall() -> None:
    while _installed:
        _installed.pop().__dict__.pop("forward", None)
    while _flagged:
        m, was = _flagged.pop()
        m._full_precision_mm = was
