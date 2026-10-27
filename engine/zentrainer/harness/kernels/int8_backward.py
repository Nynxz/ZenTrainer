"""A faster backward for ComfyUI's int8 ConvRot layers while training (memory.int8_backward).

ComfyUI's QuantLinearFunc keeps the int8 forward fast but, in backward, dequantizes the whole
weight to bf16 for every frozen layer on every step just to get the input gradient. For a ConvRot
weight, W = (q * s) @ blockdiag(H^T) (int8 q, a scale per output row s, a normalised Hadamard H
over groups of input columns), so

    grad_in = grad_out @ W = ((grad_out * s) @ q) @ blockdiag(H^T)

Two ways to take the first product:

- "fused" (memory.int8_backward "fused"): a Triton kernel (kernels/w8a16_backward.py) that never
  builds the bf16 weight. Still 16-bit math: saves memory, not time.
- "w8a8" (memory.int8_backward "w8a8", the default): the gradient quantized to int8 per row too,
  and the product on int8 tensor cores, as OneTrainer does. On NVIDIA cards from compute
  capability 8.0, two raw CUDA kernels with the rotation inside (kernels/w8a8_cuda.py: 27-32% less
  time than Triton's matmul and rotation on Krea 2's and MOSS's shapes, RTX 5090); else Triton's
  (kernels/w8a8_backward.py); else torch._int_mm, which wants the weight column-major, so each
  call takes a transposed copy (int8: a quarter of a bf16 one). Measured on H3's shapes at clip
  size (RTX 5090, Triton): 6.0 ms for the four GEMMs against 16.8 in bf16. Costs some gradient
  precision (an int8 step of each row's largest value), rounded stochastically so it stays
  unbiased.

Outside the CUDA kernels the rotation after either is a small per-group matmul. Training only;
anything else (other layouts, an fp8 backward, a weight that trains, no Triton) and any error
falls back to ComfyUI's own path.
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


def _count(path: str) -> None:
    """Which path a layer took, for the profile command. Not while dynamo traces: a compiled,
    checkpointed block can't mutate state outside it (the whole block fell back to eager)."""
    if not torch.compiler.is_compiling():
        stats[path] += 1


def enable(mode: str | bool) -> None:
    """Pick the backward ("" = ComfyUI's), patching ComfyUI's QuantLinearFunc once."""
    global _mode
    _mode = "fused" if mode is True else (mode or "")
    if _mode:
        _patch()
    if _mode == "w8a8" and torch.cuda.is_available():
        from . import w8a8_cuda

        w8a8_cuda._load(torch.device("cuda", torch.cuda.current_device()))  # now: a compiled block's trace can't


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


def w8a8_input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int,
                    residual: torch.Tensor | None = None) -> torch.Tensor | None:
    """grad_out @ dequantized ConvRot weight on int8 tensor cores: (grad_out * s) quantized per
    row, times q, rescaled, then rotated, plus `residual` (M, in) when given (in the CUDA kernel's
    epilogue). None when the shapes don't suit _int_mm."""
    m, n = grad_2d.shape
    k = q.shape[1]
    if m <= 16 or n % 8 or k % 8:
        return None
    from . import w8a8_cuda

    if w8a8_cuda.supported(grad_2d, q, group):  # raw CUDA, the rotation inside (kernels/w8a8_cuda.py)
        _count("w8a8: cuda")
        seed = torch.randint(0, 2**31 - 1, (1,), device=grad_2d.device, dtype=torch.int32)
        return w8a8_cuda.input_grad(grad_2d, q, scale, seed, group == 256, residual)
    g = _w8a8_input_grad(grad_2d, q, scale, group)
    return g if residual is None else g + residual


def _w8a8_input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    m, n = grad_2d.shape
    k = q.shape[1]
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
                _count(why or _mode)
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
        gi = _layer_input_grad(ctx.m, grad_out.reshape(-1, grad_out.shape[-1]))
        ctx.m = None
        return gi.to(grad_out.dtype).reshape(ctx.in_shape), None, None


def _layer_input_grad(m, g: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
    """A frozen int8 layer's input gradient for g (M, out), plus `residual` (M, in) when given."""
    from comfy.ops import CastBiasWeightContext

    with CastBiasWeightContext(m, g, offloadable=True, compute_dtype=g.dtype, want_requant=False) as (w, _b):
        gi = None
        if type(w).__name__ == "QuantizedTensor":
            params = w._params
            group = int(getattr(params, "convrot_groupsize", 0) or 0) if getattr(params, "convrot", False) else 0
            if _mode == "w8a8" and g.is_cuda:
                try:
                    gi = w8a8_input_grad(g, w._qdata, params.scale, group, residual)
                except Exception:
                    gi = None
            _count("int8 layer: w8a8" if gi is not None else "int8 layer: dequantized")
            if gi is not None:
                return gi
            gi = torch.mm(g.to(torch.bfloat16), w.dequantize().to(torch.bfloat16))
        else:
            _count("int8 layer: plain weight")
            gi = torch.mm(g, w.to(g.dtype))
    return gi if residual is None else gi + residual


class _LoRAInt8(torch.autograd.Function):
    """A frozen int8 layer and the LoRA beside it (bypass) as one op. Apart, the LoRA's output is
    added to the layer's in a pass of its own, scaled in another, and in the backward its input
    gradient is added to the layer's in a third, each over the whole activation (Krea 2 at 1024 px:
    ~70 ms a step of 1.03 s). Here the LoRA's output is the int8 GEMM's residual (comfy_kitchen's
    CUTLASS epilogue, as ComfyUI fuses a block's residual outside training), the scale is folded
    into the up weight, and the backward computes the LoRA's input gradient first, as the residual
    of the int8 input gradient (the CUDA kernel's epilogue). Same maths as base(x) + up(down(x)) *
    scale under autocast: the LoRA's matmuls in the activation's dtype, its gradients returned in
    its weights' own."""

    @staticmethod
    def forward(ctx, x, down, up, scale, m, original):
        x2 = x.detach().reshape(-1, x.shape[-1])
        a, b = down.detach().to(x.dtype), (up.detach() * scale).to(x.dtype)  # (rank, in), (out, rank)
        with torch.no_grad():
            z = x2 @ a.t()
            h = z @ b.t()
            out = _int8_forward(m, x2, h)
            if out is None:  # the layer can't take the residual: as the bypass does it
                _count("lora + int8: unfused")
                out = original(x2) + h  # the layer's own forward (no grad here)
        ctx.save_for_backward(x, z, a, b)  # x itself: a view of it saved breaks compiled blocks' tracing
        ctx.m, ctx.scale, ctx.in_shape, ctx.dtypes = m, scale, x.shape, (down.dtype, up.dtype)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, z, a, b = ctx.saved_tensors
        x2 = x.detach().reshape(-1, x.shape[-1])
        g = grad_out.reshape(-1, grad_out.shape[-1]).to(x2.dtype)
        gz = g @ b  # (M, rank): the gradient at the LoRA's middle, its scale in b
        d_up = (g.t() @ z) * ctx.scale
        d_down = gz.t() @ x2
        gi = _layer_input_grad(ctx.m, g, gz @ a)
        ctx.m = None
        return (gi.to(grad_out.dtype).reshape(ctx.in_shape), d_down.to(ctx.dtypes[0]), d_up.to(ctx.dtypes[1]),
                None, None, None)


def _int8_forward(m, x2: torch.Tensor, residual: torch.Tensor) -> torch.Tensor | None:
    """The int8 layer's forward with `residual` added in the GEMM's epilogue (ComfyUI's
    QuantizedTensor.linear_input_act, which outside training it uses for a block's residual), or
    None where the layer or this ComfyUI can't."""
    from comfy.ops import CastBiasWeightContext

    with CastBiasWeightContext(m, x2, offloadable=True, compute_dtype=x2.dtype, want_requant=True) as (w, bias):
        fused = getattr(w, "linear_input_act", None) if type(w).__name__ == "QuantizedTensor" else None
        if fused is None:
            return None
        ones = torch.ones(residual.shape[-1], device=x2.device, dtype=x2.dtype)
        out = fused(x2, bias, residual=residual, residual_scale=ones)
    if out is NotImplemented:
        return None
    _count("lora + int8: fused")
    return out


class LoRABeside(torch.nn.Module):
    """How a plain LoRA attaches (bypass) to a frozen int8 layer while training: as _LoRAInt8.
    Otherwise (sampling, no grad, a layer with weight functions, int8_forward off) the bypass's
    own way, f(x) + h(x). Owns no parameters of its own: the LoRA's stay where they're saved from."""

    _BYPASS_ATTRS = ("multiplier", "is_conv", "conv_dim", "kernel_size", "in_channels", "out_channels", "kw_dict")

    def __init__(self, lora, layer):
        super().__init__()
        self.lora = lora
        self.__dict__["layer"] = layer  # not a submodule: its weights aren't the adapter's
        self._alpha = float(lora.alpha)  # read once: a Parameter on the GPU would sync every call

    def _share(self) -> None:
        for a in self._BYPASS_ATTRS:
            if hasattr(self, a):
                setattr(self.lora, a, getattr(self, a))

    def h(self, x, base_out):
        self._share()
        return self.lora.h(x, base_out)

    def g(self, y):
        return y

    def __call__(self, w):  # the weight-wrapper protocol, which bypass never uses
        return self.lora(w)

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        import comfy.model_management as mm

        m = self.layer
        if (args or kwargs or not (mm.in_training and torch.is_grad_enabled()) or not x.is_cuda
                or x.dtype not in (torch.bfloat16, torch.float16) or getattr(m, "_full_precision_mm", False)
                or len(m.weight_function) or len(m.bias_function)):
            base = org_forward(x, *args, **kwargs)
            return base + self.h(x, base)
        scale = self._alpha / self.lora.rank * getattr(self, "multiplier", 1.0)
        return _LoRAInt8.apply(x, self.lora.lora_down.weight, self.lora.lora_up.weight, scale, m, org_forward)


def lora_beside(adapter, layer):
    """The adapter to attach to `layer` in bypass: LoRABeside for a plain LoRA on an int8 layer."""
    if (type(adapter).__name__ == "LoraDiff" and getattr(adapter, "lora_mid", None) is None
            and _int8_layer(layer) and _mode):
        return LoRABeside(adapter, layer)
    return adapter


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
