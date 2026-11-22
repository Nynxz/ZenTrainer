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

Packed formats (asym_w4a8_int8, w6a8_int8: comfy_kitchen's AsymW4A8Int8Layout, 4/6-bit codes with
group scales and a codebook) have no input-gradient kernel above: the qdata isn't an int8 matrix,
so in "w8a8" the codes are decoded, every backward, to the int8 grid the format's own forward
feeds its GEMM (round(clamp(level * s_rel)), in the rotated basis), and that goes to the W8A8
kernels with the per-row s_channel as the scale: the same shape as a ConvRot int8 weight. Otherwise
(and for a layer the decode doesn't cover) the backward dequantizes the weight and takes one bf16
matmul against it. That is
also the fix for ComfyUI's own path on these layers, which fails outright: a weight-only quantized
layer runs F.linear on the QuantizedTensor, and autograd's linear backward hands `mm(grad, W)` a
non-transposed RHS, which the w4a8 mm handler refuses ("W4A8 GEMM expects RHS W.T.").
"""

from __future__ import annotations

import logging

import torch

import collections
import contextlib

from ..network import _PassesOn

_mode = ""  # "", "fused" or "w8a8"
_packed_w8a8 = False  # packed 4/6-bit layers decoded to int8 for the w8a8 kernels (off if the first check fails)
_packed_checked: set = set()  # the kinds of packed layer (format, GEMV-packed) whose first gradient was checked
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
    """Pick the backward ("" = ComfyUI's), patching ComfyUI's QuantLinearFunc once. Each run starts
    clean: a fallback one run took (an out-of-memory moment in a fused forward, say) isn't kept."""
    global _mode, _packed_w8a8, _packed_checked, _fuse_failed_once
    _mode = "fused" if mode is True else (mode or "")
    _packed_w8a8 = _mode == "w8a8"
    _packed_checked = set()
    _fuse_failed_once = False
    _failed_once.clear()
    _decoder.clear()
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
    return _w8a8_input_grad(grad_2d, q, scale, group, residual)


def _w8a8_input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int,
                     residual: torch.Tensor | None = None) -> torch.Tensor:
    m, n = grad_2d.shape
    k = q.shape[1]
    try:
        from .w8a8_backward import TRITON_AVAILABLE, hadamard256_residual, int8_matmul, quantize_rows
    except Exception:
        TRITON_AVAILABLE = False
    if TRITON_AVAILABLE:  # fused Triton kernels (kernels/w8a8_backward.py), ROCm and CUDA fallback
        seed = torch.randint(0, 2**31 - 1, (1,), device=grad_2d.device, dtype=torch.int32)  # no .item(): compile-safe
        gq, step = quantize_rows(grad_2d, scale, seed)
        if group == 256 and k % 256 == 0:
            g = int8_matmul(gq, q, step)
            return hadamard256_residual(g, residual)
        if not group:
            return int8_matmul(gq, q, step, residual)
        g = int8_matmul(gq, q, step)
        h = _hadamard(group, g.device, torch.bfloat16)
        g = torch.matmul(g.view(m, k // group, group), h.T).view(m, k)
        return g if residual is None else g + residual.to(g.dtype)
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
    g = g.to(torch.bfloat16)
    return g if residual is None else g + residual.to(g.dtype)


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

    class _Unpacked:
        """ctx with its saved tensors already read: a checkpointed layer's can be unpacked only once
        ("Unpack is being triggered for a tensor that was already unpacked once"), so ComfyUI's
        backward gets the ones read here to check the layer (an fp8 or nvfp4 one, say, that isn't int8)."""

        def __init__(self, ctx, saved):
            self._ctx, self.saved_tensors = ctx, saved

        def __getattr__(self, name):
            return getattr(self._ctx, name)

    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        global _warned
        saved = None
        if not (_mode and mm.in_training and not ctx.fp8_bwd and not ctx.weight_requires_grad):
            stats["off: " + ("no mode" if not _mode else "not training" if not mm.in_training
                             else "fp8 backward" if ctx.fp8_bwd else "weight trains")] += 1
        if _mode and mm.in_training and not ctx.fp8_bwd and not ctx.weight_requires_grad:
            try:
                saved = ctx.saved_tensors
                _input, weight = saved
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
        return original(ctx if saved is None else _Unpacked(ctx, saved), grad_output)

    func.backward = staticmethod(backward)
    _patched = True
    logging.info("[ZenTrainer] int8 ConvRot backward: %s", _mode)


# ---- weight-only quantized layers (ComfyUI's int8_tensorwise and the packed 4/6-bit formats:
# asym_w4a8_int8, w6a8_int8) ---------------------------------------------------------------
# ComfyUI never routes these through QuantLinearFunc: a layer whose input needs a gradient runs
# F.linear on the QuantizedTensor under autograd, which decomposes to matmul, so the weight is
# dequantized and both the forward and the input-gradient matmul run in bf16 (profiled on H3 int8:
# every trained layer). Only grad-free layers hit the int8 kernel. While training, these layers
# get their own autograd Function instead: the int8 kernel forward, and the input gradient from
# the W8A8 kernels (mode "w8a8": an int8_tensorwise qdata as it is, a packed layer's codes decoded to
# that grid first) or a dequantized matmul (every other mode, and a packed layer the decode can't take).

INT8_FORMATS = ("int8_tensorwise",)  # the W8A8 kernels' layout: one scale per output row, int8 qdata
PACKED_FORMATS = ("asym_w4a8_int8", "w6a8_int8")  # 4/6-bit codes: decoded to int8 for the W8A8 kernels


def quant_format(m) -> str | None:
    """The layer's ComfyUI quantization format when this module has a training path for it."""
    fmt = getattr(m, "quant_format", None)
    return fmt if fmt in INT8_FORMATS + PACKED_FORMATS else None


def has_packed(model) -> bool:
    """Whether any of the model's layers is a packed 4/6-bit one."""
    return any(quant_format(m) in PACKED_FORMATS for m in model.modules())


def packed_dequantized(model, dtype: torch.dtype) -> bool:
    """Whether ComfyUI dequantizes the model's packed layers for every matmul when training in `dtype`:
    it does for a weight asked for in a dtype other than its own (comfy.ops.cast_bias_weight), e.g.
    fp16 training on a bf16 file."""
    return any(quant_format(m) in PACKED_FORMATS and getattr(m.weight, "dtype", dtype) != dtype for m in model.modules())


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


# programming errors propagate; anything else from the w8a8 path (NVRTC, Triton, a launch) falls back
_BUGS = (TypeError, AttributeError, NameError, AssertionError, IndexError, KeyError, ValueError)
_failed_once: set = set()  # "int8", "packed": which kind of layer's w8a8 failure was reported


def _kernel_failed(e: Exception, kind: str = "int8") -> None:
    if kind not in _failed_once:
        _failed_once.add(kind)
        logging.warning("[ZenTrainer] the w8a8 input gradient failed (%s: %s): %s layers use the dequantized "
                        "matmul instead", type(e).__name__, e, kind)


def _packed_failed(e: Exception) -> None:
    """A packed decode or kernel this backend refuses: packed layers dequantize for the rest of the run,
    rather than trying it again on every layer of every step."""
    global _packed_w8a8
    _packed_w8a8 = False
    _kernel_failed(e, "packed")


_fuse_failed_once = False


def _fuse_failed(e: Exception) -> None:
    global _fuse_failed_once
    if not _fuse_failed_once:
        _fuse_failed_once = True
        logging.warning("[ZenTrainer] a packed format's fused LoRA+GEMM forward failed (%s: %s): those adapters run "
                        "unfused instead", type(e).__name__, e)


def _decoders(on_gpu: bool) -> list:
    """comfy_kitchen's packed -> int8 decoders for a weight on the GPU or not, fastest first: the native
    one (hip on ROCm, cuda otherwise), Triton, eager (the only one for a CPU tensor). Bit-exact with
    each other; private functions there."""
    from comfy_kitchen.backends.eager.w4a8_int8 import _dequant_int4_grouped_to_int8 as eager

    if not on_gpu:
        return [eager]
    found = []
    try:
        if torch.version.hip:
            from comfy_kitchen.backends import hip as native
        else:
            from comfy_kitchen.backends import cuda as native
    except ImportError:
        native = None
    # the module imports without its extension (it sets _C to None), so ask whether it loaded
    if native is not None and getattr(native, "_EXT_AVAILABLE", False):
        found.append(native._dequant_int4_grouped_to_int8)
    try:
        from comfy_kitchen.backends.triton.w4a8_int8 import _dequant_int4_grouped_to_int8 as tri

        found.append(tri)
    except ImportError:
        pass
    return found + [eager]


_decoder: dict = {}  # GPU or not: the first of _decoders() that decoded a layer here


def _decode_to_int8(qdata: torch.Tensor, s_rel: torch.Tensor, codebook, group_size: int) -> torch.Tensor:
    """Packed 4/6-bit codes -> the int8 grid round(clamp(level(code) * s_rel, -127, 127)), (N, K):
    what the format's forward feeds its INT8 GEMM."""
    fn = _decoder.get(qdata.is_cuda)
    if fn is not None:
        return fn(qdata, s_rel, codebook, group_size)
    err = None
    for fn in _decoders(qdata.is_cuda):
        try:
            out = fn(qdata, s_rel, codebook, group_size)
        except torch.OutOfMemoryError:  # the card was full, not this backend unable: the trainer handles it
            raise
        except RuntimeError as e:  # a backend that imports but can't launch on this device or build: the next one
            err = e
            continue
        _decoder[qdata.is_cuda] = fn
        logging.info("[ZenTrainer] packed w8a8 backward: int8 decode by %s", fn.__module__)
        return out
    raise err


def _packed_codes(qdata: torch.Tensor, s_rel: torch.Tensor, mma_packed: bool, n: int, k: int):
    """A packed weight's codes and group scales as they're stored: a resident one ComfyUI repacked for
    small-M MMA (optimize_for_gemv) is unpacked first, as comfy_kitchen's own dequantize does."""
    if not mma_packed:
        return qdata, s_rel
    from comfy_kitchen.tensor.w4a8_packed import unpack_w4a8_mma_weight

    return unpack_w4a8_mma_weight(qdata, n, k)


def _dequantize_packed(qdata, s_rel, s_channel, codebook, correction, group_size: int, convrot_groupsize: int):
    from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight

    return dequantize_w4a8_int8_weight(qdata, s_rel, s_channel, codebook=codebook, correction=correction,
                                       group_size=group_size, convrot_groupsize=convrot_groupsize,
                                       output_dtype=torch.bfloat16)


def _check_packed(dequantized, g: torch.Tensor, gi: torch.Tensor, residual: torch.Tensor | None, kind: tuple) -> torch.Tensor:
    """Once per kind of packed layer, on the first: its w8a8 input gradient against the dequantized bf16 matmul's.
    Logs the relative difference (int8 gradient noise: about a percent). A real mismatch (a decode or
    rotation convention that changed in comfy_kitchen) turns the mode off and returns the bf16 one."""
    global _packed_w8a8
    _packed_checked.add(kind)
    ref = torch.mm(g.to(torch.bfloat16), dequantized())
    if residual is not None:
        ref = ref + residual.to(ref.dtype)
    rel = ((gi.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
    if rel > 0.15:
        _packed_w8a8 = False
        logging.warning("[ZenTrainer] packed w8a8 backward: gradient differs from the bf16 one by %.1f%% on the "
                        "first %s layer; turned off, packed layers use the dequantized matmul", rel * 100, kind[0])
        return ref
    logging.info("[ZenTrainer] packed w8a8 backward check: first %s layer within %.2f%% of the bf16 gradient",
                 kind[0], rel * 100)
    return gi


@torch.library.custom_op("zentrainer::packed_input_grad", mutates_args=())
def packed_input_grad(g: torch.Tensor, qdata: torch.Tensor, s_rel: torch.Tensor, s_channel: torch.Tensor,
                      codebook: torch.Tensor | None, correction: torch.Tensor | None, group_size: int,
                      convrot_groupsize: int, mma_packed: bool, n: int, k: int, fmt: str,
                      residual: torch.Tensor | None = None) -> torch.Tensor:
    """A packed W4A8/W6A8 layer's input gradient for g (M, out), plus `residual` (M, in): bf16 (M, in).
    In "w8a8" the codes are decoded to the int8 grid the format's own forward uses, W = (q * s_channel)
    @ blockdiag(H^T) as an int8 ConvRot weight is, and go through the same kernels; otherwise, and
    for what the decode doesn't cover (an asymmetric weight's `correction` term), the weight is
    dequantized. One opaque op, so a compiled block's trace records it whole and the decode, its
    backend and the first-layer check run on the real tensors every step."""
    qdata, s_rel = _packed_codes(qdata, s_rel, mma_packed, n, k)

    def dequantized():
        return _dequantize_packed(qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize)

    gi, decoded = None, False
    if _mode == "w8a8" and _packed_w8a8 and g.is_cuda and correction is None:
        try:
            q = _decode_to_int8(qdata, s_rel, codebook, group_size)
        except torch.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as e:  # comfy_kitchen refusing this layer or device: dequantize
            _packed_failed(e)  # (a TypeError or AttributeError, a changed signature, propagates)
            q = None
        if q is not None:
            try:
                gi = w8a8_input_grad(g, q, s_channel, convrot_groupsize, residual)
            except (*_BUGS, torch.OutOfMemoryError):
                raise
            except Exception as e:
                _packed_failed(e)
                gi = None
            del q
            kind = (fmt, mma_packed)
            if gi is not None and kind not in _packed_checked:
                checked = _check_packed(dequantized, g, gi, residual, kind)
                gi, decoded = checked, checked is gi  # a rejected check hands back the bf16 gradient
            else:
                decoded = gi is not None
    _count(f"{fmt}: w8a8 (decoded)" if decoded else f"{fmt}: dequantized")
    if gi is None:
        gi = torch.mm(g.to(torch.bfloat16), dequantized())
        if residual is not None:
            gi = gi + residual.to(gi.dtype)
    return gi.to(torch.bfloat16)  # what the fake declares: a compiled graph is built on it


@packed_input_grad.register_fake
def _(g, qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize, mma_packed, n, k, fmt,
      residual=None):
    return g.new_empty((g.shape[0], k), dtype=torch.bfloat16)


def _weight_input_grad(w, fmt: str | None, g: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
    """Compute g @ W (+ residual) for a resident/casted weight `w`."""
    gi = None
    if type(w).__name__ == "QuantizedTensor":
        params = w._params
        if fmt in PACKED_FORMATS and not getattr(params, "transposed", False):
            n, k = params.orig_shape
            return packed_input_grad(g, w._qdata, params.scale, params.s_channel, params.codebook,
                                     params.correction, int(params.group_size), int(params.convrot_groupsize),
                                     bool(params.mma_packed), n, k, fmt, residual)
        group = int(getattr(params, "convrot_groupsize", 0) or 0) if getattr(params, "convrot", False) else 0
        if _mode == "w8a8" and g.is_cuda and fmt in INT8_FORMATS:
            try:
                gi = w8a8_input_grad(g, w._qdata, params.scale, group, residual)
            except _BUGS:  # a mistake in the code, not the kernel: don't hide it behind the fallback
                raise
            except Exception as e:  # a kernel that fails to compile or launch here: dequantize
                _kernel_failed(e)
                gi = None
        _count(("int8 layer: w8a8" if gi is not None else "int8 layer: dequantized") if fmt in INT8_FORMATS
               else f"{fmt}: dequantized")
        if gi is not None:
            return gi
        gi = torch.mm(g.to(torch.bfloat16), w.dequantize().to(torch.bfloat16))
    else:
        _count("int8 layer: plain weight")
        gi = torch.mm(g, w.to(g.dtype))
    return gi if residual is None else gi + residual


def _layer_input_grad(m, g: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
    """A frozen quantized layer's input gradient for g (M, out), plus `residual` (M, in) when given."""
    from comfy.ops import CastBiasWeightContext
    from ..memory.offload import pop_recompute_weight

    fmt = quant_format(m)
    cached = pop_recompute_weight(m)
    if cached is not None:
        return _weight_input_grad(cached, fmt, g, residual)
    with CastBiasWeightContext(m, g, offloadable=True, compute_dtype=g.dtype, want_requant=False) as (w, _b):
        return _weight_input_grad(w, fmt, g, residual)


# ---- the packed forward under torch.compile -----------------------------------------------------------
# comfy_kitchen's w4a8 kernels read their tensors' raw data pointers, which a compiled trace's fake
# tensors don't have ("Cannot access data pointer of Tensor"); its int8 one is an opaque op and traces.
# While compiled blocks train a packed base, every caller of comfy_kitchen's w4a8_int8_linear (ComfyUI's
# forward, the fused LoRA one, samples) goes through the op below instead: the same call, made on the
# real tensors.

_ck_w4a8_linear = None  # comfy_kitchen's own, while opaque_packed() routes it


@torch.library.custom_op("zentrainer::packed_linear", mutates_args=())
def packed_linear(x: torch.Tensor, qdata: torch.Tensor, s_rel: torch.Tensor, s_channel: torch.Tensor,
                codebook: torch.Tensor | None, correction: torch.Tensor | None, bias: torch.Tensor | None,
                group_size: int, convrot_groupsize: int, out_dtype: torch.dtype, mma_packed: bool,
                input_act: str | None, input_act_weight: torch.Tensor | None, input_act_eps: float,
                residual: torch.Tensor | None, residual_scale: torch.Tensor | None) -> torch.Tensor:
    """comfy_kitchen's w4a8_int8_linear, which runs both packed formats (w4a8 and w6a8 share its layout
    and kernels), as one opaque op. A residual its backend can't add in the GEMM's
    epilogue (_int8_forward's fused LoRA forward, a packed format on a backend without it) is added
    here after the unfused GEMM, the same sum: compiled, _int8_forward's own fallback only ran while
    tracing, so a backend refusing it at run time stopped the step."""
    def linear(residual, residual_scale):
        return _ck_w4a8_linear(x, qdata, s_rel, s_channel, codebook=codebook, correction=correction, bias=bias,
                               group_size=group_size, convrot_groupsize=convrot_groupsize, out_dtype=out_dtype,
                               mma_packed=mma_packed, input_act=input_act, input_act_weight=input_act_weight,
                               input_act_eps=input_act_eps, residual=residual, residual_scale=residual_scale)

    if residual is None or not _fuse_failed_once:
        try:
            return linear(residual, residual_scale)
        except (*_BUGS, torch.OutOfMemoryError):
            raise
        except Exception as e:
            if residual is None:
                raise
            _fuse_failed(e)
    out = linear(None, None)
    return torch.addcmul(residual.to(out.dtype), out, residual_scale.to(out.dtype))


@packed_linear.register_fake
def _(x, qdata, s_rel, s_channel, codebook, correction, bias, group_size, convrot_groupsize, out_dtype,
      mma_packed, input_act, input_act_weight, input_act_eps, residual, residual_scale):
    return x.new_empty((*x.shape[:-1], s_channel.shape[0]), dtype=out_dtype)


def _packed_linear_routed(x, qdata, s_rel, s_channel, codebook=None, correction=None, bias=None, group_size=16,
                        convrot_groupsize=256, out_dtype=torch.bfloat16, mma_packed=False, input_act=None,
                        input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None):
    return packed_linear(x, qdata, s_rel, s_channel, codebook, correction, bias, group_size, convrot_groupsize,
                       out_dtype, mma_packed, input_act, input_act_weight, float(input_act_eps), residual,
                       residual_scale)


@contextlib.contextmanager
def opaque_packed(on: bool):
    """Route comfy_kitchen's packed forward through the opaque op while `on` (compiled blocks, a packed base)."""
    global _ck_w4a8_linear
    if not on:
        yield
        return
    from comfy_kitchen.tensor import w4a8_int8 as ck

    _ck_w4a8_linear = ck.w4a8_int8_linear
    ck.w4a8_int8_linear = _packed_linear_routed
    try:
        yield
    finally:
        ck.w4a8_int8_linear = _ck_w4a8_linear


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
        a = down.detach().to(x.dtype)
        b = up.detach().to(x.dtype) if scale == 1.0 else (up.detach() * scale).to(x.dtype)  # (rank, in), (out, rank)
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
        d_up = g.t() @ z if ctx.scale == 1.0 else (g.t() @ z) * ctx.scale
        d_down = gz.t() @ x2
        # the int8 input gradient only when something before the layer needs it
        gi = _layer_input_grad(ctx.m, g, gz @ a).to(grad_out.dtype).reshape(ctx.in_shape) if ctx.needs_input_grad[0] else None
        ctx.m = None
        return gi, d_down.to(ctx.dtypes[0]), d_up.to(ctx.dtypes[1]), None, None, None


_ones_cache: dict[tuple, torch.Tensor] = {}


def _ones(width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached all-ones vector of length `width`, avoiding a GPU allocation per linear layer."""
    if torch.compiler.is_compiling():
        return torch.ones(width, device=device, dtype=dtype)
    key = (device, dtype)
    buf = _ones_cache.get(key)
    if buf is None or buf.shape[0] < width:
        buf = torch.ones(width, device=device, dtype=dtype)
        _ones_cache[key] = buf
    return buf[:width]


def _int8_forward(m, x2: torch.Tensor, residual: torch.Tensor) -> torch.Tensor | None:
    """The int8 layer's forward with `residual` added in the GEMM's epilogue (ComfyUI's
    QuantizedTensor.linear_input_act, which outside training it uses for a block's residual), or
    None where the layer or this ComfyUI can't."""
    if torch.version.hip:  # comfy_kitchen's HIP backend applies residual via addcmul(residual, out, ones), not in epilogue
        return None
    from comfy.ops import CastBiasWeightContext

    with CastBiasWeightContext(m, x2, offloadable=True, compute_dtype=x2.dtype, want_requant=True) as (w, bias):
        fused = getattr(w, "linear_input_act", None) if type(w).__name__ == "QuantizedTensor" else None
        if fused is None or (_fuse_failed_once and quant_format(m) in PACKED_FORMATS):  # failed once: it fails again
            return None
        ones = _ones(residual.shape[-1], x2.device, x2.dtype)
        try:
            out = fused(x2, bias, residual=residual, residual_scale=ones)
        except (*_BUGS, torch.OutOfMemoryError):  # a mistake in the code, or a full card: not a missing epilogue
            raise
        except Exception as e:  # a packed format's fused epilogue can be missing on a backend: unfused is the same maths
            if quant_format(m) not in PACKED_FORMATS:
                raise
            _fuse_failed(e)
            return None
    if out is NotImplemented:
        return None
    _count("lora + int8: fused")
    return out


class LoRABeside(_PassesOn, torch.nn.Module):
    """How a plain LoRA attaches (bypass) to a frozen int8 layer while training: as _LoRAInt8.
    Otherwise (sampling, no grad, a layer with weight functions, int8_forward off) the bypass's
    own way, f(x) + h(x). Owns no parameters of its own: the LoRA's stay where they're saved from."""

    def __init__(self, lora, layer):
        super().__init__()
        self.lora = lora
        self.__dict__["layer"] = layer  # not a submodule: its weights aren't the adapter's
        self._alpha = float(lora.alpha)  # read once: a Parameter on the GPU would sync every call

    def _inner(self) -> list:
        return [self.lora]

    def h(self, x, base_out):
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
    return (quant_format(m) is not None and getattr(m, "layout_type", None) is not None
            and type(w).__name__ == "QuantizedTensor" and hasattr(m, "forward_comfy_cast_weights"))


def install(model) -> int:
    """Give the model's weight-only quantized layers the training path above (undone by uninstall)."""
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
    """memory.int8_forward off: the model's quantized layers dequantize to bf16 for every matmul. ComfyUI
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
