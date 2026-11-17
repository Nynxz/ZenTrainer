"""Weight-only quantized layers (harness/kernels/int8_backward.py): which formats take the training
path, and that a packed 4/6-bit weight reaches the int8 W8A8 kernels only decoded to int8.

ComfyUI marks both int8_tensorwise and the packed formats (asym_w4a8_int8, w6a8_int8) weight-only: a
layer whose input needs a gradient runs F.linear on the QuantizedTensor, and autograd's linear
backward hands the w4a8 mm handler a non-transposed RHS, which it refuses ("W4A8 GEMM expects RHS
W.T."). install() gives those layers an autograd Function instead. The W8A8 kernels read an int8
qdata with a scale per output row: a packed layer's 4/6-bit codes and group scales handed over as they
are would give a wrong gradient, so in "w8a8" they're decoded to the int8 grid the format's forward
uses first, and every other mode dequantizes.
"""

import sys
import types

import pytest
import torch

from zentrainer.harness.kernels import int8_backward as ib


class QuantizedTensor:
    """The parts of comfy_kitchen's QuantizedTensor the path reads. The class name is what it checks."""

    def __init__(self, weight, qdata, scale):
        self._qdata = qdata
        self._params = types.SimpleNamespace(scale=scale, convrot=True, convrot_groupsize=256)
        self._weight = weight

    def dequantize(self):
        return self._weight


class Layer(torch.nn.Module):
    """A ComfyUI mixed-precision Linear, as much of it as this path asks for."""

    def __init__(self, fmt, weight, layout="AsymW4A8Int8Layout"):
        super().__init__()
        self.quant_format, self.layout_type = fmt, layout
        self.weight = weight
        self.weight_function, self.bias_function = [], []

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        return input


def fake_comfy(monkeypatch):
    ops = types.ModuleType("comfy.ops")

    class CastBiasWeightContext:
        def __init__(self, module, *args, **kwargs):
            self.module = module

        def __enter__(self):
            return self.module.weight, None

        def __exit__(self, *exc):
            return False

    ops.CastBiasWeightContext = CastBiasWeightContext
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = True
    comfy = types.ModuleType("comfy")
    comfy.ops, comfy.model_management = ops, mm
    for name, mod in (("comfy", comfy), ("comfy.ops", ops), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_packed_formats_take_the_training_path():
    for fmt in ("int8_tensorwise", "asym_w4a8_int8", "w6a8_int8"):
        m = Layer(fmt, QuantizedTensor(torch.zeros(4, 8), torch.zeros(4, 4, dtype=torch.int8), torch.ones(4)))
        assert ib.quant_format(m) == fmt and ib._int8_layer(m)
    fp8 = Layer("float8_e4m3fn", QuantizedTensor(torch.zeros(4, 8), None, None), layout="TensorCoreFP8Layout")
    assert ib.quant_format(fp8) is None and not ib._int8_layer(fp8)


def test_an_int8_weight_still_takes_the_w8a8_kernel(monkeypatch):
    fake_comfy(monkeypatch)
    q = QuantizedTensor(torch.randn(8, 16), torch.zeros(8, 16, dtype=torch.int8), torch.ones(8))
    sentinel = torch.full((3, 16), 7.0)

    class CudaGradient:  # the kernels are CUDA-only and CI has no GPU: the gate is what's tested here
        is_cuda, dtype = True, torch.bfloat16

    monkeypatch.setattr(ib, "_mode", "w8a8")
    monkeypatch.setattr(ib, "w8a8_input_grad", lambda *a, **k: sentinel)
    ib.stats.clear()
    got = ib._layer_input_grad(Layer("int8_tensorwise", q, layout="TensorWiseINT8Layout"), CudaGradient())
    assert got is sentinel and ib.stats["int8 layer: w8a8"] == 1


def test_install_covers_a_packed_layer(monkeypatch):
    fake_comfy(monkeypatch)
    m = Layer("asym_w4a8_int8", QuantizedTensor(torch.zeros(4, 8), None, None))
    holder = torch.nn.Module()
    holder.add_module("proj", m)
    assert ib.install(holder) == 1 and "forward" in m.__dict__
    ib.uninstall()
    assert "forward" not in m.__dict__


class PackedQuantizedTensor(QuantizedTensor):
    """A packed layer's weight with the params packed_input_grad reads (group scales, s_channel, a codebook)."""

    def __init__(self, weight, correction=None, transposed=False):
        super().__init__(weight, torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.int8), None)
        self._params = types.SimpleNamespace(
            scale=torch.ones(weight.shape[0], weight.shape[1] // 16), s_channel=torch.full((weight.shape[0],), 0.5),
            correction=correction, codebook=torch.zeros(16), group_size=16, convrot_groupsize=256,
            mma_packed=False, transposed=transposed, orig_shape=tuple(weight.shape))


PackedQuantizedTensor.__name__ = "QuantizedTensor"  # the layer check reads the class name, as the fake above does


class CudaGradient(torch.Tensor):  # the gate is CUDA-only and CI has no GPU: a CPU tensor read as CUDA exercises it
    is_cuda = True


def dequantizes_to(monkeypatch, w):
    """comfy_kitchen's dequantize, standing in for a fake packed weight: it gives back `w`."""
    monkeypatch.setattr(ib, "_dequantize_packed", lambda *args: w.to(torch.bfloat16))


def test_packed_w8a8_hands_the_decoded_int8_to_the_kernel(monkeypatch):
    fake_comfy(monkeypatch)
    w = torch.randn(8, 32)
    decoded = torch.randint(-127, 127, (8, 32), dtype=torch.int8)
    seen = {}

    def kernel(g, q, scale, group, residual=None):
        seen.update(q=q, scale=scale, group=group)
        return torch.full((g.shape[0], q.shape[1]), 7.0)

    monkeypatch.setattr(ib, "_mode", "w8a8")
    monkeypatch.setattr(ib, "_packed_w8a8", True)
    monkeypatch.setattr(ib, "_packed_checked", {(f, False) for f in ib.PACKED_FORMATS})  # that check has its own test
    monkeypatch.setattr(ib, "_decode_to_int8", lambda qdata, s_rel, codebook, group_size: decoded)
    monkeypatch.setattr(ib, "w8a8_input_grad", kernel)
    for fmt in ib.PACKED_FORMATS:
        ib.stats.clear()
        g = torch.randn(3, 8).as_subclass(CudaGradient)
        got = ib._layer_input_grad(Layer(fmt, PackedQuantizedTensor(w)), g)
        assert (got == 7.0).all()
        assert seen["q"] is decoded and seen["group"] == 256
        torch.testing.assert_close(seen["scale"], torch.full((8,), 0.5))  # s_channel, the per-row scale
        assert ib.stats[f"{fmt}: w8a8 (decoded)"] == 1


@pytest.mark.parametrize("variant", ["fused mode", "off", "correction", "transposed", "decode fails"])
def test_packed_w8a8_keeps_the_dequantized_path_where_it_cant_apply(monkeypatch, variant):
    fake_comfy(monkeypatch)
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    qt = PackedQuantizedTensor(w, correction=torch.zeros(2, 8) if variant == "correction" else None,
                               transposed=variant == "transposed")

    def kernel_should_not_run(*args, **kwargs):
        raise AssertionError("the w8a8 kernel ran")

    def decode_fails(*args, **kwargs):
        raise ValueError("a layout comfy_kitchen refuses")

    monkeypatch.setattr(ib, "_mode", "fused" if variant == "fused mode" else "w8a8")
    monkeypatch.setattr(ib, "_packed_w8a8", variant != "off")
    monkeypatch.setattr(ib, "w8a8_input_grad", kernel_should_not_run)
    dequantizes_to(monkeypatch, w)
    if variant == "decode fails":
        monkeypatch.setattr(ib, "_decode_to_int8", decode_fails)
    monkeypatch.setattr(ib, "_failed_once", {"int8", "packed"})  # no warning noise
    ib.stats.clear()
    g = torch.randn(3, 8).as_subclass(CudaGradient)
    got = ib._layer_input_grad(Layer("asym_w4a8_int8", qt), g)
    torch.testing.assert_close(got.float(), g.float() @ w, rtol=2e-2, atol=5e-2)
    assert ib.stats["asym_w4a8_int8: dequantized"] == 1


def test_a_model_with_a_packed_layer_is_found():
    plain = torch.nn.Sequential(Layer("int8_tensorwise", torch.zeros(1)), torch.nn.Linear(2, 2))
    assert not ib.has_packed(plain)
    assert ib.has_packed(torch.nn.Sequential(plain, Layer("w6a8_int8", torch.zeros(1))))


def test_packed_layers_take_the_w8a8_kernels_only_in_w8a8_mode(monkeypatch):
    monkeypatch.setattr(ib, "_patch", lambda: None)
    monkeypatch.setattr(ib.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ib, "_mode", "")  # enable() writes these: monkeypatch puts them back
    monkeypatch.setattr(ib, "_packed_w8a8", False)
    monkeypatch.setattr(ib, "_packed_checked", {("w6a8_int8", False)})
    monkeypatch.setattr(ib, "_fuse_failed_once", True)
    monkeypatch.setattr(ib, "_failed_once", {"packed"})
    ib.enable("w8a8")
    assert ib._packed_w8a8 and not ib._packed_checked  # a new run checks its first layers again
    assert not ib._fuse_failed_once and not ib._failed_once  # and tries what the last one fell back from
    ib.enable("fused")
    assert not ib._packed_w8a8
    ib.enable("")
    assert not ib._packed_w8a8


def test_the_first_packed_layer_of_each_kind_is_checked_against_bf16(monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    g = torch.randn(3, 8)
    ref = g.bfloat16() @ w.bfloat16()
    w6, w4_gemv = ("w6a8_int8", False), ("asym_w4a8_int8", True)
    monkeypatch.setattr(ib, "_packed_w8a8", True)
    monkeypatch.setattr(ib, "_packed_checked", set())
    assert ib._check_packed(lambda: w.bfloat16(), g, ref * 1.01, None, w6).equal(ref * 1.01)  # close: kept
    assert ib._packed_w8a8 and ib._packed_checked == {w6}  # another kind still gets its own check
    got = ib._check_packed(lambda: w.bfloat16(), g, -ref, None, w4_gemv)  # wrong: bf16 result, mode off
    assert got.equal(ref) and not ib._packed_w8a8 and ib._packed_checked == {w6, w4_gemv}


def test_a_changed_decode_signature_is_not_hidden_behind_the_fallback(monkeypatch):
    fake_comfy(monkeypatch)
    monkeypatch.setattr(ib, "_mode", "w8a8")
    monkeypatch.setattr(ib, "_packed_w8a8", True)

    def decode(*args, **kwargs):
        raise TypeError("_dequant_int4_grouped_to_int8() takes 3 positional arguments but 4 were given")

    monkeypatch.setattr(ib, "_decode_to_int8", decode)
    g = torch.randn(3, 8).as_subclass(CudaGradient)
    with pytest.raises(TypeError):
        ib._layer_input_grad(Layer("w6a8_int8", PackedQuantizedTensor(torch.randn(8, 32))), g)


def test_decoded_int8_reproduces_the_dequantized_gradient():
    """The identity the whole mode rests on, against comfy_kitchen itself: for a symmetric packed weight,
    ((g * s_channel) @ decoded_int8) rotated == g @ dequantize()."""
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.exceptions import NoCapableBackendError
    from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight, quantize_w4a8_int8_weight

    torch.manual_seed(0)
    n, k = 64, 512
    try:
        qdata, s_rel, s_channel, correction, codebook = quantize_w4a8_int8_weight(torch.randn(n, k), group_size=16,
                                                                                  convrot_groupsize=256)
    except NoCapableBackendError as e:  # no backend here that quantizes on this device; anything else fails
        pytest.skip(f"comfy_kitchen can't quantize here: {e}")
    assert correction is None
    q = ib._decode_to_int8(qdata, s_rel, codebook, 16)
    assert q.dtype == torch.int8 and q.shape == (n, k)
    g = torch.randn(5, n)
    h = ib._hadamard(256, g.device, torch.float32)
    got = ((g * s_channel.float()[None]) @ q.float()).view(5, k // 256, 256) @ h.T
    ref = g @ dequantize_w4a8_int8_weight(qdata, s_rel, s_channel, codebook=codebook, group_size=16,
                                          convrot_groupsize=256, output_dtype=torch.float32)
    torch.testing.assert_close(got.reshape(5, k), ref, rtol=1e-3, atol=1e-3)


def test_a_native_backend_without_its_extension_is_not_a_decoder(monkeypatch):
    """comfy_kitchen's cuda/hip modules import with no extension built (_C is None) and their decoders
    then fail: they're only offered when the extension loaded, so Triton and eager get their turn."""
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.backends import cuda, hip

    native = hip if torch.version.hip else cuda
    monkeypatch.setattr(native, "_EXT_AVAILABLE", False)
    names = [fn.__module__ for fn in ib._decoders(on_gpu=True)]
    assert native.__name__ not in names and names[-1] == "comfy_kitchen.backends.eager.w4a8_int8"
    assert [fn.__module__ for fn in ib._decoders(on_gpu=False)] == ["comfy_kitchen.backends.eager.w4a8_int8"]


@pytest.mark.parametrize("bits", [4, 6])
def test_a_gemv_packed_weight_decodes_to_the_same_int8(bits):
    """A resident packed weight ComfyUI repacked for small-M MMA (optimize_for_gemv, CUDA only) is
    unpacked back to codes and group scales first: the int8 it decodes to must be the plain weight's."""
    pytest.importorskip("comfy_kitchen")
    if not torch.cuda.is_available() or torch.version.hip or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the MMA packing exists on NVIDIA cards of compute capability 8.0+ only")
    import dataclasses

    from comfy_kitchen.backends import cuda
    from comfy_kitchen.tensor.base import QuantizedTensor as CKTensor

    if not cuda._EXT_AVAILABLE:
        pytest.skip("comfy_kitchen's CUDA extension isn't built here")
    torch.manual_seed(0)
    plain = CKTensor.from_float(torch.randn(256, 512, device="cuda") * 0.02, "AsymW4A8Int8Layout", bits=bits,
                                group_size=16, codebook=bits == 4)
    q, s = plain._qdata, plain._params.scale
    storage = torch.empty(q.nbytes + s.nbytes, dtype=torch.uint8, device="cuda")  # how ComfyUI keeps it resident
    q_in = storage[:q.nbytes].view(q.dtype).view(q.shape)
    s_in = storage[q.nbytes:].view(s.dtype).view(s.shape)
    q_in.copy_(q)
    s_in.copy_(s)
    resident = CKTensor(q_in, plain._layout_cls, dataclasses.replace(plain._params, scale=s_in))
    packed = resident.optimize_for_gemv(storage)
    assert packed._params.mma_packed
    n, k = plain._params.orig_shape
    want = ib._decode_to_int8(plain._qdata, plain._params.scale, plain._params.codebook, 16)
    got = ib._decode_to_int8(*ib._packed_codes(packed._qdata, packed._params.scale, True, n, k), packed._params.codebook, 16)
    assert torch.equal(got, want)
    g = torch.randn(300, n, device="cuda", dtype=torch.bfloat16)  # and the gradient through the whole op matches
    grads = [ib.packed_input_grad(g, t._qdata, t._params.scale, t._params.s_channel, t._params.codebook, None, 16, 256,
                                  t._params.mma_packed, n, k, "asym_w4a8_int8") for t in (plain, packed)]
    torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=0)


def _real_packed(n=64, k=512, bits=6):
    """A comfy_kitchen packed weight on the CPU (its eager backend), or a skip where it can't quantize."""
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.exceptions import NoCapableBackendError
    from comfy_kitchen.tensor.base import QuantizedTensor as CKTensor

    torch.manual_seed(0)
    try:
        return CKTensor.from_float(torch.randn(n, k) * 0.02, "AsymW4A8Int8Layout", bits=bits, group_size=16,
                                   codebook=bits == 4)
    except NoCapableBackendError as e:
        pytest.skip(f"comfy_kitchen can't quantize here: {e}")


def test_the_packed_backward_compiles_as_one_op(monkeypatch):
    """A compiled block traces the backward with fake tensors: the packed input gradient must be one opaque
    op there (fullgraph: no graph break), and give what it gives eagerly."""
    w = _real_packed()
    p = w._params
    n, k = p.orig_shape
    monkeypatch.setattr(ib, "_mode", "bf16")  # the dequantized path: no CUDA kernel needed
    args = (w._qdata, p.scale, p.s_channel, p.codebook, None, 16, 256, False, n, k, "w6a8_int8")
    g = torch.randn(5, n, dtype=torch.bfloat16)

    def grad(g):
        return ib.packed_input_grad(g, *args) * 2

    torch._dynamo.reset()
    torch.testing.assert_close(torch.compile(grad, backend="aot_eager", fullgraph=True)(g), grad(g))


def test_the_w4a8_forward_compiles_while_routed():
    """comfy_kitchen's w4a8 forward reads raw data pointers, which fake tensors don't have: opaque_w4a8()
    routes it through an op a compiled block traces whole (fullgraph), with the same result."""
    w = _real_packed()
    x = torch.randn(3, 512, dtype=torch.bfloat16)

    def forward(x):
        return torch.nn.functional.linear(x, w) * 2

    want = forward(x)
    with ib.opaque_w4a8(True):
        torch._dynamo.reset()
        got = torch.compile(forward, backend="aot_eager", fullgraph=True)(x)
    from comfy_kitchen.tensor import w4a8_int8

    assert w4a8_int8.w4a8_int8_linear is ib._ck_w4a8_linear  # put back after
    torch.testing.assert_close(got, want)


def test_the_packed_backward_stays_bf16_with_an_fp16_residual(monkeypatch):
    """fp16 training hands the packed op an fp16 residual (a LoRA's input gradient): bf16 + fp16 would
    come out fp32, against the bf16 a compiled graph is built on."""
    w = _real_packed()
    p = w._params
    n, k = p.orig_shape
    monkeypatch.setattr(ib, "_mode", "bf16")
    args = (w._qdata, p.scale, p.s_channel, p.codebook, None, 16, 256, False, n, k, "w6a8_int8")
    g, residual = torch.randn(5, n, dtype=torch.float16), torch.randn(5, k, dtype=torch.float16)

    def grad(g, residual):
        return ib.packed_input_grad(g, *args, residual)

    eager = grad(g, residual)
    assert eager.dtype == torch.bfloat16
    torch._dynamo.reset()
    torch.testing.assert_close(torch.compile(grad, backend="aot_eager", fullgraph=True)(g, residual), eager)


@pytest.mark.parametrize("compiled", [False, True])
def test_a_refused_fused_residual_is_added_after_the_gemm(monkeypatch, compiled):
    """A backend without the packed format's fused epilogue: the op adds the residual itself, at run time
    (compiled, _int8_forward's own fallback only ran while tracing), and stops asking for the fused one."""
    w = torch.randn(6, 8)
    calls = []

    def kitchen(x, *args, residual=None, residual_scale=None, out_dtype=torch.float32, **kwargs):
        calls.append(residual is not None)
        if residual is not None:
            raise RuntimeError("no fused epilogue on this backend")
        return (x @ w.T).to(out_dtype)

    monkeypatch.setattr(ib, "_ck_w4a8_linear", kitchen)
    monkeypatch.setattr(ib, "_fuse_failed_once", False)
    x, h, scale = torch.randn(3, 8), torch.randn(3, 6), torch.full((6,), 0.5)
    t = torch.zeros(1)

    def forward(x, h):
        return ib.w4a8_linear(x, t, t, torch.zeros(6), None, None, None, 16, 256, torch.float32, False, None,
                              None, 0.0, h, scale)

    if compiled:
        torch._dynamo.reset()
        forward = torch.compile(forward, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(forward(x, h), h + scale * (x @ w.T))
    assert ib._fuse_failed_once and calls[-2:] == [True, False]  # tried fused, then the plain GEMM
    calls.clear()
    forward(x, h)
    assert calls == [False]  # once refused, not asked again this run
