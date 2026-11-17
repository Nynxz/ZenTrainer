"""Weight-only quantized layers (harness/kernels/int8_backward.py): which formats take the training
path, and that a packed 4/6-bit weight never reaches the int8 W8A8 kernels.

ComfyUI marks both int8_tensorwise and the packed formats (asym_w4a8_int8, w6a8_int8) weight-only: a
layer whose input needs a gradient runs F.linear on the QuantizedTensor, and autograd's linear
backward hands the w4a8 mm handler a non-transposed RHS, which it refuses ("W4A8 GEMM expects RHS
W.T."). install() gives those layers an autograd Function instead. Its input gradient comes from a
dequantized matmul for a packed weight: the W8A8 kernels read a tensor-wise int8 qdata, and 4/6-bit
codes with group scales would give a wrong gradient.
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


def test_a_packed_weight_never_reaches_the_w8a8_kernels(monkeypatch):
    fake_comfy(monkeypatch)
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    q = QuantizedTensor(w, torch.zeros(8, 8, dtype=torch.int8), torch.ones(8))

    def w8a8_should_not_run(*args, **kwargs):
        raise AssertionError("the W8A8 kernels got a packed weight")

    class CudaGradient(torch.Tensor):  # the gate is CUDA-only and CI has no GPU: a CPU tensor read as CUDA exercises it
        is_cuda = True

    monkeypatch.setattr(ib, "_mode", "w8a8")
    monkeypatch.setattr(ib, "w8a8_input_grad", w8a8_should_not_run)
    for fmt in ib.PACKED_FORMATS:
        ib.stats.clear()
        g = torch.randn(3, 8).as_subclass(CudaGradient)
        got = ib._layer_input_grad(Layer(fmt, q), g)
        torch.testing.assert_close(got.float(), (g.float() @ w), rtol=2e-2, atol=5e-2)  # bf16 on both sides
        assert ib.stats[f"{fmt}: dequantized"] == 1


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
    """A packed layer's weight with the params packed_int8() reads (group scales, s_channel, a codebook)."""

    def __init__(self, weight, correction=None, transposed=False):
        super().__init__(weight, torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.int8), torch.ones(weight.shape[0], 1))
        self._params = types.SimpleNamespace(
            scale=torch.ones(weight.shape[0], weight.shape[1] // 16), s_channel=torch.full((weight.shape[0],), 0.5),
            correction=correction, codebook=torch.zeros(16), group_size=16, convrot_groupsize=256,
            mma_packed=False, transposed=transposed, orig_shape=tuple(weight.shape))


PackedQuantizedTensor.__name__ = "QuantizedTensor"  # the layer check reads the class name, as the fake above does


class CudaGradient(torch.Tensor):  # the gate is CUDA-only and CI has no GPU: a CPU tensor read as CUDA exercises it
    is_cuda = True


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
    monkeypatch.setattr(ib, "_packed_checked", True)  # the first-layer comparison has its own test
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


@pytest.mark.parametrize("variant", ["off", "correction", "transposed", "decode fails"])
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

    monkeypatch.setattr(ib, "_mode", "w8a8")
    monkeypatch.setattr(ib, "_packed_w8a8", variant != "off")
    monkeypatch.setattr(ib, "w8a8_input_grad", kernel_should_not_run)
    if variant == "decode fails":
        monkeypatch.setattr(ib, "_decode_to_int8", decode_fails)
    monkeypatch.setattr(ib, "_failed_once", True)  # no warning noise
    ib.stats.clear()
    g = torch.randn(3, 8).as_subclass(CudaGradient)
    got = ib._layer_input_grad(Layer("asym_w4a8_int8", qt), g)
    torch.testing.assert_close(got.float(), g.float() @ w, rtol=2e-2, atol=5e-2)
    assert ib.stats["asym_w4a8_int8: dequantized"] == 1


def test_packed_layers_take_the_w8a8_kernels_only_in_w8a8_mode(monkeypatch):
    monkeypatch.setattr(ib, "_patch", lambda: None)
    monkeypatch.setattr(ib.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ib, "_mode", "")  # enable() writes these: monkeypatch puts them back
    monkeypatch.setattr(ib, "_packed_w8a8", False)
    monkeypatch.setattr(ib, "_packed_checked", True)
    ib.enable("w8a8")
    assert ib._packed_w8a8 and not ib._packed_checked  # a new run checks its first layer again
    ib.enable("fused")
    assert not ib._packed_w8a8
    ib.enable("")
    assert not ib._packed_w8a8


def test_the_first_packed_layer_is_checked_against_bf16(monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    g = torch.randn(3, 8)
    ref = g.bfloat16() @ w.bfloat16()
    monkeypatch.setattr(ib, "_packed_w8a8", True)
    monkeypatch.setattr(ib, "_packed_checked", False)
    assert ib._check_packed(QuantizedTensor(w, None, None), g, ref * 1.01, None).equal(ref * 1.01)  # close: kept
    assert ib._packed_w8a8 and ib._packed_checked
    monkeypatch.setattr(ib, "_packed_checked", False)
    got = ib._check_packed(QuantizedTensor(w, None, None), g, -ref, None)  # wrong: bf16 result, mode off
    assert got.equal(ref) and not ib._packed_w8a8


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
