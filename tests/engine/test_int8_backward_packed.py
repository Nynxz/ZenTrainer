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
