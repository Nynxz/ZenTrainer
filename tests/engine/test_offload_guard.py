"""The backward guard on streamed weights (memory/offload.py guard_backward)."""

import sys
import types

import pytest
import torch

from zentrainer.harness.memory import offload


@pytest.fixture
def comfy_ops(monkeypatch):
    ops = types.ModuleType("comfy.ops")
    ops.result = None
    ops.cast_bias_weight = lambda *a, **k: ops.result
    pkg = types.ModuleType("comfy")
    pkg.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    yield ops
    offload.guard_backward(False)


def test_streamed_weights_are_copied_while_training(comfy_ops):
    original = comfy_ops.cast_bias_weight
    offload.guard_backward(True)
    w, b = torch.ones(2, 2), torch.ones(2)
    comfy_ops.result = (w, b, ("offload-stream", w, b))
    weight, bias, state = comfy_ops.cast_bias_weight("layer", offloadable=True)
    assert weight is not w and bias is not b and torch.equal(weight, w) and torch.equal(bias, b)
    assert state == ("offload-stream", w, b)  # ComfyUI still releases its own


def test_left_alone_when_nothing_to_guard(comfy_ops):
    offload.guard_backward(True)
    w, b = torch.ones(2, 2), torch.ones(2)
    converted = w.to(torch.float64)  # already the compute stream's own (a dtype or LoRA change)
    for result in ((w, b), (w, b, None), (w, b, (None, w, b)), (converted, None, ("s", w, None))):
        comfy_ops.result = result
        assert all(x is y for x, y in zip(comfy_ops.cast_bias_weight("layer"), result))
    comfy_ops.result = (w, b, ("s", w, b))
    with torch.no_grad():  # samples, the no-grad teacher passes: nothing keeps it
        assert comfy_ops.cast_bias_weight("layer")[0] is w


def test_guard_comes_off(comfy_ops):
    original = comfy_ops.cast_bias_weight
    offload.guard_backward(True)
    offload.guard_backward(True)  # twice is once
    offload.guard_backward(False)
    assert comfy_ops.cast_bias_weight is original


def test_quantized_weights_copy_their_storage():
    kitchen = pytest.importorskip("comfy_kitchen.tensor")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    q = kitchen.QuantizedTensor.from_float(torch.randn(64, 32, device="cuda", dtype=torch.bfloat16), "TensorWiseINT8Layout")
    c = offload._own(q)
    assert type(c) is type(q) and c._qdata.data_ptr() != q._qdata.data_ptr()
    assert torch.equal(c.dequantize(), q.dequantize())


def test_skip_guard_avoids_clone_during_checkpoint_forward(comfy_ops):
    offload.guard_backward(True)
    w, b = torch.ones(2, 2), torch.ones(2)
    comfy_ops.result = (w, b, ("offload-stream", w, b))
    with offload.skip_guard():
        weight, bias, _ = comfy_ops.cast_bias_weight("layer", offloadable=True)
        assert weight is w and bias is b
    # outside skip_guard (recompute): cloned again
    weight, bias, _ = comfy_ops.cast_bias_weight("layer", offloadable=True)
    assert weight is not w and bias is not b


def test_recompute_cache_stashes_and_pops_streamed_quantized_weights(comfy_ops):
    class QuantizedTensor:
        def __init__(self, qdata):
            self._qdata = qdata

        def _copy_with(self, qdata=None):
            return QuantizedTensor(qdata if qdata is not None else self._qdata.clone())

    m = torch.nn.Module()
    m.quant_format = "int8_tensorwise"
    w = QuantizedTensor(torch.zeros(64, 64, dtype=torch.int8))
    offload.guard_backward(True)
    comfy_ops.result = (w, None, ("offload-stream", w, None))
    with offload.recompute_scope():
        with torch.no_grad():
            comfy_ops.cast_bias_weight(m, offloadable=True)
    got = offload.pop_recompute_weight(m)
    assert got is not None and got is not w and got._qdata.data_ptr() != w._qdata.data_ptr()
    assert offload.pop_recompute_weight(m) is None  # popped once, freed immediately
    offload.clear_recompute_weights()


def test_pop_recompute_weight_does_not_guard_module_id_under_compile():
    """Empty recompute cache must not emit ___check_obj_id guards across compiled blocks."""
    from torch._dynamo.testing import CompileCounter

    offload.clear_recompute_weights()
    torch._dynamo.reset()

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            cached = offload.pop_recompute_weight(self)
            return x @ (cached if cached is not None else self.w)

    blocks = [_Block() for _ in range(4)]
    x = torch.randn(2, 4)
    counter = CompileCounter()
    compiled = torch.compile(_Block.forward, backend=counter)
    for b in blocks:
        compiled(b, x)
    assert counter.frame_count == 1
    torch._dynamo.reset()


def test_keep_small_resident_promotes_only_tiny_offloaded_modules(monkeypatch):
    mm = types.ModuleType("comfy.model_management")
    mm.module_size = lambda mod: sum(p.nbytes for p in mod.parameters(recurse=False))
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(64)  # 512 B -> promoted
            self.big = torch.nn.Linear(512, 512, bias=False)  # 1 MB -> stays offloaded
            self.norm.comfy_cast_weights = True
            self.norm.prev_comfy_cast_weights = False
            self.model_lowvram = True
            self.model_loaded_weight_memory = 0

    model = TinyModel()
    moved = []

    def fake_to(self, dev):
        moved.append(self)
        return self

    monkeypatch.setattr(torch.nn.Module, "to", fake_to)
    unpinned = []
    patcher = types.SimpleNamespace(
        model=model,
        load_device=torch.device("cuda", 0),
        unpin_weight=lambda k: unpinned.append(k),
    )
    n, nbytes = offload.keep_small_resident(patcher, max_bytes=4096)
    assert n == 1 and nbytes == 512
    assert moved == [model.norm]
    assert model.norm.comfy_patched_weights is True
    assert model.norm.comfy_cast_weights is False and not hasattr(model.norm, "prev_comfy_cast_weights")
    assert not getattr(model.big, "comfy_patched_weights", False)
    assert set(unpinned) == {"norm.weight", "norm.bias"}
    assert model.model_loaded_weight_memory == 512
