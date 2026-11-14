"""H3 attention casting preserves container ownership and releases temporary references."""

import sys
import types
import weakref

import pytest
import torch

from zentrainer.harness import compat
from zentrainer.harness.compat import minimax_h3 as patch


class AttentionTensorContainer:
    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("attention tensor container has already been consumed")
        return self.tensor

    def take(self):
        tensor = self.peek()
        self.tensor = None
        return tensor


def _install(monkeypatch, attention, in_training):
    h3 = types.ModuleType("comfy.ldm.minimax.model")
    h3.Attention = type("Attention", (), {"forward": lambda *a, **k: None})
    h3._mod_gate = lambda *a, **k: None
    h3.AttentionTensorContainer = AttentionTensorContainer
    h3.optimized_attention = attention
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = in_training
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.minimax", types.ModuleType("comfy.ldm.minimax")),
                      ("comfy.ldm.minimax.model", h3), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    patch.apply()
    return h3.optimized_attention


@pytest.mark.parametrize("in_training", [False, True])
@pytest.mark.parametrize("q_dtype,k_dtype", [
    (torch.float32, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
])
def test_containers_release_inputs_inside_the_backend(monkeypatch, in_training, q_dtype, k_dtype):
    callers = tuple(AttentionTensorContainer(torch.randn(1, 2, 4, 8, dtype=dtype))
                    for dtype in (q_dtype, k_dtype, torch.bfloat16))
    originals = tuple(weakref.ref(c.peek()) for c in callers)
    expected = tuple(c.peek().to(torch.bfloat16).clone() for c in callers)
    options = {"skip_reshape": True, "transformer_options": {"sentinel": True}}

    def backend(q, k, v, heads, **kwargs):
        assert all(isinstance(c, AttentionTensorContainer) for c in (q, k, v))
        assert heads == 2 and kwargs == options
        if q_dtype == k_dtype == torch.bfloat16:
            assert all(a is b for a, b in zip((q, k, v), callers))
        tensors = (q.take(), k.take(), v.take())
        assert all(c.tensor is None for c in callers)
        torch.testing.assert_close(tensors, expected, rtol=0, atol=0)
        consumed = tuple(weakref.ref(t) for t in tensors)
        del tensors
        assert all(ref() is None for ref in originals + consumed)
        return "attention"

    attention = _install(monkeypatch, backend, in_training)
    assert attention(*callers, 2, **options) == "attention"


@pytest.mark.parametrize("in_training", [False, True])
@pytest.mark.parametrize("q_dtype,k_dtype", [
    (torch.float32, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
])
@pytest.mark.parametrize("containers", [False, True])
def test_cast_keeps_values_and_gradients(monkeypatch, in_training, q_dtype, k_dtype, containers):
    inputs = tuple(torch.randn(1, 2, 4, 8, dtype=dtype, requires_grad=True)
                   for dtype in (q_dtype, k_dtype, torch.bfloat16))
    reference = tuple(t.detach().clone().requires_grad_() for t in inputs)

    def backend(q, k, v, heads, **kwargs):
        if isinstance(q, AttentionTensorContainer):
            q, k, v = q.take(), k.take(), v.take()
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        return q + k + v

    attention = _install(monkeypatch, backend, in_training)
    args = tuple(AttentionTensorContainer(t) for t in inputs) if containers else inputs
    got = attention(*args, 2)
    if containers:
        assert all(c.tensor is None for c in args)
    expected = backend(*(t.to(torch.bfloat16) for t in reference), 2)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    got.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(tuple(t.grad for t in inputs), tuple(t.grad for t in reference), rtol=0, atol=0)
