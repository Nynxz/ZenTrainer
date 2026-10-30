"""The Cosmos Predict2 / Anima attention patch (harness/compat/cosmos.py): in training, self-attention's
q/k norms and split-half rope train (ComfyUI's fused kernel has no backward) and compute what that
kernel does; sampling, and cross-attention, stay ComfyUI's."""

import sys
import types

import torch
import torch.nn.functional as F

from zentrainer.harness import compat
from zentrainer.harness.compat import cosmos as patch

B, L, H, D = 2, 6, 2, 8


def _rope():
    """Cosmos's layout as the blocks get it: (1, L, 1, D/2, 2, 2), each pair's [[cos, -sin], [sin, cos]]."""
    a = torch.arange(L, dtype=torch.float64)[:, None] / (10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    return torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], -1).reshape(L, D // 2, 2, 2).unsqueeze(1).unsqueeze(0)


def _kitchen(x, w, rope, eps):
    """rms_rope_split_half's maths: RMSNorm over each head, then pairs (i, i + D/2) rotated."""
    n = F.rms_norm(x, (D,), weight=w, eps=eps)
    a, b = n[..., : D // 2], n[..., D // 2:]
    f = rope.squeeze(2)[:, :, None]  # (1, L, 1, D/2, 2, 2)
    return torch.cat([f[..., 0, 0] * a + f[..., 0, 1] * b, f[..., 1, 0] * a + f[..., 1, 1] * b], -1)


def _install(monkeypatch, in_training):
    calls = []

    class Attention(torch.nn.Module):
        def __init__(self, selfattn=True):
            super().__init__()
            self.is_selfattn, self.n_heads, self.head_dim = selfattn, H, D
            self.q_proj, self.k_proj, self.v_proj = (torch.nn.Linear(H * D, H * D, bias=False, dtype=torch.float64)
                                                     for _ in range(3))
            self.q_norm = torch.nn.RMSNorm(D, eps=1e-6, dtype=torch.float64).requires_grad_(False)
            self.k_norm = torch.nn.RMSNorm(D, eps=1e-6, dtype=torch.float64).requires_grad_(False)
            self.v_norm = torch.nn.Identity()

        def compute_qkv(self, x, context=None, rope_emb=None, transformer_options={}):
            calls.append(rope_emb is not None)
            return "comfy"

    predict2 = types.ModuleType("comfy.ldm.cosmos.predict2")
    predict2.Attention = Attention
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = in_training
    mm.cast_to = lambda w, dtype=None, device=None: w.to(dtype=dtype, device=device)
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.cosmos", types.ModuleType("comfy.ldm.cosmos")),
                      ("comfy.ldm.cosmos.predict2", predict2), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    patch.apply()
    return Attention, calls


def test_training_trains_through_the_norms_and_rope_and_computes_the_kernels_maths(monkeypatch):
    Attention, calls = _install(monkeypatch, in_training=True)
    torch.manual_seed(0)
    a = Attention()
    a.q_norm.weight.data, a.k_norm.weight.data = 1 + torch.randn(D, dtype=torch.float64) * 0.3, 1 + torch.randn(D, dtype=torch.float64) * 0.3
    x, rope = torch.randn(B, L, H * D, dtype=torch.float64), _rope()
    q, k, v = a.compute_qkv(x, None, rope_emb=rope)
    assert calls == []  # not ComfyUI's
    # the norms' weights in fp32, as the kernel takes them
    want_q = _kitchen(a.q_proj(x).view(B, L, H, D), a.q_norm.weight.float(), rope, 1e-6)
    want_k = _kitchen(a.k_proj(x).view(B, L, H, D), a.k_norm.weight.float(), rope, 1e-6)
    assert torch.allclose(q, want_q, atol=1e-10) and torch.allclose(k, want_k, atol=1e-10)
    assert torch.equal(v, a.v_proj(x).view(B, L, H, D))
    (q.square().sum() + k.sum()).backward()  # the kernel's way raised here: no autograd formula
    assert a.q_proj.weight.grad is not None and a.k_proj.weight.grad.abs().sum() > 0


def test_a_norm_that_trains_gets_its_gradient(monkeypatch):
    Attention, _ = _install(monkeypatch, in_training=True)
    a = Attention()
    a.q_norm.requires_grad_(True)
    q, _, _ = a.compute_qkv(torch.randn(B, L, H * D, dtype=torch.float64), None, rope_emb=_rope())
    q.sum().backward()
    assert a.q_norm.weight.grad is not None


def test_sampling_and_cross_attention_stay_comfyuis(monkeypatch):
    Attention, calls = _install(monkeypatch, in_training=False)
    x = torch.zeros(B, L, H * D, dtype=torch.float64)
    assert Attention().compute_qkv(x, None, rope_emb=_rope()) == "comfy"
    Attention, calls = _install(monkeypatch, in_training=True)
    assert Attention(selfattn=False).compute_qkv(x, x, rope_emb=None) == "comfy"
    assert Attention().compute_qkv(x, None, rope_emb=None) == "comfy"  # no rope: nothing to fuse
    assert calls == [False, False]
