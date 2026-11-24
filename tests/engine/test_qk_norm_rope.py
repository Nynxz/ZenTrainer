"""The fused q/k RMSNorm + rope (kernels/qk_norm_rope.py) against the separate ops it replaces, in
Krea 2's form and MOSS's: the same forward, and a backward that matches autograd and finite
differences (fp64, CPU)."""

import pytest
import torch
import torch.nn.functional as F

from zentrainer.harness.kernels.qk_norm_rope import qk_norm_rope

B, H, KV, L, D = 2, 4, 2, 9, 16


def _freqs():
    """ComfyUI's rope layout (EmbedND): (1, 1, L, D/2, 2, 2), each pair's [[cos, -sin], [sin, cos]]."""
    a = torch.arange(L, dtype=torch.float64)[:, None] / (10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    return torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], -1).reshape(1, 1, L, D // 2, 2, 2)


def _krea2(x, w, f):
    """Krea 2's norm then ComfyUI's _apply_rope1 (pairs 2i, 2i+1)."""
    n = F.rms_norm(x, (D,), weight=w, eps=1e-5)
    p = n.reshape(*n.shape[:-1], -1, 1, 2)
    return (f[..., 0] * p[..., 0] + f[..., 1] * p[..., 1]).reshape(x.shape)


def _moss(x, w, f):
    """MOSS's norm then its rope_split_half (pairs i, i + D/2)."""
    n = F.rms_norm(x, (D,), weight=w, eps=1e-5)
    p = n.unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    return (f[..., 0] * p[..., 0] + f[..., 1] * p[..., 1]).movedim(-1, -2).flatten(-2)


@pytest.mark.parametrize("model", ["krea2", "krea2_seq_first", "moss"])
def test_the_fused_op_trains_like_the_separate_ones(model):
    from zentrainer.harness.compat.krea2 import _apply_rope_blhd

    torch.manual_seed(0)
    moss = model == "moss"
    seq_first = model in ("moss", "krea2_seq_first")
    f = _freqs().transpose(1, 2) if seq_first else _freqs()  # seq_first: (1, L, 1, D/2, 2, 2)
    separate = _moss if moss else _krea2
    wq, wk = 1 + torch.randn(D, dtype=torch.float64) * 0.3, 1 + torch.randn(D, dtype=torch.float64) * 0.3
    q0 = torch.randn(B, L, H * D, dtype=torch.float64).view(B, L, H, D)
    k0 = torch.randn(B, L, KV * D, dtype=torch.float64).view(B, L, KV, D)
    if not seq_first:  # as Krea 2's rearrange leaves them: (B, H, L, D), not contiguous
        q0, k0 = q0.transpose(1, 2), k0.transpose(1, 2)
    op = lambda q, k: qk_norm_rope(q, k, wq, wk, f, 1e-5, split_half=moss, seq_first=seq_first)  # noqa: E731
    assert torch.autograd.gradcheck(op, (q0.clone().requires_grad_(), k0.clone().requires_grad_()), eps=1e-6, atol=1e-7)
    q, k = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    qs, ks = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    gq, gk = torch.randn_like(q0), torch.randn_like(k0)
    a, b = op(q, k)
    torch.autograd.backward((a, b), (gq, gk))
    torch.autograd.backward((separate(qs, wq, f), separate(ks, wk, f)), (gq, gk))
    assert torch.allclose(a, separate(q0, wq, f)) and torch.allclose(b, separate(k0, wk, f))
    assert torch.allclose(q.grad, qs.grad, atol=1e-12) and torch.allclose(k.grad, ks.grad, atol=1e-12)
    if model == "krea2_seq_first":
        # Verify _apply_rope_blhd forward and backward (gradcheck) match Krea 2's rope
        nq = F.rms_norm(q0, (D,), weight=wq, eps=1e-5)
        assert torch.allclose(_apply_rope_blhd(nq, _freqs()), a)
        assert torch.autograd.gradcheck(lambda t: _apply_rope_blhd(t, _freqs()),
                                        (nq.detach().clone().requires_grad_(),), eps=1e-6, atol=1e-7)


def test_krea2_patch_delegates_outside_zentrainer_run(monkeypatch):
    """Outside compat.in_run (e.g. a later native ComfyUI training job), Krea 2's patch delegates untouched."""
    import sys
    import types

    from zentrainer.harness import compat
    from zentrainer.harness.compat import krea2 as krea2_patch

    calls = []

    class FakeAttention:
        def forward(self, x, freqs=None, mask=None, transformer_options={}):
            calls.append(x)
            return "comfy"

    k2 = types.ModuleType("comfy.ldm.krea2.model")
    k2.Attention = FakeAttention
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = True
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.krea2", types.ModuleType("comfy.ldm.krea2")),
                      ("comfy.ldm.krea2.model", k2), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    monkeypatch.setattr(compat, "in_run", False)
    krea2_patch.apply()

    assert FakeAttention().forward("x") == "comfy" and calls == ["x"]


@pytest.mark.parametrize("rot", [8, 12])  # H3 turns only the first rot dims; 12: pairs not a power of 2
def test_h3s_partial_rope_trains_like_its_separate_ops(rot):
    from zentrainer.harness.compat.minimax_h3 import _rope_split_half

    torch.manual_seed(0)
    a = torch.arange(L, dtype=torch.float64)[:, None] / (10000 ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    f = torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], -1).reshape(1, L, 1, rot // 2, 2, 2)
    wq, wk = 1 + torch.randn(D, dtype=torch.float64) * 0.3, 1 + torch.randn(D, dtype=torch.float64) * 0.3

    def separate(x, w):  # H3's q/k RMSNorm, then its partial split-half rope
        return _rope_split_half(F.rms_norm(x, (D,), weight=w, eps=1e-5), f, rot)

    q0 = torch.randn(1, L, H, D, dtype=torch.float64)
    k0 = torch.randn(1, L, KV, D, dtype=torch.float64)
    op = lambda q, k: qk_norm_rope(q, k, wq, wk, f, 1e-5, split_half=True, seq_first=True, rot=rot)  # noqa: E731
    assert torch.autograd.gradcheck(op, (q0.clone().requires_grad_(), k0.clone().requires_grad_()), eps=1e-6, atol=1e-7)
    q, k = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    qs, ks = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    gq, gk = torch.randn_like(q0), torch.randn_like(k0)
    oq, ok = op(q, k)
    torch.autograd.backward((oq, ok), (gq, gk))
    torch.autograd.backward((separate(qs, wq), separate(ks, wk)), (gq, gk))
    assert torch.allclose(oq, separate(q0, wq)) and torch.allclose(ok, separate(k0, wk))
    assert torch.allclose(q.grad, qs.grad, atol=1e-12) and torch.allclose(k.grad, ks.grad, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton backward runs on CUDA")
@pytest.mark.parametrize("rot", [0, 32, 96])  # all of D, and two partial ropes (96: 48 pairs, masked)
def test_the_triton_backward_matches_torch(rot):
    from zentrainer.harness.kernels import qk_norm_rope as m

    if not m.TRITON_AVAILABLE:
        pytest.skip("no Triton")
    torch.manual_seed(0)
    d, r = 128, rot or 128
    a = torch.arange(64, dtype=torch.float32)[:, None] / (10000 ** (torch.arange(0, r, 2, dtype=torch.float32) / r))
    f = torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], -1).reshape(1, 64, 1, r // 2, 2, 2).cuda()
    x = torch.randn(1, 64, 6, d, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(x)
    w = (1 + torch.randn(d) * 0.3).cuda()
    fast = m._backward(g, x, w, f, 1e-5, True, True, rot)
    slow = m._backward_torch(g, x, w, f, 1e-5, True, True, rot)
    assert torch.allclose(fast.float(), slow.float(), atol=2e-2, rtol=2e-2)
