"""Tests for Lotus: LoRA-native optimizer (parity, preconditioning, auto-LR bounds, resume, bf16)."""

import math
import weakref

import pytest
import torch
import torch.nn as nn

from zentrainer.harness import optim
from zentrainer.harness.optim.lotus import Lotus
from zentrainer.spec import OptimSpec


def _make_lora_pairs(n_layers=3, in_dim=64, out_dim=64, rank=8, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(42)
    pairs = []
    for _ in range(n_layers):
        a = nn.Parameter((torch.randn(rank, in_dim, generator=g) / math.sqrt(in_dim)).to(device=device, dtype=dtype))
        b = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
        a._zt_lora_side, a._zt_lora_partner = "down", weakref.ref(b)
        b._zt_lora_side, b._zt_lora_partner = "up", weakref.ref(a)
        pairs.append((a, b))
    return pairs


def test_lotus_discovers_weakref_pairs_and_honors_schedule_and_set_lr():
    pairs = _make_lora_pairs(2, 32, 32, 4)
    bias = nn.Parameter(torch.zeros(32))
    params = [p for ab in pairs for p in ab] + [bias]
    spec = OptimSpec(optimizer="Lotus", learning_rate=1e-4, warmup_steps=4, lr_scheduler="cosine", steps=20)
    opt, sched = optim.build(params, spec)
    assert isinstance(opt, Lotus) and len(opt._pairs) == 2

    # Warmup starts below 1e-4 and ramps up
    lr0 = optim.current_lr(opt)
    assert lr0 < 1e-4
    for _ in range(4):
        for p in params:
            p.grad = torch.randn_like(p) * 0.05
        opt.step()
        sched.step()
    assert optim.current_lr(opt) >= 0.5e-4

    # Mid-run set_lr changes scheduled base LR cleanly
    optim.set_lr(opt, sched, 5e-4)
    assert optim.current_lr(opt) > 2e-4


def test_lotus_resumes_state_and_adaptive_multiplier_cleanly():
    def run_from(seed_list, saved=None, init_params=None):
        torch.manual_seed(0)
        pairs = _make_lora_pairs(2, 32, 32, 4)
        params = [p for ab in pairs for p in ab]
        if init_params is not None:
            with torch.no_grad():
                for p, q in zip(params, init_params):
                    p.copy_(q)
        opt = Lotus(params, lr=1e-3, weight_decay=0.01)
        if saved is not None:
            opt.load_state_dict(saved)
        for sd in seed_list:
            g = torch.Generator().manual_seed(sd)
            for _ in range(6):
                for p in params:
                    # Consistent trend so lr_mult actually moves away from 1.0
                    p.grad = torch.ones_like(p) * 0.2 + torch.randn(p.shape, generator=g) * 0.02
                opt.step()
        return params, opt

    a_params, a_opt = run_from([1, 2])
    b_params, b_opt = run_from([1])
    assert b_opt.get_avg_learning_rate() > 1e-3  # steady trend grew lr_mult
    sd = b_opt.state_dict()
    c_params, c_opt = run_from([2], saved=sd, init_params=b_params)

    # bf16 momentum stochastic rounding adds tiny O(1e-3) variance across RNG streams,
    # while weights and effective LR stay tightly aligned
    for pa, pc in zip(a_params, c_params):
        torch.testing.assert_close(pc, pa, atol=2e-3, rtol=2e-2)
    assert c_opt.get_avg_learning_rate() == pytest.approx(a_opt.get_avg_learning_rate(), rel=2e-2)


def test_lotus_bounded_auto_lr_holds_steady_at_noisy_optimum_and_climbs_on_trend():
    # At optimum + noise: stays calm near 1.0 (neither sinks like Automagic3 nor blows up like Automagic4)
    torch.manual_seed(0)
    w = nn.Parameter(torch.zeros(32, 32))
    opt = Lotus([w], lr=1e-3)
    for _ in range(150):
        w.grad = w.detach() + 0.1 * torch.randn_like(w)
        opt.step()
    assert 0.8e-3 <= opt.get_avg_learning_rate() <= 1.2e-3

    # Persistent trend: climbs cleanly up to the lr_max_mult rail (2.5x), never beyond
    w2 = nn.Parameter(torch.zeros(32, 32))
    opt2 = Lotus([w2], lr=1e-4, lr_max_mult=2.5)
    for _ in range(120):
        w2.grad = torch.ones_like(w2) * 0.1
        opt2.step()
    assert 1.5e-4 < opt2.get_avg_learning_rate() <= 2.5e-4 + 1e-9

    # Persistent overshoot (alternating signs): backs off cleanly down to lr_min_mult (0.5x), never below
    w3 = nn.Parameter(torch.zeros(32, 32))
    opt3 = Lotus([w3], lr=1e-4, lr_min_mult=0.5)
    for i in range(120):
        w3.grad = torch.ones_like(w3) * (0.1 if i % 2 == 0 else -0.1)
        opt3.step()
    assert 0.5e-4 - 1e-9 <= opt3.get_avg_learning_rate() < 0.85e-4


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lotus_low_precision_stochastic_rounding_trains(dtype):
    # Unpaired 2D tensors starting at 1.0 with updates smaller than 1 bf16/fp16 ULP (~1e-4 << 1/128)
    params = [nn.Parameter(torch.ones((16, 64), dtype=dtype)) for _ in range(4)]
    opt = Lotus(params, lr=1e-4, weight_decay=0.0)
    for _ in range(40):
        for p in params:
            p.grad = torch.ones_like(p) * 0.1
        opt.step()
    mean_v = sum(p.detach().float().mean().item() for p in params) / len(params)
    assert mean_v < 0.998 and all(torch.isfinite(p).all() for p in params)


def test_lotus_preconditioner_is_rank_independent_even_on_rank1_collapse():
    from zentrainer.harness.optim.lotus import _gram_preconditioner

    # A rank-1 collapsed Gram matrix at rank 64 must stay well-conditioned (<= 5x at damp=0.25)
    v = torch.ones(1, 64, 1)
    G = v @ v.transpose(1, 2)
    P, ok = _gram_preconditioner(G, damp=0.25)
    assert bool(ok.item())
    evals = torch.linalg.eigvalsh(P[0])
    assert evals.min() > 0.15 and (evals.max() / evals.min()).item() <= 5.1


def test_lotus_lokr_exact_moments_and_kronecker_scale_balancing():
    # Direct LoKr: tiny zero-init w1 (4x4) and large Kaiming-init w2 (64x64) combined via kron(w1, w2).
    torch.manual_seed(0)
    w1 = nn.Parameter(torch.zeros(4, 4))
    w2 = nn.Parameter(torch.randn(64, 64) / math.sqrt(64))
    w1._zt_kron_side, w1._zt_kron_partner = "w1", weakref.ref(w2)
    w2._zt_kron_side, w2._zt_kron_partner = "w2", weakref.ref(w1)

    # Kronecker-structured target so global optimum is 0: compare balanced vs unpaired Lotus
    target = torch.kron(torch.randn(4, 4) * 0.2, torch.randn(64, 64) / math.sqrt(64))
    w1_u, w2_u = nn.Parameter(w1.detach().clone()), nn.Parameter(w2.detach().clone())
    opt_u = Lotus([w1_u, w2_u], lr=2e-3, auto_lr=False)
    opt = Lotus([w1, w2], lr=2e-3, auto_lr=False)
    assert len(opt._kron_pairs) == 1 and len(opt_u._kron_pairs) == 0

    losses, losses_u = [], []
    for _ in range(80):
        loss = (torch.kron(w1, w2) - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        loss_u = (torch.kron(w1_u, w2_u) - target).pow(2).mean()
        loss_u.backward()
        opt_u.step()
        losses_u.append(loss_u.item())

    # Tiny w1 keeps exact fp32 moments; large w2 uses factored bf16 momentum + row/col variance
    assert "exp_avg_sq" in opt.state[w1] and opt.state[w1]["exp_avg"].dtype == torch.float32
    assert "exp_avg_sq_row" in opt.state[w2] and opt.state[w2]["exp_avg"].dtype == torch.bfloat16
    # Kronecker scale balancing converges substantially lower than unbalanced Lotus
    assert losses[-1] < losses[0] * 0.25 and losses[-1] < losses_u[-1] * 0.5

    # Verify scale balancing actively boosts w2 relative to w1 while rms(w1) < rms(w2)
    with torch.no_grad():
        w1_a, w2_a = nn.Parameter(torch.full((4, 4), 1e-3)), nn.Parameter(torch.full((64, 64), 4e-2))
        w1_b, w2_b = nn.Parameter(w1_a.clone()), nn.Parameter(w2_a.clone())
    w1_a._zt_kron_side, w1_a._zt_kron_partner = "w1", weakref.ref(w2_a)
    w2_a._zt_kron_side, w2_a._zt_kron_partner = "w2", weakref.ref(w1_a)
    oa = Lotus([w1_a, w2_a], lr=1e-3, auto_lr=False)
    ob = Lotus([w1_b, w2_b], lr=1e-3, auto_lr=False)  # unpaired baseline
    for p in (w1_a, w2_a, w1_b, w2_b):
        p.grad = torch.ones_like(p) * 0.1
    oa.step()
    ob.step()
    # Balanced step moves w1 half as far (0.5x) and w2 twice as far (2.0x) when rms(w1) << rms(w2)
    dw1_a = (w1_a - 1e-3).abs().mean().item()
    dw1_b = (w1_b - 1e-3).abs().mean().item()
    dw2_a = (w2_a - 4e-2).abs().mean().item()
    dw2_b = (w2_b - 4e-2).abs().mean().item()
    assert dw1_a == pytest.approx(dw1_b * 0.5, rel=1e-2)
    assert dw2_a == pytest.approx(dw2_b * 2.0, rel=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_lotus_cuda_triton_matches_cpu_fallback(dtype):
    from zentrainer.harness.optim import _lotus_triton

    dev = torch.device("cuda")
    # Include both large Triton-stacked LoRA pairs (>1024 numel), small exact 2D, and 1D tensors
    pairs_a = _make_lora_pairs(2, 192, 192, 8, device=dev, dtype=dtype)
    pairs_b = _make_lora_pairs(2, 192, 192, 8, device=dev, dtype=dtype)
    extra_a = [nn.Parameter(torch.randn(7, 13, device=dev, dtype=dtype)), nn.Parameter(torch.randn(19, device=dev, dtype=dtype))]
    extra_b = [nn.Parameter(t.detach().clone()) for t in extra_a]
    with torch.no_grad():
        for (aa, ba), (ab, bb) in zip(pairs_a, pairs_b):
            ab.copy_(aa)
            bb.copy_(ba)

    pa = [p for ab in pairs_a for p in ab] + extra_a
    pb = [p for ab in pairs_b for p in ab] + extra_b
    opt_tri = Lotus(pa, lr=1e-3, weight_decay=0.01)
    opt_ref = Lotus(pb, lr=1e-3, weight_decay=0.01)

    g = torch.Generator(device="cpu").manual_seed(77)
    orig = _lotus_triton.supported
    for step in range(10):
        grads = [torch.randn(p.shape, generator=g).to(device=dev, dtype=dtype) * 0.2 + 0.05 for p in pa]
        for p, gr in zip(pa, grads):
            p.grad = gr.clone()
        for p, gr in zip(pb, grads):
            p.grad = gr.clone()
        opt_tri.step()
        _lotus_triton.supported = lambda t, d: False
        try:
            opt_ref.step()
        finally:
            _lotus_triton.supported = orig

    tol = 2e-3 if dtype == torch.float32 else 2e-2
    for x, y in zip(pa, pb):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x.detach().float(), y.detach().float(), atol=tol, rtol=tol)
    assert opt_tri.get_avg_learning_rate() == pytest.approx(opt_ref.get_avg_learning_rate(), rel=2e-2)
