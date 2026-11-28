"""Tests for Automagic3Geo and Lotus adapter geometry (polar orth, LoRA+ 2x, exact lokr_w1, gauge balance, resume)."""

import math

import pytest
import torch
import torch.nn as nn

from zentrainer.harness import optim
from zentrainer.harness.network import Network
from zentrainer.harness.optim.automagic3_geo import Automagic3Geo
from zentrainer.harness.optim.lotus import Lotus, _orth_eligible
from zentrainer.spec import OptimSpec, TrainSpec, locked_changes


def _make_lora(n_layers=2, dim=128, rank=16, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(42)
    params, pairs = [], []
    for _ in range(n_layers):
        a = nn.Parameter((torch.randn(rank, dim, generator=g) / math.sqrt(dim)).to(device=device, dtype=dtype))
        b = nn.Parameter(torch.zeros(dim, rank, device=device, dtype=dtype))
        params.extend([a, b])
        pairs.append((a, b))
    return params, {"lora_pairs": pairs, "lokr_pairs": []}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_automagic3_geo_cuda_triton_matches_cpu_reference():
    g = torch.Generator(device="cpu").manual_seed(7)
    p_cpu, pairs_cpu = _make_lora(2, 128, 16, device="cpu", dtype=torch.float32)
    p_gpu = [nn.Parameter(p.detach().cuda()) for p in p_cpu]
    pairs_gpu = {
        "lora_pairs": [(p_gpu[i], p_gpu[i + 1]) for i in range(0, len(p_gpu), 2)],
        "lokr_pairs": [],
    }
    o_cpu = Automagic3Geo(p_cpu, lr=2e-4, weight_decay=0.01, adapter_pairs=pairs_cpu)
    o_gpu = Automagic3Geo(p_gpu, lr=2e-4, weight_decay=0.01, adapter_pairs=pairs_gpu)

    for step in range(10):
        sign = 1.0 if step < 6 else (-1.0 if step % 2 == 0 else 1.0)
        for pc, pg in zip(p_cpu, p_gpu):
            gr = torch.randn(pc.shape, generator=g) * 0.05 + sign * 0.1
            pc.grad = gr.clone()
            pg.grad = gr.cuda()
        o_cpu.step()
        o_gpu.step()

    for pc, pg in zip(p_cpu, p_gpu):
        torch.testing.assert_close(pg.detach().cpu(), pc.detach(), atol=1e-4, rtol=1e-4)
        assert torch.equal(o_gpu.state[pg]["sign_history"].cpu(), o_cpu.state[pc]["sign_history"])
        torch.testing.assert_close(o_gpu.state[pg]["lr"].cpu(), o_cpu.state[pc]["lr"], atol=1e-6, rtol=1e-5)


def test_automagic3_geo_lora_plus_single_group_scales_b_without_double_multiply():
    p, pairs = _make_lora(1, 128, 16)
    a, b = p
    opt = Automagic3Geo(p, lr=1e-3, polar_orth=False, lora_b_mult=2.0, adapter_pairs=pairs)
    a_init = a.detach().clone()
    g = torch.ones_like(a) * 0.1
    a.grad = g.clone()
    b.grad = g.t().clone()
    opt.step()
    da = (a_init - a.detach()).abs().mean().item()
    db = b.detach().abs().mean().item()
    assert db == pytest.approx(2.0 * da, rel=1e-3)

    # When explicit LoRA+ param groups already set lr_mult > 1, don't double-multiply lora_b_mult
    a2 = nn.Parameter(torch.zeros(16, 128))
    b2 = nn.Parameter(torch.zeros(128, 16))
    groups = [{"params": [a2], "lr": 1e-3}, {"params": [b2], "lr": 4e-3, "lr_mult": 4.0}]
    opt2 = Automagic3Geo(
        groups, lr=1e-3, polar_orth=False, lora_b_mult=2.0, adapter_pairs={"lora_pairs": [(a2, b2)], "lokr_pairs": []}
    )
    a2.grad = g.clone()
    b2.grad = g.t().clone()
    opt2.step()
    assert b2.detach().abs().mean().item() == pytest.approx(4.0 * a2.detach().abs().mean().item(), rel=1e-3)


def test_automagic3_geo_lokr_exact_w1_and_delta_w_preserving_gauge():
    torch.manual_seed(0)
    w1 = nn.Parameter(torch.randn(8, 8) * 0.02)
    w2 = nn.Parameter(torch.randn(128, 128) * 0.5)
    opt = Automagic3Geo(
        [w1, w2], lr=1e-3, lokr_gauge_alpha=0.25, adapter_pairs={"lora_pairs": [], "lokr_pairs": [(w1, w2)]}
    )
    w1.grad = torch.randn_like(w1) * 0.05
    w2.grad = torch.randn_like(w2) * 0.05
    opt.step()
    assert "exp_avg_sq" in opt.state[w1] and opt.state[w1]["exp_avg_sq"].dtype == torch.float32
    assert "exp_avg_sq_row" in opt.state[w2]

    # Directly verify gauge balancing preserves kron(w1, w2) exactly while shrinking the norm ratio
    r_before = w2.detach().square().mean().sqrt().item() / w1.detach().square().mean().sqrt().item()
    dw_before = torch.kron(w1.detach(), w2.detach())
    opt._balance_lokr_gauge()
    dw_after = torch.kron(w1.detach(), w2.detach())
    r_after = w2.detach().square().mean().sqrt().item() / w1.detach().square().mean().sqrt().item()
    torch.testing.assert_close(dw_after, dw_before, atol=1e-5, rtol=1e-5)
    assert r_after < r_before


def test_lotus_adapter_geometry_lora_plus_and_lokr_gauge():
    torch.manual_seed(1)
    p, pairs = _make_lora(1, 128, 16)
    a, b = p
    opt = Lotus(p, lr=1e-3, lora_orth=False, lora_b_mult=2.0, adapter_pairs=pairs)
    a_init = a.detach().clone()
    g = torch.ones_like(a) * 0.1
    a.grad = g.clone()
    b.grad = g.t().clone()
    opt.step()
    da = (a_init - a.detach()).abs().mean().item()
    db = b.detach().abs().mean().item()
    assert db == pytest.approx(2.0 * da, rel=1e-3)

    w1 = nn.Parameter(torch.randn(8, 8) * 0.02)
    w2 = nn.Parameter(torch.randn(128, 128) * 0.5)
    lopt = Lotus([w1, w2], lr=1e-3, lokr_gauge_alpha=0.25, adapter_pairs={"lora_pairs": [], "lokr_pairs": [(w1, w2)]})
    r_before = w2.detach().square().mean().sqrt().item() / w1.detach().square().mean().sqrt().item()
    w1.grad = torch.zeros_like(w1)
    w2.grad = torch.zeros_like(w2)
    dw_before = torch.kron(w1.detach(), w2.detach())
    lopt.step()
    dw_after = torch.kron(w1.detach(), w2.detach())
    r_after = w2.detach().square().mean().sqrt().item() / w1.detach().square().mean().sqrt().item()
    torch.testing.assert_close(dw_after, dw_before, atol=1e-5, rtol=1e-5)
    assert r_after < r_before


def test_automagic3_geo_build_set_lr_floor_and_resume_parity():
    p, pairs = _make_lora(1, 128, 16)
    spec = OptimSpec(optimizer="Automagic3", adapter_geo=True, learning_rate=1e-4, lr_floor=0.5)
    opt, sched = optim.build(p, spec, adapter_pairs=pairs)
    assert isinstance(opt, Automagic3Geo)
    assert opt.param_groups[0]["min_lr"] == pytest.approx(5e-5)
    assert opt.param_groups[0]["max_lr"] == pytest.approx(6e-4)

    # Lotus adapter_geo toggle wires polar orth, LoRA+ 2x, and LoKr gauge balancing; off keeps defaults
    p_l, pairs_l = _make_lora(1, 128, 16)
    l_on, _ = optim.build(p_l, OptimSpec(optimizer="Lotus", adapter_geo=True, learning_rate=1e-4), adapter_pairs=pairs_l)
    assert isinstance(l_on, Lotus)
    assert l_on.param_groups[0]["lora_orth"] is True
    assert l_on.param_groups[0]["lora_b_mult"] == pytest.approx(2.0)
    assert l_on.param_groups[0]["lokr_gauge_alpha"] == pytest.approx(0.25)
    l_off, _ = optim.build(p_l, OptimSpec(optimizer="Lotus", adapter_geo=False, learning_rate=1e-4), adapter_pairs=pairs_l)
    assert l_off.param_groups[0]["lora_orth"] is False
    assert l_off.param_groups[0]["lora_b_mult"] == pytest.approx(1.0)
    assert l_off.param_groups[0]["lokr_gauge_alpha"] == pytest.approx(0.0)

    g = torch.Generator().manual_seed(3)
    for step in range(8):
        s = 1.0 if step < 5 else -1.0
        for t in p:
            t.grad = torch.randn(t.shape, generator=g) * 0.05 + s * 0.1
        opt.step()
        sched.step()

    # Mid-run set_lr moves adapted LR cleanly
    before = optim.current_lr(opt)
    optim.set_lr(opt, sched, before / 3)
    assert optim.current_lr(opt) == pytest.approx(before / 3, rel=1e-5)

    # Save & resume on fp16 tiny lokr_w1 must preserve exact fp32 second-moment VALUES (not round ~1e-11 to 0)
    # and match an uninterrupted optimizer's next parameter update.
    torch.manual_seed(9)
    w1_a = nn.Parameter(torch.randn(8, 8, dtype=torch.float16))
    w2_a = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
    w1_b = nn.Parameter(w1_a.detach().clone())
    w2_b = nn.Parameter(w2_a.detach().clone())
    o_cont = Automagic3Geo([w1_a, w2_a], lr=1e-4, adapter_pairs={"lora_pairs": [], "lokr_pairs": [(w1_a, w2_a)]})
    o_save = Automagic3Geo([w1_b, w2_b], lr=1e-4, adapter_pairs={"lora_pairs": [], "lokr_pairs": [(w1_b, w2_b)]})
    g1 = torch.full((8, 8), 1e-4, dtype=torch.float16)  # sq ~ 1e-8 -> (1 - 0.999) * 1e-8 = 1e-11 (underflows in fp16!)
    g2 = torch.full((64, 64), 1e-2, dtype=torch.float16)
    for w1, w2, opt_i in ((w1_a, w2_a, o_cont), (w1_b, w2_b, o_save)):
        torch.manual_seed(77)
        w1.grad = g1.clone()
        w2.grad = g2.clone()
        opt_i.step()
    assert torch.equal(w1_a.detach(), w1_b.detach())
    saved_v = o_save.state[w1_b]["exp_avg_sq"].clone()
    assert saved_v.dtype == torch.float32 and saved_v.min().item() > 0.0 and saved_v.max().item() < 1e-9

    o_resumed = Automagic3Geo([w1_b, w2_b], lr=1e-4, adapter_pairs={"lora_pairs": [], "lokr_pairs": [(w1_b, w2_b)]})
    o_resumed.load_state_dict(o_save.state_dict())
    assert o_resumed.state[w1_b]["exp_avg_sq"].dtype == torch.float32
    torch.testing.assert_close(o_resumed.state[w1_b]["exp_avg_sq"], saved_v, atol=0.0, rtol=0.0)

    # Next parameter update matches uninterrupted run (same RNG seed for stochastic rounding)
    g1_next = torch.full((8, 8), 2e-4, dtype=torch.float16)
    g2_next = torch.full((64, 64), 2e-2, dtype=torch.float16)
    torch.manual_seed(123)
    w1_a.grad, w2_a.grad = g1_next.clone(), g2_next.clone()
    o_cont.step()
    torch.manual_seed(123)
    w1_b.grad, w2_b.grad = g1_next.clone(), g2_next.clone()
    o_resumed.step()
    torch.testing.assert_close(w1_b.detach(), w1_a.detach(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("adapter_geo,ceil_mult", [(False, 4.0), (True, 6.0)])
def test_automagic3_resume_preserves_builder_lr_ceiling(adapter_geo, ceil_mult):
    p_cont, pairs_cont = _make_lora(1, 64, 8)
    p_res = [nn.Parameter(t.detach().clone()) for t in p_cont]
    pairs_res = {"lora_pairs": [(p_res[0], p_res[1])], "lokr_pairs": []}
    spec = OptimSpec(optimizer="Automagic3", adapter_geo=adapter_geo, learning_rate=3e-4, lr_floor=0.25)

    o_cont, _ = optim.build(p_cont, spec, adapter_pairs=pairs_cont)
    o_save, _ = optim.build(p_res, spec, adapter_pairs=pairs_res)
    expected_max = 3e-4 * ceil_mult
    expected_min = 3e-4 * 0.25
    assert o_save.param_groups[0]["max_lr"] == pytest.approx(expected_max)
    assert o_save.param_groups[0]["min_lr"] == pytest.approx(expected_min)

    for _ in range(6):
        for t in p_cont:
            t.grad = torch.ones_like(t) * 0.1
        for t in p_res:
            t.grad = torch.ones_like(t) * 0.1
        o_cont.step()
        o_save.step()

    o_resumed, _ = optim.build(p_res, spec, adapter_pairs=pairs_res)
    o_resumed.load_state_dict(o_save.state_dict())
    assert o_resumed.param_groups[0]["max_lr"] == pytest.approx(expected_max)
    assert o_resumed.param_groups[0]["min_lr"] == pytest.approx(expected_min)

    # Consistent gradients drive the adapted LR right into the ceiling; uninterrupted and resumed must match
    for _ in range(25):
        for t in p_cont:
            t.grad = torch.ones_like(t) * 0.1
        for t in p_res:
            t.grad = torch.ones_like(t) * 0.1
        o_cont.step()
        o_resumed.step()
    assert optim.current_lr(o_resumed) == pytest.approx(expected_max, rel=1e-5)
    assert optim.current_lr(o_resumed) == pytest.approx(optim.current_lr(o_cont), rel=1e-5)
    for pa, pb in zip(p_cont, p_res):
        torch.testing.assert_close(pb.detach(), pa.detach(), atol=1e-6, rtol=1e-5)


def test_adapter_geo_locked_at_resume_and_shared_orth_eligibility():
    base = TrainSpec().to_dict()
    toggled = TrainSpec().to_dict()
    toggled["optim"]["adapter_geo"] = True
    assert "optim.adapter_geo" in locked_changes(base, toggled)

    # Standalone Lotus preserves live geometry flags across load_state_dict rather than silently reverting
    p, pairs = _make_lora(1, 64, 8)
    o_on = Lotus(p, lr=1e-4, lora_orth=True, lora_b_mult=2.0, lokr_gauge_alpha=0.25, adapter_pairs=pairs)
    for t in p:
        t.grad = torch.ones_like(t) * 0.1
    o_on.step()
    o_off = Lotus(p, lr=1e-4, lora_orth=False, lora_b_mult=1.0, lokr_gauge_alpha=0.0, adapter_pairs=pairs)
    o_off.load_state_dict(o_on.state_dict())
    assert o_off.param_groups[0]["lora_orth"] is False
    assert o_off.param_groups[0]["lora_b_mult"] == pytest.approx(1.0)
    assert o_off.param_groups[0]["lokr_gauge_alpha"] == pytest.approx(0.0)

    # Shared polar orthogonalization eligibility applies to 8x8 LoKr cores on BOTH Automagic3Geo and Lotus
    w_core = nn.Parameter(torch.randn(8, 8))
    w_big = nn.Parameter(torch.randn(128, 128))
    assert _orth_eligible(w_core) is True
    assert _orth_eligible(w_big) is False
    torch.manual_seed(0)
    g_ill = torch.randn(8, 8)
    g_ill[0] *= 20.0
    w_on, w_off = nn.Parameter(torch.zeros(8, 8)), nn.Parameter(torch.zeros(8, 8))
    o_geo_on = Automagic3Geo([w_on], lr=1e-2, polar_orth=True)
    o_geo_off = Automagic3Geo([w_off], lr=1e-2, polar_orth=False)
    w_on.grad, w_off.grad = g_ill.clone(), g_ill.clone()
    o_geo_on.step()
    o_geo_off.step()
    assert not torch.allclose(w_on.detach(), w_off.detach(), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_automagic3_geo_low_precision_stochastic_rounding(dtype):
    p, pairs = _make_lora(1, 128, 16, device="cuda", dtype=dtype)
    with torch.no_grad():
        for t in p:
            t.fill_(1.0)
    opt = Automagic3Geo(p, lr=1e-4, weight_decay=0.0, adapter_pairs=pairs)
    for _ in range(30):
        for t in p:
            t.grad = torch.ones_like(t) * 0.1
        opt.step()
    mean_v = sum(t.detach().float().mean().item() for t in p) / len(p)
    assert mean_v < 0.999 and all(torch.isfinite(t).all() for t in p)


def test_network_adapter_pairs_extracts_lora_and_lokr():
    net = object.__new__(Network)
    a, b = nn.Parameter(torch.zeros(8, 16)), nn.Parameter(torch.zeros(16, 8))
    w1, w2 = nn.Parameter(torch.zeros(4, 4)), nn.Parameter(torch.zeros(16, 16))
    alpha = nn.Parameter(torch.tensor(8.0), requires_grad=False)
    net.params = {
        "blocks.0.attn.q.lora_down.weight": a,
        "blocks.0.attn.q.lora_up.weight": b,
        "blocks.0.attn.q.alpha": alpha,
        "blocks.0.mlp.up.lokr_w1": w1,
        "blocks.0.mlp.up.lokr_w2": w2,
    }
    pairs = net.adapter_pairs()
    assert pairs["lora_pairs"] == [(a, b)]
    assert pairs["lokr_pairs"] == [(w1, w2)]
