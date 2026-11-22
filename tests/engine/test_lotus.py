"""Tests for Lotus: adapter-native optimizer (parity, exact small moments, auto-LR, resume, bf16)."""

import math

import pytest
import torch
import torch.nn as nn

from zentrainer.harness import optim
from zentrainer.harness.optim.lotus import Lotus
from zentrainer.spec import OptimSpec


def _make_lora_weights(n_layers=3, in_dim=64, out_dim=64, rank=8, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(42)
    params = []
    for _ in range(n_layers):
        params.append(nn.Parameter((torch.randn(rank, in_dim, generator=g) / math.sqrt(in_dim)).to(device=device, dtype=dtype)))
        params.append(nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype)))
    return params


def test_lotus_honors_schedule_and_set_lr():
    params = _make_lora_weights(2, 32, 32, 4) + [nn.Parameter(torch.zeros(32))]
    spec = OptimSpec(optimizer="Lotus", learning_rate=1e-4, warmup_steps=4, lr_scheduler="cosine", steps=20)
    opt, sched = optim.build(params, spec)
    assert isinstance(opt, Lotus)

    # Warmup starts below 1e-4 and ramps up
    assert optim.current_lr(opt) < 1e-4
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
        params = _make_lora_weights(2, 32, 32, 4)
        if init_params is not None:
            with torch.no_grad():
                for p, q in zip(params, init_params):
                    p.copy_(q)
        opt = Lotus(params, lr=1e-3, weight_decay=0.01, auto_lr=True)
        if saved is not None:
            opt.load_state_dict(saved)
        for sd in seed_list:
            g = torch.Generator().manual_seed(sd)
            for _ in range(16):  # > 8 steps so group hypergradient multiplier warms up and moves
                for p in params:
                    p.grad = torch.ones_like(p) * 0.2 + torch.randn(p.shape, generator=g) * 0.02
                opt.step()
        return params, opt

    a_params, a_opt = run_from([1, 2])
    b_params, b_opt = run_from([1])
    assert b_opt.get_avg_learning_rate() > 1.05e-3
    sd = b_opt.state_dict()
    c_params, c_opt = run_from([2], saved=sd, init_params=b_params)

    for pa, pc in zip(a_params, c_params):
        torch.testing.assert_close(pc, pa, atol=2e-3, rtol=2e-2)
    assert c_opt.get_avg_learning_rate() == pytest.approx(a_opt.get_avg_learning_rate(), rel=2e-2)


def test_lotus_group_mean_cos_auto_lr_climbs_brakes_honors_warmup_and_holds_steady():
    # Default (auto_lr=False): AdamW-style scheduled mode never mutates LR multiplier
    w_sched = nn.Parameter(torch.zeros(32, 32))
    opt_sched, _ = optim.build([w_sched], OptimSpec(optimizer="Lotus", learning_rate=1e-4, lotus_auto_lr=False))
    for _ in range(30):
        w_sched.grad = torch.ones_like(w_sched) * 0.1
        opt_sched.step()
    assert opt_sched.get_avg_learning_rate() == pytest.approx(1e-4)

    # auto_lr=True: steady trend climbs strongly; alternating overshoot brakes cleanly
    torch.manual_seed(0)
    w_trend = nn.Parameter(torch.zeros(64, 64))
    opt_trend, _ = optim.build([w_trend], OptimSpec(optimizer="Lotus", learning_rate=1e-4, lotus_auto_lr=True))
    w_osc = nn.Parameter(torch.zeros(64, 64))
    opt_osc, _ = optim.build([w_osc], OptimSpec(optimizer="Lotus", learning_rate=1e-4, lotus_auto_lr=True))
    for i in range(80):
        w_trend.grad = torch.ones_like(w_trend) * 0.1 + 0.05 * torch.randn_like(w_trend)
        w_osc.grad = torch.ones_like(w_osc) * (0.1 if i % 2 == 0 else -0.1)
        opt_trend.step()
        opt_osc.step()
    assert opt_trend.get_avg_learning_rate() > 3.0e-4
    assert opt_osc.get_avg_learning_rate() < 0.3e-4

    # Linear warmup ramp: auto_lr must not wind up multiplier while scheduled LR is still rising
    w_warm = nn.Parameter(torch.zeros(32, 32))
    opt_warm = Lotus([w_warm], lr=1e-6, auto_lr=True)
    for i in range(1, 35):
        opt_warm.param_groups[0]["lr"] = 1e-4 * (i / 34.0)
        w_warm.grad = torch.ones_like(w_warm) * 0.1
        opt_warm.step()
    assert opt_warm.get_avg_learning_rate() == pytest.approx(1e-4, rel=1e-3)

    # Pure zero-mean diffusion noise at optimum: holds steady (no decay stall, no runaway)
    torch.manual_seed(1)
    w_noise = nn.Parameter(torch.zeros(64, 64))
    opt_noise = Lotus([w_noise], lr=1e-3, auto_lr=True)
    for _ in range(150):
        w_noise.grad = w_noise.detach() * 0.1 + 0.5 * torch.randn_like(w_noise)
        opt_noise.step()
    assert 0.7e-3 <= opt_noise.get_avg_learning_rate() <= 1.4e-3


def test_lotus_auto_lr_handles_missing_first_param_grad_and_old_lr_ctrl_resume():
    # Step 1: params[0] has no grad while group_lr_ctrl is created on it; step 2: params[0] gets grad
    p0 = nn.Parameter(torch.randn(16, 16))
    p1 = nn.Parameter(torch.randn(16, 16))
    opt = Lotus([p0, p1], lr=1e-4, auto_lr=True)
    p1.grad = torch.ones_like(p1) * 0.1
    opt.step()
    assert "group_lr_ctrl" in opt.state[p0] and "exp_avg" not in opt.state[p0]
    p0.grad = torch.ones_like(p0) * 0.1
    p1.grad = torch.ones_like(p1) * 0.1
    opt.step()
    assert "exp_avg" in opt.state[p0]

    # Resuming an older checkpoint that stored per-tensor lr_ctrl migrates its mean multiplier
    q0 = nn.Parameter(torch.randn(16, 16))
    q1 = nn.Parameter(torch.randn(16, 16))
    opt_old = Lotus([q0, q1], lr=1e-4, auto_lr=True)
    opt_old.state[q0]["lr_ctrl"] = torch.tensor([2.0, 0.0, 0.0])
    opt_old.state[q1]["lr_ctrl"] = torch.tensor([3.0, 0.0, 0.0])
    assert opt_old.get_avg_learning_rate() == pytest.approx(2.5e-4)
    q0.grad = torch.ones_like(q0) * 0.1
    q1.grad = torch.ones_like(q1) * 0.1
    opt_old.step()
    assert opt_old.state[q0]["group_lr_ctrl"][0].item() == pytest.approx(2.5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lotus_low_precision_stochastic_rounding_trains(dtype):
    # 2D tensors starting at 1.0 with updates smaller than 1 bf16/fp16 ULP (~1e-4 << 1/128)
    params = [nn.Parameter(torch.ones((16, 64), dtype=dtype)) for _ in range(4)]
    opt = Lotus(params, lr=1e-4, weight_decay=0.0)
    for _ in range(40):
        for p in params:
            p.grad = torch.ones_like(p) * 0.1
        opt.step()
    mean_v = sum(p.detach().float().mean().item() for p in params) / len(params)
    assert mean_v < 0.998 and all(torch.isfinite(p).all() for p in params)


def test_lotus_lokr_uses_exact_fp32_moments_for_tiny_w1_and_factored_for_w2():
    # Direct LoKr: tiny zero-init w1 (4x4 <= 1024 elems) keeps exact fp32 moments while large w2 uses factored bf16.
    torch.manual_seed(0)
    w1 = nn.Parameter(torch.zeros(4, 4))
    w2 = nn.Parameter(torch.randn(64, 64) / math.sqrt(64))
    target = torch.kron(torch.randn(4, 4) * 0.2, torch.randn(64, 64) / math.sqrt(64))
    opt = Lotus([w1, w2], lr=2e-3, auto_lr=False)

    losses = []
    for _ in range(80):
        loss = (torch.kron(w1, w2) - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert "exp_avg_sq" in opt.state[w1] and opt.state[w1]["exp_avg"].dtype == torch.float32
    assert "exp_avg_sq_row" in opt.state[w2] and opt.state[w2]["exp_avg"].dtype == torch.bfloat16
    assert losses[-1] < losses[0] * 0.25


def test_lotus_fp16_resume_keeps_second_moments():
    params = _make_lora_weights(1, 256, 256, 8, dtype=torch.float16)  # big enough for factored moments
    opt = Lotus(params, lr=1e-4, auto_lr=True)
    for _ in range(3):
        for p in params:
            p.grad = (torch.randn(p.shape) * 1e-5).to(p.dtype)
        opt.step()
    fresh = Lotus(params, lr=1e-4, auto_lr=True)
    fresh.load_state_dict(opt.state_dict())
    for p in params:
        for k, v in opt.state[p].items():
            if torch.is_tensor(v):
                assert fresh.state[p][k].dtype == v.dtype and torch.equal(fresh.state[p][k], v), k
        assert opt.state[p]["exp_avg_sq_row"].min() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("auto_lr", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_lotus_cuda_triton_matches_fallback(auto_lr, dtype):
    from zentrainer.harness.optim import _lotus_triton

    dev = torch.device("cuda")
    torch.manual_seed(11)
    # Mix of 2D in-place Triton LoRA weights, tiny exact 2D (lokr_w1), 4D conv stacked Triton, and 1D bias
    pa = _make_lora_weights(2, 192, 192, 8, device=dev, dtype=dtype)
    w1_a = nn.Parameter(torch.full((4, 4), 2e-3, device=dev, dtype=dtype))
    w2_a = nn.Parameter(torch.randn(64, 64, device=dev, dtype=dtype) * 0.05)
    conv_a = nn.Parameter(torch.randn(16, 16, 3, 3, device=dev, dtype=dtype) * 0.05)
    bias_a = nn.Parameter(torch.randn(19, device=dev, dtype=dtype) * 0.05)
    pa = pa + [w1_a, w2_a, conv_a, bias_a]
    pb = [nn.Parameter(p.detach().clone()) for p in pa]

    opt_tri = Lotus(pa, lr=1e-3, weight_decay=0.01, auto_lr=auto_lr)
    opt_ref = Lotus(pb, lr=1e-3, weight_decay=0.01, auto_lr=auto_lr)

    g = torch.Generator(device="cpu").manual_seed(55)
    orig_can, orig_sup = _lotus_triton.can_fused_inplace, _lotus_triton.supported
    for _ in range(15):
        grads = [torch.randn(p.shape, generator=g).to(device=dev, dtype=dtype) * 0.2 + 0.05 for p in pa]
        for p, gr in zip(pa, grads):
            p.grad = gr.clone()
        for p, gr in zip(pb, grads):
            p.grad = gr.clone()
        opt_tri.step()
        _lotus_triton.can_fused_inplace = lambda ps, ms: False
        _lotus_triton.supported = lambda t, d: False
        try:
            opt_ref.step()
        finally:
            _lotus_triton.can_fused_inplace = orig_can
            _lotus_triton.supported = orig_sup

    tol = 2e-3 if dtype == torch.float32 else 2e-2
    for x, y in zip(pa, pb):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x.detach().float(), y.detach().float(), atol=tol, rtol=tol)
    assert opt_tri.get_avg_learning_rate() == pytest.approx(opt_ref.get_avg_learning_rate(), rel=2e-2)
