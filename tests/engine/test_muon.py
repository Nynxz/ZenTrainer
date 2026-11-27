"""Tests for the adapter-native Muon optimizer and cosine/linear lr_floor."""

from __future__ import annotations

import math

import pytest
import torch

from zentrainer.harness import optim as optim_mod
from zentrainer.harness.optim.muon import Muon, _ns5_2d
from zentrainer.spec import OptimSpec, TrainSpec


def test_muon_2d_polar_rms_and_zero_grad():
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.zeros(8, 8))
    w2 = torch.nn.Parameter(torch.randn(64, 16) * 0.05)
    bias = torch.nn.Parameter(torch.zeros(16))
    opt = Muon([w1, w2, bias], lr=1e-3, momentum=0.9, weight_decay=0.0)

    # Zero grad on w2 (like step 0 of LoKr when w1 == 0) must stay finite and leave w2 untouched
    w1.grad = torch.randn_like(w1)
    w2.grad = torch.zeros_like(w2)
    bias.grad = torch.randn_like(bias)
    w2_before = w2.detach().clone()
    opt.step()
    assert torch.equal(w2, w2_before)
    assert torch.isfinite(w1).all() and torch.isfinite(bias).all()

    # After a real step on w2, update delta / lr should have near-unit RMS and compressed NS-5 spectrum
    w2.grad = torch.randn_like(w2)
    opt.step()
    delta = (w2_before - w2.detach()) / 1e-3
    rms = float(delta.pow(2).mean().sqrt())
    assert 0.80 <= rms <= 1.20
    svals = torch.linalg.svdvals(delta / math.sqrt(max(w2.shape)))
    assert float(svals.min()) >= 0.55 and float(svals.max()) <= 1.35

    # Zero-norm rectangular input to _ns5_2d preserves original (tall) shape
    tall_zero = torch.zeros(32, 8)
    assert _ns5_2d(tall_zero).shape == (32, 8)

    # State dict round-trip
    sd = opt.state_dict()
    opt2 = Muon([torch.nn.Parameter(torch.zeros_like(w1)),
                 torch.nn.Parameter(torch.zeros_like(w2)),
                 torch.nn.Parameter(torch.zeros_like(bias))], lr=1e-3)
    opt2.load_state_dict(sd)


def test_muon_oft_blocks_stay_skew_symmetric():
    torch.manual_seed(1)
    blocks = torch.nn.Parameter(torch.zeros(4, 16, 16))
    opt = Muon([blocks], lr=1e-3, momentum=0.9, weight_decay=0.01)
    for _ in range(3):
        blocks.grad = torch.randn_like(blocks)
        opt.step()
        assert torch.allclose(blocks, -blocks.transpose(1, 2), atol=1e-6)
        assert blocks.norm() > 0


def test_optim_build_muon_and_schedule_lr_floor():
    p = torch.nn.Parameter(torch.randn(16, 8))
    spec = OptimSpec(optimizer="Muon", learning_rate=3e-4, lr_scheduler="cosine", warmup_steps=10, steps=100, lr_floor=0.2)
    opt, _sched = optim_mod.build([p], spec)
    assert isinstance(opt, Muon)
    fn = optim_mod._schedule(spec)
    assert math.isclose(fn(0), 0.1, rel_tol=1e-6)
    assert math.isclose(fn(9), 1.0, rel_tol=1e-6)
    assert math.isclose(fn(100), 0.2, rel_tol=1e-6)
    # Default lr_floor=0 reproduces exact decay to 0
    fn0 = optim_mod._schedule(OptimSpec(lr_scheduler="cosine", warmup_steps=0, steps=100, lr_floor=0.0))
    assert math.isclose(fn0(100), 0.0, abs_tol=1e-7)
    fn_lin = optim_mod._schedule(OptimSpec(lr_scheduler="linear", warmup_steps=0, steps=100, lr_floor=0.15))
    assert math.isclose(fn_lin(100), 0.15, rel_tol=1e-6)
    # Constant schedule ignores lr_floor
    fn_const = optim_mod._schedule(OptimSpec(lr_scheduler="constant", warmup_steps=0, steps=100, lr_floor=0.2))
    assert math.isclose(fn_const(100), 1.0, rel_tol=1e-6)


def test_spec_accepts_muon_and_lr_floor():
    base = {"model": {"kind": "checkpoint", "ckpt_name": "m.safetensors"}, "dataset": {"folder": "cats"}}
    spec = TrainSpec.from_dict({**base, "optim": {"optimizer": "Muon", "lr_scheduler": "cosine", "lr_floor": 0.15}})
    assert spec.optim.optimizer == "Muon"
    assert spec.optim.lr_floor == 0.15
    with pytest.raises(ValueError):
        TrainSpec.from_dict({**base, "optim": {"optimizer": "Muon", "lr_floor": 1.5}})
