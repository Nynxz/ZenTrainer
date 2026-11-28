"""Tests for 1-level 2D Haar wavelet high-frequency detail loss (`objective.haar_hf`)."""

from __future__ import annotations

import pytest
import torch

from zentrainer.harness.objective import Objective, _haar_hf_loss
from zentrainer.spec import ObjectiveSpec, SpecError, TrainSpec

BASE = {"model": {"kind": "checkpoint", "ckpt_name": "m.safetensors"}, "dataset": {"folder": "cats"}}


def test_spec_haar_hf_defaults_off_and_validates_bounds():
    s_def = TrainSpec.from_dict(BASE)
    assert s_def.objective.haar_hf == 0.0

    s_on = TrainSpec.from_dict({**BASE, "objective": {"haar_hf": 0.5}})
    assert s_on.objective.haar_hf == 0.5

    with pytest.raises(SpecError):
        TrainSpec.from_dict({**BASE, "objective": {"haar_hf": -0.1}})
    with pytest.raises(SpecError):
        TrainSpec.from_dict({**BASE, "objective": {"haar_hf": 6.0}})


def test_haar_hf_loss_exact_properties_and_shapes():
    # Zero error -> zero Haar-HF loss
    z = torch.zeros(2, 16, 16, 16)
    assert _haar_hf_loss(z, None).abs().max().item() == 0.0

    # Constant (pure DC) error -> zero high-frequency energy
    dc = torch.ones(2, 16, 16, 16) * 3.5
    assert _haar_hf_loss(dc, None).abs().max().item() == pytest.approx(0.0, abs=1e-6)

    # High-freq error -> positive energy, differentiable, works on 4D, 5D & odd H/W
    for shape in ((2, 16, 16, 16), (2, 16, 1, 15, 17)):
        err = torch.randn(shape, requires_grad=True)
        hf = _haar_hf_loss(err, None)
        assert hf is not None and hf.shape == (2,)
        assert torch.isfinite(hf).all() and (hf > 0).all()
        (g,) = torch.autograd.grad(hf.sum(), err)
        assert torch.isfinite(g).all() and g.abs().sum().item() > 0.0

    # 1D audio latents safely return None (no-op)
    assert _haar_hf_loss(torch.randn(2, 32, 64), None) is None

    # Masked loss: zero weight in right half ignores error there
    e = torch.zeros(1, 4, 8, 8)
    e[..., :, 4:] = torch.randn(1, 4, 8, 4)
    mask = torch.zeros(1, 8, 8)
    mask[:, :, :4] = 1.0
    assert _haar_hf_loss(e, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_objective_train_loss_adds_haar_hf_while_validation_loss_stays_plain():
    obj_off = Objective(ObjectiveSpec(loss="mse", target="x0", haar_hf=0.0))
    obj_on = Objective(ObjectiveSpec(loss="mse", target="x0", haar_hf=0.5))

    x_t = torch.randn(2, 16, 12, 12)
    x0 = torch.randn(2, 16, 12, 12)
    x0_pred = (x0 + 0.25 * torch.randn_like(x0)).requires_grad_(True)
    sigma = torch.tensor([0.3, 0.7])

    # Validation loss() must stay identical with or without haar_hf
    torch.testing.assert_close(obj_off.loss(x_t, x0, x0_pred, sigma), obj_on.loss(x_t, x0, x0_pred, sigma))

    # train_loss() must be strictly larger when haar_hf > 0 and backprop cleanly
    tl_off = obj_off.train_loss(x_t, x0, x0_pred, sigma)
    tl_on = obj_on.train_loss(x_t, x0, x0_pred, sigma)
    assert (tl_on > tl_off).all()
    (g,) = torch.autograd.grad(tl_on.mean(), x0_pred)
    assert torch.isfinite(g).all() and g.abs().sum().item() > 0.0
