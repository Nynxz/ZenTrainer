"""Tests for Base-Residual Self-Focusing (`objective.residual_focus`)."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from zentrainer.harness.trainer import Trainer
from zentrainer.spec import SpecError, TrainSpec


def _fake_trainer(x0_base: torch.Tensor):
    disabled_calls = []

    class _Net:
        @contextlib.contextmanager
        def disabled(self):
            disabled_calls.append(True)
            yield

    fake = types.SimpleNamespace(
        network=_Net(),
        _autocast=contextlib.nullcontext,
        denoiser=lambda x_t, sigma, conds, tread=None: x0_base,
    )
    return fake, disabled_calls


def test_residual_focus_spec_validation():
    base = {"model": {"kind": "checkpoint", "ckpt_name": "m.safetensors"}, "dataset": {"folder": "cats"}}
    spec = TrainSpec.from_dict({**base, "objective": {"residual_focus": 1.5}})
    assert spec.objective.residual_focus == 1.5
    with pytest.raises(SpecError):
        TrainSpec.from_dict({**base, "objective": {"residual_focus": -0.1}})
    with pytest.raises(SpecError):
        TrainSpec.from_dict({**base, "objective": {"residual_focus": 5.5}})
    with pytest.raises(SpecError, match="sliders"):
        TrainSpec.from_dict({
            **base,
            "objective": {"residual_focus": 1.5},
            "slider": {"kind": "text", "positive": "old", "negative": "young"},
        })


def test_residual_focus_weighting_and_mask_composition():
    x0 = torch.zeros(2, 4, 8, 8)
    x0_base = torch.zeros(2, 4, 8, 8)
    x0_base[:, :, :3, :3] = 5.0
    fake, disabled_calls = _fake_trainer(x0_base)

    w = Trainer._residual_focus_weight(fake, x0, x0, torch.ones(2), [], None, alpha=1.5)
    assert disabled_calls == [True]
    assert w.shape == (2, 8, 8)
    assert torch.allclose(w.mean(dim=(1, 2)), torch.ones(2), atol=1e-5)
    assert float(w[:, :3, :3].mean()) > float(w[:, 5:, 5:].mean())

    # Composes cleanly with an existing spatial mask
    mask = torch.ones(2, 8, 8)
    mask[:, 4:, :] = 0.0
    wm = Trainer._residual_focus_weight(fake, x0, x0, torch.ones(2), [], mask, alpha=1.5)
    assert float(wm[:, 4:, :].abs().max()) == 0.0
    assert torch.allclose(wm.mean(dim=(1, 2)), torch.ones(2), atol=1e-5)


def test_residual_focus_5d_video_and_shape_mismatch():
    # 5D video latent (B, C, T, H, W) averages across frames into a (B, H, W) spatial map
    x0 = torch.zeros(1, 4, 3, 8, 8)
    x0_base = torch.zeros(1, 4, 3, 8, 8)
    x0_base[:, :, :, :3, :3] = 4.0
    fake, _ = _fake_trainer(x0_base)
    w = Trainer._residual_focus_weight(fake, x0, x0, torch.ones(1), [], None, alpha=1.5)
    assert w.shape == (1, 8, 8)
    assert torch.allclose(w.mean(dim=(1, 2)), torch.ones(1), atol=1e-5)
    assert float(w[:, :3, :3].mean()) > float(w[:, 5:, 5:].mean())

    # Uniform disagreement produces uniform unit weights (padded border zeros excluded from avg_pool2d)
    uni_fake, _ = _fake_trainer(torch.ones(1, 4, 8, 8))
    wu = Trainer._residual_focus_weight(uni_fake, torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8),
                                        torch.ones(1), [], None, alpha=1.5)
    assert torch.allclose(wu, torch.ones(1, 8, 8), atol=1e-6)

    # Mismatched base output shape falls back to the incoming weight untouched
    bad_fake, _ = _fake_trainer(torch.zeros(1, 4, 4, 4))
    orig = torch.ones(1, 8, 8)
    assert Trainer._residual_focus_weight(bad_fake, x0[:, :, 0], x0[:, :, 0], torch.ones(1), [], orig, alpha=1.5) is orig
