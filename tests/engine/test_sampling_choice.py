"""Custom sigmas and a try's own sampling (spec.check_sigmas, callbacks._try_sampling)."""

import sys
import types

import pytest

from zentrainer.harness.callbacks import _try_sampling
from zentrainer.spec import SampleSpec, SpecError, TrainSpec, check_sigmas


def test_sigmas_go_high_to_low():
    assert check_sigmas([]) == []
    assert check_sigmas(["1", 0.5, 0]) == [1.0, 0.5, 0.0]
    for bad in ([1.0], [0.5, 0.8, 0], [1, -0.1], ["a", 0]):
        with pytest.raises(SpecError):
            check_sigmas(bad)


def test_a_run_with_bad_sigmas_is_refused():
    s = TrainSpec().to_dict()
    s["model"].update(ckpt_name="x.safetensors")
    s["dataset"]["folder"] = "d"
    s["samples"]["sigmas"] = [0.2, 0.9]
    with pytest.raises(SpecError):
        TrainSpec.from_dict(s)


@pytest.fixture
def samplers(monkeypatch):
    mod = types.ModuleType("comfy.samplers")
    mod.KSampler = types.SimpleNamespace(SAMPLERS=["euler", "res_multistep"], SCHEDULERS=["normal", "simple", "beta"])
    pkg = types.ModuleType("comfy")
    pkg.samplers = mod
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.samplers", mod)


def test_a_try_overrides_only_what_it_sets(samplers):
    run = SampleSpec(steps=20, cfg=5.0, sampler="euler", scheduler="normal")
    assert _try_sampling(run, {"prompt": "x"}) is None
    s = _try_sampling(run, {"prompt": "x", "sampler": "res_multistep", "steps": "8", "cfg": 1})
    assert (s.sampler, s.steps, s.cfg, s.scheduler) == ("res_multistep", 8, 1.0, "normal")
    assert run.sampler == "euler"  # the run's own settings are untouched


def test_what_comfyui_doesnt_have_stays_the_runs(samplers):
    run = SampleSpec(sampler="euler", scheduler="normal", steps=20)
    s = _try_sampling(run, {"sampler": "warp_drive", "scheduler": "beta", "steps": "lots", "sigmas": [0.1, 0.9]})
    assert (s.sampler, s.scheduler, s.steps, s.sigmas) == ("euler", "beta", 20, [])


def test_sigmas_and_schedulers_replace_each_other(samplers):
    custom = SampleSpec(sigmas=[1.0, 0.5, 0.0])
    assert _try_sampling(custom, {"scheduler": "simple"}).sigmas == []  # its own scheduler: not the run's sigmas
    assert _try_sampling(SampleSpec(), {"scheduler": "custom", "sigmas": [1, 0.3, 0]}).sigmas == [1.0, 0.3, 0.0]
    assert _try_sampling(SampleSpec(steps=500), {"steps": 9999}).steps == 200


def test_a_try_can_render_without_the_turbo_lora(samplers):
    run = SampleSpec(turbo_lora="turbo.safetensors", turbo_strength=0.75)
    assert _try_sampling(run, {"turbo_strength": 0}).turbo_strength == 0.0
    assert _try_sampling(run, {"turbo_strength": 9}).turbo_strength == 1.5


def _trained():
    s = TrainSpec().to_dict()
    s["model"].update(ckpt_name="x.safetensors")
    s["dataset"]["folder"] = "d"
    return s


def test_a_training_run_changes_only_its_live_sample_settings():
    from zentrainer.spec import live_samples

    s = _trained()
    s["samples"]["sigmas"] = [1.0, 0.5, 0.0]
    got = live_samples(s, {"steps": 25, "cfg": 2.5, "turbo_strength": 0, "scheduler": "simple"})
    assert (got["samples"]["steps"], got["samples"]["cfg"], got["samples"]["turbo_strength"]) == (25, 2.5, 0)
    assert got["samples"]["sigmas"] == []  # its own scheduler replaces the custom sigmas
    assert s["samples"]["steps"] != 25  # the spec handed in is left alone
    for bad in ({}, {"width": 512}, {"turbo_lora": None}, {"steps": 0}, {"turbo_strength": -1}):
        with pytest.raises(SpecError):
            live_samples(s, bad)
