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


def test_sampling_reuses_training_patcher_when_preview_model_matches():
    import contextlib
    from zentrainer.harness.callbacks import Samples

    def fake_trainer(kind="split", unet="moss.safetensors", ckpt=None, train_dt="default",
                     sample_model="moss.safetensors", sample_dt="default", lora_strength=1.0, scalable=False):
        d = TrainSpec().to_dict()
        d["dataset"]["folder"] = "d"
        d["model"].update(kind=kind, unet_name=unet, ckpt_name=ckpt, unet_weight_dtype=train_dt,
                          clip_name="c.safetensors" if kind == "split" else None,
                          clip_type="moss_image" if kind == "split" else None,
                          vae_name="v.safetensors" if kind == "split" else None)
        d["samples"].update(model_name=sample_model, model_weight_dtype=sample_dt, lora_strength=lora_strength)
        patcher = object()
        seen = []

        @contextlib.contextmanager
        def at_strength(st):
            seen.append(st)
            yield

        net = types.SimpleNamespace(scalable=scalable, at_strength=at_strength, seen=seen)
        return types.SimpleNamespace(spec=TrainSpec.from_dict(d), models=types.SimpleNamespace(patcher=patcher),
                                     network=net)

    cb = Samples(every=100)
    # Same split model + dtype + lora_strength=1.0 -> reuses training patcher without loading a second copy
    t_same = fake_trainer()
    assert not cb._uses_alt_model(t_same)
    assert cb._sampling_model(t_same) is t_same.models.patcher
    # Same split model + dtype + lora_strength=0.8 on scalable network -> still reuses training patcher
    t_scaled = fake_trainer(lora_strength=0.8, scalable=True)
    assert not cb._uses_alt_model(t_scaled)
    assert cb._sampling_model(t_scaled) is t_scaled.models.patcher
    with cb._at_strength(t_scaled, t_scaled.models.patcher, 0.5) as m:
        assert m is t_scaled.models.patcher
    assert t_scaled.network.seen == [pytest.approx(0.4)]
    # Same checkpoint model + lora_strength=1.0 -> reuses training patcher
    t_ckpt = fake_trainer(kind="checkpoint", unet=None, ckpt="sd.safetensors", sample_model="sd.safetensors")
    assert not cb._uses_alt_model(t_ckpt)
    assert cb._sampling_model(t_ckpt) is t_ckpt.models.patcher
    # Different model, different weight dtype, or non-1.0 lora_strength on non-scalable network -> loads alt model
    assert cb._uses_alt_model(fake_trainer(sample_model="moss_turbo.safetensors", scalable=True))
    assert cb._uses_alt_model(fake_trainer(sample_dt="fp8_e4m3fn", scalable=True))
    assert cb._uses_alt_model(fake_trainer(lora_strength=0.8, scalable=False))


def test_alt_sampling_model_applies_turbo_lora_at_requested_strength(monkeypatch):
    """When sampling loads the alternate model (e.g. lora_strength=0.99 or different preview model),
    samples.turbo_lora is still applied at s.turbo_strength (and omitted when strength is 0)."""
    from zentrainer.harness.callbacks import Samples

    applied = []
    fake_alt = types.SimpleNamespace(patches={}, model=object())

    comfy_pkg = types.ModuleType("comfy")
    comfy_sd = types.ModuleType("comfy.sd")
    comfy_utils = types.ModuleType("comfy.utils")
    folder_paths = types.ModuleType("folder_paths")
    comfy_sd.load_diffusion_model = lambda path, model_options=None: fake_alt

    def fake_load_lora(model, clip, weights, strength, clip_strength):
        applied.append((weights, strength))
        return types.SimpleNamespace(patches={}, model=model.model), clip

    comfy_sd.load_lora_for_models = fake_load_lora
    comfy_utils.load_torch_file = lambda path, safe_load=True: f"sd:{path}"
    folder_paths.get_full_path_or_raise = lambda kind, name: f"{kind}/{name}"
    comfy_pkg.sd = comfy_sd
    comfy_pkg.utils = comfy_utils
    for name, mod in (("comfy", comfy_pkg), ("comfy.sd", comfy_sd), ("comfy.utils", comfy_utils),
                      ("folder_paths", folder_paths)):
        monkeypatch.setitem(sys.modules, name, mod)

    d = TrainSpec().to_dict()
    d["dataset"]["folder"] = "d"
    d["model"].update(kind="split", unet_name="moss.safetensors", clip_name="c.safetensors",
                      clip_type="moss_image", vae_name="v.safetensors")
    d["samples"].update(model_name="moss.safetensors", lora_strength=0.99,
                        turbo_lora="turbo.safetensors", turbo_strength=0.75)
    t = types.SimpleNamespace(
        spec=TrainSpec.from_dict(d),
        models=types.SimpleNamespace(patcher=object()),
        network=types.SimpleNamespace(state_dict=lambda: {"w": 1}, wide=None),
    )
    cb = Samples(every=100)
    cb._sampling_model(t)
    assert applied == [({"w": 1}, 0.99), ("sd:loras/turbo.safetensors", 0.75)]

    # Overriding turbo_strength to 0 omits turbo_lora while keeping trained LoRA at 0.99
    applied.clear()
    over = SampleSpec(**{**d["samples"], "turbo_strength": 0.0})
    cb._sampling_model(t, over)
    assert applied == [({"w": 1}, 0.99)]
