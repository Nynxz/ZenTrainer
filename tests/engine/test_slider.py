"""Slider LoRAs: the text slider's targets (harness/slider.py) and its spec."""

import pytest
import torch

from zentrainer.harness import slider
from zentrainer.spec import SpecError, TrainSpec


def _preds(seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (2, 4, 8, 8)
    x_t = torch.randn(shape, generator=g)
    sigma = torch.tensor([0.9, 0.3])
    return x_t, sigma, *(torch.randn(shape, generator=g) for _ in range(3))


@pytest.mark.parametrize("target", ["native", "x0"])
def test_targets_round_trip_and_guidance_zero_is_the_neutral_prediction(target):
    x_t, sigma, pos, neu, neg = _preds()
    assert torch.allclose(slider.from_space(target, x_t, slider.to_space(target, x_t, neu, sigma), sigma), neu, atol=1e-5)
    plus, minus = slider.text_targets(target, x_t, sigma, pos, neu, neg, guidance=0.0)
    assert torch.allclose(plus, neu, atol=1e-4) and torch.allclose(minus, neu, atol=1e-4)


def test_the_two_ends_mirror_each_other_and_keep_the_neutral_scale():
    x_t, sigma, pos, neu, neg = _preds(1)
    plus, minus = slider.text_targets("native", x_t, sigma, pos, neu, neg, guidance=3.0)
    sp = lambda t: slider.to_space("native", x_t, t, sigma)  # noqa: E731
    n, a, b = sp(neu), sp(plus), sp(minus)
    dims = (1, 2, 3)
    for t in (a, b):  # renormalised to the neutral prediction's mean and std, per sample
        assert torch.allclose(t.mean(dims), n.mean(dims), atol=1e-4) and torch.allclose(t.std(dims), n.std(dims), atol=1e-4)
    # +1 minus -1 points along positive - negative (renormalising scales both ends towards neutral)
    gap, delta = (a - b).flatten(1), (sp(pos) - sp(neg)).flatten(1)
    assert (torch.nn.functional.cosine_similarity(gap, delta) > 0.95).all()


MODEL = {"kind": "split", "unet_name": "m.safetensors", "clip_name": "c.safetensors", "clip_type": "krea2", "vae_name": "v.safetensors"}


def _spec(**slider):
    return TrainSpec.from_dict({"model": MODEL, "dataset": {"folder": "pictures"}, **({"slider": slider} if slider else {})})


def test_slider_spec_needs_both_ends():
    TrainSpec.from_dict({"model": MODEL, "dataset": {"folder": "pictures"}, "slider": {"kind": "text", "positive": "old", "negative": "young", "neutral": "a person"}})
    with pytest.raises(SpecError, match="both ends"):
        _spec(kind="text", positive="old")
    with pytest.raises(SpecError, match="slider.kind"):
        _spec(kind="images")
    assert _spec().slider.kind == ""  # a normal LoRA, as before


def test_a_sliders_file_says_what_its_ends_do():
    import json

    from zentrainer.harness.callbacks import write_metadata

    spec = {"network": {}, "slider": {"kind": "text", "positive": "old", "negative": "young", "neutral": "a person",
                                      "directions": "positive"}}
    meta = json.loads(write_metadata(spec, 100)["zentrainer.slider"])
    assert meta == {"kind": "text", "positive": "old", "negative": "young", "neutral": "a person", "range": [0.0, 1.0]}
    assert "zentrainer.slider" not in write_metadata({"network": {}, "slider": {"kind": ""}}, 100)


def test_the_sweep_strip_labels_each_strength():
    import torch

    from zentrainer.harness.callbacks import SLIDER_SWEEP, _strip

    strip = _strip([torch.rand(64, 48, 3) for _ in SLIDER_SWEEP], SLIDER_SWEEP)
    w, h = strip.size
    assert w > 5 * 48 and h > 64  # five side by side, a band of labels on top


def test_the_sample_models_sweep_rescales_its_lora_in_place():
    import types

    from zentrainer.harness.callbacks import Samples

    lora, other = object(), object()  # the trained LoRA's patch, a base LoRA's
    model = types.SimpleNamespace(patches={"w": [(0.8, lora, 1.0, None, None), (1.0, other, 1.0, None, None)]},
                                  patches_uuid="start")
    t = types.SimpleNamespace(spec=types.SimpleNamespace(samples=types.SimpleNamespace(model_name="turbo", lora_strength=0.8)))
    cb = Samples.__new__(Samples)
    cb._own_patches = {id(lora)}
    with cb._at_strength(t, model, -0.5) as m:
        assert m is model  # the same model, not a copy
        assert [p[0] for p in m.patches["w"]] == [-0.4, 1.0] and m.patches_uuid != "start"
    assert [p[0] for p in model.patches["w"]] == [0.8, 1.0]


def _pair_spec(**slider):
    from zentrainer.spec import TrainSpec

    s = TrainSpec().to_dict()
    s["model"].update(ckpt_name="x.safetensors")
    s["dataset"]["folder"] = "d"
    s["slider"].update(kind="pairs", **slider)
    return s


def test_an_image_slider_names_its_pair_set():
    import pytest

    from zentrainer.spec import SpecError, TrainSpec

    assert TrainSpec.from_dict(_pair_spec(pair_set="cool")).slider.pair_set == "cool"
    for bad in ({"pair_set": ""}, {"pair_set": "../up"}, {"pair_set": "cool", "dataset_end": "sideways"}):
        with pytest.raises(SpecError):
            TrainSpec.from_dict(_pair_spec(**bad))
    s = _pair_spec(pair_set="cool")
    s["dataset"]["edit"] = True
    with pytest.raises(SpecError):
        TrainSpec.from_dict(s)


def test_pictures_without_a_partner_leave_the_batches():
    import types

    import pytest
    import torch

    from zentrainer.harness import data
    from zentrainer.harness.trainer import Trainer

    lat = torch.zeros(4, 8, 8)
    samples = [data.Sample(lat, 0, refs=[lat]), data.Sample(lat, 0, refs=None), data.Sample(lat, 0, refs=[lat])]
    ds = data.Dataset(samples, [None], [""], 0, buckets={(4, 8, 8): [0, 1, 2]})
    t = Trainer.__new__(Trainer)
    t.spec = types.SimpleNamespace(dataset=types.SimpleNamespace(folder="d"), slider=types.SimpleNamespace(pair_set="cool"))
    t._keep_pairs(ds)
    assert ds.buckets == {(4, 8, 8): [0, 2]}
    alone = data.Dataset([data.Sample(lat, 0)], [None], [""], 0, buckets={(4, 8, 8): [0]})
    with pytest.raises(ValueError):
        t._keep_pairs(alone)


def test_the_data_build_gets_each_pictures_partner():
    import types

    from zentrainer.harness.trainer import Trainer
    from zentrainer.spec import DatasetSpec

    t = Trainer.__new__(Trainer)
    t.spec = types.SimpleNamespace(slider=types.SimpleNamespace(kind="pairs", pair_set="cool"))
    built = t._data_spec(DatasetSpec(folder="d"))
    assert (built.edit, built.edit_method, built.control_sets, built.edit_grounded) == (True, "concat", ["cool"], False)
    t.spec.slider.kind = ""
    assert t._data_spec(DatasetSpec(folder="d")).edit is False


def test_a_loras_metadata_has_how_it_was_trained_not_where():
    # the share export's scrub: a shared LoRA doesn't carry the dataset folder (often a person's
    # name), the folders LoRAs go in, a LoRA it continued from, or an edit sample's picture
    import json

    from zentrainer.harness.callbacks import write_metadata

    spec = {"name": "mylora", "network": {"rank": 16, "init_from": "/home/someone/old.safetensors"},
            "dataset": {"folder": "Jane Doe photos", "trigger": "janedoe", "resolution": 1024},
            "validation": {"dataset": "Jane Doe held out"}, "output": {"lora_dir": "/home/someone/loras"},
            "samples": {"prompts": [{"prompt": "janedoe in a cafe", "control": "input/Jane Doe photos/a.png"}, "janedoe"]}}
    meta = json.loads(write_metadata(spec, 100)["zentrainer.spec"])
    assert "folder" not in meta["dataset"] and "dataset" not in meta["validation"]
    assert "lora_dir" not in meta["output"] and "init_from" not in meta["network"]
    assert meta["samples"]["prompts"] == [{"prompt": "janedoe in a cafe"}, "janedoe"]  # the picture gone
    assert meta["dataset"]["trigger"] == "janedoe" and meta["network"]["rank"] == 16  # how it was trained stays
    assert spec["dataset"]["folder"] == "Jane Doe photos"  # the run's own spec untouched
