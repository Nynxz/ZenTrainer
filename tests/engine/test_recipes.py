"""Simple mode's recipes (engine/catalog/recipes/): every file well formed, and its settings a spec
the engine takes, so a typo in one fails here rather than when someone trains with it."""

import json

import pytest

from zentrainer.models import recipes
from zentrainer.spec import TrainSpec

KEYS = {"id", "title", "model", "family", "kind", "validated", "blurb", "note", "notes", "icon", "thumb",
        "photos", "files", "steps", "caption", "samples", "spec"}
REQUIRED = {"id", "title", "model", "family", "kind", "blurb", "icon", "photos", "files", "steps", "caption",
            "samples", "spec"}
FILES = recipes.files()


def test_there_are_recipes_and_the_shared_file_names_only_them():
    assert len(recipes.load()) == len(FILES)  # every shipped recipe passes the loader's checks
    common = recipes.shared()
    ids = [p.stem for p in FILES]
    assert ids and len(set(common["order"])) == len(common["order"])
    assert set(common["order"]) <= set(ids), "an id in _shared.json's order has no file"
    assert [r["id"] for r in recipes.load()][: len(common["order"])] == common["order"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_a_recipe_file_is_well_formed(path):
    r = json.loads(path.read_text("utf-8"))
    common = recipes.shared()
    assert r["id"] == path.stem
    assert REQUIRED <= set(r) <= KEYS, f"missing {REQUIRED - set(r)}, unknown {set(r) - KEYS}"
    assert r["kind"] in ("character", "style", "video", "general")
    assert isinstance(r["photos"], dict) or r["photos"] in common["photos"]
    assert set(r["files"]) <= {"diffusion_models", "text_encoders", "vae"} and all(r["files"].values())
    assert r["samples"] and all("[trigger]" in s for s in r["samples"])
    full = recipes.resolve(r, common)
    assert full["caption"]["prompt"] and full["caption"]["prompt"] not in common["captions"]  # a name, written out


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_a_recipes_settings_make_a_spec(path):
    # as Simple mode builds one (web/src/lib/recipes.ts buildSpec): its settings over the defaults,
    # with the model files, a dataset and its length
    r = json.loads(path.read_text("utf-8"))
    spec = json.loads(json.dumps(r["spec"]))
    first = {k: v[0] for k, v in r["files"].items()}
    # (the text encoder's type is the family's, from the catalog when it runs: any stands in here)
    spec["model"] = {"kind": "split", "unet_name": first["diffusion_models"], "clip_name": first.get("text_encoders"),
                     "clip_type": "stand-in", "vae_name": first.get("vae")}
    spec.setdefault("dataset", {}).update(folder="photos", trigger="mylora")
    spec.setdefault("optim", {})["steps"] = r["steps"]
    spec.setdefault("samples", {})["prompts"] = r["samples"]
    TrainSpec.from_dict({"name": r["id"], **spec}, strict=True)  # SpecError on a field or value it doesn't take


def test_a_bad_recipe_is_left_out_not_the_rest(tmp_path, monkeypatch):
    # what the API serves: only this folder's files, each loaded on its own; one that doesn't load
    # (bad JSON, an id that isn't its name, an unknown shared name, a card picture from another site)
    # is left out with a warning, the good one still served
    common = recipes.shared()
    (tmp_path / "_shared.json").write_text(json.dumps({**common, "order": ["good"]}))
    good = json.loads(FILES[0].read_text("utf-8"))
    (tmp_path / "good.json").write_text(json.dumps({**good, "id": "good", "thumb": {"src": "recipes/good.webp"}}))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "misnamed.json").write_text(json.dumps({**good, "id": "something-else"}))
    (tmp_path / "unknown.json").write_text(json.dumps({**good, "id": "unknown", "photos": "nope"}))
    for name, src in (("offsite", "//example.com/x.png"), ("scheme", "https://example.com/x.png"),
                      ("upward", "../../secret.png"), ("rooted", "/etc/x.png")):
        (tmp_path / f"{name}.json").write_text(json.dumps({**good, "id": name, "thumb": {"src": src}}))
    # valid JSON, a field of the wrong type: left out too, never served or raised past load()
    for name, change in (("captionobj", {"caption": {"prompt": {"x": 1}, "subject": True}}), ("photosnum", {"photos": 3}),
                         ("stepstext", {"steps": "2000"}), ("nosamples", {"samples": []}), ("kindodd", {"kind": "poster"}),
                         ("filesbad", {"files": {"loras": ["x.safetensors"]}}), ("notitle", {"title": None})):
        (tmp_path / f"{name}.json").write_text(json.dumps({**good, "id": name, **change}))
    monkeypatch.setattr(recipes, "DIR", tmp_path)
    assert [r["id"] for r in recipes.load()] == ["good"]
