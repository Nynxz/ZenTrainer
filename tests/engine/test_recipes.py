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
