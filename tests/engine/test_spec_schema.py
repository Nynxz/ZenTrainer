"""The spec's schema (spec.setting): every field documented, its types, choices and bounds checked
by validate() and published as JSON Schema and docs/spec.md, which stays current."""

import re
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from zentrainer import spec as S
from zentrainer.spec import SpecError, TrainSpec

BASE = {"model": {"kind": "checkpoint", "ckpt_name": "m.safetensors"}, "dataset": {"folder": "cats"}}


def _all_fields():
    for sec in fields(TrainSpec):
        sub = sec.default_factory() if callable(sec.default_factory) else None
        yield sec.name, sec
        if is_dataclass(sub):
            for f in fields(sub):
                yield f"{sec.name}.{f.name}", f


def test_every_field_says_what_it_does():
    missing = [name for name, f in _all_fields() if not f.metadata.get("doc")]
    assert not missing, f"fields without a setting() doc: {missing}"


def test_the_reference_is_current():
    # regenerate: python -m zentrainer.spec > docs/spec.md (in engine/)
    doc = Path(__file__).resolve().parents[2] / "docs" / "spec.md"
    assert doc.read_text() == S.reference()


def test_the_schema_file_is_current():
    # regenerate: python -m zentrainer.spec --schema > docs/spec.schema.json (in engine/)
    doc = Path(__file__).resolve().parents[2] / "docs" / "spec.schema.json"
    assert doc.read_text() == S.schema_json()


def test_a_spec_file_can_name_the_schema():
    s = TrainSpec.from_dict({"$schema": S.SCHEMA_URL, "spec_version": 2, **BASE})
    assert s.dataset.folder == "cats" and "$schema" not in s.to_dict()


def test_the_defaults_pass_their_own_checks():
    s = TrainSpec.from_dict(BASE)
    assert s.memory.compile_blocks == "auto" and s.validation.noise_levels == 4


@pytest.mark.parametrize("section,field,value,says", [
    ("optim", "steps", "1000", "optim.steps is whole number"),  # a number as text
    ("network", "rank", True, "network.rank is whole number"),  # a bool isn't a number
    ("optim", "learning_rate", 0, "optim.learning_rate must be > 0"),
    ("optim", "optimizer", "AdamWW", "optim.optimizer is one of"),
    ("memory", "attention", "flash", "memory.attention is one of"),
    ("memory", "compile_blocks", "yes", 'memory.compile_blocks is true/false or "auto"'),  # (null is v1's auto)
    ("network", "lora_dtype", "fp64", "network.lora_dtype is one of"),
    ("dataset", "caption_mix", "tags", "dataset.caption_mix is list of text"),
    ("objective", "logit_std", 0, "objective.logit_std must be > 0"),
])
def test_a_wrong_value_is_refused_by_name(section, field, value, says):
    with pytest.raises(SpecError, match=re.escape(says)):
        TrainSpec.from_dict({**BASE, section: {**BASE.get(section, {}), field: value}})


def test_whole_numbers_are_fine_where_numbers_go():
    assert TrainSpec.from_dict({**BASE, "optim": {"learning_rate": 1}}).optim.learning_rate == 1


def test_loading_fills_in_and_validate_only_checks():
    s = TrainSpec.from_dict({**BASE, "dataset": {"folder": "cats", "control_sets": ["input"]}})
    assert s.dataset.edit is True  # control sets mean edit training
    s = TrainSpec.from_dict({**BASE, "dataset": {"folder": "cats", "edit": True}})
    assert s.dataset.control_sets == ["control"]  # specs from before control sets
    before = s.to_dict()
    s.validate()
    assert s.to_dict() == before


def test_the_schema_has_every_field_with_its_rules():
    sch = S.schema()
    assert sch["properties"]["spec_version"]["const"] == S.SPEC_VERSION
    names = {name for name, _ in _all_fields()} | {"$schema"}
    seen = set(sch["properties"])
    for sec, body in sch["properties"].items():
        if body.get("type") == "object" and "properties" in body:
            seen |= {f"{sec}.{k}" for k in body["properties"]}
            assert body["additionalProperties"] is False
    assert seen == names
    opt = sch["properties"]["optim"]["properties"]
    assert opt["steps"] == {"type": "integer", "description": opt["steps"]["description"], "default": 1000, "minimum": 1}
    assert sch["properties"]["memory"]["properties"]["compile_blocks"]["enum"] == ["auto", True, False]


def test_the_harness_choices_match_the_specs():
    pytest.importorskip("torch")
    from zentrainer.harness import optim
    from zentrainer.harness.memory import TRAINABLE_ATTENTION

    assert set(S.OPTIMIZERS) == set(optim.OPTIMIZERS)
    assert S.TRAIN_ATTENTION == TRAINABLE_ATTENTION


def test_the_webs_field_list_is_current():
    # regenerate: python -m zentrainer.spec --ts > web/src/api/engine/spec.fields.ts (in engine/);
    # the web's type check then says which of its fields differ from the engine's
    doc = Path(__file__).resolve().parents[2] / "web" / "src" / "api" / "engine" / "spec.fields.ts"
    assert doc.read_text() == S.typescript()
