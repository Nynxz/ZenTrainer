"""Spec versions (spec.migrate): an older layout renamed into this one as it's read, wherever a spec
comes from (stored runs, presets, profiles, shared files), and fields no version has kept apart."""

import json

import pytest

from zentrainer import presets, sharing
from zentrainer.runs.store import RunStore
from zentrainer.spec import SPEC_VERSION, SpecError, TrainSpec, migrate

MODEL = {"kind": "split", "unet_name": "u", "clip_name": "c", "clip_type": "t", "vae_name": "v"}
V1 = {"name": "old", "save_every": 100, "model": MODEL,
      "dataset": {"folder": "d", "keyframes": "first", "clip_stills": True},
      "optim": {"scheduler": "cosine"}, "objective": {"scheduler": "beta", "scheduler_steps": 8},
      "memory": {"w8a8_backward": False, "fused_int8_backward": True, "quantized_backward": True}}


def test_a_v1_spec_reads_as_v2():
    s = TrainSpec.from_dict(V1)
    assert s.spec_version == SPEC_VERSION == 2
    assert s.output.save_every == 100 and s.optim.lr_scheduler == "cosine"
    assert (s.objective.timestep_scheduler, s.objective.timestep_scheduler_steps) == ("beta", 8)
    assert s.memory.int8_backward == "fused" and s.memory.fp8_backward
    assert s.dataset.video_keyframes == "first" and s.dataset.video_stills
    assert TrainSpec.from_dict(s.to_dict()) == s  # v2 round-trips as it is


def test_v1s_two_int8_switches_become_one_choice():
    mode = lambda **m: TrainSpec.from_dict({**V1, "memory": m}).memory.int8_backward
    assert mode() == mode(w8a8_backward=True) == mode(w8a8_backward=True, fused_int8_backward=True) == "w8a8"
    assert mode(w8a8_backward=False, fused_int8_backward=True) == "fused"
    assert mode(w8a8_backward=False) == "bf16"


def test_a_field_no_version_has_is_an_error_when_written_and_left_out_when_stored():
    gone = {**V1, "optim": {"scheduler": "cosine", "anchor_decay": 0.9}}
    with pytest.raises(SpecError, match="optim.anchor_decay"):
        TrainSpec.from_dict(gone)  # a typo in a spec written by hand
    assert TrainSpec.from_dict(gone, strict=False).optim.lr_scheduler == "cosine"
    assert migrate(gone)[1] == ["optim.anchor_decay"]
    with pytest.raises(SpecError, match="unknown"):  # a v2 spec with a v1 name isn't migrated: it's a typo
        TrainSpec.from_dict({**TrainSpec.from_dict(V1).to_dict(), "save_every": 5})


def test_a_newer_spec_says_to_update():
    with pytest.raises(SpecError, match="newer ZenTrainer"):
        migrate({**V1, "spec_version": SPEC_VERSION + 1})


def test_fragments_stay_fragments():
    assert migrate({"optim": {"scheduler": "linear"}, "save_every": 0}) == (
        {"optim": {"lr_scheduler": "linear"}, "output": {"save_every": 0}}, [])


def test_stored_runs_presets_and_shared_files_come_in_this_layout(tmp_path):
    store = RunStore(tmp_path / "runs")
    run_id = "r1"
    store.create(run_id, TrainSpec.from_dict(V1).to_dict())
    (store.dir(run_id) / "spec.json").write_text(json.dumps({**V1, "optim": {"anchor_decay": 1}}))  # as v1 left it
    spec = store.spec(run_id)
    assert spec["spec_version"] == 2 and spec["output"]["save_every"] == 100 and "anchor_decay" not in spec["optim"]
    TrainSpec.from_dict(spec)  # loads strict: a resume or a duplicate works

    (tmp_path / "presets" / "run").mkdir(parents=True)
    (tmp_path / "presets" / "run" / "mine.json").write_text(json.dumps({"name": "Mine", "data": {"save_every": 50}}))
    mine = next(p for p in presets.list_presets("run", tmp_path / "presets") if p["id"] == "mine")
    assert mine["data"] == {"output": {"save_every": 50}}

    got = sharing.read({**V1, "optim": {"scheduler": "linear", "anchor_decay": 1}})
    assert got["spec"]["optim"] == {"lr_scheduler": "linear"} and any("anchor_decay" in w for w in got["warnings"])


def test_the_new_names_without_a_version_and_old_beside_new():
    v2 = TrainSpec.from_dict(V1).to_dict()
    v2.pop("spec_version")  # written by hand with the new names, no version: read as it is
    assert TrainSpec.from_dict(v2).output.save_every == 100
    both = {**V1, "output": {"save_every": 7}}  # v1's top-level one beside v2's: the v2 one stands
    assert TrainSpec.from_dict(both).output.save_every == 7


def test_what_isnt_renamed_stays():
    s = TrainSpec.from_dict({**V1, "samples": {"scheduler": "karras", "steps": 9, "prompts": [{"prompt": "a", "seed": 3}]}})
    assert s.samples.scheduler == "karras" and s.samples.prompts == [{"prompt": "a", "seed": 3}]  # sampling's own


def test_nulls_mean_what_the_form_showed():
    mem = lambda m: TrainSpec.from_dict({**V1, "memory": m}).memory
    assert mem({"w8a8_backward": None}).int8_backward == "w8a8"
    assert mem({"int8_forward": None}).int8_forward is True
    m = mem({"compile_blocks": None, "fused_qk_rope": None, "checkpoint_skip": 2})  # v1's null meant auto
    assert m.compile_blocks == m.fused_qk_rope == "auto" and m.grad_checkpoint_skip == 2
    s = TrainSpec.from_dict({**V1, "output": None})  # a null section, and v1's save_every to move into it
    assert s.output.save_every == 100 and s.output.keep == "all"


def test_v2s_first_draft_keeps_its_settings():
    # saved by v2 before these were renamed (unreleased, but on disk): carried over, not dropped
    d = {"spec_version": 2, "model": MODEL, "dataset": {"folder": "d"}, "validation": {"timesteps": 6},
         "memory": {"checkpoint_skip": 4, "compile_blocks": None, "fused_qk_rope": False}}
    s = TrainSpec.from_dict(d)
    assert s.validation.noise_levels == 6 and s.memory.grad_checkpoint_skip == 4
    assert s.memory.compile_blocks == "auto" and s.memory.fused_qk_rope is False
