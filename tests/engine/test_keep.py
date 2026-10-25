"""Which checkpoints a run keeps (harness/keep.py, output.keep), and that removing one touches
nothing but that checkpoint's own files."""

import pytest

from zentrainer.harness import keep
from zentrainer.spec import SpecError, TrainSpec


def _ck(step, kind="step", final=False, run="r1"):
    name = "final" if final else f"{kind}{step:06d}"
    return {"step": step, "final": final, "path": f"/loras/zentrainer/checkpoints/{run}/x_{run}_{name}.safetensors"}


SAVES = [_ck(100), _ck(200), _ck(300), _ck(400), _ck(500)]


def test_all_keeps_everything():
    assert keep.to_prune(SAVES, "all", 1, [], []) == []


def test_latest_keeps_the_newest():
    gone = keep.to_prune(SAVES + [_ck(500, final=True)], "latest", 2, [], [])
    assert [c["step"] for c, _ in gone] == [300, 200, 100]


def test_hand_and_stop_saves_and_the_final_always_stay():
    rows = [_ck(100), _ck(150, "manual-step"), _ck(200), _ck(250, "stopped-step"), _ck(300), _ck(300, final=True)]
    assert [c["step"] for c, _ in keep.to_prune(rows, "latest", 1, [], [])] == [200, 100]


def test_the_newest_stays_whatever_its_score():
    val = [{"step": s, "val_loss": v} for s, v in ((100, 0.1), (200, 0.2), (300, 0.3), (400, 0.4), (500, 0.9))]
    gone = keep.to_prune(SAVES, "val_loss", 2, val, [])
    assert [c["step"] for c, _ in gone] == [300, 400]  # 100 and 200 kept, 500 is what a resume needs
    assert gone[0][1] == "validation loss 0.3000; the 2 lowest kept"


def test_likeness_ranks_highest_first_without_manual_tries_or_unscored():
    like = [{"step": 100, "score": 0.5}, {"step": 100, "score": 0.7}, {"step": 200, "score": 0.3},
            {"step": 300, "score": 0.9}, {"step": 300, "score": 0.1, "manual": True}, {"step": 400, "score": 0.65}]
    # means: 100 0.6, 200 0.3, 300 0.9 (manual left out), 400 0.65; 500 unscored: kept until it is
    gone = keep.to_prune(SAVES, "likeness", 2, [], like)
    assert [c["step"] for c, _ in gone] == [100, 200]
    assert gone[0][1] == "likeness 0.60; the 2 highest kept"


def test_new_best_keeps_every_save_that_beat_all_before_it():
    saves = SAVES + [_ck(600)]
    like = [{"step": s, "score": v} for s, v in ((100, 0.4), (200, 0.3), (300, 0.6), (400, 0.5), (500, 0.7), (600, 0.65))]
    gone = keep.to_prune(saves, "likeness_new_best", 1, [], like)
    # 100, 300 and 500 each beat all before them; 600 is the newest (a resume needs it)
    assert [c["step"] for c, _ in gone] == [200, 400]
    assert gone[1][1] == "likeness 0.50, not above 0.60 (step 300): only each new best kept"
    # the newest goes once it isn't the newest and didn't beat 500
    gone = keep.to_prune(saves + [_ck(700)], "likeness_new_best", 1, [], like + [{"step": 700, "score": 0.8}])
    assert [c["step"] for c, _ in gone] == [200, 400, 600]


def test_new_best_waits_for_an_unscored_save_before_judging_later_ones():
    like = [{"step": 100, "score": 0.4}, {"step": 300, "score": 0.3}, {"step": 400, "score": 0.5}]  # 200 unscored yet
    assert keep.to_prune(SAVES, "likeness_new_best", 1, [], like) == []


def test_new_best_by_validation_keeps_each_new_lowest():
    val = [{"step": s, "val_loss": v} for s, v in ((100, 0.5), (200, 0.4), (300, 0.45), (400, 0.3), (500, 0.35))]
    assert [c["step"] for c, _ in keep.to_prune(SAVES + [_ck(600)], "val_loss_new_best", 1, val, [])] == [300, 500]


def test_validation_at_a_checkpoint_is_the_latest_before_it():
    rows = [{"step": 0, "val_loss": 1.0}, {"step": 90, "val_loss": 0.5}, {"step": 150, "val_loss": 0.1}]
    assert keep.val_at(rows, 100) == 0.5


def test_remove_deletes_only_that_checkpoint(tmp_path):
    run = tmp_path / "loras" / "zentrainer" / "checkpoints" / "r1"
    states = tmp_path / "states" / "r1"
    run.mkdir(parents=True)
    states.mkdir(parents=True)
    lora, other = run / "x_r1_step000100.safetensors", run / "x_r1_step000200.safetensors"
    state, other_state = states / "x_r1_step000100.pt", states / "x_r1_step000200.pt"
    for f in (lora, other, state, other_state):
        f.write_bytes(b"12345")
    assert keep.remove({"path": str(lora), "state": str(state)}, "r1", states) == 10
    assert not lora.exists() and not state.exists()
    assert other.exists() and other_state.exists()


@pytest.mark.parametrize("where", ["another run", "not a checkpoint folder", "a hand save", "a state elsewhere"])
def test_remove_leaves_anything_else_alone(tmp_path, where):
    states = tmp_path / "states" / "r1"
    states.mkdir(parents=True)
    lora = {"another run": tmp_path / "checkpoints" / "r2" / "x_r2_step000100.safetensors",
            "not a checkpoint folder": tmp_path / "loras" / "r1" / "x_r1_step000100.safetensors",
            "a hand save": tmp_path / "checkpoints" / "r1" / "x_r1_manual-step000100.safetensors",
            "a state elsewhere": tmp_path / "checkpoints" / "r1" / "x_r1_step000100.safetensors"}[where]
    state = tmp_path / "elsewhere" / "x.pt" if where == "a state elsewhere" else states / "x.pt"
    for f in (lora, state):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"12345")
    freed = keep.remove({"path": str(lora), "state": str(state)}, "r1", states)
    if where == "a state elsewhere":  # its LoRA is this run's, so that goes; the state stays
        assert freed == 5 and not lora.exists() and state.exists()
    else:
        assert freed is None and lora.exists() and state.exists()


def _spec(**over):
    return {"name": "x", "dataset": {"folder": "d"}, "model": {"ckpt_name": "m.safetensors"}, **over}


def test_spec_defaults_to_keeping_everything():
    assert TrainSpec.from_dict(_spec()).output.keep == "all"


def test_spec_rules_need_their_scores():
    with pytest.raises(SpecError, match="validation"):
        TrainSpec.from_dict(_spec(output={"keep": "val_loss"}, validation={"every": 0}))
    with pytest.raises(SpecError, match="likeness"):
        TrainSpec.from_dict(_spec(output={"keep": "likeness"}))  # no sample prompts
    with pytest.raises(SpecError, match="keep_count"):
        TrainSpec.from_dict(_spec(output={"keep": "latest", "keep_count": 0}))
    TrainSpec.from_dict(_spec(output={"keep": "likeness"}, samples={"prompts": ["a photo of sks"]}))
    with pytest.raises(SpecError, match="likeness"):
        TrainSpec.from_dict(_spec(output={"keep": "likeness_new_best"}))
    with pytest.raises(SpecError, match="validation"):
        TrainSpec.from_dict(_spec(output={"keep": "val_loss_new_best"}, validation={"every": 0}))


def test_keeping_by_likeness_needs_faces_on_for_the_dataset(monkeypatch):
    from zentrainer.datasets import folders

    faces = {"d": False}
    monkeypatch.setattr(folders, "get_settings", lambda name: {"faces": faces[name]})
    for rule in ("likeness", "likeness_new_best"):
        spec = TrainSpec.from_dict(_spec(output={"keep": rule}, samples={"prompts": ["a photo of sks"]}, save_every=100))
        assert "Check faces" in keep.faces_missing(spec)  # without faces, no scores: it would keep everything
    faces["d"] = True
    assert keep.faces_missing(spec) is None
    faces["d"] = False
    assert keep.faces_missing(TrainSpec.from_dict(_spec(output={"keep": "latest"}, save_every=100))) is None
