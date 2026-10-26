"""Caption jobs: which images they take, that undo puts back exactly what they changed, and how
answers are generated (length limits, cut-off answers, re-rolling with a seed)."""

from pathlib import Path

import pytest
from PIL import Image

from zentrainer.datasets import captioner, folders as datasets
from zentrainer.harness.data import scan


@pytest.fixture
def ds(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    root.mkdir()
    for n in ("a", "b", "c"):
        Image.new("RGB", (64, 32)).save(root / f"{n}.png")
    (root / "a.txt").write_text("hand written a\n")
    (root / "b.nl.txt").write_text("old nl b\n")
    monkeypatch.setattr(captioner, "_file", lambda: tmp_path / "jobs.json")
    monkeypatch.setattr(captioner, "_dataset_root", lambda job: root)
    monkeypatch.setattr(captioner, "_jobs", {})
    monkeypatch.setattr(captioner, "_loaded", True)
    monkeypatch.setattr(captioner.events, "emit", lambda *a, **k: None)
    return root


def job(**kw):
    base = dict(dataset="ds", set_name="default", clip_name="x", clip_type="y", prompt="p")
    return captioner.create(**{**base, **kw})


def names(items):
    return sorted(it.path.name for it in items)


def test_modes_pick_images(ds):
    items = scan(ds, warn=False)
    assert names(captioner.todo(job(mode="missing"), items)) == ["b.png", "c.png"]
    assert names(captioner.todo(job(mode="missing", set_name="nl"), items)) == ["a.png", "c.png"]
    assert names(captioner.todo(job(mode="all"), items)) == ["a.png", "b.png", "c.png"]
    assert names(captioner.todo(job(mode="selected", paths=["c.png"]), items)) == ["c.png"]
    assert len(captioner.todo(job(mode="all", preview=2), items)) == 2


def write(j, ds, rel, text):
    """What run() does for one image."""
    from zentrainer.harness.data import caption_path

    target = caption_path(ds / rel, j.set_name)
    if target.is_file():
        j.previous[rel] = target.read_text()
    target.write_text(text + "\n")
    j.written[rel] = text


def test_undo_restores_and_removes(ds):
    j = job(mode="all")
    write(j, ds, "a.png", "auto a")  # replaced a hand-written caption
    write(j, ds, "c.png", "auto c")  # a new one
    j.status = "done"
    assert captioner.undo(j) == 2
    assert (ds / "a.txt").read_text() == "hand written a\n"
    assert not (ds / "c.txt").exists()
    assert j.status == "undone"


def test_undo_keeps_later_edits(ds):
    j = job(mode="all")
    write(j, ds, "a.png", "auto a")
    write(j, ds, "c.png", "auto c")
    (ds / "c.txt").write_text("fixed by hand\n")  # edited after the job
    j.status = "done"
    assert captioner.undo(j) == 1
    assert (ds / "a.txt").read_text() == "hand written a\n"
    assert (ds / "c.txt").read_text() == "fixed by hand\n"
    assert "kept 1" in j.message


def test_jobs_survive_a_restart(ds, monkeypatch):
    j = job(mode="missing")
    captioner.update(j, status="running", written={"c.png": "x"})
    monkeypatch.setattr(captioner, "_jobs", {})
    monkeypatch.setattr(captioner, "_loaded", False)
    back = captioner.get(j.job_id)
    assert back.written == {"c.png": "x"}
    assert back.status == "stopped"  # it was running when ComfyUI went down


def test_mask_stats(tmp_path):
    m = Image.new("L", (100, 50), 0)
    m.paste(255, (0, 0, 50, 50))  # left half trained
    path = tmp_path / "m.png"
    m.save(path)
    cover, w, h = datasets.mask_stats(Path(path))
    assert abs(cover - 0.5) < 0.02 and (w, h) == (100, 50)


def test_clean_all_states_keeps_resumable_runs(tmp_path, monkeypatch):
    from zentrainer.runs import disk as storage
    from zentrainer.runs.store import RunStore

    states = tmp_path / "states"
    monkeypatch.setattr(storage, "_roots", lambda: {"states": states})
    store = RunStore(tmp_path / "runs")
    for run_id, status in (("done1", "done"), ("stopped1", "stopped")):
        store.create(run_id, {"name": run_id})
        store.update(run_id, status=status)
        (states / run_id).mkdir(parents=True)
        (states / run_id / "s.pt").write_bytes(b"x" * 100)
    freed, runs = storage.clean_all_states(store)
    assert (freed, runs) == (100, 1)
    assert not (states / "done1").exists()
    assert (states / "stopped1" / "s.pt").exists()  # it can still resume


class _Clip:
    """A stand-in captioning model: an answer of `n` tokens, then (when it ends itself) an end token."""

    def __init__(self, n, ends=True):
        self.n, self.ends, self.calls = n, ends, []

    def generate(self, tokens, max_length, **kw):
        self.calls.append(dict(max_length=max_length, **kw))
        ids = [1] * min(self.n, max_length)
        return ids + [0] if self.ends and len(ids) < max_length else ids

    def decode(self, ids):
        return " ".join("w" for i in ids if i)


def test_an_answer_that_uses_every_token_is_cut_off(ds):
    assert captioner._answer(_Clip(50), None, job(max_length=300)) == (" ".join(["w"] * 50), False)
    text, cut = captioner._answer(_Clip(400, ends=False), None, job(max_length=300))
    assert cut and len(text.split()) == 300
    clip = _Clip(5000, ends=False)
    assert captioner._answer(clip, None, job(max_length=0))[1]  # "no limit" still stops at the cap
    assert clip.calls[-1]["max_length"] == captioner.NO_LIMIT


def test_a_seed_samples_and_no_seed_is_the_same_every_time(ds):
    clip = _Clip(10)
    captioner._answer(clip, None, job())
    captioner._answer(clip, None, job(seed=7))
    assert clip.calls[0]["do_sample"] is False
    assert clip.calls[1]["do_sample"] is True and clip.calls[1]["seed"] == 7


def test_sampling_settings_reach_the_model(ds):
    clip = _Clip(10)
    captioner._answer(clip, None, job(seed=3, temperature=1.1, top_p=0.8, top_k=20, repetition_penalty=1.1))
    call = clip.calls[-1]
    assert (call["temperature"], call["top_p"], call["top_k"], call["repetition_penalty"]) == (1.1, 0.8, 20, 1.1)


def test_hiding_masked_areas_shows_only_what_the_mask_keeps(tmp_path):
    im = Image.new("RGB", (100, 80), (200, 0, 0))  # the background: red
    im.paste((0, 0, 200), (60, 20, 90, 60))  # the subject: blue
    mask = Image.new("L", (100, 80), 0)
    mask.paste(255, (60, 20, 90, 60))
    seen = captioner.hide_masked(im, mask)
    pad = round(captioner.MASK_MARGIN * 100)
    assert seen.size == (min(100, 90 + pad) - (60 - pad), 60 + pad - (20 - pad))  # cropped to it, with a margin
    assert seen.getpixel((pad + 5, pad + 5)) == (0, 0, 200)  # the subject as it is
    assert seen.getpixel((1, 1)) == (captioner.MASK_GREY,) * 3  # the rest grey, no red left
    assert captioner.hide_masked(im, Image.new("L", (100, 80), 0)) is im  # keeps nothing: left as it was
    assert captioner.hide_masked(im, mask.resize((50, 40))).size == seen.size  # a mask at another size fits


def test_the_mask_note_is_only_for_masked_pictures():
    job = captioner.CaptionJob(job_id="", dataset="d", set_name="default", clip_name="c", clip_type="t",
                               prompt="Describe it.", hide_masked=True)
    assert captioner.MASK_NOTE in captioner.instruction(job, masked=True)
    assert captioner.MASK_NOTE not in captioner.instruction(job)
