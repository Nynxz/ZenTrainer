"""A run's likeness rows carry their ceiling: how alike the dataset's own photos are (the face
report's median), so a score can be read as a share of what it can reach."""

import numpy as np
from zentrainer import paths
from zentrainer.datasets import faces


class _Store:
    def __init__(self):
        self.rows = []

    def append(self, run_id, stream, row):
        self.rows.append((stream, row))


def test_likeness_rows_carry_the_datasets_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(faces, "dataset_faces", lambda name, face_model=None: {"identity": 0.72, "one_identity": True})
    monkeypatch.setattr(faces, "dataset_centroid", lambda name, model=None: np.ones(512, np.float32))
    monkeypatch.setattr(faces, "likeness_detail", lambda path, centroid, face_model=None: (0.648, 1))
    monkeypatch.setattr(faces, "model", lambda: "buffalo_l")
    monkeypatch.setattr(paths, "resolve_view", lambda kind, sub, name: tmp_path / name)
    store = _Store()
    w = faces.LikenessWorker(store, "r1", "ds")
    w.submit({"step": 250, "index": 0, "subfolder": "s", "filename": "a.png"})
    w.close(wait=5)
    assert store.rows == [("likeness", {"run_id": "r1", "step": 250, "index": 0, "score": 0.648, "faces": 1,
                                        "model": "buffalo_l", "ceiling": 0.72})]


def test_sample_prompt_likeness_flag_skips_excluded_prompts():
    from types import SimpleNamespace
    import pytest
    from zentrainer.harness.callbacks import Samples
    from zentrainer.spec import SpecError, TrainSpec, sample_prompts

    base = {"model": {"kind": "checkpoint", "ckpt_name": "m.safetensors"}, "dataset": {"folder": "ds"}}
    spec = TrainSpec.from_dict({
        **base,
        "samples": {
            "prompts": [
                "[trigger], portrait",
                {"prompt": "a generic woman", "likeness": False},
            ],
        },
        "output": {"keep": "likeness"},
    })
    items = sample_prompts(spec.samples)
    assert [it.likeness for it in items] == [True, False]

    # Disabling likeness on every prompt fails validation when keep="likeness"
    with pytest.raises(SpecError, match="at least one sample prompt with likeness on"):
        TrainSpec.from_dict({
            **base,
            "samples": {"prompts": [{"prompt": "a generic woman", "likeness": False}]},
            "output": {"keep": "likeness"},
        })

    # Non-boolean likeness fails validation
    with pytest.raises(SpecError, match="true or false"):
        TrainSpec.from_dict({
            **base,
            "samples": {"prompts": [{"prompt": "a generic woman", "likeness": "no"}]},
        })

    # Samples._like skips index 1 (likeness=False), scores index 0, and still scores manual tries
    cb = Samples(every=100)
    cb.items = items
    submitted = []
    cb._liker = SimpleNamespace(submit=lambda row: submitted.append(row))
    t = SimpleNamespace(spec=spec, store=_Store())
    cb._like(t, {"step": 100, "index": 0, "filename": "a.png"})
    cb._like(t, {"step": 100, "index": 1, "filename": "b.png"})
    cb._like(t, {"step": 100, "index": 1, "filename": "c.png", "manual": True})
    assert [r["filename"] for r in submitted] == ["a.png", "c.png"]
