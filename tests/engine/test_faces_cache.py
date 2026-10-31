import os
import types

from zentrainer.datasets import faces


def _setup(tmp_path, monkeypatch):
    """A two-picture dataset with the scan, the face detector and the cache folder stood in."""
    root = tmp_path / "ds"
    root.mkdir()
    for n in ("a.png", "b.png"):
        (root / n).write_bytes(b"picture " + n.encode())
    cache_dir = tmp_path / "faces"
    cache_dir.mkdir()
    looked = []

    def frame_faces(path, name=None):
        looked.append(path.name)
        return [[{"box": [0.1, 0.1, 0.5, 0.5], "share": 0.2, "emb": [1.0, 0.0]}]]

    from zentrainer.datasets import folders
    from zentrainer.harness import data

    monkeypatch.setattr(folders, "folder", lambda name: root)
    monkeypatch.setattr(data, "scan", lambda r, warn=False: [types.SimpleNamespace(path=p, mask=None) for p in sorted(r.glob("*.png"))])
    monkeypatch.setattr(faces, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(faces, "model", lambda: faces.DEFAULT)
    monkeypatch.setattr(faces, "frame_faces", frame_faces)
    return root, cache_dir, looked


def _check(name="ds"):
    seen = []
    out = faces.dataset_faces(name, progress=lambda done, total: seen.append((done, total)))
    return out, seen


def test_a_recheck_of_an_unchanged_dataset_looks_at_nothing_and_reports_no_progress(tmp_path, monkeypatch):
    root, cache_dir, looked = _setup(tmp_path, monkeypatch)
    out, seen = _check()
    assert sorted(looked) == ["a.png", "b.png"] and seen == [(1, 2), (2, 2)] and out["total"] == 2
    (cache_file,) = cache_dir.iterdir()
    os.utime(cache_file, ns=(1, 1))  # an old stamp: a rewrite would move it

    looked.clear()
    out, seen = _check()
    assert looked == [] and seen == []  # cached: no detection, nothing to count
    assert out["total"] == 2 and out["with_face"] == 2
    assert cache_file.stat().st_mtime_ns == 1  # and the cache wasn't written again


def test_only_a_changed_picture_is_looked_at_and_counted(tmp_path, monkeypatch):
    root, cache_dir, looked = _setup(tmp_path, monkeypatch)
    _check()
    (root / "b.png").write_bytes(b"a different picture, longer")
    looked.clear()
    _, seen = _check()
    assert looked == ["b.png"] and seen == [(1, 1)]


def test_a_removed_picture_leaves_the_cache(tmp_path, monkeypatch):
    root, cache_dir, looked = _setup(tmp_path, monkeypatch)
    _check()
    (root / "a.png").unlink()
    looked.clear()
    out, seen = _check()
    import json

    (cache_file,) = cache_dir.iterdir()
    assert looked == [] and seen == [] and out["total"] == 1
    assert list(json.loads(cache_file.read_text())) == ["b.png"]
