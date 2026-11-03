import json
import os
import time

import torch

from zentrainer.datasets import folders as datasets
from zentrainer.harness import data


def _cache(root, name, folder, texts, captions, age=0):
    """A dataset cache as _build writes one: its dataset.pt and source.json naming its captions."""
    d = root / name
    d.mkdir(parents=True)
    (d / "dataset.pt").write_bytes(b"x")
    (d / data.CACHE_SOURCE).write_text(json.dumps({"folder": folder, "text": texts.dir.name,
                                                   "captions": [texts.key(c) for c in captions]}))
    t = time.time() - age
    os.utime(d / "dataset.pt", (t, t))
    return d


def test_captions_are_encoded_once_and_shared(tmp_path):
    texts = data.TextCache(tmp_path, "encoder-a")
    cond = [[torch.ones(1, 3, 4), {"pooled_output": None}]]
    assert texts.get("a cat") is None
    texts.put("a cat", cond)
    again = data.TextCache(tmp_path, "encoder-a").get("a cat")  # another cache, the same encoder
    assert torch.equal(again[0][0], cond[0][0])
    assert data.TextCache(tmp_path, "encoder-b").get("a cat") is None  # not another encoder's


def test_pruning_keeps_the_encodings_a_cache_still_uses(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    for c in ("one", "two", "three"):
        texts.put(c, [[torch.zeros(1)]])
    old = time.time() - 2 * data.TEXT_GRACE
    for f in texts.dir.glob("*.pt"):
        os.utime(f, (old, old))
    _cache(tmp_path, "k1", "cats", texts, ["one", "two"])
    data.prune_text(tmp_path)
    assert {f.stem for f in texts.dir.glob("*.pt")} == {texts.key("one"), texts.key("two")}


def test_a_recent_encoding_waits_out_its_grace(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    texts.put("being built", [[torch.zeros(1)]])  # its cache isn't written yet
    data.prune_text(tmp_path)
    assert texts.get("being built") is not None


def test_clearing_a_dataset_leaves_other_folders_and_others_captions(tmp_path, monkeypatch):
    from zentrainer.runs import disk as storage

    monkeypatch.setattr(datasets, "root", lambda: (tmp_path / "input").resolve())
    (tmp_path / "input").mkdir()
    datasets.create("cats")
    cache = tmp_path / "cache"
    texts = data.TextCache(cache, "enc")
    for c in ("mine", "shared", "theirs"):
        texts.put(c, [[torch.zeros(1)]])
    _cache(cache, "k1", "cats", texts, ["mine", "shared"])
    _cache(cache, "k2", "dogs", texts, ["shared", "theirs"])
    for listing in ("hf", "prompts", "lora_thumbs"):  # no dataset's: never cleared with one
        (cache / listing).mkdir()
        (cache / listing / "x.json").write_text("{}")

    class Store:
        def list(self):
            return []

    monkeypatch.setattr(storage, "_roots", lambda: {"cache": cache})
    storage.clean_cache(Store(), "cats")
    assert sorted(p.name for p in cache.iterdir()) == ["hf", "k2", "lora_thumbs", "prompts", "text"]
    assert {f.stem for f in texts.dir.glob("*.pt")} == {texts.key("shared"), texts.key("theirs")}


def test_a_cache_from_before_caches_said_whose_goes_after_a_week(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    old = tmp_path / "old"
    old.mkdir()
    (old / "dataset.pt").write_bytes(b"x")  # no source.json
    t = time.time() - (data.ORPHAN_DAYS + 1) * 86400
    os.utime(old / "dataset.pt", (t, t))
    recent = tmp_path / "recent"
    recent.mkdir()
    (recent / "dataset.pt").write_bytes(b"x")
    _cache(tmp_path, "k1", "cats", texts, [])
    data.prune_caches(tmp_path, "cats")
    assert not old.exists() and recent.exists()


def test_a_dataset_saves_its_captions_by_name_and_loads_them_back(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    conds = [[[torch.full((1, 2), float(i))]] for i in range(2)]
    for c, cond in zip(["", "a cat"], conds):
        texts.put(c, cond)
    ds = data.Dataset([data.Sample(torch.zeros(4, 2, 2), 1, None, [1])], conds, ["", "a cat"], 0, owners=[0])
    plain = data._to_plain(ds, texts)
    assert plain["conditionings"] is None and plain["version"] == 4  # the encodings live in the text cache
    back = data._from_plain(plain, texts)
    assert torch.equal(back.conditionings[1][0][0], conds[1][0][0])
    old = data._to_plain(ds)  # an older cache, its encodings inside: still loads as it is
    assert data._from_plain(old).conditionings is conds


def test_caption_texts_are_what_a_build_encodes(tmp_path):
    from PIL import Image

    from zentrainer.spec import DatasetSpec

    for name, caption in (("a", "[trigger] in a hat"), ("b", "a red door")):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.txt").write_text(caption)
    spec = DatasetSpec(folder="x", trigger="zk", trigger_prepend=True)
    assert data.caption_texts(spec, tmp_path) == ["", "zk in a hat", "zk, a red door"]
    with_classes = data.caption_texts(spec, tmp_path, ("a man",))
    assert "a man in a hat" in with_classes and set(with_classes) > {"", "zk in a hat", "zk, a red door"}
