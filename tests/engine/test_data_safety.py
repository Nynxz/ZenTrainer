"""Nothing of the user's gets overwritten, half-written or deleted by accident."""

import io

import pytest
from PIL import Image

from zentrainer.datasets import folders as datasets
from zentrainer import safeio
from zentrainer.runs.store import RunStore


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def ds(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "root", lambda: tmp_path.resolve())
    datasets.create("cats")
    return tmp_path / "cats"


def test_an_upload_never_replaces_a_different_file(ds):
    assert datasets.save_upload("cats", "IMG_0001.png", _png("red")) == "IMG_0001.png"
    datasets.set_caption("cats", "IMG_0001.png", "the first cat")
    # the same file again is a no-op; another camera's IMG_0001 gets its own name
    assert datasets.save_upload("cats", "IMG_0001.png", _png("red")) == "IMG_0001.png"
    assert datasets.save_upload("cats", "IMG_0001.png", _png("blue")) == "IMG_0001 (2).png"
    assert datasets.save_upload("cats", "IMG_0001.txt", b"a dropped caption") == "IMG_0001 (2).txt"
    assert (ds / "IMG_0001.txt").read_text().strip() == "the first cat"
    with Image.open(ds / "IMG_0001.png") as im:
        assert im.getpixel((0, 0)) == (255, 0, 0)


def test_deleting_a_picture_keeps_files_another_picture_owns(ds):
    for n in ("frame.png", "frame.001.png", "foo.png", "foo.jpg"):
        (ds / n).write_bytes(_png("red"))
    datasets.set_caption("cats", "frame.001.png", "frame one's own caption")  # frame.001.txt
    datasets.set_caption("cats", "frame.png", "frame")
    datasets.set_caption("cats", "foo.png", "shared by foo.png and foo.jpg")
    datasets.delete("cats", "frame.png")
    assert (ds / "frame.001.txt").is_file() and not (ds / "frame.txt").exists()
    datasets.delete("cats", "foo.png")
    assert (ds / "foo.txt").is_file()  # foo.jpg still uses it


def test_split_videos_never_replaces_an_earlier_original(ds):
    (ds / "_long").mkdir()
    (ds / "_long" / "talk.mp4").write_bytes(b"the first one")
    assert safeio.free_name(ds / "_long" / "talk.mp4").name == "talk (2).mp4"


def test_atomic_write_keeps_the_old_file_when_writing_fails(tmp_path, monkeypatch):
    f = tmp_path / "caption.txt"
    f.write_text("hand written")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(safeio.os, "fsync", boom)
    with pytest.raises(OSError):
        safeio.write_text(f, "new")
    assert f.read_text() == "hand written"
    assert [p.name for p in tmp_path.iterdir()] == ["caption.txt"]  # no temp left behind


def test_a_streamed_save_cut_short_keeps_the_old_file(tmp_path):
    f = tmp_path / "run.pt"
    f.write_bytes(b"the last good state")

    def half(out):  # torch.save killed halfway through
        out.write(b"half a ")
        raise OSError("disk full")

    with pytest.raises(OSError):
        safeio.write_with(f, half)
    assert f.read_bytes() == b"the last good state"
    assert [p.name for p in tmp_path.iterdir()] == ["run.pt"]
    safeio.write_with(f, lambda out: out.write(b"a new state"))
    assert f.read_bytes() == b"a new state"


def test_a_corrupt_settings_file_is_kept_not_overwritten(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text('{"cats": {"caption_subject": "Tom"')  # cut short
    assert safeio.read_json(f, {}) == {}
    kept = [p for p in tmp_path.iterdir() if p.name.startswith("settings.json.corrupt-")]
    assert len(kept) == 1 and "Tom" in kept[0].read_text()


def test_run_streams_survive_a_cut_short_line(tmp_path):
    store = RunStore(tmp_path)
    store.create("r1", {"name": "r1"})
    store.append("r1", "checkpoints", {"step": 250, "path": "a"})
    with open(store.dir("r1") / "checkpoints.jsonl", "a") as f:
        f.write('{"step": 500, "pa')  # the disk filled mid-append
    assert store.read("r1", "checkpoints") == [{"step": 250, "path": "a"}]
    store.truncate("r1", "checkpoints", 1000)
    assert store.read("r1", "checkpoints") == [{"step": 250, "path": "a"}]


def test_clearing_caches_only_touches_zentrainers_own(tmp_path, monkeypatch):
    import json as _json

    from zentrainer.runs import disk as storage
    from zentrainer.harness import data

    monkeypatch.setattr(datasets, "root", lambda: (tmp_path / "input").resolve())
    (tmp_path / "input").mkdir()
    datasets.create("cats")
    theirs = tmp_path / "input" / "cats" / "_latent_cache"  # another trainer's
    theirs.mkdir()
    (theirs / "x.npz").write_bytes(b"theirs")
    cache = tmp_path / "cache"
    for i, folder in enumerate(["cats", "cats", "cats", "cats", "dogs"]):
        d = cache / f"k{i}"
        d.mkdir(parents=True)
        (d / "dataset.pt").write_bytes(b"x")
        (d / data.CACHE_SOURCE).write_text(_json.dumps({"folder": folder}))
        __import__("os").utime(d / "dataset.pt", (i, i))
    data.prune_caches(cache, "cats")  # the 3 most recent stay
    assert sorted(p.name for p in cache.iterdir()) == ["k1", "k2", "k3", "k4"]

    class Store:
        def list(self):
            return []

    monkeypatch.setattr(storage, "_roots", lambda: {"cache": cache})
    storage.clean_cache(Store(), "cats")
    assert sorted(p.name for p in cache.iterdir()) == ["k4"]  # dogs' stays
    assert (theirs / "x.npz").is_file()


def _mask(size, white_box=None, black_box=None, base=0) -> bytes:
    m = Image.new("L", size, base)
    if white_box:
        m.paste(255, white_box)
    if black_box:
        m.paste(0, black_box)
    buf = io.BytesIO()
    m.save(buf, "PNG")
    return buf.getvalue()


def test_a_watermark_mask_applies_to_same_size_images_and_undoes(ds, tmp_path, monkeypatch):
    import base64

    monkeypatch.setattr(datasets, "_mask_undo", lambda name: tmp_path / "undo" / name)
    for n in ("a.png", "b.png", "c.png"):
        (ds / n).write_bytes(_png("red"))
    Image.new("RGB", (16, 16), "red").save(ds / "big.png")
    subject = _mask((8, 8), white_box=(0, 0, 4, 8))  # a's own mask: its left half
    (ds / "a-masklabel.png").write_bytes(subject)
    (ds / "_masks").mkdir()
    (ds / "_masks" / "c.png").write_bytes(_mask((8, 8), white_box=(0, 0, 8, 8)))  # the older layout
    watermark = "data:image/png;base64," + base64.b64encode(_mask((8, 8), black_box=(2, 6, 4, 8), base=255)).decode()

    out = datasets.apply_mask("cats", ["a.png", "b.png", "c.png", "big.png"], watermark, "cutout")
    assert [r.get("skipped", "") != "" for r in out["results"]] == [False, False, False, True]
    with Image.open(ds / "a-masklabel.png") as m:  # the subject stays, the box comes out of it
        assert m.getpixel((1, 1)) == 255 and m.getpixel((6, 1)) == 0 and m.getpixel((3, 7)) == 0
    with Image.open(ds / "b-masklabel.png") as m:  # no mask before: the watermark mask itself
        assert m.getpixel((6, 1)) == 255 and m.getpixel((3, 7)) == 0
    assert (ds / "c-masklabel.png").is_file() and not (ds / "_masks" / "c.png").exists()

    datasets.undo_masks("cats", out["batch"])
    assert (ds / "a-masklabel.png").read_bytes() == subject  # exactly as it was
    assert not (ds / "b-masklabel.png").exists()  # it had none
    assert (ds / "_masks" / "c.png").is_file() and not (ds / "c-masklabel.png").exists()
    with pytest.raises(datasets.DatasetError):
        datasets.undo_masks("cats", out["batch"])  # once
    with pytest.raises(datasets.DatasetError):
        datasets.undo_masks("cats", "../../etc")
