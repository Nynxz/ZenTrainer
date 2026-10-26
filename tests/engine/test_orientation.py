"""Pictures stored turned a quarter with an EXIF orientation (most phone photos): training loads
them upright, so their bucket, and the edit-input shape check, go by the upright size too."""

import logging
from pathlib import Path

import pytest
from PIL import Image

from zentrainer.harness import data
from zentrainer.spec import DatasetSpec

from test_disk_cache import FakeClip, FakeVAE


def _sideways(path: Path, upright: tuple[int, int]) -> None:
    """A JPEG of an upright w x h picture, stored turned a quarter (as a phone saves it)."""
    w, h = upright
    im = Image.new("RGB", (h, w), "red")  # stored landscape for a portrait
    exif = im.getexif()
    exif[0x0112] = 6  # "rotate 90 to show"
    im.save(path, exif=exif.tobytes())


@pytest.mark.parametrize("disk", [False, True])
def test_a_sideways_stored_portrait_trains_in_a_portrait_bucket(tmp_path, disk):
    ds = tmp_path / "phone"
    ds.mkdir()
    _sideways(ds / "a.jpg", (768, 1024))
    (ds / "a.txt").write_text("a person")
    spec = DatasetSpec(folder="phone", resolution=512, buckets=True, disk_cache=disk)
    built = data.build(spec, FakeClip(), FakeVAE(), tmp_path, tmp_path / "cache", "enc")
    (h, w) = built.samples[0].latent.shape[-2:]
    assert h > w  # portrait, as it's shown and trained (it was cropped to landscape)


@pytest.mark.parametrize("disk", [False, True])
def test_a_matched_edit_pair_draws_no_shape_warning(tmp_path, disk, caplog):
    # a 2:3 pair trains in an 832x1216-ish bucket, 2.6% off 2:3: the input matches its target,
    # so there's nothing to warn about (it was compared to the bucket)
    ds = tmp_path / "pairs"
    (ds / "_controls" / "input").mkdir(parents=True)
    Image.new("RGB", (400, 600), "red").save(ds / "a.png")
    Image.new("RGB", (400, 600), "white").save(ds / "_controls" / "input" / "a.png")
    _sideways(ds / "b.jpg", (400, 600))  # its input stored upright: the same shape as shown
    Image.new("RGB", (400, 600), "white").save(ds / "_controls" / "input" / "b.png")
    for n in ("a", "b"):
        (ds / f"{n}.txt").write_text(f"make {n} green")
    Image.new("RGB", (600, 400), "white").save(ds / "_controls" / "input" / "c.png")  # a wrong one
    Image.new("RGB", (400, 600), "blue").save(ds / "c.png")
    (ds / "c.txt").write_text("make c green")
    spec = DatasetSpec(folder="pairs", resolution=512, buckets=True, edit=True, control_sets=["input"],
                       edit_method="concat", disk_cache=disk)
    with caplog.at_level(logging.WARNING):
        data.build(spec, FakeClip(), FakeVAE(), tmp_path, tmp_path / "cache", "enc")
    warned = [r.getMessage() for r in caplog.records if "different shape" in r.getMessage()]
    assert len(warned) == 1 and "input/c.png" in warned[0]


def test_only_a_dataset_with_a_turned_picture_encodes_again(tmp_path, monkeypatch):
    upright, turned = tmp_path / "up.jpg", tmp_path / "turned.jpg"
    Image.new("RGB", (64, 96), "red").save(upright)
    _sideways(turned, (64, 96))
    assert data.upright_size(turned) == (64, 96) == data.upright_size(upright)
    spec = DatasetSpec(folder="x", resolution=64)

    def key(paths):
        return data.cache_key([data.Item(path=p, caption="c") for p in paths], spec, "enc")

    now = {"up": key([upright]), "turned": key([upright, turned])}
    monkeypatch.setattr(data, "_quarter_turned", lambda path: False)  # as the key was made before
    assert key([upright]) == now["up"]  # no turned picture: its cache stays
    assert key([upright, turned]) != now["turned"]  # one turned: it encodes again
