"""Dataset analysis: buckets as the trainer makes them (with EXIF rotation), caption phrases counted
once per caption with the subject left out, and the new picture checks (transparency, JPEG quality)."""

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402
from zentrainer.datasets import analysis, stats  # noqa: E402


def test_the_crop_is_what_the_trainer_cuts_off():
    from zentrainer.harness.data import bucket_sizes, pick_bucket

    options = bucket_sizes(1024, 64)
    square = pick_bucket(1000, 1000, options)
    assert analysis.kept_share(1000, 1000, square) == 1.0
    wide = (3000, 1000)  # 3:1
    assert 0.6 < analysis.kept_share(*wide, pick_bucket(*wide, options)) <= 1.0
    b = analysis.buckets({"a.png": (1000, 1000), "b.png": (1024, 1024), "c.png": (5000, 1000)}, 1024)  # 5:1: past 4:1
    assert b["rows"][0]["count"] == 2 and b["lonely"] == ["c.png"]
    assert [c["path"] for c in b["cropped"]] == ["c.png"]


def test_a_rotated_photo_is_measured_the_way_up_training_sees_it(tmp_path):
    path = tmp_path / "turned.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, shown portrait
    Image.new("RGB", (400, 300)).save(path, exif=exif)
    assert stats.size(path) == (300, 400)


def test_caption_phrases_count_each_caption_once_and_leave_the_subject_out():
    caps = [f"zorbo the fox wearing a white shirt, white shirt, in a park {i}" for i in range(4)]
    caps += ["zorbo the fox in a red hat by the sea", "a close-up of zorbo the fox"]
    out = analysis.captions(caps, subject="Zorbo")
    texts = {p["text"]: p for p in out["phrases"]}
    assert out["style"] == "prose" and out["captions"] == 6
    assert texts["white shirt"]["count"] == 4  # twice in a caption counts once
    assert "fox" in texts and not any("zorbo" in t for t in texts)
    assert "shirt" not in texts  # every mention is inside "white shirt"
    assert {p["text"] for p in out["frequent"]} >= {"fox", "white shirt"}


def test_tag_captions_are_read_as_tags():
    caps = ["1girl, solo, red hair, smile", "1girl, solo, red hair, outdoors", "1girl, red hair, night, rain"]
    out = analysis.captions(caps)
    assert out["style"] == "tags"
    assert {p["text"]: p["share"] for p in out["phrases"]}["red hair"] == 1.0


def test_transparency_and_heavy_jpeg_compression_are_seen(tmp_path):
    clear = tmp_path / "cutout.png"
    im = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    im.paste((0, 0, 0, 0), (0, 0, 100, 200))
    im.save(clear)
    rough, fine = tmp_path / "rough.jpg", tmp_path / "fine.jpg"
    Image.new("RGB", (200, 200), (90, 120, 200)).save(rough, quality=25)
    Image.new("RGB", (200, 200), (90, 120, 200)).save(fine, quality=95)
    assert stats._look(clear)["alpha"] == pytest.approx(0.5, abs=0.02)
    assert abs(stats._look(rough)["jpeg_q"] - 25) <= 3 and abs(stats._look(fine)["jpeg_q"] - 95) <= 3
    flags = stats.image_flags({str(clear): stats._look(clear), str(rough): stats._look(rough)}, {})
    assert "transparent" in flags[str(clear)] and "compressed" in flags[str(rough)]


def test_a_cutout_is_judged_by_what_shows_not_the_black_under_its_transparency(tmp_path):
    cutout = tmp_path / "cutout.png"
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))  # black under the see-through part, as most PNGs keep it
    im.paste((230, 200, 160, 255), (50, 30, 150, 170))
    im.paste((90, 60, 40, 255), (80, 60, 120, 140))  # some detail of its own: a plain one would be flat
    im.save(cutout)
    flags = stats.image_flags({str(cutout): stats._look(cutout)}, {})[str(cutout)]
    assert "transparent" in flags and not {"dark", "bars", "flat"} & set(flags)


def _scene(seed: int) -> Image.Image:
    """A picture with structure: random blocks of colour (a seed is a different picture)."""
    import random

    from PIL import ImageDraw

    rnd = random.Random(seed)
    im = Image.new("RGB", (320, 240), (120, 120, 120))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x, y = rnd.randrange(300), rnd.randrange(220)
        d.rectangle((x, y, x + rnd.randrange(20, 140), y + rnd.randrange(20, 120)),
                    fill=(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    return im


def test_copies_pair_up_and_different_pictures_dont(tmp_path):
    from PIL import ImageOps

    paths = {}
    original = _scene(1)
    for name, im in {"a.png": original, "a_copy.jpg": original.resize((160, 120)), "a_flipped.png": ImageOps.mirror(original),
                     "b.png": _scene(2), "c.png": _scene(3)}.items():
        paths[name] = tmp_path / name
        im.save(paths[name], quality=70) if name.endswith(".jpg") else im.save(paths[name])
    looks = {str(p): stats._look(p) for p in paths.values()}
    found = {(min(q["a"], q["b"]), max(q["a"], q["b"])): q for q in stats.pairs(looks)}
    pa, copy, flip = str(paths["a.png"]), str(paths["a_copy.jpg"]), str(paths["a_flipped.png"])
    assert found[tuple(sorted((pa, copy)))]["kind"] == "near-duplicate"
    assert found[tuple(sorted((pa, flip)))]["mirrored"]
    assert not any(str(paths["b.png"]) in k or str(paths["c.png"]) in k for k in found)
    kinds = stats.pair_kinds(list(found.values()))
    assert kinds[pa] == "near-duplicate" and str(paths["b.png"]) not in kinds


def test_bars_are_letterboxing_not_a_plain_sky(tmp_path):
    from PIL import ImageDraw

    boxed = _scene(4)
    letterbox = Image.new("RGB", (320, 320), (0, 0, 0))
    letterbox.paste(boxed, (0, 40))  # black above and below
    sky = Image.new("RGB", (320, 240), (200, 215, 230))  # a plain sky over a busy ground
    ImageDraw.Draw(sky).rectangle((0, 150, 320, 240), fill=(90, 110, 60))
    for i in range(0, 320, 7):
        ImageDraw.Draw(sky).line((i, 150, i + 30, 240), fill=(40, 60, 30), width=2)
    fade = Image.linear_gradient("L").resize((320, 240)).convert("RGB")  # a smooth gradient top to bottom
    for name, im in (("letterbox.png", letterbox), ("sky.png", sky), ("fade.png", fade)):
        im.save(tmp_path / name)
    assert stats._look(tmp_path / "letterbox.png")["border"] > 0.2
    assert stats._look(tmp_path / "sky.png")["border"] == 0
    assert stats._look(tmp_path / "fade.png")["border"] == 0
