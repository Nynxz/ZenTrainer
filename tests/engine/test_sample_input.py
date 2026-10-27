"""An edit sample's input picture, found as training finds it: the run's dataset, ComfyUI's input,
or the path as given (a picture on another drive). The run page shows it from the same lookup."""

import sys
import types

from PIL import Image

from zentrainer import paths


def test_a_sample_input_is_found_where_training_finds_it(tmp_path, monkeypatch):
    ds, inp, elsewhere = tmp_path / "datasets" / "pairs", tmp_path / "input", tmp_path / "other-drive" / "eval"
    for d in (ds, inp / "held_out", elsewhere):
        d.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(ds / "a.png")
    Image.new("RGB", (8, 8)).save(inp / "held_out" / "b.png")
    Image.new("RGB", (8, 8)).save(elsewhere / "c_control.png")
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(inp)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setattr(paths, "find_dataset", lambda name: tmp_path / "datasets" / name)

    assert paths.sample_input("pairs", "a.png") == ds / "a.png"  # the dataset's
    assert paths.sample_input("pairs", "held_out/b.png") == inp / "held_out" / "b.png"  # ComfyUI's input
    # the path as given: what a Windows user's "C:/.../c_control.png" is (the page used to ask for
    # a dataset named "C:")
    full = (elsewhere / "c_control.png").as_posix()
    assert paths.sample_input("pairs", full) == elsewhere / "c_control.png"
    assert paths.sample_input("pairs", "missing.png") is None


def test_the_thumbnail_maker_reads_any_picture_file(tmp_path):
    from zentrainer.datasets import folders

    src = tmp_path / "far" / "c_control.png"
    src.parent.mkdir()
    Image.new("RGBA", (600, 300), (255, 0, 0, 128)).save(src)
    out = folders.thumbnail_of(src, 128, tmp_path / "cache")
    with Image.open(out) as im:
        assert im.format == "JPEG" and max(im.size) == 128
