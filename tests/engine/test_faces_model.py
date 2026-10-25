"""The face models: the setting takes only the models on offer, the files are found where downloads
put them, the networks still fit the published safetensors (their parameter names and shapes are
the file format), and the alignment and resizing steps do what InsightFace's OpenCV calls do."""

import hashlib
import sys
import types

import numpy as np
import pytest
from zentrainer import paths
from zentrainer.analysis import faces as analysis, files
from zentrainer.datasets import faces

# sha256 of each published file's sorted "key shape" lines (huggingface.co/nynxz/Zentrainer at faces.REVISION)
PUBLISHED = {"det_10g": ("scrfd", 119, "2a87549dcef38998"), "w600k_r50": ("r50", 237, "9a46d739d6acd1ab"),
             "glintr100": ("r100", 462, "ddd3074a93c13c9d")}


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(tmp_path / "comfy" / "input")
    fp.get_output_directory = lambda: str(tmp_path / "comfy" / "output")
    fp.get_user_directory = lambda: str(tmp_path / "comfy" / "user")
    fp.get_folder_paths = lambda _kind: []
    detection = tmp_path / "comfy" / "models" / "detection"

    def get_full_path(kind, name):
        p = detection / name
        return str(p) if kind == "detection" and p.is_file() else None

    fp.get_full_path = get_full_path
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    paths.reset()
    yield tmp_path / "comfy"
    paths.reset()


def test_the_setting_takes_only_the_models_on_offer(comfy):
    assert faces.model() == faces.DEFAULT == "buffalo_l"
    paths.save_settings({"face_model": "antelopev2"})
    assert faces.model() == "antelopev2" and paths.read_settings()["folder"] == ""
    with pytest.raises(paths.SettingsError):
        paths.save_settings({"face_model": "something_else"})
    assert faces.model() == "antelopev2"
    assert {m["id"] for m in paths.describe()["face_models"]} == set(faces.MODELS)


def test_a_model_is_downloaded_once_its_detector_and_recognition_files_are_there(comfy):
    d = comfy / "models" / "detection" / files.SUBDIR
    d.mkdir(parents=True)
    (d / analysis.DETECTOR).write_bytes(b"x")
    assert not analysis.downloaded("buffalo_l")
    (d / "w600k_r50.safetensors").write_bytes(b"x")
    assert analysis.downloaded("buffalo_l") and not analysis.downloaded("antelopev2")
    assert {m["id"]: m["downloaded"] for m in paths.describe()["face_models"]} == {"buffalo_l": True, "antelopev2": False}


@pytest.mark.parametrize("file", PUBLISHED)
def test_the_networks_fit_the_published_files(file):
    pytest.importorskip("torch")
    from zentrainer.analysis.nets.iresnet import IResNet
    from zentrainer.analysis.nets.scrfd import SCRFD

    kind, n, sig = PUBLISHED[file]
    net = SCRFD() if kind == "scrfd" else IResNet(kind)
    sd = {k: v for k, v in net.state_dict().items() if not k.endswith("num_batches_tracked")}
    lines = "\n".join(f"{k} {tuple(v.shape)}" for k, v in sorted(sd.items()))
    assert len(sd) == n and hashlib.sha256(lines.encode()).hexdigest()[:16] == sig
    assert f"{file}.safetensors" in analysis.SIZES


def test_the_networks_run_at_the_sizes_the_pipeline_feeds_them():
    torch = pytest.importorskip("torch")
    from zentrainer.analysis.nets.iresnet import IResNet
    from zentrainer.analysis.nets.scrfd import SCRFD

    with torch.inference_mode():
        outs = SCRFD().eval()(torch.zeros(1, 3, analysis.DET_SIZE, analysis.DET_SIZE))
        for (s, b, k), stride in zip(outs, analysis.STRIDES):
            cells = 2 * (analysis.DET_SIZE // stride) ** 2  # two anchors a cell
            assert s.shape == (cells, 1) and b.shape == (cells, 4) and k.shape == (cells, 10)
        assert IResNet("r50").eval()(torch.zeros(2, 3, 112, 112)).shape == (2, 512)


def test_the_alignment_finds_the_similarity_that_maps_the_keypoints():
    a = 0.3
    M = np.array([[1.7 * np.cos(a), -1.7 * np.sin(a), 12.0], [1.7 * np.sin(a), 1.7 * np.cos(a), -5.0]])
    src = np.array([[10, 20], [40, 22], [25, 35], [14, 50], [37, 49]], np.float64)
    dst = src @ M[:, :2].T + M[:, 2]
    assert np.allclose(analysis._similarity(src, dst), M, atol=1e-9)


def test_the_warp_samples_like_opencv():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    # a whole-pixel shift copies pixels; outside the picture is black
    out = analysis._warp(img, np.array([[1.0, 0, -3], [0, 1.0, -2]]), 30)
    assert (out == img[2:32, 3:33]).all()
    edge = analysis._warp(img, np.array([[1.0, 0, 45], [0, 1.0, 0]]), 30)
    assert (edge[:, :] == 0).all()
    # half a pixel: the average of the two neighbours, rounded
    half = analysis._warp(img, np.array([[1.0, 0, -0.5], [0, 1.0, 0]]), 20).astype(int)
    assert (np.abs(half - (img[:20, :20].astype(int) + img[:20, 1:21]) / 2) <= 0.5).all()


def test_the_resize_halves_and_shrinks_like_opencv():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    half = analysis._resize(img, 40, 32).astype(int)
    x = img.astype(int)
    assert (half == (x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2] + 2) // 4).all()
    flat = np.full((90, 120, 3), 77, np.uint8)
    assert (analysis._resize(flat, 47, 33) == 77).all()  # weights sum to one at any scale
    assert analysis._resize(img, 47, 33).shape == (33, 47, 3)
