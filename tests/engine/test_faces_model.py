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

# sha256 of each published file's sorted "key shape" lines (huggingface.co/nynxz/Zentrainer at faces.REVISIONS)
PUBLISHED = {"det_10g": ("scrfd", 119, "2a87549dcef38998"), "w600k_r50": ("r50", 237, "9a46d739d6acd1ab"),
             "glintr100": ("r100", 462, "ddd3074a93c13c9d"), "yunet_2023mar": ("yunet", 106, "754a6a421051bd5d"),
             "auraface_r100": ("r100", 462, "ddd3074a93c13c9d")}  # AuraFace is the same IResNet-100


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
    (d / "auraface_r100.safetensors").write_bytes(b"x")
    assert not analysis.downloaded("auraface")  # its own detector, not InsightFace's
    (d / "yunet_2023mar.safetensors").write_bytes(b"x")
    assert {m["id"]: m["downloaded"] for m in paths.describe()["face_models"]} == {"buffalo_l": True, "antelopev2": False, "auraface": True}


@pytest.mark.parametrize("file", PUBLISHED)
def test_the_networks_fit_the_published_files(file):
    pytest.importorskip("torch")
    from zentrainer.analysis.nets.iresnet import IResNet
    from zentrainer.analysis.nets.scrfd import SCRFD
    from zentrainer.analysis.nets.yunet import YuNet

    kind, n, sig = PUBLISHED[file]
    net = SCRFD() if kind == "scrfd" else YuNet() if kind == "yunet" else IResNet(kind)
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
        from zentrainer.analysis.nets.yunet import YuNet

        for (c, o, b, k), stride in zip(YuNet().eval()(torch.zeros(1, 3, analysis.DET_SIZE, analysis.DET_SIZE)), analysis.STRIDES):
            cells = (analysis.DET_SIZE // stride) ** 2  # one prediction a cell
            assert c.shape == o.shape == (1, cells, 1) and b.shape == (1, cells, 4) and k.shape == (1, cells, 10)


def test_yunet_decodes_as_opencv_does():
    """A face predicted in one cell: its box from the cell corner (centre offset, log size) and its
    keypoints from the corner, both in strides, back in the picture's pixels; the score is the root
    of class times objectness, and overlapping weaker boxes go."""
    torch = pytest.importorskip("torch")
    cells = [(analysis.DET_SIZE // s) ** 2 for s in analysis.STRIDES]
    logit = lambda p: float(np.log(p / (1 - p)))  # noqa: E731

    def det(_x):
        outs = []
        for n, stride in zip(cells, analysis.STRIDES):
            c, o = torch.full((1, n, 1), -20.0), torch.full((1, n, 1), -20.0)
            b, k = torch.zeros(1, n, 4), torch.zeros(1, n, 10)
            if stride == 16:  # row 3, col 5 of a 40-wide grid
                i = 3 * (analysis.DET_SIZE // 16) + 5
                c[0, i], o[0, i] = logit(0.81), logit(0.81)  # score 0.81
                b[0, i] = torch.tensor([0.5, 0.5, np.log(4.0), np.log(5.0)])  # centre (88, 56), 64 x 80: x 56-120, y 16-96
                k[0, i] = torch.tensor([0.0, 0.0, 1, 0, 0.5, 1, 0, 2, 1, 2])
                j = i + 1  # a weaker twin one cell right: suppressed
                c[0, j], o[0, j] = logit(0.7), logit(0.7)
                b[0, j] = torch.tensor([-0.5, 0.5, np.log(4.0), np.log(5.0)])
            outs.append((c, o, b, k))
        return outs

    rgb = np.zeros((analysis.DET_SIZE * 2, analysis.DET_SIZE * 2, 3), np.uint8)  # letterboxed at half size
    dets, kps = analysis._detect_yunet(det, rgb)
    assert dets.shape == (1, 5) and np.allclose(dets[0], [112, 32, 240, 192, 0.81], atol=1e-3)
    assert np.allclose(kps[0], np.array([[5, 3], [6, 3], [5.5, 4], [5, 5], [6, 5]]) * 16 * 2)


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
