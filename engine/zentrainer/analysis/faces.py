"""Faces in a picture: InsightFace's detection and recognition (FaceAnalysis with buffalo_l or
antelopev2) in plain PyTorch, on the CPU so it never competes with training for VRAM.

InsightFace's pipeline: the picture letterboxed to 640 px for SCRFD, its anchors decoded and
overlaps suppressed, each face aligned on its 5 keypoints to 112 px, then ArcFace. OpenCV's resize
is reproduced to the byte when shrinking, and the alignment warps the way OpenCV 5 does, so the
embeddings are InsightFace's (to a cosine of 0.99999).

The weights are InsightFace's pretrained models (non-commercial research use only) as safetensors on
Hugging Face, downloaded into models/detection/zentrainer/ the first time faces run.
"""

from __future__ import annotations

import threading

import numpy as np

REPO = "nynxz/Zentrainer"
REVISION = "b23239f14dfe1374452340866d66e3faa9b24066"  # the files below as converted: a later commit can't change them
DETECTOR = "det_10g.safetensors"
SIZES = {DETECTOR: 16914108, "w600k_r50.safetensors": 87203152, "glintr100.safetensors": 130354288}
# id -> (label, what it is, recognition file, IResNet depth); the first is the default
MODELS = {
    "buffalo_l": ("Standard (buffalo_l)", "ArcFace R50 trained on WebFace600K: quick on the CPU. A 104 MB download the first time.",
                  "w600k_r50.safetensors", "r50"),
    "antelopev2": ("Larger (antelopev2)", "ArcFace R100 trained on Glint360K: tells similar faces apart better, about 1.5 "
                                          "times slower on the CPU. A 147 MB download the first time.",
                   "glintr100.safetensors", "r100"),
}
DEFAULT = "buffalo_l"

DET_SIZE = 640
DET_THRESH = 0.5
NMS_THRESH = 0.4
STRIDES = (8, 16, 32)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

_nets: dict[str, object] = {}  # file -> loaded network
_lock = threading.RLock()


# ---- the model files ----------------------------------------------------------------------------

def _files(name: str) -> list[str]:
    return [DETECTOR, MODELS[name][2]]


def downloaded(name: str) -> bool:
    from .files import find

    return all(find("detection", f) for f in _files(name))


def _net(file: str, depth: str | None):
    """A network, loaded once (fp16 files run in fp32), downloading its file first if needed."""
    import torch
    from safetensors.torch import load_file

    from .nets.iresnet import IResNet
    from .nets.scrfd import SCRFD

    from .files import fetch

    with _lock:
        if file not in _nets:
            path = fetch("detection", file, REPO, REVISION, f"analysis/faces/{file}", SIZES[file], "faces")
            net = SCRFD() if depth is None else IResNet(depth)
            net.load_state_dict({k: v.float() for k, v in load_file(str(path)).items()})
            _nets[file] = net.eval().to(memory_format=torch.channels_last)
        return _nets[file]


# ---- OpenCV's resize and warp, without OpenCV ---------------------------------------------------

def _resize_coeffs(src: int, dst: int):
    """cv::resize's per-output source indices and 11-bit weights along one axis (INTER_LINEAR)."""
    f = ((np.arange(dst) + 0.5) * (src / dst) - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int64)
    f = f - s
    lo, hi = s < 0, s >= src - 1
    f[lo], s[lo] = 0, 0
    f[hi], s[hi] = 0, src - 1
    a1 = np.rint(f.astype(np.float64) * 2048).astype(np.int32)
    a0 = np.rint((1 - f.astype(np.float64)) * 2048).astype(np.int32)
    return s, np.minimum(s + 1, src - 1), a0, a1


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """cv2.resize(img, (w, h)) for uint8 (H, W, 3): the horizontal pass in ints, the vertical one
    rounded as OpenCV's SIMD loop rounds it (exact when shrinking; enlarging can differ by 1 on the
    first and last rows)."""
    sh, sw = img.shape[:2]
    if sw == 2 * w and sh == 2 * h:  # OpenCV takes its INTER_AREA path for an exact halving
        x = img.astype(np.uint16)
        return ((x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2] + 2) >> 2).astype(np.uint8)
    xs0, xs1, xa0, xa1 = _resize_coeffs(sw, w)
    ys0, ys1, yb0, yb1 = _resize_coeffs(sh, h)
    rows = np.unique(np.concatenate([ys0, ys1]))  # only the source rows the output reads
    src = img[rows]
    hz = src[:, xs0].astype(np.int32) * xa0[None, :, None] + src[:, xs1].astype(np.int32) * xa1[None, :, None]
    r0, r1 = hz[np.searchsorted(rows, ys0)] >> 4, hz[np.searchsorted(rows, ys1)] >> 4
    out = ((r0 * yb0[:, None, None]) >> 16) + ((r1 * yb1[:, None, None]) >> 16)
    return np.clip((out + 2) >> 2, 0, 255).astype(np.uint8)


def _warp(img: np.ndarray, M: np.ndarray, size: int) -> np.ndarray:
    """cv2.warpAffine(img, M, (size, size), borderValue=0) for uint8 (H, W, 3) as OpenCV 5 does it:
    bilinear with float weights."""
    inv = np.linalg.inv(np.vstack([M.astype(np.float64), [0, 0, 1]]))[:2]
    ys, xs = np.mgrid[:size, :size].astype(np.float64)
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    h, w = img.shape[:2]
    pad = np.zeros((h + 2, w + 2, img.shape[2]), np.float64)  # the black border around it
    pad[1:-1, 1:-1] = img
    tap = lambda y, x: pad[np.clip(y + 1, 0, h + 1), np.clip(x + 1, 0, w + 1)]
    v = ((1 - fy) * ((1 - fx) * tap(y0, x0) + fx * tap(y0, x0 + 1))
         + fy * ((1 - fx) * tap(y0 + 1, x0) + fx * tap(y0 + 1, x0 + 1)))
    out = np.clip(np.rint(v), 0, 255)
    out[(x0 < -1) | (x0 >= w) | (y0 < -1) | (y0 >= h)] = 0  # all four neighbours outside
    return out.astype(np.uint8)


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The similarity transform (2x3) taking src points onto dst (Umeyama, as skimage's
    SimilarityTransform estimates it)."""
    n, dim = src.shape
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    A = dd.T @ sd / n
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1)
    U, S, V = np.linalg.svd(A)
    if np.linalg.matrix_rank(A) == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s, d[dim - 1] = d[dim - 1], -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / sd.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dm - scale * (T[:dim, :dim] @ sm)
    T[:dim, :dim] *= scale
    return T[:2]


# ---- detection and recognition ------------------------------------------------------------------

def _nms(dets: np.ndarray) -> list[int]:
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= NMS_THRESH)[0] + 1]
    return keep


def _detect(det, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(boxes with their score (N, 5), keypoints (N, 5, 2)), best first."""
    import torch

    h, w = rgb.shape[:2]
    nh, nw = (DET_SIZE, int(DET_SIZE / (h / w))) if h / w > 1 else (int(DET_SIZE * (h / w)), DET_SIZE)
    scale = nh / h
    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), np.uint8)
    canvas[:nh, :nw] = _resize(rgb, nw, nh)
    blob = (canvas.astype(np.float32) - 127.5) * np.float32(1 / 128)
    outs = det(torch.from_numpy(blob).permute(2, 0, 1)[None].contiguous(memory_format=torch.channels_last))
    scores, boxes, kpss = [], [], []
    for (s, b, k), stride in zip(outs, STRIDES):
        s, b, k = s.numpy(), b.numpy() * stride, k.numpy() * stride
        n = DET_SIZE // stride
        c = (np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
        c = np.repeat(c, 2, axis=0)  # two anchors per cell
        pos = np.where(s >= DET_THRESH)[0]
        scores.append(s[pos])
        boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=-1)[pos])
        kpss.append(np.stack([c[:, i % 2] + k[:, i] for i in range(10)], axis=-1).reshape(-1, 5, 2)[pos])
    s = np.vstack(scores)
    if not len(s):
        return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
    order = s.ravel().argsort()[::-1]
    pre = np.hstack((np.vstack(boxes) / scale, s)).astype(np.float32)[order]
    kps = (np.vstack(kpss) / scale)[order]
    order = pre[:, 4].argsort()[::-1]  # InsightFace sorts the candidates again before suppressing
    pre, kps = pre[order], kps[order]
    keep = _nms(pre)
    return pre[keep], kps[keep]


def faces(rgb: np.ndarray, name: str) -> list[dict]:
    """The faces in an (H, W, 3) uint8 RGB picture, best detection first: {"bbox": [x0, y0, x1, y1]
    in pixels, "score", "emb": the unit ArcFace embedding (512,)} from face model `name`."""
    import torch

    _, _, file, depth = MODELS[name]
    with _lock, torch.inference_mode():  # one picture at a time: the CPU threads are shared
        dets, kpss = _detect(_net(DETECTOR, None), rgb)
        if not len(dets):
            return []
        crops = np.stack([_warp(rgb, _similarity(k.astype(np.float32), ARCFACE_DST), 112) for k in kpss])
        blob = (crops.astype(np.float32) - 127.5) * np.float32(1 / 127.5)
        embs = _net(file, depth)(torch.from_numpy(blob).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)).numpy()
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return [{"bbox": [float(v) for v in d[:4]], "score": float(d[4]), "emb": e.astype(np.float32)} for d, e in zip(dets, embs)]
