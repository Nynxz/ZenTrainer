"""Faces in datasets and samples (the faces themselves come from analysis/faces.py):

- A dataset's face report: how many faces each picture has, how big the main one is, and how much
  it looks like the rest (cosine to the dataset's average face). Outliers stand out. A face mostly
  in the black (not trained) part of a picture's mask doesn't count: mask out a bystander and they
  stop being an "extra face" (with Masked training on, they aren't trained either).
- Likeness: how much a sample's face looks like the dataset's, per sample and checkpoint, so the
  best checkpoint of a character LoRA is a number instead of a guess.

The recognition model is a setting (face_model): buffalo_l's ArcFace R50, antelopev2's larger
R100, which tells similar faces apart better and is slower on the CPU, or AuraFace with YuNet for
commercial use (InsightFace's are non-commercial). Scores from different models don't compare, so
caches are per model and each likeness row says which one scored it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

from ..analysis.faces import DEFAULT, MODELS

MASKED_OUT = 0.5  # a face whose box averages below this in the picture's mask is masked out
MAX_FACES = 6  # faces kept per frame (largest first)
OUTLIER = 0.3  # cosine to the other pictures' average face below this: probably not the same person
ONE_IDENTITY = 0.35  # the set's median below this: its faces aren't one person (likeness means little)


def model() -> str:
    """The face model the settings pick."""
    from .. import paths

    try:
        name = paths.read_settings().get("face_model")
    except (ImportError, AttributeError):  # outside ComfyUI (tests, tools)
        name = None
    return name if name in MODELS else DEFAULT


def detect(rgb: np.ndarray, name: str | None = None) -> list[dict]:
    """Faces in an (H, W, 3) uint8 RGB picture, largest first: bbox, detector score, the share of
    the picture's height the face spans, and its unit ArcFace embedding (from `name`, or the
    model the settings pick)."""
    from ..analysis import faces

    h = rgb.shape[0]
    out = [{**f, "share": (f["bbox"][3] - f["bbox"][1]) / max(h, 1)} for f in faces.faces(rgb, name or model())]
    return sorted(out, key=lambda f: -f["share"])


def _frames(path: Path, n: int) -> list[np.ndarray]:
    """A picture, or n frames spread over a clip, as uint8 RGB."""
    from ..harness.data import video

    if video.is_video(path):
        p = video.probe(path)
        if p is None or not p.frames:
            return []
        idx = sorted({min(p.frames - 1, round((p.frames - 1) * (i + 0.5) / n)) for i in range(n)})
        scale = min(1.0, 1024 / max(p.width, p.height))
        px = video.read(path, idx, (max(32, round(p.width * scale)), max(32, round(p.height * scale))))
        return [(f.numpy() * 255).astype(np.uint8) for f in px]
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600))
        return [np.asarray(im)]


def frame_faces(path: Path, frames: int = 3, name: str | None = None) -> list[list[dict]]:
    """Per frame, its faces (largest first) with boxes as fractions of the picture, so a mask of
    any size can be laid over them later: [[{"box": [x0, y0, x1, y1], "share", "emb"}, ...], ...]."""
    out = []
    for rgb in _frames(path, frames):
        h, w = rgb.shape[:2]
        out.append([{"box": [f["bbox"][0] / w, f["bbox"][1] / h, f["bbox"][2] / w, f["bbox"][3] / h],
                     "share": f["share"], "emb": f["emb"].tolist()} for f in detect(rgb, name)[:MAX_FACES]])
    return out


def _mask_array(path: Path | None) -> np.ndarray | None:
    """A mask as a (256, 256) float array in [0, 1] (its own aspect doesn't matter: boxes are fractions)."""
    if path is None:
        return None
    from PIL import Image

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize((256, 256), Image.BILINEAR), np.float32) / 255
    except OSError:
        return None


def _trained(box: list[float], mask: np.ndarray | None) -> bool:
    """Whether a face's box is mostly in the trained (white) part of the mask."""
    if mask is None:
        return True
    n = mask.shape[0]
    x0, y0 = max(0, int(box[0] * n)), max(0, int(box[1] * n))
    x1, y1 = min(n, max(x0 + 1, int(np.ceil(box[2] * n)))), min(n, max(y0 + 1, int(np.ceil(box[3] * n))))
    return float(mask[y0:y1, x0:x1].mean()) >= MASKED_OUT


def _summarize(frames: list[list[dict]], mask: np.ndarray | None) -> dict:
    """A picture's faces after its mask: how many trained faces in its busiest frame, how many masked
    out, and the largest trained face (its share and embedding)."""
    most = masked = 0
    best = None
    for fs in frames:
        kept = [f for f in fs if _trained(f["box"], mask)]
        most = max(most, len(kept))
        masked = max(masked, len(fs) - len(kept))
        if kept and (best is None or kept[0]["share"] > best["share"]):
            best = kept[0]
    return {"faces": most, "masked": masked, "share": best["share"] if best else 0.0, "emb": best["emb"] if best else None}


# ---- a dataset's faces --------------------------------------------------------------------------

def _cache_dir() -> Path:
    import folder_paths

    d = Path(folder_paths.get_user_directory()) / "zentrainer" / "faces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_faces(name: str, progress=None, face_model: str | None = None) -> dict:
    """{"items": {rel: {"faces", "masked", "share", "likeness", "outlier"}}, "with_face", "total"};
    each picture's faces are cached by its mtime and size, so a re-check only looks at what changed
    (masks are applied afresh each time: redrawing one needs no detection)."""
    out, _ = _report(name, progress, face_model)
    return out


def _report(name: str, progress=None, face_model: str | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """The face report, and each picture's main trained face embedding (for the centroid)."""
    from . import folders as datasets
    from ..harness.data import scan

    fm = face_model or model()
    root = datasets.folder(name)
    key = hashlib.sha1(name.encode()).hexdigest()[:16]
    # one cache per model, so switching back finds the old one (buffalo_l's keeps its first name)
    cache_file = _cache_dir() / (f"{key}.json" if fm == DEFAULT else f"{key}.{fm}.json")
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    items = scan(root, warn=False)
    rels = [str(it.path.relative_to(root)) for it in items]
    stamps = {}
    for rel, it in zip(rels, items):
        st = it.path.stat()
        stamps[rel] = f"{fm}|2|{st.st_mtime_ns}|{st.st_size}"  # 2: every face per frame, boxes as fractions
    # only what's new or changed is looked at, and only that counts as progress: re-checking an
    # unchanged dataset reports none, so the page doesn't show a scan that isn't happening
    todo = [(rel, it) for rel, it in zip(rels, items) if (cache.get(rel) or {}).get("stamp") != stamps[rel]]
    for k, (rel, it) in enumerate(todo):
        cache[rel] = {"stamp": stamps[rel], "frames": frame_faces(it.path, name=fm)}
        if progress:
            progress(k + 1, len(todo))
    rows = {rel: _summarize(cache[rel]["frames"], _mask_array(it.mask)) for rel, it in zip(rels, items)}
    gone = [k for k in cache if k not in rows]
    if todo or gone:  # unchanged: the file stays as it is
        for k in gone:
            del cache[k]
        cache_file.write_text(json.dumps(cache))
    embs = {rel: np.asarray(r["emb"], np.float32) for rel, r in rows.items() if r["emb"]}
    likes = _leave_one_out(embs)
    out = {}
    for rel, r in rows.items():
        like = likes.get(rel)
        out[rel] = {"faces": r["faces"], "masked": r["masked"], "share": round(r["share"], 4),
                    "likeness": round(like, 4) if like is not None else None,
                    "outlier": like is not None and like < OUTLIER}
    vals = sorted(likes.values())
    median = vals[len(vals) // 2] if vals else None
    return {"items": out, "with_face": len(embs), "total": len(rows), "pack": fm,
            "identity": round(median, 4) if median is not None else None,
            "one_identity": median is not None and median >= ONE_IDENTITY}, embs


def _leave_one_out(embs: dict[str, np.ndarray]) -> dict[str, float]:
    """Each face against the average of the OTHER faces. Against an average that includes it, a face
    always looks alike (with 4 different people, each is a quarter of the average: ~0.5 regardless)."""
    if len(embs) < 2:
        return {}
    keys = list(embs)
    e = np.stack([embs[k] for k in keys])
    total = e.sum(0)
    out = {}
    for i, k in enumerate(keys):
        others = total - e[i]
        out[k] = float(np.dot(e[i], others / (np.linalg.norm(others) + 1e-9)))
    return out


def _centroid(embs: list[np.ndarray]) -> np.ndarray | None:
    """The dataset's average face for scoring samples, without the strays (faces that don't look like
    the rest, each judged against the others), so a few don't pull it."""
    if not embs:
        return None
    loo = _leave_one_out({str(i): e for i, e in enumerate(embs)})
    keep = [e for i, e in enumerate(embs) if loo.get(str(i), 1.0) >= OUTLIER] if loo else embs
    if len(keep) < max(1, len(embs) // 2):
        keep = embs
    c = np.stack(keep).mean(0)
    return c / (np.linalg.norm(c) + 1e-9)


def dataset_centroid(name: str, face_model: str | None = None) -> np.ndarray | None:
    _, embs = _report(name, face_model=face_model)  # cached: cheap after the first look
    return _centroid(list(embs.values()))


def likeness(path: Path, centroid: np.ndarray, frames: int = 4, face_model: str | None = None) -> float | None:
    """How much a rendered sample's main face looks like the dataset's (cosine, -1..1; same
    person is typically above 0.5): the best face in each of a few frames, averaged. The centroid
    must come from the same face model."""
    scores = []
    for rgb in _frames(path, frames):
        fs = detect(rgb, face_model)
        if fs:
            scores.append(max(float(np.dot(f["emb"], centroid)) for f in fs[:3]))
    return round(float(np.mean(scores)), 4) if scores else None


class LikenessWorker:
    """Scores a run's samples off the training thread (CPU), appending {"step", "index", "score",
    "model", "ceiling"} rows to the run's "likeness" stream as each finishes. The face model is the
    one set when the run started, for the whole run, so its scores compare with each other.
    `ceiling` is how alike the dataset's own pictures are (their median against the others): a
    LoRA that caught the face perfectly would still score about that, so a score reads best as a
    share of it."""

    def __init__(self, store, run_id: str, dataset: str):
        import queue

        self.store, self.run_id, self.dataset = store, run_id, dataset
        self.model = model()
        self.q: queue.Queue = queue.Queue()
        self.centroid = None
        self.ceiling = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="zentrainer-likeness")
        self.thread.start()

    def submit(self, row: dict) -> None:
        self.q.put(row)

    def close(self, wait: float = 0) -> None:
        """Stop once what's queued is scored; `wait`: up to that many seconds for it."""
        self.q.put(None)
        if wait:
            self.thread.join(wait)

    def _run(self) -> None:
        from .. import paths

        try:
            report = dataset_faces(self.dataset, face_model=self.model)
            self.ceiling = report.get("identity")
            if report.get("one_identity"):
                self.centroid = dataset_centroid(self.dataset, self.model)
            else:
                logging.info("[ZenTrainer] likeness off: the dataset's faces aren't one person (median %s)",
                             report.get("identity"))
        except Exception:
            logging.warning("[ZenTrainer] likeness: couldn't read the dataset's faces", exc_info=True)
        if self.centroid is None:
            logging.info("[ZenTrainer] likeness off: no faces in the dataset")
        while True:
            row = self.q.get()
            if row is None:
                return
            if self.centroid is None:
                continue
            try:
                path = paths.resolve_view(row.get("type", "output"), row["subfolder"], row["filename"])
                if path is None:
                    raise FileNotFoundError(row["filename"])
                score = likeness(path, self.centroid, face_model=self.model)
                out = {"run_id": self.run_id, "step": row["step"], "index": row["index"], "score": score,
                       "model": self.model, "ceiling": self.ceiling}
                if row.get("manual"):
                    out.update(manual=True, base=bool(row.get("base")))
                self.store.append(self.run_id, "likeness", out)
                from .. import events

                events.emit(events.LIKENESS, out)
            except Exception:
                logging.warning("[ZenTrainer] likeness failed for %s", row.get("filename"), exc_info=True)
