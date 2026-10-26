"""A dataset's health, measured, and kept over time.

Each look computes the metrics below; when the dataset has changed since the last snapshot (files,
sizes, captions), the new one is appended to its history, so the panel can show what an edit did
(recaptioned, pruned, cropped) and the card can draw the latest five scores as a pentagon.

    captions      share captioned (the default set), and per caption set
    consistency   how even the captions are: length spread, identical captions, very short ones
    quality       share of pictures with no image issue: training-ready size, sharp (against the set's
                  own median), sensibly exposed, no letterbox bars, no transparency, not heavily
                  JPEG-compressed (per-picture flags for the view)
    uniqueness    share of pictures without a near-duplicate of themselves (pairs())
    faces/balance with faces on: how much the faces agree (each against the others)
                  times the share of pictures without a face issue;
                  without one, how evenly the pictures spread over aspect ratios

Scores are 0-1 (1 = good) for the pentagon; the panel shows the real numbers behind them.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

from .. import safeio

TARGET_SIDE = 768  # a picture's short side at or above this is training-ready at the usual sizes
# Two pictures are near-duplicates when all three agree: their 256-bit difference hashes (unrelated
# pictures sit near 128 bits apart), the correlation of their 16x16 greyscale shapes, and their average
# colour (0-255). One alone pairs any two foggy pictures, or a street with a forest; together they
# catch re-saved, resized, lightly edited and mirrored copies and nothing else. "The same shot taken
# again" needs more than this (landscapes share a composition): that's for image embeddings.
NEAR_DUP = (24, 0.95, 15)
TRANSPARENT = 0.01  # share of a picture's pixels see-through for it to count as transparent
JPEG_LOW = 60  # estimated JPEG quality below this: compression artifacts a LoRA can learn
LOOK_VERSION = 5  # 2: transparency and JPEG quality; 3: the pair signature (_signature); 4: bars need an edge;
# 5: a transparent picture measured on what shows


def _store_dir() -> Path:
    import folder_paths

    d = Path(folder_paths.get_user_directory()) / "zentrainer" / "dataset_stats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:16]


def size(path: Path) -> tuple[int, int] | None:
    """A picture's (or clip's) width and height as training sees them: EXIF rotation applied."""
    from ..harness.data import upright_size, video

    try:
        if video.is_video(path):
            p = video.probe(path)
            return (p.width, p.height) if p else None
        return upright_size(path)
    except Exception:
        return None


def jpeg_quality(im) -> int | None:
    """A JPEG's quality (1-100) estimated from its luminance quantization table against the
    standard one (IJG's scaling), or None for anything else."""
    tables = getattr(im, "quantization", None)
    if not tables or 0 not in tables:
        return None
    std = (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
           14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
           49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99)
    scale = 100 * sum(tables[0]) / sum(std)
    q = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return max(1, min(100, round(q)))


def _look(path: Path) -> dict | None:
    """One pass over a picture (a clip's middle frame), at 256 px: its 64-bit difference hash
    (near-identical pictures land a few bits apart), its image statistics, how much of it is
    see-through (most models train on what's under it; a four-channel VAE, Qwen Image 2.1's, keeps
    it) and its JPEG quality."""
    import numpy as np
    from PIL import Image

    from ..harness.data import video

    try:
        if video.is_video(path):
            p = video.probe(path)
            if p is None or not p.frames:
                return None
            scale = 256 / max(p.width, p.height)
            px = video.read(path, [p.frames // 2], (max(16, round(p.width * scale)), max(16, round(p.height * scale))))[0]
            im = Image.fromarray((px.numpy() * 255).astype("uint8"))
            alpha, jpeg_q = 0.0, None
        else:
            im = Image.open(path)
            jpeg_q = jpeg_quality(im)
            im.draft("RGB", (512, 512))  # JPEGs decode small
            seen = None
            if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                im.thumbnail((256, 256))
                seen = np.asarray(im)[..., 3] / 255.0  # how much each pixel shows
                alpha = float((seen < 250 / 255).mean())
            else:
                im = im.convert("RGB")
                im.thumbnail((256, 256))
                alpha = 0.0
            im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.float32) / 255.0
        luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        lap = -4 * luma[1:-1, 1:-1] + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
        mx, mn = rgb.max(-1), rgb.min(-1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        border = _border(luma)
        bright, contrast, sharp, sat = float(luma.mean()), float(luma.std()), float(lap.var()), float(sat.mean())
        if alpha > TRANSPARENT and seen.sum() > 0.01 * seen.size:
            # the picture as it looks: what's under its see-through parts (black, mostly) isn't dark,
            # flat or a bar ("transparent" says it's there), so each pixel counts as much as it shows,
            # and sharpness is where a pixel and its neighbours all show (the cut-out's edge isn't)
            def mean(x, w):
                return float((x * w).sum() / w.sum())

            bright = mean(luma, seen)
            contrast, sat, border = mean((luma - bright) ** 2, seen) ** 0.5, mean(sat, seen), 0.0
            inner = np.minimum.reduce([seen[1:-1, 1:-1], seen[:-2, 1:-1], seen[2:, 1:-1], seen[1:-1, :-2], seen[1:-1, 2:]])
            if inner.sum() > 0:
                sharp = mean((lap - mean(lap, inner)) ** 2, inner)
        return {**_signature(im), "bright": round(bright, 4), "contrast": round(contrast, 4),
                "sharp": round(sharp * 1000, 4), "sat": round(sat, 4),
                "border": round(border, 3), "alpha": round(alpha, 4), "jpeg_q": jpeg_q}
    except Exception:
        return None


def _dhash(im) -> str:
    """A 256-bit difference hash (16 rows of 16 left-right brightness steps), as hex."""
    import numpy as np
    from PIL import Image

    g = np.asarray(im.convert("L").resize((17, 16), Image.Resampling.BILINEAR), np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1]).tobytes().hex()


def _signature(im) -> dict:
    """What pairs() compares: the picture's difference hash and its mirror's (a flipped copy is
    still a copy), its 16x16 greyscale shape (zero mean, unit length, stored as int8) and its
    average colour."""
    import base64

    import numpy as np
    from PIL import Image, ImageOps

    g = np.asarray(im.convert("L").resize((16, 16), Image.Resampling.BILINEAR), np.float32).flatten()
    g -= g.mean()
    g /= np.linalg.norm(g) + 1e-6
    colour = np.asarray(im.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR), np.float32).mean((0, 1))
    return {"dhash": _dhash(im), "dhash_m": _dhash(ImageOps.mirror(im)),
            "shape": base64.b64encode(np.round(g * 127).astype(np.int8).tobytes()).decode(),
            "colour": [round(float(c), 1) for c in colour]}


def pairs(looks: dict[str, dict | None]) -> list[dict]:
    """Every pair of pictures that are near-duplicates of each other (mirrored or not), closest
    first: {"a", "b", "kind", "bits", "corr", "colour", "mirrored"}. The shapes' correlation, one
    matrix product per block, shortlists; the hash and colour decide."""
    import base64

    import numpy as np

    keys = [p for p, lk in looks.items() if lk and "shape" in lk]
    if len(keys) < 2:
        return []
    shape = np.stack([np.frombuffer(base64.b64decode(looks[p]["shape"]), np.int8) for p in keys]).astype(np.float32)
    shape /= np.linalg.norm(shape, axis=1, keepdims=True) + 1e-6
    mirror = shape.reshape(-1, 16, 16)[:, :, ::-1].reshape(len(keys), -1)
    hashes = np.stack([np.frombuffer(bytes.fromhex(looks[p]["dhash"]), np.uint8) for p in keys])
    hashes_m = np.stack([np.frombuffer(bytes.fromhex(looks[p]["dhash_m"]), np.uint8) for p in keys])
    colour = np.array([looks[p]["colour"] for p in keys], np.float32)
    found: dict[tuple[int, int], dict] = {}
    for start in range(0, len(keys), 512):
        block = shape[start:start + 512]
        for mirrored, against, other in ((False, shape, hashes), (True, mirror, hashes_m)):
            for r, j in np.argwhere(block @ against.T >= NEAR_DUP[1]):
                i = start + int(r)
                j = int(j)
                if j <= i:
                    continue
                bits = int(np.unpackbits(hashes[i] ^ other[j]).sum())
                diff = float(np.abs(colour[i] - colour[j]).mean())
                corr = float(block[r] @ against[j])
                near = bits <= NEAR_DUP[0] and corr >= NEAR_DUP[1] and diff <= NEAR_DUP[2]
                if near and ((i, j) not in found or bits < found[(i, j)]["bits"]):
                    found[(i, j)] = {"a": keys[i], "b": keys[j], "kind": "near-duplicate", "bits": bits,
                                     "corr": round(corr, 3), "colour": round(diff, 1), "mirrored": mirrored}
    return sorted(found.values(), key=lambda q: q["bits"])


def pair_kinds(found: list[dict]) -> dict[str, str]:
    """Each paired picture's kind of pair ("near-duplicate")."""
    return {p: q["kind"] for q in found for p in (q["a"], q["b"])}


def _border(luma) -> float:
    """The share of the picture taken by bars along its edges (letterboxing, frames). A bar is an
    even band of one tone that ends in a hard edge where the picture starts: a plain sky or wall
    fades into the scene and isn't one."""
    h, w = luma.shape

    def bar(lines) -> int:
        """How many of these lines, from the edge in, are a bar: flat lines of the edge's own tone
        (a gradient drifts away from it), up to where the picture starts."""
        tone = float(lines[0].mean())
        n = 0
        while n < len(lines) // 3 and float(lines[n].std()) < 0.02 and abs(float(lines[n].mean()) - tone) < 0.03:
            n += 1
        if n == 0 or n >= len(lines) // 3:
            return 0
        # where the picture starts, its pixels differ from the bar's tone (each, not on average: a
        # dark scene against a black bar still has texture); haze out of a sky doesn't
        return n if float(abs(lines[n] - tone).mean()) > 0.08 else 0

    top, bottom = bar(luma), bar(luma[::-1])
    left, right = bar(luma.T), bar(luma.T[::-1])
    kept = max(0, h - top - bottom) * max(0, w - left - right)
    return 1 - kept / max(1, h * w)


def _looks(name: str, items: list) -> dict[str, dict | None]:
    """Each picture's look (hash + statistics), cached by mtime and size."""
    cache_file = _store_dir() / f"{_key(name)}.looks.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    out, keep = {}, {}
    for it in items:
        st_ = it.path.stat()
        stamp = f"{LOOK_VERSION}|{st_.st_mtime_ns}|{st_.st_size}"
        hit = cache.get(str(it.path))
        if not hit or hit[0] != stamp:
            hit = [stamp, _look(it.path)]
        keep[str(it.path)] = hit
        out[str(it.path)] = hit[1]
    safeio.write_json(cache_file, keep, indent=None)
    return out


def image_flags(looks: dict[str, dict | None], sides: dict[str, int], found: list[dict] | None = None) -> dict[str, list[str]]:
    """Why each picture might need a look: dark, blown out, flat, blurry (against the set's own
    median sharpness: what's sharp depends on the content), bars, greyscale, transparent, compressed,
    small, near-duplicate (`found`: pairs(looks), when it's already worked out)."""
    vals = [lk for lk in looks.values() if lk]
    sharp_med = st.median(lk["sharp"] for lk in vals) if vals else 0
    sat_med = st.median(lk["sat"] for lk in vals) if vals else 0
    kinds = pair_kinds(pairs(looks) if found is None else found)
    out = {}
    for p, lk in looks.items():
        f = []
        if lk:
            if lk["bright"] < 0.15:
                f.append("dark")
            if lk["bright"] > 0.85:
                f.append("blown out")
            if lk["contrast"] < 0.06:
                f.append("flat")
            if sharp_med and lk["sharp"] < 0.25 * sharp_med:
                f.append("blurry")
            if lk["border"] > 0.08:
                f.append("bars")
            if lk["sat"] < 0.03 and sat_med > 0.08:
                f.append("greyscale")
            if lk.get("alpha", 0) > TRANSPARENT:
                f.append("transparent")
            if lk.get("jpeg_q") is not None and lk["jpeg_q"] < JPEG_LOW:
                f.append("compressed")
        if sides.get(p, TARGET_SIDE) < TARGET_SIDE:
            f.append("small")
        if p in kinds:
            f.append(kinds[p])
        if f:
            out[p] = f
    return out


def compute(name: str, set_name: str = "default") -> dict:
    """The dataset's metrics and 0-1 scores, now, for one caption set: its captions, and without
    the pictures muted in it."""
    from . import folders as datasets
    from ..harness.data import video
    from ..harness.data import muted, scan

    root = datasets.folder(name)
    off = muted(root).get(set_name, set())
    items = [it for it in scan(root, warn=False) if str(it.path.relative_to(root)) not in off]
    n = len(items)
    caps = [(it.captions.get(set_name) or "").strip() for it in items]
    captioned = [c for c in caps if c]
    words = [len(c.split()) for c in captioned]
    sets = Counter(s for it in items for s, t in it.captions.items() if t.strip())

    sides, aspects, side_of = [], [], {}
    for it in items:
        w, h = size(it.path) or (0, 0)
        if w and h:
            sides.append(min(w, h))
            aspects.append(round(w / h, 1))
            side_of[str(it.path)] = min(w, h)
    dup_caps = sum(c - 1 for c in Counter(captioned).values() if c > 1)
    looks = _looks(name, items)
    found = pairs(looks)
    kinds = pair_kinds(found)
    near = len(kinds)
    closeness = [1.0] * near  # each picture with a near-duplicate counts against uniqueness
    flags = image_flags(looks, side_of, found)
    lv = [lk for lk in looks.values() if lk]
    bright = sorted(lk["bright"] for lk in lv)
    count = lambda tag: sum(tag in f for f in flags.values())  # noqa: E731

    m = {
        "pictures": n,
        "videos": sum(video.is_video(it.path) for it in items),
        "captioned": len(captioned),
        "caption_sets": dict(sets),
        "caption_words_median": st.median(words) if words else 0,
        "caption_words_spread": round(st.pstdev(words) / st.mean(words), 3) if len(words) > 1 and st.mean(words) else 0.0,
        "caption_short": sum(w < 5 for w in words),
        "caption_identical": dup_caps,
        "short_side_median": int(st.median(sides)) if sides else 0,
        "below_target": sum(s < TARGET_SIDE for s in sides),
        "aspect_top_share": round(max(Counter(aspects).values()) / len(aspects), 3) if aspects else 0.0,
        "near_duplicates": near,
        "brightness_median": round(st.median(bright), 3) if bright else 0,
        "brightness_spread": round(bright[int(len(bright) * 0.9)] - bright[int(len(bright) * 0.1)], 3) if len(bright) > 2 else 0,
        "too_dark": count("dark"),
        "blown_out": count("blown out"),
        "flat": count("flat"),
        "blurry": count("blurry"),
        "bars": count("bars"),
        "greyscale": count("greyscale"),
        "transparent": count("transparent"),
        "compressed": count("compressed"),
        "saturation_median": round(st.median(lk["sat"] for lk in lv), 3) if lv else 0,
        "image_issues": sum(1 for f in flags.values() if set(f) - {"near-duplicate"}),
        "muted": len(off),
        "masked": sum(1 for it in items if it.mask),
        "pairs": sum(1 for it in items if it.control_sets),
    }
    face_scores = None
    report = _cached_faces(name)
    if report is not None:
        rows = report["items"].values()
        issues = sum(1 for r in rows if r["faces"] != 1 or r["outlier"])
        likes = [r["likeness"] for r in rows if r["likeness"] is not None]
        identity = report.get("identity")
        m.update(face_coverage=report["with_face"], face_issues=issues,
                 face_likeness_median=round(st.median(likes), 3) if likes else None,
                 face_likeness_min=round(min(likes), 3) if likes else None,
                 face_one_identity=bool(report.get("one_identity")))
        # how much the faces agree (each against the others: 0.2 = strangers, 0.7 = clearly one
        # person), times the share of pictures without a face issue
        agree = 0.0 if identity is None else max(0.0, min(1.0, (identity - 0.2) / 0.5))
        face_scores = agree * (1 - issues / max(1, len(report["items"])))

    def clamp(x):
        return max(0.0, min(1.0, x))

    scores = {
        "captions": clamp(len(captioned) / n) if n else 0.0,
        "consistency": clamp((1 - min(1.0, m["caption_words_spread"]) * 0.5)
                             * (1 - (dup_caps + m["caption_short"]) / max(1, len(captioned)))) if captioned else 0.0,
        # quality: training-ready size, sharp, sensibly exposed, no bars
        "quality": clamp(1 - m["image_issues"] / n) if n else 0.0,
        "uniqueness": clamp(1 - sum(closeness) / n) if n else 0.0,
    }
    if face_scores is not None:
        scores["faces"] = clamp(face_scores)
    else:
        # one aspect ratio holding every picture is fine for a fixed-size set; the score only
        # dips when one ratio dominates a set that's otherwise mixed
        scores["balance"] = clamp(1.0 if len(set(aspects)) <= 1 else 1.2 - m["aspect_top_share"])
    root_s = str(root) + "/"
    return {"metrics": m, "scores": {k: round(v, 3) for k, v in scores.items()},
            "flags": {p.replace(root_s, "", 1): f for p, f in flags.items()}}


def _cached_faces(name: str) -> dict | None:
    """The faces report when the dataset has faces on (its setting; cached per picture)."""
    from . import folders as datasets, faces

    return faces.dataset_faces(name) if datasets.get_settings(name).get("faces") else None


def _fingerprint(name: str, set_name: str) -> str:
    """What the dataset is, for this set: its files, the set's captions, masks, pairs, mutes."""
    from . import folders as datasets
    from ..harness.data import muted, scan

    root = datasets.folder(name)
    h = hashlib.sha256(json.dumps(sorted(muted(root).get(set_name, ()))).encode())
    for it in scan(root, warn=False):
        s = it.path.stat()
        h.update(f"{it.path.name}|{s.st_mtime_ns}|{s.st_size}|{it.captions.get(set_name, '')}|{bool(it.mask)}|{sorted((k, len(v)) for k, v in it.control_sets.items())}".encode())
    return h.hexdigest()[:20]


def _history_file(name: str, set_name: str) -> Path:
    return _store_dir() / (f"{_key(name)}.jsonl" if set_name == "default" else f"{_key(name)}.{_key(set_name)}.jsonl")


def history(name: str, set_name: str = "default") -> list[dict]:
    try:
        lines = _history_file(name, set_name).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:  # cut short: skip it, keep the rest
            continue
    return out


def snapshot(name: str, set_name: str = "default") -> dict:
    """Compute now. A new snapshot only when the dataset changed since the last one; otherwise the
    latest is refreshed in place (faces turned on, measured again), so the history is a record of
    edits. Returns {"current", "history"}."""
    now = compute(name, set_name)
    fp = _fingerprint(name, set_name)
    past = history(name, set_name)
    row = {"at": time.time(), "set": set_name, "fingerprint": fp, "metrics": now["metrics"], "scores": now["scores"]}
    if past and past[-1].get("fingerprint", "").split("|")[0] == fp:
        row["at"] = past[-1]["at"]
        past[-1] = row
    else:
        past.append(row)
    safeio.write_text(_history_file(name, set_name), "".join(json.dumps(r) + "\n" for r in past))
    return {"current": now, "history": past[-60:]}


def latest_scores(name: str) -> dict | None:
    """The scores of the caption set looked at most recently (its history was written last), cheap,
    for the datasets list: {"scores", "set"}; None before a first look."""
    files = [f for f in _store_dir().glob(f"{_key(name)}*.jsonl")]
    if not files:
        return None
    newest = max(files, key=lambda f: f.stat().st_mtime)
    try:
        last = json.loads(newest.read_text().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
    return {"scores": last["scores"], "set": last.get("set", "default")}
