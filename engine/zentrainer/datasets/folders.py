"""Dataset folders (in ComfyUI's input dir, or the ZenTrainer folder's datasets/): list, inspect, edit captions, upload, delete.

Every path from a request is resolved and checked to stay inside its dataset folder.
"""

from __future__ import annotations


import os
import re
from pathlib import Path

from .. import safeio
from ..harness.data import video
from ..harness.data import (CONTROL_EXTS, CONTROLS_DIR, IMAGE_EXTS, MASK_SUFFIXES, MEDIA_EXTS, SET_NAME, caption_path,
                            caption_sets, find_control_sets, find_mask, glob_escape, scan)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,99}$")


class DatasetError(ValueError):
    pass


def root() -> Path:
    from .. import paths

    return paths.datasets_root().resolve()


def folder(name: str, must_exist: bool = True) -> Path:
    if not _NAME.match(name or "") or name.startswith((".", "_")):
        raise DatasetError(f"bad dataset name {name!r}")
    d = (root() / name).resolve()
    if d.parent != root():
        raise DatasetError(f"bad dataset name {name!r}")
    if must_exist and not d.is_dir():
        raise FileNotFoundError(name)
    return d


def _inside(d: Path, rel: str) -> Path:
    p = (d / rel).resolve()
    if d not in p.parents:
        raise DatasetError(f"path {rel!r} is outside the dataset")
    return p


# each dataset's summary, kept while none of its folders changed (a file added, removed or
# rewritten: ZenTrainer's writes all swap a file in, which touches its folder), so the datasets
# page doesn't scan thousands of pictures on every visit
_summaries: dict[str, tuple[tuple, dict | None]] = {}


def _stamp(d: Path) -> tuple:
    out = []
    for dirpath, _dirs, _files in os.walk(d):
        try:
            out.append((dirpath, os.stat(dirpath).st_mtime_ns))
        except OSError:
            pass
    return tuple(out)


def summary() -> list[dict]:
    r = root()
    out = []
    for d in sorted(p for p in r.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        stamp = _stamp(d)
        kept = _summaries.get(str(d))
        if kept is None or kept[0] != stamp:
            kept = (stamp, _summary(d, r))
            _summaries[str(d)] = kept
        if kept[1] is not None:
            # stats snapshots and settings change on their own
            out.append({**kept[1], "health": _health(d.name), "faces": bool(get_settings(d.name).get("faces"))})
    return out


def _summary(d: Path, r: Path) -> dict | None:
    items = scan(d, warn=False)
    if not items:
        return None
    stats = [i.path.stat() for i in items if i.path.exists()]
    return {
        "folder": d.name,
        "images": len(items),
        "masked": sum(1 for i in items if getattr(i, "mask", None)),
        "pairs": sum(1 for i in items if i.control_sets),
        "control_sets": control_counts(items),
        "health": _health(d.name),
        "bytes": sum(st.st_size for st in stats),
        "modified": max((st.st_mtime for st in stats), default=0),
        "videos": sum(1 for i in items if video.is_video(i.path)),
        "steps_per_epoch": sum(i.repeats for i in items),
        "captioned": sum(1 for i in items if i.caption),
        "sets": sorted({s for i in items for s in i.captions} | {"default"}, key=lambda s: (s != "default", s)),
        "thumbs": [{"subfolder": str(i.path.parent.relative_to(r)), "filename": i.path.name,
                    "path": str(i.path.relative_to(d))} for i in items[:8]],
    }


def _set(name: str) -> str:
    if name != "default" and not SET_NAME.match(name or ""):
        raise DatasetError(f"bad caption set name {name!r}: lowercase letters, digits, - and _")
    return name


_mask_stats: dict[tuple, tuple] = {}


def mask_stats(mask: Path) -> tuple[float, int, int] | None:
    """(coverage, width, height) of a mask: coverage is the mean brightness, so the share of
    the image trained at full weight. Remembered per file version."""
    from PIL import Image, ImageStat

    try:
        st = mask.stat()
    except OSError:
        return None
    key = (str(mask), st.st_mtime_ns, st.st_size)
    if key not in _mask_stats:
        try:
            with Image.open(mask) as im:
                w, h = im.size
                g = im.convert("L")
                g.thumbnail((256, 256))
                _mask_stats[key] = (ImageStat.Stat(g).mean[0] / 255, w, h)
        except OSError:
            return None
    return _mask_stats[key]


def _health(name: str) -> dict | None:
    """The last stats snapshot's 0-1 scores (dataset_stats), or None before a first look."""
    try:
        from .stats import latest_scores

        return latest_scores(name)
    except Exception:
        return None


_sizes: dict[tuple, tuple] = {}


def _size(path: Path) -> tuple[int, int] | tuple[None, None]:
    """A picture's or clip's width and height (the header only), or Nones if it can't be read.
    Remembered per file version: a big dataset reopens without reading every header again."""
    from PIL import Image

    try:
        st = path.stat()
    except OSError:
        return None, None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _sizes:
        clip = video.probe(path) if video.is_video(path) else None
        if clip is not None:
            _sizes[key] = (clip.width, clip.height)
        else:
            try:
                with Image.open(path) as im:
                    _sizes[key] = im.size
            except OSError:
                return None, None
    return _sizes[key]


def items(name: str, caption_ext: str = ".txt") -> dict:

    d, r = folder(name), root()
    from ..harness.data import muted as _muted

    mutes = _muted(d)
    rows = []
    for it in scan(d, caption_ext, warn=False):
        clip = video.probe(it.path) if video.is_video(it.path) else None
        w, h = (clip.width, clip.height) if clip is not None else _size(it.path)
        rows.append({
            "path": str(it.path.relative_to(d)),
            "subfolder": str(it.path.parent.relative_to(r)),  # for /files/view?type=input
            "filename": it.path.name,
            "caption": it.caption,
            "repeats": it.repeats,
            "width": w,
            "height": h,
            "bytes": it.path.stat().st_size,
            "mask": str(it.mask.relative_to(d)) if it.mask else None,
            "captions": it.captions,
            # edit training's inputs, per control set
            "control_sets": {k: [str(c.relative_to(d)) for c in v] for k, v in it.control_sets.items()},
            # each input's width and height, to compare with the picture's
            "control_sizes": {str(c.relative_to(d)): _size(c) for v in it.control_sets.values() for c in v},
            "muted": sorted(s for s, paths in mutes.items() if str(it.path.relative_to(d)) in paths),
        })
        if video.is_video(it.path):
            rows[-1].update(kind="video", fps=round(clip.fps, 3) if clip else None,
                            frames=clip.frames if clip else None, duration=round(clip.duration, 2) if clip else None)
        if it.mask and (ms := mask_stats(it.mask)):
            cover, mw, mh = ms
            rows[-1]["mask_coverage"] = round(cover, 4)
            # a mask is stretched to its image: a different shape means it won't line up
            rows[-1]["mask_mismatch"] = bool(w and h and abs(mw / mh - w / h) > 0.02)
    sets = sorted({s for r in rows for s in r["captions"]} | {"default"}, key=lambda s: (s != "default", s))
    controls = sorted({c for r in rows for c in r["control_sets"]} | set(_empty_control_sets(d)))
    return {"folder": name, "items": rows, "sets": sets, "control_sets": controls, "settings": get_settings(name)}


def set_muted(name: str, set_name: str, paths: list[str], on: bool) -> dict[str, list[str]]:
    """Mute (or unmute) pictures in a caption set: kept, captions and all, but not trained on in it.
    Stored in the dataset folder (.zentrainer.json) so the curation travels with the dataset."""
    from ..harness.data import MUTES_FILE, muted as _muted

    d = folder(name)
    set_name = _set(set_name)
    for p in paths:  # only pictures inside this dataset
        if not (d / p).resolve().is_relative_to(d.resolve()):
            raise DatasetError(f"{p} isn't in this dataset")
    mutes = _muted(d)
    cur = mutes.setdefault(set_name, set())
    cur.update(paths) if on else cur.difference_update(paths)
    f = d / MUTES_FILE
    data = safeio.read_json(f, {})
    data = data if isinstance(data, dict) else {}
    data["muted"] = {s: sorted(v) for s, v in mutes.items() if v}
    safeio.write_json(f, data)
    return data["muted"]


# ---- per-dataset settings (kept beside ZenTrainer's other state, not in the dataset folder) ------

# faces: face report on open, faces in the stats, sample likeness; caption_subject: what auto-captions
# call the main subject (a trigger word, a name): it belongs to the dataset, not the browser
DEFAULT_SETTINGS = {"faces": False, "caption_subject": ""}


def _settings_file() -> Path:
    import folder_paths

    p = Path(folder_paths.get_user_directory()) / "zentrainer" / "dataset_settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_settings(name: str) -> dict:
    try:
        saved = safeio.read_json(_settings_file(), {}).get(name, {})
    except (OSError, ImportError, AttributeError):  # ImportError: outside ComfyUI (tests, tools)
        saved = {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def put_settings(name: str, changes: dict) -> dict:
    folder(name)  # must exist
    f = _settings_file()
    allx = safeio.read_json(f, {})
    allx = allx if isinstance(allx, dict) else {}
    cur = {**get_settings(name), **{k: type(v)(changes[k]) for k, v in DEFAULT_SETTINGS.items() if k in changes}}
    cur["caption_subject"] = cur["caption_subject"].strip()
    allx[name] = cur
    safeio.write_json(f, allx)
    return cur


def set_caption(name: str, rel: str, caption: str, caption_ext: str = ".txt", set_name: str = "default") -> None:
    img = _inside(folder(name), rel)
    if img.suffix.lower() not in MEDIA_EXTS or not img.is_file():
        raise DatasetError(f"{rel!r} is not an image or video in this dataset")
    safeio.write_text(caption_path(img, _set(set_name), caption_ext), caption.strip() + "\n")


def bulk(name: str, op: str, text: str = "", find: str = "", caption_ext: str = ".txt", set_name: str = "default") -> int:
    """prefix / suffix / replace across every caption of one set. Returns how many changed."""
    changed = 0
    for row in items(name, caption_ext)["items"]:
        old = row["captions"].get(_set(set_name), "")
        if op == "prefix":
            new = old if old.startswith(text) else (f"{text}, {old}" if old else text)
        elif op == "suffix":
            new = old if old.endswith(text) else (f"{old}, {text}" if old else text)
        elif op == "replace":
            if not find:
                raise DatasetError("replace needs 'find'")
            new = old.replace(find, text)
        else:
            raise DatasetError(f"unknown op {op!r}; use prefix, suffix or replace")
        if new != old:
            set_caption(name, row["path"], new, caption_ext, set_name)
            changed += 1
    return changed


def delete(name: str, rel: str, caption_ext: str = ".txt") -> None:
    img = _inside(folder(name), rel)
    if img.suffix.lower() not in MEDIA_EXTS or not img.is_file():
        raise DatasetError(f"{rel!r} is not an image or video in this dataset")
    mask = find_mask(img)
    sets = caption_sets(img, caption_ext)
    # files another picture also owns stay: `foo.jpg` + `foo.png` share foo.txt and the mask,
    # and `frame.001.png`'s own caption, frame.001.txt, looks like a caption set of frame.png
    others = [p for p in img.parent.iterdir() if p != img and p.suffix.lower() in MEDIA_EXTS and p.is_file()]
    shared = any(o.stem == img.stem for o in others)
    theirs = {caption_path(o, "default", caption_ext) for o in others}
    img.unlink()
    if shared:
        return
    for set_name in sets:
        cap = caption_path(img, set_name, caption_ext)
        if cap not in theirs:
            cap.unlink(missing_ok=True)
    if mask:
        mask.unlink(missing_ok=True)
    for inputs in find_control_sets(img).values():  # its input pictures go with it
        for c in inputs:
            c.unlink(missing_ok=True)


# ---- control sets: named layers of input pictures for edit training (harness/data) ----------------

def control_counts(items) -> dict[str, int]:
    """How many pictures have an input in each control set."""
    out: dict[str, int] = {}
    for it in items:
        for k in it.control_sets:
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _empty_control_sets(d: Path) -> list[str]:
    """Sets made but not filled yet (a folder under _controls), so they show up to upload into."""
    base = d / CONTROLS_DIR
    return [p.name for p in base.iterdir() if p.is_dir() and SET_NAME.match(p.name)] if base.is_dir() else []


def _control_set(name: str) -> str:
    if not SET_NAME.match(name or ""):
        raise DatasetError(f"bad control set name {name!r}: lowercase letters, digits, - and _")
    return name


def _input_name(stem: str, n: int) -> str:
    return stem if n == 1 else f"{stem}.{n}"


def create_control_set(name: str, set_name: str) -> None:
    (folder(name) / CONTROLS_DIR / _control_set(set_name)).mkdir(parents=True, exist_ok=True)


def save_control(name: str, set_name: str, filename: str, data: bytes) -> str:
    """An input picture for a control set, matched to its picture by name: `<stem>.<ext>` is the
    input of the picture `<stem>`, `<stem>.2.<ext>` its second. Stored beside that picture as
    `_controls/<set>/<stem>.<ext>`, replacing the input it had there."""
    _control_set(set_name)
    p = Path(Path(filename).name)
    if p.suffix.lower() not in CONTROL_EXTS or p.name.startswith("."):
        raise DatasetError(f"{p.name!r}: inputs are images, videos or audio")
    d = folder(name)
    stem, n = p.stem, 1
    if (m := re.fullmatch(r"(.+)\.([2-9])", stem)) and not any(i.path.stem == stem for i in scan(d, warn=False)):
        stem, n = m.group(1), int(m.group(2))
    owners = [i.path for i in scan(d, warn=False) if i.path.stem == stem]
    if not owners:
        raise DatasetError(f"{p.name!r}: no picture called {stem!r} in this dataset to be its input")
    if len({o.parent for o in owners}) > 1:
        raise DatasetError(f"{p.name!r}: pictures called {stem!r} sit in several folders; rename one first")
    dest = owners[0].parent / CONTROLS_DIR / set_name
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob(f"{glob_escape(_input_name(stem, n))}.*"):
        if old.stem == _input_name(stem, n):
            old.unlink()
    target = dest / f"{_input_name(stem, n)}{p.suffix.lower()}"
    safeio.write_bytes(target, data)
    return str(target.relative_to(d))


def delete_control(name: str, set_name: str, rel: str) -> None:
    """One input picture out of a set (the picture itself stays)."""
    d = folder(name)
    p = _inside(d, rel)
    owned = any(p in v for it in scan(d, warn=False) for k, v in it.control_sets.items() if k == set_name)
    if not owned or not p.is_file():
        raise DatasetError(f"{rel!r} isn't an input in control set {set_name!r}")
    p.unlink()


def _set_inputs(d: Path, set_name: str) -> list[tuple[Path, int, Path]]:
    """(picture, input number, input file) for every input in a set."""
    return [(it.path, k + 1, c) for it in scan(d, warn=False) for k, c in enumerate(it.control_sets.get(set_name, []))]


def delete_control_set(name: str, set_name: str) -> int:
    """A whole control set, every input in it (the old <stem>_control files too, for "control")."""
    d = folder(name)
    inputs = _set_inputs(d, _control_set(set_name))
    for _, _, c in inputs:
        c.unlink(missing_ok=True)
    for base in d.rglob(CONTROLS_DIR):
        sub = base / set_name
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    return len(inputs)


def rename_control_set(name: str, old: str, new: str) -> int:
    """Rename a set: its inputs move to _controls/<new>/ (the old <stem>_control files of
    "control" move too, into the folder layout)."""
    d = folder(name)
    _control_set(old), _control_set(new)
    if new in {k for it in scan(d, warn=False) for k in it.control_sets} or new in _empty_control_sets(d):
        raise DatasetError(f"there's already a control set called {new!r}")
    inputs = _set_inputs(d, old)
    for pic, n, c in inputs:
        dest = pic.parent / CONTROLS_DIR / new
        dest.mkdir(parents=True, exist_ok=True)
        c.rename(dest / f"{_input_name(pic.stem, n)}{c.suffix.lower()}")
    for base in d.rglob(CONTROLS_DIR):
        sub = base / old
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    if (d / CONTROLS_DIR / old).is_dir():  # an empty set: just the folder
        (d / CONTROLS_DIR / old).rename(d / CONTROLS_DIR / new)
    return len(inputs)


def create(name: str) -> None:
    d = folder(name, must_exist=False)
    if d.exists():
        raise DatasetError(f"{name!r} already exists")
    d.mkdir()


def save_upload(name: str, filename: str, data: bytes) -> str:
    """An image, a caption file, or a mask (`<stem>-masklabel.png` / `<stem>.mask.png`,
    stored as `_masks/<stem>.png`), written into the dataset folder. Never over an existing
    file: the same bytes again are skipped, different ones get `name (2).ext`."""
    base = Path(filename).name
    p = Path(base)
    if p.suffix.lower() not in MEDIA_EXTS | {".txt"} or base.startswith("."):
        raise DatasetError(f"{base!r}: only images, videos, masks and .txt captions")
    d = folder(name)
    if p.suffix.lower() in IMAGE_EXTS and p.stem.endswith(MASK_SUFFIXES):
        stem = next(p.stem[: -len(x)] for x in MASK_SUFFIXES if p.stem.endswith(x))
        (d / "_masks").mkdir(exist_ok=True)
        target = _inside(d, f"_masks/{stem}{p.suffix.lower()}")
    else:
        target = _inside(d, base)
    if target.exists():
        if target.is_file() and target.stat().st_size == len(data) and target.read_bytes() == data:
            return str(target.relative_to(d))  # already there
        target = safeio.free_name(target)
    safeio.write_bytes(target, data)
    return str(target.relative_to(d))


def save_mask(name: str, rel: str, data_url: str) -> dict:
    """Save (or with no data, remove) an image's mask: <stem>-masklabel.png beside it, replacing
    whatever mask file it had. The PNG is greyscale: white trains at full weight."""
    import base64
    import io

    from PIL import Image

    d = folder(name)
    img = (d / rel).resolve()
    if d.resolve() not in img.parents or not img.is_file():
        raise DatasetError("no such image")
    old = find_mask(img)
    if not data_url:
        if old:
            old.unlink()
        return {"mask": None}
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    with Image.open(io.BytesIO(raw)) as m:
        with Image.open(img) as src:
            size = src.size
        mask = m.convert("L").resize(size, Image.Resampling.BILINEAR) if m.size != size else m.convert("L")
    target = img.with_name(img.stem + "-masklabel.png")
    safeio.write_bytes(target, _png(mask))
    if old and old != target:
        old.unlink()  # after the new one is in place
    return {"mask": str(target.relative_to(d))}


def _png(im) -> bytes:
    import io

    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


MASK_UNDO_KEEP = 20  # applied batches that can be undone, per dataset
_BATCH = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def _mask_undo(name: str) -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "undo" / "masks" / name


def apply_mask(name: str, rels: list[str], data_url: str, mode: str = "cutout") -> dict:
    """One mask onto many images of its size (the mask editor's "apply to selected"): `cutout`
    keeps each image's own mask and takes the new one's black out of it (a watermark box comes
    out of a subject mask), `replace` writes it as is. Images of another size are skipped. Every
    mask it changes is copied aside first, so the batch can be undone (undo_masks)."""
    import base64
    import io
    import secrets
    import shutil
    import time

    from PIL import Image, ImageChops

    if mode not in ("cutout", "replace"):
        raise DatasetError("mode is cutout or replace")
    d = folder(name)
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[-1]))) as m:
        applied = m.convert("L")
    batch = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    bdir = _mask_undo(name) / batch
    manifest, results = [], []
    for rel in rels:
        img = (d / rel).resolve()
        if d.resolve() not in img.parents or not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            results.append({"path": rel, "skipped": "not an image in this dataset"})
            continue
        with Image.open(img) as src:
            if src.size != applied.size:
                results.append({"path": rel, "skipped": f"{src.size[0]}×{src.size[1]}, not {applied.size[0]}×{applied.size[1]}"})
                continue
        old = find_mask(img)
        entry = {"path": rel, "had": None, "backup": None}
        new = applied
        if old:
            bdir.mkdir(parents=True, exist_ok=True)
            entry["had"] = str(old.relative_to(d))
            entry["backup"] = f"{len(manifest)}{old.suffix.lower()}"
            shutil.copy2(old, bdir / entry["backup"])
            if mode == "cutout":
                with Image.open(old) as om:
                    own = om.convert("L")
                    own = own.resize(applied.size, Image.Resampling.BILINEAR) if own.size != applied.size else own
                new = ImageChops.darker(own, applied)
        target = img.with_name(img.stem + "-masklabel.png")
        safeio.write_bytes(target, _png(new))
        if old and old != target:
            old.unlink()  # backed up above
        entry["wrote"] = str(target.relative_to(d))
        manifest.append(entry)
        results.append({"path": rel, "mask": entry["wrote"]})
    if manifest:
        bdir.mkdir(parents=True, exist_ok=True)
        safeio.write_json(bdir / "manifest.json", {"mode": mode, "entries": manifest})
        done = sorted(p for p in bdir.parent.iterdir() if p.is_dir() and _BATCH.match(p.name))
        for gone in done[:-MASK_UNDO_KEEP]:
            shutil.rmtree(gone, ignore_errors=True)
    return {"batch": batch if manifest else None, "results": results}


def undo_masks(name: str, batch: str) -> dict:
    """Put back the masks an apply_mask batch changed (and remove the ones it added)."""
    import shutil

    if not _BATCH.match(batch):
        raise DatasetError("no such batch")
    d = folder(name)
    bdir = _mask_undo(name) / batch
    data = safeio.read_json(bdir / "manifest.json", None)
    if not data:
        raise DatasetError("that edit can't be undone any more")
    restored = 0
    for e in reversed(data["entries"]):
        wrote = _inside(d, e["wrote"])
        if e["had"]:
            safeio.write_bytes(_inside(d, e["had"]), (bdir / e["backup"]).read_bytes())
            if wrote != _inside(d, e["had"]) and wrote.is_file():
                wrote.unlink()
        elif wrote.is_file():
            wrote.unlink()
        restored += 1
    shutil.rmtree(bdir, ignore_errors=True)
    return {"restored": restored}


def split_videos(name: str, seconds: float = 3.0, count: int = 20, longer_than: float = 20.0) -> dict:
    """Cut every video in the dataset longer than `longer_than` s into short clips (see
    harness/video.split) beside it; the originals move to _long/, which training ignores."""
    d = folder(name)
    clips, moved = 0, []
    for it in scan(d, warn=False):
        if not video.is_video(it.path):
            continue
        p = video.probe(it.path)
        if p is None or p.duration <= longer_than:
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "_", it.path.stem).strip("_")[:48] or "clip"
        clips += len(video.split(it.path, it.path.parent, seconds, count, stem=stem))
        keep = d / "_long" / it.path.relative_to(d)
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep = safeio.free_name(keep)  # an earlier original of the same name stays
        it.path.rename(keep)
        moved.append(str(keep.relative_to(d)))
    return {"clips": clips, "moved": moved}


def thumbnail(name: str, rel: str, size: int, cache: Path) -> Path:
    """A cached JPEG thumbnail (longest side `size`) of an image, a video (a frame a little way
    in, past fades from black) or a mask in the dataset."""
    import hashlib

    from PIL import Image, ImageOps

    src = _inside(folder(name), rel)
    if src.suffix.lower() not in MEDIA_EXTS or not src.is_file():
        raise DatasetError(f"{rel!r} is not an image or video in this dataset")
    size = max(32, min(int(size), 1024))
    st = src.stat()
    key = hashlib.sha1(f"{src}|{st.st_mtime_ns}|{st.st_size}|{size}".encode()).hexdigest()
    out = cache / key[:2] / f"{key}.jpg"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        if video.is_video(src):
            im = video.poster(src, 0.1)
            if im is None:
                raise DatasetError(f"{rel!r}: can't decode this video")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            im.convert("RGB").save(out, "JPEG", quality=85)
            return out
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            im.convert("RGB").save(out, "JPEG", quality=85)
    return out
