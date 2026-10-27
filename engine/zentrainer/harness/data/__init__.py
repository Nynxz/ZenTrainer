"""Dataset: scan → bucket → encode once → cache → batches.

Latents (VAE) and conditioning (text encoder) are computed once and cached on disk,
keyed by the files, their mtimes and every setting that changes the encoding, so a
re-run on the same data skips straight to training.

Masks (optional, white = learn, black = ignore) are found per image as
`_masks/<stem>.<ext>`, `<stem>-masklabel.<ext>` (OneTrainer) or `<stem>.mask.<ext>`, get
the image's exact resize/crop, and are stored at latent resolution for the loss.

Control sets (edit training's input pictures) are named layers beside the pictures, so a dataset
stays a dataset: `_controls/<set>/<stem>.<ext>` is a picture's input in that set, `<stem>.2.<ext>`
its second. Older `<stem>_control.<ext>` files beside the picture are the set "control". A run
picks the sets it trains from (DatasetSpec.control_sets).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ...spec import DatasetSpec
from . import video
from .video import VIDEO_EXTS, is_video

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


MASK_SUFFIXES = ("-masklabel", ".mask")
# edit pairs: <stem>_control.<ext> is the input picture for <stem> (the target), the caption is
# the instruction; _control2, _control3... add more references
CONTROL = re.compile(r"_control\d*$")
SET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")


@dataclass
class Item:
    path: Path
    caption: str  # the "default" set (<stem>.txt)
    repeats: int = 1
    mask: Path | None = None
    captions: dict[str, str] = field(default_factory=dict)  # set name -> text, "default" included
    controls: list[Path] = field(default_factory=list)  # the input picture(s) this run uses (pick_controls)
    control_sets: dict[str, list[Path]] = field(default_factory=dict)  # every set's input(s), by set name


@dataclass
class Sample:
    latent: torch.Tensor  # (C, [T,] h, w) raw VAE latent (before process_latent_in); T for video VAEs
    caption: int  # index into Dataset.conditionings
    mask: torch.Tensor | None = None  # (h, w) in [0, 1] at latent resolution
    variants: list[int] = field(default_factory=list)  # caption choices when mixing sets
    audio: torch.Tensor | None = None  # (C, 2, Ta): a clip's soundtrack latent (joint audio-video models)
    # edit pairs: the control pictures' raw latents, as reference_latents; for channel inputs
    # ("concat"), one per control set in order, None where the picture has no input in that set
    refs: list[torch.Tensor | None] | None = None
    # grounded edits: this image's own conditioning (the encoder saw its inputs), and the empty
    # caption with them for caption dropout
    cond: list | None = None
    cond_empty: list | None = None


@dataclass
class Dataset:
    samples: list[Sample]
    conditionings: list  # one CLIP conditioning per unique caption
    captions: list[str]
    empty_caption: int  # index of "" (for caption dropout)
    buckets: dict[tuple, list[int]] = field(default_factory=dict)  # latent shape -> sample idx
    owners: list[int] = field(default_factory=list)  # sample -> the scanned item it came from (a video gives several)


DISK_FIELDS = ("latent", "mask", "refs", "cond", "cond_empty")


@functools.lru_cache(maxsize=8)  # a step reads a sample's file several times; a few stay
def _disk_load(file: str) -> dict:
    return torch.load(file, weights_only=True)


class DiskSample:
    """A Sample whose tensors stay in a file of its own until a step reads them (dataset.disk_cache):
    the same fields, read through a small LRU, so a dataset needn't fit in RAM."""

    def __init__(self, file: Path, shape, caption: int, variants: list[int], has_refs: bool, has_cond: bool,
                 has_mask: bool = False):
        self.file, self.shape, self.caption, self.variants = str(file), tuple(shape), caption, variants
        self.has_refs, self.has_cond, self.has_mask = has_refs, has_cond, has_mask
        self.audio = None

    def __getattr__(self, name):
        if name in DISK_FIELDS:
            return _disk_load(self.file).get(name)
        raise AttributeError(name)


def _bf16(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.bfloat16) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: _bf16(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_bf16(v) for v in x)
    return x


def _write_sample(file: Path, fields: dict) -> None:
    tmp = file.with_suffix(".tmp")
    torch.save(_bf16(fields), tmp)
    os.replace(tmp, file)


DISK_BATCH = 32  # pictures of one size per VAE call: ComfyUI stages the VAE for every call (and splits a batch by free memory)


def _inputs_of(spec, it) -> list:
    """The pictures a pair trains from: its inputs, or for channel inputs each set's first (None where
    a set has none)."""
    if not (spec.edit and it.controls):
        return []
    if spec.edit_method == "concat":
        return [next(iter(it.control_sets.get(c, [])), None) for c in spec.control_sets]
    return it.controls


def _disk_samples(spec, items, variants, cap_index, sizes, vae, folder: Path, progress, total: int):
    """disk_cache's pictures, each to folder/NNNNNN.pt. A file already there (a build cut short:
    they're written whole) is kept; the rest are encoded a batch of one size at a time, the next
    batch decoding while the VAE works. Returns (samples, owners, grounded still to encode)."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    grounding = spec.edit and spec.edit_grounded and spec.edit_method != "concat"
    samples: list = [None] * len(items)
    grounded = []
    todo: dict[tuple, list[int]] = {}
    for f in folder.glob("*.tmp"):
        f.unlink(missing_ok=True)
    for n, it in enumerate(items):
        idx = [cap_index[t] for t in variants[n]]
        file = folder / f"{n:06d}.pt"
        if file.is_file():
            try:
                kept = torch.load(file, weights_only=True, mmap=True)
            except Exception:  # noqa: BLE001 - unreadable: encode it again
                kept = None
            if kept is not None:
                has = kept.get("cond") is not None
                samples[n] = DiskSample(file, kept["latent"].shape, idx[0], idx, kept.get("refs") is not None,
                                        has or bool(grounding and it.controls), kept.get("mask") is not None)
                if grounding and not has and it.controls:
                    grounded.append((n, it.controls, variants[n][0]))
                continue
        size = pick_bucket(*upright_size(it.path), sizes)
        todo.setdefault(size, []).append(n)
    if any(s is not None for s in samples):
        logging.info("[ZenTrainer] disk cache: %d of %d pictures already encoded, carrying on",
                     sum(s is not None for s in samples), len(items))

    mode = pixel_mode(vae)

    def load(n: int, size):
        it = items[n]
        inputs = _inputs_of(spec, it)
        if any(c is not None for c in inputs):
            target = upright_size(it.path)
            for c in inputs:
                if c is not None:
                    _warn_if_shaped_apart(c, it.path, target, spec.edit_method == "concat")
        mask = load_image(it.mask, size, mode="L")[None, None] if it.mask else None
        return (load_image(it.path, size, mode), [load_image(c, size, mode) if c is not None else None for c in inputs],
                mask)

    batches = [(size, ns[i:i + DISK_BATCH]) for size, ns in todo.items() for i in range(0, len(ns), DISK_BATCH)]
    done = len(items) - sum(len(ns) for ns in todo.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        ahead = [pool.submit(load, n, batches[0][0]) for n in batches[0][1]] if batches else []
        for b, (size, ns) in enumerate(batches):
            loaded = [f.result() for f in ahead]
            if b + 1 < len(batches):  # the next batch decodes while this one encodes
                ahead = [pool.submit(load, n, batches[b + 1][0]) for n in batches[b + 1][1]]
            targets = _encode_batch(vae, torch.stack([t for t, _, _ in loaded]), [items[n].path for n in ns])
            pics = [(j, k, px) for j, (_, ins, _) in enumerate(loaded) for k, px in enumerate(ins) if px is not None]
            refs_out = _encode_batch(vae, torch.stack([px for _, _, px in pics]), [items[ns[j]].path for j, _, _ in pics]) if pics else []
            for j, n in enumerate(ns):
                it, (_, ins, m) = items[n], loaded[j]
                refs = None
                if ins:
                    refs = [None] * len(ins)
                    for (jj, k, _), lat in zip(pics, refs_out):
                        if jj == j:
                            refs[k] = lat
                latent = targets[j]
                mask = torch.nn.functional.interpolate(m, size=tuple(latent.shape[-2:]), mode="area")[0, 0] if m is not None else None
                file = folder / f"{n:06d}.pt"
                _write_sample(file, {"latent": latent, "mask": mask, "refs": refs})
                idx = [cap_index[t] for t in variants[n]]
                samples[n] = DiskSample(file, latent.shape, idx[0], idx, refs is not None,
                                        bool(grounding and it.controls), mask is not None)
                if grounding and it.controls:
                    grounded.append((n, it.controls, variants[n][0]))
            done += len(ns)
            if progress:
                progress(done, total, f"encoding {items[ns[-1]].path.name}")
    grounded.sort()
    return samples, list(range(len(items))), grounded


def _encode_batch(vae, pixels: torch.Tensor, paths: list) -> list:
    """(B, H, W, 3) pictures of one size -> their latents, one VAE call; as _encode, never a NaN cached."""
    latents = vae.encode(pixels).detach().cpu()
    for lat, p in zip(latents, paths):
        if not torch.isfinite(lat).all():
            raise ValueError(f"the VAE returned NaN/inf for {Path(p).name}. The VAE is likely broken or missing "
                             "weights (check ComfyUI's log for 'Missing VAE keys'); try a separate VAE file")
    return list(latents)


def has_inputs(s) -> bool:
    """An edit pair: it has reference latents or its own (grounded) conditioning, without reading
    a disk sample's file."""
    if isinstance(s, DiskSample):
        return s.has_refs or s.has_cond
    return s.refs is not None or s.cond is not None


def has_mask(s) -> bool:
    return s.has_mask if isinstance(s, DiskSample) else s.mask is not None


def sample_shape(s) -> tuple:
    return s.shape if isinstance(s, DiskSample) else tuple(s.latent.shape)


def caption_index(files, caption_ext: str = ".txt") -> dict[str, dict[str, str]]:
    """A folder's caption sets from its file names, read once: stem -> {set: file name}. (A set
    name has no dot, so <stem>.<set>.txt splits at its last one.)"""
    out: dict[str, dict[str, str]] = {}
    for f in files:
        if not f.endswith(caption_ext):
            continue
        stem, _, name = f[: -len(caption_ext)].rpartition(".")
        if stem and SET_NAME.match(name):
            out.setdefault(stem, {})[name] = f
    return out


def caption_sets(img: Path, caption_ext: str = ".txt", index: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """All caption sets for an image: <stem>.txt is "default", <stem>.<set>.txt the rest. `index`:
    its folder's caption_index (scan passes it: globbing the folder per picture is quadratic)."""
    out = {}
    default = img.with_suffix(caption_ext)
    if default.is_file():
        out["default"] = default.read_text(encoding="utf-8").strip()
    if index is None:
        index = caption_index((p.name for p in img.parent.glob(f"{glob_escape(img.stem)}.*{caption_ext}")), caption_ext)
    for name, f in index.get(img.stem, {}).items():
        out[name] = (img.parent / f).read_text(encoding="utf-8").strip()
    return out


def caption_path(img: Path, set_name: str, caption_ext: str = ".txt") -> Path:
    return img.with_suffix(caption_ext) if set_name == "default" else img.with_name(f"{img.stem}.{set_name}{caption_ext}")


def glob_escape(s: str) -> str:
    return re.sub(r"([*?\[\]])", r"[\1]", s)


def is_mask_file(p: Path) -> bool:
    return p.stem.endswith(MASK_SUFFIXES)


def is_control_file(p: Path) -> bool:
    return bool(CONTROL.search(p.stem))


CONTROL_EXTS = MEDIA_EXTS | {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}


CONTROLS_DIR = "_controls"
LEGACY_SET = "control"


def control_index(folder: Path) -> dict[str, dict[str, Path]]:
    """A folder's control sets: set name -> {file stem ("a", "a.2"): path}, read once per folder."""
    out: dict[str, dict[str, Path]] = {}
    base = folder / CONTROLS_DIR
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if d.is_dir() and SET_NAME.match(d.name):
            out[d.name] = {f.stem: f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in CONTROL_EXTS}
    return out


def find_control_sets(img: Path, index: dict[str, dict[str, Path]] | None = None) -> dict[str, list[Path]]:
    """A picture's inputs in every control set: <stem>, then <stem>.2, <stem>.3... in the set."""
    index = control_index(img.parent) if index is None else index
    out = {}
    legacy = find_controls(img)
    if legacy:
        out[LEGACY_SET] = legacy
    for name, files in index.items():
        hits = []
        for n in ["", *(f".{k}" for k in range(2, 10))]:
            if (hit := files.get(img.stem + n)) is None:
                break
            hits.append(hit)
        if hits:
            out.setdefault(name, hits)  # _control files beside the picture win over a _controls/control folder
    return out


def pick_controls(item: Item, sets: list[str]) -> list[Path]:
    """The inputs a run trains from: each chosen set's, in the sets' order."""
    return [p for s in sets for p in item.control_sets.get(s, [])]


def find_controls(img: Path) -> list[Path]:
    """An edit pair's input(s): <stem>_control, then _control2, _control3... Pictures for every
    model; videos and audio too for MiniMax H3's references."""
    out = []
    for n in ["", *map(str, range(2, 10))]:
        hit = next((c for e in sorted(CONTROL_EXTS) if (c := img.with_name(f"{img.stem}_control{n}{e}")).is_file()), None)
        if hit is None:
            break
        out.append(hit)
    return out


def find_mask(img: Path) -> Path | None:
    if is_video(img):
        return None  # masks are per image
    for cand in [img.parent / "_masks" / (img.stem + e) for e in sorted(IMAGE_EXTS)] + [
        img.with_name(img.stem + suffix + e) for suffix in MASK_SUFFIXES for e in sorted(IMAGE_EXTS)
    ]:
        if cand.is_file():
            return cand
    return None


def scan(root: Path, caption_ext: str = ".txt", warn: bool = True) -> list[Item]:
    """Images and videos under root with same-stem captions (and masks, see module doc). "N_name"
    subfolders repeat N times; folders starting with "." or "_" are ignored."""
    items = []
    for dirpath, dirs, files in os.walk(root):
        # skip hidden and tool folders: other trainers leave .thumbs/, _latent_cache/, ...
        dirs[:] = sorted(x for x in dirs if not x.startswith((".", "_")))
        d = Path(dirpath)
        repeats = 1
        head = d.name.split("_", 1)[0] if d != root else ""
        if head.isdigit() and "_" in d.name:
            repeats = max(1, int(head))
        index = control_index(d)
        captions = caption_index(files, caption_ext)
        for f in sorted(files):
            p = d / f
            if p.suffix.lower() not in MEDIA_EXTS or is_mask_file(p) or is_control_file(p):
                continue
            caps = caption_sets(p, caption_ext, captions)
            if "default" not in caps and warn:
                logging.warning("[ZenTrainer] no caption for %s; training it with an empty caption", p.name)
            sets = find_control_sets(p, index)
            items.append(Item(p, caps.get("default", ""), repeats, find_mask(p), caps, sets.get(LEGACY_SET, []), sets))
    return items


def bucket_sizes(resolution: int, step: int, max_ratio: float = 4.0) -> list[tuple[int, int]]:
    """(w, h) pairs, multiples of `step`, with w*h <= resolution², aspect within max_ratio."""
    area = resolution * resolution
    out = set()
    w = step
    while w <= resolution * max_ratio:
        h = (area // w) // step * step
        if h >= step and max(w / h, h / w) <= max_ratio:
            out.add((w, h))
        w += step
    return sorted(out)


def _quarter_turned(path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.getexif().get(0x0112) in (5, 6, 7, 8)
    except Exception:  # noqa: BLE001 - unreadable: the build says so
        return False


def upright_size(path) -> tuple[int, int]:
    """A picture's width and height as it's shown and trained: its EXIF orientation applied (phone
    photos are often stored turned a quarter, with a tag to turn them back). load_image turns it,
    so the bucket and shape checks have to go by this, not the stored size, or a portrait photo
    trains cropped to a landscape bucket."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return (h, w) if _quarter_turned(path) else (w, h)


def _warn_if_shaped_apart(c, target, target_wh: tuple[int, int], concat: bool) -> None:
    """An input a different shape from its target: both are cropped from the centre to the target's
    bucket, so a different shape loses a different part. Against the target itself, not the bucket:
    buckets come in 64 px steps, so a matched 2:3 pair sits a few percent off its 832x1216."""
    cw, ch = upright_size(c)
    tw, th = target_wh
    if abs((cw / ch) / (tw / th) - 1) > (0.02 if concat else 0.08):
        logging.warning("[ZenTrainer] %s is a different shape from %s: it's cropped to match, which cuts part of "
                        "it off%s", f"{c.parent.name}/{c.name}", target.name,
                        " (channel inputs line up pixel for pixel: make it the same shape)" if concat else "")


def pick_bucket(w: int, h: int, sizes: list[tuple[int, int]]) -> tuple[int, int]:
    ar = math.log(w / h)
    return min(sizes, key=lambda s: (abs(math.log(s[0] / s[1]) - ar), -s[0] * s[1]))


def pixel_mode(vae) -> str:
    """How pictures reach this VAE: "RGBA" for a four-channel one (Qwen Image 2.1's), so a
    transparent PNG trains as transparent; "RGB" for the rest, which never see alpha. An RGB file
    loaded as RGBA gets alpha 1, as ComfyUI pads it."""
    return "RGBA" if getattr(vae, "output_channels", 3) == 4 else "RGB"


def load_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> torch.Tensor:
    """Resize to cover `size`, center-crop. Returns (H, W, 3) float in [0, 1], (H, W, 4) for
    mode "RGBA", or (H, W) for mode "L" (masks get exactly the same geometry as their image)."""
    from PIL import Image, ImageOps

    import numpy as np

    img = ImageOps.exif_transpose(Image.open(path)).convert(mode)
    tw, th = size
    scale = max(tw / img.width, th / img.height)
    rw, rh = max(tw, round(img.width * scale)), max(th, round(img.height * scale))
    img = img.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - tw) // 2, (rh - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)


def apply_trigger(text: str, trigger: str, prepend: bool) -> str:
    """ai-toolkit semantics: "[trigger]" -> the word; if the caption still doesn't
    contain it, prepend "trigger, ". A structured (JSON) caption gets it at the start of its
    short_description instead, so it stays JSON."""
    if not trigger:
        return text.replace("[trigger]", "").strip(" ,")
    text = text.replace("[trigger]", trigger)
    if prepend and trigger.lower() not in text.lower():
        from ...datasets import structured

        if structured.is_json_caption(text):
            return structured.with_trigger(text, trigger)
        text = f"{trigger}, {text}" if text else trigger
    return text


def cache_key(items: list[Item], spec: DatasetSpec, encoder_id: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([spec.resolution, spec.buckets, spec.bucket_step, encoder_id]).encode())
    if spec.disk_cache:
        h.update(b"disk")
    if any(is_video(it.path) for it in items):  # only then: image datasets keep their caches
        h.update(json.dumps([spec.video_frames, spec.video_fps, spec.video_mode, spec.video_clips,
                             spec.video_resolution]).encode())
        if spec.video_audio:
            h.update(b"audio")
        if spec.video_avoid_cuts:
            h.update(b"avoid_cuts")
        if spec.video_stills:
            h.update(b"clip_stills")  # v1's name: the caches made before keep their key
    # a picture stored turned a quarter was bucketed by its stored size before (cropped to the wrong
    # shape): a dataset with one encodes again, once; the rest keep their caches
    if any(not is_video(it.path) and _quarter_turned(it.path) for it in items):
        h.update(b"upright")
    for it in items:
        st = it.path.stat()
        h.update(f"{it.path}|{st.st_mtime_ns}|{st.st_size}|{it.caption}".encode())
        if it.mask:
            ms = it.mask.stat()
            h.update(f"|mask:{it.mask}|{ms.st_mtime_ns}|{ms.st_size}".encode())
        if spec.video_keyframes:
            h.update(f"|keyframes:{spec.video_keyframes}".encode())  # v1's name, as above
        if spec.edit:
            if spec.edit_grounded and spec.edit_method != "concat":
                h.update(b"grounded")
            if spec.edit_method == "concat":  # one slot per set, in order
                h.update(f"|concat:{','.join(spec.control_sets)}".encode())
            for c in it.controls:
                cs = c.stat()
                h.update(f"|control:{c}|{cs.st_mtime_ns}|{cs.st_size}".encode())
    return h.hexdigest()[:20]


def build(spec: DatasetSpec, clip, vae, root: Path, cache_dir: Path, encoder_id: str, progress=None,
          audio_vae=None, h3: bool = False) -> Dataset:
    # The harness runs outside ComfyUI's inference_mode, where the frozen VAE/text
    # encoder parameters still require grad: encode under no_grad, or every cached
    # latent drags an autograd graph (and a VAE backward) into training.
    with torch.no_grad():
        return _build(spec, clip, vae, root, cache_dir, encoder_id, progress, audio_vae, h3)


MUTES_FILE = ".zentrainer.json"  # in the dataset folder: {"muted": {set: [relative paths]}}; travels with it


def muted(root: Path) -> dict[str, set[str]]:
    """The pictures muted in each caption set (kept, captions and all, but not trained on in that set)."""
    try:
        data = json.loads((root / MUTES_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return {s: set(v) for s, v in (data.get("muted") or {}).items()}


def _build(spec, clip, vae, root, cache_dir, encoder_id, progress, audio_vae=None, h3: bool = False) -> Dataset:
    items = scan(root, spec.caption_ext)
    if not items:
        raise FileNotFoundError(f"no images found in {root}")
    mutes = muted(root)
    if mutes:  # a picture trains from the sets it isn't muted in; muted in all of them, it sits out
        used = spec.caption_mix or [spec.caption_set]
        before = len(items)
        items = [it for it in items if any(str(it.path.relative_to(root)) not in mutes.get(s, ()) for s in used)]
        if len(items) < before:
            logging.info("[ZenTrainer] %d muted picture(s) sit out (caption set %s)", before - len(items), ", ".join(used))
        if not items:
            raise FileNotFoundError(f"every picture in {root} is muted in {', '.join(used)}")
    for it in items:  # the input pictures this run trains from, in its control sets' order
        it.controls = pick_controls(it, spec.control_sets) if spec.edit else []
    if spec.edit and len(spec.control_sets) > 1:
        partial = sum(1 for it in items if it.controls and not all(it.control_sets.get(c) for c in spec.control_sets))
        if partial:
            logging.warning("[ZenTrainer] %d picture(s) have inputs in only some of the control sets %s: "
                            "they train with the inputs they have", partial, ", ".join(spec.control_sets))
    variants: list[list[str]] = []
    for it in items:  # pick each image's caption(s), trigger applied, before the cache key
        rel = str(it.path.relative_to(root))
        if spec.caption_mix:
            texts = [it.captions[s] for s in spec.caption_mix if it.captions.get(s) and rel not in mutes.get(s, ())]
        else:
            texts = [it.captions.get(spec.caption_set, "")] if it.captions.get(spec.caption_set) else []
        texts = texts or [it.captions.get("default", "")]
        texts = [apply_trigger(t, spec.trigger, spec.trigger_prepend) for t in texts]
        if spec.edit and spec.edit_template:
            from .edit import instruction

            texts = [instruction(t, spec.edit_template) for t in texts]
        variants.append(texts)
        it.caption = " ||| ".join(texts)  # only feeds the cache key

    # H3 builds references/keyframes its own way, a four-channel VAE encodes alpha (caches from
    # before it was kept were opaque); other runs keep their caches
    key = cache_key(items, spec, encoder_id + ("|h3" if h3 and (spec.edit or spec.video_keyframes) else "")
                    + ("|rgba" if pixel_mode(vae) == "RGBA" else ""))
    path = cache_dir / key
    texts = TextCache(cache_dir, encoder_id)
    if (path / "dataset.pt").is_file():
        try:
            ds = _from_plain(torch.load(path / "dataset.pt", weights_only=True), texts, path)
        except Exception as e:  # noqa: BLE001 - cut short or from an older build: rebuild it
            logging.warning("[ZenTrainer] dataset cache %s couldn't be read (%s); rebuilding it", key, e)
            shutil.rmtree(path, ignore_errors=True)
        else:
            logging.info("[ZenTrainer] dataset cache hit %s (%d images)", key, len(items))
            os.utime(path / "dataset.pt")  # recently used: kept by prune_caches
            return _expand_repeats(ds, items)

    sizes = bucket_sizes(spec.resolution, spec.bucket_step) if spec.buckets else [(spec.resolution, spec.resolution)]
    captions = sorted({t for v in variants for t in v} | {""})
    cap_index = {c: i for i, c in enumerate(captions)}

    from PIL import Image

    video_sizes = (bucket_sizes(spec.video_resolution, spec.bucket_step) if spec.buckets
                   else [(spec.video_resolution, spec.video_resolution)])
    lengths = None  # the clip lengths this VAE encodes exactly, worked out on the first video

    disk = spec.disk_cache
    if disk:
        if h3 or any(is_video(it.path) for it in items):
            raise ValueError("the disk cache is for picture datasets (not videos or MiniMax H3)")
        (path / "samples").mkdir(parents=True, exist_ok=True)
    samples: list = []
    owners: list[int] = []
    grounded: list[tuple[int, list[Path], str]] = []  # (sample, inputs, caption): encoded with the captions
    if disk:
        samples, owners, grounded = _disk_samples(spec, items, variants, cap_index, sizes, vae, path / "samples",
                                                  progress, len(items) + len(captions))
    # MiniMax H3: (sample, inputs, first frame, last frame, caption, (w, h), frames), each sample's
    # conditioning built by ComfyUI's own H3 nodes after the pictures
    h3_jobs: list[tuple] = []
    sounds: list[tuple[int, Path, float, float]] = []  # (sample, video, t0, t1): encoded after the pictures
    for n, it in enumerate([] if disk else items):
        idx = [cap_index[t] for t in variants[n]]
        if is_video(it.path):
            p = video.probe(it.path)
            if p is None or not p.frames:
                logging.warning("[ZenTrainer] can't read %s; skipping it", it.path.name)
                continue
            if lengths is None:
                lengths = video.valid_lengths(vae, max(spec.video_frames, 1))
            if spec.video_avoid_cuts:
                spans = video.shots(p, video.cuts(it.path))
                windows = video.plan_shots(p, spec.video_fps, spec.video_frames, spec.video_mode, spec.video_clips, lengths, spans)
            else:
                windows = video.plan(p, spec.video_fps, spec.video_frames, spec.video_mode, spec.video_clips, lengths)
            if not windows:
                logging.warning("[ZenTrainer] %s is too short for a clip; skipping it", it.path.name)
            size = pick_bucket(p.width, p.height, video_sizes)
            for k, frames in enumerate(windows):
                px = video.read(it.path, frames, size)
                latent = _encode(vae, px, it.path)
                if h3 and ((spec.edit and it.controls) or spec.video_keyframes):
                    first = px[:1].clone() if spec.video_keyframes else None
                    last = px[-1:].clone() if spec.video_keyframes == "first_last" else None
                    h3_jobs.append((len(samples), it.controls if spec.edit else [], first, last, variants[n][0],
                                    size, len(frames)))
                if spec.video_audio and audio_vae is not None:
                    sounds.append((len(samples), it.path, frames[0] / p.fps, (frames[-1] + 1) / p.fps))
                samples.append(Sample(latent, idx[0], None, idx))
                owners.append(n)
                if progress:
                    progress(n + 1, len(items) + len(captions), f"encoding {it.path.name} ({k + 1}/{len(windows)})")
            if spec.video_stills and windows:  # its sharpest frame, trained as a photo with the clip's caption
                f = video.sharpest_frame(it.path, p)
                still = video.read(it.path, [f], pick_bucket(p.width, p.height, sizes))
                samples.append(Sample(_encode(vae, still, it.path), idx[0], None, idx))
                owners.append(n)
            continue
        w, h = upright_size(it.path)
        size = pick_bucket(w, h, sizes)
        latent = _encode(vae, load_image(it.path, size, pixel_mode(vae)).unsqueeze(0), it.path)
        mask = None
        if it.mask:
            m = load_image(it.mask, size, mode="L")[None, None]  # (1, 1, H, W)
            mask = torch.nn.functional.interpolate(m, size=tuple(latent.shape[-2:]), mode="area")[0, 0]
        refs = None
        if h3 and spec.edit and it.controls:  # H3 takes them in its own conditioning, built below
            h3_jobs.append((len(samples), it.controls, None, None, variants[n][0], size, 1))
        elif spec.edit and it.controls:  # the input picture(s) at the target's size, as the model's references
            concat = spec.edit_method == "concat"
            # channel inputs: each set's (first) input in its slot, lined up with the target
            inputs = [next(iter(it.control_sets.get(c, [])), None) for c in spec.control_sets] if concat else it.controls
            for c in inputs:
                if c is not None:
                    _warn_if_shaped_apart(c, it.path, (w, h), concat)
            refs = [_encode(vae, load_image(c, size, pixel_mode(vae)).unsqueeze(0), c) if c is not None else None
                    for c in inputs]
            if spec.edit_grounded and not concat:  # channel inputs reach the model, not its encoder
                grounded.append((len(samples), it.controls, variants[n][0]))
        samples.append(Sample(latent, idx[0], mask, idx, refs=refs))
        owners.append(n)
        if progress:
            progress(n + 1, len(items) + len(captions), f"encoding {it.path.name}")
    if not samples:
        raise FileNotFoundError(f"nothing in {root} could be encoded")
    # soundtracks in one pass after every picture: the video and audio VAEs each load once,
    # instead of swapping in and out per clip
    for k, (i, clip, t0, t1) in enumerate(sounds):  # not `path`: that's the cache folder, written below
        samples[i].audio = _encode_audio(audio_vae, clip, t0, t1)
        if progress and k % 10 == 0:
            progress(k + 1, len(sounds), f"encoding soundtracks ({clip.name})")

    # disk-cached grounded pairs carry their own conditioning: their plain captions aren't needed
    # (thousands of them would be most of RAM)
    needed = None
    if disk:
        needed = {""} | {captions[v] for sm in samples if not sm.has_cond for v in sm.variants}
    conditionings = []
    for n, c in enumerate(captions):
        if needed is not None and c not in needed:
            conditionings.append(None)
            continue
        cond = texts.get(c)
        if cond is None:
            cond = _to_cpu(clip.encode_from_tokens_scheduled(clip.tokenize(c)))
            texts.put(c, cond)
        conditionings.append(cond)
        if progress:
            progress(len(items) + n + 1, len(items) + len(captions), "encoding captions")

    if grounded:  # each with the encoder looking at its own inputs; the empty one only if dropout uses it
        from .edit import encode_grounded, grounding_pixels

        for k, (i, inputs, text) in enumerate(grounded):
            pics = [grounding_pixels(c, clip) for c in inputs]
            cond = _to_cpu(encode_grounded(clip, text, pics))
            empty = _to_cpu(encode_grounded(clip, "", pics)) if spec.caption_dropout else None
            if disk:  # after every picture, as in RAM: the VAE and the text encoder each load once
                fields = dict(torch.load(samples[i].file, weights_only=True))
                _write_sample(Path(samples[i].file), {**fields, "cond": cond, "cond_empty": empty})
            else:
                samples[i].cond, samples[i].cond_empty = cond, empty
            if progress:
                progress(k + 1, len(grounded), "encoding instructions with their pictures")

    if h3_jobs:  # H3: references or keyframes, and the text encoder seeing them
        from .edit import h3_keyframe_cond, h3_reference_cond

        def h3_cond(text, inputs, first, last, size, nframes):
            if inputs:
                return h3_reference_cond(clip, vae, audio_vae, text, inputs, size[0], size[1], nframes)
            return h3_keyframe_cond(clip, vae, text, first, last, size[0], size[1], nframes)

        for k, (i, inputs, first, last, text, size, nframes) in enumerate(h3_jobs):
            samples[i].cond = _to_cpu(h3_cond(text, inputs, first, last, size, nframes))
            if spec.caption_dropout:
                samples[i].cond_empty = _to_cpu(h3_cond("", inputs, first, last, size, nframes))
            if progress:
                progress(k + 1, len(h3_jobs), "encoding captions with their references")

    ds = Dataset(samples, conditionings, captions, cap_index[""], owners=owners)
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / ".dataset.pt.tmp"  # whole or not at all: a cut-short cache would fail every later run
    torch.save(_to_plain(ds, texts), tmp)
    os.replace(tmp, path / "dataset.pt")
    (path / CACHE_SOURCE).write_text(json.dumps({"folder": spec.folder, "text": texts.dir.name,
                                                 "captions": [texts.key(c) for c in captions]}))
    prune_caches(cache_dir, spec.folder)
    return _expand_repeats(ds, items)


# Each caption or mask edit makes a new cache (the key covers them); the older ones are kept for a
# few edits back, then dropped. CACHE_SOURCE says which dataset a cache belongs to, and which
# captions it uses from the text cache.
CACHE_SOURCE = "source.json"
CACHES_PER_DATASET = 3
ORPHAN_DAYS = 7  # a cache from before they recorded their dataset: dropped once this old
TEXT_GRACE = 86400  # s an encoding no cache names is kept anyway (a build may be writing its cache)


class TextCache:
    """Captions' text encodings, one file each, shared by every dataset cache with the same text
    encoder: resolution stages, bucket changes and later runs on the same captions reuse them
    (a caption's encoding doesn't depend on the pictures). They're most of a cache's size: Fibo
    2.0's encoder hands the model 18 layers, ~100 MB for one structured caption."""

    def __init__(self, cache_dir: Path, encoder_id: str):
        self.dir = cache_dir / "text" / hashlib.sha256(encoder_id.encode()).hexdigest()[:16]

    @staticmethod
    def key(caption: str) -> str:
        return hashlib.sha256(caption.encode()).hexdigest()[:24]

    def get(self, caption: str):
        f = self.dir / f"{self.key(caption)}.pt"
        if not f.is_file():
            return None
        try:
            cond = torch.load(f, weights_only=True)
        except Exception as e:  # noqa: BLE001 - cut short: encode it again
            logging.warning("[ZenTrainer] text cache %s couldn't be read (%s); encoding it again", f.name, e)
            return None
        os.utime(f)
        return cond

    def put(self, caption: str, cond) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / f"{self.key(caption)}.pt"
        tmp = f.with_suffix(".tmp")
        torch.save(cond, tmp)
        os.replace(tmp, f)


def cache_folder(path: Path) -> str | None:
    try:
        return json.loads((path / CACHE_SOURCE).read_text()).get("folder")
    except (OSError, ValueError, AttributeError):
        return None  # from before caches recorded it


def prune_caches(cache_dir: Path, folder: str, keep: int = CACHES_PER_DATASET) -> None:
    import time

    caches = [p for p in cache_dir.iterdir() if p.is_dir() and (p / "dataset.pt").is_file()]
    mine = sorted((p for p in caches if cache_folder(p) == folder), key=lambda p: (p / "dataset.pt").stat().st_mtime, reverse=True)
    gone = set(mine[keep:])
    # caches from before they said whose they are: nothing would ever drop them
    gone |= {p for p in caches if cache_folder(p) is None and time.time() - (p / "dataset.pt").stat().st_mtime > ORPHAN_DAYS * 86400}
    for old in gone:
        shutil.rmtree(old, ignore_errors=True)
    prune_text(cache_dir)


def text_files(cache: Path) -> list[Path]:
    """The text-cache files one dataset cache uses."""
    try:
        src = json.loads((cache / CACHE_SOURCE).read_text())
    except (OSError, ValueError):
        return []
    if not src.get("text"):
        return []
    return [cache.parent / "text" / src["text"] / f"{k}.pt" for k in src.get("captions") or []]


def prune_text(cache_dir: Path, grace: float = TEXT_GRACE) -> int:
    """Drop the text encodings no dataset cache uses (past `grace` seconds: a build may be writing
    its cache); returns the bytes freed."""
    import time

    used = {f for p in cache_dir.iterdir() if p.is_dir() and (p / "dataset.pt").is_file() for f in text_files(p)}
    freed = 0
    text_root = cache_dir / "text"
    for f in text_root.glob("*/*.pt") if text_root.is_dir() else []:
        if f not in used and time.time() - f.stat().st_mtime > grace:
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
    return freed


def _encode(vae, pixels: torch.Tensor, path: Path) -> torch.Tensor:
    """pixels (T, H, W, 3) in [0, 1]: one image, or a clip's frames -> the latent, batch dim off."""
    latent = vae.encode(pixels)[0].detach().cpu()
    if not torch.isfinite(latent).all():
        # never cache this: every later run with the same key would train on it
        raise ValueError(f"the VAE returned NaN/inf for {path.name}. The VAE is likely broken or missing "
                         "weights (check ComfyUI's log for 'Missing VAE keys'); try a separate VAE file")
    return latent


# The cache holds only tensors, lists, dicts and strings. Pickling our own classes
# would tie the file to a module path, and ComfyUI imports custom nodes by path.
def _encode_audio(audio_vae, path: Path, t0: float, t1: float) -> torch.Tensor | None:
    """The soundtrack from t0 to t1 s as the audio VAE's latent (C, 2, Ta); None for a silent video."""
    import comfy.audio

    got = video.read_audio(path, t0, t1)
    if got is None:
        return None
    wave, sr = got  # (channels, samples)
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    wave = wave[None, :2]
    if sr != vae_sr:
        wave = comfy.audio.resample(wave, sr, vae_sr)
    return audio_vae.encode(wave.movedim(1, -1))[0].float().cpu()  # (C, 2, Ta)


def _to_plain(ds: Dataset, texts: "TextCache | None" = None) -> dict:
    if ds.samples and isinstance(ds.samples[0], DiskSample):
        return {
            "version": 5,  # disk samples: their tensors in samples/, captions in the text cache
            "text_keys": [texts.key(c) if cond is not None else None for c, cond in zip(ds.captions, ds.conditionings)],
            "disk": [[Path(s.file).name, list(s.shape), s.caption, s.variants, s.has_refs, s.has_cond, s.has_mask]
                     for s in ds.samples],
            "captions": ds.captions,
            "empty_caption": ds.empty_caption,
            "owners": ds.owners,
        }
    return {
        # 4: the captions' encodings are in the text cache (named by "text_keys"), not in here
        "version": 4 if texts else 3,
        "text_keys": [texts.key(c) for c in ds.captions] if texts else None,
        "latents": [s.latent for s in ds.samples],
        "variants": [s.variants for s in ds.samples],
        "masks": [s.mask for s in ds.samples],
        "caption_idx": [s.caption for s in ds.samples],
        "conditionings": None if texts else ds.conditionings,
        "captions": ds.captions,
        "empty_caption": ds.empty_caption,
        "owners": ds.owners,
        "audio": [s.audio for s in ds.samples],
        "refs": [s.refs for s in ds.samples],
        "conds": [s.cond for s in ds.samples],
        "conds_empty": [s.cond_empty for s in ds.samples],
    }


def _from_plain(d: dict, texts: "TextCache | None" = None, where: Path | None = None) -> Dataset:
    if d.get("disk") is not None:
        if texts is None or where is None:
            raise ValueError("a disk cache reads its samples from its folder and captions from the text cache")
        samples = [DiskSample(where / "samples" / name, *rest) for name, *rest in d["disk"]]
        if not all(Path(sm.file).is_file() for sm in samples):
            raise FileNotFoundError("a sample's file is gone from the disk cache")
        conditionings = [texts.get(c) if k is not None else None for c, k in zip(d["captions"], d["text_keys"])]
        if any(cond is None and k is not None for cond, k in zip(conditionings, d["text_keys"])):
            raise FileNotFoundError("a caption's encoding is gone from the text cache")
        return Dataset(samples, conditionings, d["captions"], d["empty_caption"], owners=d["owners"])
    masks = d.get("masks") or [None] * len(d["latents"])
    variants = d.get("variants") or [[c] for c in d["caption_idx"]]
    audio = d.get("audio") or [None] * len(d["latents"])
    n = len(d["latents"])
    refs, conds, empties = (d.get(k) or [None] * n for k in ("refs", "conds", "conds_empty"))
    samples = [Sample(lat, c, m, v, a, r, g, e)
               for lat, c, m, v, a, r, g, e in zip(d["latents"], d["caption_idx"], masks, variants, audio, refs, conds, empties)]
    owners = d.get("owners") or list(range(len(samples)))  # before videos: one sample per item
    conditionings = d.get("conditionings")
    if conditionings is None:  # version 4: each from the text cache (missing one: build it again)
        if texts is None:
            raise ValueError("this cache keeps its captions in the text cache")
        conditionings = [texts.get(c) for c in d["captions"]]
        if any(c is None for c in conditionings):
            raise FileNotFoundError("a caption's encoding is gone from the text cache")
    return Dataset(samples, conditionings, d["captions"], d["empty_caption"], owners=owners)


def _expand_repeats(ds: Dataset, items: list[Item]) -> Dataset:
    """Repeats are an index trick, not copies; buckets are recomputed per run."""
    order: list[int] = []
    owners = ds.owners or list(range(len(ds.samples)))
    for i, owner in enumerate(owners):
        order += [i] * items[owner].repeats
    buckets: dict[tuple, list[int]] = {}
    for i in order:
        buckets.setdefault(sample_shape(ds.samples[i]), []).append(i)
    ds.buckets = buckets
    return ds


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x


class Batches:
    """Endless iterator of same-shape index batches, reshuffled each epoch."""

    def __init__(self, ds: Dataset, batch_size: int, seed: int):
        self.ds = ds
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.epoch = 0
        self._queue: list[list[int]] = []

    def _refill(self) -> None:
        batches = []
        for idx in self.ds.buckets.values():
            idx = idx[:]
            self.rng.shuffle(idx)
            batches += [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        self.rng.shuffle(batches)
        self._queue = batches
        self.epoch += 1

    @property
    def per_epoch(self) -> int:
        """Batches in one pass through the dataset (repeats included): each bucket in whole batches."""
        return sum(-(-len(idx) // self.batch_size) for idx in self.ds.buckets.values())

    def reseed(self, seed: int) -> None:
        """A fresh shuffle order from `seed` (a resume: from its step on), starting a new pass."""
        self.rng.seed(seed)
        self._queue = []

    def __next__(self) -> list[int]:
        if not self._queue:
            self._refill()
        return self._queue.pop()

    def __iter__(self):
        return self
