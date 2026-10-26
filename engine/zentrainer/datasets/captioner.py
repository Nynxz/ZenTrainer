"""Auto-captioning with ComfyUI's native text generation (the same path as its
TextGenerate node): a vision-language text encoder (Qwen3-VL, Qwen2.5-VL, Gemma, ...)
sees each image and writes its caption into a caption set.

A job is queued as a one-node ComfyUI prompt (ZenCaptionSession) so it takes its turn
on the GPU like a training run. Jobs are kept on disk (user/zentrainer/caption_jobs.json):
what they wrote, and what any caption they replaced said before, so a job can be reviewed
and undone after a restart. A preview job writes nothing; it only returns its captions.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from typing import NamedTuple

from .. import safeio
from .. import events
from . import structured


class Style(NamedTuple):
    label: str
    prompt: str
    # "text": the answer as written; "fibo_json": a structured caption (structured.py), checked and
    # kept compact, or not written at all
    format: str = "text"
    # tokens the answer may take; 0 = no limit (NO_LIMIT): the model ends a caption itself, and a
    # limit only cuts one off (a looping model is what repetition_penalty is for)
    max_length: int = 0


STYLES = {
    "nl": Style("Natural language",
                "Describe this image for training an image model: the subject, their appearance, clothing and pose, "
                "the setting, lighting and photographic or artistic style. One detailed paragraph. "
                "Start directly with the description; no preamble."),
    "short": Style("Short caption", "Write a single-sentence caption for this image. Start directly; no preamble.",
                   max_length=300),  # one sentence: past this it's rambling
    "tags": Style("Tags",
                  "List comma-separated tags for this image: subject, features, clothing, pose, setting, lighting, "
                  "style, most important first. Output only the tags."),
    # Fibo 2.0's own caption format, with Bria's VGL prompts (structured.py)
    "fibo_json": Style("Fibo structured JSON", structured.caption_instruction(), "fibo_json"),
}
JSON_TRIES = 3  # answers a structured caption gets (greedy, then sampled) before the image is skipped
# max_length 0 = no limit: the answer runs until the model ends it. Still capped, so a model that
# never stops (repeating itself) can't run forever
NO_LIMIT = 4096

CAPTION = "zentrainer.caption"  # {job_id, dataset, set, status, done, total, file, text, message, ...}
MODES = ("missing", "all", "selected")  # which images: no caption in the set yet / every one / `paths`
ACTIVE = ("queued", "running")
PREVIEW_KEEP = 600  # s a finished try-on stays listed
KEEP = 60  # jobs remembered


@dataclass
class CaptionJob:
    job_id: str
    dataset: str
    set_name: str
    clip_name: str
    clip_type: str
    prompt: str
    mode: str = "missing"
    paths: list[str] = field(default_factory=list)  # dataset-relative; mode "selected"
    subject: str = ""  # what to call the main subject (a trigger word, a name)
    preview: int = 0  # > 0: caption this many images, write nothing
    max_length: int = 0  # tokens an answer may take; 0 = no limit (NO_LIMIT)
    # None: the model's single most likely answer (the same every time). A number: sampled with
    # that seed, so another number words it differently (re-rolling a caption)
    seed: int | None = None
    # sampling (with a seed): ComfyUI's generate defaults, but a cooler temperature
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    # > 1 discourages repeating tokens: a model looping ("and a red scarf, and a red scarf") runs
    # until it's cut off. Applies with or without sampling
    repetition_penalty: float = 1.0
    # LoRAs on the captioning model (abliterated / uncensored / prompt-style extractions):
    # [{"name": file in loras/, "strength": 1.0}]. Captioning only; training never uses them.
    loras: list = field(default_factory=list)
    format: str = "text"  # Style.format: "fibo_json" checks the answer is a structured caption
    # a picture with a mask (masked training) shows the model only what the mask keeps: the rest in
    # flat grey, cropped to the kept part, so its caption describes what the LoRA learns from
    hide_masked: bool = False
    # prompts to rewrite instead of images to caption (no dataset, nothing written; results only)
    rewrite: list[str] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | error | stopped | undone
    done: int = 0
    total: int = 0
    file: str = ""
    text: str = ""
    message: str = ""
    prompt_id: str = ""
    written: dict[str, str] = field(default_factory=dict)  # path -> the caption this job wrote
    previous: dict[str, str] = field(default_factory=dict)  # path -> the caption it replaced
    results: list[dict] = field(default_factory=list)  # preview: [{path, text, cut}]
    cut: list[str] = field(default_factory=list)  # paths whose answer reached max_length: cut off mid-way
    log: list[dict] = field(default_factory=list)  # [{t, level, msg}]: the job's own log
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("previous")
        d["written"] = list(self.written)
        d["replaced"] = len(self.previous)
        return d


_jobs: dict[str, CaptionJob] = {}
_lock = threading.Lock()
_loaded = False


def _file() -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "caption_jobs.json"


def _load() -> None:
    """Jobs from disk, once. One that was queued or running when ComfyUI went down didn't
    finish: what it wrote stays (and can be undone)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        rows = json.loads(_file().read_text())
    except (OSError, ValueError):
        return
    names = {f.name for f in fields(CaptionJob)}
    for r in rows:
        job = CaptionJob(**{k: v for k, v in r.items() if k in names})
        if job.status in ACTIVE:
            job.status, job.message = "stopped", "ComfyUI restarted before it finished"
        _jobs[job.job_id] = job


def _save() -> None:
    keep = sorted(_jobs.values(), key=lambda j: j.created, reverse=True)[:KEEP]
    for j in list(_jobs.values()):
        if j not in keep:
            del _jobs[j.job_id]
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(j) for j in keep]))
    tmp.replace(path)


def create(**kw) -> CaptionJob:
    with _lock:
        _load()
        job = CaptionJob(job_id="cap-" + secrets.token_hex(4), **kw)
        _jobs[job.job_id] = job
        if not job.preview:
            _save()
    _emit(job)
    return job


def get(job_id: str) -> CaptionJob | None:
    with _lock:
        _load()
        return _jobs.get(job_id)


def update(job: CaptionJob, **kw) -> None:
    with _lock:
        for k, v in kw.items():
            setattr(job, k, v)
        if not job.preview:
            _save()
    _emit(job)


GENERIC_TYPE = "stable_diffusion"  # CLIPLoader type that loads a Qwen-VL file as the stock chat model
_senses_cache: dict[str, tuple[float, tuple[bool, bool]]] = {}


def _senses(path: str) -> tuple[bool, bool]:
    """Whether a text encoder file carries a vision tower (it can see the picture) and an audio
    one (it can hear a video's sound: Gemma 4 E2B/E4B/12B): read from its safetensors header,
    cached by mtime."""
    import os
    import struct

    try:
        mtime = os.path.getmtime(path)
        hit = _senses_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = f.read(min(n, 64 << 20)).decode("utf-8", "replace")
        hears = any(k in header for k in ('"audio_model.', '"audio_tower.'))
        # Qwen-VL towers, or Gemma 4's (told apart from Gemma 3, whose chat format isn't wired, by its audio tower)
        sees = '"model.visual.' in header or '"visual.' in header or (hears and '"vision_model.' in header)
        _senses_cache[path] = (mtime, (sees, hears))
        return sees, hears
    except (OSError, struct.error):
        return False, False


def _has_vision(path: str) -> bool:
    return _senses(path)[0]


def model_senses() -> dict[str, dict]:
    """What each captioning model takes in: pictures (all), video ("clip": a timestamped clip, as
    Gemma 4 does; "stills": VIDEO_FRAMES frames) and sound."""
    import folder_paths

    out = {}
    for n in models():
        p = folder_paths.get_full_path("text_encoders", n)
        _sees, hears = _senses(p) if p else (True, False)
        out[n] = {"image": True, "video": "clip" if hears else "stills", "audio": hears, "frames": VIDEO_FRAMES}
    return out


def hearing_models() -> list[str]:
    """The captioning models that also hear a video's soundtrack."""
    import folder_paths

    return [n for n in models() if (p := folder_paths.get_full_path("text_encoders", n)) and _senses(p)[1]]


def models() -> list[str]:
    """Text encoders that can caption: ones that resolve to a file (dangling symlinks are
    listed by ComfyUI but can't load) and carry a vision tower."""
    import os

    import folder_paths

    out = []
    for n in folder_paths.get_filename_list("text_encoders"):
        p = folder_paths.get_full_path("text_encoders", n)
        if p and os.path.exists(p) and p.endswith(".safetensors") and _has_vision(p):
            out.append(n)
    return out


def history() -> list[dict]:
    """Every kept job, newest first, for the Jobs page (try-ons only while fresh)."""
    now = time.time()
    with _lock:
        _load()
        jobs = [j for j in _jobs.values() if not j.preview or j.status in ACTIVE or now - (j.finished or j.created) < PREVIEW_KEEP]
    return [_light(j) for j in sorted(jobs, key=lambda j: j.created, reverse=True)]


def recent(dataset: str | None = None) -> list[dict]:
    """The latest jobs (one dataset's, or all), plus any try-on still running or just
    finished, so a page opened mid-job (or reopened) picks it up."""
    now = time.time()
    with _lock:
        _load()
        jobs = [j for j in _jobs.values() if (dataset is None or j.dataset == dataset)
                and (not j.preview or j.status in ACTIVE or now - (j.finished or j.created) < PREVIEW_KEEP)]
    return [j.public() for j in sorted(jobs, key=lambda j: j.created, reverse=True)[:20]]


def active() -> list[dict]:
    """Every queued or running job, try-ons included: what the sidebar shows."""
    with _lock:
        _load()
        return [_light(j) for j in sorted(_jobs.values(), key=lambda j: j.created) if j.status in ACTIVE]


def _light(job: CaptionJob) -> dict:
    d = job.public()
    for k in ("prompt", "written", "paths", "log"):
        d.pop(k)
    return d


LOG_KEEP = 1000


def log(job: CaptionJob, msg: str, level: str = "info") -> None:
    """A line in the job's log (kept with the job, shown on its page)."""
    with _lock:
        job.log.append({"t": time.time(), "level": level, "msg": msg})
        del job.log[:-LOG_KEEP]


def _emit(job: CaptionJob) -> None:
    events.emit(CAPTION, _light(job))
    # mirrored so ZenKit's taskbar (in ComfyUI) shows it too, as it does training runs
    events.emit(events.ZENKIT_JOB, {
        "id": f"zentrainer:{job.job_id}",
        "name": "Rewrite prompts" if job.rewrite else f"{'Try captions' if job.preview else 'Caption'} {job.dataset}",
        "status": {"queued": "start", "running": "progress", "done": "done"}.get(job.status, "error"),
        "current": job.done, "total": job.total or job.preview or 0, "message": job.message or job.status,
        "source": "ZenTrainer",
    })


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:  # the template opened the think block: keep what follows it
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"^\s*(caption|description|tags)\s*:\s*", "", text.strip(), flags=re.I)
    return text.strip().strip('"').strip()


VIDEO_FRAMES = 4  # frames a video clip is shown as
VIDEO_NOTE = (f"These are {VIDEO_FRAMES} frames, in order, from one short video clip. Describe the clip as a whole, "
              "not each frame: also how it moves (subject and camera) and how it is edited (cuts, split screens, "
              "overlays, text, effects), if it is. Treat \"image\" in what follows as the clip.\n")


SOUND_NOTE = ("This is a short video clip, with its soundtrack. Describe the clip as a whole: what is in it, how "
              "it moves (subject and camera) and how it is edited (cuts, split screens, overlays, text, effects). "
              "Also describe what you hear: the music (genre, tempo, energy, drops), voices or other sounds, and how "
              "the cuts and motion follow the beat. Treat \"image\" in what follows as the clip.\n")


SUBJECT_NOTE = "\nRefer to the main subject as \"{subject}\" (exactly that, not a description of who they are)."
MASK_NOTE = ("\nThe flat grey parts of the picture are hidden on purpose: don't describe them, the background "
             "behind them or what might be there. Describe only what is shown.")


def instruction(job: CaptionJob, is_video: bool = False, hears: bool = False, masked: bool = False) -> str:
    """Everything the model is told: the UI shows the same pieces (routes: caption info "notes")."""
    text = (SOUND_NOTE if is_video and hears else VIDEO_NOTE if is_video else "") + job.prompt
    text = text + SUBJECT_NOTE.format(subject=job.subject) if job.subject else text
    return text + MASK_NOTE if masked else text


def todo(job: CaptionJob, items: list) -> list:
    """The images this job captions, from the dataset's scan."""
    from ..harness.data import caption_path

    root = None
    if job.mode == "selected":
        want = set(job.paths)
        out = []
        for it in items:
            root = root or _dataset_root(job)
            if str(it.path.relative_to(root)) in want:
                out.append(it)
    elif job.mode == "all":
        out = list(items)
    else:
        out = [it for it in items if not caption_path(it.path, job.set_name).is_file()]
    return out[: job.preview] if job.preview else out


def _dataset_root(job: CaptionJob) -> Path:
    from . import folders as datasets

    return datasets.folder(job.dataset)


def _answer(clip, tokens, job: CaptionJob) -> tuple[str, bool]:
    """The model's answer for these tokens, in the job's format, and whether it was cut off: it
    used every token it was allowed (an answer that ends itself stops on an end token first). A
    structured caption that isn't valid JSON gets more tries (sampled, so they differ); ValueError
    when none is."""
    limit = job.max_length or NO_LIMIT
    sampling = dict(temperature=job.temperature, top_p=job.top_p, top_k=job.top_k, repetition_penalty=job.repetition_penalty)
    if job.format != "fibo_json":
        ids = clip.generate(tokens, do_sample=job.seed is not None, seed=job.seed or 0, max_length=limit, **sampling)
        return clean(clip.decode(ids)), len(ids) >= limit
    err = None
    for i in range(JSON_TRIES):
        ids = clip.generate(tokens, do_sample=i > 0 or job.seed is not None, seed=(job.seed or 0) + i, max_length=limit,
                            **sampling)
        try:
            return structured.compact(structured.parse(clean(clip.decode(ids)))), False
        except ValueError as e:
            err = e
            log(job, f"answer {i + 1} wasn't a structured caption ({e}); trying again" if i + 1 < JSON_TRIES else
                f"answer {i + 1} wasn't a structured caption ({e})", "warn")
    raise ValueError(f"no structured caption in {JSON_TRIES} answers: {err}")


def _load_clip(job: CaptionJob):
    import nodes

    # the stock chat model, not an image model's conditioning wrapper of the same file
    # (MOSS's Qwen3.5 drops the picture and leaves a think block open)
    ctype = GENERIC_TYPE if job.clip_type in ("", "auto") else job.clip_type
    clip = _with_loras(nodes.CLIPLoader().load_clip(job.clip_name, ctype)[0], job)
    if not hasattr(clip, "generate"):
        raise RuntimeError(f"{job.clip_name} can't generate text; pick a language model (Qwen-VL, Gemma)")
    return clip


def _run_rewrite(job: CaptionJob) -> None:
    """Rewrite the job's prompts (no pictures): each into a structured caption, or by its prompt."""
    import comfy.model_management as mm
    import comfy.utils

    update(job, total=len(job.rewrite))
    clip = _load_clip(job)
    pbar = comfy.utils.ProgressBar(len(job.rewrite))
    for p in job.rewrite:
        if job.status != "running":
            break
        ask = structured.generate_instruction(p) if job.format == "fibo_json" else f"{job.prompt}\n\n{p}"
        try:
            out, cut = _answer(clip, clip.tokenize(ask, min_length=1), job)
            if cut:
                log(job, f"{p[:80]}: cut off at {job.max_length or NO_LIMIT} tokens", "warn")
        except ValueError as e:
            out = ""
            log(job, f"{p[:80]}: {e}", "warn")
        job.results.append({"path": "", "prompt": p, "text": out})
        log(job, f"{p[:80]} -> {out[:200]}")
        update(job, done=job.done + 1, text=out, message="")
        pbar.update_absolute(job.done)
        mm.throw_exception_if_processing_interrupted()


def run(job: CaptionJob) -> None:
    """Caption the job's images (or rewrite its prompts). Runs inside a ComfyUI prompt."""
    from . import segment

    segment.release()  # the mask editor's SAM3, if it's still held: the captioner wants the memory
    import comfy.model_management as mm
    import comfy.utils
    import folder_paths
    import torch

    from ..harness.data import caption_path, scan
    from ..harness.trainer import _comfy_compiler

    if job.status != "queued":  # cancelled while it waited
        return
    log(job, f"started: {job.clip_name}, rewriting {len(job.rewrite)} prompts" if job.rewrite else
        f"started: {job.clip_name}, into “{job.set_name}”, {job.mode} images" + (f", try-on of {job.preview}" if job.preview else ""))
    log(job, f"answers up to {job.max_length or f'{NO_LIMIT} (no limit)'} tokens, "
        + (f"sampled: seed {job.seed}, temperature {job.temperature}, top-p {job.top_p}, top-k {job.top_k}" if job.seed is not None
           else "the most likely answer (the same every run)")
        + (f", repetition penalty {job.repetition_penalty}" if job.repetition_penalty != 1.0 else ""))
    update(job, status="running", message=f"loading {job.clip_name}")
    # ComfyUI's model compiler (CUDA graphs) breaks token-by-token generation: a device-side
    # assert on the second token that takes the whole server down. Off while captioning, as
    # it is while training.
    _comfy_compiler(False)
    try:
        with torch.inference_mode():
            if job.rewrite:
                _run_rewrite(job)
                if job.status == "running":
                    update(job, status="done", finished=time.time())
                return
            root = _dataset_root(job)
            items = todo(job, scan(root, warn=False))
            update(job, total=len(items))
            log(job, f"{len(items)} to caption")
            if not items:
                log(job, "nothing to caption: every image already has one in this set")
                update(job, status="done", message="nothing to caption: every image already has one in this set")
                return
            clip = _load_clip(job)
            from ..harness.data import video
            # Gemma 4 takes a clip as a timestamped video and can hear it (its own chat template
            # carries the media tokens); the others see VIDEO_FRAMES stills
            gemma4 = "gemma4" in getattr(clip.tokenizer, "clip_name", "")
            hears = gemma4 and _senses(folder_paths.get_full_path("text_encoders", job.clip_name) or "")[1]
            if hears:
                log(job, "this model hears: videos are captioned with their sound")
            pbar = comfy.utils.ProgressBar(len(items))
            for it in items:
                if job.status != "running":  # cancelled
                    break
                is_video = video.is_video(it.path)
                mask = it.mask if job.hide_masked and not is_video else None  # masks are per image
                if gemma4:
                    tokens = _gemma4_tokens(clip, job, it.path, is_video, hears, mask)
                else:
                    image = _picture(it.path, mask)  # (1 or VIDEO_FRAMES, H, W, 3) in [0, 1]
                    tokens = clip.tokenize(instruction(job, is_video, masked=mask is not None), image=image, min_length=1)
                rel = str(it.path.relative_to(root))
                try:
                    text, cut = _answer(clip, tokens, job)
                except ValueError as e:  # no usable structured caption: this image keeps what it had
                    log(job, f"{rel}: skipped, {e}", "warn")
                    update(job, done=job.done + 1, file=it.path.name, text="", message="")
                    pbar.update_absolute(job.done)
                    mm.throw_exception_if_processing_interrupted()
                    continue
                if cut:
                    job.cut.append(rel)
                    log(job, f"{rel}: cut off at {job.max_length or NO_LIMIT} tokens (raise the length, or re-caption the cut ones)", "warn")
                if job.preview:
                    job.results.append({"path": rel, "text": text, "cut": cut})
                else:
                    target = caption_path(it.path, job.set_name)
                    if target.is_file() and rel not in job.previous:
                        job.previous[rel] = target.read_text(encoding="utf-8")
                    safeio.write_text(target, text + "\n")
                    job.written[rel] = text
                log(job, f"{rel}: {text}")
                update(job, done=job.done + 1, file=it.path.name, text=text, message="")  # loaded: the first one's in
                pbar.update_absolute(job.done)  # also raises on ComfyUI interrupt
                mm.throw_exception_if_processing_interrupted()
        if job.status == "running":
            log(job, (f"done: {len(job.written)} written, {len(job.previous)} of them replaced"
                      + (f", {len(job.cut)} cut off" if job.cut else "")) if not job.preview else "done (try-on: nothing saved)")
            update(job, status="done", finished=time.time())
    except BaseException as e:
        stopped = type(e).__name__ == "InterruptProcessingException"
        log(job, "interrupted" if stopped else f"failed: {type(e).__name__}: {e}", "warn" if stopped else "error")
        update(job, status="stopped" if stopped else "error", message=str(e) or type(e).__name__, finished=time.time())
        if not stopped:
            raise
    finally:
        _comfy_compiler(True)


def _with_loras(clip, job: CaptionJob):
    """The captioning model with the job's text-encoder LoRAs patched on."""
    import comfy.sd
    import comfy.utils
    import folder_paths

    for lo in job.loras or []:
        name, strength = str(lo.get("name", "")), float(lo.get("strength", 1.0))
        path = folder_paths.get_full_path("loras", name)
        if not path:
            raise FileNotFoundError(f"LoRA {name} not found in loras/")
        _model, clip = comfy.sd.load_lora_for_models(None, clip, comfy.utils.load_torch_file(path, safe_load=True), 0.0, strength)
        log(job, f"LoRA {name} at {strength:g}")
    return clip


TE_LORA_SUBDIR = "text_encoder_loras"

# LoRAs worth having on a captioner, each for the base it was extracted from (matched against
# the selected model's file name). Add one: an entry here.
TE_LORA_CATALOG = [
    {"id": "qwen3vl-8b-abliterated-v3", "label": "Abliterated v3", "repo": "nynxz/Qwen3-VL-8B-ComfyUI",
     "file": "loras/qwen3vl-8b-c_abliterated-v3_lora_rank1_fp32.safetensors", "size": 3559416,
     "base": r"qwen3[-_.]?vl[-_]?8b", "note": "Describes everything it sees instead of refusing or softening it"},
    {"id": "qwen3vl-8b-h3-rewriter", "label": "MiniMax H3 prompt style", "repo": "nynxz/Qwen3-VL-8B-ComfyUI",
     "file": "loras/minimax-h3-prompt-rewriter-8b_lora_rank256_bf16.safetensors", "size": 1396778384,
     "base": r"qwen3[-_.]?vl[-_]?8b", "note": "Writes in the style of H3's own prompts: captions for H3 training"},
    {"id": "qwen35-9b-uncensored", "label": "Uncensored", "repo": "nynxz/Qwen3.5-9b",
     "file": "loras/qwen35-9b-hauhaucs_uncensored_lora_rank4_fp32.safetensors", "size": 14044264,
     "base": r"qwen3\.?5[-_]9b", "note": "Describes everything it sees instead of refusing or softening it"},
]


def te_lora_catalog() -> list[dict]:
    """The catalog with where each one is if it's already downloaded (its loras/ name)."""
    have = {Path(n).name: n for n in te_loras()}
    return [{**e, "installed": have.get(Path(e["file"]).name)} for e in TE_LORA_CATALOG]


def download_te_lora(entry_id: str) -> dict:
    """Download a catalog LoRA into loras/text_encoder_loras/."""
    from ..models import downloads

    e = next((e for e in TE_LORA_CATALOG if e["id"] == entry_id), None)
    if e is None:
        raise ValueError(f"no catalog LoRA {entry_id!r}")
    name = Path(e["file"]).name
    url = f"https://huggingface.co/{e['repo']}/resolve/main/{e['file']}"
    job = downloads.manager().enqueue("caption", "loras", name, url, downloads.target_dir("loras") / TE_LORA_SUBDIR, e["size"])
    return {"download": job.public(), "name": f"{TE_LORA_SUBDIR}/{name}"}
_te_cache: dict[str, tuple[float, bool]] = {}


def _is_te_lora(path: str) -> bool:
    """A LoRA for a text encoder (ComfyUI's text_encoders.* keys, or kohya's lora_te*): read from
    its header, cached by mtime."""
    import os
    import struct

    try:
        mtime = os.path.getmtime(path)
        hit = _te_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = f.read(min(n, 16 << 20)).decode("utf-8", "replace")
        ok = '"text_encoders.' in header or '"lora_te' in header
        _te_cache[path] = (mtime, ok)
        return ok
    except (OSError, struct.error):
        return False


def te_loras() -> list[str]:
    """LoRAs in loras/ that patch a text encoder (the ones that make sense on a captioner)."""
    import folder_paths

    return [n for n in folder_paths.get_filename_list("loras")
            if n.endswith(".safetensors") and (p := folder_paths.get_full_path("loras", n)) and _is_te_lora(p)]


def add_te_lora(link: str, file: str | None = None) -> dict:
    """Download a text-encoder LoRA from a Hugging Face link (a repo, or a file in it) into
    loras/text_encoder_loras/. A repo with several .safetensors and no file picked returns
    {choices}."""
    from ..models import downloads, families
    from ..models.adapters import _HF_LINK

    m = _HF_LINK.match(link.strip())
    if not m:
        raise ValueError("paste a Hugging Face link: https://huggingface.co/owner/repo (or a file in it)")
    repo, path = m.group(1), (file or m.group(2) or "").strip("/")
    try:
        rows = [r for r in families.repo_files(repo, refresh=True) if r["path"].endswith(".safetensors")]
    except Exception as e:
        raise ValueError(f"couldn't list {repo}: {e}") from e
    if path:
        rows = [r for r in rows if r["path"] == path]
    if not rows:
        raise ValueError(f"no such .safetensors file in {repo}")
    if len(rows) > 1:
        return {"choices": [{"file": r["path"], "size": r.get("size")} for r in rows], "repo": repo}
    row = rows[0]
    target = downloads.target_dir("loras") / TE_LORA_SUBDIR
    url = f"https://huggingface.co/{repo}/resolve/main/{row['path']}"
    job = downloads.manager().enqueue("caption", "loras", Path(row["path"]).name, url, target, row.get("size"))
    return {"download": job.public(), "name": f"{TE_LORA_SUBDIR}/{Path(row['path']).name}"}


GEMMA_FPS = 2  # frames a second Gemma 4 sees of a clip (it subsamples whatever it's given to `fps`)


def _gemma4_tokens(clip, job: CaptionJob, path: Path, is_video: bool, hears: bool, mask: Path | None = None):
    """Gemma 4's prompt: the picture, or the clip as frames at GEMMA_FPS with its soundtrack."""
    from ..harness.data import video

    if not is_video:
        return clip.tokenize(instruction(job, masked=mask is not None), image=_picture(path, mask), skip_template=False,
                             min_length=1)
    p = video.probe(path)
    n = max(p.frames, 1)
    step = max(1.0, p.fps / GEMMA_FPS)
    idx = sorted({min(n - 1, round(i * step)) for i in range(max(1, int(n / step)))})[:40]
    w, h = p.width, p.height
    scale = min(1.0, 640 / max(w, h))
    frames = video.read(path, idx, (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)))
    sound = None
    if hears and (got := video.read_audio(path, 0.0, n / p.fps)) is not None:
        wave, sr = got
        sound = {"waveform": wave[None], "sample_rate": sr}
    return clip.tokenize(instruction(job, True, sound is not None), video=frames, fps=1, audio=sound,
                         skip_template=False, min_length=1)


MASK_GREY = 128  # what hidden parts show as: flat, mid grey (no edges or colour to describe)
MASK_MARGIN = 0.08  # around the kept part when cropping to it, as a share of the picture's longer side


def hide_masked(im, mask):
    """`im` (RGB) with what `mask` leaves out (black; soft edges blend) in flat grey, cropped to the
    part it keeps plus MASK_MARGIN: a small subject fills the model's view. A mask that keeps
    nothing leaves the picture as it was (nothing would be left to describe)."""
    from PIL import Image

    m = mask.convert("L").resize(im.size, Image.Resampling.BILINEAR)
    box = m.point(lambda v: 255 if v >= 128 else 0).getbbox()
    if box is None:
        return im
    im = Image.composite(im, Image.new("RGB", im.size, (MASK_GREY,) * 3), m)
    pad = round(MASK_MARGIN * max(im.size))
    return im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad)))


def _picture(path: Path, mask: Path | None = None):
    """What the model looks at: the image at most 1024 px a side, or VIDEO_FRAMES frames spread
    over a video at most 640 px (several pictures cost tokens); sides in multiples of 28 (the
    vision encoder's patch grid). With `mask`, only what it keeps (hide_masked)."""
    import numpy as np
    import torch
    from PIL import Image

    from ..harness.data import video

    if video.is_video(path):
        p = video.probe(path)
        n = max(p.frames, 1) if p else 1
        idx = sorted({min(n - 1, round((n - 1) * (i + 0.5) / VIDEO_FRAMES)) for i in range(VIDEO_FRAMES)})
        w, h = (p.width, p.height) if p else (640, 640)
        scale = min(1.0, 640 / max(w, h))
        size = (max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28))
        return video.read(path, idx, size)
    from PIL import ImageOps

    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    if mask is not None:
        with Image.open(mask) as m:
            rgb = hide_masked(rgb, ImageOps.exif_transpose(m))
    w, h = rgb.size
    scale = min(1.0, 1024 / max(w, h))
    size = (max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28))
    rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)[None]


def cancel(job: CaptionJob) -> None:
    """Stop at the next image (what's written stays, and can be undone)."""
    if job.status in ACTIVE:
        log(job, "cancelled", "warn")
        update(job, status="stopped", message="cancelled", finished=time.time())


def undo(job: CaptionJob) -> int:
    """Put every caption the job wrote back as it was: replaced ones get their old text,
    new ones are removed. Only while nothing else has changed them since."""
    from ..harness.data import caption_path

    if job.status in ACTIVE:
        raise ValueError("stop the job before undoing it")
    if job.status == "undone":
        return 0
    root = _dataset_root(job)
    restored = kept = 0
    for rel, wrote in job.written.items():
        target = caption_path(root / rel, job.set_name)
        current = target.read_text(encoding="utf-8").strip() if target.is_file() else None
        if current != wrote.strip():  # edited since: that edit wins
            kept += 1
            continue
        if rel in job.previous:
            safeio.write_text(target, job.previous[rel])
        else:
            target.unlink(missing_ok=True)
        restored += 1
    msg = f"undid {restored} caption{'s' if restored != 1 else ''}"
    if kept:
        msg += f"; kept {kept} you edited since"
    log(job, msg)
    update(job, status="undone", message=msg)
    return restored
