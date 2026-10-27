"""Model downloads: one at a time, resumable, with progress on the websocket.

Only files the family catalog knows can be downloaded (callers name a family/folder/file, never
a URL), and they land in the user's own model library: the first folder of that type outside
this ComfyUI's models/ (an extra_model_paths entry), so the main ComfyUI sees them too.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import events
from . import families

log = logging.getLogger("zentrainer")
EVENT = "zentrainer.download"  # Download.public()
CHUNK = 1 << 20


@dataclass
class Download:
    id: str
    family: str
    folder: str
    name: str
    url: str
    target: str  # directory
    total: int | None = None
    done: int = 0
    status: str = "queued"  # queued | downloading | done | error | cancelled
    error: str | None = None
    speed: float = 0.0  # bytes/s
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("url")
        return d


def _hf_request(url: str, headers: dict) -> urllib.request.Request:
    """A download's request, its Hugging Face token only ever to Hugging Face itself and not on to
    where it redirects (its CDN takes a signed URL, not the token): a template's file on another
    host never sees it."""
    headers = dict(headers)
    token = headers.pop("Authorization", None)
    req = urllib.request.Request(url, headers=headers)
    if token and (urllib.parse.urlsplit(url).hostname or "") in ("huggingface.co", "www.huggingface.co"):
        req.add_unredirected_header("Authorization", token)
    return req


def target_dir(folder: str) -> Path:
    """Where a download of this folder type goes: the user's library before this install's own."""
    import folder_paths

    paths = [Path(p) for p in folder_paths.get_folder_paths(folder)]
    own = Path(folder_paths.models_dir).resolve()
    outside = [p for p in paths if own not in p.resolve().parents and p.resolve() != own]
    for p in outside + paths:
        if p.is_dir() and os.access(p, os.W_OK):
            return p
    if not paths:
        raise ValueError(f"ComfyUI has no {folder} folder")
    paths[0].mkdir(parents=True, exist_ok=True)
    return paths[0]


class Downloads:
    def __init__(self) -> None:
        self._jobs: dict[str, Download] = {}
        self._cancel: set[str] = set()
        self._cv = threading.Condition()
        self._worker: threading.Thread | None = None

    def list(self) -> list[dict]:
        with self._cv:
            return [j.public() for j in sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)]

    def start(self, family: str, folder: str, name: str) -> Download:
        f = families.find_file(family, folder, name)
        if f is None:
            raise ValueError(f"{name!r} isn't a {folder} file of {family!r}")
        return self.enqueue(family, folder, name, f.url, target_dir(folder), f.size)

    def enqueue(self, family: str, folder: str, name: str, url: str, target: Path, size: int | None) -> Download:
        """Queue a file whose URL the caller has already vetted (a catalog entry, never a raw URL)."""
        if os.sep in name or "/" in name or name.startswith("."):
            raise ValueError("bad file name")
        with self._cv:
            for j in self._jobs.values():
                if j.name == name and j.folder == folder and j.status in ("queued", "downloading"):
                    return j
            target.mkdir(parents=True, exist_ok=True)
            job = Download(uuid.uuid4().hex[:10], family, folder, name, url, str(target), total=size)
            self._jobs[job.id] = job
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="zentrainer-downloads", daemon=True)
                self._worker.start()
            self._cv.notify_all()
        events.emit(EVENT, job.public())
        return job

    def cancel(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("queued", "downloading"):
                return False
            self._cancel.add(job_id)
            if job.status == "queued":
                job.status = "cancelled"
                events.emit(EVENT, job.public())
        return True

    def clear(self) -> None:
        """Forget finished downloads."""
        with self._cv:
            for k in [k for k, j in self._jobs.items() if j.status in ("done", "error", "cancelled")]:
                del self._jobs[k]

    def _next(self) -> Download | None:
        with self._cv:
            while True:
                q = [j for j in self._jobs.values() if j.status == "queued"]
                if q:
                    return min(q, key=lambda j: j.created)
                if not self._cv.wait(timeout=60):
                    self._worker = None  # idle: this thread ends; start() makes a new one
                    return None

    def _run(self) -> None:
        while (job := self._next()) is not None:
            try:
                self._fetch(job)
            except Exception as e:
                job.status, job.error = "error", str(e)
                log.warning("[ZenTrainer] download %s failed: %s", job.name, e)
            events.emit(EVENT, job.public())

    def _fetch(self, job: Download) -> None:
        final = Path(job.target) / job.name
        if final.exists():
            job.status, job.done = "done", final.stat().st_size
            return
        part = final.with_name(final.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if job.total and shutil.disk_usage(job.target).free < job.total - have:
            raise OSError(f"not enough disk space in {job.target} for {job.total / 1e9:.1f} GB")
        headers = dict(families._hf_headers())
        if have:
            headers["Range"] = f"bytes={have}-"
        job.status = "downloading"
        events.emit(EVENT, job.public())
        req = _hf_request(job.url, headers)
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                m = families.HF_URL.match(job.url)
                where = f"https://huggingface.co/{m.group(1)}" if m else "the repo"
                if req.has_header("Authorization"):
                    raise PermissionError(f"{e.code}: accept the license at {where} with the account your HF_TOKEN belongs to") from e
                raise PermissionError(f"{e.code}: gated repo: accept the license at {where}, then set HF_TOKEN and restart") from e
            raise
        with resp:
            if have and resp.status != 206:  # server ignored the range: start over
                have = 0
            length = resp.headers.get("Content-Length")
            if length:
                job.total = have + int(length)
            job.done = have
            last_emit, window_t, window_b = 0.0, time.time(), have
            with open(part, "ab" if have else "wb") as out:
                while chunk := resp.read(CHUNK):
                    if job.id in self._cancel:
                        job.status = "cancelled"
                        return
                    out.write(chunk)
                    job.done += len(chunk)
                    now = time.time()
                    if now - window_t >= 1.0:
                        job.speed = (job.done - window_b) / (now - window_t)
                        window_t, window_b = now, job.done
                    if now - last_emit >= 0.5:
                        last_emit = now
                        events.emit(EVENT, job.public())
        if job.total and job.done != job.total:
            raise OSError(f"incomplete: {job.done} of {job.total} bytes")
        part.replace(final)
        job.status = "done"


_downloads: Downloads | None = None


def manager() -> Downloads:
    global _downloads
    if _downloads is None:
        _downloads = Downloads()
    return _downloads
