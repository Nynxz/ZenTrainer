"""The analysis models' files: found in a ComfyUI model folder's zentrainer/ subfolder, or fetched
there from Hugging Face (at a pinned commit) through the app's downloads, which show the progress."""

from __future__ import annotations

import time
from pathlib import Path

SUBDIR = "zentrainer"


def find(folder: str, file: str) -> Path | None:
    import folder_paths

    p = folder_paths.get_full_path(folder, f"{SUBDIR}/{file}")
    return Path(p) if p else None


def fetch(folder: str, file: str, repo: str, revision: str, path: str, size: int, what: str) -> Path:
    """The file, downloading it first if it isn't there (waits for the download). `what` names it
    in the error and the downloads list."""
    from ..models import downloads

    if (p := find(folder, file)) is not None:
        return p
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    job = downloads.manager().enqueue(what, folder, file, url, downloads.target_dir(folder) / SUBDIR, size)
    while job.status in ("queued", "downloading"):
        time.sleep(0.25)
    if job.status != "done":
        raise RuntimeError(f"the {what} model {file} didn't download ({job.error or job.status})")
    return Path(job.target) / file
