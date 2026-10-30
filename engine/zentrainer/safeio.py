"""Writes that can't leave a user's file half-written, and reads that don't throw data away.

A crash, a kill or a full disk mid-write must leave the old file whole: everything is written
to a temp file beside the target, flushed, then swapped in with os.replace (atomic on one
filesystem). A file that won't parse is set aside as <name>.corrupt-<time> before anything can
save over it, so a bad read never turns into "start from empty and overwrite"."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any


def _tmp(path: Path) -> Path:
    # unique per thread, hidden, and not a media/caption extension a dataset scan would pick up
    return path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")


def write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    tmp = _tmp(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _replace(tmp: Path, path: Path) -> None:
    # Windows: os.replace fails with WinError 5 while another process (the web app polling run.json,
    # an antivirus scan) has the target open; the lock is brief, so retry for a few seconds
    for attempt in range(100):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 99:
                raise
            time.sleep(0.05)


def write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> None:
    write_bytes(path, text.encode(encoding))


def write_json(path: str | os.PathLike, data: Any, indent: int | None = 2) -> None:
    write_text(path, json.dumps(data, indent=indent))


def set_aside(path: str | os.PathLike) -> Path | None:
    """Move an unreadable file out of the way (kept, never deleted), so the next save can't
    destroy what's left in it."""
    path = Path(path)
    keep = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        os.replace(path, keep)
    except OSError:
        return None
    logging.warning("[ZenTrainer] %s couldn't be read; kept it as %s and started a fresh one", path, keep.name)
    return keep


def read_json(path: str | os.PathLike, default: Any) -> Any:
    """The file's JSON, `default` when it doesn't exist, and when it won't parse: `default`
    after setting the broken file aside (see set_aside)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(text)
    except ValueError:
        set_aside(path)
        return default


def free_name(path: Path) -> Path:
    """`path`, or `name (2).ext`, `name (3).ext`... the first that doesn't exist."""
    if not path.exists():
        return path
    n = 2
    while True:
        alt = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not alt.exists():
            return alt
        n += 1
