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
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO


def _tmp(path: Path) -> Path:
    # unique per thread, hidden, and not a media/caption extension a dataset scan would pick up
    return path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")


def write_bytes(path: str | os.PathLike, data: bytes) -> None:
    write_with(path, lambda f: f.write(data))


def write_with(path: str | os.PathLike, write: Callable[[BinaryIO], Any], sync_dir: bool = False) -> None:
    """write_bytes for a writer that streams into an open file (torch.save). `sync_dir` also puts
    the rename itself on disk, for a file whose older copies are deleted once it's written."""
    def fill(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())

    _swap_in(Path(path), fill, sync_dir)


def write_to_path(path: str | os.PathLike, write: Callable[[str], Any], sync_dir: bool = False) -> None:
    """write_with for a writer that only takes a file name (safetensors' save_file, which writes
    straight from the tensors rather than building the file in memory first)."""
    def fill(tmp: Path) -> None:
        write(str(tmp))
        with _open_held(tmp, "rb+") as f:
            os.fsync(f.fileno())

    _swap_in(Path(path), fill, sync_dir)


def _swap_in(path: Path, fill: Callable[[Path], None], sync_dir: bool) -> None:
    tmp = _tmp(path)
    try:
        fill(tmp)
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if sync_dir and os.name != "nt":  # Windows can't open a folder to flush it
        try:
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:  # some shares and mounts can't flush a folder; the file itself is on disk
            pass


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


def _open_held(path: Path, mode: str):
    # the same brief Windows lock as _replace, on opening a file another writer just closed
    for attempt in range(100):
        try:
            return open(path, mode)
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
