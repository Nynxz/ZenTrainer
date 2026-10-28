"""Commands into a running session: pause, resume, stop, save, sample, samples, set_lr.

The HTTP routes push commands; the training loop drains them between optimizer
steps, so every command lands at a clean step boundary.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any

COMMANDS = {"pause", "resume", "stop", "save", "sample", "samples", "set_lr", "profile"}


@dataclass
class Command:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


class Control:
    def __init__(self):
        self._q: queue.Queue[Command] = queue.Queue()
        self._resume = threading.Event()
        self._resume.set()

    def send(self, name: str, **args) -> None:
        if name not in COMMANDS:
            raise ValueError(f"unknown command {name!r}; have {', '.join(sorted(COMMANDS))}")
        if name == "pause":
            self._resume.clear()
        elif name in ("resume", "stop"):
            self._resume.set()  # a stop must wake a paused loop
        self._q.put(Command(name, args))

    def drain(self) -> list[Command]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def wait_resumed(self, timeout: float) -> bool:
        return self._resume.wait(timeout)


_sessions: dict[str, Control] = {}
_lock = threading.Lock()


def register(run_id: str) -> Control:
    with _lock:
        c = _sessions[run_id] = Control()
        return c


def unregister(run_id: str) -> None:
    with _lock:
        _sessions.pop(run_id, None)


def get(run_id: str) -> Control | None:
    return _sessions.get(run_id)


def active() -> bool:
    return bool(_sessions)
