"""The training queue: an ordered, persistent list of runs, dispatched one at a time.

Runs wait here (status "queued"), not in ComfyUI's prompt queue, so they can be
reordered, removed or held while the queue is paused, and the order survives a
restart. The worker hands the head of the queue to ComfyUI (a one-node session job)
only when no run is active, so training jobs never pile up behind each other there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from .. import state

TERMINAL = {"done", "stopped", "error", "interrupted", "rejected"}
_lock = threading.Lock()


class Queue:
    def __init__(self, path: Path):
        self.path = path
        self.current: str | None = None  # dispatched to ComfyUI, not yet finished
        # runs paused that let go of ComfyUI for a workflow (Trainer._park): nothing else starts
        # until they're resumed or stopped, so the GPU they freed goes to that workflow. In memory:
        # after a restart a parked run is interrupted, to resume as any other
        self.parked: set[str] = set()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        return {"paused": bool(data.get("paused", False)), "order": list(data.get("order", [])),
                "next": data.get("next")}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    def snapshot(self) -> dict:
        with _lock:
            return {**self._read(), "current": self.current}

    def _back_in_line(self, run_id: str) -> None:
        """A run going back in the queue isn't the one running: its last job is over. The worker lets
        go of the current run once it sees it end, but a resume or retry right after a stop can
        requeue it first, and the worker then waited on it for good (it showed twice, as running
        and as waiting). Nothing starts early: the worker still waits for no run to be active."""
        if self.current == run_id:
            self.current = None

    def add(self, run_id: str, front: bool = False) -> None:
        with _lock:
            self._back_in_line(run_id)
            d = self._read()
            d["order"] = [r for r in d["order"] if r != run_id]
            d["order"].insert(0, run_id) if front else d["order"].append(run_id)
            self._write(d)

    def insert(self, run_id: str, at: int) -> None:
        with _lock:
            self._back_in_line(run_id)
            d = self._read()
            d["order"] = [r for r in d["order"] if r != run_id]
            d["order"].insert(max(0, at), run_id)
            self._write(d)

    def remove(self, run_id: str) -> bool:
        with _lock:
            d = self._read()
            if run_id not in d["order"]:
                return False
            d["order"].remove(run_id)
            self._write(d)
            return True

    def reorder(self, order: list[str]) -> None:
        """Apply a new order to the queued runs; ids not in the queue are ignored and
        queued ids missing from `order` keep their place at the end."""
        with _lock:
            d = self._read()
            known = set(d["order"])
            head = [r for r in order if r in known]
            d["order"] = head + [r for r in d["order"] if r not in head]
            self._write(d)

    def set_paused(self, paused: bool) -> None:
        with _lock:
            d = self._read()
            d["paused"] = paused
            self._write(d)

    def start_next(self, run_id: str) -> None:
        """Start this queued run next even while the queue is paused (a swap); the rest stay held."""
        with _lock:
            d = self._read()
            d["order"] = [run_id] + [r for r in d["order"] if r != run_id]
            d["next"] = run_id
            self._write(d)

    def pop_next(self) -> str | None:
        with _lock:
            d = self._read()
            nxt, d["next"] = d.get("next"), None
            if nxt in d["order"]:
                d["order"].remove(nxt)
                self._write(d)
                return nxt
            if d["paused"] or not d["order"]:
                if nxt:
                    self._write(d)
                return None
            run_id = d["order"].pop(0)
            self._write(d)
            return run_id


_queue: Queue | None = None


def get() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(state.store().root.parent / "queue.json")
    return _queue


async def worker(dispatch, swapped=None, parked=None, interval: float = 1.5) -> None:
    """Dispatch the next queued run whenever nothing is active (or parked). `dispatch(run_id)` is
    the coroutine that submits the session job to ComfyUI; `swapped(run_id)` requeues a run
    stopped by a swap, and `parked(run_id)` keeps one that paused for a ComfyUI workflow paused,
    once it has stopped."""
    from ..harness import control

    q = get()
    store = state.store()
    while True:
        try:
            if q.current is not None:
                try:
                    meta = store.meta(q.current)
                except (FileNotFoundError, ValueError):
                    meta = {"status": "error"}
                if meta["status"] in TERMINAL and control.get(q.current) is None:
                    if meta.get("swap_resume") and swapped is not None:
                        await swapped(q.current)
                    elif meta.get("park") and parked is not None:
                        await parked(q.current)
                    q.current = None
            if q.current is None and not control.active() and not q.parked:
                run_id = q.pop_next()
                if run_id is not None:
                    q.current = run_id
                    await dispatch(run_id)
        except Exception:
            logging.exception("[ZenTrainer] queue worker")
        await asyncio.sleep(interval)
