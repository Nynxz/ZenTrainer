"""The run store: one directory per run, written by the engine, read by the UI.

    <root>/<run_id>/
        spec.json       the TrainSpec as submitted
        run.json        {run_id, name, prompt_id, status, created, updated, final_step}
        metrics.jsonl   one line per optimizer step
        checkpoints.jsonl

Files rather than a database so a run can be inspected, copied or deleted by hand.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from .. import safeio

_lock = threading.Lock()
ACTIVE = ("loading", "training", "paused")   # handed to ComfyUI: started
ENDED = ("done", "stopped", "error", "interrupted", "rejected")


class RunStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)

    def dir(self, run_id: str) -> Path:
        # run ids come from URLs; never let one escape the root
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            raise ValueError(f"bad run id {run_id!r}")
        return self.root / run_id

    def create(self, run_id: str, spec: dict[str, Any], prompt_id: str | None = None) -> dict[str, Any]:
        d = self.dir(run_id)
        d.mkdir(parents=True, exist_ok=False)
        safeio.write_json(d / "spec.json", spec)
        now = time.time()
        meta = {
            "run_id": run_id,
            "name": spec.get("name", run_id),
            "prompt_id": prompt_id,
            "status": "queued",
            "created": now,
            "updated": now,
            "final_step": None,
        }
        self._write_meta(run_id, meta)
        return meta

    def set_spec(self, run_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Replace a run's spec (only for a run that hasn't started: the caller checks)."""
        with _lock:
            safeio.write_json(self.dir(run_id) / "spec.json", spec)
            meta = self.meta(run_id)
            meta.update(name=spec.get("name", run_id), updated=time.time())
            self._write_meta(run_id, meta)
        return meta

    def update(self, run_id: str, **fields: Any) -> dict[str, Any]:
        with _lock:
            meta = self.meta(run_id)
            now = time.time()
            new, old = fields.get("status"), meta.get("status")
            # sampling a saved checkpoint queues the run and puts it back as it was: not a go of its
            # own, so its times stay as they are
            sampling = bool(meta.get("sample_job") or fields.get("sample_job"))
            # when it ran: `started` is this go (a resume starts a new one), `first_started` the
            # first; `ended` when it last stopped, finished or failed (cleared when it's queued again)
            if sampling:
                pass
            elif new in ACTIVE and old not in ACTIVE:
                fields.setdefault("started", now)
                fields.setdefault("ended", None)
                if not meta.get("first_started"):
                    fields.setdefault("first_started", now)
            elif new in ENDED and old not in ENDED:
                fields.setdefault("ended", now)
            elif new == "queued":
                fields.setdefault("ended", None)
            # a finished run resumes from nothing: what a resume started from (and its training
            # state, which the clean-up may delete) shouldn't follow it into later jobs
            if new == "done" and not sampling:
                fields.setdefault("resume_from", None)
                fields.setdefault("resume_state", None)
            meta.update(fields, updated=now)
            self._write_meta(run_id, meta)
        return meta

    def meta(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.dir(run_id) / "run.json").read_text())

    def spec(self, run_id: str) -> dict[str, Any]:
        """The run's spec in this version's layout (spec.migrate): renamed fields carried over,
        ones this version no longer has left out, so an older run still loads, resumes and copies."""
        from ..spec import migrate

        return migrate(json.loads((self.dir(run_id) / "spec.json").read_text()))[0]

    def list(self) -> list[dict[str, Any]]:
        out = []
        for d in self.root.iterdir():
            if (d / "run.json").is_file():
                try:
                    out.append(json.loads((d / "run.json").read_text()))
                except (OSError, json.JSONDecodeError):
                    continue
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def append(self, run_id: str, stream: str, row: dict[str, Any]) -> None:
        with _lock, open(self.dir(run_id) / f"{stream}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def truncate(self, run_id: str, stream: str, keep_through_step: int) -> None:
        """Drop rows after a step (they're about to be re-trained on resume)."""
        rows = [r for r in self.read(run_id, stream) if r.get("step", 0) <= keep_through_step]
        self.rewrite(run_id, stream, rows)

    def rewrite(self, run_id: str, stream: str, rows: list[dict[str, Any]]) -> None:
        # swapped in whole: a crash or a full disk mid-write leaves the old stream, never an empty one
        with _lock:
            safeio.write_text(self.dir(run_id) / f"{stream}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))

    def read(self, run_id: str, stream: str) -> list[dict[str, Any]]:
        p = self.dir(run_id) / f"{stream}.jsonl"
        if not p.is_file():
            return []
        out = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                # a line cut short (the disk filled, or the trainer is mid-append): skip it, keep the rest
                logging.debug("[ZenTrainer] %s/%s.jsonl: skipped an unreadable line", run_id, stream)
        return out

    def _write_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        p = self.dir(run_id) / "run.json"
        safeio.write_json(p, meta)
