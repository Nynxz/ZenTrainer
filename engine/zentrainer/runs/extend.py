"""Training a finished run on: what moves out of the way first."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def keep_final(store, run_id: str, ckpt: dict, lora_dir: str | None) -> dict:
    """A finished run about to train on: its final LoRA (and training state) move to where a
    save at that step goes, <name>_step<N> in the run's checkpoints, and its row says so. The
    new final is written under the old final's name, so it would otherwise overwrite it."""
    from ..harness.trainer import lora_root, state_dir, state_for

    root, base = lora_root(lora_dir)
    old = Path(ckpt["path"])
    name = f"{old.stem}_step{ckpt['step']:06d}"
    new = base / "checkpoints" / run_id / f"{name}.safetensors"
    new.parent.mkdir(parents=True, exist_ok=True)
    held = state_for(ckpt, run_id)  # before the move: older runs kept it beside the LoRA
    shutil.move(str(old), str(new))
    row = {**ckpt, "path": str(new), "lora_name": os.path.relpath(new, root), "final": False}
    if held:
        moved = state_dir(run_id) / f"{name}.pt"
        moved.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(held, str(moved))
        row["state"] = str(moved)
    store.rewrite(run_id, "checkpoints", [row if c.get("path") == ckpt["path"] else c
                                          for c in store.read(run_id, "checkpoints")])
    return row


def keep_stage_steps(spec, old_steps: int) -> None:
    """A run that has trained, given more steps (Train more, or Edit & resume): its lower-resolution
    stages and their blend are shares of the run, so a longer run would put them later, and one
    already at its own resolution would drop back down and blend again (VRAM down and up, the epoch
    see-sawing). Scaled to the new total, they fall on the same steps as before."""
    new = spec.optim.steps
    if new <= old_steps or not spec.dataset.stages:
        return
    k = old_steps / new
    spec.dataset.stages = [{**st, "until": round(float(st["until"]) * k, 6)} for st in spec.dataset.stages]
    spec.dataset.stage_blend = round(spec.dataset.stage_blend * k, 6)
