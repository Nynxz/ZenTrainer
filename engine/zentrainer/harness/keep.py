"""Which checkpoints a run keeps (output.keep): all of them, the latest few, the best few by
likeness or validation loss (scored the way the run page marks its best one), or each new best:
every save that beat all the saves before it, so the run's record-setters all stay.

Only scheduled saves (save_every) are ever removed. The final LoRA, the newest checkpoint (what a
resume continues from), and saves asked for by hand or made at a stop always stay, and so does a
checkpoint without a score yet: likeness lands a little after its samples, so it's ranked at a
later save or at the end. What's removed goes from the run's checkpoints to its "pruned" stream,
with the reason.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

SCHEDULED = re.compile(r"_step\d+\.safetensors$")  # Checkpoints.on_step's f"step{step:06d}" saves


def val_at(validation: list[dict], step: int) -> float | None:
    """A checkpoint's validation loss: the latest evaluation at or before its step."""
    v = None
    for r in validation:
        if r.get("step", 0) <= step and r.get("val_loss") is not None:
            v = r["val_loss"]
    return v


def likeness_at(likeness: list[dict], step: int) -> float | None:
    """A checkpoint's likeness: its samples' mean score (manual tries left out)."""
    xs = [r["score"] for r in likeness if r.get("step") == step and not r.get("manual") and r.get("score") is not None]
    return sum(xs) / len(xs) if xs else None


def to_prune(checkpoints: list[dict], rule: str, count: int, validation: list[dict],
             likeness: list[dict]) -> list[tuple[dict, str]]:
    """The checkpoints `rule` removes, each with why: [(row, reason)]."""
    if rule == "all":
        return []
    saves = [c for c in checkpoints if not c.get("final")]
    if not saves:
        return []
    newest = max(c["step"] for c in saves)
    scheduled = [c for c in saves if SCHEDULED.search(c.get("path", ""))]
    if rule == "latest":
        ranked = sorted(scheduled, key=lambda c: -c["step"])
        return [(c, f"older than the {count} kept") for c in ranked[count:] if c["step"] != newest]
    metric = rule.removesuffix("_new_best")
    if metric not in ("val_loss", "likeness"):
        raise ValueError(f"unknown output.keep {rule!r}")
    low = metric == "val_loss"
    score = {c["step"]: val_at(validation, c["step"]) if low else likeness_at(likeness, c["step"]) for c in scheduled}
    if rule.endswith("_new_best"):
        return _not_new_bests(scheduled, score, low, newest)
    ranked = sorted((c for c in scheduled if score[c["step"]] is not None), key=lambda c: score[c["step"]],
                    reverse=not low)
    say = "validation loss {:.4f}; the {} lowest kept" if low else "likeness {:.2f}; the {} highest kept"
    return [(c, say.format(score[c["step"]], count)) for c in ranked[count:] if c["step"] != newest]


def _not_new_bests(scheduled: list[dict], score: dict, low: bool, newest: int) -> list[tuple[dict, str]]:
    """The saves that weren't a new best when they came: in step order, each that didn't beat every
    save before it. Once a save is still unscored, nothing after it is judged: its score could still
    make a later save not a new best."""
    gone, best, best_step = [], None, None
    for c in sorted(scheduled, key=lambda c: c["step"]):
        s = score[c["step"]]
        if s is None:
            break
        if best is None or (s < best if low else s > best):
            best, best_step = s, c["step"]
        elif c["step"] != newest:
            what = f"validation loss {s:.4f}, not below {best:.4f}" if low else f"likeness {s:.2f}, not above {best:.2f}"
            gone.append((c, f"{what} (step {best_step}): only each new best kept"))
    return gone


def faces_missing(spec) -> str | None:
    """Why keeping by likeness can't work for this run (a TrainSpec), or None: likeness is only
    scored for a dataset with faces on, and without scores the rule would keep every checkpoint."""
    from ..datasets import folders as datasets

    if spec.output.keep.removesuffix("_new_best") == "likeness" and spec.save_every \
            and not datasets.get_settings(spec.dataset.folder).get("faces"):
        return (f"keeping checkpoints by likeness needs faces on for the dataset {spec.dataset.folder!r} "
                "(its page: Settings, Check faces), or keep them another way")
    return None


def remove(row: dict, run_id: str, states_dir: Path) -> int | None:
    """Delete one pruned checkpoint's LoRA and resume state, or None when its LoRA isn't one of
    this run's scheduled saves (<loras>/.../checkpoints/<run_id>/..._stepNNNNNN.safetensors, wherever
    output.lora_dir had it): then nothing is deleted. A state only from this run's state folder.
    Returns the bytes freed."""
    lora = Path(row["path"]).resolve()
    if lora.parent.name != run_id or lora.parent.parent.name != "checkpoints" or not SCHEDULED.search(lora.name):
        logging.warning("[ZenTrainer] keep: left %s alone, it isn't one of this run's checkpoints", lora)
        return None
    freed = 0
    state = Path(row["state"]).resolve() if row.get("state") else None
    for f in (lora, state if state and state.parent == states_dir.resolve() else None):
        if f is not None and f.is_file():
            freed += f.stat().st_size
            f.unlink()
    return freed
