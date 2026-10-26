"""Model profiles: settings hand-picked for one model family, beyond what the generic
defaults and ComfyUI's templates give (MOSS Image wants skip-layer guidance on layer 22,
50 steps, CFG 3...). Each is a JSON file in engine/profiles/:

    families  family ids it applies to (families.py ids: "moss-image", "krea-2", ...)
    status    "recipe": the model makers' published settings, applied as written;
              "verified": we've trained with it and checked the results
    summary   one line for the UI
    sources   [{label, url}] where the settings come from
    train     partial TrainSpec merged onto the form (dataset, network, optim, objective...)
    samples   partial SampleSpec merged onto the sample settings
    notes     advice the form can't express

A profile only holds what its source says: no guessed learning rates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..spec import SampleSpec, migrate

log = logging.getLogger("zentrainer")
ROOT = Path(__file__).resolve().parents[2] / "profiles"
STATUSES = ("recipe", "verified")

_cache: dict[str, dict] | None = None


def _check(name: str, p: dict) -> None:
    if p.get("status") not in STATUSES:
        raise ValueError(f"{name}: status must be one of {STATUSES}")
    if not isinstance(p.get("families"), list) or not p["families"]:
        raise ValueError(f"{name}: needs a families list")
    sample_fields = set(SampleSpec.__dataclass_fields__)
    if unknown := set(p.get("samples", {})) - sample_fields:
        raise ValueError(f"{name}: unknown samples field(s) {sorted(unknown)}")
    _, dropped = migrate(p.get("train", {}))
    if dropped:
        raise ValueError(f"{name}: unknown train field(s) {dropped}")


def load(root: Path = ROOT) -> dict[str, dict]:
    """family id -> profile."""
    out: dict[str, dict] = {}
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            p = json.loads(path.read_text(encoding="utf-8"))
            _check(path.name, p)
        except (OSError, ValueError) as e:
            log.warning("[ZenTrainer] skipping profile %s: %s", path.name, e)
            continue
        p = {"id": path.stem, "summary": "", "sources": [], "train": {}, "samples": {}, "notes": [], **p}
        for fid in p["families"]:
            out[fid] = p
    return out


def for_family(fid: str) -> dict | None:
    global _cache
    if _cache is None:
        _cache = load()
    return _cache.get(fid)
