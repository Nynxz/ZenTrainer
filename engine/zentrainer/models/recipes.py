"""Simple mode's recipes, as data: one JSON file each in engine/catalog/recipes/, and what they share
(their order, caption prompts and photo guidance) in its _shared.json. The web app builds the same
files in (web/src/lib/recipes.ts); this reads them for the API (GET /zentrainer/api/recipes) and the
tests, which check every recipe against the spec."""

from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "catalog" / "recipes"
SHARED = "_shared.json"


def shared() -> dict:
    return json.loads((DIR / SHARED).read_text("utf-8"))


def files() -> list[Path]:
    return sorted(p for p in DIR.glob("*.json") if p.name != SHARED)


def resolve(r: dict, common: dict) -> dict:
    """A recipe as its file has it, with a named photo guidance and caption prompt written out."""
    r = dict(r)
    if isinstance(r.get("photos"), str):
        r["photos"] = common["photos"][r["photos"]]
    caption = dict(r.get("caption") or {})
    caption["prompt"] = common["captions"].get(caption.get("prompt"), caption.get("prompt"))
    r["caption"] = caption
    return r


def load() -> list[dict]:
    """Every recipe, resolved, in the order Simple mode lists them (_shared.json's, then by id)."""
    common = shared()
    order = common.get("order", [])
    out = [resolve(json.loads(p.read_text("utf-8")), common) for p in files()]
    return sorted(out, key=lambda r: (order.index(r["id"]) if r["id"] in order else len(order), r["id"]))
