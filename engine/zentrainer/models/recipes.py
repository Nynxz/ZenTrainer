"""Simple mode's recipes, as data: one JSON file each in engine/catalog/recipes/, and what they share
(their order, caption prompts and photo guidance) in its _shared.json. Served resolved at
GET /zentrainer/api/recipes, which the web app reads when it starts, so a changed or added recipe
shows on a reload, no build; the tests check every one against the spec.

Only that folder's *.json files are read (no path comes from a request). A file that doesn't load
(bad JSON, an id that isn't its name, a shared name that doesn't exist) is left out with a warning
rather than taking the others with it, and a card picture must be a path inside the web app."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("zentrainer")
DIR = Path(__file__).resolve().parents[2] / "catalog" / "recipes"
SHARED = "_shared.json"
# a card picture: a file in the web app (web/public), never another site ("//host/x.png") or a scheme
_ASSET = re.compile(r"^(?!.*\.\.)[A-Za-z0-9_][\w./-]*$")


def shared() -> dict:
    return json.loads((DIR / SHARED).read_text("utf-8"))


def files() -> list[Path]:
    return sorted(p for p in DIR.glob("*.json") if p.name != SHARED and p.is_file())


def resolve(r: dict, common: dict) -> dict:
    """A recipe as its file has it, with a named photo guidance and caption prompt written out.
    ValueError for a shared name that doesn't exist or a card picture outside the web app."""
    r = dict(r)
    if isinstance(r.get("photos"), str):
        if r["photos"] not in common["photos"]:
            raise ValueError(f"photos {r['photos']!r} isn't in {SHARED}")
        r["photos"] = common["photos"][r["photos"]]
    caption = dict(r.get("caption") or {})
    caption["prompt"] = common["captions"].get(caption.get("prompt"), caption.get("prompt"))
    r["caption"] = caption
    thumb = r.get("thumb")
    if thumb is not None:
        if not isinstance(thumb, dict) or not all(
                isinstance(thumb.get(k), str) and _ASSET.match(thumb[k]) for k in ("src", "after") if k in thumb):
            raise ValueError("thumb's pictures must be paths in the web app (web/public)")
    return r


def load() -> list[dict]:
    """Every recipe that loads, resolved, in the order Simple mode lists them (_shared.json's, then by id)."""
    common = shared()
    order = common.get("order", [])
    out = []
    for p in files():
        try:
            r = json.loads(p.read_text("utf-8"))
            if not isinstance(r, dict) or r.get("id") != p.stem:
                raise ValueError(f"its id must be its file's name, {p.stem!r}")
            out.append(resolve(r, common))
        except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
            log.warning("[ZenTrainer] recipe %s left out: %s", p.name, e)
    return sorted(out, key=lambda r: (order.index(r["id"]) if r["id"] in order else len(order), r["id"]))
