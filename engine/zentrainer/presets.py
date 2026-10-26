"""Presets: reusable bits of a run, by kind.

    run      a partial TrainSpec merged onto the form (they layer: one, then another on top)
    prompts  a sample-prompt set: {prompts, negative, ...any SampleSpec fields}

The user's live in <user>/zentrainer/presets/<kind>/. Built-ins would ship in
engine/presets/<kind>/ and lose to the user's on an id clash; none ship now (the run form's
starting points are the recipes, web/src/lib/recipes.ts).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import safeio
from .spec import slugify

KINDS = ("run", "prompts")
BUILTIN = Path(__file__).resolve().parent.parent / "presets"


class PresetError(ValueError):
    pass


def _check(kind: str) -> None:
    if kind not in KINDS:
        raise PresetError(f"unknown preset kind {kind!r}; have {', '.join(KINDS)}")


def _load(path: Path, builtin: bool) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get("data"), dict):
        return None
    data = d["data"]
    if path.parent.name == "run":  # settings saved by an older version: in this one's layout
        from .spec import SpecError, migrate

        try:
            data = migrate(data)[0]
        except SpecError:
            return None
    return {"id": path.stem, "name": d.get("name", path.stem), "description": d.get("description", ""),
            "data": data, "builtin": builtin}


def list_presets(kind: str, user_root: Path) -> list[dict]:
    _check(kind)
    found: dict[str, dict] = {}
    for root, builtin in ((BUILTIN / kind, True), (user_root / kind, False)):
        if root.is_dir():
            for p in sorted(root.glob("*.json")):
                if (row := _load(p, builtin)) is not None:
                    found[row["id"]] = row
    return sorted(found.values(), key=lambda r: (not r["builtin"], r["name"].lower()))


def save(kind: str, name: str, data: dict, user_root: Path, description: str = "") -> dict:
    _check(kind)
    if not name.strip():
        raise PresetError("a preset needs a name")
    if not isinstance(data, dict) or not data:
        raise PresetError("a preset needs data")
    if kind == "prompts" and not isinstance(data.get("prompts"), list):
        raise PresetError("a prompt set needs a 'prompts' list")
    pid = slugify(name)
    d = user_root / kind
    d.mkdir(parents=True, exist_ok=True)
    safeio.write_json(d / f"{pid}.json", {"name": name.strip(), "description": description, "data": data})
    return {"id": pid, "name": name.strip(), "description": description, "data": data, "builtin": False}


def delete(kind: str, pid: str, user_root: Path) -> bool:
    _check(kind)
    if slugify(pid) != pid:
        raise PresetError("bad preset id")
    p = user_root / kind / f"{pid}.json"
    if not p.is_file():
        return False
    p.unlink()
    return True
