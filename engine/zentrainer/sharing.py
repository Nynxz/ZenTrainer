"""Run settings as a file to share: export a run's spec, import someone else's.

The file is the spec without what only means something on the machine it came from (the dataset
folder, where LoRAs are saved, a LoRA to continue from, edit samples' pictures), so importing it
fills in everything else and leaves those to choose. Model files stay by name: they're the same
files everywhere, and the import lists any this install doesn't have.

Import takes this file or a bare spec (a run folder's spec.json), lays it over the defaults the
form already has, and drops what this version doesn't know with a warning instead of refusing:
a file from a newer ZenTrainer still loads what it can.
"""

from __future__ import annotations

import copy
from typing import Any

from . import __version__
from .spec import SpecError, migrate

FORMAT = "zentrainer-run"
VERSION = 1

# this install's own: never exported, chosen again on import
LOCAL = {
    "dataset.folder": "the dataset",
    "output.lora_dir": "where LoRAs are saved",
    "network.init_from": "a LoRA to continue from",
}
LABELS = {**LOCAL, "samples.prompts.control": "the edit samples' pictures"}

# fields naming a model file, and the ComfyUI folder it's in (samples.model_name: by model.kind)
FILES = {
    "model.ckpt_name": "checkpoints",
    "model.unet_name": "diffusion_models",
    "model.clip_name": "text_encoders",
    "model.vae_name": "vae",
    "model.audio_vae_name": "vae",
    "samples.model_name": None,
    "samples.turbo_lora": "loras",
    "network.training_adapter": "loras",
    "network.base_lora": "loras",
}


class ShareError(ValueError):
    pass


def _get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _pop(d: dict, path: str) -> bool:
    *parents, last = path.split(".")
    for k in parents:
        d = d.get(k) if isinstance(d, dict) else None
        if not isinstance(d, dict):
            return False
    return d.pop(last, None) not in (None, "")


def export(spec: dict[str, Any]) -> dict[str, Any]:
    """A run's spec (a dict, as the API and spec.json have it) as a file to share."""
    s = copy.deepcopy(spec)
    left_out = [path for path in LOCAL if _pop(s, path)]
    for p in (s.get("samples") or {}).get("prompts") or []:
        if isinstance(p, dict) and p.pop("control", None):  # a picture in the sender's dataset
            left_out.append("samples.prompts.control")
    return {"format": FORMAT, "version": VERSION, "zentrainer": __version__, "name": s.get("name", ""),
            "left_out": sorted(set(left_out)), "spec": s}


def read(doc: Any) -> dict[str, Any]:
    """What an imported file brings: {spec (partial, to lay over the form), name, warnings,
    local (what to choose here), files (model files named, with whether they're installed)}."""
    if not isinstance(doc, dict):
        raise ShareError("not a ZenTrainer run file: expected a JSON object")
    warnings: list[str] = []
    if doc.get("format") == FORMAT:
        raw = doc.get("spec")
        if not isinstance(raw, dict):
            raise ShareError("the file has no settings (spec)")
        v = doc.get("version")
        if isinstance(v, int) and v > VERSION:
            warnings.append(f"made by a newer ZenTrainer ({doc.get('zentrainer', '?')}): settings it added are left out")
    elif "format" in doc:
        raise ShareError(f"not a ZenTrainer run file (format {doc.get('format')!r})")
    elif any(k in doc for k in ("model", "optim", "network", "dataset")):
        raw = doc  # a bare spec, e.g. a run folder's spec.json
    else:
        raise ShareError("not a ZenTrainer run file: no settings in it")
    try:
        spec, dropped = migrate(raw)  # an older layout renamed, what this version lacks left out
    except SpecError as e:
        raise ShareError(str(e)) from None
    if dropped:
        warnings.append("not in this version, left out: " + ", ".join(sorted(dropped)))
    here = export(spec)  # a bare spec still has the sender's dataset, folders, pictures
    spec = here["spec"]
    gone = {"dataset.folder", *here["left_out"], *[p for p in doc.get("left_out") or [] if p in LABELS]}
    return {"spec": spec, "name": spec.get("name") or doc.get("name") or "", "warnings": warnings,
            "local": [LABELS[p] for p in LABELS if p in gone], "files": files(spec)}


def files(spec: dict) -> list[dict]:
    """The model files a spec names: [{field, folder, name, installed}] (installed None when
    ComfyUI's folders can't be read, e.g. in tests)."""
    kind = _get(spec, "model.kind") or "checkpoint"
    out = []
    for path, folder in FILES.items():
        name = _get(spec, path)
        if not isinstance(name, str) or not name:
            continue
        folder = folder or ("diffusion_models" if kind == "split" else "checkpoints")
        out.append({"field": path, "folder": folder, "name": name, "installed": _installed(folder, name)})
    return out


def _installed(folder: str, name: str) -> bool | None:
    try:
        import folder_paths
    except ImportError:
        return None
    try:
        return folder_paths.get_full_path(folder, name) is not None
    except Exception:
        return False
