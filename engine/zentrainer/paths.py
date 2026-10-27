"""Where ZenTrainer keeps datasets and results.

By default in ComfyUI's own folders: datasets in its input dir, LoRAs / samples / merges under
output/zentrainer/. The "ZenTrainer folder" setting moves both out of them, to
<folder>/datasets/<dataset> and <folder>/output/<loras|samples|...>. The "datasets" setting puts
datasets somewhere of their own instead: a folder inside ComfyUI's input (a relative path, e.g.
"datasets" for input/datasets/, so they stop sharing input/ with everything else uploaded), or
any absolute path. Both are read once per process, so a change takes effect when ComfyUI
restarts, never under a training run; nothing is moved. Caches and run state stay in ComfyUI's
user dir either way.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import safeio

DEFAULT_SETTINGS = {"folder": "", "datasets": "", "face_model": "buffalo_l"}

_UNREAD = object()
_active = _UNREAD  # the folder this process runs with (None: ComfyUI's folders)
_active_datasets = _UNREAD  # the datasets folder it runs with (None: the folder's, or ComfyUI's input)


class SettingsError(ValueError):
    pass


def settings_file() -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "settings.json"


def read_settings() -> dict:
    try:
        saved = safeio.read_json(settings_file(), {})
    except OSError:
        saved = {}
    saved = saved if isinstance(saved, dict) else {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def folder() -> Path | None:
    """The ZenTrainer folder this process uses, or None for ComfyUI's input/output layout."""
    global _active
    if _active is _UNREAD:
        try:
            raw = str(read_settings()["folder"] or "").strip()
        except (ImportError, AttributeError):  # outside ComfyUI (tests, tools): don't pin anything
            return None
        _active = Path(raw).expanduser() if raw and os.path.isabs(os.path.expanduser(raw)) else None
    return _active


def reset() -> None:
    """Forget the folders read at start (tests)."""
    global _active, _active_datasets
    _active = _active_datasets = _UNREAD


def _datasets_path(raw: str) -> Path | None:
    """The "datasets" setting as a folder: relative = inside ComfyUI's input, absolute = itself."""
    import folder_paths

    raw = str(raw or "").strip()
    if not raw:
        return None
    p = Path(os.path.expanduser(raw))
    return p if p.is_absolute() else Path(folder_paths.get_input_directory()) / p


def datasets_folder() -> Path | None:
    """The datasets folder this process uses, if the setting names one."""
    global _active_datasets
    if _active_datasets is _UNREAD:
        try:
            _active_datasets = _datasets_path(read_settings()["datasets"])
        except (ImportError, AttributeError):  # outside ComfyUI (tests, tools)
            return None
    return _active_datasets


def datasets_root() -> Path:
    import folder_paths

    if (own := datasets_folder()) is not None:
        return own
    d = folder()
    return d / "datasets" if d else Path(folder_paths.get_input_directory())


def dataset_roots() -> list[Path]:
    """Where a dataset can be, the one in use first: runs and samples from before a datasets or
    ZenTrainer folder was set still find their dataset where it was. Listing uses datasets_root()
    only."""
    import folder_paths

    out = [datasets_root()]
    d = folder()
    for r in ([d / "datasets"] if d else []) + [Path(folder_paths.get_input_directory())]:
        if all(os.path.abspath(r) != os.path.abspath(o) for o in out):
            out.append(r)
    return out


def find_dataset(name: str) -> Path:
    """A dataset folder by name: in the datasets folder in use, else where an earlier setting kept
    it (datasets_root() / name, missing, when it's nowhere)."""
    for r in dataset_roots():
        if (r / name).is_dir():
            return r / name
    return datasets_root() / name


def sample_input(dataset: str, control: str) -> Path | None:
    """Where an edit sample's input picture is, as training finds it: in the run's dataset, in
    ComfyUI's input folder, or the path as given (one on another drive, say). The first that's a
    file; None when none is."""
    import folder_paths

    for p in (find_dataset(dataset) / control, Path(folder_paths.get_input_directory()) / control, Path(control)):
        if p.is_file():
            return p
    return None


def _within(p: Path, roots: list[Path]) -> bool:
    p = Path(os.path.abspath(p))
    return any(p == Path(os.path.abspath(r)) or Path(os.path.abspath(r)) in p.parents for r in roots)


def outside_folders(spec: dict, tries: list | None = None) -> list[str]:
    """The paths a run's settings (and test samples' `tries`) name outside ZenTrainer's folders:
    its datasets, a dataset folder in them, ComfyUI's input. Training reads those from anywhere
    (a folder or picture on another drive), so the routes take them only from this machine."""
    import folder_paths

    roots = dataset_roots()
    inp = Path(folder_paths.get_input_directory())
    out = []
    ds = spec.get("dataset") or {}
    for field, name in (("dataset.folder", ds.get("folder")), ("validation.dataset", (spec.get("validation") or {}).get("dataset"))):
        if name and not _within(find_dataset(str(name)), roots):
            out.append(f"{field} {name!r}")
    folder = find_dataset(str(ds.get("folder") or ""))
    prompts = (spec.get("samples") or {}).get("prompts") or []
    for i, p in enumerate([*prompts, *(tries or [])]):
        c = p.get("control") if isinstance(p, dict) else None
        if not c:
            continue
        found = sample_input(str(ds.get("folder") or ""), str(c))
        # missing: judge the path it names (a relative one is the dataset's, unless it climbs out)
        where = found if found is not None else (Path(str(c)) if os.path.isabs(str(c)) else folder / str(c))
        if not _within(where, [folder, inp, *roots]):
            out.append(f"{'a test sample' if i >= len(prompts) else f'sample {i + 1}'}'s input {str(c)!r}")
    return out


def legacy_output_root() -> Path:
    """output/zentrainer in ComfyUI's output dir: where results went before a folder was set."""
    import folder_paths

    return Path(folder_paths.get_output_directory()) / "zentrainer"


def output_root() -> Path:
    """What output/zentrainer is by default: loras/, samples/, apps/ live under it."""
    d = folder()
    return d / "output" if d else legacy_output_root()


def output_roots() -> list[Path]:
    """Where ZenTrainer results can be, newest layout first: a folder set later leaves older runs'
    samples and LoRAs where they were written."""
    d = folder()
    return [output_root(), legacy_output_root()] if d else [output_root()]


def lora_dir() -> Path:
    return output_root() / "loras"


def ensure() -> None:
    """At startup: make the folder's dirs, so a first run or upload has somewhere to go."""
    own, ds = folder(), datasets_folder()  # both read now: what this process uses until a restart
    if own or ds:
        for p in (datasets_root(), lora_dir()):
            p.mkdir(parents=True, exist_ok=True)


# ---- the setting ----------------------------------------------------------------------------------

def _too_broad(p: Path) -> str | None:
    """Why a folder is too broad to hand to ZenTrainer (whose dataset and file routes then read,
    write and delete in it): a drive or the filesystem's root, a home folder, or above one."""
    rp = Path(os.path.abspath(p))
    if rp == Path(rp.anchor):
        return f"{rp} is a whole drive"
    home = Path(os.path.abspath(Path.home()))
    if rp == home or rp in home.parents:
        return f"{rp} is {'your home folder' if rp == home else 'above your home folder'}"
    return None


def validate_folder(raw: str) -> str:
    """The folder as it will be saved ("" for ComfyUI's folders); SettingsError if it can't work."""
    import folder_paths

    raw = str(raw or "").strip()
    if not raw:
        return ""
    p = Path(os.path.normpath(os.path.expanduser(raw)))
    if not p.is_absolute():
        raise SettingsError("the ZenTrainer folder must be an absolute path")
    if why := _too_broad(p):
        raise SettingsError(f"{why}: pick a folder of its own for ZenTrainer, e.g. {Path.home() / 'ZenTrainer'}")
    if p.exists() and not p.is_dir():
        raise SettingsError(f"{p} exists and isn't a folder")
    rp = p.resolve()
    for label, comfy in (("input", folder_paths.get_input_directory()), ("output", folder_paths.get_output_directory())):
        c = Path(comfy).resolve()
        # inside ComfyUI's input/output is the bloat this setting is for; and <folder>/output
        # being ComfyUI's own output dir would mix the two layouts
        if rp == c or c in rp.parents or c in (rp / "datasets", rp / "output"):
            raise SettingsError(f"pick a folder outside ComfyUI's {label} folder ({c})")
    try:
        p.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=p):
            pass
    except OSError as e:
        raise SettingsError(f"can't write to {p}: {e.strerror or e}") from None
    return str(p)


def validate_datasets(raw: str) -> str:
    """The datasets setting as it will be saved: "" (the default), a folder inside ComfyUI's input
    given relative to it (e.g. "datasets"), or an absolute path. SettingsError if it can't work."""
    import folder_paths

    raw = str(raw or "").strip()
    if not raw:
        return ""
    absolute = os.path.isabs(os.path.expanduser(raw))
    if not absolute:
        raw = raw.replace("\\", "/").strip("/")
    p = _datasets_path(raw)
    inp = Path(folder_paths.get_input_directory()).resolve()
    if not absolute:
        if inp not in p.resolve().parents:
            raise SettingsError("a relative datasets folder has to be inside ComfyUI's input folder, e.g. datasets")
    elif p.resolve() == inp:
        return ""  # ComfyUI's input itself: the default
    if why := _too_broad(p):
        raise SettingsError(f"{why}: pick a folder that holds your datasets, e.g. {Path.home() / 'datasets'}")
    if p.exists() and not p.is_dir():
        raise SettingsError(f"{p} exists and isn't a folder")
    try:
        p.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=p):
            pass
    except OSError as e:
        raise SettingsError(f"can't write to {p}: {e.strerror or e}") from None
    return str(Path(os.path.normpath(os.path.expanduser(raw)))) if absolute else raw


def _datasets_left(new: Path) -> list[str]:
    """Datasets (folders with pictures or clips at the top) in the places in use before, not in `new`:
    what a user switching folders would have to move themselves."""
    from .harness.data import MEDIA_EXTS

    out = []
    for r in dataset_roots():
        if os.path.abspath(r) == os.path.abspath(new) or not r.is_dir():
            continue
        for d in sorted(r.iterdir()):
            if d.is_dir() and not d.name.startswith((".", "_")) and os.path.abspath(d) != os.path.abspath(new) \
                    and not (new / d.name).exists():
                try:
                    if any(f.suffix.lower() in MEDIA_EXTS for f in d.iterdir()):
                        out.append(str(d))
                except OSError:
                    pass
    return out


def save_settings(changes: dict) -> dict:
    cur = read_settings()
    if "folder" in changes:
        cur["folder"] = validate_folder(changes["folder"])
    if "datasets" in changes:
        cur["datasets"] = validate_datasets(changes["datasets"])
    if "face_model" in changes:
        from .datasets import faces

        if changes["face_model"] not in faces.MODELS:
            raise SettingsError(f"face_model is one of {', '.join(faces.MODELS)}")
        cur["face_model"] = changes["face_model"]
    f = settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    safeio.write_json(f, cur)
    return cur


def describe() -> dict:
    """The saved settings, what this process actually uses, whether a restart is pending (the
    folder only: the face model applies at the next face check), and the face models on offer."""
    from .analysis import faces

    saved = read_settings()
    active = folder()
    active_ds = datasets_folder()
    wanted = _datasets_path(saved["datasets"])
    return {"settings": saved, "active": {"folder": str(active) if active else "", "datasets": str(datasets_root()),
                                          "output": str(output_root()), "loras": str(lora_dir())},
            "restart_needed": saved["folder"] != (str(active) if active else "")
                              or (os.path.abspath(wanted) if wanted else "") != (os.path.abspath(active_ds) if active_ds else ""),
            # datasets the saved datasets folder won't list, still where they were (moved by hand)
            "datasets_elsewhere": _datasets_left(wanted) if wanted else [],
            "face_models": [{"id": k, "label": label, "note": note, "downloaded": faces.downloaded(k)}
                            for k, (label, note, *_) in faces.MODELS.items()]}


# ---- files by ComfyUI /view address ----------------------------------------------------------------

def inside(root: Path, rel: str) -> Path:
    """root/rel, refused if it climbs out of root. Lexical (like ComfyUI's /view), so a symlinked
    folder inside the root still serves."""
    root = Path(os.path.abspath(root))
    p = Path(os.path.abspath(root / rel))
    if p != root and root not in p.parents:
        raise ValueError(f"path {rel!r} is outside its folder")
    return p


def resolve_view(kind: str, subfolder: str, filename: str) -> Path | None:
    """The file a ComfyUI-style address names ({filename, subfolder, type} — how sample rows,
    dataset items and app results carry their files), wherever it lives now: type=input subfolders
    start with the dataset, type=output "zentrainer/..." is under output_roots(). None if it
    isn't there; ValueError for an address that escapes its folder."""
    import folder_paths

    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise ValueError(f"bad file name {filename!r}")
    sub = (subfolder or "").replace("\\", "/").strip("/")
    # never a hidden folder or file (.ssh, .config, .env): nothing ZenTrainer shows lives in one
    if filename.startswith(".") or any(part.startswith(".") for part in sub.split("/") if part):
        raise ValueError(f"{'/'.join([sub, filename]).lstrip('/')!r} is hidden")
    if kind == "input":
        from .harness.data import CONTROL_EXTS

        # the datasets folder can be anywhere (a setting): only its pictures, clips, sounds and
        # masks are served from it, not whatever else is beside them
        if Path(filename).suffix.lower() not in CONTROL_EXTS:
            raise ValueError(f"{filename!r} isn't a picture, video or sound")
        cands = [(r, sub) for r in dataset_roots()]
    elif kind == "output":
        cands = []
        if sub == "zentrainer" or sub.startswith("zentrainer/"):
            rest = sub[len("zentrainer"):].lstrip("/")
            cands = [(r, rest) for r in output_roots()]
        cands.append((Path(folder_paths.get_output_directory()), sub))
    elif kind == "temp":
        cands = [(Path(folder_paths.get_temp_directory()), sub)]
    else:
        raise ValueError(f"bad file type {kind!r}")
    for root, rel in cands:
        p = inside(root, os.path.join(rel, filename))
        if p.is_file():
            return p
    return None
