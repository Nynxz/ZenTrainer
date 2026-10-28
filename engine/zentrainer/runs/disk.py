"""Disk use: what training and apps leave behind, and the room left where they write.

Per run: its LoRA checkpoints, its resume states (optimizer state saved beside every
checkpoint: often bigger than the LoRAs) and its samples. Per dataset: the images and the
latent / text-embedding caches training builds from them (in user/zentrainer/cache). Plus app outputs and free space on
each disk involved.

Cleaning is limited to what can't lose work: a finished run's resume states (only needed
to resume a stopped run) and dataset caches (rebuilt on the next run). Nothing while a run
is active: it may be writing either.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ACTIVE = ("queued", "loading", "training", "paused")
# resume states matter for these: they can still continue
RESUMABLE = ("stopped", "error", "interrupted", "rejected")


class StorageError(ValueError):
    pass


def size(path: Path) -> int:
    """Bytes under `path` (a file, or a folder walked without following links)."""
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total, stack = 0, [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _roots() -> dict[str, Path]:
    import folder_paths

    from .. import paths

    out = paths.output_root()
    user = Path(folder_paths.get_user_directory()) / "zentrainer"
    # app results come from ComfyUI's own save nodes, which only write inside its output dir
    return {"loras": out / "loras", "samples": out / "samples", "apps": paths.legacy_output_root() / "apps",
            "states": user / "states", "cache": user / "cache"}


def _caches(folder: str, include_unknown: bool = False) -> list[Path]:
    """ZenTrainer's training caches (latents and conditioning) built from one dataset. Only ever
    these: other trainers' folders inside a dataset (_latent_cache/, .thumbs/...) aren't ours."""
    from ..harness.data import cache_folder

    root = _roots()["cache"]
    if not root.is_dir():
        return []
    # a dataset cache has its dataset.pt: the other folders here (the shared text cache, model and
    # prompt listings, LoRA thumbnails) are no dataset's, and an "unknown" cache isn't one of them
    return [p for p in root.iterdir() if p.is_dir() and (p / "dataset.pt").is_file()
            and (cache_folder(p) == folder or (include_unknown and cache_folder(p) is None))]


def _cache_bytes(caches: list[Path]) -> int:
    """The caches and the captions' text encodings they use (each counted once: caches share them)."""
    from ..harness.data import text_files

    texts = {f for c in caches for f in text_files(c)}
    return sum(size(c) for c in caches) + sum(size(f) for f in texts)


def disks() -> list[dict]:
    """Free space on each disk ZenTrainer writes to, once per disk."""
    import folder_paths

    from .. import paths

    custom = paths.folder()  # the ZenTrainer folder's disk, if set, rather than ComfyUI's output/input
    places = [("Outputs", str(paths.output_root()) if custom else folder_paths.get_output_directory()),
              ("Datasets", str(paths.datasets_root())),
              ("Run state", folder_paths.get_user_directory())]
    try:
        from ..models.downloads import target_dir

        places.append(("Model downloads", str(target_dir("diffusion_models"))))
    except Exception:
        pass
    seen: dict[int, dict] = {}
    for label, p in places:
        try:
            dev = os.stat(p).st_dev
            if dev in seen:
                seen[dev]["labels"].append(label)
                continue
            u = shutil.disk_usage(p)
        except OSError:
            continue
        seen[dev] = {"labels": [label], "path": str(p), "total": u.total, "used": u.used, "free": u.free}
    return list(seen.values())


def report(store) -> dict:
    from .. import paths
    from ..datasets import folders as datasets

    roots = _roots()
    runs = []
    for meta in store.list():
        rid = meta["run_id"]
        ckpts = store.read(rid, "checkpoints")
        ck_bytes = sum(size(Path(c["path"])) for c in ckpts if c.get("path"))
        states = size(roots["states"] / rid) + sum(size(Path(c["path"]).with_suffix(".state.pt")) for c in ckpts if c.get("path"))
        runs.append({"run_id": rid, "name": meta.get("name", rid), "status": meta.get("status"), "created": meta.get("created"),
                     "checkpoints": ck_bytes, "checkpoint_count": len(ckpts), "states": states,
                     "samples": sum(size(r / "samples" / rid) for r in paths.output_roots())})
    runs.sort(key=lambda r: -(r["checkpoints"] + r["states"] + r["samples"]))
    sets = []
    for d in datasets.summary():
        cache = _cache_bytes(_caches(d["folder"]))
        sets.append({"folder": d["folder"], "count": d["images"], "images": size(datasets.root() / d["folder"]), "cache": cache})
    sets.sort(key=lambda d: -(d["images"] + d["cache"]))
    return {"disks": disks(), "runs": runs, "datasets": sets, "apps": size(roots["apps"]),
            "orphan_caches": _orphan_caches({d["folder"] for d in sets})}


def _orphan_caches(folders: set[str]) -> list[dict]:
    """Dataset caches no listed dataset owns (deleted or renamed, or from before caches recorded
    theirs): [{folder (None: unknown), cache}] by size. Only "Clear all caches" reaches them."""
    from ..harness.data import cache_folder

    root = _roots()["cache"]
    by: dict = {}
    for p in root.iterdir() if root.is_dir() else []:
        if p.is_dir() and (p / "dataset.pt").is_file() and cache_folder(p) not in folders:
            by.setdefault(cache_folder(p), []).append(p)
    return sorted(({"folder": f, "cache": _cache_bytes(ps)} for f, ps in by.items()), key=lambda d: -d["cache"])


def _no_active(store) -> None:
    if any(m.get("status") in ACTIVE for m in store.list()):
        raise StorageError("a run is active: clean up once it finishes")


def clean_states(store, run_id: str) -> int:
    """Delete a finished run's resume states. Returns the bytes freed."""
    _no_active(store)
    meta = store.meta(run_id)  # raises for an unknown run
    if meta.get("status") in RESUMABLE:
        raise StorageError(f"{meta.get('name', run_id)} is {meta.get('status')}: its states are what resuming it needs")
    freed = 0
    d = _roots()["states"] / run_id
    if d.is_dir():
        freed += size(d)
        shutil.rmtree(d)
    for c in store.read(run_id, "checkpoints"):
        legacy = Path(c["path"]).with_suffix(".state.pt") if c.get("path") else None
        if legacy and legacy.is_file():
            freed += legacy.stat().st_size
            legacy.unlink()
    return freed


def clean_all_states(store, only: set | None = None) -> tuple[int, int]:
    """Every finished run's resume states. Runs that can still be resumed keep theirs. `only`: the
    run ids that were listed; no other run's are touched. Returns (bytes freed, runs cleaned)."""
    _no_active(store)
    freed = runs = 0
    for meta in store.list():
        if meta.get("status") in RESUMABLE or (only is not None and meta["run_id"] not in only):
            continue
        n = clean_states(store, meta["run_id"])
        if n:
            freed += n
            runs += 1
    return freed, runs


def clean_all_caches(store, only: set | None = None) -> int:
    """Every dataset's training caches, a deleted dataset's too, and the captions' encodings only
    they used; the next run on a dataset rebuilds its own. The shared sample-prompt encodings and
    the LoRA thumbnails stay. `only`: the dataset folders that were listed (None in it: the caches
    that never recorded theirs); a cache built for any other since is left alone. Returns the bytes
    freed."""
    from ..harness.data import cache_folder, prune_text

    _no_active(store)
    root = _roots()["cache"]
    freed = 0
    for p in list(root.iterdir()) if root.is_dir() else []:
        if p.is_dir() and (p / "dataset.pt").is_file() and (only is None or cache_folder(p) in only):
            freed += size(p)
            shutil.rmtree(p)
    return freed + (prune_text(root, grace=0) if root.is_dir() else 0)


def trim_checkpoints(store, run_id: str, keep: int = 1, dry_run: bool = True, only: set | None = None) -> dict:
    """A finished run's scheduled checkpoints, trimmed by its own score (harness/keep.py best_rule:
    likeness, else validation loss, else the latest) to the best `keep`, with what Keep always
    leaves: the final LoRA, the newest save (a resume's), saves by hand or at a stop, and saves
    not scored yet. dry_run: only what it would remove. Returns {run_id, name, rule, remove:
    [{step, file, bytes, reason}], freed}; removed ones move to the run's "pruned" stream.
    `only`: the (run_id, step, file) a dry run showed; a real run removes nothing outside it, so
    what's deleted is never more than what was listed, whatever changed since (a score landing,
    a new save)."""
    from ..harness import keep as kp

    _no_active(store)
    meta = store.meta(run_id)  # raises for an unknown run
    if keep < 1:
        raise StorageError("keep at least one checkpoint")
    rows = store.read(run_id, "checkpoints")
    validation, likeness = store.read(run_id, "validation"), store.read(run_id, "likeness")
    rule = kp.best_rule(validation, likeness)
    gone = [(row, why) for row, why in kp.to_prune(rows, rule, keep, validation, likeness) if kp.own(row, run_id)
            and (only is None or (run_id, row["step"], Path(row["path"]).name) in only)]
    states = _roots()["states"] / run_id

    def nbytes(row) -> int:
        n = Path(row["path"]).stat().st_size if Path(row["path"]).is_file() else 0
        st = Path(row["state"]).resolve() if row.get("state") else None  # as keep.remove decides
        return n + (st.stat().st_size if st and st.is_file() and st.parent == states.resolve() else 0)

    plan = [{"step": row["step"], "file": Path(row["path"]).name, "bytes": nbytes(row), "reason": why} for row, why in gone]
    out = {"run_id": run_id, "name": meta.get("name", run_id), "rule": rule, "remove": plan,
           "freed": sum(r["bytes"] for r in plan)}
    if not dry_run and gone:
        at = max((c["step"] for c in rows), default=0)
        removed = kp.apply(store, run_id, gone, states, at)
        done = {row["step"] for row, _, _ in removed}
        out["remove"] = [r for r in plan if r["step"] in done]
        out["freed"] = sum(n for _, _, n in removed)
    return out


def trim_all_checkpoints(store, keep: int = 1, dry_run: bool = True, only: set | None = None) -> dict:
    """trim_checkpoints over every run with something to remove. {runs: [...], freed}. `only` as
    trim_checkpoints: a real run touches no run, and no checkpoint, the dry run didn't list."""
    _no_active(store)
    runs = []
    for meta in store.list():
        if only is not None and not any(o[0] == meta["run_id"] for o in only):
            continue
        r = trim_checkpoints(store, meta["run_id"], keep, dry_run, only)
        if r["remove"]:
            runs.append(r)
    return {"runs": runs, "freed": sum(r["freed"] for r in runs)}


def clean_cache(store, folder: str) -> int:
    """Delete the training caches built from a dataset (and any from before caches recorded their
    dataset); the next run rebuilds what it needs. Nothing inside the dataset folder is touched."""
    from ..datasets import folders as datasets

    _no_active(store)
    datasets.folder(folder)  # must be a real dataset
    from ..harness.data import prune_text

    freed = 0
    for d in _caches(folder, include_unknown=True):
        freed += size(d)
        shutil.rmtree(d)
    # and the captions' encodings only those used: another dataset's that share one stay
    return freed + prune_text(_roots()["cache"], grace=0)
