"""This process and machine: health, themes, memory telemetry, disk use, ZenTrainer's and ComfyUI's
versions, restart, rebuilding the web app."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

from aiohttp import web

from ... import __version__, paths, state
from .. import system
from ...harness import control
from .common import WEB_DIST, _bad, _capabilities, _not_local, is_local, _code_stale, _comfy_version, _exec_self, _web_state, _zt_version, comfy_too_old


def register(server) -> None:
    routes = server.routes

    @routes.get("/zentrainer/api/health")
    async def health(_request):
        return web.json_response({"version": __version__, "capabilities": _capabilities(), "stale": _code_stale(),
                                  "comfy_old": comfy_too_old(), **_web_state()})

    def _revalidated(request, text: str, content_type: str):
        """`text` with an ETag the browser revalidates: it keeps its copy between loads (a 304 is a
        few hundred bytes) yet sees an edit at once."""
        import hashlib

        tag = f'"{hashlib.sha1(text.encode()).hexdigest()[:20]}"'
        headers = {"ETag": tag, "Cache-Control": "no-cache"}
        if request.headers.get("If-None-Match") == tag:
            return web.Response(status=304, headers=headers)
        return web.Response(text=text, content_type=content_type, headers=headers)

    @routes.get("/zentrainer/api/themes")
    async def themes(request):
        from ..themes import list_themes

        packs = await asyncio.to_thread(list_themes, None, False)
        return _revalidated(request, json.dumps({"themes": packs}), "application/json")

    @routes.get("/zentrainer/api/themes/{theme_id}/css")
    async def one_theme_css(request):
        """One pack's CSS (the list leaves it out: it's mostly fonts, and only the shown pack's is needed)."""
        from ..themes import theme_css

        css = await asyncio.to_thread(theme_css, request.match_info["theme_id"])
        if css is None:
            return web.Response(status=404, text="no CSS for that theme")
        return _revalidated(request, css, "text/css")

    @routes.get("/zentrainer/api/system/models")
    async def loaded_models(_request):
        """Each model ComfyUI holds: its size, how much is in VRAM, per block (see residency.py)."""
        from ...models import residency

        return web.json_response({"models": await asyncio.to_thread(residency.report)})

    @routes.get("/zentrainer/api/system")
    async def get_system(request):
        """Memory samples newer than ?since=<unix time> (all buffered ones by default)."""
        try:
            since = float(request.query.get("since", 0))
        except ValueError:
            since = 0.0
        return web.json_response({"interval": system.INTERVAL, "samples": system.history(since)})

    # ---- storage --------------------------------------------------------------------------------
    @routes.get("/zentrainer/api/storage")
    async def get_storage(request):
        """The full report; ?disks=1 for just free space (cheap: the sidebar polls it)."""
        from ...runs import disk as storage

        if request.query.get("disks") == "1":
            return web.json_response({"disks": await asyncio.to_thread(storage.disks)})
        return web.json_response(await asyncio.to_thread(storage.report, state.store()))

    @routes.get("/zentrainer/api/storage/places")
    async def storage_places(_request):
        """Where a run can save its LoRAs: ComfyUI's loras folders, with free space."""
        import folder_paths

        default = paths.lora_dir()
        base = Path(folder_paths.base_path).resolve()
        out = []
        for p in folder_paths.get_folder_paths("loras"):
            rp = Path(p).resolve()
            try:
                u = shutil.disk_usage(rp)
            except OSError:
                continue
            out.append({"path": str(rp), "free": u.free, "total": u.total, "default": rp == default.resolve(),
                        "inside_install": rp.is_relative_to(base)})
        return web.json_response({"loras": out})

    # what a cleanup's dialog listed, sent back with its Delete: nothing outside it is removed
    def _only_ids(b: dict) -> set | None:
        o = b.get("only")
        return None if o is None else {str(x) for x in o}

    def _only_folders(b: dict) -> set | None:
        o = b.get("only")
        return None if o is None else {None if x is None else str(x) for x in o}

    def _only_checkpoints(b: dict) -> set | None:
        o = b.get("only")
        return None if o is None else {(str(x["run_id"]), int(x["step"]), str(x["file"])) for x in o}

    @routes.post("/zentrainer/api/storage/clean")
    async def clean_storage(request):
        """{"kind": "states", "run_id"} | {"kind": "states_all"} | {"kind": "cache", "folder"}:
        returns {"freed": bytes} ("runs" too for states_all: how many it cleaned).
        {"kind": "cache_all", "only": [folders]}: those datasets' caches (a deleted dataset's too; null
        for caches that never recorded theirs). states_all takes "only": [run ids] the same way, and a
        trim that isn't a dry run "only": [{run_id, step, file}] from its dry run: nothing outside
        what was listed is ever removed.
        {"kind": "checkpoints", "run_id", "keep"?: 1, "dry_run"?: true} | {"kind": "checkpoints_all", ...}:
        trims finished runs' checkpoints to their best (runs/disk.py trim_checkpoints); a dry run
        (the default) only says what would go."""
        from ...datasets import folders as datasets
        from ...runs import disk as storage

        b = await request.json()
        try:
            if b.get("kind") == "states":
                freed = await asyncio.to_thread(storage.clean_states, state.store(), str(b.get("run_id", "")))
            elif b.get("kind") == "states_all":
                if b.get("only") is None:
                    return _bad("this deletes only the runs its dialog listed: send their ids as \"only\"")
                freed, runs = await asyncio.to_thread(storage.clean_all_states, state.store(), _only_ids(b))
                return web.json_response({"freed": freed, "runs": runs})
            elif b.get("kind") in ("checkpoints", "checkpoints_all"):
                keep, dry = int(b.get("keep", 1)), b.get("dry_run", True) is not False
                only = _only_checkpoints(b)
                if not dry and only is None:
                    return _bad("a trim deletes only what its dry run listed: send that list as \"only\"")
                if b["kind"] == "checkpoints":
                    out = await asyncio.to_thread(storage.trim_checkpoints, state.store(), str(b.get("run_id", "")), keep, dry, only)
                    return web.json_response({**out, "runs": [out] if out["remove"] else []})
                return web.json_response(await asyncio.to_thread(storage.trim_all_checkpoints, state.store(), keep, dry, only))
            elif b.get("kind") == "cache_all":
                if b.get("only") is None:
                    return _bad("this clears only the datasets its dialog listed: send their folders as \"only\"")
                freed = await asyncio.to_thread(storage.clean_all_caches, state.store(), _only_folders(b))
            elif b.get("kind") == "cache":
                freed = await asyncio.to_thread(storage.clean_cache, state.store(), str(b.get("folder", "")))
            else:
                return _bad("kind must be states, states_all, checkpoints, checkpoints_all, cache or cache_all")
        except (storage.StorageError, datasets.DatasetError, ValueError) as e:
            return _bad(str(e))
        except FileNotFoundError:
            return _bad("not found", 404)
        return web.json_response({"freed": freed})

    @routes.get("/zentrainer/api/settings")
    async def get_settings(_request):
        """ZenTrainer's own settings, and the dataset/output folders this process actually uses."""
        return web.json_response(await asyncio.to_thread(paths.describe))

    @routes.put("/zentrainer/api/settings")
    async def put_settings(request):
        """{"folder": absolute path, or "" for ComfyUI's input/output; "datasets": "" (as the folder
        decides), a folder inside ComfyUI's input ("datasets"), or an absolute path}: saved now, used
        from the next ComfyUI start (paths don't change under a run that's training)."""
        body = await request.json()
        if not isinstance(body, dict):
            return _bad("expected an object")
        # where ZenTrainer reads, writes and deletes: from this machine only (the rest from anywhere)
        if ("folder" in body or "datasets" in body) and not is_local(request):
            return _not_local("the ZenTrainer and datasets folders can be changed")
        try:
            await asyncio.to_thread(paths.save_settings, body)
        except paths.SettingsError as e:
            return _bad(str(e))
        out = await asyncio.to_thread(paths.describe)
        return web.json_response({**out, "message": "takes effect after restarting ComfyUI" if out["restart_needed"] else ""})

    @routes.get("/zentrainer/api/files/view")
    async def view_file(request):
        """ComfyUI's /view for ZenTrainer's files: the same ?filename=&subfolder=&type= address, but
        found in the ZenTrainer folder too (/view only knows ComfyUI's dirs). Ranges work, for videos."""
        q = request.query
        try:
            path = await asyncio.to_thread(paths.resolve_view, q.get("type", "output"), q.get("subfolder", ""),
                                           q.get("filename", ""))
        except ValueError as e:
            return _bad(str(e))
        if path is None:
            return _bad("not found", 404)
        return web.FileResponse(path)

    @routes.get("/zentrainer/api/comfyui")
    async def comfyui_version(request):
        """This ComfyUI's version and commit, the pin in comfy.lock, and (with ?check=1)
        how far upstream has moved. The check fetches, so it only runs when asked."""
        return web.json_response(await asyncio.to_thread(_comfy_version, request.query.get("check") == "1"))

    @routes.get("/zentrainer/api/version")
    async def zt_version(request):
        """ZenTrainer's version and commit, and (with ?check=1) the new commits upstream: what
        `zt update` would bring. The check fetches, so it only runs when asked."""
        return web.json_response(await asyncio.to_thread(_zt_version, request.query.get("check") == "1"))

    building = asyncio.Lock()

    @routes.post("/zentrainer/api/web/build")
    async def build_web(request):
        """Rebuild web/dist from the source on disk: after a pull, since dist isn't in git. This
        machine only: pnpm install runs packages' install scripts."""
        if not is_local(request):
            return _not_local("rebuilding the web app is")
        pnpm = shutil.which("pnpm")
        if not pnpm:
            return _bad("pnpm isn't installed here: run `pnpm install && pnpm build` in web/", 501)
        async with building:
            # CI: pnpm never stops to ask (it would wait on a prompt nobody sees)
            for args in (["install", "--frozen-lockfile"], ["build"]):
                proc = await asyncio.create_subprocess_exec(pnpm, *args, cwd=WEB_DIST.parent, env={**os.environ, "CI": "1"},
                                                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await proc.communicate()
                if proc.returncode:
                    return _bad(f"pnpm {args[0]} failed:\n{out.decode(errors='replace')[-2000:]}", 500)
        return web.json_response({"ok": True, **_web_state()})

    @routes.post("/zentrainer/api/restart")
    async def restart(request):
        """Restart this ComfyUI process (same interpreter and arguments)."""
        body = await request.json() if request.can_read_body else {}
        if control.active() and not body.get("force"):
            return _bad("a run is training; stop it first (or pass force)", 409)
        if control.active() and not is_local(request):
            return _not_local("a restart that stops a training run is")
        asyncio.get_running_loop().call_later(0.5, _exec_self)
        return web.json_response({"ok": True})
