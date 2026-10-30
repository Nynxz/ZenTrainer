"""Model families, downloads, training adapters, a model's noise schedule, and inspecting or
copying a .safetensors file."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from .common import _bad


def register(server) -> None:
    routes = server.routes

    # ---- model families + downloads ------------------------------------------------------------

    @routes.get("/zentrainer/api/families")
    async def list_families(request):
        """Model families from ComfyUI's templates, with what's installed of each."""
        from ...models import families

        return web.json_response({"families": await asyncio.to_thread(families.list_families)})

    @routes.get("/zentrainer/api/recipes")
    async def list_recipes(_request):
        """Simple mode's recipes (engine/catalog/recipes/), each a run's settings for a model."""
        from ...models import recipes

        return web.json_response({"recipes": await asyncio.to_thread(recipes.load)})

    @routes.post("/zentrainer/api/families/relist")
    async def relist_families(_request):
        """Re-list the model makers' Hugging Face repos now (new models, without waiting for the cache)."""
        from ...models import families

        try:
            return web.json_response(await asyncio.to_thread(families.relist))
        except Exception as e:
            return _bad(f"can't list Hugging Face: {e}", 502)

    @routes.get("/zentrainer/api/families/{fid}")
    async def get_family(request):
        """One family; ?repo=1 adds every build in its Hugging Face repo (cached a day,
        ?refresh=1 refetches)."""
        from ...models import families

        fam = await asyncio.to_thread(families.get, request.match_info["fid"],
                                      request.query.get("repo") == "1", request.query.get("refresh") == "1")
        return web.json_response(fam) if fam else _bad("no such family", 404)

    @routes.get("/zentrainer/api/downloads")
    async def list_downloads(request):
        from ...models import downloads, families

        targets = {}
        for folder in ("diffusion_models", "text_encoders", "vae", "checkpoints"):
            try:
                targets[folder] = str(downloads.target_dir(folder))
            except ValueError:
                pass
        return web.json_response({"downloads": downloads.manager().list(), "targets": targets,
                                  "hf_token": bool(families._hf_headers())})

    @routes.post("/zentrainer/api/downloads")
    async def start_download(request):
        """Download a file the family catalog knows: {family, folder, name}."""
        from ...models import downloads

        body = await request.json()
        try:
            job = await asyncio.to_thread(downloads.manager().start, str(body.get("family", "")),
                                          str(body.get("folder", "")), str(body.get("name", "")))
        except (ValueError, OSError) as e:
            return _bad(str(e))
        return web.json_response(job.public())

    @routes.get("/zentrainer/api/noise")
    async def noise_profile(request):
        """A model's own noise schedule from its file's header (noise.py): ?folder=&name=."""
        from ...models import noise

        folder = request.query.get("folder", "diffusion_models")
        if folder not in ("diffusion_models", "checkpoints"):
            return _bad("folder is diffusion_models or checkpoints")
        return web.json_response(await asyncio.to_thread(noise.profile, folder, request.query.get("name", "")))

    @routes.get("/zentrainer/api/noise/schedule")
    async def noise_schedule(request):
        """One ComfyUI scheduler's exact σ on a model: ?folder=&name=&scheduler=&steps=."""
        from ...models import noise

        folder = request.query.get("folder", "diffusion_models")
        if folder not in ("diffusion_models", "checkpoints"):
            return _bad("folder is diffusion_models or checkpoints")
        try:
            steps = int(request.query.get("steps", "20"))
        except ValueError:
            return _bad("steps is a number")
        sigmas = await asyncio.to_thread(noise.schedule, folder, request.query.get("name", ""),
                                         request.query.get("scheduler", "simple"), steps)
        return web.json_response({"sigmas": sigmas})

    @routes.get("/zentrainer/api/adapters")
    async def list_adapters(request):
        """Training adapters (or ?kind=base: base LoRAs) for the picker: the family's known ones
        first, custom, then local files."""
        from ...models import adapters

        kind = "base" if request.query.get("kind") == "base" else "training"
        rows = await asyncio.to_thread(adapters.listing, request.query.get("family") or None, kind)
        return web.json_response({"adapters": rows})

    @routes.post("/zentrainer/api/adapters")
    async def add_adapter(request):
        """Add one from a Hugging Face link: {link, family?, file?, kind?}. Returns {adapter} or {choices}."""
        from ...models import adapters

        body = await request.json()
        try:
            out = await asyncio.to_thread(adapters.add, str(body.get("link", "")), body.get("family") or None,
                                          body.get("file") or None, "base" if body.get("kind") == "base" else "training")
        except ValueError as e:
            return _bad(str(e))
        return web.json_response(out)

    @routes.post("/zentrainer/api/adapters/download")
    async def download_adapter(request):
        from ...models import adapters

        body = await request.json()
        try:
            job = await asyncio.to_thread(adapters.download, str(body.get("id", "")))
        except (ValueError, OSError) as e:
            return _bad(str(e))
        return web.json_response(job.public())

    @routes.delete("/zentrainer/api/adapters")
    async def remove_adapter(request):
        from ...models import adapters

        return web.json_response({"removed": adapters.remove(request.query.get("id", ""))})

    @routes.delete("/zentrainer/api/downloads/{job_id}")
    async def cancel_download(request):
        from ...models import downloads

        ok = downloads.manager().cancel(request.match_info["job_id"])
        return web.json_response({"cancelled": ok})

    @routes.post("/zentrainer/api/downloads/clear")
    async def clear_downloads(request):
        from ...models import downloads

        downloads.manager().clear()
        return web.json_response({"ok": True})

    # ---- inspecting a .safetensors file, and saving a copy --------------------------------------

    def _tensor_file(folder: str, name: str) -> str | None:
        import folder_paths

        from ...models import tensors

        if folder not in tensors.FOLDERS or not name.lower().endswith(".safetensors"):
            return None
        return folder_paths.get_full_path(folder, name)

    @routes.get("/zentrainer/api/tensors/files")
    async def tensor_files(_request):
        """The .safetensors files ComfyUI lists in each model folder."""
        import folder_paths

        from ...models import tensors

        def listing():
            out = {}
            for folder in tensors.FOLDERS:
                try:
                    names = folder_paths.get_filename_list(folder)
                except KeyError:
                    continue
                out[folder] = sorted(n.replace("\\", "/") for n in names if n.lower().endswith(".safetensors"))
            return out

        return web.json_response({"folders": await asyncio.to_thread(listing)})

    @routes.get("/zentrainer/api/tensors")
    async def inspect_tensors(request):
        """?folder=&name=: the file's tensors (name, dtype, shape), metadata and totals, from its header."""
        from ...models import tensors

        path = _tensor_file(request.query.get("folder", ""), request.query.get("name", ""))
        if not path:
            return _bad("not found", 404)
        try:
            return web.json_response(await asyncio.to_thread(tensors.inspect, path))
        except (OSError, ValueError) as e:
            return _bad(f"can't read it: {e}")

    @routes.post("/zentrainer/api/tensors/copy")
    async def copy_tensors(request):
        """Save a copy beside the file: {folder, name, out_name, metadata?, dtype?, drop?: [names or
        prefixes], dry_run?}. metadata replaces the file's; dry_run says what the copy would hold."""
        import os
        import re as _re

        from ...models import tensors

        b = await request.json()
        folder, name = str(b.get("folder", "")), str(b.get("name", "")).replace("\\", "/")
        path = _tensor_file(folder, name)
        if not path:
            return _bad("not found", 404)
        dtype = b.get("dtype") or None
        drop = [str(x) for x in b.get("drop") or []]
        meta = b.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            return _bad("metadata must be an object of strings")
        if b.get("dry_run"):
            return web.json_response(await asyncio.to_thread(tensors.plan, path, dtype, drop))
        stem = _re.sub(r"[^A-Za-z0-9_. ()-]+", "_", str(b.get("out_name") or "")).strip("._ ")
        if stem.lower().endswith(".safetensors"):
            stem = stem[: -len(".safetensors")]
        if not stem:
            return _bad("give the copy a name")
        out = Path(path).with_name(f"{stem}.safetensors")
        try:
            stats = await asyncio.to_thread(tensors.save_copy, path, out, meta, dtype, drop, bool(b.get("overwrite")))
        except FileExistsError as e:
            return _bad(str(e), 409)
        except (OSError, ValueError) as e:
            return _bad(f"can't save the copy: {e}")
        sub = os.path.dirname(name)
        return web.json_response({"folder": folder, "name": f"{sub}/{out.name}" if sub else out.name, **stats})
