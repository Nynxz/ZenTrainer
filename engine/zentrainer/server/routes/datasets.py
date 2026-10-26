"""Datasets: folders, captions (and auto-captioning), masks (SAM3), faces, stats, uploads."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

from ...datasets import folders as datasets
from .common import _bad, _ds, _loopback


def register(server) -> None:
    routes = server.routes

    @routes.get("/zentrainer/api/datasets")
    async def list_datasets(_request):
        return web.json_response({"datasets": await asyncio.to_thread(datasets.summary)})

    @routes.post("/zentrainer/api/datasets")
    async def create_dataset(request):
        body = await request.json()
        return await _ds(lambda: datasets.create(str(body.get("name", "")).strip()) or {"ok": True})

    @routes.get("/zentrainer/api/datasets/{name}")
    async def get_dataset(request):
        return await _ds(lambda: datasets.items(request.match_info["name"]))

    @routes.get("/zentrainer/api/datasets/{name}/thumb")
    async def thumb(request):
        import folder_paths

        name, rel = request.match_info["name"], request.query.get("path", "")
        cache = Path(folder_paths.get_user_directory()) / "zentrainer" / "thumbs"
        try:
            path = await asyncio.to_thread(datasets.thumbnail, name, rel, int(request.query.get("size", 384)), cache)
        except FileNotFoundError:
            return _bad("not found", 404)
        except (datasets.DatasetError, ValueError) as e:
            return _bad(str(e))
        return web.FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @routes.put("/zentrainer/api/datasets/{name}/caption")
    async def put_caption(request):
        body = await request.json()
        name = request.match_info["name"]
        return await _ds(lambda: datasets.set_caption(name, body["path"], str(body.get("caption", "")),
                                                      set_name=str(body.get("set") or "default")) or {"ok": True})

    @routes.post("/zentrainer/api/datasets/{name}/bulk")
    async def bulk_captions(request):
        b = await request.json()
        name = request.match_info["name"]
        return await _ds(lambda: {"changed": datasets.bulk(name, b.get("op", ""), b.get("text", ""), b.get("find", ""),
                                                           set_name=str(b.get("set") or "default"))})

    @routes.put("/zentrainer/api/datasets/{name}/muted")
    async def dataset_muted(request):
        """{"set", "paths", "muted": bool}: mute or unmute pictures in a caption set."""
        b = await request.json()
        try:
            return web.json_response({"muted": datasets.set_muted(request.match_info["name"], str(b.get("set") or "default"),
                                                                  [str(p) for p in b.get("paths") or []], bool(b.get("muted", True)))})
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        except datasets.DatasetError as e:
            return _bad(str(e))

    @routes.put("/zentrainer/api/datasets/{name}/settings")
    async def dataset_settings(request):
        """{"faces": bool}: this dataset's settings (faces: face report on open, in the stats, likeness)."""
        try:
            return web.json_response(datasets.put_settings(request.match_info["name"], await request.json()))
        except FileNotFoundError:
            return _bad("dataset not found", 404)

    @routes.get("/zentrainer/api/datasets/{name}/stats")
    async def dataset_stats(request):
        """The dataset's health now and over time: {"current": {metrics, scores}, "history"}; a
        snapshot is kept whenever the dataset changed since the last look."""
        from ...datasets import stats as dataset_stats

        try:
            return web.json_response(await asyncio.to_thread(dataset_stats.snapshot, request.match_info["name"],
                                                             request.query.get("set") or "default"))
        except FileNotFoundError:
            return _bad("dataset not found", 404)

    @routes.get("/zentrainer/api/datasets/{name}/analysis")
    async def dataset_analysis(request):
        """The detail behind the health scores (datasets/analysis.py): buckets at ?resolution= (default
        1024), caption phrase frequencies, per-picture values and flags, near-duplicate pairs."""
        from ...datasets import analysis

        res = request.query.get("resolution", "1024")
        if not res.isdigit() or not 256 <= int(res) <= 4096:
            return _bad("resolution is a number of pixels, 256 to 4096")
        try:
            return web.json_response(await asyncio.to_thread(analysis.analyze, request.match_info["name"],
                                                             request.query.get("set") or "default", int(res)))
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        except datasets.DatasetError as e:
            return _bad(str(e))

    @routes.get("/zentrainer/api/datasets/{name}/faces")
    async def dataset_faces(request):
        """The dataset's faces: per picture, faces found, the main face's size, how much it looks
        like the rest, outliers. Cached per file; the first look takes a while (and downloads the
        face model the first time). available: false, with a hint, when the model can't download."""
        from ... import events
        from ...datasets import faces

        name = request.match_info["name"]
        last = 0.0

        def progress(done: int, total: int) -> None:  # at most 4 a second, and the last one
            nonlocal last
            if done == total or time.monotonic() - last >= 0.25:
                last = time.monotonic()
                events.emit(events.FACES, {"dataset": name, "done": done, "total": total})

        try:
            out = await asyncio.to_thread(faces.dataset_faces, name, progress)
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        except RuntimeError as e:  # the face model didn't download
            return web.json_response({"available": False, "hint": str(e)})
        return web.json_response({"available": True, **out})

    @routes.post("/zentrainer/api/datasets/{name}/split")
    async def split_videos(request):
        """Cut long videos into short clips: {seconds?, count? (per video, 0 = all), longer_than?}."""
        name = request.match_info["name"]
        b = await request.json() if request.can_read_body else {}
        return await _ds(lambda: datasets.split_videos(name, float(b.get("seconds") or 3.0), int(b.get("count", 20)),
                                                       float(b.get("longer_than") or 20.0)))

    @routes.get("/zentrainer/api/caption")
    async def caption_info(request):
        """Styles, models, and the recent jobs (?dataset= for one dataset's)."""
        from ...datasets import captioner

        if request.query.get("active") == "1":
            return web.json_response({"jobs": captioner.active()})
        if request.query.get("all") == "1":
            return web.json_response({"jobs": captioner.history()})
        return web.json_response({
            "styles": [{"id": k, "label": v.label, "prompt": v.prompt, "format": v.format, "max_length": v.max_length}
                       for k, v in captioner.STYLES.items()],
            "formats": list(captioner.structured.FORMATS),
            # what goes around a style's prompt, so the panel can show exactly what the model is told
            "notes": {"video": captioner.VIDEO_NOTE, "sound": captioner.SOUND_NOTE, "subject": captioner.SUBJECT_NOTE,
                      "mask": captioner.MASK_NOTE},
            "models": await asyncio.to_thread(captioner.models),
            "hearing": await asyncio.to_thread(captioner.hearing_models),
            "senses": await asyncio.to_thread(captioner.model_senses),
            "loras": await asyncio.to_thread(captioner.te_loras),
            "lora_catalog": await asyncio.to_thread(captioner.te_lora_catalog),
            "jobs": captioner.recent(request.query.get("dataset") or None),
        })

    @routes.post("/zentrainer/api/caption/loras")
    async def add_caption_lora(request):
        """{"link", "file"?}: download a text-encoder LoRA for captioning from Hugging Face."""
        from ...datasets import captioner

        b = await request.json()
        try:
            if b.get("id"):  # a catalog entry
                return web.json_response(await asyncio.to_thread(captioner.download_te_lora, str(b["id"])))
            return web.json_response(await asyncio.to_thread(captioner.add_te_lora, str(b.get("link", "")), b.get("file")))
        except (ValueError, OSError) as e:
            return _bad(str(e))

    @routes.post("/zentrainer/api/datasets/{name}/caption")
    async def caption_dataset(request):
        """{"clip_name", "clip_type", "set", "style" | "prompt" (+ "format"), "mode": missing|all|selected,
        "paths" (selected), "subject", "preview": n, "max_length", "replace_default"}.
        Rewriting captions that already exist in the "default" set (the one training reads
        unless told otherwise) needs replace_default: true, so it never happens by accident."""
        from ...datasets import captioner
        from ...harness.data import caption_path, scan

        name, b = request.match_info["name"], await request.json()
        try:
            root = datasets.folder(name)
            set_name = datasets._set(str(b.get("set") or "default"))
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        except datasets.DatasetError as e:
            return _bad(str(e))
        if not b.get("clip_name"):
            return _bad("pick a vision-language text encoder")
        mode = str(b.get("mode") or ("all" if b.get("overwrite") else "missing"))
        if mode not in captioner.MODES:
            return _bad(f"mode is one of {', '.join(captioner.MODES)}")
        paths = [str(p) for p in b.get("paths") or []]
        if mode == "selected" and not paths:
            return _bad("select some images first")
        preview = max(0, min(int(b.get("preview") or 0), 8))
        style = captioner.STYLES.get(str(b.get("style")), captioner.STYLES["nl"])
        prompt = str(b.get("prompt") or style.prompt)
        fmt = str(b.get("format") or (style.format if not b.get("prompt") else "text"))
        if fmt not in captioner.structured.FORMATS:
            return _bad(f"format is one of {', '.join(captioner.structured.FORMATS)}")
        job = captioner.CaptionJob(job_id="", dataset=name, set_name=set_name, clip_name=str(b["clip_name"]),
                                   clip_type=str(b.get("clip_type") or "auto"), prompt=prompt, mode=mode, paths=paths,
                                   subject=str(b.get("subject") or "").strip(), preview=preview)
        if not preview and set_name == "default" and not b.get("replace_default"):
            items = await asyncio.to_thread(lambda: captioner.todo(job, scan(root, warn=False)))
            existing = sum(caption_path(it.path, "default").is_file() for it in items)
            if existing:
                return _bad(f"this would rewrite {existing} existing caption{'s' if existing != 1 else ''} in the "
                            "default set", 409, replaces=existing)
        length = b.get("max_length")
        job = captioner.create(**{f: getattr(job, f) for f in ("dataset", "set_name", "clip_name", "clip_type", "prompt",
                                                               "mode", "paths", "subject", "preview")},
                               # 0 = no limit; unset = the style's own
                               max_length=_length(length) if length is not None else (style.max_length if not b.get("prompt") else 0),
                               seed=int(b["seed"]) if b.get("seed") is not None else None,
                               **_sampling(b),
                               format=fmt, hide_masked=bool(b.get("hide_masked")),
                               loras=[{"name": str(lo.get("name")), "strength": float(lo.get("strength", 1.0))}
                                      for lo in (b.get("loras") or []) if lo.get("name")])
        return await _submit_caption(job, f"caption {name}", b.get("client_id"))

    def _sampling(b: dict) -> dict:
        """A caption request's sampling settings, clamped to what makes sense (unset: the job's defaults)."""
        out = {}
        for key, cast, lo, hi in (("temperature", float, 0.05, 2.0), ("top_p", float, 0.05, 1.0), ("top_k", int, 0, 500),
                                  ("repetition_penalty", float, 1.0, 2.0)):
            if b.get(key) is not None:
                out[key] = min(max(cast(b[key]), lo), hi)
        return out

    def _length(v) -> int:
        """A caption's token limit from a request: 0 (no limit) or 16 to captioner.NO_LIMIT."""
        from ...datasets import captioner

        n = int(v)
        return 0 if n <= 0 else max(16, min(n, captioner.NO_LIMIT))

    async def _submit_caption(job, title: str, client_id):
        """Queue a caption job as a one-node ComfyUI prompt, so it takes its turn on the GPU."""
        from ...datasets import captioner
        from .. import jobqueue

        prompt_id = str(uuid.uuid4())
        busy = jobqueue.get().current
        captioner.update(job, prompt_id=prompt_id, message="waits for the training run to finish" if busy else "")
        payload = {"prompt": {"cap": {"class_type": "ZenCaptionSession", "inputs": {"job_id": job.job_id},
                                      "_meta": {"title": title}}},
                   "prompt_id": prompt_id, "client_id": client_id}
        async with aiohttp.ClientSession() as s:
            async with s.post(_loopback(server) + "/prompt", json=payload) as r:
                result = await r.json()
        if r.status != 200:
            captioner.update(job, status="error", message=str(result.get("error")))
            return _bad("ComfyUI rejected the caption job", 422, comfy=result)
        return web.json_response({"job": job.public()})

    @routes.post("/zentrainer/api/caption/rewrite")
    async def rewrite_prompts(request):
        """{"clip_name", "prompts": [...], "format": "fibo_json"} rewrites each prompt as a structured
        caption; {"prompt": instruction} (format "text") rewrites each by that instruction instead.
        A job like a caption try-on: GET /caption/{job_id} has the results, nothing is written."""
        from ...datasets import captioner

        b = await request.json()
        prompts = [str(p).strip() for p in b.get("prompts") or [] if str(p).strip()][:32]
        if not b.get("clip_name"):
            return _bad("pick a language model to rewrite with")
        if not prompts:
            return _bad("no prompts to rewrite")
        fmt = str(b.get("format") or ("text" if b.get("prompt") else "fibo_json"))
        if fmt not in captioner.structured.FORMATS or (fmt == "text" and not str(b.get("prompt") or "").strip()):
            return _bad("rewrite as fibo_json, or give an instruction (prompt) for plain text")
        job = captioner.create(dataset="", set_name="", clip_name=str(b["clip_name"]), clip_type=str(b.get("clip_type") or "auto"),
                               prompt=str(b.get("prompt") or ""), preview=len(prompts), rewrite=prompts, format=fmt,
                               max_length=_length(b["max_length"]) if b.get("max_length") is not None else 0,
                               seed=int(b["seed"]) if b.get("seed") is not None else None, **_sampling(b))
        return await _submit_caption(job, "rewrite prompts", b.get("client_id"))

    @routes.post("/zentrainer/api/datasets/{name}/segment")
    async def segment_image(request):
        """SAM3 on one image: {path, text?, positive?: [{x,y}], negative?: [{x,y}], threshold?, refine?}.
        Waits up to ~20 s for the result ({job} with status done -> GET .../segment/{id}.png);
        when the GPU is busy (a training run) it returns the queued job to poll."""
        from .. import jobqueue
        from ...datasets import segment

        name = request.match_info["name"]
        b = await request.json()
        try:
            root = datasets.folder(name)
            img = (root / str(b.get("path", ""))).resolve()
            if root.resolve() not in img.parents or not img.is_file():
                return _bad("no such image in the dataset", 404)
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        job = segment.create(img, text=str(b.get("text") or ""), positive=list(b.get("positive") or []),
                             negative=list(b.get("negative") or []), threshold=float(b.get("threshold") or 0.5),
                             refine=int(b.get("refine", 2)))
        if jobqueue.get().current:
            job.message = "waits for the training run to finish"
        payload = {"prompt": {"seg": {"class_type": "ZenSegmentSession", "inputs": {"job_id": job.job_id},
                                      "_meta": {"title": f"mask {name}"}}}, "prompt_id": str(uuid.uuid4())}
        async with aiohttp.ClientSession() as s:
            async with s.post(_loopback(server) + "/prompt", json=payload) as r:
                res = await r.json()
        if r.status != 200:
            return _bad("ComfyUI rejected the segment job", 422, comfy=res)
        for _ in range(100):
            if job.status not in segment.ACTIVE:
                break
            await asyncio.sleep(0.2)
        return web.json_response({"job": job.public()})

    @routes.get("/zentrainer/api/segment/{job_id}")
    async def segment_job(request):
        from ...datasets import segment

        jid = request.match_info["job_id"]
        if jid.endswith(".png"):
            p = segment.result(jid[:-4])
            return web.FileResponse(p, headers={"Cache-Control": "no-store"}) if p else _bad("no mask yet", 404)
        job = segment.get(jid)
        return web.json_response({"job": job.public()}) if job else _bad("no such job", 404)

    @routes.put("/zentrainer/api/datasets/{name}/mask")
    async def save_mask(request):
        """Write an image's mask from the editor: {path, png: data URL}; empty png deletes it."""
        name = request.match_info["name"]
        b = await request.json()
        return await _ds(lambda: datasets.save_mask(name, str(b.get("path", "")), str(b.get("png") or "")))

    @routes.post("/zentrainer/api/datasets/{name}/masks/apply")
    async def apply_masks(request):
        """One mask onto many images of its size: {paths, png, mode: cutout|replace}.
        Returns {batch (to undo), results: [{path, mask} | {path, skipped}]}."""
        name, b = request.match_info["name"], await request.json()
        paths = [str(p) for p in (b.get("paths") or [])]
        return await _ds(lambda: datasets.apply_mask(name, paths, str(b.get("png") or ""), str(b.get("mode") or "cutout")))

    @routes.post("/zentrainer/api/datasets/{name}/masks/undo")
    async def undo_masks(request):
        """Undo an apply: {batch}."""
        name, b = request.match_info["name"], await request.json()
        return await _ds(lambda: datasets.undo_masks(name, str(b.get("batch") or "")))

    @routes.get("/zentrainer/api/caption/{job_id}")
    async def caption_job(request):
        from ...datasets import captioner

        job = captioner.get(request.match_info["job_id"])
        if job is None:
            return _bad("no such job", 404)
        return web.json_response({"job": {**job.public(), "captions": dict(job.written)}})

    @routes.post("/zentrainer/api/caption/{job_id}/{action}")
    async def caption_action(request):
        """cancel: stop it (before it starts, or at the next image); undo: put back what it changed."""
        from ...datasets import captioner

        job, action = captioner.get(request.match_info["job_id"]), request.match_info["action"]
        if job is None:
            return _bad("no such job", 404)
        if action == "cancel":
            was = job.status
            captioner.cancel(job)
            if was == "queued" and job.prompt_id:
                async with aiohttp.ClientSession() as s:
                    await s.post(_loopback(server) + "/queue", json={"delete": [job.prompt_id]})
            return web.json_response({"job": job.public()})
        if action == "undo":
            try:
                n = await asyncio.to_thread(captioner.undo, job)
            except (ValueError, FileNotFoundError) as e:
                return _bad(str(e))
            return web.json_response({"job": job.public(), "restored": n})
        return _bad("action is cancel or undo", 404)

    @routes.delete("/zentrainer/api/datasets/{name}/item")
    async def delete_item(request):
        name, rel = request.match_info["name"], request.query.get("path", "")
        return await _ds(lambda: datasets.delete(name, rel) or {"ok": True})

    # ---- control sets: edit training's input pictures, a named layer per set ------------------
    @routes.post("/zentrainer/api/datasets/{name}/controls/{set}")
    async def control_set_create_or_upload(request):
        """Upload input pictures into a set (each matched to its picture by name), or with no
        files, just make the set."""
        name, set_name = request.match_info["name"], request.match_info["set"]
        if not request.content_type.startswith("multipart/"):
            return await _ds(lambda: datasets.create_control_set(name, set_name) or {"ok": True})
        saved, errors = [], []
        reader = await request.multipart()
        async for part in reader:
            if part.filename:
                data = await part.read(decode=False)
                try:
                    saved.append(await asyncio.to_thread(datasets.save_control, name, set_name, part.filename, bytes(data)))
                except (datasets.DatasetError, FileNotFoundError) as e:
                    errors.append(str(e))
        return web.json_response({"saved": saved, "errors": errors})

    @routes.delete("/zentrainer/api/datasets/{name}/controls/{set}")
    async def control_set_delete(request):
        """One input (?path=) or, without one, the whole set."""
        name, set_name, rel = request.match_info["name"], request.match_info["set"], request.query.get("path")
        if rel:
            return await _ds(lambda: datasets.delete_control(name, set_name, rel) or {"ok": True})
        return await _ds(lambda: {"deleted": datasets.delete_control_set(name, set_name)})

    @routes.post("/zentrainer/api/datasets/{name}/controls/{set}/rename")
    async def control_set_rename(request):
        name, set_name = request.match_info["name"], request.match_info["set"]
        to = (await request.json()).get("to", "")
        return await _ds(lambda: {"moved": datasets.rename_control_set(name, set_name, to)})

    @routes.post("/zentrainer/api/datasets/{name}/upload")
    async def upload(request):
        name = request.match_info["name"]
        saved, errors = [], []
        reader = await request.multipart()
        async for part in reader:
            if part.filename:
                data = await part.read(decode=False)
                try:
                    saved.append(await asyncio.to_thread(datasets.save_upload, name, part.filename, bytes(data)))
                except (datasets.DatasetError, FileNotFoundError) as e:
                    errors.append(str(e))
        return web.json_response({"saved": saved, "errors": errors})
