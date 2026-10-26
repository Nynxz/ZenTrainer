"""Runs: create, edit while queued, resume, sample, control; presets; the training queue and its worker."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from ... import presets, sharing, state
from ...runs.thin import thin
from .. import jobqueue, system
from ...harness import control
from ...harness.keep import faces_missing
from ...spec import SpecError, TrainSpec, locked_changes, slugify
from . import common
from .common import RETRYABLE, _bad, _dequeue, _keep_later, _migrate_states, _queue_session, _remember_client


def register(server) -> None:
    routes = server.routes

    @routes.get("/zentrainer/api/runs")
    async def list_runs(_request):
        return web.json_response({"runs": state.store().list()})

    @routes.get("/zentrainer/api/spec/schema")
    async def spec_schema(_request):
        """The run spec's JSON Schema: every field's type, default, choices, bounds and meaning
        (docs/spec.md is the same, to read)."""
        from ...spec import schema

        return web.json_response(schema())

    def _with_size(c: dict) -> dict:
        # a cleaned-up training state can't be resumed from: say so rather than offer it
        c = {**c, "state": c["state"] if c.get("state") and Path(c["state"]).is_file() else None}
        try:
            return {**c, "size": Path(c["path"]).stat().st_size}
        except (OSError, KeyError):
            return {**c, "size": None}  # moved or deleted since

    @routes.get("/zentrainer/api/runs/{run_id}")
    async def get_run(request):
        """The run's meta, spec and logs. ?since=N: only the metrics from row N on (a monitor
        polling a long run), with metrics_count the run's total so far. Otherwise a long run's
        metrics come thinned (runs/thin.py: metrics_every steps per row) unless ?full=1."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            metrics = store.read(run_id, "metrics")
            since = request.query.get("since")
            if since and since.isdigit():
                rows, every = metrics[max(0, int(since)):], 1
            elif request.query.get("full") == "1":
                rows, every = metrics, 1
            else:
                rows, every = thin(metrics)
            return web.json_response({
                "run": store.meta(run_id),
                "spec": store.spec(run_id),
                "metrics": rows,
                "metrics_every": every,
                "metrics_count": len(metrics),
                "checkpoints": [_with_size(c) for c in store.read(run_id, "checkpoints")],
                "samples": store.read(run_id, "samples"),
                "validation": store.read(run_id, "validation"),
                "likeness": store.read(run_id, "likeness"),
                "pruned": store.read(run_id, "pruned"),
                "live": control.get(run_id) is not None,
            })
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)

    @routes.post("/zentrainer/api/runs")
    async def create_run(request):
        body = await request.json()
        try:
            spec = TrainSpec.from_dict(body.get("spec", {}))
        except (SpecError, TypeError) as e:
            return _bad(str(e))
        if (why := faces_missing(spec)) is not None:
            return _bad(why)

        store = state.store()
        run_id = store.new_id()
        store.create(run_id, spec.to_dict())
        # "next" jumps the queue (starts now when idle); "end" waits its turn
        jobqueue.get().add(run_id, front=body.get("position", "next") == "next")
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id), "queue": jobqueue.get().snapshot()})

    @routes.put("/zentrainer/api/runs/{run_id}/spec")
    async def edit_queued_run(request):
        """Change a run's settings: {"spec"}. A run waiting in the queue keeps its place; one
        that ended before it trained a step (failed loading, out of VRAM at step one, stopped
        while queued) goes back to the front of the queue with them. A run that has trained
        keeps its spec: its checkpoints were made with it (Duplicate makes a new one)."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("no such run", 404)
        waiting = run_id in jobqueue.get().snapshot()["order"] and meta.get("status") == "queued"
        retry = (control.get(run_id) is None and meta.get("status") in RETRYABLE
                 and not store.read(run_id, "metrics") and not store.read(run_id, "checkpoints"))
        # a stopped run that has trained: settings the saved LoRA and optimizer state don't
        # depend on can change before it resumes (memory, samples, LR, length...)
        resume_edit = (control.get(run_id) is None and meta.get("status") in RETRYABLE and not retry
                       and bool(store.read(run_id, "checkpoints")))
        if not (waiting or retry or resume_edit):
            return _bad(f"this run is {meta.get('status')}: it can't be edited", 409)
        body = await request.json()
        try:
            spec = TrainSpec.from_dict(body.get("spec", {}))
            spec.validate()
        except (SpecError, TypeError) as e:
            return _bad(str(e))
        if (why := faces_missing(spec)) is not None:
            return _bad(why)
        if resume_edit:
            # through the dataclass so fields added since it was saved compare at their defaults
            changed = locked_changes(TrainSpec.from_dict(store.spec(run_id)).to_dict(), spec.to_dict())
            if changed:
                return _bad("these can't change once a run has trained (its LoRA and optimizer state were "
                            f"made with them; Duplicate starts over): {', '.join(changed)}", 409, locked=changed)
        store.set_spec(run_id, spec.to_dict())
        if retry:
            for stream in ("samples", "validation"):
                store.rewrite(run_id, stream, [])  # its step-0 samples come again with the new settings
            store.update(run_id, message=None, final_step=None, resume_from=None, resume_state=None, status="queued")
            jobqueue.get().add(run_id, front=True)
            _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id), "requeued": retry})

    @routes.post("/zentrainer/api/runs/{run_id}/resume")
    async def resume_run(request):
        """Continue a stopped/failed run from its latest resumable checkpoint, or from
        a given one ({"checkpoint": path}); with none, it restarts from step 0."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)
        if control.get(run_id) is not None or meta["status"] in ("queued", "loading", "training", "paused"):
            return _bad("run is already active", 409)
        body = await request.json() if request.can_read_body else {}
        from ...harness.trainer import state_for

        resumable = [c for c in store.read(run_id, "checkpoints") if state_for(c, run_id)]
        if body.get("checkpoint"):
            resumable = [c for c in resumable if c["path"] == body["checkpoint"]]
            if not resumable:
                return _bad("that checkpoint has no saved training state", 400)
        ckpt = max(resumable, key=lambda c: c["step"]) if resumable else None
        if ckpt is None and (store.read(run_id, "checkpoints") or store.read(run_id, "metrics")):
            # starting over would drop its history and retrain over its saved LoRAs and samples
            return _bad("this run has no saved training state to continue from (Resume states was off, or "
                        "they were cleaned up). Duplicate it to train again from the start.", 409)
        common.queue_resume(store, run_id, ckpt, at=0)
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id)})

    @routes.get("/zentrainer/api/runs/{run_id}/download")
    async def download_checkpoint(request):
        """One of the run's saved LoRAs as a file to save (?lora=<its lora_name>; the final if left
        out). Only files the run's checkpoint rows name are served."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            rows = store.read(run_id, "checkpoints")
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)
        want = request.query.get("lora")
        row = next((c for c in rows if c.get("lora_name") == want), None) if want else \
            next((c for c in reversed(rows) if c.get("final")), None)
        path = Path(row["path"]) if row and row.get("path") else None
        if not path or not path.is_file():
            return _bad("no such checkpoint on disk", 404)
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{path.name}"'})

    @routes.post("/zentrainer/api/runs/{run_id}/extend")
    async def extend_run(request):
        """Train a finished run on, to {"steps": the new total}, from the state its final save kept:
        optimizer, step, RNG and average carry on. Its final LoRA becomes a checkpoint first
        (<name>_step<N>), so the new final doesn't overwrite it."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)
        if control.get(run_id) is not None or meta["status"] != "done":
            return _bad("only a finished run trains on (a stopped one resumes)", 409)
        body = await request.json() if request.can_read_body else {}
        ckpt = common.latest_resumable(store, run_id)
        if ckpt is None or not ckpt.get("final") or not Path(ckpt["path"]).is_file():
            return _bad("this run kept no training state at its end (Resume states was off, or it was cleaned "
                        "up): start a new run with Continue from set to its LoRA", 409)
        try:
            steps = int(body.get("steps"))
        except (TypeError, ValueError):
            return _bad("steps is the new total")
        if steps <= ckpt["step"]:
            return _bad(f"steps is the new total: more than the {ckpt['step']} it has trained")
        spec = TrainSpec.from_dict(store.spec(run_id))
        spec.optim.steps = steps
        try:
            spec.validate()
        except SpecError as e:
            return _bad(str(e))
        from ...runs.extend import keep_final

        ckpt = keep_final(store, run_id, ckpt, spec.output.lora_dir)
        store.set_spec(run_id, spec.to_dict())
        common.queue_resume(store, run_id, ckpt, at=0)
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id)})

    @routes.post("/zentrainer/api/runs/{run_id}/sample")
    async def sample_run(request):
        """Sample a finished run's checkpoint: {"checkpoint"?: path (default the latest), "run_prompts"?:
        bool (default true), "tries"?: [{prompt, width?, height?, seed?}]}. Renders at that checkpoint's
        step, replacing the run's samples already there; the run keeps its status."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)
        if control.get(run_id) is not None or meta["status"] in ("queued", "loading", "training", "paused", "sampling"):
            return _bad("run is active: use Sample now while it trains", 409)
        body = await request.json() if request.can_read_body else {}
        ckpts = [c for c in store.read(run_id, "checkpoints") if c.get("path") and Path(c["path"]).is_file()]
        if body.get("checkpoint"):
            ckpts = [c for c in ckpts if c["path"] == body["checkpoint"]]
        if not ckpts:
            return _bad("no checkpoint file to sample", 400)
        ckpt = max(ckpts, key=lambda c: (c["step"], bool(c.get("final"))))
        run_prompts = body.get("run_prompts", True)
        tries = [t for t in body.get("tries") or [] if isinstance(t, dict) and str(t.get("prompt", "")).strip()]
        if not run_prompts and not tries:
            return _bad("nothing to sample", 400)
        if run_prompts:  # a re-sample replaces the run's samples (and their likeness) at that step
            for stream in ("samples", "likeness"):
                rows = store.read(run_id, stream)
                store.rewrite(run_id, stream, [r for r in rows if r.get("step") != ckpt["step"] or r.get("manual")])
        store.update(run_id, status="queued", message=f"sampling step {ckpt['step']}", sample_job={
            "path": ckpt["path"], "step": ckpt["step"], "run_prompts": bool(run_prompts), "tries": tries,
            "prev_status": meta["status"], "prev_message": meta.get("message")})
        jobqueue.get().add(run_id, front=True)
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id)})

    @routes.post("/zentrainer/api/runs/{run_id}/control")
    async def control_run(request):
        """{"command": "pause"|"resume"|"stop"|"save"|"sample"|"set_lr", "args": {...}}"""
        run_id = request.match_info["run_id"]
        body = await request.json()
        ctl = control.get(run_id)
        if ctl is None and run_id in jobqueue.get().parked:  # paused, having let go of ComfyUI
            return await _unpark(run_id, body)
        if ctl is None:
            if body.get("command") == "stop":  # waiting in our queue, or dispatched but not started
                if jobqueue.get().remove(run_id):
                    state.store().update(run_id, status="stopped", message="removed from queue")
                    return web.json_response({"ok": True})
                return await _dequeue(server, run_id)
            return _bad("run is not training", 409)
        try:
            ctl.send(body.get("command", ""), **(body.get("args") or {}))
        except (ValueError, TypeError) as e:
            return _bad(str(e))
        if body.get("command") == "sample":  # acknowledge now; it renders at the next step boundary
            from ... import events
            events.emit(events.SAMPLE_PROGRESS, {"run_id": run_id, "phase": "requested"})
        return web.json_response({"ok": True})

    async def _unpark(run_id: str, body: dict):
        """A parked run (paused, ComfyUI let go): resume puts it back first in the queue to continue
        from its stop's state; stop leaves it stopped. Either way the queue moves again."""
        store, q = state.store(), jobqueue.get()
        cmd = body.get("command")
        if cmd not in ("resume", "stop"):
            return _bad("this run is paused and has let go of ComfyUI: resume or stop it", 409)
        q.parked.discard(run_id)
        store.update(run_id, parked=None)
        if cmd == "stop":
            store.update(run_id, status="stopped", message="stopped while paused")
            return web.json_response({"ok": True})
        ckpt = common.latest_resumable(store, run_id)
        if ckpt is None:  # its state was cleaned up while it waited: starting over would drop its history
            store.update(run_id, status="stopped", message="its saved training state is gone; Duplicate it to train again")
            return _bad("this run's saved training state is gone (cleaned up?): it can't continue", 409)
        common.queue_resume(store, run_id, ckpt, at=0)
        _remember_client(body.get("client_id"))
        return web.json_response({"ok": True, "run": store.meta(run_id)})

    def _preset_root() -> Path:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "zentrainer" / "presets"

    @routes.get("/zentrainer/api/presets/{kind}")
    async def list_presets(request):
        try:
            return web.json_response({"presets": presets.list_presets(request.match_info["kind"], _preset_root())})
        except presets.PresetError as e:
            return _bad(str(e))

    @routes.post("/zentrainer/api/presets/{kind}")
    async def save_preset(request):
        b = await request.json()
        try:
            row = presets.save(request.match_info["kind"], str(b.get("name", "")), b.get("data"), _preset_root(),
                               str(b.get("description", "")))
        except presets.PresetError as e:
            return _bad(str(e))
        return web.json_response({"preset": row})

    # ---- run settings as a file to share (sharing.py) ----------------------------------------
    @routes.get("/zentrainer/api/runs/{run_id}/export")
    async def export_run(request):
        try:
            spec = state.store().spec(request.match_info["run_id"])
        except (FileNotFoundError, ValueError):
            return _bad("no such run", 404)
        doc = sharing.export(spec)
        name = slugify(doc["name"] or "run") or "run"
        return web.json_response(doc, headers={"Content-Disposition": f'attachment; filename="{name}.zentrainer.json"'})

    @routes.post("/zentrainer/api/share/export")
    async def export_spec(request):
        b = await request.json()
        if not isinstance(b.get("spec"), dict):
            return _bad("spec required")
        return web.json_response(sharing.export(b["spec"]))

    @routes.post("/zentrainer/api/share/import")
    async def import_spec(request):
        try:
            b = await request.json()
        except ValueError:
            return _bad("not JSON")
        try:
            return web.json_response(sharing.read(b.get("doc") if isinstance(b, dict) and "doc" in b else b))
        except sharing.ShareError as e:
            return _bad(str(e))

    @routes.delete("/zentrainer/api/presets/{kind}/{id}")
    async def delete_preset(request):
        try:
            ok = presets.delete(request.match_info["kind"], request.match_info["id"], _preset_root())
        except presets.PresetError as e:
            return _bad(str(e))
        return web.json_response({"ok": ok}) if ok else _bad("no such user preset (built-ins can't be deleted)", 404)

    @routes.get("/zentrainer/api/queue")
    async def get_queue(_request):
        q, store = jobqueue.get().snapshot(), state.store()
        def meta(rid):
            try:
                return store.meta(rid)
            except (FileNotFoundError, ValueError):
                return None
        return web.json_response({
            "paused": q["paused"],
            "current": meta(q["current"]) if q["current"] else None,
            "items": [m for m in map(meta, q["order"]) if m],
        })

    @routes.post("/zentrainer/api/queue")
    async def edit_queue(request):
        """{"action": "pause"|"resume"} or {"action": "order", "order": [...]} or
        {"action": "top"|"remove"|"swap", "run_id": ...}. Swap trains a queued run now: the current
        run stops (it saves its state, as a stop does) and goes back in the queue right after it,
        to resume from there."""
        b, q = await request.json(), jobqueue.get()
        action = b.get("action")
        if action == "swap":
            rid = str(b.get("run_id"))
            snap = q.snapshot()
            if rid not in snap["order"]:
                return _bad("only a queued run can be swapped to", 409)
            q.start_next(rid)  # starts even while the queue is paused; the rest stay held
            cur = snap["current"]
            ctl = control.get(cur) if cur else None
            if ctl is not None:
                state.store().update(cur, swap_resume=True)
                ctl.send("stop")
            return web.json_response(q.snapshot())
        if action in ("pause", "resume"):
            q.set_paused(action == "pause")
        elif action == "order":
            q.reorder([str(r) for r in b.get("order", [])])
        elif action == "top":
            q.add(str(b.get("run_id")), front=True)
        elif action == "remove":
            if q.remove(str(b.get("run_id"))):
                state.store().update(str(b["run_id"]), status="stopped", message="removed from queue")
        else:
            return _bad(f"unknown action {action!r}")
        return web.json_response(q.snapshot())

    async def _start_worker(_app):
        await asyncio.to_thread(_migrate_states)
        # runs that were mid-flight when this process last stopped can't be running now
        store, waiting = state.store(), set(jobqueue.get().snapshot()["order"])
        for m in store.list():
            job = m.get("sample_job")
            if job and (m["status"] == "sampling" or (m["status"] == "queued" and m["run_id"] not in waiting)):
                # a checkpoint was being sampled: the run itself was finished, so it goes back as it was
                store.update(m["run_id"], status=job.get("prev_status") or "done", message=job.get("prev_message"),
                             sample_job=None)
                continue
            if m["status"] in ("loading", "training", "paused") or (m["status"] == "queued" and m["run_id"] not in waiting):
                store.update(m["run_id"], status="interrupted", message="ComfyUI stopped during this run; resume it")

        async def dispatch(run_id: str):
            try:
                name = state.store().meta(run_id)["name"]
            except (FileNotFoundError, ValueError):
                return
            await _queue_session(server, run_id, name, common._client_id)
        async def swapped(run_id: str):
            # stopped by "swap to": back in the queue after the run it made room for, to resume from
            # its stop's state (a stop that saved none leaves it stopped, and says so)
            store = state.store()
            store.update(run_id, swap_resume=None)
            if store.meta(run_id)["status"] != "stopped":
                return  # it finished or failed on its own before the stop landed
            ckpt = common.latest_resumable(store, run_id)
            if ckpt is None:
                store.update(run_id, message="swapped out, but it saved no training state to resume from")
                return
            common.queue_resume(store, run_id, ckpt, at=1)

        async def parked(run_id: str):
            # paused and let go of ComfyUI for a workflow (Trainer._park): paused, holding the queue,
            # to continue from the state its stop saved when it's resumed
            store = state.store()
            store.update(run_id, park=None)
            if store.meta(run_id)["status"] != "stopped":
                return  # it finished or failed on its own first
            ckpt = common.latest_resumable(store, run_id)
            if ckpt is None:
                store.update(run_id, message="paused for a ComfyUI workflow, but it saved no training state to resume from")
                return
            store.update(run_id, status="paused", parked=True,
                         message=f"paused at step {ckpt['step']}, and let go of ComfyUI for a workflow: Resume continues from there")
            jobqueue.get().parked.add(run_id)

        _app["zentrainer_queue"] = asyncio.create_task(jobqueue.worker(dispatch, swapped, parked))
        _app["zentrainer_system"] = asyncio.create_task(system.sampler())

    server.app.on_startup.append(_start_worker)
