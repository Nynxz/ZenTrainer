"""Helpers the API's areas share: errors, queueing a run as a ComfyUI job, ComfyUI's version,
restarting, and what the engine can do (capabilities)."""

from __future__ import annotations

import asyncio
import logging
import re
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

from ... import paths, safeio, state
from ...datasets import folders as datasets


WEB_DIST = Path(__file__).resolve().parents[4] / "web" / "dist"

# ended without training a step: editable, and saving puts it back in the queue
RETRYABLE = ("error", "stopped", "interrupted", "rejected")


def queue_resume(store, run_id: str, ckpt: dict | None, at: int = 0) -> None:
    """Put a stopped run back in the queue to continue from `ckpt` (None: from step 0), at
    position `at`: what Resume does, and what swapping to another run does for the one it stops."""
    from ...harness.trainer import state_for
    from .. import jobqueue

    step = ckpt["step"] if ckpt else 0
    _keep_later(store, run_id, step)
    for stream in ("metrics", "samples", "checkpoints", "validation", "likeness", "pruned"):
        store.truncate(run_id, stream, step)
    store.update(run_id, resume_from=ckpt["path"] if ckpt else None,
                 resume_state=state_for(ckpt, run_id) if ckpt else None, message=None, final_step=None,
                 status="queued")
    jobqueue.get().insert(run_id, at)


def latest_resumable(store, run_id: str) -> dict | None:
    from ...harness.trainer import state_for

    resumable = [c for c in store.read(run_id, "checkpoints") if state_for(c, run_id)]
    return max(resumable, key=lambda c: c["step"]) if resumable else None


def _keep_later(store, run_id: str, step: int) -> None:
    """Before a resume retrains past `step`, move the LoRAs and sample pictures saved after it into
    a `replaced-<time>` folder beside them: retraining writes the same file names, and those files
    are someone's results until they say otherwise."""
    import shutil

    tag = f"replaced-{time.strftime('%Y%m%d-%H%M%S')}"
    files = [Path(c["path"]) for c in store.read(run_id, "checkpoints") if c.get("step", 0) > step and c.get("path")]
    files += [paths.resolve_view("output", r.get("subfolder", ""), r["filename"]) for r in store.read(run_id, "samples")
              if r.get("step", 0) > step and r.get("filename") and r.get("type", "output") == "output"]
    for f in files:
        if f and f.is_file():
            keep = f.parent / tag / f.name
            keep.parent.mkdir(exist_ok=True)
            shutil.move(str(f), str(keep))


def _bad(message: str, status: int = 400, **extra) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


# An upload part read in chunks, up to a size: a whole part used to be read into memory with no
# limit, so one request could fill RAM. Past the limit that file is refused, not the request.
UPLOAD_MAX = 2 * 1024**3


async def read_upload(part, limit: int = UPLOAD_MAX) -> bytes | None:
    """A multipart part's bytes, or None (the part drained) when it's over `limit`."""
    chunks, size = [], 0
    while chunk := await part.read_chunk(1 << 20):
        size += len(chunk)
        if size > limit:
            while await part.read_chunk(1 << 20):  # drain it, so the next part reads
                pass
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# Requests from this machine. The API has no login (like ComfyUI's), so what reaches past
# ZenTrainer's own folders (a path anywhere on disk, where datasets live, a forced restart) is
# taken only from someone at this machine, not from anyone on the network (--listen) or through a
# tunnel. A tunnel or proxy connects from 127.0.0.1 itself, so a request carrying a proxy's
# headers counts as remote; and the Host has to be this machine's too, so a page on another site
# whose name has been pointed at 127.0.0.1 (DNS rebinding) doesn't pass for it.
_PROXY_HEADERS = ("Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP", "CF-Connecting-IP",
                  "True-Client-IP", "X-Client-IP")
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


def _is_loopback(host: str | None) -> bool:
    import ipaddress

    if not host:
        return False
    host = host.strip("[]").lower()
    if host in _LOCAL_NAMES or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_local(request: web.Request) -> bool:
    """The request is from someone at this machine (see above)."""
    if any(h in request.headers for h in _PROXY_HEADERS):
        return False
    import urllib.parse

    try:
        name = urllib.parse.urlsplit("//" + request.headers.get("Host", "")).hostname  # localhost:8189, [::1]:8189
    except ValueError:
        return False
    return _is_loopback(request.remote) and _is_loopback(name)


def _not_local(what: str) -> web.Response:
    return _bad(f"{what} only from this machine (not over the network or a tunnel): ZenTrainer has no "
                "login, so this is kept to someone at the computer it runs on", 403)


# The oldest ComfyUI ZenTrainer is tested with (comfy.lock pins newer). Installed as a node pack
# into someone's ComfyUI, an older one would fail in odd places; this says so plainly instead.
MIN_COMFY = (0, 41, 0)


def comfy_too_old() -> str | None:
    try:
        from comfyui_version import __version__ as v
    except Exception:
        return None
    nums = tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    if nums and nums < MIN_COMFY:
        return (f"ComfyUI {v} is older than ZenTrainer needs ({'.'.join(map(str, MIN_COMFY))} or newer): "
                "update ComfyUI (Manager > Update ComfyUI, or git pull)")
    return None


_client_id: str | None = None


def _remember_client(client_id: str | None) -> None:
    """Session jobs report progress to the last UI client that queued something."""
    global _client_id
    if client_id:
        _client_id = client_id


def _exec_self() -> None:
    import os
    import sys

    logging.info("[ZenTrainer] restarting ComfyUI")
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


async def _queue_session(server, run_id: str, name: str, client_id: str | None) -> web.Response:
    """Queue the one-node job that hosts the run. Its only job is to hold ComfyUI's
    execution lane; the run itself is defined by the spec in the store."""
    store = state.store()
    prompt_id = str(uuid.uuid4())
    prompt = {"session": {"class_type": "ZenTrainSession", "inputs": {"run_id": run_id},
                          "_meta": {"title": f"train {name}"}}}
    payload = {"prompt": prompt, "prompt_id": prompt_id, "extra_data": {"zentrainer_run": run_id}}
    if client_id:
        payload["client_id"] = client_id
    store.update(run_id, prompt_id=prompt_id, status="queued")
    async with aiohttp.ClientSession() as s:
        async with s.post(_loopback(server) + "/prompt", json=payload) as r:
            result = await r.json()
    if r.status != 200:
        store.update(run_id, status="rejected", message=str(result.get("error")))
        return _bad("ComfyUI rejected the run", 422, run_id=run_id, comfy=result)
    return web.json_response({"run": store.meta(run_id)})


async def _dequeue(server, run_id: str) -> web.Response:
    store = state.store()
    try:
        prompt_id = store.meta(run_id)["prompt_id"]
    except (FileNotFoundError, ValueError):
        return _bad("run not found", 404)
    async with aiohttp.ClientSession() as s:
        await s.post(_loopback(server) + "/queue", json={"delete": [prompt_id]})
    store.update(run_id, status="stopped", message="removed from queue")
    return web.json_response({"ok": True})


async def _ds(fn) -> web.Response:
    """Run a dataset operation off the event loop, mapping its errors to HTTP."""
    try:
        return web.json_response(await asyncio.to_thread(fn))
    except FileNotFoundError:
        return _bad("dataset not found", 404)
    except (datasets.DatasetError, KeyError) as e:
        return _bad(str(e))
    except Exception as e:  # noqa: BLE001
        # a picture that can't be opened, or too many pixels to (PIL's guard against a tiny file
        # that decodes huge): a clear refusal rather than a 500
        from PIL import Image, UnidentifiedImageError

        if isinstance(e, (Image.DecompressionBombError, UnidentifiedImageError)):
            return _bad(f"not a picture that can be read: {e}")
        raise


def _migrate_states() -> None:
    """Move training-state files out of the LoRA folders (older versions saved
    <name>.state.pt beside each checkpoint, where ComfyUI lists it as a LoRA)."""

    from ...harness.trainer import state_dir

    store = state.store()
    for meta in store.list():
        rid = meta["run_id"]
        rows = store.read(rid, "checkpoints")
        changed = False
        for row in rows:
            old = Path(row["path"]).with_suffix(".state.pt")
            if not row.get("state") and old.is_file():
                new = state_dir(rid) / f"{Path(row['path']).stem}.pt"
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(new))
                row["state"] = str(new)
                changed = True
        if changed:
            store.rewrite(rid, "checkpoints", rows)
            logging.info("[ZenTrainer] moved training state of %s out of the LoRA folder", rid)
    # strays not tied to a run row
    for root in (r / "loras" for r in paths.output_roots()):
        for stray in root.rglob("*.state.pt") if root.is_dir() else []:
            dest = safeio.free_name(state_dir("_orphaned") / stray.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stray), str(dest))


def _comfy_version(check: bool) -> dict:
    import json
    import subprocess

    import folder_paths

    root = Path(folder_paths.__file__).resolve().parent
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=60)  # noqa: E731
    out: dict = {"path": str(root)}
    try:
        import comfyui_version

        out["version"] = comfyui_version.__version__
    except Exception:
        pass
    head = git("rev-parse", "HEAD")
    if head.returncode == 0:
        out["commit"] = head.stdout.strip()
        out["describe"] = git("describe", "--tags", "--always").stdout.strip()
        out["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    lock = Path(__file__).resolve().parents[4] / "comfy.lock"
    if lock.is_file():
        pin = json.loads(lock.read_text())["comfyui"]
        out["pin"] = {"ref": pin["ref"], "describe": pin.get("describe")}
        out["matches_pin"] = out.get("commit") == pin["ref"]
    if check and "commit" in out:
        fetched = git("fetch", "--quiet", "origin")
        if fetched.returncode == 0:
            upstream = git("rev-parse", "origin/HEAD").stdout.strip() or git("rev-parse", "origin/master").stdout.strip()
            behind = git("rev-list", "--count", f"HEAD..{upstream}").stdout.strip()
            out["upstream"] = {"commit": upstream, "describe": git("describe", "--tags", "--always", upstream).stdout.strip(),
                               "behind": int(behind or 0)}
        else:
            out["upstream_error"] = fetched.stderr.strip()[:300]
    return out


def _loopback(server) -> str:
    host = server.address if server.address not in ("0.0.0.0", "::", "", None) else "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{server.port}"


_LOADED_AT = time.time()


def _code_stale() -> bool:
    """The engine's code on disk is newer than what this process loaded (an update or an edit):
    a restart picks it up. The sidebar offers one."""
    root = Path(__file__).resolve().parents[2]
    return any(p.stat().st_mtime > _LOADED_AT for p in root.rglob("*.py") if "__pycache__" not in p.parts)


REPO = Path(__file__).resolve().parents[4]


def _git(*args: str, timeout: float = 60) -> str:
    """A git command's output in ZenTrainer's checkout, or "" when it fails (or there's no git)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


# the commit this process loaded: the checkout can move on (a pull) while it runs
_LOADED_COMMIT = _git("rev-parse", "HEAD", timeout=10)


def _zt_version(check: bool) -> dict:
    """ZenTrainer's version and the commit running, a newer one on disk (restart to load it), and
    (check) what its upstream has that this checkout lacks: what `zt update` would bring. The
    check fetches, so it only runs when asked; like `zt update` it never changes the checkout."""
    from ... import __version__

    out: dict = {"version": __version__}
    if not _LOADED_COMMIT:
        return out  # not a git checkout: nothing to compare against
    out["commit"] = _LOADED_COMMIT
    out["describe"] = _git("describe", "--tags", "--always", _LOADED_COMMIT)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    out["branch"] = None if branch == "HEAD" else branch  # HEAD: detached, e.g. checked out at a tag
    out["dirty"] = bool(_git("status", "--porcelain", "--untracked-files=no"))
    head = _git("rev-parse", "HEAD")
    if head and head != _LOADED_COMMIT:
        out["on_disk"] = {"commit": head, "describe": _git("describe", "--tags", "--always", head)}
    if not check:
        return out
    up = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not up:
        out["upstream_error"] = (f"the branch {branch} doesn't track a remote branch, so there's nothing to compare with" if out["branch"]
                                 else "this checkout isn't on a branch (it's at a fixed commit), so there's nothing to compare with")
        return out
    # never stop to ask for a password or passphrase: nobody would see the prompt
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        fetched = subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", up.split("/", 1)[0]],
                                 capture_output=True, text=True, timeout=60, env=env)
        err = fetched.stderr.strip() if fetched.returncode else ""
    except (OSError, subprocess.SubprocessError) as e:
        err = str(e)
    if err:
        out["upstream_error"] = err[:300]
        return out
    new = _git("log", "--format=%h %s", f"HEAD..{up}")
    out["upstream"] = {"ref": up, "describe": _git("describe", "--tags", "--always", up),
                       "behind": len(new.splitlines()), "ahead": int(_git("rev-list", "--count", f"{up}..HEAD") or 0),
                       "commits": new.splitlines()[:12]}
    return out


def _web_state() -> dict:
    """The web app as built against its source on disk. web/dist isn't in git, so a pull brings new
    source while the page stays the old build until it's rebuilt (`web_stale`). `web_built` changes
    when a rebuild lands, so an open tab can offer to reload."""
    web_root = WEB_DIST.parent
    index = WEB_DIST / "index.html"
    built = index.stat().st_mtime if index.is_file() else None
    sources = [*(web_root / "src").rglob("*"), *(web_root / "public").rglob("*"),
               *(web_root / n for n in ("index.html", "package.json", "pnpm-lock.yaml", "vite.config.ts"))]
    newest = max((p.stat().st_mtime for p in sources if p.is_file()), default=0.0)
    return {"web_built": built, "web_stale": built is not None and newest > built, "web_buildable": shutil.which("pnpm") is not None}


def _capabilities() -> dict:
    from ...harness import optim

    caps: dict = {"optimizers": optim.catalog()}
    try:
        from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS

        from ...harness.memory import TRAINABLE_ATTENTION

        # Only backends with a backward pass are offered for training.
        caps["train_attention"] = [n for n in REGISTERED_ATTENTION_FUNCTIONS if n in TRAINABLE_ATTENTION]
        caps["infer_attention"] = list(REGISTERED_ATTENTION_FUNCTIONS)
    except Exception:
        pass
    try:
        import comfy.model_management as mm

        dev = mm.get_torch_device()
        caps["device"] = str(dev)
        caps["device_name"] = mm.get_torch_device_name(dev)
        caps["vram_total"] = mm.get_total_memory(dev)
        caps["vram_free"] = mm.get_free_memory(dev)
    except Exception:
        pass
    return caps
