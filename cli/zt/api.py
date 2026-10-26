"""Headless: drive ZenTrainer over its HTTP API (everything the app does goes through it).

These are thin: a spec file in, a run id out, `follow` to block until it ends (exit 0 done,
1 failed or stopped), and `monitor` (zt.monitor) to watch everything at once.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from . import comfy, ports, repo, ui

TERMINAL = ("done", "error", "stopped", "interrupted", "rejected")
STATUS_STYLE = {"running": "bold green", "queued": "cyan", "paused": "yellow", "done": "green", "error": "red",
                "stopped": "yellow", "interrupted": "yellow", "rejected": "red"}


def call(method: str, path: str, body=None, port: int | None = None, timeout: float = 60):
    """One API request; dies with the server's own message when it says no."""
    url = f"http://127.0.0.1:{ports.current(port)}/zentrainer/api{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("error") or e.reason
        except Exception:
            msg = e.reason
        ui.die(f"{method} {path}: {msg}")
    except urllib.error.URLError:
        ui.die(f"no ZenTrainer on :{ports.current(port)} (zt start)")


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v  # a bare string


def apply_sets(spec: dict, sets: list[str] | None) -> dict:
    """--set optim.learning_rate=1e-4 --set samples.prompts='["a cat"]': dotted keys, JSON values."""
    for item in sets or []:
        key, eq, val = item.partition("=")
        if not eq:
            ui.die(f"--set wants key=value, got {item!r}")
        node = spec
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = _value(val)
    return spec


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ---- runs ----------------------------------------------------------------------------------------

def _engine_spec():
    """The engine's spec module (plain Python, no ComfyUI or torch): the same checks as the server."""
    if str(repo.ENGINE) not in sys.path:
        sys.path.insert(0, str(repo.ENGINE))
    from zentrainer import spec

    return spec


def _profile(profile: str | None) -> dict:
    if not profile:
        return {}
    prof = repo.ENGINE / "profiles" / f"{profile}.json"
    if not prof.is_file():
        have = ", ".join(p.stem for p in (repo.ENGINE / "profiles").glob("*.json"))
        ui.die(f"no profile {profile!r}; have {have}")
    return json.loads(prof.read_text()).get("train", {})


def load_spec(spec_path: str, profile: str | None = None, sets: list[str] | None = None) -> dict:
    """A spec file (- for stdin) over a profile's settings, then --set on top."""
    try:
        spec = json.loads(Path(spec_path).read_text()) if spec_path != "-" else json.load(sys.stdin)
    except (OSError, ValueError) as e:
        ui.die(f"{spec_path}: {e}")
    if not isinstance(spec, dict):
        ui.die(f"{spec_path}: a spec is a JSON object")
    return apply_sets(_deep_merge(_profile(profile), spec), sets)  # the file wins over the profile


def _report(spec: dict) -> bool:
    """Print every problem with the spec; True when there are none."""
    _, errors, notes = _engine_spec().problems(spec)
    for n in notes:
        ui.say(n)
    for e in errors:
        ui.console.print(Text.assemble(("  x ", "red"), e))
    return not errors


def check(spec_path: str, profile: str | None = None, sets: list[str] | None = None) -> None:
    """Every problem in a spec, without queuing anything or needing ZenTrainer running (exit 1 if
    any). The rules are the server's; whether the model files and dataset exist is checked when
    it's queued."""
    if not _report(load_spec(spec_path, profile, sets)):
        sys.exit(1)
    ui.ok("the spec is fine")


def new(profile: str | None = None, sets: list[str] | None = None) -> None:
    """A complete spec with every setting at its default (a profile's on top, then --set), to start
    from: fill in the model and dataset, and remove what you don't change."""
    sp = _engine_spec()
    spec = apply_sets(_deep_merge(sp.TrainSpec().to_dict(), sp.migrate(_profile(profile))[0]), sets)
    print(json.dumps({"$schema": sp.SCHEMA_URL, **spec}, indent=2))


def train(spec_path: str, profile: str | None = None, sets: list[str] | None = None, end: bool = False,
          follow_it: bool = False, port: int | None = None, **follow_opts) -> None:
    spec = load_spec(spec_path, profile, sets)
    if not _report(spec):  # all of them now, rather than the server's first
        ui.die("not queued: fix the spec (zt check shows the same)")
    port = comfy.background(port)
    run = call("POST", "/runs", {"spec": spec, "position": "end" if end else "next"}, port)["run"]
    print(run["run_id"])
    if follow_it:
        follow(run["run_id"], port=port, **follow_opts)


def runs(as_json: bool = False, port: int | None = None) -> None:
    rows = call("GET", "/runs", port=port)["runs"]
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    t = Table(box=None, padding=(0, 2), header_style="dim")
    for col in ("run", "status", "steps", "name"):
        t.add_column(col, justify="right" if col == "steps" else "left")
    for r in rows:
        status = r.get("status", "?")
        t.add_row(Text(r["run_id"], style="dim"), Text(status, style=STATUS_STYLE.get(status, "")),
                  str(r.get("final_step") or ""), r.get("name", ""))
    ui.console.print(t)


def new_rows(d: dict, seen: int | None) -> tuple[list[dict], int]:
    """The metric rows after `seen`, and the run's count so far. A server from before ?since
    sends them all every time: those are sliced here instead."""
    rows = d.get("metrics") or []
    if "metrics_count" in d:
        return rows, d["metrics_count"]
    return rows[seen or 0:], len(rows)


def _line(m: dict, total, phases: bool) -> str:
    rate = m.get("rate") or m.get("it_s") or 0
    ph = m.get("phases") or {}
    extra = " ".join(f"{k} {v:.2f}s" for k, v in ph.items()) if phases else ""
    return (f"step {m.get('step')}/{total}  loss {m.get('loss') or 0:.4f}  lr {m.get('lr') or 0:.2e}"
            + (f"  {1 / rate:.2f}s/step" if rate else "") + (f"  {extra}" if extra else ""))


def follow(run_id: str, every: int = 10, interval: float = 5.0, phases: bool = False, port: int | None = None) -> None:
    """Show progress until the run ends; the exit status says how it ended. A live bar on a
    terminal, a line every `every` steps otherwise (logs, scripts)."""
    live = ui.console.is_terminal
    seen, last_status, last_samples, last_val = None, None, None, None
    progress = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                        TextColumn("{task.fields[detail]}"), TimeRemainingColumn(), console=ui.console,
                        disable=not live)
    task = progress.add_task(run_id, total=None, detail="")
    with progress:
        out = progress.console
        while True:
            d = call("GET", f"/runs/{run_id}" + (f"?since={seen}" if seen is not None else ""), port=port)
            run, spec = d["run"], d.get("spec") or {}
            status = run.get("status")
            total = spec.get("optim", {}).get("steps")
            if status != last_status:
                msg = run.get("message")
                out.print(Text.assemble(("› ", "magenta"), (status or "?", STATUS_STYLE.get(status, "")),
                                        f": {msg}" if msg else ""))
                last_status = status
            rows, count = new_rows(d, seen)
            if seen is None:  # joining a run partway: start at its latest step
                rows = rows[-1:]
            if rows:
                m = rows[-1]
                rate = m.get("rate") or m.get("it_s") or 0
                progress.update(task, description=run.get("name") or run_id, total=total, completed=m.get("step") or 0,
                                detail=f"loss {m.get('loss') or 0:.4f}" + (f"  {1 / rate:.2f}s/step" if rate else ""))
                if not live:
                    for m in rows:
                        if m.get("step", 0) % every == 0 or m is rows[-1]:
                            print(_line(m, total, phases), flush=True)
            seen = count
            n = len(d.get("samples") or [])
            if last_samples is not None and n > last_samples:
                out.print(Text.assemble(("› ", "magenta"), f"{n - last_samples} new sample(s)"))
            last_samples = n
            val = (d.get("validation") or [None])[-1]
            if val and val != last_val and last_val is not None:
                out.print(Text.assemble(("› ", "magenta"), f"validation at step {val.get('step')}: {val.get('val_loss', 0):.4f}"))
            last_val = val
            if status in TERMINAL:
                ck = d.get("checkpoints") or []
                if ck:
                    out.print(ck[-1]["path"])
                sys.exit(0 if status == "done" else 1)
            time.sleep(interval)


def spec(run_id: str, port: int | None = None) -> None:
    print(json.dumps(call("GET", f"/runs/{run_id}?since=999999999", port=port)["spec"], indent=2))


def control(command: str, run_id: str, port: int | None = None) -> None:
    call("POST", f"/runs/{run_id}/control", {"command": {"unpause": "resume"}.get(command, command)}, port)
    ui.ok(f"{command}: {run_id}")


def resume(run_id: str, sets: list[str] | None = None, follow_it: bool = False, port: int | None = None,
           **follow_opts) -> None:
    port = comfy.background(port)
    d = call("GET", f"/runs/{run_id}?since=999999999", port=port)
    if sets:  # settings that may change before it continues (memory, samples, LR, length...)
        call("PUT", f"/runs/{run_id}/spec", {"spec": apply_sets(d["spec"], sets)}, port)
    if d["run"].get("status") == "queued":  # waiting its turn already: the edit is all it needs
        ui.ok(f"{run_id} is queued; settings updated")
    else:
        call("POST", f"/runs/{run_id}/resume", {}, port)
        ui.ok(f"resumed {run_id}")
    if follow_it:
        follow(run_id, port=port, **follow_opts)


def sample(run_id: str, prompt: str, base: bool = False, size: str | None = None, seed: int | None = None,
           port: int | None = None) -> None:
    args: dict = {"prompt": prompt, "base": base}
    if size:
        w, _, h = size.lower().partition("x")
        args.update(width=int(w), height=int(h))
    if seed is not None:
        args["seed"] = seed
    call("POST", f"/runs/{run_id}/control", {"command": "sample", "args": args}, port)
    ui.ok("queued; it renders at the next step boundary")


def profile(run_id: str, lines: int = 40, port: int | None = None) -> None:
    """Profile the run's next training step; print where the GPU time went."""
    folder = repo.COMFY / "user" / "zentrainer" / "runs" / run_id
    before = set(folder.glob("profile-step*.txt"))
    call("POST", f"/runs/{run_id}/control", {"command": "profile"}, port)
    with ui.console.status("profiling the next step…"):
        for _ in range(600):
            new = sorted(set(folder.glob("profile-step*.txt")) - before)
            if new:
                break
            time.sleep(1)
        else:
            ui.die("no profile came back (is the run training?)")
    print("\n".join(new[-1].read_text().splitlines()[:lines]))
    ui.say(f"full table and Chrome trace: {new[-1].with_suffix('')}.*")


def queue(action: str | None = None, port: int | None = None) -> None:
    if action:
        call("POST", "/queue", {"action": action}, port)
    q = call("GET", "/queue", port=port)
    head = Text("paused" if q["paused"] else "running", style="yellow" if q["paused"] else "green")
    if q.get("current"):
        head.append(f", training {q['current']['run_id']}  {q['current'].get('name', '')}")
    ui.console.print(head)
    for r in q["items"]:
        ui.console.print(Text.assemble("  ", (r["run_id"], "dim"), "  ", r.get("name", "")))


def caption(dataset: str, model: str, style: str = "nl", set_name: str = "default", all_: bool = False,
            follow_it: bool = False, port: int | None = None) -> None:
    port = comfy.background(port)
    body = {"clip_name": model, "style": style, "mode": "all" if all_ else "missing", "set": set_name,
            "replace_default": all_}
    job = call("POST", f"/datasets/{dataset}/caption", body, port)["job"]
    print(job["job_id"])
    if not follow_it:
        return
    with Progress(TextColumn("captioning"), BarColumn(), TextColumn("{task.completed}/{task.total}  {task.fields[msg]}"),
                  console=ui.console) as progress:
        task = progress.add_task("", total=None, msg="")
        while True:
            j = call("GET", f"/caption/{job['job_id']}", port=port)
            j = j.get("job", j)
            progress.update(task, completed=j.get("done", 0), total=j.get("total"), msg=(j.get("message") or "")[:60])
            if j.get("status") in ("done", "error", "cancelled", "stopped"):
                break
            time.sleep(2)
    sys.exit(0 if j.get("status") == "done" else 1)
