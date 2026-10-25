"""zt: set up, run, update and watch ZenTrainer from a terminal.

    zt                         the menu: first-run setup, update check, start, monitor
    zt setup [--models-from DIR]... [--extra-model-paths YAML]... [--torch VARIANT] [--zenkit DIR]
    zt start [--port N] [--open] [-- extra ComfyUI args]
    zt dev [--port N]          ComfyUI in the background + the web app with hot reload
    zt restart [--force]       restart the running managed ComfyUI
    zt monitor                 a live dashboard of the queue, runs and memory
    zt extras optimizers
    zt update [--ref REF]      pull ZenTrainer (fast-forward only), re-sync ComfyUI and the web app
    zt attach COMFYUI_DIR      zt doctor      zt web [--build]

Headless (starts ComfyUI in the background when it isn't running):
    zt train SPEC.json [--profile ID] [--set key=value]... [--end] [--follow]
    zt runs [--json]           zt follow RUN       zt spec RUN > spec.json
    zt pause|unpause|stop|save RUN                 zt resume RUN [--set key=value]...
    zt sample RUN "prompt" [--base] [--size WxH] [--seed N]
    zt queue [pause|resume]    zt caption DATASET --model FILE [--style nl|tags] [--all]
    zt profile RUN             zt shrink PATH... [--dtype bf16|fp16] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from . import api, comfy, repo, ui


def _client(sub, name: str, help: str):
    s = sub.add_parser(name, help=help)
    s.add_argument("--port", type=int)
    return s


def _follow_flags(s) -> None:
    s.add_argument("--every", type=int, default=10, help="print every Nth step (off a terminal)")
    s.add_argument("--interval", type=float, default=5.0, help="seconds between checks")
    s.add_argument("--phases", action="store_true", help="show the step's timing breakdown (off a terminal)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zt", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("setup", help="clone the pinned ComfyUI, install it, link the engine")
    s.add_argument("--models-from", action="append", metavar="DIR",
                   help="an existing ComfyUI install or models folder to use in place (repeatable)")
    s.add_argument("--extra-model-paths", action="append", metavar="YAML",
                   help="a ComfyUI extra_model_paths.yaml to use as-is (repeatable)")
    s.add_argument("--torch", help="torch build: cu130, cu128, cu126, rocm10.1.0, cpu (default: detect)")
    s.add_argument("--zenkit", metavar="DIR", help="link a ComfyUI-ZenKit checkout for its themes")
    s.add_argument("--build-web", action="store_true", help="rebuild web/dist even if it's current")

    s = sub.add_parser("start", help="run the managed ComfyUI with ZenTrainer")
    s.add_argument("--port", type=int)
    s.add_argument("--open", action="store_true", help="open the app in a browser")
    s.add_argument("rest", nargs=argparse.REMAINDER, help="after --: extra ComfyUI arguments")

    _client(sub, "dev", "ComfyUI in the background + the web app with hot reload")
    s = _client(sub, "restart", "restart the running managed ComfyUI (picks up engine changes)")
    s.add_argument("--force", action="store_true", help="even if a run is training")
    _client(sub, "monitor", "a live dashboard of the queue, runs and memory")

    s = sub.add_parser("extras", help="install optional packages into the managed ComfyUI")
    s.add_argument("groups", nargs="+", choices=list(comfy.EXTRAS),
                   help="optimizers: Prodigy, 8-bit, schedule-free, CAME")
    s = sub.add_parser("update", help="pull ZenTrainer and re-sync ComfyUI and the web app, or move the pin with --ref")
    s.add_argument("--ref", help="a ComfyUI commit, tag or branch to pin")
    s = sub.add_parser("attach", help="link the engine into an existing ComfyUI instead")
    s.add_argument("comfyui")
    sub.add_parser("doctor", help="show what's installed and where models come from")
    s = sub.add_parser("web", help="bring the web app up to date: the prebuilt one, or built with pnpm")
    s.add_argument("--build", action="store_true", help="rebuild it even if it's current")

    s = _client(sub, "train", "queue a training run from a spec file (- for stdin); prints its id")
    s.add_argument("spec")
    s.add_argument("--profile", help="start from an engine profile's settings (e.g. minimax-h3)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting (dotted key, JSON value)")
    s.add_argument("--end", action="store_true", help="join the end of the queue instead of starting next")
    s.add_argument("--follow", action="store_true", help="show progress until it ends")
    _follow_flags(s)
    s = _client(sub, "runs", "list runs")
    s.add_argument("--json", action="store_true")
    s = _client(sub, "spec", "print a run's settings as JSON")
    s.add_argument("run")
    s = _client(sub, "follow", "show a run's progress until it ends (exit 0 = done)")
    s.add_argument("run")
    _follow_flags(s)
    for name in ("pause", "unpause", "stop", "save"):
        _client(sub, name, f"{name} a training run").add_argument("run")
    s = _client(sub, "resume", "continue a stopped run from its latest checkpoint")
    s.add_argument("run")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="change a setting first (not model/dataset/network)")
    s.add_argument("--follow", action="store_true")
    _follow_flags(s)
    s = _client(sub, "sample", "render a prompt with the current LoRA mid-run")
    s.add_argument("run")
    s.add_argument("prompt")
    s.add_argument("--base", action="store_true", help="without the LoRA, to compare")
    s.add_argument("--size", metavar="WxH")
    s.add_argument("--seed", type=int)
    s = _client(sub, "profile", "profile one training step: where the GPU time goes")
    s.add_argument("run")
    s.add_argument("--lines", type=int, default=40)
    s = sub.add_parser("shrink", help="rewrite fp32 LoRA files in bf16 (half the size)")
    s.add_argument("paths", nargs="+", help=".safetensors files or folders of them")
    s.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    s.add_argument("--dry-run", action="store_true", help="only say what it would save")
    s = _client(sub, "queue", "show the queue, or pause/resume it")
    s.add_argument("action", nargs="?", choices=["pause", "resume"])
    s = _client(sub, "caption", "auto-caption a dataset")
    s.add_argument("dataset")
    s.add_argument("--model", required=True, help="a vision text encoder file (as listed in the app)")
    s.add_argument("--style", default="nl")
    s.add_argument("--set", dest="set_name", default="default", help="caption set")
    s.add_argument("--all", action="store_true", help="rewrite existing captions too")
    s.add_argument("--follow", action="store_true")
    return ap


def dispatch(a: argparse.Namespace) -> None:
    from . import menu, ports, update

    port = getattr(a, "port", None)
    follow = {k: getattr(a, k) for k in ("every", "interval", "phases") if hasattr(a, k)}
    match a.cmd:
        case None:
            menu.menu()
        case "setup":
            comfy.setup(a.models_from, a.extra_model_paths, a.torch, a.zenkit, a.build_web)
        case "start":
            rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
            comfy.start(port, a.open, rest)
        case "dev":
            comfy.dev(port)
        case "restart":
            comfy.restart(port, a.force)
        case "monitor":
            from . import monitor

            up = ports.running(port)
            if not up:
                ui.die(f"no ZenTrainer answering on :{ports.configured(port)}; start it with `zt start` (or `zt`)")
            monitor.run(up)
        case "extras":
            comfy.extras(a.groups)
        case "update":
            if a.ref:
                comfy.repin(a.ref)
                return
            found = update.check()
            if found and not update.apply(found, sync=False):
                return
            if not found:
                ui.ok("ZenTrainer is up to date")
            comfy.setup(torch=repo.local().get("torch"))
        case "attach":
            comfy.attach(a.comfyui)
        case "doctor":
            comfy.doctor()
        case "web":
            from . import web

            web.ensure(force=a.build)
        case "train":
            api.train(a.spec, a.profile, a.set, a.end, a.follow, port, **follow)
        case "runs":
            api.runs(a.json, port)
        case "spec":
            api.spec(a.run, port)
        case "follow":
            api.follow(a.run, port=port, **follow)
        case "pause" | "unpause" | "stop" | "save":
            api.control(a.cmd, a.run, port)
        case "resume":
            api.resume(a.run, a.set, a.follow, port, **follow)
        case "sample":
            api.sample(a.run, a.prompt, a.base, a.size, a.seed, port)
        case "profile":
            api.profile(a.run, a.lines, port)
        case "shrink":
            comfy.shrink(a.paths, a.dtype, a.dry_run)
        case "queue":
            api.queue(a.action, port)
        case "caption":
            api.caption(a.dataset, a.model, a.style, a.set_name, a.all, a.follow, port)


def main() -> None:
    a = parser().parse_args()
    try:
        dispatch(a)
    except subprocess.CalledProcessError as e:
        ui.die(f"command failed ({e.returncode}): {' '.join(map(str, e.cmd))}")
    except KeyboardInterrupt:
        ui.console.print()
        sys.exit(130)
