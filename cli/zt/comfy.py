"""The managed ComfyUI: setting it up at the pinned commit, starting it, and checking on it.

It lives in ./comfyui (pinned by comfy.lock, own .venv) with its own input/, output/ and user/
folders, as any ComfyUI does: all gitignored. Models are never copied: --models-from points it at
existing ComfyUI installs or model folders.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.table import Table

from . import ports, repo, ui, web


def uv() -> str:
    """uv: on PATH, or where its installer puts it (the start scripts install it)."""
    found = shutil.which("uv") or shutil.which("uv", path=str(Path.home() / ".local" / "bin"))
    if not found:
        ui.die("uv not found. Run start-linux.sh or start-windows.bat (they install it), or see https://docs.astral.sh/uv/")
    return found


def need(tool: str, hint: str) -> None:
    if not shutil.which(tool):
        ui.die(f"{tool} not found. {hint}")


def link(target: Path, at: Path) -> None:
    """Symlink `at` -> `target`; replaces a stale link, refuses to clobber a real folder."""
    if at.is_symlink():
        at.unlink()
    elif at.exists():
        ui.die(f"{at} exists and is not a link; remove it or pick another name")
    at.parent.mkdir(parents=True, exist_ok=True)
    try:
        at.symlink_to(target, target_is_directory=True)
    except OSError as e:
        if repo.WINDOWS:  # symlinks need developer mode; a junction doesn't
            ui.run(["cmd", "/c", "mklink", "/J", str(at), str(target)])
        else:
            ui.die(f"could not link {at} -> {target}: {e}")
    ui.ok(f"linked {at.relative_to(repo.ROOT) if at.is_relative_to(repo.ROOT) else at} -> {target}")


# ---- the PyTorch build for this machine ---------------------------------------------------------

def _nvml():
    """NVML (NVIDIA's library, part of the driver) ready to query, or None without an NVIDIA driver."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def driver_cuda() -> tuple[int, int] | None:
    """The newest CUDA the NVIDIA driver supports, e.g. (13, 0)."""
    nvml = _nvml()
    if not nvml:
        return None
    try:
        v = nvml.nvmlSystemGetCudaDriverVersion_v2()  # 13040 -> 13.4
        return v // 1000, (v % 1000) // 10
    finally:
        nvml.nvmlShutdown()


def gpus() -> list[tuple[str, int]]:
    """Each NVIDIA and AMD GPU's name and memory in bytes."""
    nvml = _nvml()
    if not nvml:
        return [(name, mem) for name, mem, _ in amd_gpus()]
    try:
        out = []
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(h)
            out.append((name.decode() if isinstance(name, bytes) else name, nvml.nvmlDeviceGetMemoryInfo(h).total))
        return out + [(name, mem) for name, mem, _ in amd_gpus()]
    finally:
        nvml.nvmlShutdown()


KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def amd_gpus() -> list[tuple[str, int, str]]:
    """Each AMD GPU's name, memory in bytes and target (gfx1100, ...), from its display driver alone:
    ROCm's PyTorch wheels bring their own runtime, so no ROCm install is needed or looked for."""
    if repo.WINDOWS:
        clinfo = shutil.which("clinfo")  # the Adrenalin driver installs it
        try:  # it also runs on NVIDIA machines, where a stuck OpenCL driver mustn't hang setup
            out = subprocess.run([clinfo], capture_output=True, text=True, timeout=10).stdout if clinfo else ""
        except subprocess.TimeoutExpired:
            out = ""
        return parse_clinfo(out)
    out = []
    for node in sorted(KFD_NODES.glob("*"), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        props = parse_kfd_properties((node / "properties").read_text() if (node / "properties").is_file() else "")
        if not props.get("simd_count") or not props.get("gfx_target_version"):
            continue  # a CPU node
        v = props["gfx_target_version"]  # major*10000 + minor*100 + stepping, the stepping printed in hex
        gfx = "gfx%d%d%x" % (v // 10000, (v // 100) % 100, v % 100)
        mem = sum(parse_kfd_properties(p.read_text()).get("size_in_bytes", 0)
                  for p in (node / "mem_banks").glob("*/properties"))
        out.append((f"AMD Radeon ({gfx})", mem, gfx))
    return out


def parse_kfd_properties(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (line.split() for line in text.splitlines() if len(line.split()) == 2) if v.isdigit()}


def parse_clinfo(text: str) -> list[tuple[str, int, str]]:
    out = []
    for block in re.split(r"\n\s*Device Type:", text)[1:]:
        gfx = re.search(r"^\s*Name:\s*(gfx[0-9a-f]+)", block, re.M)
        if not block.lstrip().startswith("CL_DEVICE_TYPE_GPU") or "Advanced Micro Devices" not in block or not gfx:
            continue
        board = re.search(r"^\s*Board name:\s*(.+)$", block, re.M)
        mem = re.search(r"^\s*Global memory size:\s*(\d+)", block, re.M)
        out.append((board[1].strip() if board else "AMD Radeon", int(mem[1]) if mem else 0, gfx[1]))
    return out


def pick_torch(variants: dict) -> str:
    cuda = driver_cuda()
    if cuda:
        for name, req in variants.items():
            if name.startswith("cu") and "min_driver_cuda" in req:
                major, minor = (int(x) for x in req["min_driver_cuda"].split("."))
                if cuda >= (major, minor):
                    return name
        ui.die(f"NVIDIA driver reports CUDA {cuda[0]}.{cuda[1]}, older than any pinned torch build; update the driver")
    rocm = next((v for v in variants if v.startswith("rocm")), None)
    if rocm and amd_gpus():
        return rocm
    return "cpu"


def torch_packages(variant: str, req: dict, version: str) -> list[str]:
    """What uv installs for a torch build. AMD's index serves one torch for every GPU, and each GPU's
    kernels as an extra (device-gfx1100, ...). Its torchvision and torchaudio don't name the torch they
    were built for, so comfy.lock pins them. uv prefers the extra index, so everything AMD serves comes
    from AMD and only the rest from PyPI."""
    if "index" not in req:
        return [f"torch=={version}", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{variant}"]
    targets = [gfx for _, _, gfx in amd_gpus()]
    devices = sorted({f"device-{gfx}" for gfx in targets if gfx in req["devices"]})
    if targets and not devices:
        ui.die(f"this AMD GPU ({', '.join(targets)}) has no ROCm PyTorch build; use --torch cpu to set up without it")
    extra = f"[{','.join(devices or ['device-all'])}]"
    return [f"torch{extra}=={version}+{variant}", f"torchvision{extra}=={req['torchvision']}+{variant}",
            f"torchaudio=={req['torchaudio']}+{variant}", "--index-url", "https://pypi.org/simple",
            "--extra-index-url", req["index"]]


def amd_device(req: dict, py: Path) -> int | None:
    """With an AMD GPU that AMD's torch has no kernels for beside one it has (an older APU's graphics
    next to a Radeon, say), the supported one's index for ComfyUI's --cuda-device, which otherwise
    takes device 0 whichever that is. The index comes from torch, as HIP's order needn't be clinfo's."""
    if "devices" not in req or all(gfx in req["devices"] for *_, gfx in amd_gpus()):
        return None
    found = hip_devices(py)
    ok = [i for i, gfx in enumerate(found) if gfx in req["devices"]]
    if not ok:
        ui.warn("an AMD GPU here has no ROCm PyTorch kernels, and torch didn't name the one that has: "
                "if ComfyUI picks the wrong one, start it with `zt start -- --cuda-device N`")
        return None
    return ok[0] if len(ok) < len(found) else None


def hip_devices(py: Path) -> list[str]:
    """Each GPU's target (gfx1100, ...) in the order the installed torch numbers them."""
    code = ("import torch; print(*(torch.cuda.get_device_properties(i).gcnArchName.split(':')[0]"
            " for i in range(torch.cuda.device_count())))")
    try:
        return subprocess.run([py, "-I", "-c", code], capture_output=True, text=True, timeout=120).stdout.split()
    except subprocess.TimeoutExpired:
        return []


# ---- where models come from ----------------------------------------------------------------------

def models_yaml(sources: list[Path]) -> Path | None:
    """extra_model_paths.yaml covering each source's model folders: an install's models/, or a
    plain folder of checkpoints/, loras/, ... A ComfyUI install's own yaml is passed to ComfyUI
    separately (see install_configs)."""
    lines = ["# written by zt setup; regenerate with `zt setup --models-from ...`"]
    for i, d in enumerate(sources):
        base = d / "models" if (d / "models").is_dir() else d
        found = [f for f in repo.MODEL_FOLDERS if (base / f).is_dir()]
        if not found:
            continue
        lines.append(f"zentrainer_{i}:")
        lines.append(f"  base_path: {json.dumps(str(base))}")
        lines += [f"  {f}: {f}/" for f in found]
    if len(lines) == 1:
        return None
    path = repo.ROOT / ".zentrainer" / "extra_model_paths.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def install_configs(sources: list[Path]) -> list[str]:
    """Each existing ComfyUI install's own extra_model_paths.yaml, used as-is."""
    return [str(s / "extra_model_paths.yaml") for s in sources
            if (s / "main.py").is_file() and (s / "extra_model_paths.yaml").is_file()]


# ---- setup -------------------------------------------------------------------------------------

def setup(models_from: list[str] | None = None, extra_model_paths: list[str] | None = None, torch: str | None = None,
          zenkit: str | None = None, build_web: bool = False) -> None:
    """Clone the pinned ComfyUI, give it its own environment, link the engine in."""
    need("git", "Install git.")
    uv_bin = uv()
    L = repo.lock()
    ref = L["comfyui"]["ref"]

    # 1. ComfyUI at the pinned commit
    if not (repo.COMFY / ".git").is_dir():
        ui.run(["git", "clone", "--filter=blob:none", L["comfyui"]["repo"], repo.COMFY])
    else:
        ui.run(["git", "-C", repo.COMFY, "fetch", "--quiet", "origin"])
    if repo.git("status", "--porcelain", "--untracked-files=no", cwd=repo.COMFY):
        ui.die(f"{repo.COMFY} has local changes; commit or discard them before re-pinning")
    ui.run(["git", "-C", repo.COMFY, "checkout", "--quiet", ref])

    # 2. its own environment
    variants = L["torch"]["variants"]
    if torch and torch not in variants:  # a build an older comfy.lock pinned
        ui.warn(f"comfy.lock no longer has the {torch} torch build: picking one again")
        torch = None
    variant = torch or pick_torch(variants)
    py = repo.venv_python()
    if not py.is_file():
        ui.run([uv_bin, "venv", "--python", L["python"], repo.COMFY / ".venv"])
    tv = L["torch"]["version"]
    ui.run([uv_bin, "pip", "install", "--python", py, *torch_packages(variant, variants[variant], tv)])
    ui.run([uv_bin, "pip", "install", "--python", py, "-r", repo.COMFY / "requirements.txt"])

    # 3. the engine, and optionally ZenKit
    link(repo.ENGINE, repo.COMFY / "custom_nodes" / "zentrainer")
    if zenkit:
        link(Path(zenkit).expanduser().resolve(), repo.COMFY / "custom_nodes" / "ComfyUI-ZenKit")

    # 4. where models come from (never copied)
    cfg = repo.local()
    if models_from or extra_model_paths:
        sources = [Path(p).expanduser().resolve() for p in models_from or []]
        for s in sources:
            if not s.is_dir():
                ui.die(f"--models-from {s}: not a directory (for a yaml file use --extra-model-paths)")
        yamls = [Path(p).expanduser().resolve() for p in extra_model_paths or []]
        for y in yamls:
            if not y.is_file():
                ui.die(f"--extra-model-paths {y}: no such file")
        gen = models_yaml(sources)
        cfg["extra_model_paths"] = [str(y) for y in yamls] + install_configs(sources) + ([str(gen)] if gen else [])
    cfg.update(torch=variant, ref=ref)
    cfg.pop("cuda_device", None)
    device = amd_device(variants[variant], py)
    if device is not None:
        cfg["cuda_device"] = device
        ui.warn(f"an AMD GPU here has no ROCm PyTorch kernels: ComfyUI will use GPU {device}")
    repo.save_local(cfg)

    # 5. the web app: kept, fetched prebuilt, or built
    web.ensure(force=build_web)

    ui.ok(f"ready: ComfyUI {L['comfyui']['describe']}, torch {tv}+{variant}. Start it with `zt start`.")
    if not cfg.get("extra_model_paths"):
        ui.warn("no model sources configured: add your existing ones with `zt setup --models-from /path/to/ComfyUI`")


def repin(ref: str) -> None:
    """Move comfy.lock's ComfyUI pin to a commit, tag or branch, then set up at it."""
    L = repo.lock()
    L["comfyui"]["ref"] = ref
    if (repo.COMFY / ".git").is_dir():
        ui.run(["git", "-C", repo.COMFY, "fetch", "--quiet", "--tags", "origin"])
        sha = repo.git("rev-parse", ref + "^{commit}", cwd=repo.COMFY)
        if not sha:
            ui.die(f"{ref}: not a commit, tag or branch of ComfyUI")
        L["comfyui"].update(ref=sha, describe=repo.git("describe", "--tags", sha, cwd=repo.COMFY) or sha[:9])
    repo.LOCK.write_text(json.dumps(L, indent=2) + "\n")
    ui.ok(f"comfy.lock -> {L['comfyui']['ref']} ({L['comfyui']['describe']}). Test before committing the lock.")
    setup(torch=repo.local().get("torch"))


def attach(comfyui: str) -> None:
    """Link the engine into an existing ComfyUI instead of the managed one."""
    target = Path(comfyui).expanduser().resolve()
    if not (target / "main.py").is_file():
        ui.die(f"{target} doesn't look like a ComfyUI checkout (no main.py)")
    link(repo.ENGINE, target / "custom_nodes" / "zentrainer")
    ver = target / "comfyui_version.py"
    m = re.search(r'"([^"]+)"', ver.read_text()) if ver.is_file() else None
    ui.ok(f"attached to ComfyUI {m[1] if m else '(unknown version)'}; tested pin is {repo.lock()['comfyui']['describe']}")
    ui.say("restart that ComfyUI, then open /zentrainer/ on its port")


def extras(groups: list[str]) -> None:
    """Optional packages into the managed ComfyUI."""
    if not repo.venv_python().is_file():
        ui.die("no managed ComfyUI yet: run `zt setup` first")
    pkgs = [p for g in groups for p in EXTRAS[g]]
    ui.run([uv(), "pip", "install", "--python", repo.venv_python(), *pkgs])
    ui.ok("installed; restart ComfyUI (`zt restart`) to use them")


EXTRAS = {
    "optimizers": ["prodigyopt", "bitsandbytes", "schedulefree", "came-pytorch"],
}


# ---- running it ----------------------------------------------------------------------------------

def args(port: int) -> list[str]:
    cmd = ["--port", str(port)]  # ComfyUI's own input/, output/, user/ folders
    cfg = repo.local()
    for y in cfg.get("extra_model_paths", []):
        cmd += ["--extra-model-paths-config", y]
    if "cuda_device" in cfg:
        cmd += ["--cuda-device", str(cfg["cuda_device"])]
    return cmd


def _need_setup() -> None:
    if not repo.venv_python().is_file():
        ui.die("no managed ComfyUI yet: run `zt setup` (or just `zt` for the guided setup) first")


def start(port: int | None = None, open_browser: bool = False, rest: list[str] | None = None) -> None:
    """Run the managed ComfyUI in the foreground; if ZenTrainer is already up, say where."""
    _need_setup()
    up = ports.running(port)
    if up:
        url = f"http://127.0.0.1:{up}/zentrainer/"
        ui.ok(f"ZenTrainer is already running at {url}")
        if open_browser:
            ui.open_browser(url)
        return
    port = ports.free(port)
    ports.remember(port)
    url = f"http://127.0.0.1:{port}/zentrainer/"
    ui.say(f"ZenTrainer at {url}  (Ctrl-C stops it)")
    if open_browser:
        ui.open_browser(url)
    cmd = [str(repo.venv_python()), "main.py", *args(port), *(rest or [])]
    os.chdir(repo.COMFY)
    if repo.WINDOWS:
        sys.exit(subprocess.call(cmd))
    os.execv(cmd[0], cmd)


def _launch(port: int, detach: bool) -> subprocess.Popen:
    """The managed ComfyUI in the background, its output in .zentrainer/comfy.log, waited on until
    it answers. detach: it keeps running after zt exits."""
    log = repo.ROOT / ".zentrainer" / "comfy.log"
    log.parent.mkdir(exist_ok=True)
    ports.remember(port)
    proc = subprocess.Popen([str(repo.venv_python()), "main.py", *args(port)], cwd=repo.COMFY, stdout=open(log, "ab"),
                            stderr=subprocess.STDOUT, start_new_session=detach)
    with ui.console.status(f"starting ComfyUI on :{port} (log: {log.relative_to(repo.ROOT)})…"):
        for _ in range(480):
            if ports.alive(port):
                ui.ok(f"ZenTrainer is up at http://127.0.0.1:{port}/zentrainer/")
                return proc
            if proc.poll() is not None:
                ui.die(f"ComfyUI exited ({proc.returncode}); see {log}")
            time.sleep(0.5)
    ui.die(f"ComfyUI didn't come up; see {log}")


def background(port: int | None = None) -> int:
    """The port ZenTrainer runs on, starting the managed ComfyUI in the background if nothing
    answers (it keeps running after zt exits)."""
    up = ports.running(port)
    if up:
        return up
    _need_setup()
    port = ports.free(port)
    _launch(port, detach=True)
    return port


def dev(port: int | None = None) -> None:
    """ComfyUI (reused if already up) plus Vite's dev server: web/ edits show instantly,
    engine (Python) edits need `zt restart`. A ComfyUI started here stops with it."""
    need("pnpm", "Install pnpm: https://pnpm.io/installation")
    comfy = None
    up = ports.running(port)
    if up:
        ui.say(f"using the ComfyUI already running on :{up}")
        port = up
    else:
        _need_setup()
        port = ports.free(port)
        comfy = _launch(port, detach=False)
    if not (repo.WEB / "node_modules").is_dir():
        ui.run(["pnpm", "install"], cwd=repo.WEB)
    ui.say("app with hot reload: http://localhost:5173/zentrainer/")
    try:
        subprocess.call(["pnpm", "dev"], cwd=repo.WEB, env={**os.environ, "COMFY_URL": f"http://127.0.0.1:{port}"})
    except KeyboardInterrupt:
        pass
    finally:
        if comfy and comfy.poll() is None:
            ui.say("stopping ComfyUI")
            comfy.terminate()
            try:
                comfy.wait(15)
            except subprocess.TimeoutExpired:
                comfy.kill()


def restart(port: int | None = None, force: bool = False) -> None:
    port = ports.running(port)
    if not port:
        ui.die("nothing answering; start it with `zt start` or `zt dev`")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/zentrainer/api/restart", method="POST",
                                 data=json.dumps({"force": force}).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        ui.die(json.loads(e.read() or b"{}").get("error", str(e)))
    with ui.console.status("restarting…"):
        time.sleep(2)
        for _ in range(240):
            if ports.alive(port):
                ui.ok(f"back up on :{port}")
                return
            time.sleep(0.5)
    ui.die("ComfyUI didn't come back; check its console")


def doctor() -> None:
    """What's installed, and where models come from."""
    L, cfg = repo.lock(), repo.local()
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()

    def row(name: str, good: bool | None, text: str) -> None:
        mark = {True: "[green]✓[/]", False: "[red]✗[/]", None: "[yellow]![/]"}[good]
        t.add_row(name, f"{mark} {text}")

    t.add_row("pin", f"  {L['comfyui']['describe']}  [dim]({L['comfyui']['ref'][:12]})[/]")
    if (repo.COMFY / ".git").is_dir():
        head = repo.git("rev-parse", "HEAD", cwd=repo.COMFY)
        row("checkout", head == L["comfyui"]["ref"], head[:12] + ("" if head == L["comfyui"]["ref"] else "  differs from the pin: run zt setup"))
    else:
        row("checkout", False, "missing: run zt setup")
    linked = (repo.COMFY / "custom_nodes" / "zentrainer").is_symlink()
    row("engine", linked, "linked" if linked else "not linked")
    own = repo.COMFY / "extra_model_paths.yaml"  # ComfyUI always loads this one itself
    sources = ([f"{own} (auto)"] if own.is_file() else []) + cfg.get("extra_model_paths", [])
    row("models", bool(sources) or None, ", ".join(sources) or "none: put an extra_model_paths.yaml in comfyui/, or zt setup --models-from …")
    have_web = (repo.DIST / "index.html").is_file()
    current = have_web and web.built_from() in ("", web.source())
    row("web", have_web and (current or None), ("built" if current else "built, for an older version") if have_web else "not built")
    uv_bin = shutil.which("uv") or shutil.which("uv", path=str(Path.home() / ".local" / "bin"))
    uv_ver = subprocess.run([uv_bin, "--version"], capture_output=True, text=True).stdout.strip() if uv_bin else ""
    row("uv", bool(uv_bin), f"{uv_ver}  [dim]{uv_bin}[/]" if uv_bin else "not found")
    cuda = driver_cuda()
    if cuda or not amd_gpus():
        row("driver", bool(cuda) or None, "CUDA %d.%d" % cuda if cuda else "no NVIDIA or AMD GPU found")
    for i, (name, mem) in enumerate(gpus()):
        row(f"gpu {i}", True, f"{name}, {mem / 2**30:.0f} GB")
    if repo.venv_python().is_file():
        code = ("import torch;print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'no-cuda',"
                " torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")
        out = subprocess.run([str(repo.venv_python()), "-c", code], capture_output=True, text=True)
        text = out.stdout.strip() or (out.stderr.strip().splitlines() or ["?"])[-1]
        pinned = L["torch"]["version"]
        same = text.startswith(pinned + "+") or text.startswith(pinned + " ")
        if out.returncode == 0 and not same:
            text += f"  (comfy.lock pins {pinned})"
        row("torch", ("no-cuda" not in text and out.returncode == 0 and same) or None, text)
    else:
        row("torch", False, "no venv")
    up = ports.running()
    row("server", bool(up) or None, f"running on :{up}" if up else f"not running (it would use :{ports.configured()})")
    ui.console.print(t)


_SHRINK = r"""
import math, os, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

to = {"bf16": torch.bfloat16, "fp16": torch.float16}[sys.argv[1]]
dry = sys.argv[2] == "1"
files = []
for p in map(Path, sys.argv[3:]):
    files += sorted(p.rglob("*.safetensors")) if p.is_dir() else [p]
saved = 0
for f in files:
    with safe_open(str(f), "pt") as sf:
        meta = sf.metadata()
        keys = list(sf.keys())
        wide = [k for k in keys if sf.get_slice(k).get_dtype() in ("F32", "F64")]
        if not wide:
            continue
        before = f.stat().st_size
        if dry:
            # an fp32 tensor halves; the rest stays
            gain = sum(4 * math.prod(sf.get_slice(k).get_shape()) for k in wide) // 2
            print(f"{f}  {before / 1e6:,.0f} MB -> {(before - gain) / 1e6:,.0f} MB")
            saved += gain
            continue
        out = {k: (sf.get_tensor(k).to(to) if k in wide else sf.get_tensor(k)) for k in keys}
    part = f.with_name(f".{f.name}.part")
    save_file(out, str(part), metadata=meta)
    os.chmod(part, 0o644)
    os.replace(part, f)
    saved += before - f.stat().st_size
    print(f"{f}  {before / 1e6:,.0f} MB -> {f.stat().st_size / 1e6:,.0f} MB")
print(f"{'would save' if dry else 'saved'} {saved / 1e6:,.0f} MB")
"""


def shrink(paths: list[str], dtype: str = "bf16", dry_run: bool = False) -> None:
    """Rewrite LoRA files' fp32 tensors in bf16 (or fp16), in place: half the size, nothing a
    sample shows. Other tensors and the metadata stay. A checkpoint resumed from afterwards
    continues from the rounded weights."""
    if not repo.venv_python().is_file():
        ui.die("no managed ComfyUI venv: run zt setup")
    for p in paths:
        if not Path(p).exists():
            ui.die(f"not found: {p}")
    subprocess.run([str(repo.venv_python()), "-I", "-c", _SHRINK, dtype, "1" if dry_run else "0", *paths], check=True)
