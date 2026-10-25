"""zt keeps a user's checkout safe: an update only fast-forwards a clean one, the web app is the only
thing it replaces (and only by a complete build), and it finds a free port rather than crash.
Everything runs on throwaway git repositories under tmp_path."""

import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("rich")

from zt import api, comfy, ports, repo, update, web  # noqa: E402

LOCK = {"comfyui": {"repo": "x", "ref": "x", "describe": "x"}, "python": "3.12",
        "torch": {"version": "2.0", "variants": {"cpu": {}}}, "port": 8189}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clones(tmp_path):
    """An upstream, a maintainer's clone that pushes to it, and the user's clone zt works in."""
    up, dev, user = tmp_path / "up.git", tmp_path / "dev", tmp_path / "user"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(up))
    git(tmp_path, "clone", "-q", str(up), str(dev))
    (dev / "comfy.lock").write_text(json.dumps(LOCK))
    (dev / "web").mkdir()
    (dev / "web" / "index.html").write_text("v1")
    (dev / "README.md").write_text("hi\n")
    git(dev, "add", "-A")
    git(dev, "commit", "-qm", "first")
    git(dev, "push", "-q", "origin", "HEAD:main")
    git(tmp_path, "clone", "-q", str(up), str(user))
    repo.set_root(user)
    # the user's own things, which nothing may touch
    (user / "comfyui" / "models").mkdir(parents=True)
    (user / "comfyui" / "models" / "mine.safetensors").write_text("precious")
    (user / ".zentrainer").mkdir()
    (user / ".zentrainer" / "local.json").write_text('{"port": 8189}')
    yield up, dev, user
    repo.set_root(Path(__file__).resolve().parents[2])


def upstream_commit(dev: Path, msg: str = "more") -> None:
    with open(dev / "README.md", "a") as f:
        f.write(msg + "\n")
    git(dev, "commit", "-qam", msg)
    git(dev, "push", "-q", "origin", "HEAD:main")


def untouched(user: Path) -> bool:
    return ((user / "comfyui" / "models" / "mine.safetensors").read_text() == "precious"
            and (user / ".zentrainer" / "local.json").read_text() == '{"port": 8189}')


def test_an_update_fast_forwards_a_clean_checkout(clones):
    _, dev, user = clones
    upstream_commit(dev, "new feature")
    found = update.check()
    assert found and found[1][0].endswith("new feature")
    assert update.apply(found, sync=False)
    assert git(user, "log", "-1", "--format=%s") == "new feature"
    assert untouched(user)


def test_an_update_leaves_local_changes_alone(clones):
    _, dev, user = clones
    upstream_commit(dev)
    (user / "README.md").write_text("my edit\n")
    head = git(user, "rev-parse", "HEAD")
    assert not update.apply(update.check(), sync=False)
    assert git(user, "rev-parse", "HEAD") == head and (user / "README.md").read_text() == "my edit\n"
    assert untouched(user)


def test_an_update_leaves_local_commits_alone(clones):
    _, dev, user = clones
    upstream_commit(dev)
    (user / "notes.txt").write_text("mine")
    git(user, "add", "notes.txt")
    git(user, "commit", "-qm", "my own commit")
    head = git(user, "rev-parse", "HEAD")
    assert not update.apply(update.check(), sync=False)
    assert git(user, "rev-parse", "HEAD") == head


def test_nothing_new_is_nothing_to_do(clones):
    assert update.check() is None


def test_the_prebuilt_web_app_is_used_only_when_it_matches(clones, monkeypatch, tmp_path):
    up, dev, user = clones
    monkeypatch.setattr(web.shutil, "which", lambda name, *a, **k: None if name == "pnpm" else "x")
    want = web.source()

    def publish(source: str) -> None:
        build = tmp_path / f"build-{source[:6]}"
        build.mkdir()
        (build / "index.html").write_text(f"built from {source}")
        (build / "SOURCE").write_text(source + "\n")
        git(build, "init", "-q", "-b", "web-dist")
        git(build, "add", "-A")
        git(build, "commit", "-qm", "build")
        git(build, "push", "-qf", str(up), "HEAD:refs/heads/web-dist")

    publish("0" * 40)  # a build of some other version: not used
    web.ensure()
    assert not (repo.DIST / "index.html").exists()
    publish(want)
    web.ensure()
    assert (repo.DIST / "index.html").read_text() == f"built from {want}"
    assert web.built_from() == want and not (repo.DIST / "SOURCE").exists()
    assert not (repo.WEB / ".dist-new").exists() and not (repo.WEB / ".dist-old").exists()
    assert untouched(user)


def test_a_taken_port_moves_to_the_next_free_one(clones):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        taken = s.getsockname()[1]
        (repo.LOCAL).write_text(json.dumps({"port": taken}))
        assert ports.taken(taken)
        got = ports.free()
        assert got != taken and not ports.taken(got)
        with pytest.raises(SystemExit):  # a --port given on the command line is kept to
            ports.free(taken)


@pytest.mark.skipif(os.name == "nt", reason="TIME_WAIT doesn't block a bind on Windows")
def test_a_port_a_server_just_let_go_of_is_free():
    # the server closes its side first, as one that stops does: its connection lingers in TIME_WAIT.
    # It sets SO_REUSEADDR as ComfyUI's (aiohttp's) does; Linux lets a bind through only when both do
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        conn, _ = server.accept()
        conn.close()
    client.close()
    assert not ports.taken(port)


def test_follow_reads_only_new_rows_from_either_server():
    rows = [{"step": i} for i in range(1, 6)]
    assert api.new_rows({"metrics": rows[3:], "metrics_count": 5}, 3) == (rows[3:], 5)  # with ?since
    assert api.new_rows({"metrics": rows}, 3) == (rows[3:], 5)  # an older server sends them all
    assert api.new_rows({"metrics": rows}, None) == (rows, 5)


def test_settings_on_the_command_line():
    spec = api.apply_sets({"optim": {"lr": 1}}, ["optim.lr=0.0002", "samples.prompts=[\"a cat\"]", "name=plain"])
    assert spec == {"optim": {"lr": 0.0002}, "samples": {"prompts": ["a cat"]}, "name": "plain"}


CLINFO = """Number of platforms:\t\t\t\t 1
  Platform Vendor:\t\t\t\t Advanced Micro Devices, Inc.
  Device Type:\t\t\t\t\t CL_DEVICE_TYPE_GPU
  Board name:\t\t\t\t\t AMD Radeon RX 7900 XTX
  Global memory size:\t\t\t\t 25753026560
  Name:\t\t\t\t\t\t gfx1100
  Vendor:\t\t\t\t\t Advanced Micro Devices, Inc.
  Device Type:\t\t\t\t\t CL_DEVICE_TYPE_CPU
  Name:\t\t\t\t\t\t AMD Ryzen 9 7950X
  Vendor:\t\t\t\t\t AuthenticAMD
"""


def test_amd_gpus_are_found_from_the_driver_alone(monkeypatch, tmp_path):
    assert comfy.parse_clinfo(CLINFO) == [("AMD Radeon RX 7900 XTX", 25753026560, "gfx1100")]
    for node, props in {"0": "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\n",
                        "1": "simd_count 192\ngfx_target_version 90010\n",
                        "2": "simd_count 96\ngfx_target_version 120001\n"}.items():
        (tmp_path / node / "mem_banks" / "0").mkdir(parents=True)
        (tmp_path / node / "properties").write_text(props)
        (tmp_path / node / "mem_banks" / "0" / "properties").write_text(f"heap_type 1\nsize_in_bytes {int(node) << 34}\n")
    monkeypatch.setattr(comfy, "KFD_NODES", tmp_path)
    monkeypatch.setattr(repo, "WINDOWS", False)
    assert comfy.amd_gpus() == [("AMD Radeon (gfx90a)", 1 << 34, "gfx90a"), ("AMD Radeon (gfx1201)", 2 << 34, "gfx1201")]


def test_amd_torch_comes_with_each_gpus_kernels(monkeypatch):
    req = {"index": "https://amd/", "devices": ["gfx1100", "gfx1201"], "torchvision": "0.28.0", "torchaudio": "2.11.0.3"}
    found = []
    monkeypatch.setattr(comfy, "amd_gpus", lambda: found)
    found[:] = [("a", 0, "gfx1201"), ("b", 0, "gfx1100"), ("c", 0, "gfx1201")]
    assert comfy.torch_packages("rocm10.1.0", req, "2.13.0") == [
        "torch[device-gfx1100,device-gfx1201]==2.13.0+rocm10.1.0",
        "torchvision[device-gfx1100,device-gfx1201]==0.28.0+rocm10.1.0", "torchaudio==2.11.0.3+rocm10.1.0",
        "--index-url", "https://pypi.org/simple", "--extra-index-url", "https://amd/"]
    found[:] = []  # --torch rocm... with no AMD GPU seen: every GPU's kernels
    assert comfy.torch_packages("rocm10.1.0", req, "2.13.0")[0] == "torch[device-all]==2.13.0+rocm10.1.0"
    found[:] = [("old", 0, "gfx906")]
    with pytest.raises(SystemExit):
        comfy.torch_packages("rocm10.1.0", req, "2.13.0")
    assert comfy.torch_packages("cu130", {}, "2.13.0") == [
        "torch==2.13.0", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu130"]


def test_a_stuck_clinfo_means_no_amd_gpu(monkeypatch):
    def run(cmd, **kw):
        assert kw["timeout"] == 10
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(repo, "WINDOWS", True)
    monkeypatch.setattr(comfy.shutil, "which", lambda name, **kw: "clinfo")
    monkeypatch.setattr(comfy.subprocess, "run", run)
    assert comfy.amd_gpus() == []


def test_comfyui_runs_on_the_amd_gpu_torch_has_kernels_for(monkeypatch):
    req = {"index": "https://amd/", "devices": ["gfx1100", "gfx1200"]}
    found, hip = [], []
    monkeypatch.setattr(comfy, "amd_gpus", lambda: found)
    monkeypatch.setattr(comfy, "hip_devices", lambda py: hip)
    found[:] = [("r", 0, "gfx1200"), ("igpu", 0, "gfx1036")]
    hip[:] = ["gfx1036", "gfx1200"]
    assert comfy.amd_device(req, Path("py")) == 1
    hip[:] = ["gfx1200", "gfx1036"]  # still kept off the unsupported one, though it isn't the default
    assert comfy.amd_device(req, Path("py")) == 0
    hip[:] = ["gfx1200"]  # HIP doesn't list it at all
    assert comfy.amd_device(req, Path("py")) is None
    hip[:] = []  # torch couldn't say: left to the user
    assert comfy.amd_device(req, Path("py")) is None
    found[:] = [("a", 0, "gfx1100"), ("b", 0, "gfx1200")]
    monkeypatch.setattr(comfy, "hip_devices", lambda py: pytest.fail("no need to ask torch"))
    assert comfy.amd_device(req, Path("py")) is None
    assert comfy.amd_device({}, Path("py")) is None
    monkeypatch.setattr(repo, "local", lambda: {"cuda_device": 1})
    assert comfy.args(8189) == ["--port", "8189", "--cuda-device", "1"]
