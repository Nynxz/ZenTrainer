"""What a caller who can reach ZenTrainer's API (it has no login, like ComfyUI's) can't do: read,
write or delete outside its folders, name paths anywhere from another machine, or get the HF token
sent elsewhere. And what a shared settings file can't point at."""

import asyncio
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from zentrainer import paths, sharing


def _req(remote="127.0.0.1", host="127.0.0.1:8189", **headers):
    return types.SimpleNamespace(remote=remote, headers={"Host": host, **headers})


def test_only_a_request_from_this_machine_counts_as_local():
    from zentrainer.server.routes.common import is_local

    assert is_local(_req())
    assert is_local(_req(host="localhost:8189")) and is_local(_req(remote="::1", host="[::1]:8189"))
    assert not is_local(_req(remote="192.168.1.20", host="192.168.1.5:8189"))  # someone on the LAN
    assert not is_local(_req(**{"X-Forwarded-For": "203.0.113.7"}))  # a tunnel connects from 127.0.0.1
    assert not is_local(_req(**{"CF-Connecting-IP": "203.0.113.7"}))
    assert not is_local(_req(host="attacker.example:8189"))  # DNS rebinding: a page whose name points here
    assert not is_local(_req(host=""))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    inp, ds = tmp_path / "input", tmp_path / "datasets"
    for d in (inp, ds / "cats", tmp_path / "home" / ".ssh", tmp_path / "elsewhere"):
        d.mkdir(parents=True)
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(inp)
    fp.get_output_directory = lambda: str(tmp_path / "output")
    fp.get_temp_directory = lambda: str(tmp_path / "temp")
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setattr(paths, "dataset_roots", lambda: [ds, inp])
    monkeypatch.setattr(paths, "find_dataset", lambda name: ds / name)
    return types.SimpleNamespace(tmp=tmp_path, inp=inp, ds=ds)


def test_files_view_serves_media_only_and_nothing_hidden(roots):
    Image.new("RGB", (8, 8)).save(roots.ds / "cats" / "a.png")
    (roots.ds / "cats" / "notes.txt").write_text("x")
    (roots.ds / ".ssh").mkdir()
    (roots.ds / ".ssh" / "id_rsa.png").write_text("key")
    assert paths.resolve_view("input", "cats", "a.png") == roots.ds / "cats" / "a.png"
    with pytest.raises(ValueError, match="isn't a picture"):
        paths.resolve_view("input", "cats", "notes.txt")  # the datasets folder can be anywhere: only media
    with pytest.raises(ValueError, match="hidden"):
        paths.resolve_view("input", ".ssh", "id_rsa.png")
    with pytest.raises(ValueError, match="hidden"):
        paths.resolve_view("output", "zentrainer", ".env.png")


def test_a_whole_drive_or_a_home_folder_is_refused_as_a_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home" / "me"))
    assert paths._too_broad(Path(Path.cwd().anchor))  # the root (or a drive)
    assert paths._too_broad(tmp_path / "home" / "me") and paths._too_broad(tmp_path / "home")  # it, or above it
    assert paths._too_broad(tmp_path / "home" / "me" / "datasets") is None  # a folder of its own


def test_segment_results_take_a_job_id_never_a_path():
    from zentrainer.datasets import segment

    for bad in ("/home/user/secret", "../../x", "..\\..\\x", "seg-1234"):
        with pytest.raises(ValueError):
            segment._out(bad)


def test_paths_outside_zentrainers_folders_are_named(roots):
    Image.new("RGB", (8, 8)).save(roots.ds / "cats" / "a.png")
    far = roots.tmp / "elsewhere" / "eval.png"
    Image.new("RGB", (8, 8)).save(far)
    ok = {"dataset": {"folder": "cats"}, "samples": {"prompts": [{"prompt": "x", "control": "a.png"}]}}
    assert paths.outside_folders(ok) == []
    assert paths.outside_folders({"dataset": {"folder": str(roots.tmp / "elsewhere")}})  # a folder anywhere
    assert paths.outside_folders({"dataset": {"folder": "../elsewhere"}})  # or climbing out
    assert paths.outside_folders({"dataset": {"folder": "cats"}, "validation": {"dataset": "/etc"}})
    far_input = {"dataset": {"folder": "cats"}, "samples": {"prompts": [{"prompt": "x", "control": str(far)}]}}
    assert paths.outside_folders(far_input) == [f"sample 1's input {str(far)!r}"]
    assert paths.outside_folders(ok, [{"prompt": "y", "control": "../../elsewhere/eval.png"}])  # a test sample's
    assert paths.outside_folders(ok, [{"prompt": "y", "control": "missing.png"}]) == []  # the dataset's, if it were there


def test_a_shared_settings_file_points_at_nothing_on_this_machine():
    spec = {"spec_version": 2, "name": "theirs", "dataset": {"folder": "/their/data"},
            "validation": {"dataset": "/home/them/.ssh", "every": 100},
            "output": {"lora_dir": "/their/loras"}, "network": {"init_from": "theirs.safetensors"},
            "samples": {"prompts": [{"prompt": "x", "control": "/their/pic.png"}]}}
    got = sharing.read(sharing.export(spec))["spec"]
    assert not got.get("dataset", {}).get("folder")
    assert not got.get("validation", {}).get("dataset")
    assert not got.get("output", {}).get("lora_dir") and not got.get("network", {}).get("init_from")
    assert all(not (p.get("control") if isinstance(p, dict) else None) for p in got.get("samples", {}).get("prompts", []))


def test_the_hf_token_goes_only_to_hugging_face_and_not_on_redirects():
    from zentrainer.models.downloads import _hf_request

    auth = {"Authorization": "Bearer hf_secret", "User-Agent": "zt"}
    hf = _hf_request("https://huggingface.co/org/repo/resolve/main/x.safetensors", auth)
    assert hf.unredirected_hdrs.get("Authorization") == "Bearer hf_secret"  # sent, but not carried to its CDN
    assert "Authorization" not in hf.headers
    other = _hf_request("https://files.example.com/x.safetensors", auth)  # a template's file elsewhere
    assert not other.has_header("Authorization")


def test_an_upload_over_the_limit_is_refused_and_drained():
    from zentrainer.server.routes.common import read_upload

    class Part:
        def __init__(self, size):
            self.left = size

        async def read_chunk(self, n):
            take = min(n, self.left)
            self.left -= take
            return b"x" * take

    small, big = Part(3000), Part(5000)
    assert asyncio.run(read_upload(small, limit=4096)) == b"x" * 3000
    assert asyncio.run(read_upload(big, limit=4096)) is None and big.left == 0  # drained for the next part


def test_a_training_state_that_holds_code_is_not_loaded(tmp_path):
    import pickle

    import torch

    class Boom:
        def __reduce__(self):
            return (print, ("ran code",))

    plain, evil = tmp_path / "plain.pt", tmp_path / "evil.pt"
    torch.save({"step": 5, "optimizer": {"state": {0: {"exp_avg": torch.zeros(2)}}}, "rng": torch.ByteTensor(4)}, plain)
    torch.save({"step": 5, "x": Boom()}, evil)
    assert torch.load(plain, weights_only=True)["step"] == 5  # what a state holds loads
    with pytest.raises(pickle.UnpicklingError):
        torch.load(evil, weights_only=True)  # what trainer._restore now uses: refused, not run
