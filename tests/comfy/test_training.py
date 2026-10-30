"""Training through a real ComfyUI, one model family at a time (smoke.py), on CPU.

The ComfyUI is ZT_COMFYUI, else this checkout's comfyui/; its Python is ZT_COMFYUI_PYTHON, else
its .venv's, else this one. Without a ComfyUI these skip (CI's `engine` job); the `comfyui` job
runs them on the pinned ComfyUI and, nightly, on its latest.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMFYUI = Path(os.environ.get("ZT_COMFYUI") or ROOT / "comfyui")
FAMILIES = ["moss", "krea2", "fibo2", "flux3", "h3", "anima", "flux2"]  # smoke.py's FAMILIES


def _python() -> str:
    if os.environ.get("ZT_COMFYUI_PYTHON"):
        return os.environ["ZT_COMFYUI_PYTHON"]
    for venv in (COMFYUI / ".venv" / "bin" / "python", COMFYUI / ".venv" / "Scripts" / "python.exe"):
        if venv.is_file():
            return str(venv)
    return sys.executable


pytestmark = pytest.mark.skipif(not (COMFYUI / "comfy" / "sd.py").is_file(), reason=f"no ComfyUI at {COMFYUI}")


@pytest.mark.parametrize("family", FAMILIES)
def test_trains_samples_saves_and_resumes(family, tmp_path):
    r = subprocess.run([_python(), str(Path(__file__).with_name("smoke.py")), family, str(COMFYUI), str(tmp_path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    # the patched model computes what ComfyUI's code does (a patch gone stale drifts)
    assert out["drift"] < 1e-5, out
    assert out["status"] == "done" and out["steps"] == 4 and math.isfinite(out["loss"]), out
    assert out["trained"], out  # the LoRA moved off its zero start
    assert out["patched"] > 0 and out["unloaded"] == [], out  # stock ComfyUI loads every key
    assert out["samples"] and all(Path(p).is_file() for p in out["samples"]), out
    # output.keep "val_loss", 1: validation on each save (1, 2, 3); the best and the newest (3, what
    # a resume needs) stay, the other one goes, files and all; the final stays
    assert out["validated"][:4] == [0, 1, 2, 3], out
    assert sorted(out["kept"]) == sorted({out["best"], 3}) and out["kept_files"], out
    assert sorted(out["pruned"] + out["kept"]) == [1, 2, 3] and not any(out["pruned_files"]), out
    assert out["resumed"] == "done" and out["resumed_steps"] == 6 and math.isfinite(out["resumed_loss"]), out
