"""runs/preflight.py: what only this install can say about a spec at submit, against ComfyUI's
own lists (faked here): files that aren't there, unknown samplers, schedulers and text encoder
types, missing dataset folders, each with the name it likely meant."""

import enum
import sys
import types

import pytest

from zentrainer import paths
from zentrainer.runs import preflight
from zentrainer.spec import TrainSpec


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    lists = {"checkpoints": ["sdxl.safetensors"], "diffusion_models": ["flux2/klein.safetensors"],
             "text_encoders": ["qwen3_4b.safetensors"], "vae": ["flux2-vae.safetensors"], "loras": ["turbo.safetensors"]}
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda kind: lists[kind]
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSampler = types.SimpleNamespace(SAMPLERS=["euler", "dpmpp_2m"])
    samplers.SAMPLER_NAMES = ["euler", "dpmpp_2m"]
    samplers.SCHEDULER_HANDLERS = {"simple": None, "normal": None, "beta": None}
    sd = types.ModuleType("comfy.sd")
    sd.CLIPType = enum.Enum("CLIPType", "STABLE_DIFFUSION FLUX2 QWEN_IMAGE")
    nodes = types.ModuleType("nodes")
    nodes.VAELoader = types.SimpleNamespace(vae_list=lambda _self: ["flux2-vae.safetensors", "pixel_space"])
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.samplers, comfy_pkg.sd = samplers, sd
    for name, mod in (("folder_paths", fp), ("comfy", comfy_pkg), ("comfy.samplers", samplers), ("comfy.sd", sd), ("nodes", nodes)):
        monkeypatch.setitem(sys.modules, name, mod)
    (tmp_path / "datasets" / "cats").mkdir(parents=True)
    monkeypatch.setattr(paths, "find_dataset", lambda name: tmp_path / "datasets" / name)
    monkeypatch.setattr(paths, "datasets_root", lambda: tmp_path / "datasets")


def spec(**over):
    d = {"spec_version": 2, "model": {"kind": "split", "unet_name": "flux2/klein.safetensors", "clip_name": "qwen3_4b.safetensors",
                                      "clip_type": "flux2", "vae_name": "flux2-vae.safetensors"},
         "dataset": {"folder": "cats"}, "samples": {"sampler": "euler", "scheduler": "simple"}}
    for k, v in over.items():
        d[k] = {**d.get(k, {}), **v}
    return TrainSpec.from_dict(d)


def test_a_spec_this_install_can_run_has_no_problems(comfy):
    assert preflight.problems(spec()) == []


def test_each_problem_is_named_with_what_it_likely_meant(comfy):
    found = preflight.problems(spec(model={"unet_name": "flux2/kelin.safetensors", "clip_type": "qwen_imgae"},
                                    samples={"sampler": "eular", "scheduler": "karras"}, dataset={"folder": "dogs"},
                                    network={"init_from": "turbo"}))
    assert found == ["model.unet_name: no diffusion model 'flux2/kelin.safetensors' here; did you mean 'flux2/klein.safetensors'?",
                     "model.clip_type: no text encoder type 'qwen_imgae' here; did you mean 'qwen_image'?",
                     "network.init_from: no LoRA 'turbo' here",
                     "samples.sampler: no sampler 'eular' here; did you mean 'euler'?",
                     "samples.scheduler: no scheduler 'karras' here",
                     f"dataset.folder: no dataset 'dogs' in {paths.datasets_root()}"]


def test_a_vae_comfyui_makes_itself_is_fine(comfy):
    assert preflight.problems(spec(model={"vae_name": "pixel_space"})) == []
