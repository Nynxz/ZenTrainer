"""What only this install can check about a spec, when a run is submitted: that the files it names
are here, its dataset folders exist, and the sampler, scheduler and text encoder type are ones this
ComfyUI has. spec.validate() checks the spec's own rules anywhere; these need ComfyUI, and without
them a typo in a file name fails when the run loads, or at its first sample hundreds of steps in."""

from __future__ import annotations

from .. import paths
from ..spec import _near


def _one_of(field: str, value: str | None, options, what: str) -> str | None:
    if not value or value in options:
        return None
    near = _near(value, options)
    return f"{field}: no {what} {value!r} here" + (f"; did you mean {near!r}?" if near else "")


def problems(spec) -> list[str]:
    """Each problem with a TrainSpec on this install (an empty list when there's none)."""
    import comfy.samplers
    import comfy.sd
    import folder_paths
    import nodes

    def files(folder: str) -> list[str]:
        try:
            return folder_paths.get_filename_list(folder)
        except KeyError:
            return []

    out: list[str | None] = []
    m, s, n = spec.model, spec.samples, spec.network
    if m.kind == "checkpoint":
        out.append(_one_of("model.ckpt_name", m.ckpt_name, files("checkpoints"), "checkpoint"))
        out.append(_one_of("samples.model_name", s.model_name, files("checkpoints"), "checkpoint"))
    else:
        vaes = list(dict.fromkeys([*files("vae"), *nodes.VAELoader.vae_list(nodes.VAELoader)]))
        # what CLIPLoader looks the type up in: an unknown one quietly loads as Stable Diffusion's
        clip_types = [t.lower() for t in comfy.sd.CLIPType.__members__]
        out += [_one_of("model.unet_name", m.unet_name, files("diffusion_models"), "diffusion model"),
                _one_of("model.clip_name", m.clip_name, files("text_encoders"), "text encoder"),
                _one_of("model.clip_type", m.clip_type, clip_types, "text encoder type"),
                _one_of("model.vae_name", m.vae_name, vaes, "VAE"),
                _one_of("model.audio_vae_name", m.audio_vae_name, vaes, "VAE"),
                _one_of("samples.model_name", s.model_name, files("diffusion_models"), "diffusion model")]
    loras = files("loras")
    for field, name in (("network.init_from", n.init_from), ("network.training_adapter", n.training_adapter),
                        ("network.base_lora", n.base_lora), ("samples.turbo_lora", s.turbo_lora)):
        out.append(_one_of(field, name, loras, "LoRA"))
    # as they are now: custom nodes add samplers and schedulers after ComfyUI's own lists are made
    samplers = list(dict.fromkeys([*comfy.samplers.KSampler.SAMPLERS, *comfy.samplers.SAMPLER_NAMES]))
    schedulers = list(comfy.samplers.SCHEDULER_HANDLERS)
    out.append(_one_of("samples.sampler", s.sampler, samplers, "sampler"))
    out.append(_one_of("samples.scheduler", s.scheduler, schedulers, "scheduler"))
    if spec.objective.timesteps == "scheduler":
        out.append(_one_of("objective.timestep_scheduler", spec.objective.timestep_scheduler, schedulers, "scheduler"))
    for field, folder in (("dataset.folder", spec.dataset.folder), ("validation.dataset", spec.validation.dataset)):
        if folder and not paths.find_dataset(folder).is_dir():
            out.append(f"{field}: no dataset {folder!r} in {paths.datasets_root()}")
    return [p for p in out if p]
