"""Load the models a run needs, through ComfyUI's own loaders.

The diffusion model is loaded fresh and non-dynamic (`disable_dynamic=True`): the
harness patches its forwards for gradient checkpointing and attaches adapters, so
it must not share module objects with whatever the server has cached for inference.
Text encoder and VAE are frozen and only used for preprocessing/sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..spec import ModelSpec


@dataclass
class Models:
    patcher: Any  # comfy.model_patcher.ModelPatcher (the one being trained)
    _clip: Any  # comfy.sd.CLIP, or None until first used (see `clip`)
    vae: Any  # comfy.sd.VAE
    _clip_loader: Any = None
    audio_vae: Any = None  # comfy.sd.VAE for a joint audio-video model's sound, when the spec names one

    @property
    def clip(self):
        """The text encoder, loaded on first use: a run whose captions and sample prompts are
        all cached never loads it (for the big VLM encoders that's minutes and tens of GB)."""
        if self._clip is None and self._clip_loader is not None:
            self._clip = self._clip_loader()
        return self._clip

    @clip.setter
    def clip(self, value):
        self._clip = value

    def release_clip(self) -> None:
        """Free the loaded text encoder; it loads again if something needs it later (a manual
        sample's new prompt). A checkpoint's built-in encoder can't be reloaded alone, so it's
        only freed for split models."""
        if self._clip_loader is not None:
            self._clip = None


_WEIGHT_DTYPES = {"fp8_e4m3fn": "float8_e4m3fn", "fp8_e4m3fn_fast": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}


def load_text(spec: ModelSpec) -> Models | None:
    """Only a split model's text encoder, loaded on first use: the run's text is encoded with it
    before the diffusion model loads (Trainer._text_first). None for a checkpoint, whose encoder
    is inside it."""
    if spec.kind != "split":
        return None
    import nodes

    return Models(None, None, None, lambda: nodes.CLIPLoader().load_clip(spec.clip_name, spec.clip_type)[0])


def load(spec: ModelSpec) -> Models:
    import comfy.sd
    import folder_paths
    import nodes
    import torch

    if spec.kind == "checkpoint":
        path = folder_paths.get_full_path_or_raise("checkpoints", spec.ckpt_name)
        patcher, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"), disable_dynamic=True,
        )
        return Models(patcher, clip, vae)

    opts: dict[str, Any] = {}
    if spec.unet_weight_dtype in _WEIGHT_DTYPES:
        opts["dtype"] = getattr(torch, _WEIGHT_DTYPES[spec.unet_weight_dtype])
        if spec.unet_weight_dtype.endswith("_fast"):
            opts["fp8_optimizations"] = True
    path = folder_paths.get_full_path_or_raise("diffusion_models", spec.unet_name)
    patcher = comfy.sd.load_diffusion_model(path, model_options=opts, disable_dynamic=True)
    vae = nodes.VAELoader().load_vae(spec.vae_name)[0]
    audio_vae = nodes.VAELoader().load_vae(spec.audio_vae_name)[0] if spec.audio_vae_name else None
    return Models(patcher, None, vae, lambda: nodes.CLIPLoader().load_clip(spec.clip_name, spec.clip_type)[0], audio_vae)
