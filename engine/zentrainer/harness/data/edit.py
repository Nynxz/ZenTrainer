"""Edit training: the model sees an input picture (the pair's _control image) as reference
latents beside the noisy target, and learns to produce the target from it and the caption (the
instruction). The same path ComfyUI's ReferenceLatent node takes, so any family whose model reads
`reference_latents` (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit, ...) trains and samples
this way unchanged."""

from __future__ import annotations

import inspect

import torch


def _reads(model, key: str) -> bool:
    for cls in type(model).__mro__:
        fn = cls.__dict__.get("extra_conds")
        if fn is None:
            continue
        try:
            if key in inspect.getsource(fn):
                return True
        except (OSError, TypeError):
            pass
    return False


def takes_references(model) -> bool:
    """Whether this ComfyUI model takes reference inputs: reference_latents, or MiniMax H3's own
    reference blocks; or MOSS Image, which ZenTrainer's patch teaches them (moss_refs.py)."""
    from ..moss_refs import supports

    return _reads(model, "reference_latents") or is_h3(model) or supports(model)


def is_h3(model) -> bool:
    """MiniMax H3: references (images, videos with sound, audio) and keyframes ride in its own
    conditioning, and its text encoder sees them, so each sample's conditioning is its own."""
    return _reads(model, "minimax_refs")


def method(model, chosen: str = "") -> str | None:
    """How the references join the sequence ("index", "offset", "index_timestep_zero"...): the
    spec's choice, else the model's own default; a model without one (Krea 2) would ignore them,
    so it gets "index" (each reference tagged on the position axis the image doesn't use, the
    NK2E / Kontext-style scheme)."""
    if chosen:
        return chosen
    from ..moss_refs import DEFAULT, supports

    if supports(model):  # ai-toolkit's MOSS-Image Edit layout (ComfyUI's "ai-toolkit" format)
        return DEFAULT
    dm = getattr(model, "diffusion_model", None)
    own = getattr(dm, "default_ref_method", None) or getattr(getattr(dm, "params", None), "default_ref_method", None)
    return None if own else "index"


def with_refs(conditioning: list, refs: list[torch.Tensor], ref_method: str | None = None) -> list:
    """A copy of a CLIP conditioning ([[cond, {...}]]) carrying these raw latents, as the
    ReferenceLatent node adds them (batch dim on, appended to any already there)."""
    out = []
    for cond, opts in conditioning:
        opts = dict(opts)
        opts["reference_latents"] = list(opts.get("reference_latents") or []) + [r if r.ndim == 4 else r[None] for r in refs]
        if ref_method:
            opts["reference_latents_method"] = ref_method
        out.append([cond, opts])
    return out


def with_concat(conditioning: list, latent: torch.Tensor) -> list:
    """A copy of a CLIP conditioning carrying `latent` ((1, C × inputs, h, w), processed; zeros
    for a missing input) as the model's channel inputs: c_concat, the conds ComfyUI's concat_cond
    makes. Set as a model cond directly, so it reaches the model whatever ComfyUI's own
    concat_cond would make of it (see wide_input.py)."""
    import comfy.conds

    out = []
    for cond, opts in conditioning:
        opts = dict(opts)
        conds = dict(opts.get("model_conds") or {})
        conds["c_concat"] = comfy.conds.CONDNoiseShape(latent if latent.ndim == 4 else latent[None])
        opts["model_conds"] = conds
        out.append([cond, opts])
    return out


VISION = "<|vision_start|><|image_pad|><|vision_end|>"
GROUNDING_PX = 768  # longest side of the picture the text encoder sees (krea2edit's grounding_px)
# MOSS Image's encoder sees pictures as ai-toolkit's MOSS-Image Edit training shows them (and ComfyUI's
# Text Encode MOSS-Image Edit, its "ai-toolkit" format): each scaled to at most this area, "Picture N:"
# before it. Qwen3.5 makes a token of every 32 px square: ~144 tokens a picture
MOSS_GROUNDING_AREA = 384 * 384


def is_moss(clip) -> bool:
    return type(getattr(clip, "tokenizer", None)).__name__ == "MOSSImageTokenizer"


def moss_grounding(image: torch.Tensor) -> torch.Tensor:
    """(1, H, W, 3) -> as MOSS's encoder sees it: area-scaled to at most MOSS_GROUNDING_AREA."""
    import math

    import comfy.utils

    scale = min(1.0, math.sqrt(MOSS_GROUNDING_AREA / (image.shape[1] * image.shape[2])))
    size = (round(image.shape[2] * scale), round(image.shape[1] * scale))
    return comfy.utils.common_upscale(image.movedim(-1, 1), size[0], size[1], "area", "disabled").movedim(1, -1)


def instruction(text: str, template: str) -> str:
    return template.replace("{}", text) if template else text


def grounded_template(clip, n: int) -> str | None:
    """The text encoder's own chat template with n pictures before the instruction, or None
    when it isn't a vision-language chat encoder. Its own template matters: Krea 2's carries a
    system prompt that qwen3vl's default picture template doesn't."""
    tok = getattr(clip, "tokenizer", None)
    if is_moss(clip):  # its conditioning template, an open think block and all; ai-toolkit's numbered pictures
        pictures = "".join(f"Picture {i + 1}: {VISION}" for i in range(n))
        return "<|im_start|>user\n" + pictures + "{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    base = getattr(tok, "llama_template", None)
    if not isinstance(base, str) or "user\n" not in base:
        return None
    return base.replace("user\n", "user\n" + VISION * n, 1)


def flatten(im):
    """An RGB picture of `im`, its transparent parts over white: what a vision encoder is shown
    (ComfyUI's Qwen edit node does the same; the VAE gets all four channels)."""
    from PIL import Image

    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
    return im.convert("RGB")


def grounding_pixels(path, clip=None) -> torch.Tensor:
    """An input picture as the text encoder sees it: whole (never cropped to the target), longest
    side at most GROUNDING_PX (MOSS: moss_grounding). (1, H, W, 3) in [0, 1]."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = flatten(ImageOps.exif_transpose(im))
        if is_moss(clip):
            return moss_grounding(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None])
        if max(im.size) > GROUNDING_PX:
            s = GROUNDING_PX / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def encode_grounded(clip, text: str, pictures: list[torch.Tensor]):
    """A caption's conditioning with the encoder looking at these pictures."""
    template = grounded_template(clip, len(pictures))
    if template is None:
        raise ValueError("grounded edits need a vision-language text encoder (Qwen3-VL: Krea 2, Qwen Image; Qwen3.5: MOSS Image)")
    if is_moss(clip):
        text = text.strip()  # as MOSS's own template does
    tokens = clip.tokenize(text, images=pictures, llama_template=template)
    return clip.encode_from_tokens_scheduled(tokens)


# ---- MiniMax H3: built by ComfyUI's own nodes, so training sees exactly what inference does ----

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}


def _image(path) -> torch.Tensor:
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32) / 255.0)[None]


def _audio(path, seconds: float | None = None) -> dict | None:
    from . import video

    got = video.read_audio(path, 0.0, seconds if seconds else 1e9)
    if got is None:
        return None
    wave, sr = got
    return {"waveform": wave[None], "sample_rate": sr}


def _video(path, max_frames: int) -> torch.Tensor:
    """A reference clip's frames at 24 fps, at its own size (H3's node fits the canvas)."""
    from . import video

    p = video.probe(path)
    n = min(max_frames, int(p.frames * 24 / p.fps))
    idx = [min(p.frames - 1, round(i * p.fps / 24)) for i in range(n)]
    return video.read(path, idx, (p.width, p.height))


def h3_reference_cond(clip, vae, audio_vae, text: str, inputs: list, width: int, height: int, frames: int):
    """MiniMaxH3ReferenceToVideo's conditioning for this caption and these inputs: images,
    videos (with their soundtrack) and audio files, in order."""
    from pathlib import Path

    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    from . import IMAGE_EXTS, VIDEO_EXTS

    images, videos, sounds, audios = {}, {}, {}, {}
    for p in map(Path, inputs):
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images[f"ref_image_{len(images) + 1}"] = _image(p)
        elif ext in VIDEO_EXTS:
            k = len(videos) + 1
            videos[f"ref_video_{k}"] = _video(p, 15 * 24)
            if audio_vae is not None and (snd := _audio(p, 15.0)) is not None:
                sounds[f"ref_video_audio_{k}"] = snd
        elif ext in AUDIO_EXTS and audio_vae is not None and (snd := _audio(p)) is not None:
            audios[f"ref_audio_{len(audios) + 1}"] = snd
    out = MiniMaxH3ReferenceToVideo.execute(clip, text, width, height, frames, "match", vae, audio_vae,
                                            ref_images=images, ref_videos=videos, ref_video_audios=sounds, ref_audios=audios)
    return out.args[0]


def h3_keyframe_cond(clip, vae, text: str, first: torch.Tensor | None, last: torch.Tensor | None,
                     width: int, height: int, frames: int):
    """MiniMaxH3ImageToVideo's conditioning: the clip anchored on its first (and last) frame."""
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

    return MiniMaxH3ImageToVideo.execute(clip, vae, text, width, height, frames, first, last).args[0]
