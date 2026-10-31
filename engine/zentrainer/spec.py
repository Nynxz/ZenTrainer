"""Run specs — the declarative contract between the UI and the harness.

A spec says *what* to train; the harness decides *how*. Pure Python, no ComfyUI
imports, so it can be validated anywhere.
"""

from __future__ import annotations

import copy
import json
import re
import types
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

ModelKind = Literal["checkpoint", "split"]
Auto = Literal["auto"]

# choices that live in the harness, which imports torch: tests keep them in step
OPTIMIZERS = ("AdamW", "Adam", "Adafactor", "Lion", "SGD", "RMSprop", "AdamW8bit", "Lion8bit", "Automagic",
              "Automagic3", "Automagic4", "Prodigy", "AdamWScheduleFree", "CAME")  # harness/optim OPTIMIZERS
TRAIN_ATTENTION = ("pytorch", "xformers", "sub_quad", "split")  # harness/memory TRAINABLE_ATTENTION
WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")  # ComfyUI's weight_dtype
DTYPES = ("bf16", "fp16", "fp32")


def setting(default: Any = MISSING, doc: str = "", *, factory: Any = MISSING, choices: tuple | None = None,
            ge: float | None = None, gt: float | None = None, le: float | None = None, lt: float | None = None):
    """A spec field and what it means: `doc` is its line in the reference (docs/spec.md) and the
    schema; `choices` and the bounds are checked by validate() and published in the schema."""
    meta = {"doc": doc, "choices": choices, "ge": ge, "gt": gt, "le": le, "lt": lt}
    if factory is not MISSING:
        return field(default_factory=factory, metadata=meta)
    return field(default=default, metadata=meta)


@dataclass
class ModelSpec:
    """The model to train: one checkpoint file, or a diffusion model, text encoder and VAE."""
    kind: ModelKind = setting("checkpoint", '"checkpoint": one file with the model, text encoder and VAE (SD 1.5, '
                              'SDXL). "split": separate files (Flux, Qwen Image, Z-Image, ...).',
                              choices=("checkpoint", "split"))
    ckpt_name: str | None = setting(None, "checkpoint models: the file in ComfyUI's checkpoints folder.")
    unet_name: str | None = setting(None, "split models: the diffusion model, in diffusion_models.")
    unet_weight_dtype: str = setting("default", "How the diffusion model's weights load (ComfyUI's weight_dtype).",
                                     choices=WEIGHT_DTYPES)
    clip_name: str | None = setting(None, "split models: the text encoder, in text_encoders.")
    clip_type: str | None = setting(None, "split models: the text encoder's type, as ComfyUI's CLIPLoader names it "
                                          "(flux2, qwen_image, ...).")
    vae_name: str | None = setting(None, "split models: the VAE, in vae.")
    audio_vae_name: str | None = setting(None, "Joint audio-video models (MiniMax H3): their audio VAE, needed to "
                                               "train on a video's sound (dataset.video_audio) and for sound in samples.")


@dataclass
class DatasetSpec:
    """The pictures and videos, their captions, and how they're prepared."""
    folder: str = setting("", "The dataset folder: pictures and videos with same-name caption files. A kohya-style "
                              '"N_name" subfolder repeats its pictures N times.')
    resolution: int = setting(1024, "Training resolution: pictures are resized to about this many pixels squared.",
                              ge=64)
    stages: list[dict] = setting(doc='Lower resolutions first: [{"resolution": 768, "until": 0.6}] trains at 768 for '
                                     "the first 60% of the run, then at `resolution`. Every resolution is cached "
                                     "before training. A stage may add its own \"batch_size\" (pictures per step at that "
                                     "resolution; `optim.batch_size` is the final resolution's). "
                                     "[] = one resolution throughout.", factory=list)
    stage_blend: float = setting(0.2, "How much of the run each resolution change blends over, so nothing switches "
                                      "at once (0.6 to 0.8: the next resolution goes from no steps to every step).",
                                 ge=0, lt=1)
    caption_ext: str = setting(".txt", "The caption files' extension.")
    buckets: bool = setting(True, "Aspect-ratio buckets: each picture is resized to the closest shape of about "
                                  "resolution² pixels. Off = centre-cropped squares.")
    bucket_step: int = setting(64, "Bucket sides are multiples of this.", ge=8)
    edit: bool = setting(False, "Edit training: each picture's input picture(s) from `control_sets` are what the model "
                                "edits from (as reference latents) and the caption is the instruction. For models "
                                "that take reference images (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit, ...). Any "
                                "control set turns it on.")
    control_sets: list[str] = setting(doc="Control sets, in order: named folders of input pictures beside the dataset "
                                          "(_controls/<set>/<name>.<ext>; older <name>_control files are the "
                                          '"control" set). Several sets give each picture several inputs. Edit '
                                          'training with none uses "control".', factory=list)
    edit_method: str = setting("", 'How the input picture joins the sequence: "" = the model\'s own way, or "index", '
                                   '"offset", "index_timestep_zero". "concat": channel inputs instead, each control '
                                   "set's picture (aligned with the target, the same size) as extra latent channels "
                                   "into a widened input layer, like ComfyUI's MOSS Image control adapters.",
                               choices=("", "index", "offset", "index_timestep_zero", "negative_index", "uxo", "concat"))
    control_dropout: float = setting(0.0, "Channel inputs: the share of steps each input is left out (zeros), "
                                          "independently, so the adapter also works with some or none of them.",
                                     ge=0, lt=1)
    edit_template: str = setting("", 'Wraps each instruction: "{}" is the caption (e.g. a generalist edit model\'s '
                                     "own wording, for a specialist trained on top of it). \"\" = the caption as it is.")
    edit_grounded: bool = setting(False, "The text encoder sees the input pictures too (Qwen3-VL based: Krea 2, Qwen "
                                         "Image): conditioning becomes per picture, encoded when the dataset caches.")
    disk_cache: bool = setting(False, "Each picture's latents (and its own conditioning) in a file of its own, read "
                                      "when a step uses it: for datasets bigger than RAM (thousands of grounded edit "
                                      "pairs).")
    video_keyframes: str = setting("", 'MiniMax H3 image-to-video: each clip conditioned on its own first frame '
                                       '("first") or first and last ("first_last"), as the fl2va model is at '
                                       'inference. "" = off.', choices=("", "first", "first_last"))
    caption_dropout: float = setting(0.0, "The share of steps that train on an empty caption.", ge=0, le=1)
    caption_set: str = setting("default", 'Which captions to train on: <name>.txt is "default", <name>.<set>.txt the '
                                          "others (tags, nl, ...).")
    caption_mix: list[str] = setting(doc="Several caption sets to mix: each step picks one of the picture's at "
                                         "random. [] = caption_set only.", factory=list)
    trigger: str = setting("", '"[trigger]" in captions and sample prompts becomes this, and captions that don\'t '
                               'mention it get it prepended ("ohwx, a photo of ...").')
    trigger_prepend: bool = setting(True, "Prepend the trigger to captions that don't mention it.")
    masked_loss: bool = setting(False, "Masked training: pictures with a mask only learn where it's white.")
    unmasked_weight: float = setting(0.0, "Masked training: how much the rest of the picture still trains "
                                          "(0 = not at all, 0.1 = lightly).", ge=0, le=1)
    video_frames: int = setting(49, "Videos train as clips this many frames long at most (snapped to a length the "
                                    "model's VAE encodes exactly). An image model trains a video's frames as pictures.",
                                ge=1)
    video_fps: float = setting(0.0, "Clips resampled to this frame rate. 0 = the video's own.", ge=0)
    video_mode: str = setting("uniform", 'Which parts of a video become clips: "uniform" spread over it, "head" its '
                                         'start, "chunks" consecutive.', choices=("uniform", "head", "chunks"))
    video_clips: int = setting(1, "Clips per video, at most.", ge=1)
    video_resolution: int = setting(512, "Each clip at about this many pixels squared (a clip costs frames x pixels, "
                                         "so videos usually train smaller than pictures).", ge=64)
    video_audio: bool = setting(False, "Train on each clip's soundtrack too (joint audio-video models with "
                                       "model.audio_vae_name), so cuts can learn the beat.")
    video_avoid_cuts: bool = setting(False, "Keep clips inside one shot: no clip spans a hard cut. Off for edits where "
                                            "the cutting is the point (music videos).")
    video_stills: bool = setting(False, "Each clip also trains its sharpest frame as a picture (with the clip's "
                                        "caption). Stills are never TREAD-routed and anchor the LoRA's detail.")


@dataclass
class NetworkSpec:
    """The adapter that trains: its kind, size, and which layers get one."""
    algorithm: str = setting("LoRA", "The adapter (ComfyUI's weight adapters).", choices=("LoRA", "LoHa", "LoKr", "OFT"))
    rank: int = setting(16, "The adapter's rank.", ge=1)
    alpha: float | None = setting(None, "The adapter's alpha. null = the rank (a scale of 1).", gt=0)
    lora_dtype: str = setting("fp32", "What the adapter trains in. fp32: AdamW's steps on a LoRA are smaller than bf16 "
                                      "can represent at the weights' size, so most would round away. bf16 only with an "
                                      "optimizer that rounds stochastically (Automagic).", choices=DTYPES)
    targets: str | list[str] = setting("attn_mlp", 'Which layers get adapters: "attn" (attention projections), "mlp" '
                                                   '(feed-forward), "attn_mlp" (both), "all" (every adaptable layer, '
                                                   "like ComfyUI's native trainer), or a list of regexes over module "
                                                   "names.")
    train_bias_norm: bool = setting(False, "Train the biases and norm weights too.")
    init_from: str | None = setting(None, "A LoRA in the loras folder to continue from.")
    training_adapter: str | None = setting(None, "A LoRA applied while training only, never in samples or the saved "
                                                 "file: e.g. a de-distillation adapter for a guidance-distilled "
                                                 "model (MiniMax H3), so training sits on an undistilled base.")
    base_lora: str | None = setting(None, "A LoRA the trained one sits on everywhere (training, samples and use), e.g. "
                                          "a fix the model always needs. The saved LoRA leaves it out and records it: "
                                          "load it beside, at base_strength.")
    base_strength: float = setting(1.0, "base_lora's strength.", ge=-4, le=4)
    blocks: str = setting("", 'Only these transformer blocks get adapters, e.g. "20-49" or "10-": the backward stops '
                              "at the first one, so steps are quicker, and earlier blocks stay untouched. \"\" = every "
                              "block.")
    skip_blocks: str = setting("", 'These blocks get no adapters, e.g. "22" or "22, 30-31" (a block that skip-layer '
                                   "guidance skips stays the base's).")


@dataclass
class OptimSpec:
    """The optimizer, learning rate and schedule, and the run's length."""
    optimizer: str = setting("AdamW", "The optimizer.", choices=OPTIMIZERS)
    learning_rate: float = setting(1e-4, "The learning rate.", gt=0)
    weight_decay: float = setting(0.0, "Weight decay.", ge=0)
    lr_scheduler: str = setting("constant", "The learning rate over the run.", choices=("constant", "linear", "cosine"))
    warmup_steps: int = setting(0, "Steps the learning rate ramps up over, from 0.", ge=0)
    max_grad_norm: float = setting(1.0, "Gradients clipped to this norm. 0 = off.", ge=0)
    steps: int = setting(1000, "Training steps.", ge=1)
    batch_size: int = setting(1, "Pictures per step.", ge=1)
    grad_accumulation_steps: int = setting(1, "Batches per optimizer step.", ge=1)
    lora_plus: float = setting(0.0, "LoRA+: the up matrices learn this many times faster than the down ones, which "
                                    "trains a LoRA faster for the same stability (paper: 16; 4 is a safe start). "
                                    "0 or 1 = off.", ge=0)
    ema: float = setting(0.0, "Weight averaging: checkpoints, samples and the saved LoRA use a running average of the "
                              "adapter (0.98 = roughly the last 50 steps). 0 = off.", ge=0, lt=1)
    lr_floor: float = setting(0.0, "Automagic v3/v4: the adaptive rate never drops below this fraction of its start "
                                   "(v3's decays and can stall under-strength). 0 = no floor.", ge=0, le=1)


@dataclass
class ObjectiveSpec:
    """What the loss compares, and which noise levels each step trains at."""
    loss: str = setting("mse", "The loss.", choices=("mse", "l1", "huber", "smooth_l1"))
    target: str = setting("native", '"native": each model\'s own prediction (noise for SD-style models, velocity for '
                                    'flow models). "x0": the loss on the denoised picture (ComfyUI\'s native trainer).',
                          choices=("native", "x0"))
    timesteps: str = setting("uniform", 'How each step\'s noise level is drawn: "uniform"; "logit_normal" (AI '
                                        'Toolkit\'s "sigmoid"), shaped by logit_mean and logit_std; or "scheduler": '
                                        "the levels ComfyUI's scheduler `timestep_scheduler` samples at in "
                                        "`timestep_scheduler_steps` steps, each as often (trains where the LoRA will "
                                        "be used; few steps for a fast LoRA).",
                             choices=("uniform", "logit_normal", "scheduler"))
    timestep_scheduler: str = setting("simple", 'timesteps "scheduler": the ComfyUI scheduler.')
    timestep_scheduler_steps: int = setting(20, 'timesteps "scheduler": its step count.', ge=1, le=1000)
    logit_mean: float = setting(0.0, "logit_normal's centre: positive leans to heavy noise (composition), negative to "
                                     "light (detail).", ge=-3, le=3)
    logit_std: float = setting(1.0, "logit_normal's spread.", gt=0, le=3)
    uniform_mix: float = setting(0.0, "logit_normal: the share of steps drawn uniformly instead, keeping a little "
                                      "training at the extremes it rarely visits (0.01).", ge=0, le=1)
    shift: float = setting(0.0, "Flow-matching models: the training timestep shift. 0 = the model's own sampling "
                                "shift, which favours heavy noise (composition, colour); a lower one (1-3) spends more "
                                "steps on detail, such as faces.", ge=0)
    bias: str = setting("balanced", 'Lean the noise levels toward one end, whatever the distribution: "high_noise" '
                                    'for composition, layout and pose, "low_noise" for detail and texture (AI '
                                    "Toolkit's \"content\" and \"style\"). \"balanced\" = as drawn.",
                        choices=("balanced", "high_noise", "low_noise"))
    bias_power: float = setting(3.0, "How hard the bias leans: the draw to this power (3, as AI Toolkit, puts half the "
                                     "steps in the noisiest eighth for high_noise). 1 = not at all.", ge=1, le=10)
    weighting: str = setting("", 'Per-noise-level loss weights. "flex" (flow models): AI Toolkit\'s "weighted", '
                                 'measured on Flex.1. "balance": measures the model as it trains and evens the pull of '
                                 "every noise level (EDM2's idea). \"\" = none.", choices=("", "flex", "balance"))
    spread: bool = setting(False, "Noise levels from an even, low-discrepancy sequence instead of random draws: a "
                                  "small batch covers the schedule in a few steps, for a steadier loss.")
    preserve: str = setting("", 'Class words, e.g. "woman, man, person" (AI Toolkit\'s differential output '
                                "preservation): a step also trains its caption with one of them in the trigger's place "
                                "to give what the base model gives, so the adapter changes only what the trigger asks "
                                "for and \"a man\" stays a man. Needs a trigger. A step with it takes about 2.3x as "
                                'long (the base model\'s pass and the adapter\'s own); every 2nd step, 1.7x. "" = off.')
    preserve_weight: float = setting(1.0, "objective.preserve: its loss's weight beside the training loss.", gt=0)
    preserve_every: int = setting(2, "objective.preserve: on every Nth step. 2 held the class as well as every step "
                                     "(MOSS: \"a woman\" 0.07 likeness to the trained one either way) at 1.7x the time "
                                     "against 2.3x.", ge=1)
    preserve_plain: float = setting(0.0, 'objective.preserve: the share of its passes on a plain class prompt ("a photo '
                                         'of a woman") at heavy noise instead of the step\'s own caption: holds the '
                                         "prompts people write, where generating from noise takes on the trained "
                                         "subject. 0 = the captions only.", ge=0, le=1)


@dataclass
class MemorySpec:
    """Precision, checkpointing, offloading and kernels: how the run fits and how fast it goes."""
    training_dtype: str = setting("bf16", "The precision the model computes in.", choices=DTYPES)
    gradient_checkpointing: bool = setting(True, "Recompute activations in the backward instead of keeping them: much "
                                                 "less VRAM, some speed.")
    grad_checkpoint_depth: int = setting(1, "Gradient checkpointing's granularity.", ge=1)
    grad_checkpoint_skip: int | Auto = setting("auto", "Leave N blocks un-checkpointed (the biggest, the last first): "
                                                     "they keep their activations (more VRAM) and skip the recompute, the cheapest "
                                                     'place to trade memory for speed. "auto": as many as the VRAM '
                                                     "the first steps left free holds (none with offloading).", ge=0)
    checkpoint_save: str = setting("attention", "What checkpointed blocks keep instead of recomputing: "
                                                '"attention": attention\'s output (about 50 MB a block at 1024 px), so '
                                                "the backward doesn't run attention again (Krea 2 6%, MOSS 15% a step, "
                                                'for 1-2 GB). "none": everything recomputed, the least VRAM. Not with '
                                                "offloading.", choices=("attention", "none"))
    offloading: bool = setting(False, "Keep part of the model in RAM and stream it in: for models bigger than the "
                                      "card.")
    vram_headroom: float = setting(0.0, "With offloading: the VRAM kept free for activations, the LoRA and its "
                                        "optimizer (GB); the rest holds as much of the model as fits. 0 = auto (8 GB, "
                                        "more for long video clips).", ge=0)
    tread: float = setting(0.0, "TREAD token routing on video steps (MiniMax H3): this share of a clip's tokens skips "
                                "the middle blocks and rejoins unchanged (0.5 roughly halves their work). Stills, "
                                "samples and validation always run whole. 0 = off.", ge=0, lt=1)
    tread_steps: float = setting(1.0, "The share of clip steps that route; the rest run whole. A dataset of clips only, "
                                      "routed every step, lets the LoRA learn to make up for the routing "
                                      "(over-sharpened samples): below 1 keeps whole steps.", ge=0, le=1)
    int8_forward: bool = setting(True, "int8 bases (MiniMax H3, Krea 2 int8): trained layers run ComfyUI's int8 kernel "
                                       "forward instead of a dequantized bf16 matmul. Off is slower: for comparing "
                                       "quality.")
    int8_backward: str = setting("w8a8", 'int8 bases: how the backward gets each frozen layer\'s input gradient. "w8a8": '
                                         "quantized to int8 too, on int8 tensor cores (about 2x the bf16 backward), "
                                         'for some gradient precision. "fused": a fused Triton kernel at full '
                                         'precision, without rebuilding the layer\'s bf16 weight. "bf16": the weight '
                                         "rebuilt in bf16 (ComfyUI's way; the slowest).",
                                 choices=("w8a8", "fused", "bf16"))
    fp8_backward: bool = setting(False, "fp8 bases: the backward's matmuls in fp8 too (ComfyUI's training_fp8_bwd). "
                                        "Faster, less precise.")
    compile_blocks: bool | Auto = setting(False, "torch.compile the repeated transformer block, fusing the elementwise "
                                                 "work around the matmuls; the first steps (and first sample) are "
                                                 "slower while it compiles, minutes on ROCm. Off unless turned on: "
                                                 '"auto" (an older spec\'s) is off too.', choices=("auto", True, False))
    fused_qk_rope: bool | Auto = setting("auto", "Krea 2, MOSS, MiniMax H3: the attention's q/k norms and rope as one "
                                                 "op with its own backward (Triton): the same maths, 6-9% a step. "
                                                 '"auto": on while the blocks run uncompiled (compiled ones fuse '
                                                 "these already).", choices=("auto", True, False))
    bypass_mode: bool = setting(False, "Adapters as forward hooks beside each layer instead of patched into its weight. "
                                       "Quantized bases always train this way.")
    attention: str = setting("pytorch", "The attention backend while training (only ones with a backward pass; flash "
                                        "and sage have none).", choices=TRAIN_ATTENTION)


@dataclass
class SamplePrompt:
    """One sample: its prompt and, optionally, its own size/length/seed (else the run's)."""
    prompt: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    seed: int | None = None
    # edit runs: the picture to edit (a path in the dataset folder, or under ComfyUI's input dir)
    control: str | None = None


@dataclass
class SampleSpec:
    """Sample pictures (or clips) during the run: what, how often, and how they're made."""
    likeness: bool = setting(True, "Score each sample's face against the dataset's (with faces on for the dataset).")
    every: int = setting(0, "Sample every this many steps. 0 = off.", ge=0)
    prompts: list[str | dict] = setting(doc='The samples: each a prompt, or {"prompt", "width", "height", "frames", '
                                            '"seed", "control"} with its own size, length, seed or (edit runs) '
                                            "picture to edit.", factory=list)
    negative: str = setting("", "The negative prompt.")
    width: int = setting(1024, "Sample width.", ge=16)
    height: int = setting(1024, "Sample height.", ge=16)
    frames: int = setting(1, "Video models: frames per sample (1 = a still), snapped to a length the VAE decodes and "
                             "saved as an .mp4 at fps.", ge=1)
    fps: float = setting(24.0, "Video samples' frame rate.", gt=0)
    steps: int = setting(20, "Sampling steps.", ge=1)
    cfg: float = setting(5.0, "CFG scale.", ge=0)
    sampler: str = setting("euler", "ComfyUI's sampler.")
    scheduler: str = setting("normal", "ComfyUI's scheduler.")
    sigmas: list[float] = setting(doc="Custom sigmas, high to low (e.g. [1.0, 0.94, 0.85, 0.7, 0.5, 0.3, 0.12, 0]), in "
                                      "place of the scheduler and steps (len - 1 steps). [] = the scheduler.",
                                  factory=list)
    seed: int = setting(42, "The first sample's seed; each next prompt the next one.", ge=0)
    baseline: bool = setting(True, "Render the prompts once before training: the base model's look, to compare "
                                   "against.")
    model_name: str | None = setting(None, "Sample with another model of the same family than the one training (e.g. "
                                           "train a raw model, preview on its turbo): a diffusion_models file for "
                                           "split models, a checkpoints file for checkpoint ones.")
    model_weight_dtype: str = setting("default", "How model_name's weights load.", choices=WEIGHT_DTYPES)
    lora_strength: float = setting(1.0, "The trained LoRA's strength in samples.")
    turbo_lora: str | None = setting(None, "A turbo LoRA under every sample (not training, not the saved LoRA): "
                                           "few-step previews.")
    turbo_strength: float = setting(0.75, "turbo_lora's strength; 0 samples without it (it stays loaded, so it can be "
                                          "turned back up while the run trains).", ge=0)
    slg_mode: str = setting("off", 'Skip-layer guidance, as ComfyUI\'s SkipLayerGuidanceDiT: "uncond" skips the layers '
                                   'in the negative pass only (no extra cost), "standard" adds a pass without them and '
                                   "pushes away from it by slg_scale.", choices=("off", "uncond", "standard"))
    slg_layers: list[int] = setting(doc="Skip-layer guidance: the main (single-stream) blocks to skip.", factory=list)
    slg_double_layers: list[int] = setting(doc="Skip-layer guidance: double-stream blocks, for models that have them.",
                                           factory=list)
    slg_scale: float = setting(3.0, 'Skip-layer guidance "standard": how hard it pushes.', ge=0)
    slg_start: float = setting(0.01, "Skip-layer guidance: where in sampling it starts (0 to 1).", ge=0, le=1)
    slg_end: float = setting(1.0, "Skip-layer guidance: where it ends (0 to 1).", ge=0, le=1)


@dataclass
class ValidationSpec:
    """A smooth "is it learning" curve: loss on fixed pictures at fixed noise levels with fixed
    noise, so the only thing that changes between evaluations is the LoRA."""
    every: int = setting(100, "Measure every this many steps (and before training). 0 = off.", ge=0)
    images: int = setting(4, "How many pictures to measure.", ge=1)
    noise_levels: int = setting(4, "Noise levels each picture is measured at, spread over the schedule.", ge=1)
    seed: int = setting(1234, "Picks the pictures and their noise.", ge=0)
    dataset: str = setting("", "A held-out dataset folder to measure on instead, prepared like the training one (edit "
                               "inputs, masks, captions): loss on pictures the LoRA never trains on, so the curve says "
                               'whether it generalises rather than how well it fits. "" = pictures from the training '
                               "dataset.")


@dataclass
class OutputSpec:
    """Where the LoRAs go, how often, in what, and which checkpoints stay."""
    save_every: int = setting(250, "A checkpoint (the LoRA and its resume state) every this many steps. 0 = only the "
                                   "final LoRA.", ge=0)
    lora_dir: str | None = setting(None, "One of ComfyUI's loras folders to save in (e.g. on a roomier disk). null = "
                                         "output/zentrainer/loras. LoRAs go in a zentrainer/<run>/ subfolder of it.")
    keep_states: str = setting("latest", "Resume states (optimizer, schedule, RNG) beside each checkpoint, often bigger "
                                         'than the LoRA: "latest" keeps only the newest, "all" one per checkpoint, '
                                         '"none" skips them (the run can\'t be resumed).',
                               choices=("latest", "all", "none"))
    save_dtype: str = setting("bf16", "What the LoRA files are written in (training keeps network.lora_dtype): bf16 "
                                      "halves an fp32 file and loses nothing a sample shows.", choices=DTYPES)
    keep: str = setting("all", 'Which scheduled checkpoints a run keeps: "all"; the keep_count "latest"; the keep_count '
                               'best by "likeness" (its samples\' faces) or by "val_loss"; or each new best, '
                               '"likeness_new_best" / "val_loss_new_best". The final LoRA, the newest checkpoint and '
                               "saves asked for by hand always stay.",
                        choices=("all", "latest", "likeness", "val_loss", "likeness_new_best", "val_loss_new_best"))
    keep_count: int = setting(3, 'How many checkpoints "latest", "likeness" and "val_loss" keep.', ge=1)


@dataclass
class SliderSpec:
    """A slider LoRA: one whose strength moves a picture along one attribute, +1 towards `positive`,
    -1 towards `negative`, 0 the base model. "pairs" learns it from picture pairs that differ only in
    it; "text" from the base model's own predictions for the prompts (no captions needed)."""

    kind: str = setting("", '"" (a normal LoRA), "pairs" or "text".', choices=("", "pairs", "text"))
    pair_set: str = setting("", "pairs: the control set holding each picture's partner (_controls/<pair_set>/, the "
                                "same names).")
    dataset_end: str = setting("positive", "pairs: which end the dataset's own pictures are.",
                               choices=("positive", "negative"))
    positive: str = setting("", 'What +1 adds, e.g. "a person, old, wrinkled skin, grey hair".')
    negative: str = setting("", 'What -1 adds, e.g. "a person, young, smooth skin".')
    neutral: str = setting("", 'text: what the LoRA trains on, without either end, e.g. "a person".')
    anchors: list[str] = setting(doc="text: prompts the slider should leave alone (pulled to the base model at +1 and "
                                     "-1).", factory=list)
    guidance: float = setting(3.0, "text: how far each end is pushed past the base model's own difference.", gt=0, le=20)
    anchor_weight: float = setting(1.0, "text: the anchors' weight.", ge=0)
    directions: str = setting("both", '"both", or "positive" (only +1 is trained).', choices=("both", "positive"))


@dataclass
class TrainSpec:
    """A training run: what to train and how. Each section has its own table below."""
    spec_version: int = setting(2, "The spec's layout. An older one is brought up to date as it's read; a newer one "
                                   "(from a newer ZenTrainer) is refused.")
    name: str = setting("untitled", "The run's name.")
    seed: int = setting(0, "The run's seed: noise, shuffling and dropout.", ge=0)
    model: ModelSpec = setting(doc="The model.", factory=ModelSpec)
    dataset: DatasetSpec = setting(doc="The dataset.", factory=DatasetSpec)
    network: NetworkSpec = setting(doc="The adapter.", factory=NetworkSpec)
    optim: OptimSpec = setting(doc="The optimizer and schedule.", factory=OptimSpec)
    objective: ObjectiveSpec = setting(doc="The loss and noise levels.", factory=ObjectiveSpec)
    memory: MemorySpec = setting(doc="Memory and speed.", factory=MemorySpec)
    samples: SampleSpec = setting(doc="Samples.", factory=SampleSpec)
    validation: ValidationSpec = setting(doc="Validation loss.", factory=ValidationSpec)
    output: OutputSpec = setting(doc="Saving.", factory=OutputSpec)
    slider: SliderSpec = setting(doc="Slider LoRAs.", factory=SliderSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> TrainSpec:
        """A spec from its dict, any version (migrated first). strict: a field this version doesn't
        have is an error (a typo in a spec written by hand); else it's left out, as migrate does for
        what's stored (runs, presets)."""
        raw, dropped = migrate(data)
        if strict and dropped:
            raise SpecError(f"unknown field(s): {', '.join(_unknown(k) for k in dropped)}")

        def build(dc, raw):
            return dc(**(raw or {}))

        nested = {f.name: f.default_factory for f in fields(cls) if callable(f.default_factory)}
        parts = {k: build(dc, raw.pop(k, None)) for k, dc in nested.items()}
        spec = build(cls, raw)
        for k, v in parts.items():
            setattr(spec, k, v)
        _normalize(spec)
        spec.validate()
        return spec

    def validate(self) -> None:
        """SpecError for the first thing wrong: each field's type, choices and bounds (from its
        setting()), then what depends on several fields. It only checks; from_dict fills in."""
        for sec in fields(self):
            v = getattr(self, sec.name)
            if is_dataclass(v):
                _check_fields(v, f"{sec.name}.")
        _check_fields(self, "", sections=False)
        m = self.model
        if m.kind == "checkpoint" and not m.ckpt_name:
            raise SpecError("model.ckpt_name is required for a checkpoint model")
        if m.kind == "split":
            missing = [k for k in ("unet_name", "clip_name", "clip_type", "vae_name") if not getattr(m, k)]
            if missing:
                raise SpecError(f"split model needs: {', '.join('model.' + k for k in missing)}")
        d = self.dataset
        if not d.folder:
            raise SpecError("dataset.folder is required")
        last = 0.0
        for st in d.stages:
            if not isinstance(st.get("resolution"), int) or not isinstance(st.get("until"), (int, float)) or set(st) - {"resolution", "until", "batch_size"}:
                raise SpecError('dataset.stages are {"resolution": 768, "until": 0.6, "batch_size": 4} (batch_size optional)')
            if "batch_size" in st and (not isinstance(st["batch_size"], int) or isinstance(st["batch_size"], bool) or st["batch_size"] < 1):
                raise SpecError("a stage's batch_size is a whole number, 1 or more")
            if not 256 <= st["resolution"] < d.resolution:
                raise SpecError(f"a stage's resolution is below the run's ({d.resolution}) and at least 256")
            if not last < st["until"] < 1.0:
                raise SpecError("dataset.stages end in order, each before the run does (until: 0 to 1)")
            last = st["until"]
        if any(not CONTROL_SET_NAME.match(c) for c in d.control_sets):
            raise SpecError("dataset.control_sets are set names: lowercase letters, digits, - and _")
        if d.edit_template and "{}" not in d.edit_template:
            raise SpecError("dataset.edit_template needs {} where the instruction goes")
        if isinstance(self.network.targets, str):
            _choice("network.targets", self.network.targets, ("attn", "mlp", "attn_mlp", "all"))
        elif not any(str(p).strip() for p in self.network.targets) and not self.network.train_bias_norm \
                and not (d.edit and d.edit_method == "concat" and d.control_sets):  # channel inputs train
            # nothing would train: it failed only once the model had loaded ("no layers matched")
            raise SpecError('network.targets has no patterns: add the layers to train (regexes), or use "attn", '
                            '"mlp", "attn_mlp" or "all"')
        if self.objective.preserve.strip():
            if not any(w.strip() for w in self.objective.preserve.split(",")):
                raise SpecError('objective.preserve is class words, comma-separated: e.g. "woman, man, person"')
            if not self.dataset.trigger.strip():
                raise SpecError("objective.preserve swaps the trigger for a class word: set dataset.trigger")
            if self.slider.kind:
                raise SpecError("objective.preserve is for adapters with a trigger, not sliders")
            if self.dataset.edit_grounded:
                raise SpecError("objective.preserve can't swap the caption of a grounded edit (its encoding includes the inputs)")
            if self.dataset.video_keyframes:
                raise SpecError("objective.preserve can't swap the caption of a keyframe clip (its encoding includes the keyframes)")
        if self.network.blocks and not re.fullmatch(r"\s*\d*\s*-?\s*\d*\s*", self.network.blocks):
            raise SpecError("network.blocks is a range like 20-49, 20- or 10")
        if self.network.skip_blocks and not re.fullmatch(r"\s*\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*\s*", self.network.skip_blocks):
            raise SpecError("network.skip_blocks is blocks and ranges like 22 or 22, 30-31")
        if self.objective.timesteps == "scheduler" and self.objective.shift:
            raise SpecError("objective.shift is the model's own with a scheduler (its shift is in the schedule): set it to 0")
        sl = self.slider
        if sl.kind == "pairs":
            if not CONTROL_SET_NAME.match(sl.pair_set or ""):
                raise SpecError("an image slider needs the set holding each picture's partner: slider.pair_set "
                                "(a folder under the dataset's _controls/)")
            if d.edit:
                raise SpecError("an image slider pairs pictures itself: turn edit training off (dataset.edit)")
        if sl.kind == "text" and not (sl.positive.strip() and sl.negative.strip()):
            raise SpecError("a text slider needs both ends: slider.positive and slider.negative")
        sa = self.samples
        check_sigmas(sa.sigmas)
        if sa.slg_start > sa.slg_end:
            raise SpecError("samples.slg_start is before samples.slg_end")
        for p in sa.prompts:
            if isinstance(p, dict):
                if not isinstance(p.get("prompt"), str):
                    raise SpecError("each sample needs a prompt")
                if set(p) - {"prompt", "width", "height", "frames", "seed", "control"}:
                    raise SpecError(f"a sample takes prompt, width, height, frames, seed and control, not "
                                    f"{', '.join(sorted(set(p) - {'prompt', 'width', 'height', 'frames', 'seed', 'control'}))}")
                for k in ("width", "height", "frames", "seed"):
                    if p.get(k) is not None and (not isinstance(p[k], int) or isinstance(p[k], bool) or p[k] < (0 if k == "seed" else 1)):
                        raise SpecError(f"sample {k} must be a positive whole number")
                if any(p.get(k) and p[k] % 16 for k in ("width", "height")):
                    raise SpecError("sample width/height must be multiples of 16")
                if p.get("control") is not None and not isinstance(p["control"], str):
                    raise SpecError("a sample's control is a picture's path")
        out = self.output
        if out.keep in ("val_loss", "val_loss_new_best") and not self.validation.every:
            raise SpecError("keeping the best checkpoints by validation loss needs validation on (validation.every)")
        if out.keep in ("likeness", "likeness_new_best") and not (sa.prompts and sa.likeness):
            raise SpecError("keeping the best checkpoints by likeness needs sample prompts with likeness on "
                            "(samples.prompts, samples.likeness)")
        if self.validation.dataset and self.validation.dataset == d.folder:
            raise SpecError("validation.dataset is the training dataset: pick a held-out one, or leave it empty "
                            "to measure on pictures from the training set")
        if self.validation.dataset and sl.kind == "pairs":
            raise SpecError("a held-out validation dataset isn't for image sliders: their loss is on the pairs")


def _type_ok(value: Any, hint: Any) -> bool:
    """Whether a JSON value fits a field's annotation (bools are never numbers; ints are floats)."""
    origin, args = get_origin(hint), get_args(hint)
    if hint is Any:
        return True
    if origin in (Union, types.UnionType):
        return any(_type_ok(value, a) for a in args)
    if hint is type(None):
        return value is None
    if origin is Literal:
        return any(value == a and type(value) is type(a) for a in args)
    if origin is list or hint is list:
        return isinstance(value, list) and (not args or all(_type_ok(x, args[0]) for x in value))
    if origin is dict or hint is dict:
        return isinstance(value, dict)
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is str:
        return isinstance(value, str)
    return isinstance(value, hint) if isinstance(hint, type) else True


def _type_name(hint: Any) -> str:
    """An annotation as the reference writes it: "number", "text or null", "list of text", ..."""
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        return " or ".join(_type_name(a) for a in args)
    if origin is Literal:
        return " or ".join(json.dumps(a) for a in args)
    if origin is list or hint is list:
        return f"list of {_type_name(args[0])}" if args else "list"
    return {bool: "true/false", int: "whole number", float: "number", str: "text", dict: "object",
            type(None): "null"}.get(hint, getattr(hint, "__name__", str(hint)))


def _bounds(meta) -> str:
    parts = [f"{op} {meta[k]:g}" for k, op in (("ge", ">="), ("gt", ">"), ("le", "<="), ("lt", "<")) if meta.get(k) is not None]
    return ", ".join(parts)


def _check_fields(obj, at: str, sections: bool = True, errors: list[str] | None = None) -> None:
    """One section's fields against their annotation, choices and bounds: SpecError for the first
    problem, or every problem added to `errors`."""
    hints = get_type_hints(type(obj))
    for f in fields(obj):
        if not sections and is_dataclass(getattr(obj, f.name)):
            continue
        problem = _field_problem(getattr(obj, f.name), f.metadata, hints[f.name], f"{at}{f.name}")
        if problem and errors is None:
            raise SpecError(problem)
        if problem:
            errors.append(problem)


def _field_problem(v, meta, hint, name: str) -> str | None:
    if not _type_ok(v, hint):
        return f"{name} is {_type_name(hint)}, got {json.dumps(v) if _jsonable(v) else repr(v)}"
    if meta.get("choices") is not None and isinstance(v, (str, bool)) and not any(v == c and type(v) is type(c) for c in meta["choices"]):
        near = _near(v, [c for c in meta["choices"] if isinstance(c, str)]) if isinstance(v, str) else None
        if near:
            return f"{name}: {json.dumps(v)} isn't one of its choices; did you mean {json.dumps(near)}?"
        return f"{name} is one of {', '.join(json.dumps(c) for c in meta['choices'])}, got {json.dumps(v)}"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        lo, lo_x, hi, hi_x = meta.get("ge"), meta.get("gt"), meta.get("le"), meta.get("lt")
        if (lo is not None and v < lo) or (lo_x is not None and v <= lo_x) or (hi is not None and v > hi) or (hi_x is not None and v >= hi_x):
            return f"{name} must be {_bounds(meta)}, got {v:g}"
    return None


def _near(word: str, options) -> str | None:
    """The option a typo most likely meant (case aside), if any is close."""
    import difflib

    by_lower = {o.lower(): o for o in options}
    if word.lower() in by_lower:
        return by_lower[word.lower()]
    hit = difflib.get_close_matches(word.lower(), list(by_lower), n=1, cutoff=0.8)
    return by_lower[hit[0]] if hit else None


def _unknown(path: str) -> str:
    """An unknown field, with the field it was likely meant to be."""
    *parent, leaf = path.split(".")
    dc = TrainSpec
    for k in parent:
        f = next((f for f in fields(dc) if f.name == k), None)
        sub = f.default_factory() if f is not None and callable(f.default_factory) else None
        if not is_dataclass(sub):
            return path
        dc = type(sub)
    near = _near(leaf, [f.name for f in fields(dc)])
    return f"{path} (did you mean {'.'.join([*parent, near])}?)" if near else path


def problems(data: Any) -> tuple[dict | None, list[str], list[str]]:
    """Everything wrong with a spec at once (for `zt check`), where from_dict stops at the first:
    (the spec as this version reads it, or None; the problems; notes on how it was read). Fields'
    own problems come first; the rules across fields are checked once those are right."""
    notes: list[str] = []
    try:
        raw, dropped = migrate(data)
    except SpecError as e:
        return None, [str(e)], notes
    version = data.get("spec_version", 1) if isinstance(data, dict) else None
    if version != SPEC_VERSION:
        notes.append(f"version {version}: read as version {SPEC_VERSION} (renamed fields carried over)")
    errors = [f"unknown field: {_unknown(k)}" for k in dropped]
    nested = {f.name: f.default_factory for f in fields(TrainSpec) if callable(f.default_factory)}
    spec = TrainSpec()
    for k, v in raw.items():
        if k in nested and isinstance(v, dict):
            try:
                setattr(spec, k, nested[k]().__class__(**v))
            except TypeError as e:
                errors.append(f"{k}: {e}")
        elif k in nested and v is not None:
            errors.append(f"{k} is a section ({{...}}), got {json.dumps(v)}")
        elif k not in nested:
            setattr(spec, k, v)
    for sec in fields(spec):
        if is_dataclass(getattr(spec, sec.name)):
            _check_fields(getattr(spec, sec.name), f"{sec.name}.", errors=errors)
    _check_fields(spec, "", sections=False, errors=errors)
    if not errors:
        _normalize(spec)
        try:
            spec.validate()
        except SpecError as e:
            errors.append(str(e))
    return (None if errors else spec.to_dict()), errors, notes


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _normalize(spec: TrainSpec) -> None:
    """What loading fills in, before validate(): control sets mean edit training, and edit
    training without any uses the "control" set (specs from before control sets)."""
    d = spec.dataset
    if isinstance(d.control_sets, list) and d.control_sets:
        d.edit = True
    elif d.edit is True and d.control_sets == []:
        d.control_sets = ["control"]


class SpecError(ValueError):
    pass


SPEC_VERSION = 2


def _move(d: dict, src: str, dst: str) -> None:
    """d's value at dotted path src to dst (sections made as needed), when src is there and dst
    isn't already."""
    *path, key = src.split(".")
    parent = d
    for k in path:
        parent = parent.get(k) if isinstance(parent, dict) else None
    if not isinstance(parent, dict) or key not in parent:
        return
    value = parent.pop(key)
    *path, key = dst.split(".")
    for k in path:
        if not isinstance(d.get(k), dict):  # missing, or null (a section left at its defaults)
            d[k] = {}
        d = d[k]
    d.setdefault(key, value)


def _v1_to_v2(d: dict) -> None:
    """The renames of spec v2 (2026-10-26)."""
    for src, dst in (("save_every", "output.save_every"), ("optim.scheduler", "optim.lr_scheduler"),
                     ("objective.scheduler", "objective.timestep_scheduler"),
                     ("objective.scheduler_steps", "objective.timestep_scheduler_steps"),
                     ("memory.quantized_backward", "memory.fp8_backward"),
                     ("dataset.keyframes", "dataset.video_keyframes"), ("dataset.clip_stills", "dataset.video_stills")):
        _move(d, src, dst)
    m = d.get("memory")
    if isinstance(m, dict) and ("w8a8_backward" in m or "fused_int8_backward" in m):
        w8a8, fused = m.pop("w8a8_backward", True), m.pop("fused_int8_backward", False)  # v1's defaults
        if w8a8 is None:  # null: the form showed the default, on
            w8a8 = True
        m.setdefault("int8_backward", "w8a8" if w8a8 else "fused" if fused else "bf16")
    _v2_draft(d)


def _v2_draft(d: dict) -> None:
    """v2 as its first draft wrote it, before it was released (2026-10-26/27): these were renamed
    after some runs were saved with them, and its nulls meant auto (or on)."""
    for src, dst in (("validation.timesteps", "validation.noise_levels"),
                     ("memory.checkpoint_depth", "memory.grad_checkpoint_depth"),
                     ("memory.checkpoint_skip", "memory.grad_checkpoint_skip")):
        _move(d, src, dst)
    m = d.get("memory")
    if isinstance(m, dict):
        for k in ("compile_blocks", "fused_qk_rope"):  # v1's null meant auto
            if k in m and m[k] is None:
                m[k] = "auto"
        if "int8_forward" in m and m["int8_forward"] is None:  # null: the form showed the default, on
            m["int8_forward"] = True


MIGRATIONS = {1: _v1_to_v2}  # from version n to n + 1


def _known(dc, raw: dict, at: str, dropped: list[str]) -> dict:
    """`raw` with only the fields dataclass `dc` has, nested dataclasses likewise."""
    kinds = {f.name: f for f in fields(dc)}
    out = {}
    for k, v in raw.items():
        f = kinds.get(k)
        if f is None:
            dropped.append(f"{at}{k}")
            continue
        sub = f.default_factory() if callable(f.default_factory) else None
        if is_dataclass(sub) and isinstance(v, dict):
            out[k] = _known(type(sub), v, f"{at}{k}.", dropped)
        else:
            out[k] = v
    return out


def migrate(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A spec, or part of one (a preset, a profile's settings), in this version's layout, and the
    fields it had that this version doesn't (left out). An older layout is renamed step by step
    (MIGRATIONS); one without spec_version is version 1. A newer one than this version is an error:
    its settings may mean things this version can't do."""
    if not isinstance(data, dict):
        raise SpecError("a spec is a JSON object")
    d = copy.deepcopy(data)
    d.pop("$schema", None)  # an editor's pointer to docs/spec.schema.json: not a setting
    version = d.pop("spec_version", 1)
    if not isinstance(version, int) or version < 1:
        raise SpecError(f"spec_version is a whole number from 1, got {version!r}")
    if version > SPEC_VERSION:
        raise SpecError(f"this spec is version {version}, from a newer ZenTrainer (this one reads up to "
                        f"{SPEC_VERSION}): update ZenTrainer")
    for v in range(version, SPEC_VERSION):
        MIGRATIONS[v](d)
    if version == 2:
        _v2_draft(d)
    dropped: list[str] = []
    out = _known(TrainSpec, d, "", dropped)
    if "spec_version" in data or any(k in data for k in ("model", "dataset")):
        out["spec_version"] = SPEC_VERSION  # a whole spec says what it is; a fragment stays a fragment
    return out, sorted(dropped)


# a control set's name, as caption sets' (harness/data SET_NAME)
CONTROL_SET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")


def _choice(name: str, value: str, options: tuple[str, ...]) -> None:
    if value not in options:
        raise SpecError(f"{name} must be one of {', '.join(options)}; got {value!r}")


def _json_schema_type(hint: Any) -> dict:
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        return {"anyOf": [_json_schema_type(a) for a in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if origin is list or hint is list:
        return {"type": "array", **({"items": _json_schema_type(args[0])} if args else {})}
    return {"type": {bool: "boolean", int: "integer", float: "number", str: "string", dict: "object",
                     type(None): "null"}.get(hint, "object")}


def schema() -> dict:
    """The JSON Schema of a spec, from each field's setting(): types, defaults, choices, bounds and
    what it means (served at /zentrainer/api/spec/schema). What depends on several fields
    (validate()) is beyond it, so a spec the schema passes can still be refused."""
    def section(dc) -> dict:
        hints, props = get_type_hints(dc), {}
        for f in fields(dc):
            sub = f.default_factory() if callable(f.default_factory) else None
            if is_dataclass(sub):
                props[f.name] = {**section(type(sub)), "description": f.metadata.get("doc", "")}
                continue
            m = f.metadata
            entry = {**_json_schema_type(hints[f.name]), "description": m.get("doc", "")}
            entry["default"] = f.default if f.default is not MISSING else f.default_factory()
            if m.get("choices") is not None:
                entry["enum"] = list(m["choices"])
            for k, js in (("ge", "minimum"), ("gt", "exclusiveMinimum"), ("le", "maximum"), ("lt", "exclusiveMaximum")):
                if m.get(k) is not None:
                    entry[js] = m[k]
            props[f.name] = entry
        return {"type": "object", "description": (dc.__doc__ or "").strip(), "properties": props,
                "additionalProperties": False}

    out = section(TrainSpec)
    out["properties"]["spec_version"]["const"] = SPEC_VERSION
    # a spec file can name this schema, so an editor checks and completes it as it's written
    out["properties"] = {"$schema": {"type": "string", "description": "This schema's address, for editors: ignored."},
                         **out["properties"]}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": f"ZenTrainer run spec, version {SPEC_VERSION}",
            **out, "required": ["model", "dataset"]}


SCHEMA_URL = "https://raw.githubusercontent.com/Nynxz/ZenTrainer/main/docs/spec.schema.json"


def schema_json() -> str:
    """docs/spec.schema.json, which spec files can name ("$schema": SCHEMA_URL)."""
    return json.dumps({**schema(), "$id": SCHEMA_URL}, indent=2) + "\n"


def _ts_type(hint: Any) -> str:
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        return " | ".join(_ts_type(a) for a in args)
    if origin is Literal:
        return " | ".join(json.dumps(a) for a in args)
    if origin is list or hint is list:
        inner = _ts_type(args[0]) if args else "unknown"
        return f"({inner})[]" if "|" in inner else f"{inner}[]"
    return {bool: "boolean", int: "number", float: "number", str: "string", dict: "Record<string, unknown>",
            type(None): "null"}.get(hint, "unknown")


def typescript() -> str:
    """web/src/api/engine/spec.fields.ts: every section's fields and types, which the web's
    TrainSpec (spec.ts) is checked against when the web app type-checks, so a field added or
    renamed here can't be missed there (tests check the file is current)."""
    lines = ["// Generated from engine/zentrainer/spec.py: `python -m zentrainer.spec --ts > web/src/api/engine/spec.fields.ts`",
             "// in engine/. Don't edit: spec.ts checks its TrainSpec against these at type-check time.", "",
             "export interface SpecFields {"]
    top = get_type_hints(TrainSpec)
    for f in fields(TrainSpec):
        sub = f.default_factory() if callable(f.default_factory) else None
        if not is_dataclass(sub):
            lines.append(f"  {f.name}: {_ts_type(top[f.name])}")
            continue
        hints = get_type_hints(type(sub))
        lines.append(f"  {f.name}: {{")
        lines += [f"    {g.name}: {_ts_type(hints[g.name])}" for g in fields(sub)]
        lines.append("  }")
    return "\n".join(lines + ["}"]) + "\n"


def reference() -> str:
    """docs/spec.md: every field, from its setting() (tests check the file is current)."""
    def default(f) -> str:
        v = f.default if f.default is not MISSING else f.default_factory()
        return f"`{json.dumps(v)}`"

    def allowed(f, hint) -> str:
        m = f.metadata
        if m.get("choices") is not None:
            return " or ".join(f"`{json.dumps(c)}`" for c in m["choices"])
        b = _bounds(m)
        return f"{_type_name(hint)}, {b}" if b else _type_name(hint)

    lines = [f"# Run spec reference (version {SPEC_VERSION})", "",
             "<!-- generated from engine/zentrainer/spec.py: `python -m zentrainer.spec > docs/spec.md` in engine/ -->", "",
             "A run spec is the JSON a run is made from: what the run form, `zt train`, the API and exported settings",
             "files all speak. Every field is optional apart from `model` and `dataset.folder`; a missing one takes its",
             "default. A spec without `spec_version` is version 1 and is brought up to date as it's read; a field this",
             "version doesn't have is an error.", "",
             "The same rules as JSON Schema: [spec.schema.json](spec.schema.json), or `/zentrainer/api/spec/schema` from a",
             "running ZenTrainer. Name it at the top of a spec file and editors (VS Code and others) complete and check",
             "it as you type:", "", "```json", "{", f'  "$schema": "{SCHEMA_URL}",',
             f'  "spec_version": {SPEC_VERSION},', '  "model": {"kind": "checkpoint", "ckpt_name": "sdxl.safetensors"},',
             '  "dataset": {"folder": "my_dataset"}', "}", "```", ""]
    top = get_type_hints(TrainSpec)
    lines += ["## Top level", "", "| Field | Default | Allowed | What it does |", "|---|---|---|---|"]
    sections = []
    for f in fields(TrainSpec):
        sub = f.default_factory() if callable(f.default_factory) else None
        if is_dataclass(sub):
            sections.append((f.name, type(sub)))
            lines.append(f"| `{f.name}` | | object | {f.metadata.get('doc', '')} See [{f.name}](#{f.name}). |")
        else:
            lines.append(f"| `{f.name}` | {default(f)} | {allowed(f, top[f.name])} | {f.metadata.get('doc', '')} |")
    for name, dc in sections:
        hints = get_type_hints(dc)
        lines += ["", f"## {name}", "", " ".join((dc.__doc__ or "").split()), "",
                  "| Field | Default | Allowed | What it does |", "|---|---|---|---|"]
        for f in fields(dc):
            doc = f.metadata.get("doc", "").replace("|", "\\|")
            lines.append(f"| `{name}.{f.name}` | {default(f)} | {allowed(f, hints[f.name])} | {doc} |")
    return "\n".join(lines) + "\n"


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "run"


def check_sigmas(sigmas) -> list[float]:
    """Custom sigmas as floats, or SpecError: at least two, none negative, never rising."""
    if not sigmas:
        return []
    try:
        out = [float(x) for x in sigmas]
    except (TypeError, ValueError):
        raise SpecError("samples.sigmas are numbers, high to low (e.g. 1, 0.8, 0.5, 0.2, 0)") from None
    if len(out) < 2:
        raise SpecError("samples.sigmas need at least two values (one step goes from the first to the second)")
    if any(x < 0 for x in out) or any(b > a for a, b in zip(out, out[1:])):
        raise SpecError("samples.sigmas go from high to low and none is below 0")
    return out


def sample_prompts(s: SampleSpec) -> list[SamplePrompt]:
    """The samples with every field filled from the run's defaults."""
    out = []
    for i, p in enumerate(s.prompts):
        d = p if isinstance(p, dict) else {"prompt": p}
        out.append(SamplePrompt(d["prompt"], d.get("width") or s.width, d.get("height") or s.height,
                                d.get("frames") or s.frames, s.seed + i if d.get("seed") is None else d["seed"],
                                d.get("control") or None))
    return out


# what a trained run's saved LoRA, optimizer state and dataset cache depend on: everything else
# (memory, samples, validation, LR and schedule, length, timesteps) may change at a resume
LOCKED_AT_RESUME = ("model", "dataset", "network", "optim.optimizer", "optim.lora_plus", "seed")


def _without_batch(d: dict) -> dict:
    # batch sizes (a stage's and optim.batch_size) shape each step, not the saved LoRA, optimizer state
    # or dataset cache, so a resume may change them
    d = json.loads(json.dumps(d))
    if isinstance(d.get("dataset"), dict):
        d["dataset"]["stages"] = [{k: v for k, v in st.items() if k != "batch_size"} for st in d["dataset"].get("stages") or []]
    return d


def locked_changes(old: dict, new: dict) -> list[str]:
    old, new = _without_batch(old), _without_batch(new)

    def at(d: dict, path: str):
        for k in path.split("."):
            d = d.get(k) if isinstance(d, dict) else None
        return d

    out = []
    for path in LOCKED_AT_RESUME:
        a, b = at(old, path), at(new, path)
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{path}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        elif a != b:
            out.append(path)
    return out


# sample settings a run can change while it trains (the "samples" command): they apply from its
# next sample; the rest load with the run (prompts are encoded, the turbo LoRA attached, at its start)
LIVE_SAMPLES = ("every", "steps", "cfg", "sampler", "scheduler", "sigmas", "turbo_strength", "lora_strength")


def live_samples(spec: dict, changes: dict) -> dict:
    """The run's spec with `changes` to its LIVE_SAMPLES settings, validated. SpecError for any
    other key or a bad value."""
    if not isinstance(changes, dict) or not changes:
        raise SpecError("no sample settings to change")
    other = sorted(set(changes) - set(LIVE_SAMPLES))
    if other:
        raise SpecError(f"these can't change while a run trains: {', '.join('samples.' + k for k in other)} "
                        f"(only {', '.join(LIVE_SAMPLES)})")
    d = json.loads(json.dumps(spec))
    d.setdefault("samples", {}).update(changes)
    if "scheduler" in changes and "sigmas" not in changes:
        d["samples"]["sigmas"] = []  # a scheduler of its own replaces custom sigmas, as a try's does
    new = TrainSpec.from_dict(d)
    new.validate()
    return new.to_dict()


if __name__ == "__main__":
    import sys

    sys.stdout.write(schema_json() if "--schema" in sys.argv else typescript() if "--ts" in sys.argv else reference())
