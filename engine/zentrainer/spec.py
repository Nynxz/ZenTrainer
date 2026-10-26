"""Run specs — the declarative contract between the UI and the harness.

A spec says *what* to train; the harness decides *how*. Pure Python, no ComfyUI
imports, so it can be validated anywhere.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal

ModelKind = Literal["checkpoint", "split"]


@dataclass
class ModelSpec:
    # "checkpoint": one file with MODEL+CLIP+VAE (SD1.5/SDXL style).
    # "split": separate diffusion model / text encoder / VAE (Flux, Qwen-Image, Z-Image, ...).
    kind: ModelKind = "checkpoint"
    ckpt_name: str | None = None
    unet_name: str | None = None
    unet_weight_dtype: str = "default"
    clip_name: str | None = None
    clip_type: str | None = None
    vae_name: str | None = None
    # Joint audio-video models (MiniMax H3): their audio VAE, needed to train on a video's sound
    # (dataset.video_audio) and to put sound in video samples.
    audio_vae_name: str | None = None


@dataclass
class DatasetSpec:
    # A subfolder of ComfyUI's input dir: images + same-name caption files. A
    # kohya-style "N_name" subfolder repeats its images N times.
    folder: str = ""
    resolution: int = 1024
    # Lower resolutions first: [{"resolution": 768, "until": 0.6}] trains at 768 for the first 60%
    # of the run, then at `resolution`. Each change blends over `stage_blend` of the run (0.6 ->
    # 0.8: the next resolution goes from no steps to every step), so nothing switches at once.
    # Every resolution is cached before training. [] = one resolution throughout.
    stages: list = field(default_factory=list)
    stage_blend: float = 0.2
    caption_ext: str = ".txt"
    # Aspect-ratio buckets: images are resized to the closest bucket of ~resolution²
    # pixels, sides multiples of bucket_step. Off = center-crop squares.
    buckets: bool = True
    bucket_step: int = 64
    # Edit training: each image's input picture(s) from the chosen control sets are what the model
    # edits from (passed as reference latents), the caption is the instruction. Models that take
    # reference images (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit...).
    edit: bool = False
    # Control sets, in order: named layers of input pictures beside the dataset
    # (_controls/<set>/<stem>.<ext>; older <stem>_control files are the "control" set). Several
    # sets give each image several inputs. Any set turns edit on; edit with none means "control".
    control_sets: list[str] = field(default_factory=list)
    # how the input picture joins the sequence: "" = the model's own way ("index" for models
    # without one, like Krea 2), or "index", "offset", "index_timestep_zero". "concat": channel
    # inputs instead, each control set's picture (aligned with the target, the same size) as extra
    # latent channels into a widened input layer, as ComfyUI's MOSS-Image control adapters
    # (harness/wide_input.py)
    edit_method: str = ""
    # channel inputs: the share of steps each input is left out (zeros), independently, so the
    # adapter also works with some or none of them
    control_dropout: float = 0.0
    # wrap each instruction: "{}" is the caption (NK2E's generalist was trained on "Make the
    # following edit to the reference image, ...: {}", so specialists on top of it want the same)
    edit_template: str = ""
    # the text encoder sees the input pictures too (Qwen3-VL based: Krea 2, Qwen Image), as
    # krea2edit's identity edits do: conditioning becomes per image, encoded when the dataset caches
    edit_grounded: bool = False
    # each picture's latents (and its own conditioning) in a file of its own, read when a step uses
    # it, in bf16: for picture datasets bigger than RAM (thousands of grounded edit pairs)
    disk_cache: bool = False
    # MiniMax H3 image-to-video: each clip conditioned on its own first frame ("first") or first
    # and last ("first_last"), as the fl2va model is at inference; the text encoder sees them too
    video_keyframes: str = ""
    caption_dropout: float = 0.0
    # Caption sets: <img>.txt is "default", <img>.<set>.txt the others (tags, nl, bbox, ...).
    # Train on one set, or mix several: each step picks one of the image's sets at random.
    caption_set: str = "default"
    caption_mix: list[str] = field(default_factory=list)
    # Trigger word (as in ai-toolkit): "[trigger]" in captions and sample prompts becomes
    # this, and captions that don't mention it get it prepended ("ohwx, a photo of ...").
    trigger: str = ""
    trigger_prepend: bool = True
    # Masked training: images with a mask only learn where it's white. The rest of the
    # image keeps `unmasked_weight` (0 = ignored, 0.1 = learned lightly).
    masked_loss: bool = False
    unmasked_weight: float = 0.0
    # Videos (.mp4 .webm .mov .mkv ...) train as clips: `video_frames` long at most (snapped to a
    # length the model's VAE encodes exactly), resampled to `video_fps` (0 = the video's own),
    # picked by `video_mode` (uniform: spread over the video; head: the start; chunks:
    # consecutive) up to `video_clips` per video, each at about `video_resolution`² pixels (a
    # clip costs frames x pixels, so videos usually train smaller than images). An image model
    # trains a video's frames as images.
    video_frames: int = 49
    video_fps: float = 0.0
    video_mode: str = "uniform"
    video_clips: int = 1
    video_resolution: int = 512
    # Train on each clip's soundtrack too (joint audio-video models with model.audio_vae_name):
    # the audio stream joins the loss instead of being left out, so cuts can learn the beat.
    video_audio: bool = False
    # Keep clips inside one shot: hard cuts are detected and no clip spans one. Off for edits
    # where the cutting is the point (music videos, PMVs).
    video_avoid_cuts: bool = False
    # Clip stills: each clip also trains its sharpest frame as a photo (with the clip's caption).
    # Stills are never TREAD-routed and anchor the LoRA's detail; Fizgig's H3 default.
    video_stills: bool = False


@dataclass
class NetworkSpec:
    algorithm: str = "LoRA"  # LoRA | LoHa | LoKr | OFT (comfy.weight_adapter)
    rank: int = 16
    alpha: float | None = None  # None = rank (scale 1.0)
    # fp32: AdamW's steps on a LoRA are smaller than bf16 can represent at the weights' size,
    # so in bf16 most of them round away (a toy LoRA learned 6% vs 51% in fp32, same steps).
    # bf16 only with an optimizer that rounds stochastically (Automagic).
    lora_dtype: str = "fp32"
    # Which layers get adapters: "attn" (attention projections), "mlp" (feed-forward linears),
    # "attn_mlp" (both), "all" (every adaptable layer, like ComfyUI's native trainer), or a list of
    # regexes over module names.
    targets: str | list[str] = "attn_mlp"
    train_bias_norm: bool = False
    init_from: str | None = None  # a LoRA in the loras folder to continue
    # A LoRA (loras folder) applied while training only, never in samples or the saved file:
    # e.g. a de-distillation adapter for a guidance-distilled model (MiniMax H3), so training
    # sits on an undistilled base and the result runs on the real one.
    training_adapter: str | None = None
    # A LoRA (loras folder) the trained one sits on, everywhere: training, samples and use, e.g.
    # a fix the model always needs (MOSS-Image SFT's destripe LoRA). The saved LoRA leaves it out
    # and records it (zentrainer.base_lora): load it beside, at this strength.
    base_lora: str | None = None
    base_strength: float = 1.0
    # Only these transformer blocks get adapters, e.g. "20-49" (MiniMax H3's identity blocks,
    # Fizgig's default) or "10-": the backward then stops at the first one, so steps are
    # quicker, and the model's early (layout, motion) blocks stay untouched. "" = every block.
    blocks: str = ""
    # These blocks get no adapters, e.g. "22" or "22, 30-31" (AI Toolkit's ignore_if_contains on
    # "layers.22."): a block sampling skips in the uncond (skip-layer guidance) stays the base's
    skip_blocks: str = ""


@dataclass
class OptimSpec:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: str = "constant"  # the learning rate over the run: constant | linear | cosine
    warmup_steps: int = 0
    max_grad_norm: float = 1.0  # 0 = off
    steps: int = 1000
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    # LoRA+: the up (B) matrices learn this many times faster than the down (A) ones, which
    # trains a LoRA faster for the same stability (paper: 16; 4 is a safe start). 0 or 1 = off.
    lora_plus: float = 0.0
    # Weight averaging: checkpoints, samples and the saved LoRA use a running average of the
    # adapter (0.98 = roughly the last 50 steps). 0 = off.
    ema: float = 0.0
    # Automagic v3's (and v4's) adaptive rate never drops below this fraction of its start (0.5 =
    # half). v3's rate peaks early and then decays, and on MOSS-Image runs stalled under-strength
    # (likeness flat after ~500 steps); a floor keeps it learning. v4 settles on its own. 0 = no floor.
    lr_floor: float = 0.0


@dataclass
class ObjectiveSpec:
    loss: str = "mse"  # mse | l1 | huber | smooth_l1
    # "native": each model's own prediction space (eps for SD-style, velocity for
    # flow models). "x0": loss on the denoised image, ComfyUI's native trainer default.
    target: str = "native"
    # uniform | logit_normal (AI Toolkit's "linear" | "sigmoid") | scheduler: the noise levels the
    # ComfyUI scheduler `timestep_scheduler` samples at in `timestep_scheduler_steps` steps, each
    # step as often (trains where the LoRA will be used; few steps for a fast LoRA, as AI Toolkit's
    # two/four/eight_step)
    timesteps: str = "uniform"
    timestep_scheduler: str = "simple"
    timestep_scheduler_steps: int = 20
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # share of steps drawn uniformly instead (logit-normal only): keeps a little training at
    # the extremes it rarely visits (malcolmrey: 0.01)
    uniform_mix: float = 0.0
    # Training timestep shift for flow-matching models (0 = the model's own sampling shift).
    # The sampling shift favours heavy noise (MOSS's 5 puts ~70% of steps above σ≈0.7):
    # composition and colour. A lower one (1-3) spends more steps on detail, e.g. faces.
    shift: float = 0.0
    # Lean the noise levels toward one end, whatever the distribution (AI Toolkit's "content" and
    # "style"): "high_noise" for composition, layout and pose, "low_noise" for detail and texture.
    # The draw is cubed (bias_power 3, as AI Toolkit): high_noise puts half the steps in the
    # noisiest eighth of the distribution. "balanced" = as drawn.
    bias: str = "balanced"
    bias_power: float = 3.0
    # Per-noise-level loss weights. "flex" (flow models) is AI Toolkit's "weighted", measured on
    # Flex.1: up to 1.5x around σ 0.85, down to 0.45x near clean. "balance" measures this model
    # as it trains and evens the pull of every noise level (EDM2's idea, objective.Balance). "" = none.
    weighting: str = ""
    # Noise levels from an even, low-discrepancy sequence instead of random draws: batch 1 covers
    # the schedule in a few steps instead of by luck, a steadier loss for the same distribution.
    spread: bool = False


@dataclass
class MemorySpec:
    training_dtype: str = "bf16"  # bf16 | fp16 | fp32
    gradient_checkpointing: bool = True
    checkpoint_depth: int = 1
    # leave the last N blocks un-checkpointed: they keep their activations (more VRAM) and
    # skip the recompute. Their backward runs right after their forward, so it's the
    # cheapest place to trade memory for speed.
    checkpoint_skip: int = 0
    offloading: bool = False
    # TREAD token routing on video steps (MiniMax H3): this share of a clip's tokens skips the
    # middle blocks (2 to n-3) and rejoins unchanged. 0.5 roughly halves those blocks' work;
    # stills, samples and validation always run whole. 0 = off.
    tread: float = 0.0
    # Share of clip steps that route (the rest run whole). Fizgig's recipe routes every clip but
    # runs stills in full, and stills anchor the LoRA; a dataset of clips only routes every step,
    # and the LoRA learns to make up for the routing: measured on H3 (all clips, 0.5): loss 0.22
    # routed vs 0.15 whole on the same LoRA, and over-sharpened samples. Below 1 keeps whole steps.
    tread_steps: float = 1.0
    # int8 ConvRot bases (MiniMax H3, Krea 2 int8): how the backward gets each frozen layer's
    # input gradient (kernels/int8_backward.py); no effect on other models.
    # "w8a8": the gradient quantized to int8 too, so it runs on int8 tensor cores (Triton kernels:
    #   2x the bf16 backward on H3's clip-size layers, RTX 5090; Krea 2 int8 at 1024 1.60 -> 1.35 s
    #   a step). Some gradient precision for the speed (OneTrainer's W8A8).
    # "fused": a fused Triton kernel instead of rebuilding the layer's bf16 weight (Fizgig's, by
    #   @mabseyuk): saves a transient weight per layer, full gradient precision.
    # "bf16": the weight rebuilt in bf16 for the matmul (ComfyUI's way; the slowest).
    int8_backward: str = "w8a8"
    # torch.compile the repeated transformer block: fuses the elementwise work around the matmuls.
    # The first steps (and the first sample) take longer while it compiles. None (the default) =
    # auto: on for runs long enough to repay that, with Triton and no offloading (kernels/compile.py)
    compile_blocks: bool | None = None
    # Krea 2, MOSS, MiniMax H3: the attention's q/k norms and rope as one op with its own backward (Triton),
    # instead of separate ops and casts (kernels/qk_norm_rope.py): same maths, 6-9% a step uncompiled.
    # None = auto: on while the blocks run uncompiled (compiled ones fuse these already; a custom op
    # in them kept compile from fusing what's around it: Krea 2 1.38 -> 1.42 s a step)
    fused_qk_rope: bool | None = None
    # int8 bases: trained layers run ComfyUI's int8 kernel forward (activations quantized) instead
    # of a dequantized bf16 matmul. Off = the bf16 forward (slower; for comparing quality).
    int8_forward: bool = True
    # With offloading: VRAM kept free for activations, the LoRA and its optimizer (GB);
    # the rest holds as much of the model as fits. 0 = auto: 8 GB, more for long video clips.
    vram_headroom: float = 0.0
    # fp8 bases: the backward's matmuls in fp8 too (ComfyUI's training_fp8_bwd). Faster, less precise
    fp8_backward: bool = False
    # Adapters as forward hooks beside each layer instead of patched into its weight. Quantized
    # bases always train this way (a patched weight would turn their quantized matmul off)
    bypass_mode: bool = False
    # Attention backend while training. Flash and sage have no backward pass, so the
    # default pins differentiable SDPA regardless of ComfyUI's launch flags.
    attention: str = "pytorch"


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
    # score each sample's face against the dataset's (with faces on for the dataset)
    likeness: bool = True
    every: int = 0  # 0 = off
    # each a prompt, or {"prompt", "width"?, "height"?, "frames"?, "seed"?} for its own size
    prompts: list = field(default_factory=list)
    negative: str = ""
    width: int = 1024
    height: int = 1024
    # Video models: frames per sample (1 = a still). Snapped to a length the VAE decodes,
    # saved as an .mp4 at `fps`.
    frames: int = 1
    fps: float = 24.0
    steps: int = 20
    cfg: float = 5.0
    sampler: str = "euler"
    scheduler: str = "normal"
    # Custom sigmas, high to low (e.g. [1.0, 0.94, 0.85, 0.7, 0.5, 0.3, 0.12, 0]): the noise levels
    # sampling steps through, in place of the scheduler and steps (len - 1 steps). [] = the scheduler.
    sigmas: list[float] = field(default_factory=list)
    seed: int = 42
    # Render the prompts once before training: the base model's look, to compare against.
    baseline: bool = True
    # Sample with a different model than the one being trained (e.g. train Krea raw,
    # preview on Krea turbo): a diffusion_models file for split models, a checkpoints
    # file for checkpoint models. Same family, so the training text encoder/VAE are used.
    model_name: str | None = None
    model_weight_dtype: str = "default"
    lora_strength: float = 1.0
    # A turbo LoRA under every sample (not training, not the saved LoRA): few-step previews,
    # e.g. MiniMax H3's minimax_h3_turbo_v4_step600_ema at 0.75 with 6 steps, CFG 1.
    turbo_lora: str | None = None
    turbo_strength: float = 0.75
    # Skip-layer guidance, as ComfyUI's SkipLayerGuidanceDiT nodes do it. "uncond" skips the
    # layers in the negative pass only (no extra cost; MOSS Image's recommended setting);
    # "standard" adds a pass without them and pushes away from it by slg_scale. Layers are
    # the main (single-stream) block indices; slg_double_layers is for double-stream models.
    slg_mode: str = "off"  # off | uncond | standard
    slg_layers: list[int] = field(default_factory=list)
    slg_double_layers: list[int] = field(default_factory=list)
    slg_scale: float = 3.0
    slg_start: float = 0.01
    slg_end: float = 1.0


@dataclass
class ValidationSpec:
    """A smooth "is it learning" curve: loss on fixed images at fixed noise levels with
    fixed noise, so the only thing that changes between evaluations is the LoRA."""
    every: int = 100  # 0 = off
    images: int = 4
    timesteps: int = 4
    seed: int = 1234
    # A held-out dataset folder to measure on instead, prepared like the training one (edit inputs,
    # masks, captions): loss on pictures the LoRA never trains on, so the curve says whether it
    # generalises rather than how well it fits. "" = pictures from the training dataset
    dataset: str = ""


@dataclass
class OutputSpec:
    # A checkpoint (LoRA + resume state) every this many steps. 0 = only the final LoRA
    save_every: int = 250
    # One of ComfyUI's "loras" folders (e.g. your model library's, on a roomier disk);
    # None = output/zentrainer/loras inside this install. LoRAs go in a zentrainer/
    # subfolder of it, so ComfyUI lists them as zentrainer/<run>/<file>.
    lora_dir: str | None = None
    # Resume states (optimizer, schedule, RNG) beside each checkpoint, often bigger than
    # the LoRA: "latest" keeps only the newest (resume from the last save), "all" one per
    # checkpoint, "none" skips them (the run can't be resumed).
    keep_states: str = "latest"
    # What the LoRA files are written in, apart from what trains (network.lora_dtype): bf16 halves
    # an fp32 file and loses nothing a sample shows. The resume state keeps the full weights.
    save_dtype: str = "bf16"  # bf16 | fp16 | fp32
    # Which scheduled checkpoints (save_every) a run keeps: "all"; the `keep_count` "latest"; the
    # `keep_count` best by "likeness" (its samples' faces, a character dataset with faces on) or by
    # "val_loss"; or each new best, "likeness_new_best" / "val_loss_new_best": every save that beat
    # all before it (keep_count unused). The final LoRA, the newest checkpoint (what a resume
    # continues from) and saves asked for by hand or at a stop always stay (harness/keep.py).
    keep: str = "all"
    keep_count: int = 3


@dataclass
class SliderSpec:
    """A slider LoRA (harness/slider.py): one whose strength moves an image along one attribute, +1
    towards `positive`, -1 towards `negative`, 0 the base model.
    "pairs": from picture pairs that differ only in it: the dataset's images are one end, a control
    set of the same names (_controls/<pair_set>/) the other; captions are the pair's.
    "text": a concept slider, its targets made from the base model's own predictions for the
    prompts (no captions needed: the dataset's pictures are only where the noise starts)."""

    kind: str = ""  # "" (a normal LoRA) | "pairs" | "text"
    # pairs: the control set holding each image's partner, and which end the dataset's own are
    pair_set: str = ""
    dataset_end: str = "positive"  # "positive" (the dataset's images are +1) | "negative"
    positive: str = ""  # what +1 adds, e.g. "a person, old, wrinkled skin, grey hair"
    negative: str = ""  # what -1 adds, e.g. "a person, young, smooth skin"
    neutral: str = ""  # what the LoRA trains on, without either: e.g. "a person"
    # prompts whose images the slider should leave alone (at +1 and -1, pulled to the base model)
    anchors: list[str] = field(default_factory=list)
    guidance: float = 3.0  # how far each end is pushed past the base model's own difference
    anchor_weight: float = 1.0
    directions: str = "both"  # "both" | "positive" (only +1 is trained)


@dataclass
class TrainSpec:
    # the spec's layout (SPEC_VERSION): an older one is brought up to date as it's read (migrate)
    spec_version: int = 2
    name: str = "untitled"
    seed: int = 0
    model: ModelSpec = field(default_factory=ModelSpec)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    network: NetworkSpec = field(default_factory=NetworkSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    samples: SampleSpec = field(default_factory=SampleSpec)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    slider: SliderSpec = field(default_factory=SliderSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> TrainSpec:
        """A spec from its dict, any version (migrated first). strict: a field this version doesn't
        have is an error (a typo in a spec written by hand); else it's left out, as migrate does for
        what's stored (runs, presets)."""
        raw, dropped = migrate(data)
        if strict and dropped:
            raise SpecError(f"unknown field(s): {', '.join(dropped)}")

        def build(dc, raw):
            return dc(**(raw or {}))

        nested = {f.name: f.default_factory for f in fields(cls) if callable(f.default_factory)}
        parts = {k: build(dc, raw.pop(k, None)) for k, dc in nested.items()}
        spec = build(cls, raw)
        for k, v in parts.items():
            setattr(spec, k, v)
        spec.validate()
        return spec

    def validate(self) -> None:
        m = self.model
        if m.kind not in ("checkpoint", "split"):
            raise SpecError(f"model.kind must be 'checkpoint' or 'split', got {m.kind!r}")
        if m.kind == "checkpoint" and not m.ckpt_name:
            raise SpecError("model.ckpt_name is required for a checkpoint model")
        if m.kind == "split":
            missing = [k for k in ("unet_name", "clip_name", "clip_type", "vae_name") if not getattr(m, k)]
            if missing:
                raise SpecError(f"split model needs: {', '.join('model.' + k for k in missing)}")
        if not self.dataset.folder:
            raise SpecError("dataset.folder is required")
        last = 0.0
        for st in self.dataset.stages:
            if not isinstance(st, dict) or not isinstance(st.get("resolution"), int) or not isinstance(st.get("until"), (int, float)):
                raise SpecError('dataset.stages are {"resolution": 768, "until": 0.6}')
            if not 256 <= st["resolution"] < self.dataset.resolution:
                raise SpecError(f"a stage's resolution is below the run's ({self.dataset.resolution}) and at least 256")
            if not last < st["until"] < 1.0:
                raise SpecError("dataset.stages end in order, each before the run does (until: 0 to 1)")
            last = st["until"]
        if not 0.0 <= self.dataset.stage_blend < 1.0:
            raise SpecError("dataset.stage_blend is a share of the run, 0 to below 1")
        _choice("network.algorithm", self.network.algorithm, ("LoRA", "LoHa", "LoKr", "OFT"))
        if self.network.base_lora and not -4.0 <= self.network.base_strength <= 4.0:
            raise SpecError("network.base_strength is a LoRA strength, between -4 and 4 (usually 1)")
        _choice("optim.lr_scheduler", self.optim.lr_scheduler, ("constant", "linear", "cosine"))
        _choice("objective.loss", self.objective.loss, ("mse", "l1", "huber", "smooth_l1"))
        _choice("objective.target", self.objective.target, ("native", "x0"))
        _choice("objective.timesteps", self.objective.timesteps, ("uniform", "logit_normal", "scheduler"))
        if self.objective.timesteps == "scheduler":
            if not 1 <= self.objective.timestep_scheduler_steps <= 1000:
                raise SpecError("objective.timestep_scheduler_steps is a step count, 1 to 1000")
            if self.objective.shift:
                raise SpecError("objective.shift is the model's own with a scheduler (its shift is in the schedule): set it to 0")
        _choice("objective.bias", self.objective.bias, ("balanced", "high_noise", "low_noise"))
        _choice("objective.weighting", self.objective.weighting, ("", "flex", "balance"))
        if not 1.0 <= self.objective.bias_power <= 10.0:
            raise SpecError("objective.bias_power is how hard the bias leans, 1 (not at all) to 10 (3 = AI Toolkit's)")
        _choice("memory.training_dtype", self.memory.training_dtype, ("bf16", "fp16", "fp32"))
        _choice("memory.int8_backward", self.memory.int8_backward, ("w8a8", "fused", "bf16"))
        if self.memory.int8_forward is None:  # what the form shows for it: the default, on
            self.memory.int8_forward = True
        sl = self.slider
        _choice("slider.kind", sl.kind, ("", "pairs", "text"))
        _choice("slider.directions", sl.directions, ("both", "positive"))
        _choice("slider.dataset_end", sl.dataset_end, ("positive", "negative"))
        if sl.kind == "pairs":
            if not CONTROL_SET_NAME.match(sl.pair_set or ""):
                raise SpecError("an image slider needs the set holding each picture's partner: slider.pair_set "
                                "(a folder under the dataset's _controls/)")
            if self.dataset.edit:
                raise SpecError("an image slider pairs pictures itself: turn edit training off (dataset.edit)")
        if sl.kind == "text":
            if not (sl.positive.strip() and sl.negative.strip()):
                raise SpecError("a text slider needs both ends: slider.positive and slider.negative")
            if not 0 < sl.guidance <= 20:
                raise SpecError("slider.guidance is above 0 (3 is usual)")
            if sl.anchor_weight < 0:
                raise SpecError("slider.anchor_weight can't be negative")
        _choice("samples.slg_mode", self.samples.slg_mode, ("off", "uncond", "standard"))
        check_sigmas(self.samples.sigmas)
        _choice("output.keep_states", self.output.keep_states, ("latest", "all", "none"))
        _choice("output.save_dtype", self.output.save_dtype, ("bf16", "fp16", "fp32"))
        _choice("output.keep", self.output.keep, ("all", "latest", "likeness", "val_loss", "likeness_new_best",
                                                  "val_loss_new_best"))
        if self.output.keep_count < 1:
            raise SpecError("output.keep_count must be >= 1")
        if self.output.keep in ("val_loss", "val_loss_new_best") and not self.validation.every:
            raise SpecError("keeping the best checkpoints by validation loss needs validation on (validation.every)")
        if self.output.keep in ("likeness", "likeness_new_best") and not (self.samples.prompts and self.samples.likeness):
            raise SpecError("keeping the best checkpoints by likeness needs sample prompts with likeness on "
                            "(samples.prompts, samples.likeness)")
        if self.objective.shift < 0:
            raise SpecError("objective.shift must be >= 0 (0 = the model's own)")
        if not 0.0 <= self.samples.slg_start <= self.samples.slg_end <= 1.0:
            raise SpecError("samples.slg_start/slg_end must satisfy 0 <= start <= end <= 1")
        if isinstance(self.network.targets, str):
            _choice("network.targets", self.network.targets, ("attn", "mlp", "attn_mlp", "all"))
        if self.optim.steps < 1:
            raise SpecError("optim.steps must be >= 1")
        for p in self.samples.prompts:
            if isinstance(p, dict):
                if not isinstance(p.get("prompt"), str):
                    raise SpecError("each sample needs a prompt")
                for k in ("width", "height", "frames", "seed"):
                    if p.get(k) is not None and (not isinstance(p[k], int) or p[k] < (0 if k == "seed" else 1)):
                        raise SpecError(f"sample {k} must be a positive whole number")
                if any(p.get(k) and p[k] % 16 for k in ("width", "height")):
                    raise SpecError("sample width/height must be multiples of 16")
            elif not isinstance(p, str):
                raise SpecError("samples.prompts holds prompts (text) or {prompt, width, height, frames, seed}")
        if self.network.blocks and not re.fullmatch(r"\s*\d*\s*-?\s*\d*\s*", self.network.blocks):
            raise SpecError("network.blocks is a range like 20-49, 20- or 10")
        if self.network.skip_blocks and not re.fullmatch(r"\s*\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*\s*", self.network.skip_blocks):
            raise SpecError("network.skip_blocks is blocks and ranges like 22 or 22, 30-31")
        if not 0.0 <= self.memory.tread < 1.0:
            raise SpecError("memory.tread is a share of the video tokens, from 0 (off) to below 1")
        if not 0.0 <= self.objective.uniform_mix <= 1.0:
            raise SpecError("objective.uniform_mix is a share between 0 and 1")
        if self.dataset.edit_method not in ("", "index", "offset", "index_timestep_zero", "negative_index", "uxo", "concat"):
            raise SpecError("dataset.edit_method is one of index, offset, index_timestep_zero, concat (or empty for the model's own)")
        if not 0.0 <= self.dataset.control_dropout < 1.0:
            raise SpecError("dataset.control_dropout is a share of steps, 0 to below 1")
        if not 0.0 <= self.memory.tread_steps <= 1.0:
            raise SpecError("memory.tread_steps is a share between 0 and 1")
        if any(not isinstance(c, str) or not CONTROL_SET_NAME.match(c) for c in self.dataset.control_sets):
            raise SpecError("dataset.control_sets are set names: lowercase letters, digits, - and _")
        if self.dataset.control_sets:
            self.dataset.edit = True
        elif self.dataset.edit:
            self.dataset.control_sets = ["control"]  # specs from before control sets: the _control files
        if self.dataset.video_keyframes not in ("", "first", "first_last"):
            raise SpecError("dataset.video_keyframes is first, first_last or empty")
        if self.dataset.edit_template and "{}" not in self.dataset.edit_template:
            raise SpecError("dataset.edit_template needs {} where the instruction goes")
        if not 0.0 <= self.optim.lr_floor <= 1.0:
            raise SpecError("optim.lr_floor is a fraction of the start learning rate, 0 (off) to 1")
        if not 0.0 <= self.optim.ema < 1.0:
            raise SpecError("optim.ema is a decay between 0 (off) and below 1, e.g. 0.98")
        if self.optim.lora_plus < 0:
            raise SpecError("optim.lora_plus must be >= 0 (0 = off)")
        if self.samples.frames < 1 or self.samples.fps <= 0:
            raise SpecError("samples.frames must be >= 1 and samples.fps > 0")
        if self.output.save_every < 0 or self.samples.every < 0:
            raise SpecError("output.save_every and samples.every must be >= 0 (0 = off)")
        if self.validation.every < 0 or self.validation.images < 1 or self.validation.timesteps < 1:
            raise SpecError("validation needs every >= 0, images >= 1 and timesteps >= 1")
        if self.validation.dataset and self.validation.dataset == self.dataset.folder:
            raise SpecError("validation.dataset is the training dataset: pick a held-out one, or leave it empty "
                            "to measure on pictures from the training set")
        if self.validation.dataset and self.slider.kind == "pairs":
            raise SpecError("a held-out validation dataset isn't for image sliders: their loss is on the pairs")
        if not 0.0 <= self.dataset.unmasked_weight <= 1.0:
            raise SpecError("dataset.unmasked_weight must be between 0 and 1")
        d = self.dataset
        if d.video_mode not in ("uniform", "head", "chunks"):
            raise SpecError("dataset.video_mode is uniform, head or chunks")
        if d.video_frames < 1 or d.video_clips < 1 or d.video_resolution < 64 or d.video_fps < 0:
            raise SpecError("dataset video settings: frames and clips at least 1, resolution at least 64, fps 0 or more")


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
    version = d.pop("spec_version", 1)
    if not isinstance(version, int) or version < 1:
        raise SpecError(f"spec_version is a whole number from 1, got {version!r}")
    if version > SPEC_VERSION:
        raise SpecError(f"this spec is version {version}, from a newer ZenTrainer (this one reads up to "
                        f"{SPEC_VERSION}): update ZenTrainer")
    for v in range(version, SPEC_VERSION):
        MIGRATIONS[v](d)
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
LOCKED_AT_RESUME = ("model", "dataset", "network", "optim.optimizer", "optim.batch_size", "optim.lora_plus", "seed")


def locked_changes(old: dict, new: dict) -> list[str]:
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
