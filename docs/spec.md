# Run spec reference (version 2)

<!-- generated from engine/zentrainer/spec.py: `python -m zentrainer.spec > docs/spec.md` in engine/ -->

A run spec is the JSON a run is made from: what the run form, `zt train`, the API and exported settings
files all speak. Every field is optional apart from `model` and `dataset.folder`; a missing one takes its
default. A spec without `spec_version` is version 1 and is brought up to date as it's read; a field this
version doesn't have is an error. The same rules as JSON Schema: `/zentrainer/api/spec/schema`.

## Top level

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `spec_version` | `2` | whole number | The spec's layout. An older one is brought up to date as it's read; a newer one (from a newer ZenTrainer) is refused. |
| `name` | `"untitled"` | text | The run's name. |
| `seed` | `0` | whole number, >= 0 | The run's seed: noise, shuffling and dropout. |
| `model` | | object | The model. See [model](#model). |
| `dataset` | | object | The dataset. See [dataset](#dataset). |
| `network` | | object | The adapter. See [network](#network). |
| `optim` | | object | The optimizer and schedule. See [optim](#optim). |
| `objective` | | object | The loss and noise levels. See [objective](#objective). |
| `memory` | | object | Memory and speed. See [memory](#memory). |
| `samples` | | object | Samples. See [samples](#samples). |
| `validation` | | object | Validation loss. See [validation](#validation). |
| `output` | | object | Saving. See [output](#output). |
| `slider` | | object | Slider LoRAs. See [slider](#slider). |

## model

The model to train: one checkpoint file, or a diffusion model, text encoder and VAE.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `model.kind` | `"checkpoint"` | `"checkpoint"` or `"split"` | "checkpoint": one file with the model, text encoder and VAE (SD 1.5, SDXL). "split": separate files (Flux, Qwen Image, Z-Image, ...). |
| `model.ckpt_name` | `null` | text or null | checkpoint models: the file in ComfyUI's checkpoints folder. |
| `model.unet_name` | `null` | text or null | split models: the diffusion model, in diffusion_models. |
| `model.unet_weight_dtype` | `"default"` | `"default"` or `"fp8_e4m3fn"` or `"fp8_e4m3fn_fast"` or `"fp8_e5m2"` | How the diffusion model's weights load (ComfyUI's weight_dtype). |
| `model.clip_name` | `null` | text or null | split models: the text encoder, in text_encoders. |
| `model.clip_type` | `null` | text or null | split models: the text encoder's type, as ComfyUI's CLIPLoader names it (flux2, qwen_image, ...). |
| `model.vae_name` | `null` | text or null | split models: the VAE, in vae. |
| `model.audio_vae_name` | `null` | text or null | Joint audio-video models (MiniMax H3): their audio VAE, needed to train on a video's sound (dataset.video_audio) and for sound in samples. |

## dataset

The pictures and videos, their captions, and how they're prepared.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `dataset.folder` | `""` | text | The dataset folder: pictures and videos with same-name caption files. A kohya-style "N_name" subfolder repeats its pictures N times. |
| `dataset.resolution` | `1024` | whole number, >= 64 | Training resolution: pictures are resized to about this many pixels squared. |
| `dataset.stages` | `[]` | list of object | Lower resolutions first: [{"resolution": 768, "until": 0.6}] trains at 768 for the first 60% of the run, then at `resolution`. Every resolution is cached before training. [] = one resolution throughout. |
| `dataset.stage_blend` | `0.2` | number, >= 0, < 1 | How much of the run each resolution change blends over, so nothing switches at once (0.6 to 0.8: the next resolution goes from no steps to every step). |
| `dataset.caption_ext` | `".txt"` | text | The caption files' extension. |
| `dataset.buckets` | `true` | true/false | Aspect-ratio buckets: each picture is resized to the closest shape of about resolution² pixels. Off = centre-cropped squares. |
| `dataset.bucket_step` | `64` | whole number, >= 8 | Bucket sides are multiples of this. |
| `dataset.edit` | `false` | true/false | Edit training: each picture's input picture(s) from `control_sets` are what the model edits from (as reference latents) and the caption is the instruction. For models that take reference images (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit, ...). Any control set turns it on. |
| `dataset.control_sets` | `[]` | list of text | Control sets, in order: named folders of input pictures beside the dataset (_controls/<set>/<name>.<ext>; older <name>_control files are the "control" set). Several sets give each picture several inputs. Edit training with none uses "control". |
| `dataset.edit_method` | `""` | `""` or `"index"` or `"offset"` or `"index_timestep_zero"` or `"negative_index"` or `"uxo"` or `"concat"` | How the input picture joins the sequence: "" = the model's own way, or "index", "offset", "index_timestep_zero". "concat": channel inputs instead, each control set's picture (aligned with the target, the same size) as extra latent channels into a widened input layer, like ComfyUI's MOSS Image control adapters. |
| `dataset.control_dropout` | `0.0` | number, >= 0, < 1 | Channel inputs: the share of steps each input is left out (zeros), independently, so the adapter also works with some or none of them. |
| `dataset.edit_template` | `""` | text | Wraps each instruction: "{}" is the caption (e.g. a generalist edit model's own wording, for a specialist trained on top of it). "" = the caption as it is. |
| `dataset.edit_grounded` | `false` | true/false | The text encoder sees the input pictures too (Qwen3-VL based: Krea 2, Qwen Image): conditioning becomes per picture, encoded when the dataset caches. |
| `dataset.disk_cache` | `false` | true/false | Each picture's latents (and its own conditioning) in a file of its own, read when a step uses it: for datasets bigger than RAM (thousands of grounded edit pairs). |
| `dataset.video_keyframes` | `""` | `""` or `"first"` or `"first_last"` | MiniMax H3 image-to-video: each clip conditioned on its own first frame ("first") or first and last ("first_last"), as the fl2va model is at inference. "" = off. |
| `dataset.caption_dropout` | `0.0` | number, >= 0, <= 1 | The share of steps that train on an empty caption. |
| `dataset.caption_set` | `"default"` | text | Which captions to train on: <name>.txt is "default", <name>.<set>.txt the others (tags, nl, ...). |
| `dataset.caption_mix` | `[]` | list of text | Several caption sets to mix: each step picks one of the picture's at random. [] = caption_set only. |
| `dataset.trigger` | `""` | text | "[trigger]" in captions and sample prompts becomes this, and captions that don't mention it get it prepended ("ohwx, a photo of ..."). |
| `dataset.trigger_prepend` | `true` | true/false | Prepend the trigger to captions that don't mention it. |
| `dataset.masked_loss` | `false` | true/false | Masked training: pictures with a mask only learn where it's white. |
| `dataset.unmasked_weight` | `0.0` | number, >= 0, <= 1 | Masked training: how much the rest of the picture still trains (0 = not at all, 0.1 = lightly). |
| `dataset.video_frames` | `49` | whole number, >= 1 | Videos train as clips this many frames long at most (snapped to a length the model's VAE encodes exactly). An image model trains a video's frames as pictures. |
| `dataset.video_fps` | `0.0` | number, >= 0 | Clips resampled to this frame rate. 0 = the video's own. |
| `dataset.video_mode` | `"uniform"` | `"uniform"` or `"head"` or `"chunks"` | Which parts of a video become clips: "uniform" spread over it, "head" its start, "chunks" consecutive. |
| `dataset.video_clips` | `1` | whole number, >= 1 | Clips per video, at most. |
| `dataset.video_resolution` | `512` | whole number, >= 64 | Each clip at about this many pixels squared (a clip costs frames x pixels, so videos usually train smaller than pictures). |
| `dataset.video_audio` | `false` | true/false | Train on each clip's soundtrack too (joint audio-video models with model.audio_vae_name), so cuts can learn the beat. |
| `dataset.video_avoid_cuts` | `false` | true/false | Keep clips inside one shot: no clip spans a hard cut. Off for edits where the cutting is the point (music videos). |
| `dataset.video_stills` | `false` | true/false | Each clip also trains its sharpest frame as a picture (with the clip's caption). Stills are never TREAD-routed and anchor the LoRA's detail. |

## network

The adapter that trains: its kind, size, and which layers get one.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `network.algorithm` | `"LoRA"` | `"LoRA"` or `"LoHa"` or `"LoKr"` or `"OFT"` | The adapter (ComfyUI's weight adapters). |
| `network.rank` | `16` | whole number, >= 1 | The adapter's rank. |
| `network.alpha` | `null` | number or null, > 0 | The adapter's alpha. null = the rank (a scale of 1). |
| `network.lora_dtype` | `"fp32"` | `"bf16"` or `"fp16"` or `"fp32"` | What the adapter trains in. fp32: AdamW's steps on a LoRA are smaller than bf16 can represent at the weights' size, so most would round away. bf16 only with an optimizer that rounds stochastically (Automagic). |
| `network.targets` | `"attn_mlp"` | text or list of text | Which layers get adapters: "attn" (attention projections), "mlp" (feed-forward), "attn_mlp" (both), "all" (every adaptable layer, like ComfyUI's native trainer), or a list of regexes over module names. |
| `network.train_bias_norm` | `false` | true/false | Train the biases and norm weights too. |
| `network.init_from` | `null` | text or null | A LoRA in the loras folder to continue from. |
| `network.training_adapter` | `null` | text or null | A LoRA applied while training only, never in samples or the saved file: e.g. a de-distillation adapter for a guidance-distilled model (MiniMax H3), so training sits on an undistilled base. |
| `network.base_lora` | `null` | text or null | A LoRA the trained one sits on everywhere (training, samples and use), e.g. a fix the model always needs. The saved LoRA leaves it out and records it: load it beside, at base_strength. |
| `network.base_strength` | `1.0` | number, >= -4, <= 4 | base_lora's strength. |
| `network.blocks` | `""` | text | Only these transformer blocks get adapters, e.g. "20-49" or "10-": the backward stops at the first one, so steps are quicker, and earlier blocks stay untouched. "" = every block. |
| `network.skip_blocks` | `""` | text | These blocks get no adapters, e.g. "22" or "22, 30-31" (a block that skip-layer guidance skips stays the base's). |

## optim

The optimizer, learning rate and schedule, and the run's length.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `optim.optimizer` | `"AdamW"` | `"AdamW"` or `"Adam"` or `"Adafactor"` or `"Lion"` or `"SGD"` or `"RMSprop"` or `"AdamW8bit"` or `"Lion8bit"` or `"Automagic"` or `"Automagic3"` or `"Automagic4"` or `"Prodigy"` or `"AdamWScheduleFree"` or `"CAME"` | The optimizer. |
| `optim.learning_rate` | `0.0001` | number, > 0 | The learning rate. |
| `optim.weight_decay` | `0.0` | number, >= 0 | Weight decay. |
| `optim.lr_scheduler` | `"constant"` | `"constant"` or `"linear"` or `"cosine"` | The learning rate over the run. |
| `optim.warmup_steps` | `0` | whole number, >= 0 | Steps the learning rate ramps up over, from 0. |
| `optim.max_grad_norm` | `1.0` | number, >= 0 | Gradients clipped to this norm. 0 = off. |
| `optim.steps` | `1000` | whole number, >= 1 | Training steps. |
| `optim.batch_size` | `1` | whole number, >= 1 | Pictures per step. |
| `optim.grad_accumulation_steps` | `1` | whole number, >= 1 | Batches per optimizer step. |
| `optim.lora_plus` | `0.0` | number, >= 0 | LoRA+: the up matrices learn this many times faster than the down ones, which trains a LoRA faster for the same stability (paper: 16; 4 is a safe start). 0 or 1 = off. |
| `optim.ema` | `0.0` | number, >= 0, < 1 | Weight averaging: checkpoints, samples and the saved LoRA use a running average of the adapter (0.98 = roughly the last 50 steps). 0 = off. |
| `optim.lr_floor` | `0.0` | number, >= 0, <= 1 | Automagic v3/v4: the adaptive rate never drops below this fraction of its start (v3's decays and can stall under-strength). 0 = no floor. |

## objective

What the loss compares, and which noise levels each step trains at.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `objective.loss` | `"mse"` | `"mse"` or `"l1"` or `"huber"` or `"smooth_l1"` | The loss. |
| `objective.target` | `"native"` | `"native"` or `"x0"` | "native": each model's own prediction (noise for SD-style models, velocity for flow models). "x0": the loss on the denoised picture (ComfyUI's native trainer). |
| `objective.timesteps` | `"uniform"` | `"uniform"` or `"logit_normal"` or `"scheduler"` | How each step's noise level is drawn: "uniform"; "logit_normal" (AI Toolkit's "sigmoid"), shaped by logit_mean and logit_std; or "scheduler": the levels ComfyUI's scheduler `timestep_scheduler` samples at in `timestep_scheduler_steps` steps, each as often (trains where the LoRA will be used; few steps for a fast LoRA). |
| `objective.timestep_scheduler` | `"simple"` | text | timesteps "scheduler": the ComfyUI scheduler. |
| `objective.timestep_scheduler_steps` | `20` | whole number, >= 1, <= 1000 | timesteps "scheduler": its step count. |
| `objective.logit_mean` | `0.0` | number, >= -3, <= 3 | logit_normal's centre: positive leans to heavy noise (composition), negative to light (detail). |
| `objective.logit_std` | `1.0` | number, > 0, <= 3 | logit_normal's spread. |
| `objective.uniform_mix` | `0.0` | number, >= 0, <= 1 | logit_normal: the share of steps drawn uniformly instead, keeping a little training at the extremes it rarely visits (0.01). |
| `objective.shift` | `0.0` | number, >= 0 | Flow-matching models: the training timestep shift. 0 = the model's own sampling shift, which favours heavy noise (composition, colour); a lower one (1-3) spends more steps on detail, such as faces. |
| `objective.bias` | `"balanced"` | `"balanced"` or `"high_noise"` or `"low_noise"` | Lean the noise levels toward one end, whatever the distribution: "high_noise" for composition, layout and pose, "low_noise" for detail and texture (AI Toolkit's "content" and "style"). "balanced" = as drawn. |
| `objective.bias_power` | `3.0` | number, >= 1, <= 10 | How hard the bias leans: the draw to this power (3, as AI Toolkit, puts half the steps in the noisiest eighth for high_noise). 1 = not at all. |
| `objective.weighting` | `""` | `""` or `"flex"` or `"balance"` | Per-noise-level loss weights. "flex" (flow models): AI Toolkit's "weighted", measured on Flex.1. "balance": measures the model as it trains and evens the pull of every noise level (EDM2's idea). "" = none. |
| `objective.spread` | `false` | true/false | Noise levels from an even, low-discrepancy sequence instead of random draws: a small batch covers the schedule in a few steps, for a steadier loss. |

## memory

Precision, checkpointing, offloading and kernels: how the run fits and how fast it goes.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `memory.training_dtype` | `"bf16"` | `"bf16"` or `"fp16"` or `"fp32"` | The precision the model computes in. |
| `memory.gradient_checkpointing` | `true` | true/false | Recompute activations in the backward instead of keeping them: much less VRAM, some speed. |
| `memory.grad_checkpoint_depth` | `1` | whole number, >= 1 | Gradient checkpointing's granularity. |
| `memory.grad_checkpoint_skip` | `0` | whole number, >= 0 | Leave the last N blocks un-checkpointed: they keep their activations (more VRAM) and skip the recompute, the cheapest place to trade memory for speed. |
| `memory.offloading` | `false` | true/false | Keep part of the model in RAM and stream it in: for models bigger than the card. |
| `memory.vram_headroom` | `0.0` | number, >= 0 | With offloading: the VRAM kept free for activations, the LoRA and its optimizer (GB); the rest holds as much of the model as fits. 0 = auto (8 GB, more for long video clips). |
| `memory.tread` | `0.0` | number, >= 0, < 1 | TREAD token routing on video steps (MiniMax H3): this share of a clip's tokens skips the middle blocks and rejoins unchanged (0.5 roughly halves their work). Stills, samples and validation always run whole. 0 = off. |
| `memory.tread_steps` | `1.0` | number, >= 0, <= 1 | The share of clip steps that route; the rest run whole. A dataset of clips only, routed every step, lets the LoRA learn to make up for the routing (over-sharpened samples): below 1 keeps whole steps. |
| `memory.int8_forward` | `true` | true/false | int8 bases (MiniMax H3, Krea 2 int8): trained layers run ComfyUI's int8 kernel forward instead of a dequantized bf16 matmul. Off is slower: for comparing quality. |
| `memory.int8_backward` | `"w8a8"` | `"w8a8"` or `"fused"` or `"bf16"` | int8 bases: how the backward gets each frozen layer's input gradient. "w8a8": quantized to int8 too, on int8 tensor cores (about 2x the bf16 backward), for some gradient precision. "fused": a fused Triton kernel at full precision, without rebuilding the layer's bf16 weight. "bf16": the weight rebuilt in bf16 (ComfyUI's way; the slowest). |
| `memory.fp8_backward` | `false` | true/false | fp8 bases: the backward's matmuls in fp8 too (ComfyUI's training_fp8_bwd). Faster, less precise. |
| `memory.compile_blocks` | `"auto"` | `"auto"` or `true` or `false` | torch.compile the repeated transformer block, fusing the elementwise work around the matmuls; the first steps (and first sample) are slower while it compiles. "auto": on for runs long enough to repay that, when Triton is installed. |
| `memory.fused_qk_rope` | `"auto"` | `"auto"` or `true` or `false` | Krea 2, MOSS, MiniMax H3: the attention's q/k norms and rope as one op with its own backward (Triton): the same maths, 6-9% a step. "auto": on while the blocks run uncompiled (compiled ones fuse these already). |
| `memory.bypass_mode` | `false` | true/false | Adapters as forward hooks beside each layer instead of patched into its weight. Quantized bases always train this way. |
| `memory.attention` | `"pytorch"` | `"pytorch"` or `"xformers"` or `"sub_quad"` or `"split"` | The attention backend while training (only ones with a backward pass; flash and sage have none). |

## samples

Sample pictures (or clips) during the run: what, how often, and how they're made.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `samples.likeness` | `true` | true/false | Score each sample's face against the dataset's (with faces on for the dataset). |
| `samples.every` | `0` | whole number, >= 0 | Sample every this many steps. 0 = off. |
| `samples.prompts` | `[]` | list of text or object | The samples: each a prompt, or {"prompt", "width", "height", "frames", "seed", "control"} with its own size, length, seed or (edit runs) picture to edit. |
| `samples.negative` | `""` | text | The negative prompt. |
| `samples.width` | `1024` | whole number, >= 16 | Sample width. |
| `samples.height` | `1024` | whole number, >= 16 | Sample height. |
| `samples.frames` | `1` | whole number, >= 1 | Video models: frames per sample (1 = a still), snapped to a length the VAE decodes and saved as an .mp4 at fps. |
| `samples.fps` | `24.0` | number, > 0 | Video samples' frame rate. |
| `samples.steps` | `20` | whole number, >= 1 | Sampling steps. |
| `samples.cfg` | `5.0` | number, >= 0 | CFG scale. |
| `samples.sampler` | `"euler"` | text | ComfyUI's sampler. |
| `samples.scheduler` | `"normal"` | text | ComfyUI's scheduler. |
| `samples.sigmas` | `[]` | list of number | Custom sigmas, high to low (e.g. [1.0, 0.94, 0.85, 0.7, 0.5, 0.3, 0.12, 0]), in place of the scheduler and steps (len - 1 steps). [] = the scheduler. |
| `samples.seed` | `42` | whole number, >= 0 | The first sample's seed; each next prompt the next one. |
| `samples.baseline` | `true` | true/false | Render the prompts once before training: the base model's look, to compare against. |
| `samples.model_name` | `null` | text or null | Sample with another model of the same family than the one training (e.g. train a raw model, preview on its turbo): a diffusion_models file for split models, a checkpoints file for checkpoint ones. |
| `samples.model_weight_dtype` | `"default"` | `"default"` or `"fp8_e4m3fn"` or `"fp8_e4m3fn_fast"` or `"fp8_e5m2"` | How model_name's weights load. |
| `samples.lora_strength` | `1.0` | number | The trained LoRA's strength in samples. |
| `samples.turbo_lora` | `null` | text or null | A turbo LoRA under every sample (not training, not the saved LoRA): few-step previews. |
| `samples.turbo_strength` | `0.75` | number | turbo_lora's strength. |
| `samples.slg_mode` | `"off"` | `"off"` or `"uncond"` or `"standard"` | Skip-layer guidance, as ComfyUI's SkipLayerGuidanceDiT: "uncond" skips the layers in the negative pass only (no extra cost), "standard" adds a pass without them and pushes away from it by slg_scale. |
| `samples.slg_layers` | `[]` | list of whole number | Skip-layer guidance: the main (single-stream) blocks to skip. |
| `samples.slg_double_layers` | `[]` | list of whole number | Skip-layer guidance: double-stream blocks, for models that have them. |
| `samples.slg_scale` | `3.0` | number, >= 0 | Skip-layer guidance "standard": how hard it pushes. |
| `samples.slg_start` | `0.01` | number, >= 0, <= 1 | Skip-layer guidance: where in sampling it starts (0 to 1). |
| `samples.slg_end` | `1.0` | number, >= 0, <= 1 | Skip-layer guidance: where it ends (0 to 1). |

## validation

A smooth "is it learning" curve: loss on fixed pictures at fixed noise levels with fixed noise, so the only thing that changes between evaluations is the LoRA.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `validation.every` | `100` | whole number, >= 0 | Measure every this many steps (and before training). 0 = off. |
| `validation.images` | `4` | whole number, >= 1 | How many pictures to measure. |
| `validation.noise_levels` | `4` | whole number, >= 1 | Noise levels each picture is measured at, spread over the schedule. |
| `validation.seed` | `1234` | whole number, >= 0 | Picks the pictures and their noise. |
| `validation.dataset` | `""` | text | A held-out dataset folder to measure on instead, prepared like the training one (edit inputs, masks, captions): loss on pictures the LoRA never trains on, so the curve says whether it generalises rather than how well it fits. "" = pictures from the training dataset. |

## output

Where the LoRAs go, how often, in what, and which checkpoints stay.

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `output.save_every` | `250` | whole number, >= 0 | A checkpoint (the LoRA and its resume state) every this many steps. 0 = only the final LoRA. |
| `output.lora_dir` | `null` | text or null | One of ComfyUI's loras folders to save in (e.g. on a roomier disk). null = output/zentrainer/loras. LoRAs go in a zentrainer/<run>/ subfolder of it. |
| `output.keep_states` | `"latest"` | `"latest"` or `"all"` or `"none"` | Resume states (optimizer, schedule, RNG) beside each checkpoint, often bigger than the LoRA: "latest" keeps only the newest, "all" one per checkpoint, "none" skips them (the run can't be resumed). |
| `output.save_dtype` | `"bf16"` | `"bf16"` or `"fp16"` or `"fp32"` | What the LoRA files are written in (training keeps network.lora_dtype): bf16 halves an fp32 file and loses nothing a sample shows. |
| `output.keep` | `"all"` | `"all"` or `"latest"` or `"likeness"` or `"val_loss"` or `"likeness_new_best"` or `"val_loss_new_best"` | Which scheduled checkpoints a run keeps: "all"; the keep_count "latest"; the keep_count best by "likeness" (its samples' faces) or by "val_loss"; or each new best, "likeness_new_best" / "val_loss_new_best". The final LoRA, the newest checkpoint and saves asked for by hand always stay. |
| `output.keep_count` | `3` | whole number, >= 1 | How many checkpoints "latest", "likeness" and "val_loss" keep. |

## slider

A slider LoRA: one whose strength moves a picture along one attribute, +1 towards `positive`, -1 towards `negative`, 0 the base model. "pairs" learns it from picture pairs that differ only in it; "text" from the base model's own predictions for the prompts (no captions needed).

| Field | Default | Allowed | What it does |
|---|---|---|---|
| `slider.kind` | `""` | `""` or `"pairs"` or `"text"` | "" (a normal LoRA), "pairs" or "text". |
| `slider.pair_set` | `""` | text | pairs: the control set holding each picture's partner (_controls/<pair_set>/, the same names). |
| `slider.dataset_end` | `"positive"` | `"positive"` or `"negative"` | pairs: which end the dataset's own pictures are. |
| `slider.positive` | `""` | text | What +1 adds, e.g. "a person, old, wrinkled skin, grey hair". |
| `slider.negative` | `""` | text | What -1 adds, e.g. "a person, young, smooth skin". |
| `slider.neutral` | `""` | text | text: what the LoRA trains on, without either end, e.g. "a person". |
| `slider.anchors` | `[]` | list of text | text: prompts the slider should leave alone (pulled to the base model at +1 and -1). |
| `slider.guidance` | `3.0` | number, > 0, <= 20 | text: how far each end is pushed past the base model's own difference. |
| `slider.anchor_weight` | `1.0` | number, >= 0 | text: the anchors' weight. |
| `slider.directions` | `"both"` | `"both"` or `"positive"` | "both", or "positive" (only +1 is trained). |
