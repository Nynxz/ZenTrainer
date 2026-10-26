# ZenTrainer architecture

ZenTrainer is a training and app UI that uses ComfyUI as its engine instead of
reimplementing one. ai-toolkit, OneTrainer and similar tools each carry their own model
loaders, attention code, offloading, quantization and sampling. ComfyUI already has all
of these, kept current for every new model family. ZenTrainer owns the training loop
and calls ComfyUI **as a library** for everything else.

## Shape

```
 ┌──────────── web (Vue + zenkit-ui) ─────────────────────┐
 │ sidebar: New run · Runs · Datasets · Apps · Settings   │
 │ api/engine/ (REST + control)     api/comfy.ts (/ws,    │
 │                                       /object_info)    │
 └──────────────┬────────────────────────────┬────────────┘
                │ /zentrainer/api/*          │ stock ComfyUI API
 ┌──────────────▼────────────────────────────▼────────────┐
 │ ComfyUI process                                        │
 │  engine/zentrainer                                     │
 │   server/      routes (HTTP API), jobqueue, telemetry  │
 │   nodes.py     ZenTrainSession: hosts one run          │
 │   runs/        run store (user/zentrainer/runs/<id>/)  │
 │   harness/                                             │
 │     trainer.py    the loop: setup → steps → save       │
 │     models.py     comfy.sd loaders (non-dynamic copy)  │
 │     data/         scan, buckets, cached VAE/CLIP encode│
 │     network.py    comfy.weight_adapter LoRA/LoHa/…     │
 │     denoiser.py   process_conds → apply_model          │
 │     objective.py  targets, timestep sampling, losses   │
 │     optim/        optimizers + LR schedules            │
 │     callbacks.py  metrics, checkpoints, samples        │
 │     control.py    pause/resume/stop/save/sample/set_lr │
 │     memory/       checkpointing, offload streams, TREAD│
 │     kernels/      int8 backward, torch.compile         │
 │  comfy/  model mgmt, offload, quant ops, attention,    │
 │          model_sampling, samplers, folder_paths        │
 └────────────────────────────────────────────────────────┘
```

### How a run executes

1. `POST /zentrainer/api/runs` validates the `TrainSpec`, writes it to the run store and
   adds it to **ZenTrainer's queue** (`server/jobqueue.py`): an ordered, persistent list that
   can be reordered, pruned or paused. A worker hands the head of the queue to ComfyUI
   as a **one-node** job, `ZenTrainSession(run_id)`, only when nothing is training.
2. That job exists only to claim ComfyUI's execution lane, and with it the GPU, for the
   whole run. It also makes ComfyUI's own cancel button work. Nothing about the run is
   encoded in the graph.
3. The node constructs a `Trainer` from the stored spec and runs it in-process.
4. While it trains, `POST /runs/<id>/control` pushes commands into a queue. The loop
   drains the queue between optimizer steps, so every command lands at a clean step
   boundary. Commands sent while paused apply on resume.
5. **Resume.** Every checkpoint writes a `.state.pt` (optimizer, LR schedule, step, RNG)
   beside the LoRA, including the automatic one on error/stop/interrupt. Resume puts the
   run back at the front of the queue from its latest such checkpoint (metrics past it
   are dropped and re-trained); with none, it restarts. Runs left "training" by a dead
   process are marked interrupted at startup.
6. Events (`zentrainer.metrics|checkpoint|sample|status`) go out over ComfyUI's `/ws` and
   are also appended to the run store, so a reconnecting tab rebuilds full history.
   Runs mirror `zenkit.job` events, so ZenKit's taskbar shows training too.

### What the harness takes from ComfyUI

| Need | ComfyUI piece | Notes |
|---|---|---|
| Model files | `folder_paths` + `extra_model_paths.yaml` | Existing checkpoints, diffusion_models, VAEs and text encoders are used in place, with no re-download. Trained LoRAs land in a folder registered as a `loras` path. |
| Loading | `comfy.sd.load_checkpoint_guess_config` / `load_diffusion_model` with `disable_dynamic=True` | A private copy: the harness patches forwards and attaches adapters, so it must not share modules with the server's inference cache. |
| Conditioning | `process_conds` → `get_area_and_mult` → `cond_cat` → `apply_model` | The same path the samplers use, minus CFG, so every family's `extra_conds` (SDXL pooled/size, Flux guidance, …) works unchanged. |
| Noise schedule | `model_sampling` (`percent_to_sigma`, `noise_scaling`) | Family-agnostic. The "native" target `(x_t − x0)/σ` is ε for SD-style models and velocity for flow models. |
| Layer targets | structural: a layer is "attn"/"mlp" when an enclosing module's name says so | Covers SD, Flux, Krea 2 (`attn.wq`, `mlp.up`), Llama-style blocks without per-family lists. |
| Adapters | `comfy.weight_adapter` + `ModelPatcher.add_weight_wrapper` (or bypass hooks) | Files load with ComfyUI's LoraLoader (tested: 160/160 pairs, 0 unloaded keys). Adapters are pinned to the training device so offloading never moves optimizer-owned params. |
| Memory | `load_models_gpu` (partial/lowvram), offload-aware checkpointing, `training_fp8_bwd` | Async offload and pinned memory come from ComfyUI's loader. Not yet profiled under training. |
| Attention | `optimized_attention_override` in transformer_options | **flash and sage have no backward.** Each run pins a differentiable backend, so a server launched with `--use-sage-attention` still trains. |
| comfy-kitchen | quant kernels | **Forward only.** The training speed path is fp8 base weights plus FP8 `_scaled_mm` in backward. |
| Quantized bases (int8/fp8/nvfp4) | `QuantLinearFunc` keeps the quantized matmul in training | A weight wrapper disables it (full dequant per forward), so quantized models train in **bypass mode automatically**. |
| Checkpointing | `patch` from nodes_train, applied **per block** | The native `find_modules_at_depth(depth=1)` returns the model root, i.e. one segment for the whole net: no activation savings (this OOM'd Krea 2 on a 32 GB card). We checkpoint each child of the outermost block lists. Likely the same bug in native TrainLoraNode. |
| Samples | `comfy.sample.sample` on the training patcher | Rendered in-loop with the in-progress adapters, then the training load is restored. Tested on SD1.5. |

### Upstream seams

- `harness/memory/` imports `patch`/`unpatch`/`find_modules_at_depth` from
  `comfy_extras/nodes_train.py`. That's the only private import. The proposal is to move
  these helpers into a public `comfy/train/` module.
- `harness/compat/` patches ComfyUI model code that samples fine but can't train, one
  module per model, only while `in_training` is set, so the pinned checkout stays
  untouched. MiniMax H3: the training branch of its attention calls comfy_kitchen's fused
  RMSNorm+rope, which has no backward (replaced by the plain-torch partial split-half
  rope), and `_mod_gate` adds into the residual stream in place after the norms saved it
  (made out of place). FLUX.3 fills its joint sequence with `cat(out=...)` and adds its
  residuals in place, both out of place while gradients are on. Each is the fix to send
  upstream; delete the module once ComfyUI has it.
- Possible bug in native `TrainLoraNode`: in bucket and multi-res modes the latents
  appear to skip `process_latent_in`. Unverified; worth a test before reporting.

### Gotchas found while building (each was a real failure)

- ComfyUI runs nodes under `torch.inference_mode()`. The harness disables it, and then
  the frozen VAE/CLIP parameters still `require_grad`. Dataset encodes **must** run
  under `no_grad`, or every cached latent carries an autograd graph and step N+epoch
  fails with "backward through the graph a second time".
- ComfyUI imports custom nodes under a path-based module name, so pickling the harness's
  own classes ties a cache file to one install path. The dataset cache stores plain
  tensors, lists and dicts only.
- ComfyUI's model compiler (comfy_aimdo malloc graphs) is off for a whole run, samples
  included, and while captioning. With it on, training fails ("aimdo memory compile
  error"), token-by-token caption generation hits a device-side assert that kills the
  process, and MiniMax H3 (which drives it from its own forward) aborts sampling a
  partly offloaded, LoRA-wrapped model.
- Offloading must give `load_models_gpu` a real headroom. `memory_required=1e20` leaves
  a weight budget of 0.1 bytes on NVIDIA (`MIN_WEIGHT_MEMORY_RATIO` is 0), so every layer
  crossed PCIe twice a step. Now whatever fits beside 8 GB of headroom stays resident,
  and a model that can't fit whole turns offloading on by itself.
- The text encoder is loaded lazily and released after on_start (captions and sample
  prompts are both cached), and the model is loaded for training again after on_start,
  because encoding prompts can push it out of VRAM.

## Training quality and control

- **Validation loss** (`callbacks.Validation`): fixed images × fixed noise levels × fixed
  noise, evaluated before training and every N steps. Unlike the per-step loss (random
  timestep each step), it only moves when the LoRA does: the "is it learning" curve.
- **Baseline samples** at step 0 (the untouched model), and **sampling with another
  model** (train on Krea raw, preview on turbo): the alternate model gets the current
  LoRA through ComfyUI's own `load_lora_for_models`.
- **Trigger word**, ai-toolkit style: `[trigger]` in captions and prompts becomes the
  word; captions that lack it get it prepended.
- **Optimizers** (`harness/optim/`): AdamW, Adam, Adafactor, Lion, SGD, RMSprop built
  in; AdamW/Lion 8-bit, Prodigy, schedule-free AdamW and CAME when their packages are
  installed (`scripts/zt extras optimizers`). Schedule-free switches to its eval
  weights for sampling, validation and saving.
- **Presets** (`presets.py`): partial specs the user saves that layer, and sample-prompt
  sets, under `<user>/zentrainer/presets/`. The run form's ready-made starting points are
  Simple mode's recipes (`web/src/lib/recipes.ts`), applied from its "Apply a recipe" dialog.
- **Timing**: `train_time` counts optimizer steps only; `it_s` is per step, so sampling
  and validation don't read as slowdowns.

## Data, memory, queue

- **Caption sets**: `<img>.txt` is "default", `<img>.<set>.txt` the others (tags, nl,
  bbox, ...). A run trains on one set or mixes several (random per step).
- **Auto-captioning** (`datasets/captioner.py`): ComfyUI's native text generation (the
  TextGenerate path: `clip.tokenize(image=…)` → `generate` → `decode`) with a
  vision-language text encoder, writing into a caption set. Runs as a one-node
  `ZenCaptionSession` job so it takes its turn on the GPU. Jobs persist
  (`<user>/zentrainer/caption_jobs.json`) with what they wrote and what they replaced,
  so a job can be reviewed and undone (an edit made since wins). Modes: missing, all,
  or selected images; a preview captions a few without writing. Rewriting existing
  captions in "default" needs an explicit `replace_default`. Videos caption from their
  middle frame.

- **Multi-stream models** (`harness/data/streams.py`): MiniMax H3 denoises video and stereo
  audio together. As ComfyUI's sampler does, the streams are packed into one flat latent
  with their shapes (`latent_shapes`, which conds need too). The dataset caches only
  the picture; the other streams are the model's own empty ones (`fix_empty_latent`),
  and a per-element loss weight leaves them out, so a LoRA learns the picture without
  unlearning the sound. Batch size 1 (the model's limit); use gradient accumulation.
- **Videos** (`harness/data/video.py`): videos sit in datasets beside images and train as
  clips. Each clip is resampled to `video_fps` and cut to the longest length up to
  `video_frames` that the VAE encodes exactly. Valid lengths come from the VAE's own
  temporal formulas: the counts it encodes and decodes back unchanged, 4k+1 for Wan,
  17k+5 for H3, 1 for image VAEs (so an image model trains a video's frames as images).
  `video_mode` picks the windows: uniform spread, head, or back-to-back chunks, up to
  `video_clips` per video. Each window caches as its own latent (`Dataset.owners` maps
  clips back to their video for repeats) at about `video_resolution`² pixels. The video
  settings join the cache key only when a dataset has videos.
- **Masked training**: a mask per image (`_masks/<stem>.png`, `<stem>-masklabel.png` or
  `<stem>.mask.png`; white = learn) gets the image's exact crop, is stored at latent
  resolution, and weights the loss (`unmasked_weight` for the rest), normalised per
  sample so masked and unmasked images stay on one scale.
- **Live sampling**: `zentrainer.sample_progress` events carry the prompt/step and a
  latent preview (ComfyUI's previewer, latent-to-RGB fallback when previews are off).
- **Datasets** (`datasets/folders.py`): list, per-image view with sizes, caption edit, bulk
  prefix/suffix/replace, upload, delete, create; every path is checked to stay inside
  its folder. The scanner skips `.`/`_` folders (other trainers' caches and thumbnails).
- **Faces** (`analysis/faces.py`, `datasets/faces.py`): InsightFace's SCRFD detector and
  ArcFace (buffalo_l, antelopev2) in plain PyTorch (`analysis/nets/`), with InsightFace's
  letterbox, anchors, NMS and 5-point alignment, and OpenCV's resize and warp done in numpy.
  The weights are safetensors in `nynxz/Zentrainer` (pinned by commit), downloaded into
  `models/detection/zentrainer/` through the download manager the first time faces run. The
  networks' parameter names are that file format (`tests/engine/test_faces_model.py` holds
  their fingerprints). `datasets/faces.py` builds the face reports, per-model caches and
  likeness on top; it all runs on the CPU, so it never takes VRAM from a run.
- **Memory** (`system.py`): VRAM (device, PyTorch allocated/reserved, comfy_aimdo) and
  RAM (system, process) sampled at 1 Hz into a 15-minute buffer; sidebar meters and a
  System page with charts. The System page can restart ComfyUI (`os.execv` of the same
  command); refused while a run trains unless forced.

## Model families and downloads

Picking a model is one choice (Krea 2, Z-Image, SDXL…), not three files.
`models/families.py` builds the catalog from ComfyUI's own workflow templates (the
`comfyui_workflow_templates` package that ships with ComfyUI), so a new ComfyUI brings new
families and nothing here is hand-maintained.

- **Where families come from.** Each open text-to-image template names its loaders' files,
  the folder each goes in, and a download URL (`properties.models`). Templates in a shape the
  harness trains count: one checkpoint, or one diffusion model + one text encoder + one VAE.
  Split models are grouped by their diffusion model's Hugging Face repo (every Qwen-Image
  template shares one). Checkpoints are grouped by the template index's model name, taking
  the most specific one, since the index isn't consistent about order. A prompt-upsampler
  LLM loaded through `CLIPLoader` (Flux 3 Klein's Qwen3.5) isn't counted as the text encoder.
  A template named "distilled" marks its model distilled, along with that build's other
  precisions in the repo, whatever the files are called.
- **Repos without a template.** ComfyUI supports more models than it has templates for
  (MOSS Image landed in ComfyUI with no template), so Hugging Face repos are a second source:
  every Comfy-Org repo, plus the ComfyUI-layout repos ("…-comfy") that a short list of
  model makers (`MAKERS`: Black Forest Labs, Stability, Qwen…) publish themselves. A repo laid out as diffusion model + text encoder + VAE,
  or as checkpoints, becomes a family. Its text encoder type is matched by name against
  ComfyUI's `CLIPLoader` types (`MOSS-Image` → `moss_image`), and a repo with no confident
  match is skipped rather than guessed. Video/audio/3D repos, add-ons, and repos whose
  templates load shapes the harness can't train are excluded. These families are labelled
  "from repo".
- **Dates and gates.** Each family has an `added` date: its first template's `date`, or the
  repo's `createdAt`. The list comes newest first and families from the last 30 days are
  tagged "new". A gated repo (Hugging Face `gated`) is flagged, with a hint to accept the
  license and set `HF_TOKEN`.
- **Base builds.** Templates are inference workflows and mostly load turbo/distilled builds,
  which train poorly. So a family also lists every build in its repo through the Hugging Face
  API, cached a day under `user/zentrainer/cache/hf/` and prefetched once in the background.
  The recommendation is an installed base build, then the base build in the template's
  precision, then the template's file.
- **Samples.** The family's template sampler settings (steps, CFG, sampler, scheduler) and
  an installed distilled build become a one-click "preview faster on turbo" suggestion.
- **Profiles** (`models/profiles.py`, `engine/profiles/*.json`). Settings hand-picked for one
  family, beyond the generic defaults and the templates: partial training and sample settings,
  notes, and the sources they come from. `status` is `recipe` (the makers' published settings,
  applied as written) or `verified` (trained with and checked). A profile holds only what its
  source says, never guessed learning rates. The picker shows a tick on profiled families, and
  picking one applies the profile. A card under the picker shows the notes and an Apply button
  once the form drifts from it. MOSS Image is the first one: Euler, 50 steps, CFG 3, layer 22
  skipped in the negative pass.
- **Skip-layer guidance** (`samples.slg_*`). Samples can use ComfyUI's own
  `SkipLayerGuidanceDiTSimple` (skip layers in the negative pass) or `SkipLayerGuidanceDiT` (an
  extra pass). The node code runs against a stand-in that only collects sampler options, which
  are swapped onto the sampling model for the render and restored after, because cloning the
  training patcher would reload its weights.
- **Install state.** Each file is matched by basename against ComfyUI's own loader lists,
  so files in subfolders and `extra_model_paths` libraries count.
- **Downloads** (`models/downloads.py`). Callers name a family/folder/file, never a URL, so only
  catalog files can be fetched, and only to a vetted file name. A worker runs one download
  at a time, resumes from `.part` with HTTP ranges, checks free space first and reports on
  `zentrainer.download`. `HF_TOKEN` is passed for gated repos. Files land in the first
  folder of that type outside this install's own `models/` (the user's library through
  `extra_model_paths`), so the main ComfyUI sees them too.
- **UI.** New Run's model picker and the Models page share `FileState`: installed, a
  download button with its size, or progress with cancel. Any file can still be picked by
  hand under "Custom".

## Frontend

Vue + `@nynxz/zenkit-ui`, served by ComfyUI at `/zentrainer/` (dev: Vite proxy). The
look follows lora.moe: tight large titles, soft cards and pill controls. It has a
collapsible app sidebar with a live card per training run, and a phone overlay below
720px.

The log drawer (sidebar "Log", or Ctrl+`) shows ComfyUI's console live. It overlays the
bottom of the main area, so the page never moves. `web/src/stores/logs.ts` backfills from
`/internal/logs/raw` and subscribes with `PATCH /internal/logs/subscribe`. ComfyUI drops
that subscription when the socket closes, so it's renewed on each reconnect's `status`
message. Raw writes are stitched into lines: ANSI stripped, `\r` progress bars collapsed
to their latest state, a Python traceback kept as one block, and the level taken from
ComfyUI's `[WARNING]`/`[ERROR]` tags (keywords for untagged output). It keeps 5000 lines.
Errors that arrive while the drawer is closed badge the sidebar row. The drawer's Memory tab charts VRAM and
RAM with the System page's series (`web/src/lib/memSeries.ts`), sized to the drawer. Model
loads/unloads and errors from the log are marked on the time axis. ComfyUI's zoneless local
log times are read as browser-local, which lines up when both run on one machine.

Themes are ZenKit's JSON packs, which are shadcn semantic tokens. They're fetched from
`/zenkit/themes` when ComfyUI-ZenKit is installed, with a built-in fallback otherwise.
Settings shows lora.moe-style pack cards drawn from each pack's own tokens.
`?theme=&mode=` lets an embedding host pass its theme in.

ZenKit follow-ups:

- **Publish `@nynxz/zenkit-theme` and `@nynxz/zenkit-ui`.** `web/` links both from the sibling
  checkout today (vite dedupes `vue`). zenkit-ui gained lightbox features ZenTrainer uses:
  `overlay` and `thumb` items, a `#side` panel, `grid` navigation (WASD), `searchable`, and
  keys that leave Ctrl/Cmd/Alt combos to the browser.
- **Light base tokens in zenkit-theme.** `BASE_TOKENS` are dark-only. `web/src/lib/theme.ts`
  carries a `LIGHT_BASE` workaround that belongs upstream.

## ComfyUI version

Settings shows the running ComfyUI's version and commit against the pin in
`comfy.lock`, and on request fetches upstream to say how many commits it's behind.

## Roadmap

See also [APPS.md](APPS.md) for the apps, Assets and onboarding design (apps are
experimental; training comes first).

1. **Per-family training defaults.** Families know their files; next they should carry
   what to train with: preset, learning rate, resolution, timestep shift.
2. **Settings store.** A registry of settings saved in ComfyUI's per-user settings (theme,
   motion, per-family overrides), rendered by the Settings page. Includes a reduce-motion
   setting that ZenKit honours.
3. **Multi-encoder models.** Templates that load two or more text encoders as separate
   parts (Flux.1 dev, SD3, HiDream) are skipped: the harness takes one. Their all-in-one
   checkpoints are offered.
4. **GPU runs on more families**, checking memory flags, offloading and fp8 + quantized
   backward under load.
5. **Outputs and Assets**: a library of what training and apps produced.
