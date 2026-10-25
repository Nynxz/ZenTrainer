# Changelog

What changed in each ZenTrainer release, newest first. Versions are the ones the ComfyUI Registry
and Manager show.

## Unreleased

### AMD
- **AMD GPUs on Windows and Linux:** `zt setup` finds a Radeon or Instinct card from its display
  driver alone and installs PyTorch's ROCm build from AMD's wheel index, with only that GPU's
  kernels; no ROCm or HIP SDK install is needed. The ROCm build the setup used to pick no longer
  existed for the pinned PyTorch, so an AMD machine set up before this one gets the new build on
  `zt update`. `zt doctor` and first-run setup list AMD GPUs.
- **Fitting weights in VRAM on Windows with an AMD GPU** now counts what the desktop and other
  programs use: ROCm only reported ZenTrainer's own memory there, so the fit could plan with VRAM
  that was already taken. The memory chart shows the whole card too.

### Cloud
- **Train on a rented GPU:** a container image (`docker/`) for RunPod and other cloud GPUs, or any
  machine with Docker. The app sits behind a password, ComfyUI is never reachable from outside,
  SSH takes keys only, and your datasets, runs, LoRAs and models stay on the volume. Setup for
  RunPod is in docker/README.md; the image is `ghcr.io/nynxz/zentrainer`.

### Datasets
- **Faces need nothing installed:** face reports and likeness scores now run on ZenTrainer's own
  port of InsightFace's models, in plain PyTorch, so the faces extra is gone (no insightface,
  onnxruntime or OpenCV, and none of their clashes with other custom nodes). Scores are the same
  as before (existing face reports and likeness stay valid), and it's a little faster. The face
  model downloads the first time faces run: 104 MB for Standard, 147 MB for Larger, into
  `models/detection/zentrainer/`. Models InsightFace downloaded to `~/.insightface` aren't used
  any more; delete them if nothing else needs them. While a dataset's faces are checked, its page
  shows how far along it is (and the face model's download, the first time), beside the Check
  faces switch and in Health.

### Runs
- **Likeness as a share of your photos:** a run's likeness also shows as a percentage of how alike
  the dataset's own photos are to each other (its median in the face report), on the chart, the
  stats, the checkpoints and the recommended checkpoint. A LoRA that caught the face perfectly
  would score about what the photos do, so "93% of your photos" says more than a bare 0.67. Runs
  scored before this show the plain number.
- **Keep only the checkpoints you want:** New run → Training → Keep checkpoints: All (as before),
  the latest few, the best few by likeness, or the lowest validation loss; or each new best
  likeness or validation loss, every save that beat all before it. The rest are deleted as
  the run goes, so a long run doesn't fill the disk. The final LoRA, the newest save (what a resume
  continues from) and saves made by hand or at a stop always stay. Best by likeness renders samples
  at every save to score them, and lowest validation loss runs validation at every save (it's
  measured on the training pictures, so it mostly keeps the later saves). The run's Checkpoints
  card says what it keeps and lists what it removed, with why.
- **Try other samplers as the run trains:** Manual's Sampling chip opens a sampler, scheduler,
  steps and CFG for the next tries (the run's own until changed), and each try's card says what it
  used. Custom sigmas, high to low, in place of a scheduler: for tries and for the run's samples
  (Samples → Scheduler → Custom sigmas).
- The Manual tab no longer has a gap beside it; its rendering dot sits on its corner.
- **Fixed: a resumed run's loss chart** drew its second half over its first on the time axis (each
  session's clock starts at 0), with lines back to the start; and a resume from an earlier
  checkpoint drew the abandoned stretch and its redo on top of each other. The chart now shows one
  history: time carries on across resumes, and redone steps replace the abandoned ones.
- **Lighter run page:** the noise chart starts folded (it remembers being opened), and its dots are
  drawn on a canvas instead of a page element each, so moving the mouse over it or a step landing
  doesn't rebuild hundreds of them.
- **Fixed: long runs' pages lagged, then stopped drawing:** a 150k-step run's page fetched all
  150k metric rows (62 MB), redid its charts over every one of them each step, and past ~130k steps
  the loss chart's scale hit a JavaScript limit, so the chart and everything below it never drew.
  A long run's history now comes thinned to ~4,000 rows (each carries its stretch's mean loss: the
  same curve; the latest steps stay exact; `?full=1` gets every row), a page left open keeps it
  that way, and the charts no longer hit the limit. On that run: 1.6 MB, drawn in under 2 s, 17 MB
  of memory instead of 347.
- **Slider LoRAs:** New run → Model → Add → Slider: a LoRA whose strength moves pictures along one
  attribute (+1 one way, −1 the other, 0 the base model).
  - **From image pairs:** the dataset's pictures and a picture set beside them (`_controls/<set>/`,
    the same file names) that differ only in what the slider moves (warm and cool light, smiling
    and not). Each step trains both from the same noise, at +1 and −1, with the model's own loss:
    right for flow models (Krea 2, Flux, MOSS) by construction. Pictures without a partner are left
    out. A step costs about two normal ones.
  - **From prompts:** two prompts, as AI Toolkit's concept sliders, with optional anchors for what
    should stay the same; the dataset's pictures are only where the noise starts. About three.
  - Samples render each prompt at −1, −0.5, 0, +0.5 and +1 side by side, and the saved LoRA records
    what each end does (zentrainer.slider).
- **Share run settings:** Export (a `.zentrainer.json` file) or Copy (the same as text, for a chat)
  on New run, and Export settings on a run's page; Import on New run loads one, or paste its text
  anywhere on the page. The file leaves out what's only yours (the dataset, where LoRAs are saved, a
  LoRA to continue from, edit samples' pictures) and says which model files aren't installed here.
  A file from a newer ZenTrainer loads what this one knows, and says what it left out.

### Training
- **Fixed: continuing from a LoRA used this run's alpha instead of the file's,** so a LoRA whose
  alpha differs from the new run's rank (rank 16 at alpha 16 continued in a rank-4 run, say)
  trained at a fraction of its strength, and a DoRA lost its scale. It now continues exactly as
  ComfyUI applies the file.
- **Fixed: a run with a base LoRA or a turbo LoRA could show another run's samples at step 0:**
  the "before training" samples are reused between runs with the same settings, and those two
  weren't counted, though samples render with them on.
- **Fixed: video datasets with sound** stopped as their encoding finished, with "File exists" and
  the path of one of the clips: the soundtrack pass reused the name of the folder the encoded
  dataset is saved in. Datasets already encoded before were unaffected.
- **Fixed: MOSS Image on a newer ComfyUI** stopped with "attention() got an unexpected keyword
  argument 'mask'": ComfyUI's MOSS attention takes more arguments since its MOSS-Image edit
  support, and ZenTrainer's fused q/k wrapper didn't pass them on.
- **Fixed: FLUX.3 couldn't train:** every run stopped on its first step ("functions with out=...
  arguments don't support automatic differentiation"). ComfyUI's FLUX.3 builds its token sequence
  and residuals in place, which is fine for sampling but not for training; ZenTrainer now does
  those steps out of place while training, with the same results.
- **Automagic v4 (experimental):** Ostris's newest Automagic, in the optimizer list. Where v3's
  learning rate keeps falling once the easy learning is done (what the LR floor works around), v4's
  settles at a steady level, and its most sensitive weights take smaller steps so the rest can
  learn faster. Batched like v3, so it's as quick, and it saves and resumes the same way. It's
  marked experimental upstream; try it against v3 on the same dataset.
- **Fused q/k norm + rope for MiniMax H3:** Memory → Fused q/k norm + rope now covers H3 too (it
  rotates only part of each head, which the op now handles). 9% faster a step on H3 at 512 px
  with sound, the same results.
- **Compile with offloading on:** compiling the blocks was refused whenever offloading was on,
  after a crash that turned out to be the offloaded weights themselves (fixed in 0.3.0), not
  compile. It's allowed again, Auto included: MiniMax H3 at 512 px with sound, 32% faster a step,
  whether it all fits on the card or about 1 GB of it streams. With most of a model streaming it's
  about even (the copies set the pace). Tested with weights streaming and moving on H3 and Krea 2, also with a 32 GB card
  held to 16 GB.
- **Fused q/k norm + rope on by default (Auto):** used whenever the blocks run uncompiled, off
  while they're compiled (compile fuses those ops itself). 6–9% faster a step on Krea 2, MOSS and
  MiniMax H3 without compile, with nothing to set. On or Off in Memory still force it.
- **Fixed: the automatic VRAM fit undid itself on video:** after putting more of the model on the
  card it checks the steps didn't get slower, but it compared raw step times, and video clips of
  different lengths take very different times (1.4 to 3 s a step on H3). It often "measured" a
  slowdown and moved the weights back off. It now compares steps of the same shape.

## 0.3.0 (2026-10-25)

### Training
- **Fixed: "illegal memory access" with weights offloaded:** a streamed layer's weights could be
  freed while its backward was still queued on the GPU, and the run crashed a step or two later
  (often right after a resume or once the weights moved between CPU and GPU). Most likely with
  Automagic and EMA, whose long optimizer step lets the CPU run far ahead of the GPU. The backward
  now keeps its own copy of each streamed weight (a copy on the GPU, not over PCIe).
- **Fixed: out of VRAM after samples with weights offloaded:** coming back from a sample pass, the
  model went back onto the card between the sampler's leftover cache fragments, and the next step
  ran out of VRAM; each time used one of the run's three retries, so a long run with samples
  stopped (Krea 2 int8 on 16 GB: at step 500). The cache is cleared first now, and the retries
  count only in a row.
- **Pausing lets ComfyUI work:** a paused run that sees a workflow queued in ComfyUI saves its
  state and lets go of ComfyUI (its queue and the GPU), and stays paused; Resume continues from
  there (a model reload). Nothing else in ZenTrainer's queue starts meanwhile. Needs resume states
  (Output → Resume states, on by default).
- **Faster by default on int8 models:** the int8 backward (W8A8) is now on, so the frozen layers'
  gradients run on int8 tensor cores too: about 16% faster on Krea 2 int8 at 1024. It rounds the
  gradient to int8 (stochastically, so unbiased); turn it off in Memory to train with a bf16
  backward.
- **Compile blocks is Auto by default:** on for runs of 500 steps or more when Triton is installed
  (Krea 2 at 1024: about 14% faster once compiled). Never with weights offloaded, even when turned
  on (or set by a recipe): weights moving between CPU and GPU crashed compiled runs with an illegal
  memory access on a 16 GB card.
- **Experimental: fused q/k norm + rope for Krea 2** (memory.fused_qk_rope, off by default): the
  attention's norms and rope run as one op with its own backward instead of six separate ones:
  6–7% a step on Krea 2 and MOSS at 1024. Same maths, slightly less rounding. Skipped when the
  blocks compile, which fuse them already (and faster with it off).
- **Steps wait for the GPU once instead of six times:** the loss, gradient norm and noise chart are
  read together at the step's end, and the run page's phase timings come from every 10th step.
  The GPU no longer idles between phases.
- **"int8 forward" off now does what it says:** int8 models dequantize to bf16 for every matmul.
  ComfyUI had started multiplying in int8 whenever a layer trains, so turning it off changed
  nothing.

### Datasets and captions
- **A datasets folder of their own** (Settings → Storage): input/datasets/, so datasets stop
  sharing ComfyUI's input folder with everything else uploaded there, or any folder. Nothing is
  moved: Settings lists the datasets still in the old place, and runs still find them there.
- **Caption length, cut-offs and re-rolls:** captions are no longer cut at 300 tokens: most styles
  have no limit now (up to 4096; the model ends a caption itself), a short caption keeps 300, and
  a Length choice when captioning sets one (600, 1200, or no limit); captions that ran out of room are flagged, listed on the job ("N cut off",
  to review them or re-caption them with no limit), and marked in the try-on; and Wording: Vary
  samples each run with a new seed, so captioning an image again words it differently. Advanced
  adds a fixed seed (repeatable re-rolls), temperature, top-p, top-k, a repetition penalty (for a
  model that loops until it's cut off) and a custom length; a job's log says what it used.
- **Dataset analysis** (Health → Full analysis): the pictures worth a look, grouped by issue; how
  training buckets and crops them at a resolution you pick, with every crop shown; which phrases
  most captions repeat (things the LoRA will lean towards); the spread of brightness, contrast,
  sharpness, size and shape; and near-duplicate pairs side by side.
- **Health finds copies properly:** near-duplicates are re-saved, resized, lightly edited or
  mirrored copies, and different pictures that only share a composition no longer count. Bars
  are real letterboxing, not a plain sky. New checks for transparency and heavy JPEG compression,
  and sizes follow a photo's rotation as training does. Health scores move once as a result.
- **A choice of face model** (Settings → Faces): InsightFace's standard one, or the larger
  antelopev2 (ArcFace R100), which tells similar faces apart better and is about 1.5 times slower.
  A run keeps the model it started with. antelopev2's download, which unpacks a folder too deep,
  now loads as it is.

### Install and run
- **A git clone into custom_nodes opens without building anything:** the first visit to
  /zentrainer/ fetches the prebuilt web app for that version (with the clone's own git), and again
  after a `git pull` that changed it. It said to run `pnpm build` before. A build of your own is
  left alone.
- **`zt` no longer says a port is in use just after a server stops.** Its check counted the
  stopped server's lingering connections (about a minute) as the port being taken, so
  `zt train --port` right after a restart refused a free port.

## 0.2.0 (2026-10-24)

### Install and run
- **Start scripts.** From a clone, `start-linux.sh` or `start-windows.bat` is all you run: they
  install uv if it's missing (asking first), which brings the right Python, then open `zt`.
  Only git is needed beforehand.
- **`zt`, the command line, rebuilt.** On its own it shows a menu: a first-run setup that asks
  where your models are and which PyTorch build to use, an update check, start, and more.
  Arrow keys throughout.
- **`zt monitor`:** a live dashboard in the terminal with the queue, runs, progress, loss and
  memory, and keys to pause, stop or save a checkpoint. It only needs ZenTrainer's API, so it
  can watch a machine you're connected to remotely.
- **Safe updates.** `zt` checks for a new version when it starts and updates only a clean
  checkout. Your own changes or commits are never touched; it says why it skipped instead.
- **Ports.** If ZenTrainer is already running, `zt start` says where. If another program has
  the port, it uses the next free one.
- **No Node.js needed** to run from a clone: the web app comes prebuilt.
- **A smaller node pack:** the Registry download is the engine and the built app only (3 MB).

### Training
- **Resolution stages:** train at a lower resolution first and blend into the full one.
- **A Focus slider** for where training spends its steps, plus timestep types, bias and
  ComfyUI's schedulers, with charts of the noise levels they pick.
- **Base LoRA:** train a new LoRA on top of an existing one. It stays on in samples and is
  recorded in the file.
- **Skip blocks:** leave chosen blocks as the base model's.
- **Automagic v3** batches its step across tensors (faster) and has an optional
  learning-rate floor.
- **Offloading** measures how much VRAM the first steps need and fits as many weights on the
  card as are left. Each move is a trial, undone if it makes steps slower.
- **Train a finished run further,** and download its LoRAs from the run page.
- LoRA files are saved in bf16 by default: half the size of fp32.

### Models
- **MOSS Image edits:** train edits from reference pictures, with nodes to sample them.
  Edit pairs can also be channel inputs.
- **Fibo 2.0:** a profile with Bria's own LoRA recipe, and captions in its structured JSON.
- **The model picker:** makers' logos, a catalogue of corrections, and Hugging Face re-listed
  for new models.

### Datasets and captions
- **Storage:** a ZenTrainer folder of its own, so datasets and results don't have to fill
  ComfyUI's input and output folders.
- **Big datasets:** a cache on disk for datasets bigger than RAM, and large folders open fast.
  Caption encodings are cached once and shared between runs.
- **The mask editor:** an image list, a box tool, and one mask applied to many images (with
  undo).
- **Control sets:** named layers of input pictures beside a dataset, for edit training.
- Tidier dataset cards, list and side panel; captions and health in one badge.

### Runs and samples
- **Compare runs** side by side, from the Jobs page.
- **The loss chart** shows speed and time, and its lines can be hidden. Likeness shows its
  change since the first scored samples.
- **Edit runs** show their samples beside the input pictures.
- **The sidebar** shows the queue on the running run's card, and what's loaded in VRAM.
- **Swap** to a queued run: the current one saves and goes back in the queue.

### Simple mode
- **Recipe pages** that guide you through a run, check your photos, and show the results.

### Tools (experimental)
- **Compact:** shrink a LoRA or LoKr to a size you pick. Rank goes to the layers that change
  most, optionally weighed against the base model.
- **Inspect a model file:** every layer, shape, precision and metadata entry in a .safetensors
  file. Save a copy with other metadata, another precision, or layers left out.

### Fixes
- Opening the app over plain http on a LAN address works again.
- Skip-layer guidance buttons are clickable again.
- Batches whose captions can't stack run as separate calls instead of failing.
- SAM3 lets go of its VRAM when training or captioning starts.

### Upgrading
- **Tools, Apps and Gallery are experimental pages, hidden by default.** Turn them on under
  Settings → Appearance → Experimental pages.
- **From a clone:** `scripts/zt` still works, and is now `zt`. Run `./start-linux.sh` or
  `start-windows.bat` once to set up its environment.
- **The pinned PyTorch is 2.13.0** (in `comfy.lock`). `zt setup` and `zt update` install that
  version; `zt doctor` says when yours differs.

## 0.1.0 (2026-10-18)

The first release on the ComfyUI Registry.
