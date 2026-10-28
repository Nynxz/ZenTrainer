![ZenTrainer](docs/images/banner.webp)

Train LoRAs inside ComfyUI, with an app built for it. Nothing new to download: it uses the models
and the GPU your ComfyUI already has, and it trains on ComfyUI's own model code, so whatever ComfyUI
can load, you can train.

## Training you can watch

Every run gets one page: the newest sample for each prompt beside the loss, and the numbers that
matter in a rail on the right. While a sample renders you see it denoise in its tile. Notches on
the bar mark each checkpoint and sample, so you know what's coming and when.

- **Pause, sample now, save now, change the learning rate**, all mid-run
- **Resume** a stopped or crashed run from any checkpoint, optimizer state and all
- **Validation loss** on a fixed set, so the curve only moves when the LoRA does
- **Likeness** for character LoRAs: how much each checkpoint's samples look like your dataset's face
- **Try a prompt** at the current step, or on any finished run's checkpoint
- **Progression**: every prompt across every step; in the viewer ←/→ moves through training, ↑/↓ switches prompt
- **Training time apart from total time**, so loading models and sampling don't make a run look slow
- **A queue on the Jobs page**: line runs up, reorder them, and they train one after another

![A run](docs/images/run.webp)

## Datasets

Any folder in ComfyUI's `input/` is a dataset. Browse it as a grid or a table and edit captions in
place. Click an image to open it big, with its caption beside it and its training mask overlaid.

- **Caption sets**: keep long and short captions side by side and train on either, or a mix
- **Auto-caption** with a vision model through ComfyUI's own text generation (Qwen-VL, Gemma)
- **Bulk edits**: prefix, suffix or find and replace across a set
- **Masks**: drop `name-masklabel.png` beside an image, or paint one, and it learns only where the mask is white
- **Your files are safe**: uploads never replace a file, edits are written whole (a crash can't
  leave half a caption), and nothing outside ZenTrainer's own folders is ever deleted

![A dataset](docs/images/dataset.webp)

![The dataset viewer](docs/images/dataset-viewer.webp)

## Set up a run

Pick a model by name (Krea 2, Z-Image, Qwen-Image, SDXL…) and every part is filled in: the
base build to train on, its text encoder and VAE, each marked installed or one click from
downloading off its Comfy-Org Hugging Face repo. The family's turbo build is offered for fast
previews, with the sampler settings ComfyUI's own template uses. The **Models** page lists
every family, what you have of it, and the rest.

Start from a recipe (the model field opens the models, each with its recipes; **Recipes** is the
dialog's other tab): ours for each model (a character, a style), or one
you saved from a run that went well with **Save as recipe**. Each shows what it would change before
it's applied, and keeps your run's name, dataset, trigger word and captions. The form warns you
about what usually goes wrong: no trigger word, sample prompts that never use it, a learning rate
the optimizer won't like, too few epochs.

- **Trigger word** added to every caption, and `[trigger]` in captions and prompts
- **Optimizers**: AdamW, Prodigy, Lion, Adafactor, CAME, schedule-free, 8-bit and more, each with a suggested learning rate
- **LoRA or LoKr**; masked training; caption dropout; mixed caption sets
- **Sample with another model**: train on the raw base, preview on its turbo version

![New run](docs/images/new-run.webp)

## Simple mode

Pick what you want to teach (a character, a style), drop in your photos, name it, and train. The
recipe picks the model files (downloading what's missing), captions the photos, and sets every
knob. While it trains you see each preview land, when the next one is due, and you can try a
prompt of your own at the current step. Switch to Advanced in Settings for every setting.

![Simple mode](docs/images/simple.webp)

## Apps (experimental)

ComfyUI workflows with an interface built for the job: **Test LoRA** renders checkpoints ×
strengths × prompts on a fixed seed, **Generate** is a clean txt2img on any model with your LoRAs.
They're still being finished, so they sit behind Settings → Experimental pages (a run's Try
buttons open them either way). → [Writing apps](docs/APPS.md)

## Your ComfyUI, kept safe

ZenTrainer runs on its own pinned copy of ComfyUI, so updating your main install or its custom
nodes can't break a training run. Your models stay where they are, read through
`extra_model_paths.yaml`. Themes come from [ZenKit](https://github.com/Nynxz/ZenKit), so it matches
the rest of your setup.

## Install

Two ways in, same engine:

**Into your ComfyUI, as a node pack** (the simple way). Install ZenTrainer from ComfyUI Manager or
the Registry, or clone this repository into `ComfyUI/custom_nodes/`, then restart ComfyUI and click the lotus
in its top bar (or open `/zentrainer/` on its port, e.g. http://127.0.0.1:8188/zentrainer/). It uses that ComfyUI's
models, `input/` and `output/`; it needs ComfyUI 0.41 or newer and says so if yours is older.
Nothing extra to install: everything it needs ships with ComfyUI (optional optimizers:
`pip install -r requirements-optional.txt`). A Registry/Manager install includes the built app; a
git clone fetches the prebuilt one the first time you open it (or `scripts/zt web`, which can also
build it with pnpm).

**Its own pinned ComfyUI, from a clone** (for keeping training separate, below): a
dedicated ComfyUI at the tested commit with its own venv, reading your existing models in place.
Updating your main ComfyUI or its custom nodes can't break a run. Only git is needed:

| | |
|---|---|
| **Linux** | `git clone https://github.com/Nynxz/ZenTrainer && cd ZenTrainer && ./start-linux.sh` |
| **Windows** | Install [Git for Windows](https://git-scm.com/download/win), clone the repository, then double-click `start-windows.bat` |

The first run installs [uv](https://docs.astral.sh/uv/) if it's missing (it brings its own
Python), asks where your models are and which PyTorch build to use, then sets up and starts.
After that it opens a menu (start, doctor, extras, settings) and checks for updates on the way
in: an update only ever fast-forwards a clean checkout, and it's skipped when you have local
changes or commits. Arguments pass through to `zt`, e.g. `./start-linux.sh start --open`.
If ZenTrainer is already running it says where; if another program has its port (8189), it
uses the next free one. macOS isn't supported yet.

**On a rented GPU** (RunPod, Vast.ai, …) or any machine with Docker: a container image with the
app behind a password and your work kept on the volume. On RunPod it's one click:
[deploy the ZenTrainer template](https://console.runpod.io/hub/template/2amo4895gj?ref=oe5l65t0)
(a referral link: it supports ZenTrainer at no cost to you). Elsewhere, and how it works →
[docker/README.md](docker/README.md)

## Run it (zt)

`zt` is ZenTrainer's command line (`cli/`, run through uv): the start scripts open its menu, and
every command works on its own too. `scripts/zt` runs it from a clone; to type just `zt`
anywhere, install it once with `uv tool install --editable ./cli`.

```sh
zt                      # the menu: first-run setup, update check, start, monitor
zt setup --models-from /path/to/your/ComfyUI   # clone pinned ComfyUI, install, link the engine
zt start --open         # http://127.0.0.1:8189/zentrainer/
zt monitor              # a live dashboard: queue, runs, loss, memory (p pause, s stop, c save)
zt dev                  # ComfyUI + web app with hot reload (:5173/zentrainer/)
zt restart              # restart ComfyUI (picks up engine changes)
zt doctor               # what's installed, where models come from
zt update               # pull ZenTrainer, re-sync ComfyUI and the web app
```

Headless, for scripts or a machine without a browser: `zt new > spec.json` writes a spec to start
from, `zt check spec.json` lists every problem in one, `zt train spec.json --follow` runs it; then
`zt runs`, `zt follow RUN`, `zt pause|stop|save RUN`, `zt resume RUN`, `zt sample RUN "prompt"`,
`zt queue`, `zt caption DATASET --model FILE`. `zt --help` lists them all. The guides:
[docs/cli.md](docs/cli.md) (the commands, with a worked example), [docs/api.md](docs/api.md) (the
HTTP API the app and `zt` both use) and [docs/spec.md](docs/spec.md) (every setting a run takes).

`--models-from` takes an existing ComfyUI install (its `models/` and its
`extra_model_paths.yaml`) or a plain models folder. To point at a yaml directly, use
`--extra-model-paths /path/to/extra_model_paths.yaml`. Both flags are repeatable and
can be combined. Or skip the flags and drop (or symlink) an `extra_model_paths.yaml`
into `comfyui/`: ComfyUI loads that file on its own.

Datasets go in `comfyui/input/<name>/`; trained LoRAs land in
`comfyui/output/zentrainer/loras/`, and run history in `comfyui/user/zentrainer/`. That's your
work: to reinstall, use `zt update` rather than deleting `comfyui/`, or move those three
folders out first. Theme packs are bundled; `--zenkit` is optional.

Prefer one ComfyUI? `zt attach /path/to/ComfyUI` links the engine into it
instead. It works as long as that ComfyUI is close to the pin in `comfy.lock`.

`comfy.lock` pins the ComfyUI commit and torch build that the harness is tested
against. The harness uses ComfyUI internals, so move the pin with
`zt update --ref <commit|tag>` and test before committing it.

**Compiled blocks** (Memory → Compile blocks): the first run at a resolution it hasn't compiled
before generates its kernels. That's a few minutes on Linux with an NVIDIA card, and 10-25 on
ROCm or Windows, where the first steps can take 40-60 s each and the run looks stalled. The
kernels stay in Triton's cache (`TRITON_CACHE_DIR`, else `~/.triton`), so later runs at that
resolution are fast from the start: don't delete it to clean up. To see why a run recompiles,
start ComfyUI with `TORCH_LOGS=recompiles` and `PYTHONUNBUFFERED=1` (on Windows, `set` both in the
launcher): without the second, ComfyUI's log can lag minutes behind the run.

## How it works

The training loop is ZenTrainer's own: it loads models, patches in the LoRA and steps the
optimizer with ComfyUI as a library, not through workflows. While a run trains, a one-node job holds
ComfyUI's queue, so a generation can't land on the GPU mid-step. Design and rationale are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
engine/   ComfyUI node pack: training harness (ComfyUI as a library), run API, session node
web/      Vue app (zenkit-ui, ZenKit themes), served by ComfyUI at /zentrainer/
tests/    harness tests that need no ComfyUI
scripts/  zt (setup, start, dev, …); devrun.py runs a spec through the harness without the server
docker/   the container image for cloud GPUs (RunPod, …)
docs/     architecture, apps; banner/ holds the README banner's source
```

## Develop

`zt dev` is the loop: web edits hot-reload at http://localhost:5173/zentrainer/,
engine (Python) edits need `zt restart` (or Restart on the System page).

```sh
# tests (no GPU)
uv run --with pytest --with torch --with pillow --with numpy --no-project python -m pytest

# a spec straight through the harness, using ComfyUI as a library (see examples/)
comfyui/.venv/bin/python scripts/devrun.py examples/sdxl-quick.json --comfy comfyui

# README screenshots, from an install holding only demo content (maroon-volt, dark)
uv run --with playwright --with pillow --no-project python docs/banner/shots.py --url http://127.0.0.1:8188 --run <id> --dataset <folder>
# banner and Registry icon (docs/banner/banner.html → docs/images/banner.webp, icon.png)
uv run --with playwright --with pillow --no-project python docs/banner/render.py
```
