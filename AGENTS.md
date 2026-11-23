# Working on ZenTrainer

How the engine is built is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This file is how to
change it: what good changes look like here, what must not break, and the workflow.

## Engineering style

- Keep changes small and direct. Touch the narrowest code path that explains the bug or delivers
  the feature, and the fewest files. A change spread over many files needs a reason beyond
  convenience.
- Prefer practical fixes over architecture work. Add an abstraction only when it removes real
  repeated logic or matches a pattern already here.
- Delete what newer code makes useless: dead branches, unused options, fallbacks nobody reaches,
  debug prints. Don't keep a second way of doing the same thing "for compatibility" unless
  something real still depends on it (see Compatibility).
- Code must read as hand-written for this repository. Unnecessary helper layers, vague names,
  comments that restate the code, defensive branches without a real failure, and rewrites that
  ignore the local style are rejected.
- Fail clearly. A wrong setting, an unsupported model or a bad file stops with a message that
  says what to do, never silently trains or samples something worse.
- When a feature breaks for users, disabling it beats a complicated partial fix.

## Boundaries

- **ComfyUI is the library.** Model loading, attention, quantization, offloading, sampling and
  weight adapters come from ComfyUI (`comfy.*`). The engine owns the training loop and what
  ComfyUI doesn't do; it doesn't copy or fork ComfyUI code. If ComfyUI has it, use it.
- **The engine** (`engine/zentrainer/`): `spec.py` is the contract (pure Python, no ComfyUI);
  `harness/` trains; `server/` is the HTTP API; `runs/` stores runs; `datasets/` and `models/`
  serve the app. Training code doesn't know about HTTP, and routes don't do training math.
- **The web app** (`web/`) talks to the engine only through `/zentrainer/api` (and ComfyUI's own
  API). Anything it needs comes from an endpoint, not from guessing file layouts.
- **`zt`** (`cli/`) uses the same API for everything a running ZenTrainer does; it only touches
  files for setup, updates and the web build.
- **Per-model knowledge** goes in `engine/profiles/*.json` and `models/families.py` as data, not in
  `if family == ...` branches through the code. A profile holds what its source says; no guessed
  learning rates.
- Pass the narrowest data across a boundary; keep each layer's bookkeeping in that layer.

## Compatibility: what must keep working

- **LoRA files load in stock ComfyUI** with its LoraLoader: standard key names, alpha, dtype.
  Never invent a format only ZenTrainer reads.
- **Old runs keep working.** `TrainSpec.from_dict` rejects unknown fields, so never rename or
  remove a spec field; a new field gets a default that reproduces the old behaviour. Training
  states, checkpoints and run folders from earlier versions must still resume, list and download.
- **The API is used by the web app, `zt` and people's scripts:** change a route's shape only with
  every caller updated in the same change; add fields rather than repurpose them.
- **The engine imports without ComfyUI** (tests and tools import `zentrainer` directly), which is
  why `comfy.*`, `folder_paths`, `server` and heavy packages are imported inside the functions
  that use them. Plain standard-library imports go at the top.

## Training correctness

- Treat dtype, device placement, VRAM use and offloading as correctness, not tuning. Check what a
  change does with quantized (int8, fp8) base models, offloading on, and a small GPU.
- A speed change must not change the result. Compare loss (and a sample) before and after on the
  same seed; say what you compared.
- No silent upcasts to fp32 copies of a model, no extra copies of weights, no tensors used as
  Python bookkeeping (counts, indices, shapes stay Python ints).
- Optional packages (optimizers) are extras: `zt extras ...`, imported with a plain
  `import` when the user picks them (the Registry's scanner flags `importlib.import_module`), with
  a message naming the extra when missing.

## Autograd and the frozen model

ComfyUI is inference-only: it runs nodes under `torch.inference_mode` and treats every model as
frozen. Training reverses the first and relies on the second.

- Training runs with ComfyUI's inference mode turned off (`torch.inference_mode(False)`, as
  `harness/trainer.py` does). A new path that trains or builds training inputs does the same.
- Anything that feeds training (cached latents, text encodings, masks, reference latents) is
  made under `torch.no_grad()`, never `torch.inference_mode()`: inference tensors can't be saved
  for backward, and the error shows up steps later, far from the cause. `inference_mode` is only
  for pure inference whose results never reach training (captioning, say).
- The base model is frozen once, by the harness (`requires_grad_(False)` on the loaded model).
  Don't add freeze or trainability flags to ComfyUI's model classes, or patch its model code to
  train: only the adapter's parameters train (and its alpha stays fixed: it's a saved scale).
- Don't switch the base model to train mode. It stays as ComfyUI loads it; dropout and other
  training-only behaviour belong to the adapter.

## No internet without the user

- The engine contacts the internet only for something the user started: downloading a model they
  picked, listing a repository they opened. No telemetry, analytics, tracking or background
  requests, opt-in or not.
- `zt`'s update check is for clones and can be turned off (Settings: updates never).

## The user's data comes first

- Never delete or overwrite datasets, runs, checkpoints, training states, models or outputs
  without the user saying so. Look at a target before removing it; map run ids to names first.
- Test anything that installs, updates or deletes in a throwaway clone or folder, never in the
  user's checkout or `comfyui/`. `tests/cli` shows the pattern (git repositories under tmp_path).
- Don't restart the user's ComfyUI while a run trains: check `zt queue` first.
- A question with no one to answer it stops; it never takes a default (a default can be
  "install").
- Don't search the user's whole disk for something: ask where it is.

## Python

- Match the file you edit: its names, comment density, line length and idiom.
- Comments say why, in plain sentences, and only where the code doesn't. A module starts with a
  docstring saying what it's for.
- `try`/`except` only where there's a real failure with a useful fallback; catch the specific
  exception.
- The engine adds no dependencies: `dependencies = []` in pyproject.toml, everything else ships
  with ComfyUI or is an extra. `cli/` has its own small, locked set.

## Web

- Vue 3 with `@nynxz/zenkit-ui` components and the theme's `--zen-*` variables; no hard-coded
  colours that break another theme. Check every page at phone width.
- Plain words in the interface: say what something does for the user. Settings carry a one-line
  description.
- `pnpm build` type-checks; it must pass.

## Running and checking

```sh
zt dev                     # ComfyUI + the web app with hot reload (:5173/zentrainer/)
zt restart                 # after engine (Python) changes (check `zt queue` first)
cd web && pnpm build       # type-check and build web/dist
python -m pytest -q tests  # engine tests; tests/comfy also trains each patched model family
                           # (tiny, on CPU) through comfyui/ when it's there
uv run --frozen --project cli --with pytest python -m pytest -q tests/cli   # zt's tests
```

`tests/engine/test_streams.py::test_sharpest_frame_picks_the_detailed_one` fails on main
already; anything else failing is new. New behaviour gets a test when it can break quietly
(formats, compatibility, anything touching the user's files). A patch of ComfyUI model code
(a module in harness/compat/) gets its family in tests/comfy/smoke.py, which also checks the patched model
computes what ComfyUI's code does.

## Commits, PRs and releases

- One topic per commit. Subject `Area: what changed` in plain words; a body when the why isn't
  obvious. Commit when asked; the user pushes.
- Research, experiments and one-off scripts don't belong in this repository.
- **Release notes come from PRs, not `CHANGELOG.md`.** Don't add a `CHANGELOG.md` entry in a PR.
  Instead, give the PR a plain-English title (`Area: what changed`), a short user-facing summary in
  the PR body (what changed and why — never naming a user, their files, folders or local setup),
  and a label (`enhancement`, `performance`, `bug`, `documentation`, or `internal` to omit it from
  release notes).
- **A release:** bump the version in `pyproject.toml` and `engine/zentrainer/__init__.py` (kept
  equal by `tests/engine/test_version.py`), push `main` and a `vX.Y.Z` tag (which publishes to the
  ComfyUI Registry), and publish the GitHub Release for `vX.Y.Z` from the merged PRs since the
  previous tag (`.github/release.yml` groups them by label, with highlights edited at the top if
  needed). A published version can't be republished: a fix is the next patch version.
- The Registry pack is what git tracks minus `.comfyignore` (plus `web/dist` and `engine/js` from
  `[tool.comfy] includes`). Anything only a clone needs goes in `.comfyignore`.
- CI: `ci` runs the tests and the web build on every push; `comfyui` runs tests/comfy against the
  pinned ComfyUI on every push and against ComfyUI's latest nightly; `web-dist` rebuilds the prebuilt app
  (the orphan `web-dist` branch, for clones without Node) when `web/` changes; `publish-registry`
  runs on `v*` tags.
