# Changelog

What changed in each ZenTrainer release, newest first. Versions are the ones the ComfyUI Registry
and Manager show.

## Unreleased

### Security
ZenTrainer's API has no login (as ComfyUI's hasn't), so what someone who can reach it could do
was looked over, route by route. ComfyUI only listens on this machine unless it's started with
`--listen` (or reached through a tunnel); these matter most then.
- **Fixed: changing the datasets or ZenTrainer folder setting could open your files up.** Set to a
  home folder, the dataset routes would then read, write and delete in it. Those settings now
  change only from this machine, a whole drive or a home folder is refused, and the file viewer
  serves only pictures, videos, sounds and masks from the datasets, never a hidden folder.
- **Fixed: the segmentation result route could serve any .png on disk** from a path in its job id.
- **Paths outside ZenTrainer's folders** in a run (a dataset folder, held-out validation, edit
  sample inputs, test samples) are taken only from this machine: training reads them from
  anywhere. A shared settings file now leaves out the validation dataset too.
- **From this machine only:** a restart that stops a training run, and rebuilding the web app.
  "This machine" means a request from it, to its own name, not through a proxy or tunnel (they
  connect from 127.0.0.1 themselves), which also stops a site whose name points at 127.0.0.1.
- **Training states and caches load in PyTorch's safe mode:** tensors and plain data only, never
  code. They're ZenTrainer's own files, but nothing reads them in a way that could run code now.
- **Uploads are read up to 2 GB a file** (a whole file used to be held in memory, unlimited);
  a picture too big or broken to open is refused instead of failing.
- **The Hugging Face token** goes only to huggingface.co, and not on to where a download is
  redirected.

### Run settings (spec version 2)
- **Specs have a version now** (`spec_version: 2`), and older ones are brought up to date as
  they're read: stored runs, presets, model profiles, exported settings files, and specs sent to
  the API or `zt train`. Nothing you have stops working: all of a run's settings carry over, and a
  setting this version doesn't have (one that was tried and removed) is left out with a note
  instead of stopping the run. That fixes runs that couldn't resume, be edited or be duplicated
  ("unknown OptimSpec field(s): anchor_decay"). A spec written by hand still errors on a field
  this version has never had, so typos are caught.
- **Renamed, for anyone scripting runs** (the old names still work in older specs, which are
  converted; a spec that says `spec_version: 2` uses the new ones):
  - `save_every` → `output.save_every`, beside the other saving settings
  - `optim.scheduler` → `optim.lr_scheduler`, and `objective.scheduler` / `scheduler_steps` →
    `objective.timestep_scheduler` / `timestep_scheduler_steps`: two different schedules that
    were both called `scheduler`
  - `memory.w8a8_backward` + `memory.fused_int8_backward` → `memory.int8_backward`: `"w8a8"`
    (the default), `"fused"` or `"bf16"`, the one choice the two switches made between. On the
    form: Memory → int8 backward
  - `memory.quantized_backward` → `memory.fp8_backward` (what it is: for fp8 models)
  - `dataset.keyframes` → `dataset.video_keyframes`, `dataset.clip_stills` → `dataset.video_stills`
  - `validation.timesteps` → `validation.noise_levels` (a count; `objective.timesteps` is the distribution)
  - `memory.checkpoint_depth` / `checkpoint_skip` → `memory.grad_checkpoint_depth` / `grad_checkpoint_skip`
    (gradient checkpointing, not the saved checkpoints)
  - `memory.compile_blocks` and `memory.fused_qk_rope` take `"auto"` (the default) instead of `null`
- A spec from a newer ZenTrainer than this one is refused with a note to update, instead of
  training with settings it doesn't understand.
- **Every setting is checked when a run is submitted:** its type (`"steps": "1000"` or `"rank": true` is
  refused), its choices (the optimizer, attention backend and dtypes included, which took any text
  before and failed only once training started) and its range, each error naming the field. The same
  rules are published: **[docs/spec.md](docs/spec.md)** lists every field with its default, what it
  takes and what it does, and [docs/spec.schema.json](docs/spec.schema.json) (also served at
  `GET /zentrainer/api/spec/schema`) is the JSON Schema. Both are generated from the spec itself, so
  they can't drift from it. A spec file can name the schema (`"$schema": "https://raw.githubusercontent.com/Nynxz/ZenTrainer/main/docs/spec.schema.json"`,
  as the examples do) and editors such as VS Code complete and check it as you type.
- **Compile blocks is off unless you turn it on** (Memory → Compile blocks). It was "auto", on for
  runs of 500 steps or more with Triton installed; with kept attention and the fused q/k rope
  running without it, it saves a few percent on most models, against seconds to minutes of
  compiling on NVIDIA and 10 minutes or more on ROCm. A saved spec's `"auto"` still loads, as off,
  and the log says so. No recipe turns it on: the Fibo 2.0 one did, a setting carried over from the
  run it came from rather than one chosen for Fibo.

### Command line
- **Specs from the command line:** `zt new [--profile ID] > spec.json` writes a complete spec with
  every default to start from, and `zt check spec.json` lists every problem in one at once (with
  what a typo likely meant: `optim.lerning_rate (did you mean optim.learning_rate?)`), offline and
  by the server's rules. `zt train` checks the same way first, so a bad spec never reaches the
  queue. The server's errors name the likely field or choice too.
- **Guides for scripting:** [docs/cli.md](docs/cli.md) (the commands, with a worked example) and
  [docs/api.md](docs/api.md) (the HTTP API the app and `zt` use: runs, the queue, live events, with
  a Python example).
- **zt's menus work in Git Bash and other terminals that aren't a Windows console**, which crashed
  with `NoConsoleScreenBufferError` (or, in Git Bash, could stop as "no terminal"): there it asks in
  plain lines instead (a numbered list to pick from, Enter for the default).

### Run form and look
- **Isotope is the theme a first visit starts in** (the README's look), instead of Zen. A theme
  you picked stays picked.
- **Simple mode, tidied:** recipe cards are grouped by kind (People & characters, Styles, Any
  subject) and lead with the model's maker; recipes have no colour panels until they have pictures.
  A recipe's page shows two rows of your photos (Show all for the rest), and a dataset you picked
  closes to one row with Change. A run's page puts the picture beside what it says: the preview
  being drawn and the earlier ones while it trains, before and after when it's done, with the
  recommended checkpoint, the model and how long it took. The checkpoint table's labels line up
  with their columns.
- **The mask editor fades what's left out into the theme's background;** **High-vis** (beside
  Overlay) brings back the magenta checker, for edges against dark or busy pictures.
- **The recipes dialog** no longer jumps to its search box, and a recipe without a picture has its
  name as a heading instead of a coloured band.
- **`zt dev` can save:** ComfyUI turned down every change made through the dev server ("Forbidden":
  saving, starting a run, finding a mask), its Origin not matching the address the proxy reached.
- **A run's rail reads like a dataset's side panel**, the same spacing and headers, and every card
  in it (Time, Learning, Setup, In VRAM, Checkpoints) folds from its header, remembered in this
  browser. The Datasets page and a dataset's page take the header Jobs and Models have (a dataset's
  with Datasets as its crumb), then the same sticky toolbar and view switch.
- **A grad norm or loss that blew up reads short** (6.3e16, the whole number on hover) instead of
  running out of its cell, and a checkpoint that's both final and best keeps its loss in view.
- **New run's model field opens a model browser,** as its dataset field opens the dataset picker:
  every model family (starred first, a search, all or installed) and the one highlighted, whether
  it's installed or what it downloads, where it comes from, and its recipes. A recipe there is
  applied at once with its model (Undo takes it back). Recipes is the same dialog's other view
  (every recipe, yours too, with what it changes), and a recipe's Details go to it there; Pick
  the files yourself and Re-list are in its footer.
- **The adapter and its rank together:** LoRA or LoKr is a switch beside the rank at the top of
  Training, not a field in Advanced.
- **Masked training says when it disagrees with the dataset:** a dataset with masks and the
  switch off (or the reverse) is pointed out under it.
- **Other LoRAs, in one list:** a LoRA to train more on, to sit under the new one, or to be on only
  while training (a de-distillation adapter) is one row each, its file and how it's used,
  instead of three fields (Continue from, Base LoRA, Training adapter). The settings are the same.
- **Trigger word: Use “`<dataset>`”** beside it fills in the dataset's name (lower case, `_` for
  spaces), and goes once it's used.
- **A new run picks the dataset changed last**, not the first by name (Train on this and a
  duplicate still bring their own).
- **New run's dataset picker is a dialog like Recipes:** the datasets on the left (starred first,
  a search, by recent, name or size) and the one highlighted on the right, eight of its pictures,
  its health and what's in it, before Use; a picture (or Browse all) opens the dataset in the
  lightbox with its captions. Stars follow you, as starred models do. The pickers
  close with ✕ or Esc: no Cancel button.
- **A run's full spec reads like Recipes:** every setting by part, in words, or the JSON, and
  it copies or downloads as JSON (`zt train` takes it, and a New run page takes it pasted).
- **Recipes reads more cleanly:** a shorter header over each recipe, and its changes in words (a
  resolution stage reads "768 px for the first 60%", not its JSON).
- **The page behind a dialog or the lightbox stays put:** the wheel scrolls only the dialog.
- **Smoother moving around:** Back and Forward land where each page was scrolled; the Datasets
  and Models lists show at once on a return and refresh in place; switching between Apps no longer
  reloads the whole page.
- **System's runs table stays short:** the 12 biggest first with Show all, the runs that keep
  nothing on disk out of the way behind a count, a search, a Total column, and each column's
  header sorts by it.
- **Jobs has a search** (a run by name or id, a caption job by what it captioned), in the same
  sticky toolbar Models and the datasets pages have now, with one style of switch for their
  filters, sorts and views.
- **Crop to the card works in the table view too:** off, its thumbnails show the whole picture.
- **Mask editor on the dataset page:** a button beside Add files (and Masks with a selection) opens
  it on the first selected picture, or the first one.
- **Auto-caption's Into set lists every set** under the field, one click each: the field's
  suggestions showed only sets starting with what it already held.
- **Fixed: a training step whose gradients overflowed broke the app's pages.** Its grad norm
  (infinite or NaN) went out as `Infinity` in the live updates and the run's details, which a
  browser can't read, so the Jobs page said it couldn't reach the engine. Such a value now goes out
  as empty (shown as –), and a run that's only just starting no longer reads as the engine down.
- **The adapter list is LoRA and LoKr.** LoHa and OFT aren't ready (LoHa can't train on an int8
  base), so the form no longer offers them; a run that already uses one keeps it, and specs still
  take them.
- **The adapter's rank is labelled,** and resuming a run locks the adapter and its rank with the
  rest of the network (its saved state is that adapter at that rank).
- **The optimizer list is the ones that can run:** those needing a package that isn't installed
  are one line at its end (how many, which, and the pip line that adds them); a run that already
  uses one keeps it.
- **Resume states set to None says what it costs,** in Settings and on New run: a stopped or
  crashed run can't be continued. It's remembered from the last run's choice, so it's worth seeing.
- **Mask editor: Save & next** (Ctrl+→) beside Previous and Next, and the one Save button beside
  it. It saves only what changed, so an untouched picture keeps the mask it has. Its side panel
  goes in the order you work (Tools, Find it, Mask, This size), and its list filters round as the
  theme does.
- **Holding Shift over a dataset thumbnail peeks at a sharp copy,** not the tile's small one; it
  doesn't peek in the lightbox or over the mask editor's canvas, and a Shift-click (selecting a
  range) doesn't pop it up.
- **The Datasets page has the picker's stars** (starred first, whatever the sort) and a sort for
  the cards as well as the table, remembered like the view.
- **Settings is tidier:** a General section (Interface, Experimental pages), then Training, then
  Appearance (Light or dark, Motion, eight theme packs until Show all). A web-only update
  rebuilds from Versions.
- **A paused queue resumes from the sidebar:** a play button beside Jobs while it's paused (its
  queued runs' line reads Paused instead of Next). The update button lines up with Settings, its
  icon centred.
- **Fixed: one NaN step blanked a stretch of a long run's loss chart.** Thinning averaged it in;
  it's left out now.
- **Fixed: an edit sample's input picture didn't show when it was given as a full path** (one on
  another drive, say): training found it, but the run page asked for a dataset named after the
  path's first part ("bad dataset name"). A run now hands the page each
  sample's input itself, found as training finds it. A picture the server can't read says so
  instead of failing.
- **Fixed: a run using a model file outside the catalog lost it when duplicated** (a Qwen Image
  2.1 run on `qwen3vl_8b_fp8_scaled`, say): it showed empty, "not in the catalog", and asked to
  download the catalog's. A file of yours that ComfyUI has is listed, tagged yours, and installed.
- **Comfy Classic's surfaces are tints of its colours,** not the full brand blue (dark) or neon lime
  (light): every hovered row, chip and tile was a block of it.
- **Recipes on New run** (top right): Simple mode's recipes for each model, and your own, without
  switching to Simple mode. Each shows what it would do to the run before it's applied: its model
  files (and what they'd download), its length, adapter, learning rate and resolution, and every
  setting it changes. The run's name, dataset, trigger word and captions stay, and Undo puts the
  rest back. A search narrows the list; on a phone it's the whole screen, list then recipe.
- **Save as recipe** (the ⋯ menu, beside Start, or a finished run's ⋯ menu, for one that went
  well): these settings and the model, as one of yours. It says when one of yours has the same
  name (it replaces it) or exactly these settings already.
- **No built-in presets:** Character, Style, Flow-model tweaks and Quick test are gone (the
  recipes do their job). Presets you saved show as your recipes. Export, Copy and Import moved
  into the ⋯ menu.
- **A recipe trains one length**, the one it was checked at (MOSS Image 1,750 steps; the drafts
  2,000), not one worked out from the photo count, with a checkpoint and samples every 250 steps:
  pick the best, or Train more. A set of over 150 pictures is told it may want more. Simple mode
  too.
- **MOSS Image's recipe checkpoints every block:** it kept the last four's activations for speed,
  VRAM a small card may not have. What it learns is the same.
- **Isotope**, a new theme: ink black, an isotope-lime accent, titles in condensed caps (every
  page's, the ZenTrainer name, dialogs') and one tight corner on everything. Its fonts ship inside
  it, so it works offline.
- **Themes load lighter:** the app fetches only the shown theme's extra CSS (Isotope's is mostly
  its fonts, ~160 KB), not every theme's on every load, and the browser keeps both it and the
  theme list between loads. A reload went from ~206 KB to under 1 KB.
- **The version beside the ZenTrainer name** in the sidebar; hover for the build and whether
  there's an update.
- **New here?** New run mentions Simple mode until you close the note or a run finishes.
- Recipe covers are their own colour and icon (not ComfyUI's template pictures, for now), and
  the N key no longer opens New run.

### AMD
- **AMD GPUs on Windows and Linux:** `zt setup` finds a Radeon or Instinct card from its display
  driver alone and installs PyTorch's ROCm build from AMD's wheel index, with only that GPU's
  kernels; no ROCm or HIP SDK install is needed. The ROCm build the setup used to pick no longer
  existed for the pinned PyTorch, so an AMD machine set up before this one gets the new build on
  `zt update`. `zt doctor` and first-run setup list AMD GPUs.
- **Fitting weights in VRAM on Windows with an AMD GPU** now counts what the desktop and other
  programs use: ROCm only reported ZenTrainer's own memory there, so the fit could plan with VRAM
  that was already taken. The memory chart shows the whole card too.

### Settings
- **The face models' licences:** InsightFace's non-commercial note shows only under its two
  models, with a link to Commercial (AuraFace).
- **Versions, at the top of Settings:** the ZenTrainer version running (release, commits since,
  commit), the web app's build and ComfyUI's, and whether there's an update: how many new
  commits, what they are, and what to run (`zt update`). It also says when a newer version is on
  disk but not loaded yet (restart), when the page is an older build than its source (rebuild),
  and when local changes or a checkout's own commits would stop `zt update`. **Copy versions**
  puts all of it in one block to paste when asking for help. The sidebar marks a ZenTrainer
  update as it already did ComfyUI's.
- The page is reordered: versions, appearance, training, faces, then storage folders.

### Cloud
- **Train on a rented GPU:** a container image (`docker/`) for RunPod and other cloud GPUs, or any
  machine with Docker. The app sits behind a password, ComfyUI is never reachable from outside,
  SSH takes keys only, and your datasets, runs, LoRAs and models stay on the volume. Setup for
  RunPod is in docker/README.md; the image is `ghcr.io/nynxz/zentrainer`.

### Datasets
- **Fixed: a phone photo stored turned a quarter trained cropped to the wrong shape.** Phones often
  save a portrait sideways with a tag to turn it upright; training loaded it upright but picked its
  bucket from the stored size, so a portrait was cropped to a landscape bucket (a 3:4 lost about
  half its height: heads and feet). It's bucketed upright now, as the Analysis page already showed
  it. A dataset with such a photo encodes again, once.
- **Fixed: matched edit pairs were told they're different shapes.** The check compared an input
  to the bucket its target trains at, a few percent off a 2:3 or 4:3 picture, so side-by-side
  inputs that match their target drew "make it the same shape". It compares the two pictures now.
- **The Analysis page shows how each picture is cropped:** what the crop cuts off is striped in the
  warning colour with a dashed edge, so even a 2% sliver shows (it was a shadow behind an outline on
  the picture's edge), and each says where ("cuts 4% top and bottom"). **As trained** shows each
  picture in its bucket's shape, cropped from the centre exactly as training does.
- **Fixed: transparent pictures were called dark** (and flat, or as having bars) on the Analysis
  page, from the black usually kept under their see-through parts. They're judged by what shows
  now, each pixel as much as it shows; "transparent" still says the rest is there. Analysis runs
  again once.
- **Captioning leaves masked-out areas out** (Caption → Masked parts: Hidden, the default when a
  dataset has masks): the captioning model sees what the mask hides as flat grey, cropped to what it
  keeps, and is told not to describe it, so captions stop describing a background training ignores.
  Videos and pictures without a mask caption as before. The captioner also turns phone photos the
  way their EXIF says, as training already did.
- **Faces need nothing installed:** face reports and likeness scores now run on ZenTrainer's own
  port of InsightFace's models, in plain PyTorch, so the faces extra is gone (no insightface,
  onnxruntime or OpenCV, and none of their clashes with other custom nodes). Scores are the same
  as before (existing face reports and likeness stay valid), and it's a little faster. The face
  model downloads the first time faces run: 104 MB for Standard, 147 MB for Larger, into
  `models/detection/zentrainer/`. Models InsightFace downloaded to `~/.insightface` aren't used
  any more; delete them if nothing else needs them. While a dataset's faces are checked, its page
  shows how far along it is (and the face model's download, the first time), beside the Check
  faces switch and in Health.
- **A face model for commercial use:** Settings → Faces → Commercial (AuraFace) runs face reports
  and likeness on fal's AuraFace (Apache 2.0) with OpenCV's YuNet detector (MIT), instead of
  InsightFace's models, which are non-commercial. AuraFace-v1 ships InsightFace's own detector
  beside it, so ZenTrainer pairs it with YuNet: nothing InsightFace's runs. It's a little less
  accurate (LFW 99.56%, against 99.83% for the other two), and its scores sit lower (one person
  around 0.6 rather than 0.7), so compare them only with each other. A 131 MB download the
  first time.

### Runs
- **The LoRA under samples** (was Turbo LoRA) is a row like Other LoRAs: "+ Add a LoRA under
  samples", then its picker, strength and an x to clear it. A turbo one (by its name) offers
  few-step settings, and clearing it offers the ones from before. Before, the picker's "None" was
  sorted in among the LoRA files, so it couldn't be found once one was picked. The same went for
  "The model being trained", Compact's base and an App's "Auto" file: each is first now.
- **A run's sample settings change while it trains:** a try can set the LoRA under samples'
  strength (0 renders without it), and "Use for the run's samples" makes the tuned steps, CFG,
  sampler, scheduler and strength the run's own from its next sample (kept in its settings).
- **A warning when likeness keeps the checkpoints but samples ride on a LoRA:** the scores then
  measure those renders, not your LoRA alone.
- **The log says why compiled blocks recompile** (the step, what moved the weights and how many),
  and the README says what a first compile at a new resolution costs and how to log recompiles.
- **Steadier speed and ETA:** the loss chart's speed line, and the it/s and time left on a run,
  jumped about while a run blended into a new resolution (its steps come at both sizes then).
  They're now an average over the last 50 or so steps, leaving out compiles, so they read as the
  run's actual pace. The chart's line is smoother on runs trained before this too.
- **Fixed: the queue could hang with runs waiting their turn and nothing training,** a run shown
  twice (running and waiting). Two ways in: a Resume or retry right after a stop put the run back
  before the queue had let go of it, and a run waiting when ComfyUI started was sent before
  ComfyUI listened, failed, and stayed the "running" one. Now going back in line lets go of it,
  the queue waits for ComfyUI to listen, and a run that fails to reach it is tried again.
- **Fixed: Train more on a run that starts lower went back down a resolution,** and blended up
  again: its stages are shares of the run, so a longer run put them later (VRAM dropping and
  climbing, the epoch see-sawing). Train more, and Edit & resume with more steps, now keep them
  on the steps they were on: a run that reached its own resolution stays there.
- **Fixed: a run's epoch and peak VRAM jumped about** while a resolution change blends. Each
  resolution kept its own epoch count and a step takes either, so the epoch flipped between
  them; a resume restored only one count (into the wrong resolution) and started the lower one's
  shuffle over. The epoch now follows the step (passes through the run's own batches), a resume
  reshuffles every resolution from its step, and Peak VRAM is the most a step has used so far
  rather than the last step's. An open run page reads the run afresh when it starts again,
  instead of holding on its old last step.
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
- **Trim checkpoints** (System → Storage, per run or every finished run): keeps each run's final
  LoRA, its best checkpoint (by likeness, else validation loss, else the newest), its newest save
  and saves made by hand or at a stop, and deletes the other checkpoints with their resume states.
  It's the same choice a run's Keep setting makes while it trains. On one install with 455
  checkpoints, it would free 36.9 GB from 59 runs.
- **Clear all caches** (System → Storage → Datasets): every dataset's training caches at once, a
  deleted or renamed dataset's too (those weren't listed anywhere before, so nothing could clear
  them). The samples' prompt encodings stay.
- **System's cleanups show what they'll delete first,** in a dialog: each run's or dataset's files
  with their sizes (and, for checkpoints, why each goes), the total, and what stays. Nothing goes
  until Delete there, and Delete removes only what the dialog listed: what changed meanwhile (a
  new save, a score landing, a new cache) is left alone. "Dataset images" is "Datasets" now: the folders hold videos, captions and
  masks too.

### Training
- **Fixed: an open ZenTrainer page cost training about 13% of its speed.** Progress bars and the
  sidebar's job ring eased every change, but a run updates them every step, faster than the easing
  ends, so the page redrew 60 times a second for the whole run, on the GPU training uses: MOSS at
  768 px trained at 5.2 it/s with a run page open against 6.0 with Motion set to Reduced. Progress
  now jumps to each value (one redraw a step), so Motion no longer matters to training.
- **Fixed: a run with Targets on Custom and no patterns** loaded its model for a couple of minutes
  and then failed ("no layers matched network.targets=[]"). It's refused when you start it now,
  and the form says so under the patterns box.
- **Fixed: compiled blocks recompiled every graph after a run's first sample** on a card where
  ComfyUI unloads part of the model for the render and loads it back (16 GB and the like): a cluster
  of slow steps after each sample and save, 7–8 s each with a warm kernel cache on an RX 7800 XT,
  which made a compiled run slower than an uncompiled one. The graphs stay valid across that round
  trip now: the steps after a sample under a 10 GB budget went from 3.4 and 2.2 s spikes to a flat
  0.6 s (MOSS, RTX 5090).
- **Faster steps when the card has room: Keep last blocks is now Auto.** Gradient checkpointing
  recomputes every block in the backward; with the model all on the card, ZenTrainer now measures
  what a block keeps over the first steps and leaves as many blocks un-checkpointed as the free
  VRAM holds (the biggest first), checkpointing some again if a step runs out. On an RTX 5090 at
  1024 px: MOSS 0.69 → 0.50 s a step (40 of 44 blocks kept), Krea 2 1.12 → 1.08. With offloading
  that VRAM holds weights instead, so none are kept; a number in the setting is kept as before.
- **A faster int8 backward (Krea 2, MOSS, MiniMax H3 int8 bases), in raw CUDA.** The W8A8 input
  gradient now runs as two CUDA kernels with ConvRot's rotation inside the matmul, on NVIDIA cards
  from the RTX 30 series (compute capability 8.0) on: it reads the int8 weight as stored, so no
  transposed copy, and rounds to bf16 once instead of twice. On an RTX 5090 at 1024 px: Krea 2
  1.12 → 1.05 s a step, MOSS 0.69 → 0.60, the same losses. They compile when a run first needs them
  (about 2 s, with what PyTorch ships: no CUDA toolkit); other cards and shapes use the Triton
  kernels as before.
- **A LoRA on an int8 layer trains fused with it.** Its output is added inside the int8 matmul
  (comfy_kitchen's epilogue, as ComfyUI does for a block's residual when generating) and its input
  gradient inside the CUDA backward, instead of in passes of their own over the whole activation:
  Krea 2 at 1024 px 1.05 → 1.01 s a step, the same losses. Plain LoRAs on int8 bases; others, and
  sampling, as before.
- **Validation on a held-out dataset** (Advanced training → Validate on, or `validation.dataset`):
  the validation loss is measured on another dataset folder, prepared like the training one (edit
  pairs and their inputs, masks, captions), instead of on training pictures. The curve then shows
  whether the LoRA generalises to pictures it never trained on, not just how well it fits, and
  keeping checkpoints by validation loss keeps the ones that do best on them. Validation pictures
  sets how many it measures; the chart labels the curve held-out.
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
- **Fixed: Qwen Image 2.1 couldn't train.** Every run stopped before its first step: with
  "Boolean value of Tensor with more than one value is ambiguous", or out of VRAM even at 512 px,
  or "Recomputed values ... have different metadata" with compile on. Two things in ComfyUI's
  Qwen Image 2.1 are built for sampling only. Its prefix cache (the text and reference part,
  kept between sampling steps) is now off while training: it failed with a slot per training
  pair, and would have trained on that part as the LoRA was steps before, with no gradient
  through it. And each block added its output into its input in place, which broke the gradients
  and checkpointing (a 512 px step peaked at 24.8 GB against 14.4 GB without it); that's now out
  of place while training. Samples keep ComfyUI's fast paths. An edit LoRA at 768 px now trains
  at about 1.8 s a step on a 5090, 16 GB peak, with compile on.
- **Fixed: Qwen Image 2.1 failing at the end of a run, or in samples at CFG above 1,** with "The
  size of tensor a (...) must match the size of tensor b (...)". ComfyUI's prefix cache compared
  its slots by value when it moved one, so it failed once it held two of different lengths: a
  positive and negative prompt of different lengths, or a run's last validation pass over several
  pictures. It now moves the slot it found. This fails ComfyUI's own Qwen Image 2.1 sampling the
  same way, so it's fixed for everything, not just training.
- **Fixed: a run's last validation point was computed differently from the rest:** it ran after
  training mode was off, through ComfyUI's sampling code (fused kernels, caches), so the last point
  on the validation curve wasn't quite comparable. It's measured as the others are now.
- **Fixed: Memory → int8 forward and int8 backward (W8A8) showed On while the run had them off**
  when the spec held `null` for them (an int8 model then trained on the slower bf16 path). `null`
  now means the default, on, as the form showed.
- **Transparency on Qwen Image 2.1:** its VAE has an alpha channel, but training gave it every
  picture as RGB, so it saw transparent PNGs as fully opaque (whatever colour sat under the
  transparent parts showing). Targets and edit inputs now keep their alpha for a four-channel VAE,
  so an edit LoRA can learn to cut a subject out onto transparency, and its samples save as PNGs
  with it. Such a dataset encodes again once. Grounded edits show the vision encoder transparent
  pictures over white, as ComfyUI's Qwen edit node does, on every model.
- **Layers: MLP and Custom:** adapters on the feed-forward layers only ("MLP", beside Attention,
  Attention + MLP and Every layer), or on your own patterns over layer names ("Custom", one regular
  expression a line; Blocks and Skip blocks still apply). A layer that fuses attention and MLP in
  one (Flux's single blocks) only goes with Attention + MLP. A spec with patterns used to show as
  Attention + MLP on the form; it now shows them.
- **Keeping checkpoints by likeness needs Check faces:** likeness is only scored for a dataset
  with faces on, and without scores "Best likeness" kept every checkpoint. Such a run is now
  refused with what to turn on, and the form says so on those choices before you start.
- **MOSS Image · Character recipe, validated and over twice as fast:** it now trains a LoKr on
  only the MLPs of blocks 24-41, where a full run's change concentrated (36 of 176 layers), with
  Automagic v3, a 768 px start and int8 weights where available. On 37 photos it reached 87% of
  the dataset's own likeness at 1750 steps, against about 93% for every layer, at 0.2 s a step
  instead of 0.53 and a 170 MB file instead of 620 MB. Samples preview on the trained Base model.
- **Checkpointed blocks keep attention's output** (Memory → Checkpoints keep: Attention, the
  default), so the backward runs only the rest of each block again, not attention, the costliest
  part to recompute. About 50 MB a block at 1024 px. With a card's VRAM nearly used (about 23 GB
  free, Keep last blocks on Auto): Krea 2 1.02 → 0.94 s a step, MOSS 0.51 → 0.49; with nothing
  kept un-checkpointed, Krea 2 6% and MOSS 15% faster. The same losses. Not with offloading yet.
- **Fixed: compiled blocks ran an int8 layer's training path outside the compiled graph,** split
  at every layer, because of the counters behind the profile command's path list. They now
  compile whole.
- **Fixed: on Windows, training could spill out of VRAM into system RAM and crawl.** Past the card's
  own memory the NVIDIA driver doesn't run out, it carries on in "shared GPU memory", many times
  slower, so the fit that keeps blocks un-checkpointed (or weights on the card) never saw the
  out-of-VRAM step it backs off from: a later, bigger bucket could take it over the edge and keep
  it there. A run now caps what it may take at the card's own memory (bar what other programs
  hold), from loading on: a step that would spill runs out instead, before the driver is asked,
  and backs off like on Linux, with fewer blocks kept or fewer weights on the card, and the step
  retried.
- **Class preservation (objective.preserve), AI Toolkit's differential output preservation:** a
  character LoRA stops turning every woman (or man) into the character. Give class words
  ("woman, man, person"; the run form fills them in from the dataset's captions when you switch it
  on) and every 2nd step also trains its caption with one of them in the trigger's place to give
  what the base model gives. On MOSS (37 photos of one woman), "a woman ..." without the trigger
  scored 0.07 likeness to her, against 0.43 without preservation, and the trigger kept its 0.62;
  the run took 1.7x as long (every step: 2.3x, and no better). The class captions are encoded with
  the rest when the dataset is built. `preserve_weight` weighs its loss, `preserve_every` sets how
  often, and `preserve_plain` puts a share of the passes on a plain prompt ("a photo of a woman")
  at heavy noise instead (it didn't help on that dataset). The run page charts the loss
  (`preserve_loss`) on its own scale.
- **The loss chart's speed line is smooth whatever the steps:** it averages a triangle-weighted
  window the same on both sides, which slides in at the ends instead of shrinking, so steps that
  differ in turn (preservation every Nth step) no longer make it saw.
- **Two MOSS Image recipes (drafts):** *Character for finetunes*, a LoRA on every block's MLPs at
  3e-4 that kept 78% of its likeness on Serenity v6 (the blocks 24-41 recipe keeps about 60%), for
  LoRAs meant for finetunes as well as Base; and *Style*, which leaves blocks 34-41 untrained: two
  MOSS style LoRAs cut to block ranges kept none of their style there.

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
