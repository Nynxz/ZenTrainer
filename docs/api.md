# The HTTP API

The app and `zt` ([cli.md](cli.md)) both drive ZenTrainer through this API, so anything they do, a
script can. It's served by the ComfyUI ZenTrainer runs in, under `/zentrainer/api`
(`http://127.0.0.1:8189/zentrainer/api` for a managed install). Bodies and answers are JSON. A
request that's refused answers 4xx with `{"error": "what's wrong"}`.

A run is made from a **spec**: [spec.md](spec.md) lists every setting, and
[spec.schema.json](spec.schema.json) (also `GET /spec/schema`) is the same as JSON Schema.

## A run, start to finish

```python
import json, time, urllib.request

API = "http://127.0.0.1:8189/zentrainer/api"

def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req) as r:
        return json.load(r)

spec = {"spec_version": 2, "name": "my-run",
        "model": {"kind": "checkpoint", "ckpt_name": "sdxl_base.safetensors"},
        "dataset": {"folder": "my_dataset"}, "optim": {"steps": 1500}}
run_id = call("POST", "/runs", {"spec": spec})["run"]["run_id"]

seen = 0
while True:
    d = call("GET", f"/runs/{run_id}?since={seen}")   # only the metrics not seen yet
    for m in d["metrics"]:
        print(m["step"], m["total"], m["loss"])
    seen = d["metrics_count"]
    if d["run"]["status"] in ("done", "error", "stopped", "interrupted", "rejected"):
        break
    time.sleep(10)

print([c["lora_name"] for c in d["checkpoints"]])     # the saved LoRAs, in ComfyUI's loras list
```

## Runs

| Request | What it does |
|---|---|
| `POST /runs` `{"spec", "position"?}` | Queue a run. `position`: `"next"` (the default: it starts now when nothing is training) or `"end"`. Answers `{"run", "queue"}`; a spec that's wrong is refused with the first problem, by field |
| `GET /runs` | Every run: `{"runs": [{run_id, name, status, created, final_step, ...}]}` |
| `GET /runs/{id}` | One run: `run` (its status and meta), `spec`, `metrics`, `checkpoints`, `samples`, `validation`. `?since=N` sends only the metrics from row N on, with `metrics_count` the total, for polling a long run |
| `PUT /runs/{id}/spec` `{"spec"}` | Change a run's settings. A queued run keeps its place; one that ended before its first step (failed loading, say) goes back to the front of the queue. A stopped run that has trained can change everything but what its LoRA and optimizer state were made with (the model, dataset, network, optimizer, batch size, LoRA+ and seed) before it resumes; a running or finished one can't |
| `POST /runs/{id}/control` `{"command", "args"?}` | `"pause"`, `"resume"`, `"stop"`, `"save"` (a checkpoint now), `"sample"`, or `"set_lr"` (`args: {"lr"}`) on the training run |
| `POST /runs/{id}/resume` `{"checkpoint"?}` | Continue a stopped or failed run from its latest checkpoint, or the one named |
| `POST /runs/{id}/extend` `{"steps"}` | Train a finished run on, to a new total, from where it ended |
| `POST /runs/{id}/sample` | Sample a finished run's checkpoint: `{"checkpoint"?, "run_prompts"?, "tries"?: [{prompt, width?, height?, ...}]}` |
| `GET /runs/{id}/download?lora=NAME` | One of the run's LoRAs as a file (the final one without `lora`) |
| `GET /runs/{id}/export` | The run's settings as a file to share (without your dataset or folders) |
| `POST /share/import` | Read a shared settings file: its spec in this version's layout, with `warnings` for what was left out |

A run's `status`: `queued`, `loading`, `training`, `paused`, then one of `done`, `error`, `stopped`,
`interrupted` or `rejected`. Its `message` says why when it didn't finish.

The rows in `metrics`: `{step, total, loss, lr, grad_norm, epoch, elapsed, it_s, ...}`.
`checkpoints`: `{step, path, lora_name, final}`, `lora_name` as ComfyUI's LoRA loaders list it.
`validation`: `{step, val_loss, by_noise, held_out}`. `samples`: `{step, index, prompt, filename,
subfolder, type}` (the image is at ComfyUI's `/view?filename=&subfolder=&type=`).

## The queue

| Request | What it does |
|---|---|
| `GET /queue` | `{"paused", "current", "items"}`: what's training and what's waiting |
| `POST /queue` `{"action"}` | `"pause"` / `"resume"` the queue (a paused queue starts nothing new); `{"action": "top" \| "remove" \| "swap", "run_id"}` moves a run first, takes it out (the run stays), or trains it now in place of the current one; `{"action": "order", "order": [run ids]}` |

## Specs, presets and the install

| Request | What it does |
|---|---|
| `GET /spec/schema` | The spec's JSON Schema |
| `GET /recipes` | Simple mode's recipes (`engine/catalog/recipes/`): each a run's settings for a model, with its files, length, caption prompt, photo guidance and sample prompts, in the order the app lists them |
| `GET /presets/{kind}` | Saved presets: `run`, your recipes (the settings and model a new run can start from); `prompts`, sample-prompt sets. `POST` saves one `{"name", "data", "description"?}`, `DELETE /presets/{kind}/{id}` removes it |
| `GET /themes` | The theme packs (bundled, a linked ZenKit's, the user drop-in folder), without their extra CSS: `has_css` says a pack has some |
| `GET /themes/{id}/css` | One pack's extra CSS (fonts, effects), with an ETag to revalidate it; 404 when it has none |
| `GET /health` | ZenTrainer's version, what this install can do (`capabilities`: optimizers, attention backends, the GPU), whether the code or web app changed since it started |
| `GET /version?check=1` | ZenTrainer's version and commit; with `check=1`, the new commits upstream |
| `GET /system` | Memory samples (VRAM, RAM) |
| `GET /storage?disks=1` | Free space where datasets and LoRAs go |
| `GET /settings` | ZenTrainer's own settings and the folders in use |

## Live events

Training sends events on ComfyUI's websocket (`ws://127.0.0.1:8189/ws?clientId=ANY_ID`), as
`{"type": EVENT, "data": {...}}`. Each carries the `run_id`.

| `type` | `data` |
|---|---|
| `zentrainer.status` | `{run_id, status, message?, step}`: a run started, paused, ended, ... |
| `zentrainer.metrics` | `{run_id, step, total, loss, lr, grad_norm, epoch, elapsed, it_s}`: each step |
| `zentrainer.checkpoint` | `{run_id, step, path, final, lora_name}`: a LoRA was saved |
| `zentrainer.sample` | `{run_id, step, index, prompt, filename, subfolder, type}`: a sample was rendered |
| `zentrainer.likeness` | `{run_id, step, index, score}`: a sample's face against the dataset's |
| `zentrainer.validation` | `{run_id, step, val_loss, by_noise}`: a validation pass |

Polling `GET /runs/{id}?since=N` gets the same, and survives a dropped connection: the run's history
is kept on disk.
