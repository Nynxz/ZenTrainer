# Training from the command line

Everything the app does goes through ZenTrainer's HTTP API ([api.md](api.md)), and `zt` drives the
same API from a terminal: for scripts, a machine without a browser, or a queue of runs to set up at
once. A run is a **spec**, a JSON file of its settings; [spec.md](spec.md) lists every one.

## A run, start to finish

```sh
# 1. a complete spec with every setting at its default (a profile's on top, if you name one)
zt new --profile minimax-h3 > my-run.json

# 2. fill in the model and dataset, change what you want, remove what you don't
#    (an editor completes and checks it as you type: the file names its schema)

# 3. every problem in it, all at once, without queuing anything
zt check my-run.json

# 4. queue it (it starts now when nothing else is training) and watch it to the end
zt train my-run.json --follow
```

`zt train` prints the new run's id, starting ComfyUI in the background first if it isn't running.
With `--follow` it shows progress until the run ends and exits 0 when it finished, 1 when it
failed or was stopped, so a script can chain on it.

A spec only needs what differs from the defaults. The smallest one that trains:

```json
{
  "$schema": "https://raw.githubusercontent.com/Nynxz/ZenTrainer/main/docs/spec.schema.json",
  "spec_version": 2,
  "model": {"kind": "checkpoint", "ckpt_name": "sdxl_base.safetensors"},
  "dataset": {"folder": "my_dataset"}
}
```

More in [examples/](../examples/).

## Checking a spec

`zt check` reports every problem at once, with the field and, for a typo, what it likely meant:

```
$ zt check my-run.json
› version 1: read as version 2 (renamed fields carried over)
  x unknown field: optim.lerning_rate (did you mean optim.learning_rate?)
  x dataset.resolution is whole number, got "big"
  x optim.optimizer: "Adamw" isn't one of its choices; did you mean "AdamW"?
  x optim.steps must be >= 1, got 0
```

It runs offline with the same rules the server uses, and `zt train` runs it first, so a bad spec
never reaches the queue. What only the running install can tell (whether the model files and the
dataset folder exist) is checked when the run is queued.

An older spec (no `spec_version`, or `1`) is read as the current version: its renamed settings carry
over, as the note says.

## Changing settings without editing the file

`--set` changes one setting: a dotted key and a JSON value (a bare word is text). It works on `new`,
`check` and `train`, so one file can be a template for several runs:

```sh
for lr in 5e-5 1e-4 2e-4; do
  zt train my-run.json --set name="lr-$lr" --set optim.learning_rate=$lr --end
done
```

`--end` puts each run at the end of the queue instead of starting it next. `--profile ID` starts from
an engine profile's settings (engine/profiles/), with the file's own on top.

## The commands

| Command | What it does |
|---|---|
| `zt new [--profile ID] [--set k=v]...` | Print a complete spec with every default, to start from |
| `zt check SPEC [--profile ID] [--set k=v]...` | Every problem in a spec, without queuing it (exit 1 if any) |
| `zt train SPEC [--profile ID] [--set k=v]... [--end] [--follow]` | Queue a run (`-` reads the spec from stdin); prints its id |
| `zt runs [--json]` | List the runs |
| `zt spec RUN` | Print a run's settings as a spec (`> again.json` to reuse them) |
| `zt follow RUN` | Show a run's progress until it ends (exit 0 = done) |
| `zt pause`, `unpause`, `stop`, `save RUN` | Control a training run; `save` writes a checkpoint now |
| `zt resume RUN [--set k=v]... [--follow]` | Continue a stopped run from its latest checkpoint (not model, dataset or network settings) |
| `zt sample RUN "prompt" [--base] [--size WxH] [--seed N]` | Render a prompt with the current LoRA, mid-run (`--base`: without it, to compare) |
| `zt queue [pause\|resume]` | Show the queue, or pause or resume it |
| `zt caption DATASET --model FILE [--style nl\|tags] [--all]` | Auto-caption a dataset with a vision text encoder |
| `zt profile RUN` | Profile one training step: where the GPU time goes |
| `zt shrink PATH... [--dtype bf16\|fp16]` | Rewrite fp32 LoRA files in bf16 (half the size) |
| `zt monitor` | A live dashboard of the queue, runs and memory |

Every command that talks to ZenTrainer takes `--port N` (default: the one `zt start` used).
`zt --help` and `zt COMMAND --help` say the same.
