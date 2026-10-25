"""Pure tests: specs, run store, objective, schedules, data bucketing, control.
No ComfyUI needed."""

import math
import re
import types
from pathlib import Path

import pytest
import torch

from zentrainer.harness import control, data, optim
from zentrainer.harness.objective import Objective, sample_percents
from zentrainer.runs.store import RunStore
from zentrainer.spec import ObjectiveSpec, OptimSpec, SpecError, TrainSpec


def _spec(**model):
    return {"name": "My Style", "dataset": {"folder": "style"}, "model": {"ckpt_name": "sdxl.safetensors", **model}}


def test_spec_roundtrip_and_validation():
    spec = TrainSpec.from_dict(_spec())
    assert TrainSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(SpecError, match="clip_type"):
        TrainSpec.from_dict(_spec(kind="split", unet_name="u", clip_name="c", vae_name="v", ckpt_name=None))
    with pytest.raises(SpecError, match="lr"):
        TrainSpec.from_dict({**_spec(), "optim": {"lr": 1}})
    with pytest.raises(SpecError, match="scheduler"):
        TrainSpec.from_dict({**_spec(), "optim": {"scheduler": "warp"}})
    TrainSpec.from_dict({**_spec(), "samples": {"every": 10}})  # no prompts: sampling is just skipped


def test_run_store_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    rid = store.new_id()
    store.create(rid, {"name": "x"}, prompt_id="p")
    store.append(rid, "metrics", {"step": 1, "loss": 0.5})
    store.update(rid, status="training")
    assert store.meta(rid)["status"] == "training"
    assert store.read(rid, "metrics") == [{"step": 1, "loss": 0.5}]
    assert [m["run_id"] for m in store.list()] == [rid]
    with pytest.raises(ValueError):
        store.dir("../etc")


class EpsSampling:
    """Minimal stand-in for ComfyUI's EPS model_sampling: x_t = x0 + σ·ε."""

    def noise_scaling(self, sigma, noise, latent, max_denoise=False):
        return latent + sigma.view(-1, *([1] * (noise.ndim - 1))) * noise


class FlowSampling:
    """Stand-in for CONST (flow) model_sampling: x_t = (1-σ)·x0 + σ·ε."""

    def noise_scaling(self, sigma, noise, latent, max_denoise=False):
        s = sigma.view(-1, *([1] * (noise.ndim - 1)))
        return (1 - s) * latent + s * noise


def test_native_target_is_eps_for_eps_models_and_velocity_for_flow():
    torch.manual_seed(0)
    x0, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    sigma = torch.tensor([0.3, 0.8])
    obj = Objective(ObjectiveSpec(target="native"))
    for ms, true_target in ((EpsSampling(), noise), (FlowSampling(), noise - x0)):
        x_t = obj.noisy(ms, sigma, noise, x0)
        assert torch.allclose((x_t - x0) / sigma.view(-1, 1, 1, 1), true_target, atol=1e-5)
        assert obj.loss(x_t, x0, x0, sigma).abs().max() < 1e-10  # perfect prediction -> zero loss
        assert obj.loss(x_t, x0, x0 + 0.1, sigma).shape == (2,)


def test_timestep_sampling_stays_inside_the_schedule():
    for kind in ("uniform", "logit_normal"):
        p = sample_percents(ObjectiveSpec(timesteps=kind), 10_000, torch.Generator().manual_seed(0))
        assert 0 < p.min() and p.max() < 1
    lp = sample_percents(ObjectiveSpec(timesteps="logit_normal"), 10_000, torch.Generator().manual_seed(0))
    assert abs(lp.mean().item() - 0.5) < 0.02 and lp.std() < 0.25  # concentrated mid-schedule


def test_schedules_warmup_then_decay():
    p = [torch.nn.Parameter(torch.zeros(1))]
    for name, end in (("constant", 1.0), ("linear", 0.0), ("cosine", 0.0)):
        opt, sched = optim.build(p, OptimSpec(scheduler=name, warmup_steps=10, steps=110, learning_rate=1.0))
        lrs = []
        for _ in range(110):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(end, abs=0.02)
    optim.set_lr(opt, sched, 0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * sched.lr_lambdas[0](sched.last_epoch))


def test_buckets_respect_area_step_and_pick_by_aspect():
    sizes = data.bucket_sizes(1024, 64)
    assert all(w % 64 == 0 and h % 64 == 0 and w * h <= 1024 * 1024 for w, h in sizes)
    assert (1024, 1024) in sizes
    assert data.pick_bucket(4000, 3000, sizes) == min(sizes, key=lambda s: abs(math.log(s[0] / s[1]) - math.log(4 / 3)))
    assert data.pick_bucket(1000, 1000, sizes) == (1024, 1024)


def test_scan_reads_captions_and_kohya_repeats(tmp_path):
    from PIL import Image

    (tmp_path / "3_subject").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("a cat\n")
    Image.new("RGB", (8, 8)).save(tmp_path / "3_subject" / "b.jpg")  # no caption
    for junk in (".thumbs", "_latent_cache"):  # left behind by other trainers
        (tmp_path / junk).mkdir()
        Image.new("RGB", (8, 8)).save(tmp_path / junk / "x.jpg")
    items = {i.path.name: i for i in data.scan(tmp_path)}
    assert "x.jpg" not in items
    assert items["a.png"].caption == "a cat" and items["a.png"].repeats == 1
    assert items["b.jpg"].caption == "" and items["b.jpg"].repeats == 3


def test_batches_are_same_shape_and_cover_each_epoch():
    lat = lambda h: data.Sample(torch.zeros(4, h, 8), 0)  # noqa: E731
    ds = data.Dataset([lat(8), lat(8), lat(8), lat(16)], [None], [""], 0)
    ds = data._expand_repeats(ds, [data.Item(None, "", 1)] * 4)
    b = data.Batches(ds, batch_size=2, seed=0)
    for _ in range(2):  # two epochs: each covers every sample exactly once, in same-shape batches
        b._refill()
        epoch = list(b._queue)
        assert all(len({tuple(ds.samples[i].latent.shape) for i in idx}) == 1 for idx in epoch)
        assert sorted(i for idx in epoch for i in idx) == [0, 1, 2, 3]


def test_control_commands_and_pause():
    c = control.Control()
    c.send("pause")
    assert c.paused
    c.send("set_lr", lr=1e-5)
    c.send("resume")
    assert not c.paused
    assert [x.name for x in c.drain()] == ["pause", "set_lr", "resume"]
    with pytest.raises(ValueError):
        c.send("explode")


def test_themes_later_source_wins_and_css_refs_inline(tmp_path):
    import json

    from zentrainer.server.themes import BUNDLED, list_themes

    bundled = {p["id"] for p in list_themes([BUNDLED])}
    assert {"maroon-volt", "strawberry-milk", "nynxz"} <= bundled
    user = tmp_path / "user"
    (user / "maroon-volt").mkdir(parents=True)
    (user / "maroon-volt" / "theme.json").write_text(json.dumps(
        {"id": "maroon-volt", "name": "Mine", "tokens": {"dark": {}}, "css": "extra.css"}))
    (user / "maroon-volt" / "extra.css").write_text(":root{--x:1}")
    packs = {p["id"]: p for p in list_themes([BUNDLED, user])}
    assert packs["maroon-volt"]["name"] == "Mine"
    assert packs["maroon-volt"]["css"] == ":root{--x:1}"


def test_targets_match_by_structure_across_families():
    from zentrainer.harness.network import matches

    attn = [
        "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q",  # SD/SDXL
        "diffusion_model.double_blocks.3.img_attn.qkv",  # Flux
        "diffusion_model.blocks.7.attn.wq",  # Krea 2
        "diffusion_model.layers.2.self_attn.q_proj",  # Llama-style DiT
        "diffusion_model.layers.5.proj.q",  # MOSS Image: only the layer's own name says it
        "diffusion_model.text_embedder.blocks.0.qkv",
    ]
    mlp = [
        "diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj",
        "diffusion_model.double_blocks.3.img_mlp.0",
        "diffusion_model.blocks.7.mlp.up",
        "diffusion_model.single_blocks.0.linear1",
        "diffusion_model.layers.5.proj.ffn_up_gate",
        "diffusion_model.text_embedder.blocks.0.ffn_down",
    ]
    other = ["diffusion_model.first", "diffusion_model.last.linear", "diffusion_model.time_in.in_layer",
             "diffusion_model.video_embedder", "diffusion_model.shared_block_modulation", "diffusion_model.q"]  # leaf names count only in blocks
    assert all(matches(n, "attn") for n in attn) and not any(matches(n, "attn") for n in mlp)
    fused = "diffusion_model.single_blocks.0.linear1"  # attention and MLP in one: only attn_mlp has it
    assert all(matches(n, "mlp") for n in mlp if n != fused) and not any(matches(n, "mlp") for n in attn + other + [fused])
    assert all(matches(n, "attn_mlp") for n in attn + mlp)
    assert not any(matches(n, "attn_mlp") for n in other)
    assert all(matches(n, "all") for n in other)
    assert matches("diffusion_model.blocks.7.attn.wq", [r"\.wq$"]) and not matches("x.attn.wk", [r"\.wq$"])


def test_dataset_editing_stays_inside_the_folder(tmp_path, monkeypatch):
    from PIL import Image

    from zentrainer.datasets import folders as datasets

    monkeypatch.setattr(datasets, "root", lambda: tmp_path.resolve())
    datasets.create("cats")
    with pytest.raises(datasets.DatasetError):
        datasets.create("cats")
    for bad in ("../etc", ".hidden", "_cache", "a/b"):
        with pytest.raises((datasets.DatasetError, FileNotFoundError)):
            datasets.folder(bad)
    buf = __import__("io").BytesIO()
    Image.new("RGB", (16, 8)).save(buf, "PNG")
    assert datasets.save_upload("cats", "../../evil/a.png", buf.getvalue()) == "a.png"
    with pytest.raises(datasets.DatasetError):
        datasets.save_upload("cats", "run.sh", b"#!")
    datasets.set_caption("cats", "a.png", "a cat")
    assert datasets.bulk("cats", "prefix", "zt_cat") == 1
    assert datasets.bulk("cats", "prefix", "zt_cat") == 0  # idempotent
    rows = datasets.items("cats")["items"]
    assert rows[0]["caption"] == "zt_cat, a cat" and rows[0]["width"] == 16
    with pytest.raises(datasets.DatasetError):
        datasets.set_caption("cats", "../../x.png", "nope")
    assert datasets.save_upload("cats", "a-masklabel.png", buf.getvalue()) == "_masks/a.png"
    assert datasets.items("cats")["items"][0]["mask"] == "_masks/a.png"
    thumb = datasets.thumbnail("cats", "a.png", 64, tmp_path / "thumbs")
    assert thumb.is_file() and datasets.thumbnail("cats", "a.png", 64, tmp_path / "thumbs") == thumb  # cached
    datasets.delete("cats", "a.png")
    assert datasets.items("cats")["items"] == [] and not (tmp_path / "cats" / "a.txt").exists()
    assert not (tmp_path / "cats" / "_masks" / "a.png").exists()


def test_queue_order_pause_and_persistence(tmp_path):
    from zentrainer.server.jobqueue import Queue

    q = Queue(tmp_path / "queue.json")
    for r in ("a", "b", "c"):
        q.add(r)
    q.add("c", front=True)
    assert q.snapshot()["order"] == ["c", "a", "b"]
    q.reorder(["b", "zzz", "a"])
    assert q.snapshot()["order"] == ["b", "a", "c"]
    q.set_paused(True)
    assert q.pop_next() is None
    q.set_paused(False)
    assert q.pop_next() == "b"
    assert q.remove("c") and not q.remove("c")
    assert Queue(tmp_path / "queue.json").snapshot()["order"] == ["a"]  # survives a restart


def test_a_parked_run_holds_the_queue_until_it_is_resumed(tmp_path, monkeypatch):
    # a paused run that let go of ComfyUI for a workflow (Trainer._park) stops with park set: the
    # worker hands it to `parked`, and starts nothing else while any run is parked
    import asyncio

    from zentrainer.harness import control
    from zentrainer.server import jobqueue

    class Store:
        metas = {"a": {"status": "stopped", "park": True}}

        def meta(self, run_id):
            return self.metas[run_id]

    q = jobqueue.Queue(tmp_path / "queue.json")
    q.add("b")
    q.current = "a"
    monkeypatch.setattr(jobqueue, "get", lambda: q)
    monkeypatch.setattr(jobqueue.state, "store", lambda: Store())
    monkeypatch.setattr(control, "get", lambda run_id: None)
    monkeypatch.setattr(control, "active", lambda: False)
    dispatched, parked = [], []

    async def dispatch(run_id):
        dispatched.append(run_id)

    async def on_parked(run_id):
        parked.append(run_id)
        q.parked.add(run_id)

    async def run(seconds):
        task = asyncio.ensure_future(jobqueue.worker(dispatch, None, on_parked, interval=0.01))
        await asyncio.sleep(seconds)
        task.cancel()

    asyncio.run(run(0.1))
    assert parked == ["a"] and q.current is None and dispatched == []  # held: b waits
    q.parked.discard("a")  # resumed (or stopped)
    asyncio.run(run(0.1))
    assert dispatched == ["b"]


def test_parking_only_when_another_comfyui_prompt_waits(monkeypatch):
    import sys
    import types

    from zentrainer.harness import trainer

    remaining = [1]
    fake = types.ModuleType("server")
    fake.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
        prompt_queue=types.SimpleNamespace(get_tasks_remaining=lambda: remaining[0])))
    monkeypatch.setitem(sys.modules, "server", fake)
    assert not trainer._comfy_waiting()  # only the run itself
    remaining[0] = 2
    assert trainer._comfy_waiting()  # a workflow queued behind it
    monkeypatch.setitem(sys.modules, "server", types.ModuleType("server"))  # no queue to ask
    assert not trainer._comfy_waiting()


def test_forward_only_attention_trains_with_pytorch(monkeypatch):
    # comfy_kitchen_int8 (like sage) has no backward and refuses autocast's fp32 q/k beside a bf16 v:
    # a run set to it (from before it was hidden in the form) trains with pytorch instead
    import sys
    import types

    from zentrainer.harness.memory import attention_override

    calls = []
    mod = types.ModuleType("comfy.ldm.modules.attention")
    mod.get_attention_function = lambda name: (lambda *a, **k: calls.append(name))
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", mod)

    attention_override("comfy_kitchen_int8")["optimized_attention_override"](None, 1, 2, 3)
    attention_override("xformers")["optimized_attention_override"](None, 1, 2, 3)
    assert calls == ["pytorch", "xformers"]
    assert attention_override("default") == {}


def test_checkpointing_wraps_each_block_not_the_whole_model():
    import torch.nn as nn

    from zentrainer.harness.memory import block_lists

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    class Net(nn.Module):  # like Krea 2: one block list plus a nested text model
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.blocks = nn.ModuleList([Block() for _ in range(4)])
            self.txt = nn.Sequential(nn.ModuleList([Block(), Block()]))

    net = Net()
    top = block_lists(net, 1)
    assert len(top) == 6 and all(isinstance(b, Block) for b in top)
    assert len(block_lists(net, 2)) == 12  # one list deeper


def test_masks_are_found_skipped_as_images_and_weight_the_loss(tmp_path):
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    Image.new("L", (8, 8), 255).save(tmp_path / "a-masklabel.png")  # OneTrainer naming
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    (tmp_path / "_masks").mkdir()
    Image.new("L", (8, 8), 255).save(tmp_path / "_masks" / "b.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "c.png")
    items = {i.path.name: i for i in data.scan(tmp_path, warn=False)}
    assert set(items) == {"a.png", "b.png", "c.png"}  # mask files aren't training images
    assert items["a.png"].mask.name == "a-masklabel.png"
    assert items["b.png"].mask == tmp_path / "_masks" / "b.png"
    assert items["c.png"].mask is None

    obj = Objective(ObjectiveSpec(target="x0"))
    x0 = torch.zeros(1, 4, 2, 2)
    pred = x0.clone()
    pred[..., 0, 0] = 1.0  # error only in the top-left latent pixel
    sigma = torch.tensor([0.5])
    left_only = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    right_only = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    assert obj.loss(x0, x0, pred, sigma, left_only).item() == pytest.approx(1.0)  # error is all it sees
    assert obj.loss(x0, x0, pred, sigma, right_only).item() == pytest.approx(0.0)  # error masked out
    assert obj.loss(x0, x0, pred, sigma).item() == pytest.approx(0.25)  # unmasked: averaged


def test_presets_builtin_user_and_validity(tmp_path):
    from zentrainer import presets

    runs = presets.list_presets("run", tmp_path)
    assert {"quick-test", "character", "style", "flow-tweaks"} <= {r["id"] for r in runs}
    base = TrainSpec.from_dict(_spec()).to_dict()
    for r in runs:  # every built-in merges into a valid spec
        merged = {**base, **{k: ({**base[k], **v} if isinstance(v, dict) else v) for k, v in r["data"].items()}}
        TrainSpec.from_dict(merged)
    presets.save("prompts", "Portrait checks", {"prompts": ["a portrait"], "negative": ""}, tmp_path)
    assert [p["id"] for p in presets.list_presets("prompts", tmp_path)] == ["portrait-checks"]
    with pytest.raises(presets.PresetError):
        presets.save("prompts", "bad", {"negative": "x"}, tmp_path)
    with pytest.raises(presets.PresetError):
        presets.delete("run", "../x", tmp_path)
    assert presets.delete("prompts", "portrait-checks", tmp_path)
    assert not presets.delete("run", "character", tmp_path)  # built-ins stay


def test_trigger_word_like_ai_toolkit():
    from zentrainer.harness.data import apply_trigger

    assert apply_trigger("a photo of [trigger] smiling", "ohwx", True) == "a photo of ohwx smiling"
    assert apply_trigger("a woman smiling", "ohwx", True) == "ohwx, a woman smiling"
    assert apply_trigger("OHWX at the beach", "ohwx", True) == "OHWX at the beach"  # already there
    assert apply_trigger("a woman smiling", "ohwx", False) == "a woman smiling"
    assert apply_trigger("", "ohwx", True) == "ohwx"
    assert apply_trigger("[trigger], a cat", "", True) == "a cat"  # no trigger set: placeholder removed


def test_optimizer_catalog_builds_what_is_available():
    p = [torch.nn.Parameter(torch.ones(4))]
    for row in optim.catalog():
        if not row["available"]:
            with pytest.raises(RuntimeError, match=row["pip"]):
                optim.build(p, OptimSpec(optimizer=row["id"]))
            continue
        # at the LR its catalog row recommends: an adaptive one (Prodigy: 1.0) steps by lr x its own
        # estimate (1e-6 at first), so a normal LoRA LR moves a weight of 1 by less than fp32 can show
        lr = float(re.match(r"[0-9.]+(?:e-?[0-9]+)?", row["lr_hint"]).group())
        opt, sched = optim.build([torch.nn.Parameter(torch.ones(4))], OptimSpec(optimizer=row["id"], learning_rate=lr))
        optim.set_mode(opt, training=True)  # as a run does (schedule-free ones refuse to step otherwise)
        q = opt.param_groups[0]["params"][0]
        for _ in range(2 if row["id"] == "Automagic4" else 1):  # its first step only measures
            q.grad = torch.ones_like(q)
            opt.step()
        assert not torch.equal(q.detach(), torch.ones(4)), row["id"]  # it moved


def test_caption_sets_read_write_and_mix(tmp_path, monkeypatch):
    from PIL import Image

    from zentrainer.datasets import folders as datasets
    from zentrainer.datasets.captioner import clean

    monkeypatch.setattr(datasets, "root", lambda: tmp_path.resolve())
    datasets.create("d")
    Image.new("RGB", (8, 8)).save(tmp_path / "d" / "a.png")
    (tmp_path / "d" / "a.txt").write_text("a woman")
    datasets.set_caption("d", "a.png", "1girl, smiling", set_name="tags")
    info = datasets.items("d")
    assert info["sets"] == ["default", "tags"]
    assert info["items"][0]["captions"] == {"default": "a woman", "tags": "1girl, smiling"}
    assert datasets.bulk("d", "prefix", "ohwx", set_name="tags") == 1
    assert (tmp_path / "d" / "a.tags.txt").read_text().strip() == "ohwx, 1girl, smiling"
    assert (tmp_path / "d" / "a.txt").read_text().strip() == "a woman"  # other set untouched
    with pytest.raises(datasets.DatasetError):
        datasets.set_caption("d", "a.png", "x", set_name="../evil")
    items = data.scan(tmp_path / "d", warn=False)
    assert len(items) == 1 and items[0].captions["tags"] == "ohwx, 1girl, smiling"  # sidecars aren't images
    datasets.delete("d", "a.png")
    assert not list((tmp_path / "d").glob("a.*"))
    assert clean('<think>hmm</think> Caption: "A cat on a mat."') == "A cat on a mat."


def test_apps_load_compile_chain_and_sweep(tmp_path):
    import json

    from zentrainer.apps import compile as ac
    from zentrainer.apps import manifest as am

    apps, errors = am.list_apps(None)
    assert not errors and {"text-to-image", "test-lora"} <= {a.id for a in apps}
    gen = next(a for a in apps if a.id == "text-to-image")
    assert {"krea-2", "z-image-turbo", "flux-3-image"} <= set(gen.families())

    # one workflow per model: Krea 2's; no LoRAs, so the placeholder goes and the sampler reads the loader
    [cell] = ac.compile_batch(gen, {"model": "krea-2", "prompt": "a cat", "seed": 7})
    g = cell["prompt"]
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in g.values())
    sampler = next(n for n in g.values() if n["_meta"]["title"] == "Sampler")
    loader = next(k for k, n in g.items() if n["class_type"] == "UNETLoader")
    assert sampler["inputs"]["model"] == [loader, 0] and sampler["inputs"]["seed"] == 7
    assert next(n for n in g.values() if n["_meta"]["title"] == "Prompt")["inputs"]["text"] == "a cat"
    # it starts from the workflow's own values, and Krea 2 (CFG 1) has no negative to show
    assert ac.defaults_for(gen, "krea-2")["steps"] == 8 and ac.defaults_for(gen, "krea-2")["cfg"] == 1.0
    assert "negative" not in gen.bound_in(am.pick_workflow(gen, None, "krea-2"))
    with pytest.raises(am.AppError, match="pick a model"):
        ac.compile_batch(gen, {"prompt": "a cat"})
    # a fine-tune swaps in for the workflow's build; LoRA rows switched off stay unapplied
    [cell] = ac.compile_batch(gen, {"model": "krea-2", "prompt": "a cat", "model_file": "my_krea_finetune.safetensors",
                                    "loras": [{"name": "a.safetensors", "strength": 1, "on": False}]})
    g = cell["prompt"]
    assert next(n for n in g.values() if n["class_type"] == "UNETLoader")["inputs"]["unet_name"] == "my_krea_finetune.safetensors"
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in g.values())
    # a size is a pair, not a sweepable list: both sides land
    [cell] = ac.compile_batch(gen, {"model": "krea-2", "prompt": "a cat", "size": [832, 1216]})
    size = next(n for n in cell["prompt"].values() if n["_meta"]["title"] == "Size")["inputs"]
    assert (size["width"], size["height"]) == (832, 1216)

    # two LoRAs chain in order; seeds sweep into two cells
    cells = ac.compile_batch(gen, {"model": "krea-2", "prompt": "a cat",
                                   "loras": [{"name": "a.safetensors", "strength": 0.7}, {"name": "b.safetensors"}]},
                             {"seed": [1, 2]})
    assert [c["coords"] for c in cells] == [{"seed": 1}, {"seed": 2}]
    g = cells[0]["prompt"]
    chain = [n for n in g.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert [n["inputs"]["lora_name"] for n in chain] == ["a.safetensors", "b.safetensors"]
    assert chain[1]["inputs"]["model"] == [next(k for k, n in g.items() if n is chain[0]), 0]

    # Test LoRA: checkpoints x strengths x prompts, split layout picks the split workflow
    tl = next(a for a in apps if a.id == "test-lora")
    split = {"kind": "split", "unet_name": "krea2.safetensors", "clip_name": "q.safetensors", "clip_type": "krea2", "vae_name": "v"}
    cells = ac.compile_batch(tl, {"model": split, "seed": 42},
                             {"lora": ["s250.safetensors", "s500.safetensors"], "strength": [1.0, 1.4], "prompt": ["ohwx", "ohwx smiling"]})
    assert len(cells) == 8
    c = next(c for c in cells if c["coords"] == {"lora": "s500.safetensors", "strength": 1.4, "prompt": "ohwx smiling"})
    g = c["prompt"]
    lora = next(n for n in g.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert lora["inputs"]["lora_name"] == "s500.safetensors" and lora["inputs"]["strength_model"] == 1.4
    assert next(n for n in g.values() if n["class_type"] == "UNETLoader")["inputs"]["unet_name"] == "krea2.safetensors"
    assert next(n for n in g.values() if n["_meta"]["title"] == "Positive")["inputs"]["text"] == "ohwx smiling"

    with pytest.raises(am.AppError, match="limit"):
        ac.compile_batch(gen, {"model": "krea-2"}, {"seed": list(range(65))})

    # a manifest binding to a node that isn't there fails at load
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "w.json").write_text(json.dumps({"1": {"class_type": "X", "_meta": {"title": "A"}, "inputs": {"t": ""}}}))
    (bad / "app.json").write_text(json.dumps({"name": "Bad", "workflows": [{"file": "w.json"}],
                                              "inputs": [{"id": "p", "type": "prompt", "bind": "Missing.text"}]}))
    with pytest.raises(am.AppError, match="Missing"):
        am.load(bad)


def test_app_extends_inherits_overrides_and_removes(tmp_path):
    import json

    from zentrainer.apps import compile as ac
    from zentrainer.apps import manifest as am

    child = tmp_path / "quick"
    child.mkdir()
    (child / "app.json").write_text(json.dumps({
        "id": "quick", "name": "Quick generate", "extends": "text-to-image",
        "remove": ["loras", "negative"],
        "inputs": [
            {"id": "steps", "type": "number", "label": "Steps", "bind": "Sampler.steps | Scheduler.steps", "default": 8, "optional": True},
            {"id": "denoise", "type": "number", "label": "Denoise", "bind": "Sampler.denoise", "default": 0.6, "optional": True},
        ],
    }))
    app = am.load(child, builtin=False)
    ids = [i["id"] for i in app.inputs]
    assert "loras" not in ids and "negative" not in ids
    assert ids.index("steps") < ids.index("denoise")  # override stays in the parent's place; new ones append
    assert app.input("steps")["default"] == 8 and app.public()["extends"] == "text-to-image"
    [cell] = ac.compile_batch(app, {"model": "z-image-turbo", "prompt": "hi", "seed": 1, "steps": 8, "denoise": 0.6})
    sampler = next(n for n in cell["prompt"].values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["steps"] == 8 and sampler["inputs"]["denoise"] == 0.6

    loop = tmp_path / "a"
    loop.mkdir()
    (loop / "app.json").write_text(json.dumps({"name": "A", "extends": "a", "inputs": []}))
    with pytest.raises(am.AppError, match="loop"):
        am.load(loop)


def test_training_shift_reskews_flow_sigmas():
    import torch

    from zentrainer.harness.objective import sigmas_for

    p = torch.tensor([0.5])  # mid schedule: noise fraction 0.5
    assert torch.allclose(sigmas_for(None, p, 1.0), torch.tensor([0.5]))
    assert torch.allclose(sigmas_for(None, p, 5.0), torch.tensor([5 * 0.5 / (1 + 4 * 0.5)]))  # MOSS's own: ~0.83
    assert sigmas_for(None, p, 2.0) < sigmas_for(None, p, 5.0)  # lower shift, less noise


def test_storage_sizes_and_safe_cleanup(tmp_path, monkeypatch):
    from zentrainer.runs import disk as storage

    class Store:
        def __init__(self):
            self.metas = {"a": {"run_id": "a", "name": "a", "status": "done"}, "b": {"run_id": "b", "name": "b", "status": "stopped"}}

        def list(self):
            return list(self.metas.values())

        def meta(self, rid):
            return self.metas[rid]

        def read(self, rid, stream):
            return []

    states = tmp_path / "states"
    for rid in ("a", "b"):
        (states / rid).mkdir(parents=True)
        (states / rid / "s.pt").write_bytes(b"x" * 1000)
    monkeypatch.setattr(storage, "_roots", lambda: {"states": states, "loras": tmp_path, "samples": tmp_path / "s", "apps": tmp_path / "x"})
    store = Store()
    assert storage.size(states) == 2000
    assert storage.clean_states(store, "a") == 1000 and not (states / "a").exists()
    with pytest.raises(storage.StorageError, match="resuming"):
        storage.clean_states(store, "b")  # stopped: its states are what resume needs
    store.metas["a"]["status"] = "training"
    with pytest.raises(storage.StorageError, match="active"):
        storage.clean_states(store, "a")


def test_output_spec_defaults_and_validation():
    from zentrainer.spec import SpecError, TrainSpec

    s = TrainSpec.from_dict({"model": {"kind": "checkpoint", "ckpt_name": "x.safetensors"}, "dataset": {"folder": "d"}})
    assert (s.output.lora_dir, s.output.keep_states) == (None, "latest")  # inside the install; newest state only
    s.output.keep_states = "some"
    with pytest.raises(SpecError, match="keep_states"):
        s.validate()


def _toy_lora(dtype, make, steps=1200):
    import torch

    torch.manual_seed(0)
    d, r = 128, 8
    target = torch.randn(d, r) @ torch.randn(r, d) * 0.02
    x = torch.randn(1024, d)
    y = x @ target.T
    down = torch.nn.Parameter((torch.randn(r, d) / d**0.5).to(dtype))
    up = torch.nn.Parameter(torch.zeros(d, r, dtype=dtype))
    opt = make([down, up])
    for _ in range(steps):
        idx = torch.randint(0, 1024, (64,))
        loss = torch.nn.functional.mse_loss((x[idx].to(dtype) @ down.T @ up.T).float(), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return 1 - torch.nn.functional.mse_loss((x.to(dtype) @ down.T @ up.T).float(), y).item() / y.pow(2).mean().item()


def test_bf16_lora_loses_adamw_steps_and_automagic_keeps_them():
    import torch

    from zentrainer.harness.optim import Automagic

    adam = lambda p: torch.optim.AdamW(p, lr=1e-4)  # noqa: E731
    assert _toy_lora(torch.float32, adam) > 2 * _toy_lora(torch.bfloat16, adam)  # why LoRAs default to fp32
    auto_bf16 = _toy_lora(torch.bfloat16, lambda p: Automagic(p))
    assert auto_bf16 > _toy_lora(torch.bfloat16, adam) * 2  # stochastic rounding lands the steps


def test_stochastic_rounding_is_unbiased():
    import torch

    from zentrainer.harness.optim import _stochastic_copy

    src = torch.full((20000,), 1.0 + 1e-3)  # between two bf16 values
    dst = torch.empty(20000, dtype=torch.bfloat16)
    _stochastic_copy(dst, src)
    assert abs(dst.float().mean().item() - (1.0 + 1e-3)) < 2e-4  # nearest rounding would give exactly 1.0


def test_muting_per_caption_set(tmp_path, monkeypatch):
    from PIL import Image

    from zentrainer.datasets import folders as datasets
    from zentrainer.harness.data import muted

    monkeypatch.setattr(datasets, "root", lambda: tmp_path.resolve())
    datasets.create("cats")
    d = datasets.folder("cats")
    for n in ("a", "b", "c"):
        Image.new("RGB", (8, 8)).save(d / f"{n}.png")
    datasets.set_muted("cats", "default", ["b.png"], True)
    datasets.set_muted("cats", "tags", ["c.png"], True)
    assert muted(d) == {"default": {"b.png"}, "tags": {"c.png"}}
    rows = {r["path"]: r["muted"] for r in datasets.items("cats")["items"]}
    assert rows == {"a.png": [], "b.png": ["default"], "c.png": ["tags"]}
    datasets.set_muted("cats", "default", ["b.png"], False)
    assert muted(d) == {"tags": {"c.png"}}
    with pytest.raises(datasets.DatasetError):
        datasets.set_muted("cats", "default", ["../x.png"], True)


def test_conds_that_cannot_stack_get_their_own_call():
    from zentrainer.harness.denoiser import batches

    # captions by token length; like MOSS, only equal lengths stack
    parts = [151, 129, 151, 129, 77]
    assert batches(parts, lambda a, b: a == b) == [[0, 2], [1, 3], [4]]
    assert batches(parts, lambda a, b: True) == [[0, 1, 2, 3, 4]]


def test_training_on_keeps_the_old_final_as_a_checkpoint(tmp_path, monkeypatch):
    import sys
    import types

    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path / "output")
    fp.get_user_directory = lambda: str(tmp_path / "user")
    fp.get_folder_paths = lambda _kind: []
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    from zentrainer.runs.extend import keep_final

    store = RunStore(tmp_path / "runs")
    rid = store.new_id()
    store.create(rid, {"name": "x"})
    loras = tmp_path / "output" / "zentrainer" / "loras"
    final = loras / "x" / f"x_{rid}.safetensors"
    state = tmp_path / "user" / "zentrainer" / "states" / rid / f"x_{rid}.pt"
    for f in (final, state):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"w")
    store.append(rid, "checkpoints", {"step": 250, "path": str(loras / "checkpoints" / rid / "a.safetensors"), "final": False})
    store.append(rid, "checkpoints", {"step": 500, "path": str(final), "lora_name": f"x/x_{rid}.safetensors",
                                      "final": True, "state": str(state)})
    row = keep_final(store, rid, store.read(rid, "checkpoints")[1], None)
    assert not final.exists() and not state.exists()  # the new final can take the name
    assert Path(row["path"]) == loras / "checkpoints" / rid / f"x_{rid}_step000500.safetensors"
    assert Path(row["path"]).is_file() and Path(row["state"]).is_file()
    assert row["lora_name"] == f"checkpoints/{rid}/x_{rid}_step000500.safetensors" and not row["final"]
    assert store.read(rid, "checkpoints")[1] == row and store.read(rid, "checkpoints")[0]["step"] == 250


def test_lora_files_save_in_their_own_dtype_and_the_state_keeps_full_weights(tmp_path, monkeypatch):
    import sys
    import types

    import safetensors.torch
    import torch

    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path / "output")
    fp.get_user_directory = lambda: str(tmp_path / "user")
    fp.get_folder_paths = lambda _kind: []
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    from zentrainer.harness.trainer import Trainer
    from zentrainer.spec import TrainSpec

    w = torch.nn.Parameter(torch.tensor([1.0 + 2**-12]))  # lost in bf16
    t = object.__new__(Trainer)
    t.spec, t.run_id, t.step, t.store, t.ema, t.train_seconds = TrainSpec(name="x"), "r1", 10, None, None, 0.0
    t.network = types.SimpleNamespace(dtype=torch.float32, params={"a.lokr_w2": w}, state_dict=lambda: {"a.lokr_w2": w.detach()})
    t.optimizer = torch.optim.SGD([w], lr=0.1)
    t.scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda _s: 1.0)
    t.generator, t.batches = torch.Generator(), types.SimpleNamespace(epoch=0)
    t.objective, t.emit = types.SimpleNamespace(balance=None), lambda *_a: None
    path = t._save("step000010", False, fp, safetensors.torch)
    assert safetensors.torch.load_file(path)["a.lokr_w2"].dtype == torch.bfloat16
    state = torch.load(next((tmp_path / "user").rglob("*.pt")), weights_only=False)
    assert state["live"]["a.lokr_w2"].item() == 1.0 + 2**-12 and "ema" not in state


def test_skipped_blocks_get_no_adapters():
    from zentrainer.harness.network import skipped

    assert skipped("diffusion_model.layers.22.proj.q", "22")
    assert not skipped("diffusion_model.layers.21.proj.q", "22")
    assert skipped("diffusion_model.layers.31.proj.ffn_down", "22, 30-31")
    assert not skipped("diffusion_model.final_linear_video", "22") and not skipped("diffusion_model.layers.3.x", "")
    TrainSpec.from_dict({**_spec(), "network": {"skip_blocks": "22, 30-31"}})
    with pytest.raises(SpecError, match="skip_blocks"):
        TrainSpec.from_dict({**_spec(), "network": {"skip_blocks": "layers.22"}})


def test_compile_is_auto_unless_set(monkeypatch):
    from zentrainer.harness.kernels import compile as comp

    assert comp.wanted(True, 10) == (True, "") and comp.wanted(False, 5000) == (False, "")
    monkeypatch.setattr(comp.os, "name", "posix")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "triton" else None)
    assert comp.wanted(None, comp.AUTO_STEPS)[0]
    assert not comp.wanted(None, comp.AUTO_STEPS - 1)[0]  # too short to repay the compile
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert not comp.wanted(None, 5000)[0]  # no Triton


def test_the_fused_qk_rope_is_auto_by_default():
    assert TrainSpec.from_dict(_spec()).memory.fused_qk_rope is None  # on while the blocks run uncompiled
    assert TrainSpec.from_dict({**_spec(), "memory": {"fused_qk_rope": False}}).memory.fused_qk_rope is False
