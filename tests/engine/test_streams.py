"""Multi-stream latents (MiniMax H3: video + audio) packed the way ComfyUI's sampler packs them."""

import math
import sys
import types

import pytest
import torch

from zentrainer.harness.data.streams import Streams, first_stream


class _Nested:
    is_nested = True

    def __init__(self, ts):
        self.ts = ts

    def unbind(self):
        return self.ts


class _AV:  # a two-stream latent format: the picture plus an empty (zero) second stream
    latent_channels = 4
    latent_dimensions = 3

    def fix_empty_latent(self, latent):
        return _Nested((latent, latent.new_zeros((latent.shape[0], 3, 2, 5))))


class _Plain:
    latent_channels = 4
    latent_dimensions = 2

    def fix_empty_latent(self, latent):
        return latent


def _comfy_utils(monkeypatch):
    def pack_latents(latents):
        shapes = [t.shape for t in latents]
        return torch.cat([t.reshape(t.shape[0], 1, -1) for t in latents], dim=-1), shapes

    mod = types.ModuleType("comfy.utils")
    mod.pack_latents = pack_latents
    pkg = types.ModuleType("comfy")
    pkg.utils = mod
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.utils", mod)


def test_single_stream_models_are_left_alone():
    s = Streams(types.SimpleNamespace(latent_format=_Plain()))
    assert not s.multi
    assert s.pack(torch.zeros(1, 4, 8, 8)) is None
    assert s.packed_shape((4, 8, 8)) == (1, 4, 8, 8)


def test_pack_weights_only_the_picture(monkeypatch):
    _comfy_utils(monkeypatch)
    s = Streams(types.SimpleNamespace(latent_format=_AV()))
    assert s.multi
    latent = torch.randn(1, 4, 1, 6, 8)
    p = s.pack(latent)
    n = math.prod(latent.shape[1:])
    assert p.x0.shape == (1, 1, n + 3 * 2 * 5)
    assert torch.equal(p.x0[0, 0, :n], latent.flatten())
    assert p.weight[..., :n].eq(1).all() and p.weight[..., n:].eq(0).all()
    assert s.packed_shape((4, 1, 6, 8)) == tuple(p.x0.shape)


def test_pack_spreads_a_mask_over_the_picture(monkeypatch):
    _comfy_utils(monkeypatch)
    s = Streams(types.SimpleNamespace(latent_format=_AV()))
    mask = torch.zeros(1, 6, 8)
    mask[:, :, :4] = 1  # left half
    p = s.pack(torch.randn(1, 4, 1, 6, 8), mask)
    w = p.weight[0, 0, : 4 * 6 * 8].reshape(4, 1, 6, 8)
    assert w[..., :4].eq(1).all() and w[..., 4:].eq(0).all()


def test_first_stream():
    t = torch.zeros(2)
    assert first_stream(t) is t
    assert first_stream(_Nested((t, torch.ones(1)))) is t


def test_h3_targets():
    from zentrainer.harness.network import matches

    for name in ("blocks.3.attn.qkv_proj", "blocks.3.attn.out_proj", "blocks.3.mlp.fc1", "blocks.49.mlp.fc2"):
        assert matches(name, "attn_mlp"), name
    for name in ("token_refiner.blocks.0.attn.qkv_proj", "condition_proj", "blocks.3.adaln_proj.linear",
                 "blocks.3.attn.to_gate_compress", "video_patch_proj"):
        assert not matches(name, "attn_mlp"), name
    assert matches("blocks.3.attn.qkv_proj", "attn") and not matches("blocks.3.mlp.fc1", "attn")


def test_rope_split_half_rotates_only_the_first_dims():
    from zentrainer.harness.compat.minimax_h3 import _rope_split_half

    x = torch.randn(1, 5, 2, 8)
    pe = torch.zeros(1, 5, 1, 2, 2, 2)
    pe[..., 0, 0] = 1  # identity rotation: cos 1, sin 0
    pe[..., 1, 1] = 1
    out = _rope_split_half(x, pe, 4)
    assert torch.allclose(out, x)


class _Frozen(torch.nn.Module):  # a stand-in adapter: adds a constant to the weight / the output
    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, w):
        return w + self.k

    def h(self, x, base_out):
        return torch.full_like(base_out, self.k)

    def g(self, y):
        return y


def test_training_adapter_is_switchable_and_not_saved():
    from zentrainer.harness.network import _Assisted, _Switch

    sw = _Switch()
    a = _Assisted(_Frozen(1.0), _Frozen(10.0), sw)
    w = torch.zeros(2)
    assert a(w).tolist() == [11.0, 11.0]  # both while training
    sw.on = False
    assert a(w).tolist() == [1.0, 1.0]  # samples: the trained adapter alone
    sw.on = True
    a.multiplier = 0.5  # bypass sets layer info on what it holds; the children need it too
    assert a.h(w, w).tolist() == [11.0, 11.0] and a.trained.multiplier == 0.5
    only = _Assisted(None, _Frozen(10.0), sw)  # a layer only the training adapter touches
    sw.on = False
    assert only(w).tolist() == [0.0, 0.0] and only.h(w, w) == 0


def test_base_lora_is_on_in_training_and_samples():
    from zentrainer.harness.network import _Assisted, _Switch
    from zentrainer.harness.callbacks import write_metadata

    sw = _Switch()
    a = _Assisted(_Frozen(1.0), _Frozen(10.0), sw, base=_Frozen(100.0), base_strength=0.5)
    w = torch.zeros(2)
    assert a(w).tolist() == [61.0, 61.0]  # training: base at 0.5, training adapter, trained
    assert a.h(w, w).tolist() == [61.0, 61.0]  # bypass: the same sum
    sw.on = False
    assert a(w).tolist() == [51.0, 51.0]  # samples: the base stays, the training adapter goes
    assert a.h(w, w).tolist() == [51.0, 51.0]
    only = _Assisted(None, None, sw, base=_Frozen(100.0))  # a layer only the base touches
    assert only(w).tolist() == [100.0, 100.0]
    meta = write_metadata({"network": {"base_lora": "base_loras/destripe.safetensors", "base_strength": 0.5}}, 10)
    assert meta["zentrainer.base_lora"] == "base_loras/destripe.safetensors" and meta["zentrainer.base_strength"] == "0.5"
    assert "zentrainer.base_lora" not in write_metadata({"network": {}}, 10)


def test_frozen_adapters_on_slices_of_a_fused_weight():
    from zentrainer.harness.network import _Sliced

    s = _Sliced()  # q and v of a fused qkv (rows 0-1, 4-5 of 6)
    s.add(0, 2, _Frozen(1.0))
    s.add(4, 2, _Frozen(5.0))
    w = torch.zeros(6, 3)
    assert s(w)[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 5.0, 5.0]  # k untouched
    assert w.abs().sum() == 0  # the base weight isn't changed in place
    out = torch.zeros(2, 6)
    s.multiplier = 0.5
    assert s.h(None, out)[0].tolist() == [1.0, 1.0, 0.0, 0.0, 5.0, 5.0]
    assert s.parts[1][2].multiplier == 0.5  # bypass's layer info reaches the parts


def test_pack_trains_a_real_soundtrack(monkeypatch):
    _comfy_utils(monkeypatch)
    s = Streams(types.SimpleNamespace(latent_format=_AV()))
    lat = torch.ones(2, 4, 2, 3, 3)
    sound = torch.full((3, 2, 7), 0.5)  # longer than the stream (5): cropped
    p = s.pack(lat, None, [sound, None])
    n, m = 4 * 2 * 3 * 3, 3 * 2 * 5
    assert p.x0.shape == (2, 1, n + m)
    assert torch.all(p.x0[0, 0, n:] == 0.5) and torch.all(p.x0[1, 0, n:] == 0)
    assert torch.all(p.weight[0, 0, n:] == 1) and torch.all(p.weight[1, 0, n:] == 0)  # only heard audio counts
    assert torch.all(p.weight[:, 0, :n] == 1)


def test_sample_prompts_fill_from_the_run():
    from zentrainer.spec import SampleSpec, sample_prompts

    s = SampleSpec(prompts=["a", {"prompt": "b", "width": 480, "height": 832, "frames": 73}], width=1024, height=1024, seed=7)
    a, b = sample_prompts(s)
    assert (a.prompt, a.width, a.height, a.frames, a.seed) == ("a", 1024, 1024, 1, 7)
    assert (b.prompt, b.width, b.height, b.frames, b.seed) == ("b", 480, 832, 73, 8)


def test_lora_plus_groups_keep_their_ratio_through_set_lr():
    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    down, up = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    spec = OptimSpec(learning_rate=1e-4, lora_plus=4)
    opt, sched = optim.build([{"params": [down], "lr": 1e-4}, {"params": [up], "lr": 4e-4, "lr_mult": 4.0}], spec)
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 4e-4]
    optim.set_lr(opt, sched, 2e-4)
    assert [round(g["lr"], 12) for g in opt.param_groups] == [2e-4, 8e-4]


def test_network_disabled_zeroes_then_restores():
    import types

    from zentrainer.harness.network import Network

    p = torch.nn.Parameter(torch.ones(3))
    net = types.SimpleNamespace(params={"a.lora_up.weight": p})
    with Network.disabled(net):
        assert torch.all(p == 0)
    assert torch.all(p == 1)


def test_released_text_encoder_loads_again():
    from zentrainer.harness.models import Models

    loads = []
    m = Models(patcher=None, _clip=None, vae=None, _clip_loader=lambda: loads.append(1) or "clip")
    assert m.clip == "clip"
    m.release_clip()
    assert m._clip is None
    assert m.clip == "clip" and len(loads) == 2  # a later prompt can still be encoded


def test_manual_tries_batch_into_one_session_and_pick_up_late_ones():
    from zentrainer.harness.control import Control
    from zentrainer.harness.trainer import Trainer

    calls = []

    class Samples:
        def render(self, t):
            calls.append("run samples")

        def render_manual(self, t, tries, more):
            batch = list(tries)
            t.control.send("sample", prompt="late")  # queued while the first batch renders
            nxt = more()
            calls.append(("session", [a["prompt"] for a in batch], [a["prompt"] for a in nxt]))

    t = object.__new__(Trainer)
    t.control = Control()
    t._samples = Samples()
    t.control.send("sample", prompt="a")
    t.control.send("sample", prompt="b")
    t._apply(t.control.drain())
    assert calls == [("session", ["a", "b"], ["late"])]


def test_block_range():
    from zentrainer.harness.network import in_blocks

    assert in_blocks("diffusion_model.blocks.25.attn.qkv", "20-49")
    assert not in_blocks("diffusion_model.blocks.3.attn.qkv", "20-49")
    assert in_blocks("diffusion_model.double_blocks.7.img_mlp.0", "5-")
    assert in_blocks("diffusion_model.layers.2.mlp", "-2") and not in_blocks("diffusion_model.layers.3.mlp", "-2")
    assert not in_blocks("diffusion_model.final_layer.linear", "20-49")
    assert in_blocks("diffusion_model.final_layer.linear", "")


def test_late_blocks_train_without_input_grad_and_early_blocks_skip_backward():
    import torch.utils.checkpoint as cp

    early, late = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    early.requires_grad_(False)
    late.requires_grad_(False)
    lora = torch.nn.Parameter(torch.zeros(4))  # stands in for an adapter on the late block
    hits = []
    early.register_full_backward_hook(lambda *a: hits.append("early"))
    x = torch.randn(2, 4)  # no requires_grad: the input carries none
    h = cp.checkpoint(early, x, use_reentrant=False)
    y = cp.checkpoint(lambda t: late(t) + lora, h, use_reentrant=False)
    y.sum().backward()
    assert lora.grad is not None and torch.all(lora.grad == 2)
    assert hits == []  # nothing flowed back into the untrained early block


def test_locked_changes_at_resume():
    from zentrainer.spec import TrainSpec, locked_changes

    old = TrainSpec().to_dict()
    new = TrainSpec().to_dict()
    new["memory"]["bypass_mode"] = True
    new["optim"]["learning_rate"] = 5e-5
    new["samples"]["steps"] = 6
    assert locked_changes(old, new) == []
    new["network"]["rank"] = 32
    new["optim"]["optimizer"] = "Prodigy"
    assert locked_changes(old, new) == ["network.rank", "optim.optimizer"]


def test_preview_adapter_only_while_sampling():
    from zentrainer.harness.network import _Assisted, _Switch

    sw = _Switch()
    assist = lambda w: w + 1  # training adapter: +1
    turbo = lambda w: w + 10  # preview adapter: +10, used at 0.5
    a = _Assisted(None, assist, sw, turbo, 0.5)
    w = torch.zeros(2)
    assert torch.all(a(w) == 1)  # training: the training adapter only
    sw.on = False
    assert torch.all(a(w) == 5)  # sampling: half the turbo, no training adapter


def test_tread_routes_video_tokens_around_the_middle_blocks():
    import types as _t

    from zentrainer.harness.memory.tread import patches

    n_blocks, N, C = 10, 40, 3
    layout = _t.SimpleNamespace(segments=[(0, 6, "text"), (6, 34, "video"), (34, 40, "audio")])
    seen = []

    def block(i):
        def run(args):  # each block adds its index and records the length it saw
            seen.append((i, args["img"].shape[0], len(args["mod_segments"]), args["rope_freqs"].shape[1]))
            return {"img": args["img"] + i}
        return run

    rep = patches(n_blocks, 0.5, torch.Generator().manual_seed(0), start=2, end=7)
    h = torch.zeros(N, C)
    segs = [(0, 6, 1), (6, 34, 0), (34, 40, 2)]
    rope = torch.zeros(1, N, 1, 2, 2, 2)
    for i in range(n_blocks):
        args = {"img": h, "mod_segments": segs, "rope_freqs": rope, "layout": layout, "t_emb": None}
        h = rep[("double_block", i)](args, {"original_block": block(i)})["img"] if ("double_block", i) in rep else block(i)(args)["img"]
    lengths = {i: n for i, n, _, _ in seen}
    assert lengths[0] == lengths[1] == N and lengths[7] == lengths[9] == N  # whole outside the window
    assert all(lengths[i] == N - 14 for i in range(2, 7))  # half the 28 video tokens routed
    assert all(r == n for _, n, _, r in seen)  # rope follows the tokens
    text_total = sum(range(n_blocks))
    assert torch.all(h[:6] == text_total) and torch.all(h[34:] == text_total)  # text/audio saw every block
    routed = h[6:34, 0]
    assert ((routed == text_total) | (routed == text_total - sum(range(2, 7)))).all()  # video: all, or skipped 2-6
    assert int((routed != text_total).sum()) == 14


def test_ema_average_applied_and_restored():
    from zentrainer.harness.ema import EMA

    p = torch.nn.Parameter(torch.zeros(3))
    ema = EMA({"a": p}, 0.9)
    for step in range(200):
        with torch.no_grad():
            p.fill_(1.0)
        ema.update(step)
    assert torch.allclose(ema.shadow["a"], torch.ones(3), atol=1e-3)
    with torch.no_grad():
        p.fill_(5.0)
    with ema.applied() as live:
        assert torch.allclose(p, torch.ones(3), atol=1e-3) and torch.all(live["a"] == 5.0)
    assert torch.all(p == 5.0)  # the live weights are back
    st = ema.state_dict()
    other = EMA({"a": torch.nn.Parameter(torch.zeros(3))}, 0.9)
    other.load_state_dict(st)
    assert torch.allclose(other.shadow["a"], ema.shadow["a"])


def test_automagic3_trains_and_keeps_its_own_lr():
    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
    target = torch.randn(8, 8)
    opt, sched = optim.build([w], OptimSpec(optimizer="Automagic3", learning_rate=1e-3, warmup_steps=10))
    first = None
    for _ in range(60):
        loss = ((w - target) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    assert loss.item() < first
    assert optim.current_lr(opt) != 1e-3  # it steered the LR itself


def test_adapter_shape_check_skips_other_builds_layers():
    from zentrainer.harness.network import _fits

    m = torch.nn.Module()
    m.diffusion_model = torch.nn.Module()
    m.diffusion_model.proj = torch.nn.Linear(8, 96)  # a pruned build: 8 inputs, where the LoRA expects 2688
    m.diffusion_model.conv = torch.nn.Conv2d(4, 6, 3)
    assert _fits(m, "diffusion_model.proj.weight", (96, 8))
    assert not _fits(m, "diffusion_model.proj.weight", (96, 2688))
    assert _fits(m, "diffusion_model.conv.weight", (6, 4 * 9))
    assert not _fits(m, "diffusion_model.missing.weight", (6, 4))


def test_edit_pairs_scan_and_reference_conditioning(tmp_path):
    from PIL import Image

    from zentrainer.harness.data import scan
    from zentrainer.harness.data.edit import with_refs

    for name in ("a.png", "a_control.png", "a_control2.jpg", "b.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    (tmp_path / "a.txt").write_text("make it red")
    items = {it.path.name: it for it in scan(tmp_path, warn=False)}
    assert set(items) == {"a.png", "b.png"}  # the inputs aren't trained as images of their own
    assert [c.name for c in items["a.png"].controls] == ["a_control.png", "a_control2.jpg"]
    assert items["b.png"].controls == []

    cond = [[torch.zeros(1, 4, 8), {"pooled_output": None}]]
    ref = torch.ones(16, 2, 2)
    out = with_refs(cond, [ref])
    assert out[0][1]["reference_latents"][0].shape == (1, 16, 2, 2)
    assert "reference_latents" not in cond[0][1]  # the cached caption conditioning is left alone


def test_grounded_template_keeps_the_encoders_own_system_prompt():
    from types import SimpleNamespace

    from zentrainer.harness.data.edit import VISION, grounded_template, instruction

    krea = "<|im_start|>system\nDescribe the image<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    clip = SimpleNamespace(tokenizer=SimpleNamespace(llama_template=krea))
    t = grounded_template(clip, 2)
    assert t.startswith("<|im_start|>system\nDescribe the image")  # not qwen3vl's default picture template
    assert t.count(VISION) == 2 and t.index(VISION) == t.index("user\n") + len("user\n")
    assert t.format("make it red").count("make it red") == 1
    assert grounded_template(SimpleNamespace(tokenizer=SimpleNamespace()), 1) is None  # a T5/CLIP encoder
    assert instruction("add a hat", "Make the following edit: {}") == "Make the following edit: add a hat"
    assert instruction("add a hat", "") == "add a hat"


def test_w8a8_kernels_match_torch():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (Triton kernels)")
    from zentrainer.harness.kernels.w8a8_backward import TRITON_AVAILABLE, int8_matmul, quantize_rows

    if not TRITON_AVAILABLE:
        pytest.skip("needs Triton")
    torch.manual_seed(0)
    g = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    s = torch.rand(512, device="cuda") + 0.5
    q = torch.randint(-127, 128, (512, 256), device="cuda", dtype=torch.int8)
    gq, step = quantize_rows(g, s, 7)
    x = g.float() * s
    assert torch.allclose(step, x.abs().amax(1) / 127, rtol=1e-5)
    assert ((gq.float() * step[:, None] - x).abs() <= step[:, None] + 1e-6).all()  # within one int8 step
    ref = (gq.float() @ q.float()) * step[:, None]
    out = int8_matmul(gq, q, step).float()
    assert ((out - ref).norm() / ref.norm()) < 5e-3


def test_compiled_blocks_are_undone():
    from zentrainer.harness.kernels.compile import compiled_blocks

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))

    net = Net()
    with compiled_blocks(net, True) as n:
        assert n == 3 and all("forward" in b.__dict__ for b in net.blocks)
    assert not any("forward" in b.__dict__ for b in net.blocks)  # the cached model is left as it was


def test_failed_run_frames_let_go_of_their_tensors():
    import gc
    import weakref

    from zentrainer.harness.trainer import _release_frames

    ref = None

    def step():
        nonlocal ref
        activations = torch.zeros(1000)
        ref = weakref.ref(activations)
        raise torch.OutOfMemoryError("CUDA out of memory")

    try:
        try:
            step()
        except torch.OutOfMemoryError as inner:
            raise RuntimeError("Out of VRAM at step 1") from inner
    except RuntimeError as e:
        kept = e  # ComfyUI keeps the job's exception
        _release_frames(kept)
    gc.collect()
    assert ref() is None and "step 1" in str(kept)


def test_sharpest_frame_picks_the_detailed_one(tmp_path):
    import numpy as np

    from zentrainer.harness.data import video

    frames = []
    rng = np.random.default_rng(0)
    for i in range(12):  # flat grey frames, one noisy (sharp) frame at index 7
        f = np.full((64, 64, 3), 0.5, dtype=np.float32)
        if i == 7:
            f = rng.random((64, 64, 3)).astype(np.float32)
        frames.append(f)
    path = tmp_path / "clip.mp4"
    video.write(path, torch.from_numpy(np.stack(frames)), 12)
    p = video.probe(path)
    assert abs(video.sharpest_frame(path, p, look=12) - 7) <= 0


def test_image_flags_rules():
    from zentrainer.datasets.stats import image_flags

    # (pairs of near-duplicates are tested with real pictures in test_analysis.py)
    ok = {"bright": 0.5, "contrast": 0.2, "sharp": 10.0, "sat": 0.3, "border": 0.0}
    looks = {
        "a": ok,
        "b": {**ok, "bright": 0.08},              # dark
        "c": {**ok, "sharp": 1.0},                # blurry against the set's median
        "d": {**ok, "border": 0.2, "sat": 0.01},  # bars, and no colour in a colourful set
    }
    flags = image_flags(looks, {"a": 1024, "b": 1024, "c": 600, "d": 1024})
    assert flags["b"] == ["dark"] and "a" not in flags
    assert "blurry" in flags["c"] and "small" in flags["c"]
    assert "bars" in flags["d"] and "greyscale" in flags["d"]


def test_face_likeness_leaves_each_face_out():
    import numpy as np

    from zentrainer.datasets.faces import _leave_one_out

    rng = np.random.default_rng(0)
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    strangers = {str(i): unit(rng.normal(size=512)).astype(np.float32) for i in range(4)}
    assert max(_leave_one_out(strangers).values()) < 0.2  # four different people don't look alike
    me = unit(rng.normal(size=512))
    same = {str(i): unit(me + 0.6 * unit(rng.normal(size=512))).astype(np.float32) for i in range(4)}
    assert min(_leave_one_out(same).values()) > 0.5  # one person's photos do


def test_masked_out_faces_dont_count():
    import numpy as np

    from zentrainer.datasets.faces import _summarize

    me = {"box": [0.1, 0.1, 0.4, 0.5], "share": 0.4, "emb": [1.0, 0.0]}
    bystander = {"box": [0.6, 0.2, 0.8, 0.4], "share": 0.2, "emb": [0.0, 1.0]}
    frames = [[me, bystander]]
    assert _summarize(frames, None)["faces"] == 2  # no mask: both count
    mask = np.ones((256, 256), np.float32)
    mask[:, 140:] = 0  # the right side (the bystander) masked out
    r = _summarize(frames, mask)
    assert (r["faces"], r["masked"], r["emb"]) == (1, 1, [1.0, 0.0])
    mask[:] = 0
    mask[:, 140:] = 1  # now the main face is masked out: the bystander is the trained face
    r = _summarize(frames, mask)
    assert (r["faces"], r["masked"], r["emb"], r["share"]) == (1, 1, [0.0, 1.0], 0.2)


def test_stopping_right_after_a_save_writes_no_copy():
    from types import SimpleNamespace

    from zentrainer.harness.callbacks import Checkpoints

    saved = []
    t = SimpleNamespace(step=0, spec=SimpleNamespace(optim=SimpleNamespace(steps=1000)),
                        save=lambda tag, final=False: saved.append(tag))
    cb = Checkpoints(250)
    t.step = 250
    cb.on_step(t)
    cb.on_end(t, "stopped")  # stopped at the step just saved: that file resumes
    assert saved == ["step000250"]
    t.step = 260
    cb.on_end(t, "error")  # past it: the newer weights are kept
    assert saved == ["step000250", "error-step000260"]
    t.step = 1000
    cb.on_end(t, "done")
    assert saved[-1] == "final"


def _automagic3_run(cls, dtype, steps=30, **kw):
    torch.manual_seed(0)
    # LoRA-like: repeated shapes, one odd element count (21, not a multiple of 8), a 1-D tensor
    shapes = [(4, 16), (4, 16), (16, 4), (16, 4), (3, 7), (5,)]
    ps = [torch.nn.Parameter((torch.randn(s) * 0.1).to(dtype)) for s in shapes]
    targets = [torch.randn(s) for s in shapes]
    opt = cls(ps, lr=1e-3, weight_decay=0.01, fused=False, **kw)
    torch.manual_seed(1)
    for _ in range(steps):
        loss = sum(((p.float() - t) ** 2).mean() for p, t in zip(ps, targets))
        loss.backward()
        opt.step()
    return ps, opt


def test_automagic3_batched_matches_the_vendored_step():
    from zentrainer.harness.optim.automagic3 import Automagic3
    from zentrainer.harness.optim.automagic3_batched import Automagic3Batched

    # 12 steps: the sign window (8) fills and the lr vote runs; far longer and the toy's lr grows
    # until reduction-order rounding (~1e-7) compounds visibly
    a, oa = _automagic3_run(Automagic3, torch.float32, steps=12)
    b, ob = _automagic3_run(Automagic3Batched, torch.float32, steps=12)
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, rtol=1e-5, atol=1e-7)
    assert oa.get_learning_rates() == pytest.approx(ob.get_learning_rates(), rel=1e-5)
    assert oa.get_learning_rates()[0] != 1e-3  # the window filled and the lr moved: the vote is compared too
    for pa, pb in zip(a, b):
        sa, sb = oa.state[pa], ob.state[pb]
        assert sa.keys() == sb.keys()  # same state layout: states save and resume either way
        assert sa["hist_idx"] == sb["hist_idx"] and sa["step"] == sb["step"]
        assert torch.equal(sa["sign_history"], sb["sign_history"])


def test_automagic3_batched_bf16_trains():
    from zentrainer.harness.optim.automagic3_batched import Automagic3Batched

    ps, _ = _automagic3_run(Automagic3Batched, torch.bfloat16, steps=60)
    assert all(p.dtype == torch.bfloat16 and torch.isfinite(p).all() for p in ps)


def test_automagic4_batched_matches_the_vendored_step():
    from zentrainer.harness.optim.automagic4 import Automagic4
    from zentrainer.harness.optim.automagic4_batched import Automagic4Batched

    for steps in (1, 12):  # the measuring-only first step, then past the full sign window
        a, oa = _automagic3_run(Automagic4, torch.float32, steps=steps)
        b, ob = _automagic3_run(Automagic4Batched, torch.float32, steps=steps)
        for pa, pb in zip(a, b):
            assert torch.allclose(pa, pb, rtol=1e-5, atol=1e-7)
            sa, sb = oa.state[pa], ob.state[pb]
            assert sa.keys() == sb.keys() and sa["step"] == sb["step"]  # states save and resume either way
            assert torch.allclose(sa["sens_ref"], sb["sens_ref"], rtol=1e-5)
            assert torch.equal(sa["sign_history"], sb["sign_history"])
        assert oa.get_learning_rates() == pytest.approx(ob.get_learning_rates(), rel=1e-5)
    assert oa.get_learning_rates()[0] != 1e-3  # the vote ran: it's compared too


def test_automagic4_bf16_trains():
    from zentrainer.harness.optim.automagic4_batched import Automagic4Batched

    ps, _ = _automagic3_run(Automagic4Batched, torch.bfloat16, steps=60)
    assert all(p.dtype == torch.bfloat16 and torch.isfinite(p).all() for p in ps)


def test_automagic4_first_step_only_measures():
    from zentrainer.harness.optim.automagic4_batched import Automagic4Batched

    torch.manual_seed(0)
    w, b = torch.nn.Parameter(torch.randn(8, 8)), torch.nn.Parameter(torch.randn(8))
    before = (w.detach().clone(), b.detach().clone())
    opt = Automagic4Batched([w, b], lr=1e-3, fused=False)
    w.grad, b.grad = torch.randn(8, 8), torch.randn(8)
    opt.step()
    assert torch.equal(w, before[0]) and torch.equal(b, before[1])  # nothing moved
    assert "sens_ref" in opt.state[w]  # but it has its reference for the next
    w.grad, b.grad = torch.randn(8, 8), torch.randn(8)
    opt.step()
    assert not torch.equal(w, before[0])


def test_automagic4_holds_its_rate_where_v3_sinks():
    from zentrainer.harness.optim.automagic3_batched import Automagic3Batched
    from zentrainer.harness.optim.automagic4_batched import Automagic4Batched

    # a weight already at its minimum, its gradient only the pull back plus noise: v3 reads the
    # pull's flips as overshoot and walks its LR down; v4 settles where steps are step_target
    def settle(cls):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.zeros(32, 32))
        opt = cls([w], lr=1e-3, fused=False)
        for _ in range(400):
            w.grad = w.detach() + 0.1 * torch.randn_like(w)
            opt.step()
        return opt.get_learning_rates()[0]

    assert settle(Automagic3Batched) < 1e-3
    assert settle(Automagic4Batched) > 2e-3


def test_automagic4_resumes_where_it_left_off():
    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    def make():
        torch.manual_seed(0)
        ps = [torch.nn.Parameter(torch.randn(16, 8) * 0.1), torch.nn.Parameter(torch.randn(8, 16) * 0.1)]
        return ps, optim.build(ps, OptimSpec(optimizer="Automagic4", learning_rate=1e-3))[0]

    def steps(ps, opt, seed, n):
        g = torch.Generator().manual_seed(seed)
        for _ in range(n):
            for p in ps:
                p.grad = torch.randn(p.shape, generator=g)
            opt.step()

    a, oa = make()
    steps(a, oa, 1, 10)
    steps(a, oa, 2, 10)
    b, ob = make()
    steps(b, ob, 1, 10)
    saved = ob.state_dict()
    c, oc = make()
    for pc, pb in zip(c, b):
        pc.data.copy_(pb.data)
    oc.load_state_dict(saved)
    steps(c, oc, 2, 10)
    for pa, pc in zip(a, c):
        assert torch.allclose(pa, pc, rtol=1e-5, atol=1e-7)
    assert optim.current_lr(oa) == pytest.approx(optim.current_lr(oc), rel=1e-5)


def test_old_optimizer_states_resume_without_the_alphas():
    from zentrainer.harness.optim import remap_state

    # saved while alphas trained: each adapter's alpha, up, down, numbered across the run's groups
    names = ["a.alpha", "a.lora_up.weight", "a.lora_down.weight", "b.alpha", "b.lora_up.weight", "b.lora_down.weight"]
    saved = {"state": {i: {"tag": n} for i, n in enumerate(names)},
             "param_groups": [{"lr": 1e-4, "params": list(range(6))}]}
    new = [[n for n in names if not n.endswith(".alpha")]]
    out = remap_state(saved, [names], new)
    assert out["param_groups"][0]["params"] == [0, 1, 2, 3] and out["param_groups"][0]["lr"] == 1e-4
    assert [out["state"][i]["tag"] for i in range(4)] == new[0]

    # LoRA+: (rest, up) groups, numbered across both
    old = [[n for n in names if "lora_up" not in n], [n for n in names if "lora_up" in n]]
    saved2 = {"state": {i: {"tag": n} for i, n in enumerate(old[0] + old[1])},
              "param_groups": [{"lr": 1e-4, "params": [0, 1, 2, 3]}, {"lr": 1.6e-3, "params": [4, 5]}]}
    new2 = [[n for n in old[0] if not n.endswith(".alpha")], old[1]]
    out2 = remap_state(saved2, old, new2)
    assert [g["params"] for g in out2["param_groups"]] == [[0, 1], [2, 3]]
    assert [out2["state"][i]["tag"] for i in range(4)] == new2[0] + new2[1]
    assert out2["param_groups"][1]["lr"] == 1.6e-3


def test_automagic3_lr_floor_holds_its_rate():
    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    def run(floor):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
        opt, _ = optim.build([w], OptimSpec(optimizer="Automagic3", learning_rate=1e-3, lr_floor=floor))
        for i in range(40):  # every update flips sign: the vote that walks the rate down
            w.grad = torch.ones_like(w) * (1 if i % 2 else -1)
            opt.step()
        return optim.current_lr(opt)

    assert run(0.0) < 0.5e-3  # without a floor it falls well below half
    assert run(0.5) >= 0.5e-3 * 0.999  # with one it holds there


def test_a_fit_that_slows_the_steps_is_undone(monkeypatch):
    from zentrainer.harness import trainer as tr

    class Patcher:
        def __init__(self):
            self.loaded = 4 << 30

        def loaded_size(self):
            return self.loaded

    t = tr.Trainer.__new__(tr.Trainer)
    t.models = types.SimpleNamespace(patcher=Patcher())
    fits = []
    t._fit_weights = lambda: (fits.append(t._resident), setattr(t.models.patcher, "loaded", t._resident))
    t._recompile_if_moved = lambda: None
    stats = {"num_alloc_retries": 0}
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: dict(stats))

    short, long = ((1, 2560, 64), "a"), ((1, 5376, 64), "b")  # two clip shapes: H3 took ~1.5 and ~3 s a step on them
    before = [(1, 1.5, short), (2, 3.0, long), (3, 1.5, short), (4, 3.0, long)]

    # same-shaped steps slower after: back to where it was, and capped there
    t._trial = {"at": 4, "prev": 4 << 30, "before": before, "retries": 0}
    t.models.patcher.loaded = 9 << 30
    t._warm = 11
    t._times = before + [(5, 21.0, short), (8, 2.0, short), (9, 2.0, short), (10, 3.9, long), (11, 2.0, short)]
    t._judge()
    assert t._resident == t._ceiling == 4 << 30 and fits == [4 << 30] and t._trial is None

    # as fast, but the allocator had to retry: undone too
    t._ceiling = 1 << 62
    t._trial = {"at": 4, "prev": 4 << 30, "before": before, "retries": 0}
    stats["num_alloc_retries"] = 3
    t._times = before + [(8, 1.4, short), (9, 1.4, short), (10, 1.4, short), (11, 1.4, short)]
    t._judge()
    assert t._ceiling == 4 << 30

    # faster and no retries: kept
    del t._ceiling
    fits.clear()
    t._trial = {"at": 4, "prev": 4 << 30, "before": before, "retries": 3}
    t._times = before + [(8, 1.4, short), (9, 2.9, long), (10, 1.4, short), (11, 2.9, long)]
    t._judge()
    assert fits == [] and not hasattr(t, "_ceiling")

    # only long clips since the fit, short ones before it: raw times would read twice as slow, so
    # it waits for a same-shaped step, then keeps the fit when none comes
    t._trial = {"at": 4, "prev": 4 << 30, "before": [(1, 1.5, short), (2, 1.5, short)], "retries": 3}
    t._times = t._trial["before"] + [(8, 3.0, long), (9, 3.0, long), (10, 3.0, long), (11, 3.0, long)]
    t._judge()
    assert t._trial is not None and fits == []  # not judged yet
    t._warm = 4 + tr.TRIAL_LONGEST
    t._judge()
    assert t._trial is None and fits == [] and not hasattr(t, "_ceiling")


def test_trial_ratio_compares_same_shapes_only():
    from zentrainer.harness.trainer import trial_ratio

    before = [(1, 1.0, "a"), (2, 2.0, "b"), (3, 1.2, "a")]
    assert trial_ratio(before, [(5, 2.2, "b"), (6, 2.2, "b")]) == pytest.approx(1.1)
    assert trial_ratio(before, [(5, 9.0, "c")]) is None
    assert trial_ratio(before, []) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_offloaded_weights_get_pinned_copies(monkeypatch):
    from zentrainer.harness.memory import offload

    monkeypatch.setattr(offload, "PIN_RAM_SPARE", 0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    q = torch.randint(-127, 127, (8, 8), dtype=torch.int8)
    m[1].weight = torch.nn.Parameter(torch.zeros(8, 8), requires_grad=False)
    m[1].weight._qdata = q  # an int8 weight: its bytes live in _qdata
    before = m[0].weight.detach().clone()
    n, size = offload.pin_offloaded(m)
    assert n == 4 and m[0].weight.is_pinned() and m[1].weight._qdata.is_pinned()
    assert torch.equal(m[0].weight.detach(), before) and torch.equal(m[1].weight._qdata, q)
    assert offload.pin_offloaded(m) == (0, 0)  # already pinned: left alone


def test_resolution_stages_blend_into_the_next():
    from zentrainer.harness import trainer as tr
    from zentrainer.spec import TrainSpec

    spec = TrainSpec.from_dict({"model": {"kind": "checkpoint", "ckpt_name": "x.safetensors"},
                                "dataset": {"folder": "d", "resolution": 1024, "stages": [{"resolution": 768, "until": 0.6}],
                                            "stage_blend": 0.2},
                                "optim": {"steps": 1000}})
    t = tr.Trainer.__new__(tr.Trainer)
    t.spec = spec
    t._stages = [(0.6, 768, "ds768", "b768")]
    t._final_stage = (1024, "ds1024", "b1024")

    def res_at(step):
        t.step = step
        return t._stage()[0]

    assert {res_at(s) for s in range(0, 600)} == {768}
    assert {res_at(s) for s in range(800, 1000)} == {1024}
    early = sum(res_at(s) == 1024 for s in range(600, 650)) / 50
    late = sum(res_at(s) == 1024 for s in range(750, 800)) / 50
    assert early < 0.4 < 0.6 < late  # the share rises across the blend
    assert [res_at(s) for s in range(600, 800)] == [res_at(s) for s in range(600, 800)]  # a resume picks the same


def test_the_epoch_follows_the_step_not_the_stage_a_step_took():
    """Each resolution stage has its own batches, and while a change blends a step takes either: the
    epoch a step reports comes from the step in the run's own batches, so it never jumps back."""
    import types

    from zentrainer.harness import data
    from zentrainer.harness import trainer as tr

    def batches(sizes, bs=2, seed=0):
        ds = types.SimpleNamespace(buckets={k: list(range(n)) for k, n in sizes.items()})
        return data.Batches(ds, bs, seed)

    run_own = batches({(1024, 1024): 7, (832, 1216): 4})  # 4 + 2 batches a pass
    low = batches({(768, 768): 11})
    assert run_own.per_epoch == 6 and low.per_epoch == 6
    t = tr.Trainer.__new__(tr.Trainer)
    t._final_stage = (1024, None, run_own)
    t._stages = [(0.6, 768, None, low)]
    t.batches = low  # the stage this step took: it doesn't change the count
    epochs = [t.epoch_at(s) for s in range(1, 40)]
    assert epochs == sorted(epochs) and epochs[:7] == [1, 1, 1, 1, 1, 1, 2] and t.epoch_at(0) == 0

    # one resolution: the same numbers Batches' own refills gave
    t2 = tr.Trainer.__new__(tr.Trainer)
    t2.batches = only = batches({(1024, 1024): 5}, bs=2)
    seen = []
    for _ in range(12):
        next(only)
        seen.append(only.epoch)
    assert [t2.epoch_at(s) for s in range(1, 13)] == seen

    # a resume reshuffles every stage from its step, not just the run's own
    t.spec = types.SimpleNamespace(seed=3)
    t.step = 500
    t.batches = run_own
    def order():
        t._reseed_batches()
        return [next(run_own) for _ in range(6)], [next(low) for _ in range(6)]

    at_500 = order()
    assert order() == at_500  # the same step: the same order
    t.step = 900
    own_900, low_900 = order()
    assert own_900 != at_500[0] and low_900 != at_500[1]  # the lower stage too, not replaying its start


def test_training_on_keeps_a_run_at_its_own_resolution():
    """Train more (or Edit & resume with more steps): the stages are shares of the run, so a longer
    run would drop back to the lower resolution and blend again. Scaled to the new total, a run
    that reached its own resolution stays there, and the stages fall on the same steps."""
    from zentrainer.harness import trainer as tr
    from zentrainer.runs.extend import keep_stage_steps
    from zentrainer.spec import TrainSpec

    spec = TrainSpec.from_dict({"model": {"kind": "checkpoint", "ckpt_name": "x.safetensors"},
                                "dataset": {"folder": "d", "resolution": 1024, "stages": [{"resolution": 768, "until": 0.6}],
                                            "stage_blend": 0.2},
                                "optim": {"steps": 2000}})
    t = tr.Trainer.__new__(tr.Trainer)
    t._final_stage = (1024, "ds1024", "b1024")

    def res_at(sp, step):
        t.spec, t.step = sp, step
        t._stages = [(float(st["until"]), int(st["resolution"]), "ds768", "b768") for st in sp.dataset.stages]
        return t._stage()[0]

    before = [res_at(spec, st) for st in range(0, 2000)]
    longer = TrainSpec.from_dict(spec.to_dict())
    longer.optim.steps = 4000
    assert {res_at(longer, st) for st in range(2000, 4000)} != {1024}  # the bug: back to 768 for a while
    keep_stage_steps(longer, 2000)
    longer.validate()
    assert {res_at(longer, st) for st in range(2000, 4000)} == {1024}
    assert [res_at(longer, st) for st in range(0, 2000)] == before  # the steps it trained: the same plan
    shorter = TrainSpec.from_dict(spec.to_dict())
    shorter.optim.steps = 1500
    keep_stage_steps(shorter, 2000)  # fewer steps only moves a run on: left as it was
    assert shorter.dataset.stages == spec.dataset.stages and shorter.dataset.stage_blend == spec.dataset.stage_blend


def test_resolution_stages_are_checked():
    from zentrainer.spec import SpecError, TrainSpec

    base = {"model": {"kind": "checkpoint", "ckpt_name": "x.safetensors"}, "dataset": {"folder": "d", "resolution": 1024}}
    for bad in ([{"resolution": 1024, "until": 0.5}], [{"resolution": 768, "until": 1.0}],
                [{"resolution": 512, "until": 0.6}, {"resolution": 768, "until": 0.4}]):
        with pytest.raises(SpecError):
            TrainSpec.from_dict({**base, "dataset": {**base["dataset"], "stages": bad}})
