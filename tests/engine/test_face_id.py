"""Tests for harness/face_id.py: geometry mapping, multi-VAE decoding, gating, grouping and gradients."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from zentrainer.harness import face_id as fid
from zentrainer.harness.data import Dataset, Item
from zentrainer.spec import SpecError, TrainSpec

BASE = {"model": {"kind": "checkpoint", "ckpt_name": "m.safetensors"}, "dataset": {"folder": "cats"}}


def _pts(cx: float, cy: float, scale: float = 1.0) -> np.ndarray:
    """Five landmarks roughly centred on (cx, cy)."""
    base = np.array([[-18, -14], [18, -14], [0, 4], [-14, 22], [14, 22]], dtype=np.float64)
    return base * scale + np.array([cx, cy], dtype=np.float64)


def test_spec_defaults_and_slider_rejection():
    s = TrainSpec.from_dict(BASE)
    assert s.objective.face_id == 0.0
    assert s.objective.face_id_every == 2
    assert s.objective.face_id_start == 0.25
    assert s.objective.face_id_device == "cpu"
    assert s.objective.face_id_margin == 0.0

    s2 = TrainSpec.from_dict({
        **BASE,
        "objective": {"face_id": 0.02, "face_id_every": 3, "face_id_start": 0.0, "face_id_device": "gpu", "face_id_margin": 0.82},
    })
    assert (s2.objective.face_id, s2.objective.face_id_every, s2.objective.face_id_start, s2.objective.face_id_device, s2.objective.face_id_margin) == (
        0.02, 3, 0.0, "gpu", 0.82
    )

    with pytest.raises(SpecError, match="sliders"):
        TrainSpec.from_dict({**BASE, "objective": {"face_id": 0.02},
                             "slider": {"kind": "text", "positive": "old", "negative": "young"}})


def test_bucket_landmarks_matches_load_image_crop_and_rejects_clipped():
    # 1200x800 landscape cropped to 512x512 square (buckets=False):
    # scale = max(512/1200, 512/800) = 0.64 -> resized 768x512 -> left crop 128 px
    center_pts = _pts(600, 400, scale=2.0)
    mapped = fid.bucket_landmarks((1200, 800), center_pts, (512, 512))
    assert mapped is not None
    # Centre (600, 400) maps to (256, 256); _pts' y-offsets average to +4.0 * 2.0 * 0.64 = +5.12
    np.testing.assert_allclose(mapped.mean(axis=0), [256.0, 256.0 + 4.0 * 2.0 * 0.64], atol=1e-4)

    # A face near the far left edge gets cropped out in a square crop -> returns None
    edge_pts = _pts(50, 400, scale=1.0)
    assert fid.bucket_landmarks((1200, 800), edge_pts, (512, 512)) is None


@pytest.mark.parametrize("stride,latent_hw", [(8, (64, 64)), (16, (48, 48)), (8, (96, 64))])
def test_crop_plan_bounds_and_grid_range(stride, latent_hw):
    lh, lw = latent_hw
    tw, th = lw * stride, lh * stride
    pts = _pts(tw * 0.5, th * 0.4, scale=1.5)
    plan = fid.crop_plan((tw, th), pts, latent_hw, stride)
    assert plan is not None
    left, top, right, bottom, grid = plan
    assert 0 <= left < right <= lw
    assert 0 <= top < bottom <= lh
    assert grid.shape == (1, 112, 112, 2)
    assert torch.isfinite(grid).all()


def test_probe_neg1_1_detects_both_vae_ranges():
    vae_std = SimpleNamespace(process_output=lambda x: x.clone().add_(1.0).div_(2.0).clamp_(0.0, 1.0))
    vae_01 = SimpleNamespace(process_output=lambda x: x)
    assert fid._probe_neg1_1(vae_std) is True
    assert fid._probe_neg1_1(vae_01) is False


def test_effective_noise_handles_flow_and_eps(monkeypatch):
    monkeypatch.setattr(fid, "is_flow", lambda ms: getattr(ms, "flow", False))

    sig = torch.tensor([0.2, 0.5, 1.0, 4.0])
    assert fid._effective_noise(SimpleNamespace(flow=True), sig) == pytest.approx([0.2, 0.5, 1.0, 4.0], rel=1e-5)
    assert fid._effective_noise(SimpleNamespace(flow=False), sig) == pytest.approx([0.2 / 1.2, 0.5 / 1.5, 0.5, 0.8], rel=1e-5)


class _TinyDecoder(torch.nn.Module):
    """Differentiable stand-in VAE decoder supporting 4D/5D, configurable stride, RGB/RGBA and output range."""

    def __init__(self, in_ch: int, stride: int, out_ch: int = 3, is_5d: bool = False, zero_one: bool = False,
                 detach: bool = False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.stride = stride
        self.is_5d = is_5d
        self.zero_one = zero_one
        self.detach = detach

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_5d:
            assert z.ndim == 5
            z2 = z[:, :, 0]
        else:
            assert z.ndim == 4
            z2 = z
        out = self.conv(z2)
        out = torch.nn.functional.interpolate(out, scale_factor=self.stride, mode="bilinear", align_corners=False)
        out = torch.sigmoid(out) if self.zero_one else torch.tanh(out)
        if self.is_5d:
            out = out.unsqueeze(2)
        return out.detach() if self.detach else out


class _TinyRec(torch.nn.Module):
    """Tiny differentiable stand-in for IResNet: (N, 3, 112, 112) -> (N, 512)."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (3, 112, 112)
        return self.proj(x.mean(dim=(-2, -1)))


def _make_trainer(tmp_path: Path, *, stride: int = 8, in_ch: int = 16, out_ch: int = 3,
                  is_5d: bool = False, zero_one: bool = False, detach: bool = False,
                  folder_triggers: bool = False, steps: int = 20, start: float = 0.0, every: int = 2,
                  masked: bool = False, masked_loss: bool | None = None):
    img_paths = []
    items = []
    for k in range(4):
        sub = tmp_path / ("alice" if k < 2 else "bob") if folder_triggers else tmp_path
        sub.mkdir(parents=True, exist_ok=True)
        p = sub / f"img_{k}.png"
        Image.new("RGB", (512, 512), (120 + k * 10, 90, 80)).save(p)
        mask_p = None
        if masked and k == 0:
            # Left half white (keep face 0), right half black (mask out bystander face 1)
            arr = np.zeros((512, 512), dtype=np.uint8)
            arr[:, :256] = 255
            mask_p = sub / f"img_{k}.mask.png"
            Image.fromarray(arr).save(mask_p)
        img_paths.append(p)
        trig = ("alice" if k < 2 else "bob") if folder_triggers else "subj"
        items.append(Item(path=p, caption="a photo", mask=mask_p, trigger=trig))

    spec = TrainSpec.from_dict({
        **BASE,
        "dataset": {
            "folder": str(tmp_path), "resolution": 512, "folder_triggers": folder_triggers,
            "masked_loss": masked if masked_loss is None else masked_loss,
        },
        "optim": {"steps": steps},
        "objective": {"face_id": 0.02, "face_id_every": every, "face_id_start": start},
    })
    dec = _TinyDecoder(in_ch, stride, out_ch=out_ch, is_5d=is_5d, zero_one=zero_one, detach=detach)
    vae = SimpleNamespace(
        first_stage_model=dec,
        spacial_compression_decode=lambda: stride,
        latent_dim=3 if is_5d else 2,
        latent_channels=in_ch,
        working_dtypes=[torch.float32],
        process_output=(lambda x: x) if zero_one else (lambda x: x.clone().add_(1.0).div_(2.0).clamp_(0.0, 1.0)),
        patcher=None,
    )

    class _FlowMS:
        pass
    _FlowMS.__name__ = "ModelSamplingDiscreteFlow"

    model = SimpleNamespace(process_latent_out=lambda x: x * 2.0)
    ds = Dataset(samples=[], conditionings=[], captions=[], empty_caption=0, owners=[0, 1, 2, 3], items=items)
    tr = SimpleNamespace(
        spec=spec,
        step=0,
        dataset=ds,
        models=SimpleNamespace(vae=vae, patcher=SimpleNamespace(model=model)),
        denoiser=SimpleNamespace(model_sampling=_FlowMS()),
    )
    return tr, img_paths


@pytest.mark.parametrize("stride,in_ch,out_ch,is_5d,zero_one", [
    (8, 16, 3, True, False),   # Krea 2 / Wan / Qwen-Image 5D 8x [-1, 1]
    (16, 128, 3, False, False), # Flux.2 / MOSS 4D 16x [-1, 1]
    (16, 64, 4, False, False),  # Qwen-Image 2.1 RGBA 4D 16x
    (8, 4, 3, False, True),     # [0, 1] VAE output range
])
def test_face_identity_end_to_end_backprop_and_gating(tmp_path, monkeypatch, stride, in_ch, out_ch, is_5d, zero_one):
    import zentrainer.analysis.faces as af

    rec = _TinyRec()
    monkeypatch.setattr(fid, "is_flow", lambda ms: True)
    monkeypatch.setattr(af, "_net", lambda file, kind: rec if kind in ("r50", "r100") else "det")
    monkeypatch.setattr(
        af, "_detect",
        lambda det, rgb: (np.array([[180.0, 180.0, 320.0, 320.0, 0.99]], np.float32),
                          _pts(250.0, 250.0, scale=1.5)[None].astype(np.float32)),
    )

    tr, _ = _make_trainer(tmp_path, stride=stride, in_ch=in_ch, out_ch=out_ch, is_5d=is_5d, zero_one=zero_one,
                          steps=20, start=0.25, every=2)
    face_id = fid.FaceIdentity(tr)
    assert len(face_id.targets) == 4
    assert face_id.start_step == 5

    lh = lw = 512 // stride
    shape = (2, in_ch, 1, lh, lw) if is_5d else (2, in_ch, lh, lw)

    # Before start_step -> skipped
    tr.step = 4
    pred = torch.randn(shape, requires_grad=True)
    assert face_id.loss(tr, pred, torch.tensor([0.3, 0.4]), [0, 1]) == (None, None)

    # Odd step after start_step (every=2) -> skipped
    tr.step = 6
    assert face_id.loss(tr, pred, torch.tensor([0.3, 0.4]), [0, 1]) == (None, None)

    # Out-of-range sigmas -> skipped
    tr.step = 5
    assert face_id.loss(tr, pred, torch.tensor([0.05, 0.80]), [0, 1]) == (None, None)

    # Multi-frame video latent (T > 1) -> skipped
    if is_5d:
        vid = torch.randn((1, in_ch, 3, lh, lw), requires_grad=True)
        assert face_id.loss(tr, vid, torch.tensor([0.3]), [0]) == (None, None)

    # Eligible step + noise: produces finite loss and non-zero gradient on the picked sample only
    pred = torch.randn(shape, requires_grad=True)
    loss, raw_val = face_id.loss(tr, pred, torch.tensor([0.80, 0.35]), [0, 1])
    assert loss is not None and raw_val is not None
    assert torch.isfinite(loss) and 0.0 <= raw_val <= 2.0
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[0].abs().sum().item() == 0.0  # sample 0 was sigma=0.80 (skipped)
    assert pred.grad[1].abs().sum().item() > 0.0   # sample 1 received gradient through VAE + ArcFace

    # Multi-resolution stage: same FaceIdentity works at 384 px without re-scanning
    lh2 = lw2 = 384 // stride
    shape2 = (1, in_ch, 1, lh2, lw2) if is_5d else (1, in_ch, lh2, lw2)
    pred2 = torch.randn(shape2, requires_grad=True)
    loss2, _ = face_id.loss(tr, pred2, torch.tensor([0.40]), [2])
    assert loss2 is not None
    loss2.backward()
    assert pred2.grad is not None and pred2.grad.abs().sum().item() > 0.0


def test_mask_filters_bystander_and_folder_triggers_separate_centroids(tmp_path, monkeypatch):
    import zentrainer.analysis.faces as af

    # Image 0 has TWO faces, but its mask blacks out the right half (face 1), so face 0 is accepted!
    def fake_detect(det, rgb):
        # First pixel R channel identifies image index (120, 130, 140, 150)
        idx = (int(rgb[0, 0, 0]) - 120) // 10
        if idx == 0:
            boxes = np.array([[60.0, 180.0, 180.0, 300.0, 0.99], [320.0, 180.0, 440.0, 300.0, 0.98]], np.float32)
            pts = np.stack([_pts(120.0, 240.0), _pts(380.0, 240.0)]).astype(np.float32)
            return boxes, pts
        return (np.array([[180.0, 180.0, 320.0, 320.0, 0.99]], np.float32),
                _pts(250.0, 250.0)[None].astype(np.float32))

    # Distinct embeddings for alice (idx 0,1: R<=130) vs bob (idx 2,3: R>=140)
    class _SplitRec(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((x.shape[0], 512), dtype=x.dtype)
            # R channel in [-1, 1]: 120->-0.059, 130->+0.020, 140->+0.098, 150->+0.176
            r = x[:, 0].mean(dim=(-2, -1))
            for i in range(x.shape[0]):
                if r[i] < 0.05:
                    out[i, 0] = 1.0
                else:
                    out[i, 1] = 1.0
            return out

    monkeypatch.setattr(af, "_net", lambda file, kind: _SplitRec() if kind in ("r50", "r100") else "det")
    monkeypatch.setattr(af, "_detect", fake_detect)

    tr, paths = _make_trainer(tmp_path, folder_triggers=True, masked=True)
    face_id = fid.FaceIdentity(tr)
    assert len(face_id.targets) == 4
    # Alice's target points along dim 0, Bob's along dim 1 (orthogonal!)
    assert float((face_id.targets[paths[0]].target * face_id.targets[paths[2]].target).sum()) == pytest.approx(0.0, abs=1e-5)
    assert float((face_id.targets[paths[0]].target * face_id.targets[paths[1]].target).sum()) == pytest.approx(1.0, abs=1e-5)

    # With masked_loss=False, the mask file on disk is ignored, so image 0 has 2 unmasked faces and is skipped
    tr_nomask, _ = _make_trainer(tmp_path / "nomask", folder_triggers=True, masked=True, masked_loss=False)
    assert len(fid.FaceIdentity(tr_nomask).targets) == 3


def test_detached_vae_fails_fast_at_startup(tmp_path, monkeypatch):
    import zentrainer.analysis.faces as af

    monkeypatch.setattr(af, "_net", lambda file, kind: _TinyRec() if kind in ("r50", "r100") else "det")
    monkeypatch.setattr(
        af, "_detect",
        lambda det, rgb: (np.array([[180.0, 180.0, 320.0, 320.0, 0.99]], np.float32),
                          _pts(250.0, 250.0)[None].astype(np.float32)),
    )

    tr, _ = _make_trainer(tmp_path, detach=True)
    with pytest.raises(ValueError, match="backpropagate on CPU"):
        fid.FaceIdentity(tr)


def test_gpu_crop_matches_cpu_and_offloads_back_to_cpu(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPU crop decode test")
    import zentrainer.analysis.faces as af

    rec = _TinyRec()
    monkeypatch.setattr(fid, "is_flow", lambda ms: True)
    monkeypatch.setattr(af, "_net", lambda file, kind: rec if kind in ("r50", "r100") else "det")
    monkeypatch.setattr(
        af, "_detect",
        lambda det, rgb: (np.array([[180.0, 180.0, 320.0, 320.0, 0.99]], np.float32),
                          _pts(250.0, 250.0, scale=1.5)[None].astype(np.float32)),
    )

    tr_cpu, _ = _make_trainer(tmp_path / "cpu", stride=8, in_ch=16, is_5d=True, steps=10, start=0.0, every=1)
    tr_gpu, _ = _make_trainer(tmp_path / "gpu", stride=8, in_ch=16, is_5d=True, steps=10, start=0.0, every=1)
    tr_gpu.spec.objective.face_id_device = "gpu"
    # Match decoder weights so CPU and GPU outputs can be compared directly
    tr_gpu.models.vae.first_stage_model.load_state_dict(tr_cpu.models.vae.first_stage_model.state_dict())

    f_cpu = fid.FaceIdentity(tr_cpu)
    f_gpu = fid.FaceIdentity(tr_gpu)

    base = torch.randn((1, 16, 1, 64, 64), device="cuda")
    p_cpu = base.clone().requires_grad_(True)
    p_gpu = base.clone().requires_grad_(True)

    loss_cpu, raw_cpu = f_cpu.loss(tr_cpu, p_cpu, torch.tensor([0.35]), [0])
    loss_gpu, raw_gpu = f_gpu.loss(tr_gpu, p_gpu, torch.tensor([0.35]), [0])
    assert loss_cpu is not None and loss_gpu is not None
    assert raw_gpu == pytest.approx(raw_cpu, abs=1e-4)
    # While GPU SparseID holds its private recognizer on CUDA (before backward), the shared
    # cached recognizer used by background likeness scoring MUST remain untouched on CPU!
    assert f_gpu.rec is not rec
    assert not any(t.is_cuda for t in rec.parameters())
    assert rec(torch.zeros(1, 3, 112, 112)).device.type == "cpu"

    loss_cpu.backward()
    loss_gpu.backward()
    torch.testing.assert_close(p_gpu.grad, p_cpu.grad, atol=1e-4, rtol=1e-3)
    # After backward, decoder and recognizer are offloaded back to CPU cleanly
    assert not any(t.is_cuda for t in f_gpu.decoder.parameters())
    assert not any(t.is_cuda for t in f_gpu.rec.parameters())

    # Even if a shared recognizer was somehow left on CUDA, a fresh FaceIdentity copies to CPU cleanly
    rec.cuda()
    tr_next, _ = _make_trainer(tmp_path / "next", stride=8, in_ch=16, is_5d=True, steps=10, start=0.0, every=1)
    f_next = fid.FaceIdentity(tr_next)
    assert not any(t.is_cuda for t in f_next.rec.parameters())
    f_next.close()


def test_preserve_prior_spec_validation_and_independent_latent_dop(monkeypatch):
    from contextlib import contextmanager
    from zentrainer.harness import trainer as tr_mod
    from zentrainer.harness.data import plain_class

    s = TrainSpec.from_dict(BASE)
    assert s.objective.preserve_prior == 0

    s_ok = TrainSpec.from_dict({
        **BASE,
        "dataset": {"folder": "cats", "trigger": "hanasubj"},
        "objective": {"preserve": "woman", "preserve_prior": 8, "preserve_every": 4, "preserve_weight": 1.5},
    })
    assert s_ok.objective.preserve_prior == 8

    # Turning preserve off while preserve_prior > 0 stays valid (matches preserve_weight/every/plain)
    s_off = TrainSpec.from_dict({**BASE, "objective": {"preserve": "", "preserve_prior": 8}})
    assert s_off.objective.preserve == "" and s_off.objective.preserve_prior == 8

    with pytest.raises(SpecError, match="non-edit"):
        TrainSpec.from_dict({
            **BASE,
            "dataset": {"folder": "cats", "trigger": "hanasubj", "edit": True},
            "objective": {"preserve": "woman", "preserve_prior": 8},
        })

    # Verify _preserve_loss uses independent cached prior latents (never the subject's x_t)
    w = torch.nn.Parameter(torch.tensor(0.5))
    enabled = {"on": True}
    seen_inputs = []

    @contextmanager
    def disabled():
        enabled["on"] = False
        try:
            yield
        finally:
            enabled["on"] = True

    class _FakeDenoiser:
        model_sampling = SimpleNamespace()

        def prepare(self, cond, x0, device, shapes):
            return {"cond": cond}

        def __call__(self, x_t, sigma, conds, tread):
            seen_inputs.append(x_t.detach().clone())
            return x_t * (1.0 + w) if enabled["on"] else x_t

    monkeypatch.setattr(tr_mod, "sigmas_for", lambda ms, p, shift, sched: p)

    tr = tr_mod.Trainer.__new__(tr_mod.Trainer)
    tr.spec = s_ok
    tr.step = 0
    tr.device = torch.device("cpu")
    tr.train_dtype = torch.float32
    tr.scaler = None
    tr._shift = 0.0
    tr._schedule = None
    tr._input_grad = False
    tr.generator = torch.Generator().manual_seed(0)
    tr.models = SimpleNamespace(patcher=SimpleNamespace(model=SimpleNamespace(process_latent_in=lambda x: x)))
    tr.network = SimpleNamespace(disabled=disabled)
    tr.denoiser = _FakeDenoiser()
    tr.objective = SimpleNamespace(noisy=lambda ms, sig, noise, x0: (1.0 - sig.view(-1, 1, 1, 1)) * x0 + sig.view(-1, 1, 1, 1) * noise)
    tr.dataset = Dataset(samples=[], conditionings=["PLAIN_WOMAN"], captions=[plain_class("woman")], empty_caption=0)
    tr._prior_latents = {"woman": [torch.full((4, 8, 8), 7.0, dtype=torch.bfloat16)]}
    tr._prior_conds = {}

    subject_xt = torch.full((1, 4, 8, 8), -99.0)
    loss_val = tr._preserve_loss([0], torch.zeros(1, 4, 8, 8), None, subject_xt, torch.tensor([0.3]), torch.zeros(1, 4, 8, 8), 1)
    assert loss_val is not None and torch.isfinite(loss_val) and loss_val.item() > 0.0
    assert w.grad is not None and torch.isfinite(w.grad) and w.grad.abs().item() > 0.0
    # Neither base nor adapter forward saw the subject's x_t (-99.0)
    assert len(seen_inputs) == 2
    assert all(inp.mean().item() > 0.0 for inp in seen_inputs)


def test_build_prior_latents_uses_alt_sample_model_and_unloads(tmp_path, monkeypatch):
    import sys
    import types
    from zentrainer.harness import trainer as tr_mod
    from zentrainer.harness.data import plain_class

    s = TrainSpec.from_dict({
        **BASE,
        "model": {"kind": "split", "unet_name": "krea2_raw.safetensors", "clip_name": "q.safetensors", "clip_type": "krea2", "vae_name": "v.safetensors"},
        "dataset": {"folder": "cats", "trigger": "hanasubj", "resolution": 512},
        "objective": {"preserve": "woman", "preserve_prior": 2},
        "samples": {"model_name": "krea2_turbo.safetensors", "steps": 8, "cfg": 1.0},
    })

    events = []
    alt_patcher = SimpleNamespace(
        tag="turbo_alt",
        model=SimpleNamespace(latent_format=SimpleNamespace(latent_channels=16), eval=lambda: None),
    )
    train_patcher = SimpleNamespace(
        tag="raw_train",
        model=SimpleNamespace(latent_format=SimpleNamespace(latent_channels=16), eval=lambda: None),
    )

    comfy_mod = types.ModuleType("comfy")
    mm_mod = types.ModuleType("comfy.model_management")
    mm_mod.load_models_gpu = lambda models, **kw: events.append(("load_gpu", models[0].tag))
    mm_mod.unload_model_and_clones = lambda m: events.append(("unload", m.tag))
    mm_mod.cleanup_models_gc = lambda: events.append(("gc", None))
    mm_mod.soft_empty_cache = lambda: events.append(("empty_cache", None))

    sd_mod = types.ModuleType("comfy.sd")
    sd_mod.load_diffusion_model = lambda path, model_options=None: (events.append(("load_diff", path)), alt_patcher)[1]

    sample_mod = types.ModuleType("comfy.sample")
    sample_mod.fix_empty_latent_channels = lambda p, lat: lat
    sample_mod.prepare_noise = lambda lat, seed: torch.zeros_like(lat)

    def fake_sample(patcher, noise, steps, cfg, sampler, scheduler, pos, neg, lat, **kw):
        events.append(("sample", patcher.tag, steps, cfg))
        return torch.ones_like(lat)

    sample_mod.sample = fake_sample
    comfy_mod.model_management = mm_mod
    comfy_mod.sd = sd_mod
    comfy_mod.sample = sample_mod
    fp_mod = types.ModuleType("folder_paths")
    fp_mod.get_full_path_or_raise = lambda kind, name: f"/{kind}/{name}"

    for name, mod in [
        ("comfy", comfy_mod), ("comfy.model_management", mm_mod),
        ("comfy.sd", sd_mod), ("comfy.sample", sample_mod), ("folder_paths", fp_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    tr = tr_mod.Trainer.__new__(tr_mod.Trainer)
    tr.spec = s
    tr.streams = SimpleNamespace(multi=False)
    tr.cache_dir = tmp_path / "cache"
    tr.device = torch.device("cpu")
    tr.train_dtype = torch.float32
    tr._status = lambda *a, **kw: None
    tr.models = SimpleNamespace(
        patcher=train_patcher,
        vae=SimpleNamespace(spacial_compression_encode=lambda: 8, latent_dim=2),
    )
    tr.network = SimpleNamespace(assist=True, turbo=0.0)
    tr.dataset = Dataset(samples=[], conditionings=["EMPTY", "PLAIN_WOMAN"],
                         captions=["", plain_class("woman")], empty_caption=0)

    tr._build_prior_latents()
    assert len(tr._prior_latents["woman"]) == 2
    assert ("load_diff", "/diffusion_models/krea2_turbo.safetensors") in events
    assert ("sample", "turbo_alt", 8, 1.0) in events
    assert ("unload", "turbo_alt") in events
    assert ("empty_cache", None) in events
    # Second call hits disk cache and loads nothing
    events.clear()
    tr._prior_latents = {}
    tr._build_prior_latents()
    assert len(tr._prior_latents["woman"]) == 2
    assert events == []


def test_face_id_margin_gates_gradient_above_target():
    """face_id_margin shuts off gradient once cos >= margin while keeping full gradient below margin."""
    fi = object.__new__(fid.FaceIdentity)
    fi.vae = SimpleNamespace()
    fi.decoder = _TinyDecoder(4, stride=4)
    fi.rec = _TinyRec()
    fi.latent_dim = 2
    fi.decode_dtype = torch.float32
    fi._neg1_1 = True
    fi.margin = 0.80

    z = torch.randn(1, 4, 4, 4, requires_grad=True)
    grid = torch.zeros(1, 112, 112, 2)
    with torch.no_grad():
        rgb = fi._decode_rgb(z, torch.device("cpu"))
        crop = torch.nn.functional.grid_sample(rgb + 1.0, grid, align_corners=True) - 1.0
        cur_emb = torch.nn.functional.normalize(fi.rec(crop), dim=1)

    # Identical target (cos == 1.0 >= 0.80): opt_loss is 0 and z receives zero gradient
    opt_hi, dist_hi = fi._eval_crop(z, grid, cur_emb.clone(), torch.device("cpu"))
    assert opt_hi.item() == pytest.approx(0.0, abs=1e-6)
    assert dist_hi.item() == pytest.approx(0.0, abs=1e-5)
    (g_hi,) = torch.autograd.grad(opt_hi, z)
    assert g_hi.abs().max().item() == 0.0

    # Opposite target (cos == -1.0 < 0.80): opt_loss == margin - cos == 1.80 with non-zero gradient
    opt_lo, dist_lo = fi._eval_crop(z, grid, -cur_emb.clone(), torch.device("cpu"))
    assert opt_lo.item() == pytest.approx(1.80, abs=1e-5)
    assert dist_lo.item() == pytest.approx(2.00, abs=1e-5)
    (g_lo,) = torch.autograd.grad(opt_lo, z)
    assert g_lo.abs().max().item() > 0.0


def test_face_id_rec_loaded_under_inference_mode_still_backprops(tmp_path, monkeypatch):
    """A cached recognizer built during inference_mode likeness scoring must still support autograd in SparseID."""
    import zentrainer.analysis.faces as af

    with torch.inference_mode():
        rec = _TinyRec()
    assert any(p.is_inference() for p in rec.parameters())

    monkeypatch.setattr(fid, "is_flow", lambda ms: True)
    monkeypatch.setattr(af, "_net", lambda file, kind: rec if kind in ("r50", "r100") else "det")
    monkeypatch.setattr(
        af, "_detect",
        lambda det, rgb: (np.array([[180.0, 180.0, 320.0, 320.0, 0.99]], np.float32),
                          _pts(250.0, 250.0, scale=1.5)[None].astype(np.float32)),
    )

    tr, _ = _make_trainer(tmp_path, stride=8, in_ch=16, is_5d=True, steps=10, start=0.0, every=1)
    face_id = fid.FaceIdentity(tr)
    assert not any(p.is_inference() for p in face_id.rec.parameters())
    z = torch.randn(1, 16, 1, 12, 12, requires_grad=True)
    grid = torch.zeros(1, 112, 112, 2)
    target = next(iter(face_id.targets.values())).target
    opt_loss, _ = face_id._eval_crop(z, grid, target, torch.device("cpu"))
    (g,) = torch.autograd.grad(opt_loss, z)
    assert torch.isfinite(g).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_face_id_gpu_decode_handles_comfy_kitchen_group_norm_silu_pad3d():
    ck = pytest.importorskip("comfy_kitchen")
    if not hasattr(ck, "group_norm_silu_pad3d"):
        pytest.skip("comfy_kitchen.group_norm_silu_pad3d not present")

    class _KitchenDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(16))
            self.b = torch.nn.Parameter(torch.zeros(16))
            self.conv = torch.nn.Conv3d(16, 3, kernel_size=1)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            if z.is_cuda:
                h = ck.group_norm_silu_pad3d(z, self.w, self.b, 4, 1e-6, (1, 1, 1, 1, 0), silu=True)
                h = h[..., 1:-1, 1:-1]
            else:
                h = torch.nn.functional.silu(z)
            return self.conv(h)

    fi = object.__new__(fid.FaceIdentity)
    fi.vae = SimpleNamespace(patcher=None)
    fi.decoder = _KitchenDecoder().eval().requires_grad_(False)
    fi.rec = _TinyRec().eval().requires_grad_(False)
    fi.latent_dim = 3
    fi.latent_channels = 16
    fi.decode_dtype = torch.bfloat16
    fi._neg1_1 = True
    fi.device_pref = "gpu"
    fi.margin = 0.0

    # Startup verifier succeeds on both CPU and CUDA and offloads back to CPU
    fi._verify_decoder()
    assert fi.device_pref == "gpu"
    assert not any(t.is_cuda for t in fi.decoder.parameters())

    z = torch.randn(1, 16, 1, 4, 4, device="cuda", requires_grad=True)
    rgb = fi._decode_rgb(z, torch.device("cuda"))
    (g,) = torch.autograd.grad(rgb.mean(), z)
    assert torch.isfinite(g).all() and g.abs().max().item() > 0.0
    fi.close()
    assert not any(t.is_cuda for t in fi.decoder.parameters())


def test_differentiable_kitchen_forwards_version_specific_kwargs(monkeypatch):
    ck = pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.backends import eager as ck_eager

    seen = {}

    def fake_pub(x, weight=None, bias=None, num_groups=32, eps=1e-6, pad=(0, 0, 0, 0, 0),
                 silu=True, zero_pad=False, stats=None, out=None):
        raise AssertionError("should have been routed to eager")

    def fake_eager(x, weight, bias, num_groups, eps, pad, silu, zero_pad=False, stats=None):
        seen["pad"] = pad
        seen["stats"] = stats
        return x * 2.0

    def fake_stats_pub(x, num_groups=32, eps=1e-6):
        raise AssertionError("stats3d should have been routed to eager")

    def fake_stats_eager(x, num_groups=32, eps=1e-6):
        seen["stats3d"] = num_groups
        return x.mean(dim=(-2, -1), keepdim=True)

    monkeypatch.setattr(ck, "group_norm_silu_pad3d", fake_pub)
    monkeypatch.setattr(ck_eager, "group_norm_silu_pad3d", fake_eager)
    monkeypatch.setattr(ck, "group_norm_stats3d", fake_stats_pub, raising=False)
    monkeypatch.setattr(ck_eager, "group_norm_stats3d", fake_stats_eager, raising=False)

    x = torch.randn(1, 4, 1, 2, 2, requires_grad=True)
    with fid._differentiable_kitchen(True):
        st = ck.group_norm_stats3d(x, num_groups=4)
        y = ck.group_norm_silu_pad3d(x, pad=(1, 1, 1, 1, 0), stats=st)
    (g,) = torch.autograd.grad(y.sum(), x)
    assert seen["pad"] == [1, 1, 1, 1, 0] and seen["stats3d"] == 4 and torch.is_tensor(seen["stats"])
    assert torch.allclose(g, torch.full_like(x, 2.0))


def test_cpu_without_hw_bf16_casts_bf16_vae_to_fp32_and_restores_for_inference(tmp_path, monkeypatch):
    """On CPUs without AVX-512 BF16 / AMX (e.g. Ryzen 5900X), a bf16-loaded VAE must decode
    in fp32 on CPU (avoiding >50x software-emulated bf16 stalls) and restore bf16 + contiguous
    layout before sampling or on teardown."""
    import zentrainer.analysis.faces as af

    rec = _TinyRec()
    monkeypatch.setattr(fid, "is_flow", lambda ms: True)
    monkeypatch.setattr(fid, "_cpu_bf16_fast", lambda: False)
    monkeypatch.setattr(af, "_net", lambda file, kind: rec if kind in ("r50", "r100") else "det")
    monkeypatch.setattr(
        af, "_detect",
        lambda det, rgb: (np.array([[180.0, 180.0, 320.0, 320.0, 0.99]], np.float32),
                          _pts(250.0, 250.0, scale=1.5)[None].astype(np.float32)),
    )

    tr, _ = _make_trainer(tmp_path, stride=8, in_ch=16, is_5d=False, steps=10, start=0.0, every=1)
    tr.models.vae.working_dtypes = [torch.bfloat16, torch.float32]
    tr.models.vae.first_stage_model.to(torch.bfloat16)

    face_id = fid.FaceIdentity(tr)
    assert face_id.decode_dtype == torch.float32
    assert face_id.gpu_decode_dtype == torch.bfloat16
    # After startup verification, shared VAE is restored to bf16 + contiguous for samples.on_start
    assert next(face_id.decoder.parameters()).dtype == torch.bfloat16
    assert next(face_id.decoder.parameters()).is_contiguous()

    # Running a CPU SparseID step casts decoder to fp32 and backprops cleanly
    pred = torch.randn((1, 16, 64, 64), requires_grad=True)
    loss, _ = face_id.loss(tr, pred, torch.tensor([0.35]), [0])
    assert loss is not None and torch.isfinite(loss)
    assert next(face_id.decoder.parameters()).dtype == torch.float32
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum().item() > 0.0

    # Restoring before sampling (enter_inference) or teardown puts VAE back to bf16 + contiguous
    face_id.restore_vae()
    assert next(face_id.decoder.parameters()).dtype == torch.bfloat16
    assert next(face_id.decoder.parameters()).is_contiguous()
    face_id.close()


def test_differentiable_kitchen_disables_and_restores_fused_norm_served(monkeypatch):
    """ComfyUI #17196 added in-place scratch-buffer Flux2 VAE decoding gated by fused_norm_served;
    _differentiable_kitchen(True) must disable it during CUDA SparseID backprop and restore it on exit."""
    import sys
    import types

    dm = types.ModuleType("comfy.ldm.modules.diffusionmodules.model")
    orig = lambda dev, dt: True  # noqa: E731
    dm.fused_norm_served = orig
    for pkg in ("comfy", "comfy.ldm", "comfy.ldm.modules", "comfy.ldm.modules.diffusionmodules"):
        monkeypatch.setitem(sys.modules, pkg, types.ModuleType(pkg))
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.diffusionmodules.model", dm)

    assert dm.fused_norm_served(torch.device("cpu"), torch.bfloat16) is True
    with fid._differentiable_kitchen(True):
        assert dm.fused_norm_served(torch.device("cpu"), torch.bfloat16) is False
    assert dm.fused_norm_served is orig
