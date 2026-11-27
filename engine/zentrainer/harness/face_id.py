"""SparseID: differentiable face identity loss on a cropped patch of the denoised latent.

During startup, scans the dataset's still images once with ZenTrainer's native PyTorch face
detector and ArcFace recognizer (Settings -> face_model: buffalo_l, antelopev2, or AuraFace for
commercial use) to record each single-face picture's 5-point landmarks and the subject's clean
unit identity centroid (or one centroid per subfolder trigger when dataset.folder_triggers is on).

During training, on eligible steps (every `face_id_every` steps after `face_id_start`, on at most
one mid/low-noise image per optimizer step), slices only the padded latent rectangle covering that
face, decodes that small patch on CPU (or CUDA when `face_id_device="gpu"`) through the frozen VAE
decoder, warps to 112x112 with differentiable `grid_sample`, and returns
`weight * (1 - cosine(emb, target))` (or `weight * relu(margin - cosine)` when `face_id_margin > 0`).
Zero extra VRAM on CPU and works across image VAEs (4D/5D, 8x/16x/32x, RGB/RGBA, [-1, 1] or
[0, 1] output ranges) and all resolution stages.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import is_video
from .objective import is_flow

SIGMA_MIN = 0.10  # below this the face is already clean; gradients add little
SIGMA_MAX = 0.65  # above this x0_pred is too blurry for ArcFace geometry
MIN_FACE_PX = 48  # minimum face box side (px at final training resolution) to supervise
CONTEXT_PX = 32   # border context (px) around the 5-point crop so VAE convs stay clean
DET_MAX_SIDE = 1600
MAX_CROP_CACHE = 512


@dataclass(frozen=True)
class FaceTarget:
    """One accepted picture's landmarks in upright original pixels and identity target."""

    orig_size: tuple[int, int]  # (ow, oh) upright image dimensions
    points: np.ndarray          # (5, 2) float64 landmarks in upright original pixel space
    target: torch.Tensor        # (1, 512) float32 unit embedding on CPU


def bucket_landmarks(orig_size: tuple[int, int], points: np.ndarray,
                     target_size: tuple[int, int]) -> np.ndarray | None:
    """Map 5-point landmarks from upright original pixels into `load_image`'s resized+cropped
    bucket `(tw, th)`. Returns None if any landmark fell outside the crop."""
    ow, oh = orig_size
    tw, th = target_size
    if ow <= 0 or oh <= 0 or tw <= 0 or th <= 0:
        return None
    scale = max(tw / ow, th / oh)
    rw, rh = max(tw, round(ow * scale)), max(th, round(oh * scale))
    left, top = (rw - tw) // 2, (rh - th) // 2
    pts = points.astype(np.float64) * np.array([rw / ow, rh / oh], dtype=np.float64) - np.array([left, top], dtype=np.float64)
    if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= tw) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= th):
        return None
    return pts


def crop_plan(orig_size: tuple[int, int], points: np.ndarray, latent_hw: tuple[int, int],
              stride: int) -> tuple[int, int, int, int, torch.Tensor] | None:
    """Compute `(left, top, right, bottom, grid)` in latent indices and normalized `grid_sample`
    coordinates for a `(lh, lw)` latent at spatial `stride`. Returns None if the face is outside."""
    from ..analysis import faces as af

    lh, lw = latent_hw
    tw, th = lw * stride, lh * stride
    pts = bucket_landmarks(orig_size, points, (tw, th))
    if pts is None:
        return None
    m = af._similarity(pts.astype(np.float32), af.ARCFACE_DST)
    inv = np.linalg.inv(np.vstack([m.astype(np.float64), [0.0, 0.0, 1.0]]))[:2]
    yy, xx = np.indices((112, 112))
    sx, sy = (inv @ np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))).reshape(2, 112, 112)
    pad = max(2, math.ceil(CONTEXT_PX / stride))
    left = max(0, math.floor(sx.min() / stride) - pad)
    top = max(0, math.floor(sy.min() / stride) - pad)
    right = min(lw, math.ceil((sx.max() + 1.0) / stride) + pad)
    bottom = min(lh, math.ceil((sy.max() + 1.0) / stride) + pad)
    ww, hh = stride * (right - left), stride * (bottom - top)
    if ww <= 1 or hh <= 1:
        return None
    gx = 2.0 * (sx - stride * left) / (ww - 1) - 1.0
    gy = 2.0 * (sy - stride * top) / (hh - 1) - 1.0
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1)).float()[None]
    return left, top, right, bottom, grid


def _cpu_bf16_fast() -> bool:
    """True when this CPU has hardware BF16 instructions (AVX-512 BF16 or AMX); without them,
    CPU bfloat16 conv2d/conv3d falls back to slow software emulation (10-50x slower than fp32)."""
    for fn in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        check = getattr(torch.cpu, fn, None)
        if callable(check):
            try:
                if check():
                    return True
            except Exception:
                pass
    return False


def _probe_neg1_1(vae) -> bool:
    """True when `vae.first_stage_model.decode` emits [-1, 1] (most VAEs: process_output maps -1 -> 0);
    False when it already emits [0, 1] (TAEHV, StageA, MiniMax H3)."""
    fn = getattr(vae, "process_output", None)
    if fn is None:
        return True
    try:
        with torch.no_grad():
            probe = fn(torch.tensor([-1.0, 0.0, 1.0]))
        return abs(float(probe[0])) < 0.25 and abs(float(probe[1]) - 0.5) < 0.25
    except Exception:
        return True


def _effective_noise(ms, sigma: torch.Tensor) -> list[float]:
    """Noise fraction in (0, 1) comparable across flow (sigma directly) and eps/v-pred schedules."""
    s = sigma.detach().float().cpu()
    if not is_flow(ms):
        s = s / (1.0 + s)
    return s.tolist()


@contextlib.contextmanager
def _differentiable_kitchen(enabled: bool):
    """Route inference-only comfy_kitchen VAE ops (group_norm_*, fp16_conv*) to comfy_kitchen's
    own pure-PyTorch eager implementations and disable ComfyUI's in-place fused Flux2 VAE buffer
    reuse (`fused_norm_served`, ComfyUI #17196) during CUDA SparseID decode so autograd backward
    works across ComfyUI / comfy_kitchen versions without altering inference fast paths."""
    if not enabled:
        yield
        return
    saved_fused = None
    saved = {}
    ck = None
    try:
        try:
            import comfy.ldm.modules.diffusionmodules.model as dm_mod

            if hasattr(dm_mod, "fused_norm_served"):
                saved_fused = (dm_mod, dm_mod.fused_norm_served)
                dm_mod.fused_norm_served = lambda *_, **__: False
        except ImportError:
            pass
        try:
            import inspect

            import comfy_kitchen as ck
            from comfy_kitchen.backends import eager as ck_eager
        except ImportError:
            ck = None
        if ck is not None:
            for name in dir(ck):
                if not (name.startswith("group_norm") or name.startswith("fp16_conv")):
                    continue
                orig = getattr(ck, name, None)
                eager_fn = getattr(ck_eager, name, None)
                if not (callable(orig) and callable(eager_fn)):
                    continue
                sig = inspect.signature(orig)
                eager_params = set(inspect.signature(eager_fn).parameters)

                def _make_wrap(fn, fn_sig, fn_params):
                    def _wrap(*args, **kwargs):
                        bound = fn_sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                        ba = dict(bound.arguments)
                        out = ba.pop("out", None)
                        if "pad" in ba and ba["pad"] is not None:
                            ba["pad"] = list(ba["pad"])
                        if "stride" in ba and ba["stride"] is not None:
                            s = ba["stride"]
                            ba["stride"] = [s] * 3 if isinstance(s, int) else list(s)
                        res = fn(**{k: v for k, v in ba.items() if k in fn_params})
                        if out is not None:
                            out.copy_(res)
                            return out
                        return res

                    return _wrap

                saved[name] = orig
                setattr(ck, name, _make_wrap(eager_fn, sig, eager_params))
        yield
    finally:
        if ck is not None:
            for k, v in saved.items():
                setattr(ck, k, v)
        if saved_fused is not None:
            saved_fused[0].fused_norm_served = saved_fused[1]


def _detect_on_image(path: Path, det, det_kind: str) -> tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray] | None:
    """Load upright RGB (thumbnailed for detection), run detector, return `(orig_wh, rgb_det, boxes, points)`
    with boxes and points rescaled back into upright original pixel space."""
    from PIL import Image, ImageOps

    from ..analysis import faces as af

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            ow, oh = im.size
            if max(ow, oh) > DET_MAX_SIDE:
                im.thumbnail((DET_MAX_SIDE, DET_MAX_SIDE))
            rgb = np.asarray(im)
    except OSError:
        return None
    dh, dw = rgb.shape[:2]
    detect = af._detect_yunet_refined if det_kind == "yunet" else af._detect
    boxes, points = detect(det, rgb)
    if not len(boxes):
        return (ow, oh), rgb, boxes, points
    sx, sy = ow / dw, oh / dh
    boxes_orig = boxes.copy()
    boxes_orig[:, [0, 2]] *= sx
    boxes_orig[:, [1, 3]] *= sy
    points_orig = points.astype(np.float64) * np.array([sx, sy], dtype=np.float64)
    return (ow, oh), rgb, boxes_orig, points_orig


def _mask_keeps_face(box: np.ndarray, orig_size: tuple[int, int], mask: np.ndarray | None) -> bool:
    """Whether a face's box is mostly in the white (trained) region of `mask`."""
    from ..datasets import faces as df

    if mask is None:
        return True
    ow, oh = orig_size
    n = mask.shape[0]
    x0 = max(0, int((box[0] / max(ow, 1)) * n))
    y0 = max(0, int((box[1] / max(oh, 1)) * n))
    x1 = min(n, max(x0 + 1, int(math.ceil((box[2] / max(ow, 1)) * n))))
    y1 = min(n, max(y0 + 1, int(math.ceil((box[3] / max(oh, 1)) * n))))
    return float(mask[y0:y1, x0:x1].mean()) >= df.MASKED_OUT


class FaceIdentity:
    """Differentiable face identity auxiliary loss attached to a Trainer."""

    def __init__(self, tr) -> None:
        from ..analysis import faces as af
        from ..datasets import faces as df

        s = tr.spec
        self.weight = float(s.objective.face_id)
        self.every = int(s.objective.face_id_every)
        self.start_step = int(s.optim.steps * s.objective.face_id_start)
        self.device_pref = str(s.objective.face_id_device)
        self.margin = float(getattr(s.objective, "face_id_margin", 0.0))
        self.vae = tr.models.vae
        self.decoder = getattr(self.vae, "first_stage_model", None)
        if self.decoder is None or not hasattr(self.decoder, "decode"):
            raise ValueError("objective.face_id needs a VAE with a decoder")
        self.decoder.eval().requires_grad_(False)
        self.stride = int(self.vae.spacial_compression_decode())
        self.latent_dim = int(getattr(self.vae, "latent_dim", 2))
        self.latent_channels = int(getattr(self.vae, "latent_channels", 4))
        self._neg1_1 = _probe_neg1_1(self.vae)

        param = next(self.decoder.parameters(), None)
        self._orig_dtype = param.dtype if param is not None else getattr(self.vae, "vae_dtype", torch.float32)
        working = getattr(self.vae, "working_dtypes", [torch.bfloat16, torch.float32])
        want_bf16 = (
            self._orig_dtype == torch.bfloat16
            or (torch.bfloat16 in working and self._orig_dtype != torch.float16)
        )
        # Use bf16 on CUDA when supported; on CPU only when the CPU has hardware BF16
        # (AVX-512 BF16 / AMX) — on Zen 3 (e.g. Ryzen 5900X) / older CPUs, software-emulated
        # CPU bf16 convs are >50x slower than fp32 even when ComfyUI loaded the VAE in bf16.
        self.gpu_decode_dtype = torch.bfloat16 if want_bf16 else torch.float32
        self.decode_dtype = torch.bfloat16 if (want_bf16 and _cpu_bf16_fast()) else torch.float32
        self._logged_first_crop = False
        self._cl_ready = False

        self.face_model = df.model()
        _, _, rec_file, depth, det_kind = af.MODELS[self.face_model]
        with torch.inference_mode(False), torch.no_grad():
            with af._lock:
                det = af._net(af.DETECTORS[det_kind], det_kind)
                shared_rec = af._net(rec_file, depth)
                # Own a private CPU copy of the recognizer so GPU SparseID (.to(cuda)) never mutates
                # or races with the shared CPU cache used by background sample likeness scoring.
                self.rec = copy.deepcopy(shared_rec).to("cpu").eval().requires_grad_(False)
            # If an older cached recognizer held inference tensors, replace them with regular tensors so autograd works.
            if any(t.is_inference() for t in itertools.chain(self.rec.parameters(), self.rec.buffers())):
                for m in self.rec.modules():
                    for k, p in list(m._parameters.items()):
                        if isinstance(p, torch.Tensor) and p.is_inference():
                            m._parameters[k] = torch.nn.Parameter(p.detach().clone(), requires_grad=False)
                    for k, buf in list(m._buffers.items()):
                        if isinstance(buf, torch.Tensor) and buf.is_inference():
                            m._buffers[k] = buf.detach().clone()

        self._crop_cache: dict[tuple[Path, tuple[int, int]], tuple | None] = {}
        self.targets: dict[Path, FaceTarget] = {}
        self._prepare(tr, det, det_kind)
        self._verify_decoder()

    def _prepare(self, tr, det, det_kind: str) -> None:
        from ..analysis import faces as af
        from ..datasets import faces as df
        from .data import bucket_sizes, pick_bucket

        s = tr.spec.dataset
        items = tr.dataset.items
        stills = [it for it in items if not is_video(it.path)]
        if not stills:
            raise ValueError("objective.face_id needs still pictures in the dataset (videos aren't supported)")
        sizes = bucket_sizes(s.resolution, s.bucket_step) if s.buckets else [(s.resolution, s.resolution)]

        # per-group (trigger or "") candidates: path -> (orig_wh, pts_orig, emb)
        groups: dict[str, list[tuple[Path, tuple[int, int], np.ndarray, np.ndarray]]] = {}
        with torch.no_grad():
            for it in stills:
                found = _detect_on_image(it.path, det, det_kind)
                if found is None:
                    continue
                (ow, oh), rgb_det, boxes, points = found
                if not len(boxes):
                    continue
                mask = df._mask_array(it.mask) if s.masked_loss else None
                kept = [i for i in range(len(boxes)) if _mask_keeps_face(boxes[i], (ow, oh), mask)]
                # Only an unambiguous single trained face supplies identity supervision.
                if len(kept) != 1:
                    continue
                k = kept[0]
                tw, th = pick_bucket(ow, oh, sizes)
                scale = max(tw / ow, th / oh)
                box_side = min(boxes[k, 2] - boxes[k, 0], boxes[k, 3] - boxes[k, 1]) * scale
                if box_side < MIN_FACE_PX:
                    continue
                if not s.buckets and bucket_landmarks((ow, oh), points[k], (tw, th)) is None:
                    continue
                dh, dw = rgb_det.shape[:2]
                pts_det = (points[k] * np.array([dw / ow, dh / oh], dtype=np.float64)).astype(np.float32)
                m = af._similarity(pts_det, af.ARCFACE_DST)
                crop = af._warp(rgb_det, m, 112)
                x = torch.from_numpy(crop).permute(2, 0, 1).float()[None] / 127.5 - 1.0
                emb = F.normalize(self.rec(x), dim=1).flatten().numpy()
                group = it.trigger.strip() if s.folder_triggers else ""
                groups.setdefault(group, []).append((it.path, (ow, oh), points[k], emb))

        accepted = 0
        for group, rows in groups.items():
            embs = [r[3] for r in rows]
            if not s.folder_triggers and len(embs) >= 4:
                loo = sorted(df._leave_one_out({str(i): e for i, e in enumerate(embs)}).values())
                median = loo[len(loo) // 2]
                if median < 0.25:
                    raise ValueError(
                        f"objective.face_id needs one person per dataset (median face similarity is {median:.2f}, "
                        "too low for one identity): turn off objective.face_id, or use dataset.folder_triggers "
                        "with one person per subfolder"
                    )
                if median < df.ONE_IDENTITY:
                    logging.warning("[ZenTrainer] face identity: dataset face similarity is modest (median %.2f); "
                                    "double-check that all pictures are the same person", median)
            centroid = df._centroid(embs)
            if centroid is None:
                continue
            kept_rows = [r for r in rows if float(r[3] @ centroid) >= df.OUTLIER]
            if not kept_rows:
                continue
            centroid = df._centroid([r[3] for r in kept_rows])
            target = torch.from_numpy(centroid.astype(np.float32))[None]
            for path, orig_size, pts, _ in kept_rows:
                self.targets[path] = FaceTarget(orig_size, pts, target)
                accepted += 1

        if not accepted:
            raise ValueError(
                f"objective.face_id found no clear single-person faces in {s.folder!r} "
                f"(checked {len(stills)} pictures with {self.face_model}): turn off objective.face_id "
                "or check that faces are visible and unmasked"
            )
        active_dt = self.gpu_decode_dtype if self.device_pref == "gpu" else self.decode_dtype
        logging.info("[ZenTrainer] SparseID (%s, %s/%s): %d of %d pictures accepted "
                     "(weight %.3g, every %d steps, from step %d)",
                     self.face_model, self.device_pref, str(active_dt).replace("torch.", ""),
                     accepted, len(stills), self.weight, self.every, self.start_step)

    def _detach_vae_patcher(self) -> None:
        """Detach ComfyUI's DynamicVRAM / offload hooks on the VAE patcher before raw .to() calls."""
        patcher = getattr(self.vae, "patcher", None)
        if patcher is None:
            return
        import comfy.model_management as mm

        if any(m.model is patcher for m in mm.current_loaded_models):
            mm.unload_model_and_clones(patcher, unload_additional_models=False)

    def _decoder_needs(self, device: torch.device, dtype: torch.dtype) -> bool:
        return any(
            t.device != device or (t.is_floating_point() and t.dtype != dtype)
            for t in itertools.chain(self.decoder.parameters(), self.decoder.buffers())
        )

    def _ensure_cpu_decoder(self, dtype: torch.dtype | None = None) -> None:
        self._detach_vae_patcher()
        dt = dtype if dtype is not None else getattr(self, "_orig_dtype", self.decode_dtype)
        if self._decoder_needs(torch.device("cpu"), dt):
            self.decoder.to(device="cpu", dtype=dt)

    def restore_vae(self) -> None:
        """Restore the shared VAE decoder to CPU, its original dtype, and contiguous layout
        before ComfyUI's sampler or DynamicVRAM loader uses it again."""
        self._detach_vae_patcher()
        orig_dt = getattr(self, "_orig_dtype", self.decode_dtype)
        if getattr(self, "_cl_ready", False):
            self.decoder.to(device="cpu", dtype=orig_dt, memory_format=torch.contiguous_format)
            self._cl_ready = False
        elif self._decoder_needs(torch.device("cpu"), orig_dt):
            self.decoder.to(device="cpu", dtype=orig_dt)

    def _move_to_cpu(self) -> None:
        self.restore_vae()
        if any(t.is_cuda for t in itertools.chain(self.rec.parameters(), self.rec.buffers())):
            self.rec.to("cpu")

    def close(self) -> None:
        """Restore the shared VAE decoder and offload the private recognizer on run teardown."""
        self._move_to_cpu()

    def _decode_rgb(self, z: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Decode a small cropped latent `(1, C, [1,] h, w)` on `device` (CPU by default) to float32 RGB in [-1, 1]."""
        dev = device if device is not None else torch.device("cpu")
        dt = getattr(self, "gpu_decode_dtype", self.decode_dtype) if dev.type == "cuda" else self.decode_dtype
        if dev.type == "cpu":
            self._ensure_cpu_decoder(dt)
        else:
            self._detach_vae_patcher()
            if self._decoder_needs(dev, dt):
                if getattr(self, "_cl_ready", False):
                    self.decoder.to(device=dev, dtype=dt, memory_format=torch.contiguous_format)
                    self._cl_ready = False
                else:
                    self.decoder.to(device=dev, dtype=dt)
        if self.latent_dim == 2 and z.ndim == 5:
            z = z[:, :, 0]
        z = z.to(dev, dt)
        if dev.type == "cpu" and self.latent_dim == 2 and z.ndim == 4:
            if not getattr(self, "_cl_ready", False):
                self.decoder.to(memory_format=torch.channels_last)
                self._cl_ready = True
            z = z.contiguous(memory_format=torch.channels_last)
        ctx = torch.autocast(dev.type, dtype=torch.bfloat16) if dt == torch.bfloat16 else contextlib.nullcontext()
        with ctx, _differentiable_kitchen(dev.type == "cuda" and torch.is_grad_enabled()):
            rgb = self.decoder.decode(z)
        if rgb.ndim == 5:
            rgb = rgb[:, :, 0]
        if rgb.shape[1] > 3:
            rgb = rgb[:, :3]
        rgb = rgb.float()
        return rgb.clamp(-1.0, 1.0) if self._neg1_1 else rgb.clamp(0.0, 1.0) * 2.0 - 1.0

    def _eval_crop(self, z_crop: torch.Tensor, grid: torch.Tensor, target: torch.Tensor,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the face latent patch and return `(opt_loss, raw_cosine_distance)` on `device`."""
        if device.type == "cpu":
            if any(t.is_cuda for t in itertools.chain(self.rec.parameters(), self.rec.buffers())):
                self.rec.to("cpu")
        else:
            if any(t.device != device for t in itertools.chain(self.rec.parameters(), self.rec.buffers())):
                self.rec.to(device)
        rgb = self._decode_rgb(z_crop, device)
        # Black image padding in normalized ArcFace [-1, 1] space.
        crop = F.grid_sample(rgb + 1.0, grid.to(device), align_corners=True) - 1.0
        if device.type == "cpu":
            crop = crop.contiguous(memory_format=torch.channels_last)
        emb = F.normalize(self.rec(crop), dim=1)
        cos = (emb * target.to(device)).sum()
        raw_dist = 1.0 - cos
        opt_loss = F.relu(self.margin - cos) if self.margin > 0.0 else raw_dist
        return opt_loss, raw_dist

    def _verify_decoder(self) -> None:
        """Fail fast at startup if this model's VAE decoder can't backprop on CPU (e.g. quantized
        or streaming no_grad decoders like MiniMax H3's); when GPU SparseID is requested, also
        verify CUDA backward once and fall back cleanly to CPU if a custom CUDA op lacks autograd."""
        shape = (1, self.latent_channels, 1, 2, 2) if self.latent_dim == 3 else (1, self.latent_channels, 2, 2)
        probe = torch.zeros(shape, dtype=torch.float32, requires_grad=True)
        try:
            with torch.enable_grad():
                rgb = self._decode_rgb(probe)
                if not rgb.requires_grad:
                    raise RuntimeError("VAE decoder output is detached")
                grad, = torch.autograd.grad(rgb.mean(), probe)
                if not torch.isfinite(grad).all():
                    raise RuntimeError("VAE decoder produced non-finite gradients")
        except Exception as e:
            self.restore_vae()
            raise ValueError(
                f"objective.face_id isn't supported with this model's VAE ({type(self.decoder).__name__}): "
                f"its decoder couldn't backpropagate on CPU ({e})"
            ) from e
        if self.device_pref == "gpu" and torch.cuda.is_available():
            dev = torch.device("cuda")
            try:
                probe_gpu = torch.zeros(shape, dtype=torch.float32, device=dev, requires_grad=True)
                with torch.enable_grad():
                    rgb = self._decode_rgb(probe_gpu, dev)
                    if not rgb.requires_grad:
                        raise RuntimeError("VAE decoder output is detached on CUDA")
                    grad, = torch.autograd.grad(rgb.mean(), probe_gpu)
                    if not torch.isfinite(grad).all():
                        raise RuntimeError("VAE decoder produced non-finite gradients on CUDA")
            except Exception as e:
                logging.warning("[ZenTrainer] SparseID GPU decoder couldn't backpropagate (%s); falling back to CPU", e)
                self.device_pref = "cpu"
            finally:
                self._move_to_cpu()
                torch.cuda.empty_cache()
        else:
            self.restore_vae()

    def _plan(self, path: Path, ft: FaceTarget, latent_hw: tuple[int, int]):
        key = (path, latent_hw)
        if key not in self._crop_cache:
            if len(self._crop_cache) >= MAX_CROP_CACHE:
                self._crop_cache.clear()
            self._crop_cache[key] = crop_plan(ft.orig_size, ft.points, latent_hw, self.stride)
        return self._crop_cache[key]

    def _crop_grad(self, z_crop: torch.Tensor, grid: torch.Tensor, target: torch.Tensor,
                   device: torch.device) -> tuple[torch.Tensor | None, float | None, float | None]:
        """Run crop forward + backward on `device` against a detached leaf so both forward and
        backward are covered by the GPU OOM fallback and the decoder/ArcFace can offload back to
        CPU before the DiT backward pass runs. Returns `(grad_crop, opt_val, raw_dist)`."""
        z_leaf = z_crop.detach().requires_grad_(True)
        opt_loss, raw_dist = self._eval_crop(z_leaf, grid, target, device)
        # Transfer both scalars in one sync to check finiteness and read the logged metric.
        vals = torch.stack((opt_loss.detach().float(), raw_dist.detach().float())).cpu().tolist()
        opt_val, raw_val = float(vals[0]), float(vals[1])
        if not (math.isfinite(opt_val) and math.isfinite(raw_val)):
            return None, None, None
        (grad_crop,) = torch.autograd.grad(opt_loss, z_leaf)
        return grad_crop.detach(), opt_val, raw_val

    def loss(self, tr, pred: torch.Tensor, sigma: torch.Tensor,
             idx: list[int]) -> tuple[torch.Tensor | None, float | None]:
        """Return `(weighted_loss_on_device, raw_cosine_distance)` for at most one eligible face
        in this micro-batch, or `(None, None)` when skipped."""
        if tr.step < self.start_step or (tr.step - self.start_step) % self.every != 0:
            return None, None
        if not isinstance(pred, torch.Tensor):
            return None, None
        if pred.ndim == 5 and pred.shape[2] != 1:
            return None, None  # multi-frame video clip
        if pred.ndim not in (4, 5):
            return None, None

        items = tr.dataset.items
        owners = tr.dataset.owners
        if not items or not owners:
            return None, None
        latent_hw = (int(pred.shape[-2]), int(pred.shape[-1]))
        noises = _effective_noise(tr.denoiser.model_sampling, sigma)

        picked = None
        for j, (i, noise) in enumerate(zip(idx, noises)):
            if not (SIGMA_MIN <= noise <= SIGMA_MAX):
                continue
            path = items[owners[i]].path
            ft = self.targets.get(path)
            if ft is None:
                continue
            plan = self._plan(path, ft, latent_hw)
            if plan is not None:
                picked = (j, ft, plan)
                break
        if picked is None:
            return None, None

        j, ft, (left, top, right, bottom, grid) = picked
        raw = tr.models.patcher.model.process_latent_out(pred[j : j + 1])
        z_crop = raw[..., top:bottom, left:right]
        dev = pred.device if (self.device_pref == "gpu" and pred.is_cuda) else torch.device("cpu")
        t0 = time.perf_counter() if not getattr(self, "_logged_first_crop", True) else 0.0
        if dev.type == "cuda":
            try:
                grad_crop, opt_val, raw_val = self._crop_grad(z_crop, grid, ft.target, dev)
            except torch.OutOfMemoryError:
                logging.warning("[ZenTrainer] SparseID GPU crop decode/backward ran out of VRAM; falling back to CPU")
                self._move_to_cpu()
                torch.cuda.empty_cache()
                self.device_pref = "cpu"
                dev = torch.device("cpu")
                grad_crop, opt_val, raw_val = self._crop_grad(z_crop, grid, ft.target, dev)
            finally:
                # Offload decoder + ArcFace back to CPU before DiT backward runs and so ComfyUI's
                # VAE patcher always finds it on CPU.
                self._move_to_cpu()
        else:
            grad_crop, opt_val, raw_val = self._crop_grad(z_crop, grid, ft.target, dev)
        if t0:
            self._logged_first_crop = True
            dt = getattr(self, "gpu_decode_dtype", self.decode_dtype) if dev.type == "cuda" else self.decode_dtype
            logging.info("[ZenTrainer] SparseID first crop (%s, %s, shape=%s): forward+backward %.0f ms",
                         dev.type, str(dt).replace("torch.", ""), tuple(z_crop.shape), (time.perf_counter() - t0) * 1000.0)
        if grad_crop is None or opt_val is None or raw_val is None:
            logging.warning("[ZenTrainer] face identity loss was non-finite at step %d; skipping", tr.step + 1)
            return None, None
        # Straight-through surrogate: exact scalar value (weight * opt_val) and exact gradient
        # (weight * grad_crop) w.r.t. z_crop, without keeping the VAE/ArcFace graph alive on GPU.
        z_f = z_crop.float()
        w_grad = grad_crop.to(device=pred.device, dtype=torch.float32) * self.weight
        val = torch.tensor(self.weight * opt_val, device=pred.device, dtype=torch.float32)
        surrogate = val + ((z_f - z_f.detach()) * w_grad).sum()
        return surrogate, raw_val
