"""The training loop. Owns the run from model load to final save.

ComfyUI is used as a library: its loaders, ModelPatcher and memory management, weight
adapters, attention backends, model_sampling and samplers. Nothing here goes through
the node graph. The only graph involvement is the ZenTrainSession node that hosts
the session, so a run holds ComfyUI's execution lane (and its GPU) while it trains.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .. import events, paths, vram
from ..spec import TrainSpec, slugify
from . import control as control_mod
from . import data, models, optim
from .callbacks import Callback, Checkpoints, Keep, Metrics, Samples, Validation, write_metadata
from . import compat
from .data.streams import Streams
from .denoiser import Denoiser
from .memory import attention_override, gradient_checkpointing
from . import network as network_mod
from .network import Network
from .objective import Objective, is_flow, sample_percents, scheduler_sigmas, sigmas_for

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _has_soundtracks(root, look: int = 6) -> bool:
    """Whether the dataset's clips carry audio (the first few looked at)."""
    from .data import video

    clips = [it.path for it in data.scan(root, warn=False) if video.is_video(it.path)][:look]
    for p in clips:
        try:
            if video.read_audio(p, 0.0, 0.5) is not None:
                return True
        except Exception:
            continue
    return False


def _release_frames(e: BaseException) -> None:
    """Drop the locals held by a failed run's traceback (and what it was raised from). ComfyUI
    keeps the exception as the job's error, and its frames hold the failed step's activations
    and gradients: after an out-of-VRAM error that's GBs the next run then can't load into
    (measured: a resumed H3 run got 4.9 of its 20 GB onto the GPU). The message and line
    numbers stay."""
    import traceback

    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if e.__traceback__ is not None:
            traceback.clear_frames(e.__traceback__)
        e = e.__cause__ or e.__context__


def _oom_advice(spec: TrainSpec, step: int) -> str:
    m, d, o = spec.memory, spec.dataset, spec.optim
    tips = []
    if not m.gradient_checkpointing:
        tips.append("turn on gradient checkpointing")
    elif m.checkpoint_skip:
        tips.append(f"keep fewer blocks un-checkpointed ({m.checkpoint_skip} -> 0)")
    if not m.offloading:
        tips.append("turn on weight offloading")
    if d.resolution > 768:
        tips.append(f"lower resolution ({d.resolution} -> {d.resolution * 3 // 4 // 64 * 64})")
    if d.video_frames > 22:
        tips.append(f"shorter video clips ({d.video_frames} frames) or a lower video resolution ({d.video_resolution})")
    if m.offloading:
        tips.append("more VRAM headroom (Memory -> VRAM headroom, in GB)")
    if o.batch_size > 1:
        tips.append(f"batch size {o.batch_size} -> 1 (use grad accumulation instead)")
    if m.training_dtype == "fp32":
        tips.append("train in bf16")
    where = f"at step {step}" if step else "on the first step"
    return f"Out of VRAM {where}. Try: " + ("; ".join(tips) or "a smaller rank or fewer target layers") + "."


def state_dir(run_id: str) -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "states" / run_id


def lora_root(lora_dir: str | None) -> tuple[Path, Path]:
    """(the ComfyUI loras folder LoRA names are relative to, the folder runs save under).
    A chosen folder must be one ComfyUI lists for loras: names stay loadable, and a spec
    can't point writes anywhere else."""
    import folder_paths

    default = paths.lora_dir()
    if not lora_dir:
        return default, default
    want = Path(lora_dir).resolve()
    for p in folder_paths.get_folder_paths("loras"):
        if Path(p).resolve() == want:
            return want, want / "zentrainer"
    raise ValueError(f"output.lora_dir {lora_dir!r} isn't one of ComfyUI's loras folders")


def state_for(checkpoint: dict, run_id: str) -> str | None:
    """The training-state file of a checkpoint row, if it exists. Older checkpoints kept
    it beside the LoRA as <name>.state.pt."""
    for cand in (checkpoint.get("state"), str(Path(checkpoint["path"]).with_suffix(".state.pt"))):
        if cand and Path(cand).is_file():
            return cand
    return None


def _comfy_waiting() -> bool:
    """Another ComfyUI prompt is queued behind the one running (this run's): someone wants the GPU."""
    try:
        import server

        return server.PromptServer.instance.prompt_queue.get_tasks_remaining() > 1
    except Exception:  # outside ComfyUI, or its queue changed shape: never park on a guess
        return False


class Stop(Exception):
    def __init__(self, status: str):
        self.status = status


_compiler_default: bool | None = None


def _comfy_compiler(on: bool) -> None:
    """ComfyUI's memory compiler (comfy_aimdo malloc graphs) records one forward's allocations
    and replays them on the next. That suits repeated inference, not training: autograd keeps
    tensors past each block and bucket shapes change, and the replay fails with "aimdo memory
    compile error". ComfyUI only resets it between nodes, and a whole run is one node, so it's
    off for the whole run (samples included) and back to the launch setting at the end."""
    global _compiler_default
    import comfy.model_prefetch
    from comfy.cli_args import args

    if _compiler_default is None:
        _compiler_default = args.disable_comfy_compiler
    comfy.model_prefetch.cleanup_malloc_graph()
    args.disable_comfy_compiler = _compiler_default if on else True


# VRAM left for the LoRA, optimizer state, activations and ComfyUI itself when the base model
# loads whole; below this the run offloads weights instead (see Trainer._auto_offload)
AUTO_OFFLOAD_HEADROOM = 8 * 1024 ** 3
# video clips are many tokens per step (73 frames at 512 px: ~5k; a 1024 px image on H3: ~1k):
# activations grow with them, so the headroom does too, beyond this many tokens
HEADROOM_TOKENS = 2048
HEADROOM_PER_TOKEN = int(1.0 * 1024 ** 2)  # measured: 73 frames at 512 (~5.3k tokens) peaked ~7 GB of activations
# that estimate is safe, not tight (Krea 2 at 1024 px needs ~5-6 GB beside its weights, and was
# given 9.9): after these steps the measured peak decides, and every spare byte but the margin
# holds model weights. Later passes correct the first; an out-of-VRAM step backs off by the margin
TIGHTEN_AT = (4, 12, 40)
TIGHTEN_MARGIN = 1024 ** 3
TIGHTEN_RETRIES = 3
# each fit is a trial: measured over these steps after it (the recompile it may cause is over by
# then), it stays only if steps didn't get slower than this and the allocator never had to retry.
# A card run nearly full can be slower than one streaming more weights. Steps are compared with
# steps of the same shape before the fit: H3 clips of different lengths took 1.4 to 3 s a step,
# and raw times "measured" every fit as slower. No same-shaped pair by TRIAL_LONGEST: kept.
TRIAL_STEPS = (4, 7)
TRIAL_LONGEST = 16
TRIAL_SLOWER = 1.08

# the run page's phase timings (data, forward, backward, optimizer) come from every this-many-th
# step: timing them stalls the GPU at each phase's end
PHASES_EVERY = 10


def trial_ratio(before: list, after: list) -> float | None:
    """How much slower steps got after a fit: the median of after/before over step shapes seen in
    both (each a median of its steps), or None when no shape is in both. Entries are
    (warm step, seconds, shape)."""
    def by_shape(rows):
        out: dict = {}
        for _, sec, key in rows:
            if key is not None:
                out.setdefault(key, []).append(sec)
        return {k: sorted(v)[len(v) // 2] for k, v in out.items()}

    b, a = by_shape(before), by_shape(after)
    ratios = sorted(a[k] / b[k] for k in a.keys() & b.keys() if b[k] > 0)
    return ratios[len(ratios) // 2] if ratios else None


class _Lazy:
    """Stands in for an object until it's actually used (the text encoder, for a cache hit)."""

    def __init__(self, get):
        self._get = get

    def __getattr__(self, name):
        return getattr(self._get(), name)


class Trainer:
    def __init__(self, spec: TrainSpec, run_id: str, *, store=None, emit: Callable = events.emit,
                 control: control_mod.Control | None = None, callbacks: list[Callback] | None = None,
                 resume_from: str | None = None, resume_state: str | None = None, sample_job: dict | None = None):
        self.spec = spec
        # a finished run's checkpoint to sample instead of training (routes: POST .../sample):
        # {"path", "step", "run_prompts", "tries", "prev_status", "prev_message"}
        self.sample_job = sample_job
        self.resume_from = resume_from  # a checkpoint .safetensors
        self.resume_state = resume_state  # its training state (see state_for)
        self.run_id = run_id
        self.store = store
        self.emit = emit
        self.control = control or control_mod.Control()
        self.callbacks = callbacks if callbacks is not None else [
            Metrics(), Checkpoints(spec.save_every), Samples(spec.samples.every), Validation(), Keep()]
        self.step = 0
        self.train_seconds = 0.0  # time spent in optimizer steps only (no sampling/validation/saves/pauses)
        self.last_step_seconds: float | None = None
        self.last_loss: float | None = None
        self.last_grad_norm: float | None = None
        self._samples = next((c for c in self.callbacks if isinstance(c, Samples)), None)

    # ---- lifecycle -------------------------------------------------------------

    def run(self) -> str:
        """Train to completion (or stop). Returns the final status."""
        import comfy.model_management as mm

        if self.sample_job:
            return self._sample_only()

        status = "error"
        with torch.inference_mode(False):  # ComfyUI executes nodes under inference_mode
            try:
                self._status("loading")
                self.setup()
                self._status("training")
                from .kernels.compile import compiled_blocks, wanted

                dm = self.models.patcher.model.diffusion_model
                compile_on, why = wanted(self.spec.memory.compile_blocks, self.spec.optim.steps)
                if why:
                    logging.info("[ZenTrainer] %s", why)
                if compile_on and compat.fused_qk_rope:
                    # compile fuses the norms and rope itself, and the custom op would keep it from
                    # fusing what's around them: Krea 2 at 1024, 1.38 s a step compiled, 1.42 with it
                    compat.fused_qk_rope = False
                    logging.info("[ZenTrainer] fused q/k norm + rope off while the blocks run compiled: they fuse them already")
                # compiled first, so checkpointing wraps the compiled forward
                with compiled_blocks(dm, compile_on) as self._compiled, \
                        gradient_checkpointing(dm, self.spec.memory.gradient_checkpointing,
                                               self.spec.memory.checkpoint_depth, self.spec.memory.offloading,
                                               self.spec.memory.checkpoint_skip) as ckpt:
                    self._input_grad = ckpt.input_grad
                    self.enter_training()
                    try:
                        for cb in self.callbacks:
                            cb.on_start(self)
                        self._drop_text_encoder()
                        # on_start may have loaded the text encoder (sample prompts) and the VAE
                        # (baseline samples), pushing the model out: back in before step one
                        self.enter_training()
                        status = self.loop()
                    except Stop as s:
                        status = s.status
                    except torch.OutOfMemoryError as e:
                        status = "error"
                        _release_frames(e)
                        raise RuntimeError(_oom_advice(self.spec, self.step)) from None
                    except BaseException as e:
                        status = "interrupted" if type(e).__name__ == "InterruptProcessingException" else "error"
                        raise
                    finally:
                        mm.in_training = False
                        for cb in self.callbacks:
                            try:
                                cb.on_end(self, status)
                            except Exception:
                                logging.exception("[ZenTrainer] callback on_end failed")
                        self.network.eject(self.models.patcher)
                        self.models.patcher.cleanup()
                        _comfy_compiler(True)  # after on_end: its final samples run without it too
                        from .memory.offload import guard_backward, overlap, watch_pins

                        overlap(False)
                        guard_backward(False)
                        watch_pins(False)
                        if getattr(self, "_compiled", 0):
                            torch.compiler.set_stance("default")  # the final samples left it eager
            except BaseException as e:
                interrupted = type(e).__name__ == "InterruptProcessingException"
                self._status("interrupted" if interrupted else "error", message=str(e) or type(e).__name__)
                _release_frames(e)
                raise
            finally:
                mm.training_fp8_bwd = False
                from .kernels import int8_backward

                int8_backward.uninstall()  # the model is cached: leave its layers as they were
                self._release()
                mm.soft_empty_cache()
        self._status(status, final_step=self.step)
        return status

    def _sample_only(self) -> str:
        """Sample a finished run's checkpoint: its LoRA loaded, the run's prompts (and any tries)
        rendered at the checkpoint's step as if training had sampled there, so they join the
        samples, the likeness line and the progression; then everything is given back. Nothing
        trains, and the run keeps the status it had."""
        import comfy.model_management as mm

        job = self.sample_job
        from .kernels import int8_backward

        samples = self._samples
        out = "error"
        with torch.inference_mode(False):
            try:
                self._status("sampling", message=f"sampling step {job['step']}")
                self.setup()
                self.step = int(job["step"])
                self.enter_training()  # loads the model and attaches the LoRA; sampling switches from there
                samples.on_start(self)  # encodes the prompts (no baseline: the step isn't 0)
                self._drop_text_encoder()
                if job.get("run_prompts", True):
                    samples.render(self)
                if job.get("tries"):
                    samples.render_manual(self, list(job["tries"]))
                out = "done"
            except BaseException as e:
                logging.exception("[ZenTrainer] sampling %s at step %s failed", self.run_id, job.get("step"))
                out = f"sampling failed: {e or type(e).__name__}"
                raise
            finally:
                mm.in_training = False
                try:
                    samples.on_end(self, "done")  # the likeness worker finishes what's queued
                except Exception:
                    logging.exception("[ZenTrainer] samples on_end failed")
                if getattr(self, "network", None) is not None and hasattr(self, "models"):
                    self.network.eject(self.models.patcher)
                    self.models.patcher.cleanup()
                _comfy_compiler(True)
                mm.training_fp8_bwd = False
                int8_backward.uninstall()
                self._release()
                mm.soft_empty_cache()
                prev = job.get("prev_status") or "done"
                msg = job.get("prev_message") if out == "done" else out
                self.emit(events.STATUS, {"run_id": self.run_id, "status": prev, "message": msg, "step": self.step})
                self.emit(events.ZENKIT_JOB, {"id": f"zentrainer:{self.run_id}", "name": f"Sample {self.spec.name}",
                                              "status": "done" if out == "done" else "error", "current": 1, "total": 1,
                                              "message": "sampled" if out == "done" else out, "source": "ZenTrainer"})
                if self.store:
                    self.store.update(self.run_id, status=prev, message=msg, sample_job=None)
        return prev

    def _release(self) -> None:
        """Give back what a run holds on the GPU (finished or failed): the optimizer state, the
        adapter, cached conds; the next run starts on a clean card."""
        import gc

        for name in ("optimizer", "scheduler", "ema", "network", "denoiser", "_previewer"):
            if hasattr(self, name):
                setattr(self, name, None)
        conds = getattr(self, "_conds", None)
        if isinstance(conds, dict):
            conds.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def setup(self) -> None:
        import comfy.model_management as mm
        import comfy.utils
        import folder_paths

        s = self.spec
        compat.apply(qk_rope=s.memory.fused_qk_rope is not False)  # ComfyUI model fixes for training (see compat/)
        from .kernels import int8_backward

        int8_backward.enable("w8a8" if s.memory.w8a8_backward else "fused" if s.memory.fused_int8_backward else "")
        lora_root(s.output.lora_dir)  # a bad save folder fails now, not at the first save
        torch.manual_seed(s.seed)
        random.seed(s.seed)
        self.generator = torch.Generator().manual_seed(s.seed)

        self.models = models.load(s.model)
        patcher = self.models.patcher
        compat.prune_unused(patcher)
        self._auto_offload(patcher)
        self._offload_skip()
        from .kernels import int8_backward

        # int8 layers that train: the int8 forward (activations quantized), or with int8_forward off
        # a dequantized bf16 one
        if s.memory.int8_forward:
            n8 = int8_backward.install(patcher.model)
            if n8:
                logging.info("[ZenTrainer] int8 training path on %d layers", n8)
        elif n8 := int8_backward.weight_only(patcher.model):
            logging.info("[ZenTrainer] int8 weights dequantized to bf16 for every matmul on %d layers", n8)
        self.device = patcher.load_device
        self.train_dtype = _DTYPES[s.memory.training_dtype]

        encoder_id = repr((s.model.kind, s.model.ckpt_name, s.model.clip_name, s.model.clip_type, s.model.vae_name))
        root = paths.find_dataset(s.dataset.folder)  # where it is now, or where an earlier setting kept it
        cache = Path(folder_paths.get_user_directory()) / "zentrainer" / "cache"
        self.encoder_id, self.cache_dir = encoder_id, cache
        if s.dataset.video_audio and self.models.audio_vae is None:
            raise ValueError("training on video sound needs the model's audio VAE: set Model -> Audio VAE")
        from .data.edit import is_h3

        self.h3 = is_h3(patcher.model)
        if s.dataset.keyframes and not self.h3:
            raise ValueError("keyframes (image-to-video training) are for MiniMax H3's fl2va model")
        self.dataset = data.build(self._data_spec(s.dataset), _Lazy(lambda: self.models.clip), self.models.vae, root, cache,
                                  encoder_id, progress=lambda i, n, msg: self._status("loading", message=f"{msg} ({i}/{n})"),
                                  audio_vae=self.models.audio_vae, h3=self.h3)
        if s.slider.kind == "pairs":
            self._keep_pairs(self.dataset)
        if s.slider.kind == "text":  # its prompts, encoded while the text encoder is here (and cached like captions)
            self._slider_text = self._encode_texts(self._slider_prompts())
            self._slider_conds: dict = {}
        self.edit_method = None
        self.concat_inputs = 0  # channel inputs: how many input latents ride beside the noisy one
        if s.dataset.edit and s.dataset.edit_method == "concat":
            from .wide_input import input_layer

            if input_layer(patcher.model) is None:
                raise ValueError(f"channel inputs widen the model's input layer, and {type(patcher.model).__name__}'s "
                                 "can't be (MOSS Image's can): use references for this model")
            if self.h3:
                raise ValueError("channel inputs aren't for MiniMax H3: use references")
            pairs = sum(sm.has_refs if isinstance(sm, data.DiskSample) else any(r is not None for r in sm.refs or [])
                        for sm in self.dataset.samples)
            if not pairs:
                raise ValueError(f"edit training, but no picture has an input in the control set(s) "
                                 f"{', '.join(s.dataset.control_sets)}")
            self.concat_inputs = len(s.dataset.control_sets)
            logging.info("[ZenTrainer] channel inputs: %d of %d pictures have one; %d input slot(s) (%s)",
                         pairs, len(self.dataset.samples), self.concat_inputs, ", ".join(s.dataset.control_sets))
        elif s.dataset.edit:
            from .data.edit import method, takes_references

            if not takes_references(patcher.model):
                from .wide_input import input_layer

                if input_layer(patcher.model) is not None:
                    raise ValueError(f"{type(patcher.model).__name__} takes its inputs as channels, not references: "
                                     "set Inputs join as to Channels (dataset.edit_method \"concat\")")
                raise ValueError(f"{type(patcher.model).__name__} doesn't take reference images, so it can't train "
                                 "edits: pick an edit-capable model (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit)")
            pairs = sum(data.has_inputs(sm) for sm in self.dataset.samples)
            if not pairs:
                raise ValueError(f"edit training, but no picture has an input in the control set(s) "
                                 f"{', '.join(s.dataset.control_sets)}")
            self.edit_method = None if self.h3 else method(patcher.model, s.dataset.edit_method)
            from .moss_refs import supports

            if supports(patcher.model):  # ComfyUI's MOSS reads no references: ZenTrainer's patch adds them
                from . import moss_refs

                moss_refs.apply(patcher)
            logging.info("[ZenTrainer] edit training: %d of %d images have an input picture (references: %s)",
                         pairs, len(self.dataset.samples), self.edit_method or "the model's own")
        if self.h3 and not s.dataset.video_audio and _has_soundtracks(root):
            msg = ("these clips have sound, but 'Train on sound' is off: the model sees silence beside every clip, "
                   "and the LoRA drifts the audio with nothing holding it (measured: the beat about halves). "
                   "Turn on Train on sound")
            logging.warning("[ZenTrainer] %s", msg)
            self._status("loading", message=msg)
        masked = sum(data.has_mask(sm) for sm in self.dataset.samples)
        if masked and not s.dataset.masked_loss:
            msg = (f"{masked} picture{'s have masks' if masked != 1 else ' has a mask'}, but 'Masked training' is "
                   "off: the masks are ignored and every part of those pictures trains, the parts they "
                   "leave out included. To honour them, stop and turn on Masked training (Dataset)")
            logging.warning("[ZenTrainer] %s", msg)
            self._status("loading", message=msg)
        if s.memory.tread > 0 and s.memory.tread_steps >= 1.0 and self.dataset.samples and all(
                sm.latent.ndim == 4 and sm.latent.shape[1] > 1 for sm in self.dataset.samples):
            msg = ("TREAD routes every step: this dataset is all clips, with no stills run whole to anchor the "
                   "LoRA, so it can learn to make up for the routing (over-sharpened samples). Consider "
                   "'Share of steps routed' around 0.5, or adding stills")
            logging.warning("[ZenTrainer] %s", msg)
            self._status("loading", message=msg)
        self.batches = data.Batches(self.dataset, s.optim.batch_size, s.seed)
        # lower-resolution stages, each cached now: (until, resolution, dataset, batches)
        from dataclasses import replace

        self._stages = []
        for st in s.dataset.stages:
            res = int(st["resolution"])
            ds = data.build(self._data_spec(replace(s.dataset, resolution=res, stages=[])), _Lazy(lambda: self.models.clip),
                            self.models.vae, root, cache, encoder_id, audio_vae=self.models.audio_vae, h3=self.h3,
                            progress=lambda i, n, msg, r=res: self._status("loading", message=f"{msg} at {r} px ({i}/{n})"))
            if s.slider.kind == "pairs":
                self._keep_pairs(ds)
            self._stages.append((float(st["until"]), res, ds, data.Batches(ds, s.optim.batch_size, s.seed + res)))
        self._final_stage = (s.dataset.resolution, self.dataset, self.batches)
        self._res_seen = 0
        if self._stages:
            plan = ", ".join(f"{r} px until {u:.0%}" for u, r, _, _ in self._stages)
            logging.info("[ZenTrainer] resolution: %s, then %d px; each change blends over %.0f%% of the run",
                         plan, s.dataset.resolution, s.dataset.stage_blend * 100)

        if s.memory.training_dtype != "fp32":
            if any(getattr(m, "layout_type", None) is not None for m in patcher.model.modules()):
                # a quantized base: only the cast dtype. ComfyUI's set_model_compute_dtype also
                # forces every layer to cast its weights, which turns the int8/fp8 matmuls off
                # (each layer dequantized to bf16 every pass): measured on H3 int8, every
                # forward matmul ran in bf16. Also clears the flag a previous run left on the
                # cached model.
                import uuid

                patcher.add_object_patch("manual_cast_dtype", self.train_dtype)
                patcher.force_cast_weights = False
                patcher.patches_uuid = uuid.uuid4()  # the next load re-applies the per-layer flags
            else:
                patcher.set_model_compute_dtype(self.train_dtype)
        mm.training_fp8_bwd = s.memory.quantized_backward

        init = None
        if self.resume_from:
            init = comfy.utils.load_torch_file(self.resume_from)
        elif self.sample_job:
            init = comfy.utils.load_torch_file(self.sample_job["path"])
        elif s.network.init_from:
            init = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", s.network.init_from))
        # A weight wrapper on a quantized layer turns off its quantized matmul (every forward
        # would dequantize the full weight), so quantized bases train in bypass mode.
        quantized = any(getattr(m, "layout_type", None) is not None for m in patcher.model.modules())
        bypass = s.memory.bypass_mode or quantized
        if quantized and not s.memory.bypass_mode:
            self._status("loading", message="quantized base model: training in bypass mode")
        assist = None
        if s.network.training_adapter:
            left_out: list = []
            assist = network_mod.load_training_adapter(
                patcher, folder_paths.get_full_path_or_raise("loras", s.network.training_adapter), self.device,
                torch.bfloat16, left_out)
            self._partial(s.network.training_adapter, "the training adapter", len(assist), left_out)
            logging.info("[ZenTrainer] training adapter %s on %d layers (off for samples and the saved LoRA)",
                         s.network.training_adapter, len(assist))
        preview = None
        if s.samples.turbo_lora:
            left_out = []
            preview = network_mod.load_training_adapter(
                patcher, folder_paths.get_full_path_or_raise("loras", s.samples.turbo_lora), self.device, torch.bfloat16,
                left_out)
            self._partial(s.samples.turbo_lora, "the turbo LoRA", len(preview), left_out)
            logging.info("[ZenTrainer] samples ride on %s at %.2f", s.samples.turbo_lora, s.samples.turbo_strength)
        base = None
        if s.network.base_lora:
            left_out = []
            base = network_mod.load_training_adapter(
                patcher, folder_paths.get_full_path_or_raise("loras", s.network.base_lora), self.device, torch.bfloat16,
                left_out)
            self._partial(s.network.base_lora, "the base LoRA", len(base), left_out)
            logging.info("[ZenTrainer] base LoRA %s at %.2f on %d layers (training and samples; saved beside, not in)",
                         s.network.base_lora, s.network.base_strength, len(base))
        self.network = Network(patcher, s.network, self.device, bypass=bypass, init_weights=init, assist=assist,
                               preview=preview, preview_strength=s.samples.turbo_strength,
                               base=base, base_strength=s.network.base_strength, wide_inputs=self.concat_inputs,
                               scalable=bool(s.slider.kind))
        self.optimizer, self.scheduler = optim.build(
            self.network.param_groups(s.optim.learning_rate, s.optim.lora_plus), s.optim)
        from .ema import EMA

        self.ema = EMA(self.network.params, s.optim.ema) if s.optim.ema > 0 else None
        self.objective = Objective(s.objective)
        self.denoiser = Denoiser(patcher, attention_override(s.memory.attention))
        self.streams = Streams(patcher.model)
        if self.streams.multi and s.optim.batch_size > 1:
            raise ValueError(f"{type(patcher.model).__name__} trains one image per step: set batch size 1 "
                             f"and use gradient accumulation {s.optim.batch_size} instead")
        self._shift = s.objective.shift
        if self._shift and not is_flow(self.denoiser.model_sampling):
            raise ValueError(f"objective.shift is for flow-matching models; {type(self.denoiser.model_sampling).__name__} isn't one. Set it to 0.")
        self._schedule = (scheduler_sigmas(self.denoiser.model_sampling, s.objective.scheduler, s.objective.scheduler_steps)
                          if s.objective.timesteps == "scheduler" else None)
        if s.objective.weighting == "flex" and not is_flow(self.denoiser.model_sampling):
            raise ValueError(f"objective.weighting is for flow-matching models; {type(self.denoiser.model_sampling).__name__} isn't one. Turn it off.")
        self.scaler = torch.amp.GradScaler() if self.train_dtype == torch.float16 else None

        self._conds: dict[tuple[int, tuple], dict] = {}  # (caption, shape) -> prepared cond
        self.progress = comfy.utils.ProgressBar(s.optim.steps)
        if self.resume_from:
            self._restore(Path(self.resume_state or state_for({"path": self.resume_from}, self.run_id) or ""))

    def _saved_optimizer_state(self, saved: dict) -> dict:
        """States saved before LoRA alphas stopped training numbered every adapter tensor; now
        alphas are frozen and left out. Renumber those by name (LoRA+ splits the up matrices into
        a second group, as it did then)."""
        groups = self.optimizer.param_groups
        if [len(g["params"]) for g in saved["param_groups"]] == [len(g["params"]) for g in groups]:
            return saved
        names = {id(p): k for k, p in self.network.params.items()}
        new = [[names[id(p)] for p in g["params"]] for g in groups]
        everything = list(self.network.params)
        if len(new) == 2:  # LoRA+: (rest, up)
            old = [[k for k in everything if ".lora_up." not in f"{k}."], [k for k in everything if ".lora_up." in f"{k}."]]
        else:
            old = [everything]
        if [len(g) for g in old] != [len(g["params"]) for g in saved["param_groups"]]:
            return saved  # not that layout: let load_state_dict say what doesn't match
        logging.info("[ZenTrainer] optimizer state from before alphas were frozen: renumbered")
        return optim.remap_state(saved, old, new)

    def _restore(self, state_path: Path) -> None:
        """Continue from a checkpoint: adapters were loaded from its LoRA file above;
        this brings back optimizer moments, the LR schedule position, step and RNG."""
        state = torch.load(str(state_path), map_location="cpu", weights_only=False)  # our own file
        self.optimizer.load_state_dict(self._saved_optimizer_state(state["optimizer"]))
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])
        if self.objective.balance is not None and state.get("balance"):  # its loss averages carry on
            self.objective.balance.load_state_dict(state["balance"])
        if state.get("live"):  # the checkpoint's LoRA file holds the average; training goes on from these
            with torch.no_grad():
                for k, v in state["live"].items():
                    if k in self.network.params:
                        self.network.params[k].copy_(v.to(self.network.params[k]))
        if self.ema is not None and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.generator.set_state(state["rng"])
        self.batches.epoch = int(state.get("batches_epoch", 0))
        self.train_seconds = float(state.get("train_seconds", 0.0))
        self.batches.rng.seed(self.spec.seed + self.step)  # fresh shuffle order from here
        self.progress.update_absolute(self.step)
        self._status("loading", message=f"resumed from step {self.step}")

    def _drop_text_encoder(self) -> None:
        """Captions are encoded into the dataset cache and the sample prompts at on_start: past
        that the text encoder is never used, so don't hold it in RAM for the whole run (a few
        GB for most models, tens for the big VLM encoders)."""
        import comfy.model_management as mm

        if self.models._clip is None:  # never loaded (everything was cached)
            return
        self.models.release_clip()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        logging.info("[ZenTrainer] text encoder released: captions and sample prompts are encoded")

    @property
    def headroom(self) -> int:
        """Bytes of VRAM kept free beside the (partly) loaded model: the spec's, or 8 GB plus
        room for the dataset's longest clip. Only the first load uses it: then _tighten measures."""
        if self.spec.memory.vram_headroom > 0:
            return int(self.spec.memory.vram_headroom * 1024 ** 3)
        tokens = 0
        for smp in getattr(getattr(self, "dataset", None), "samples", [])[:2000]:
            lat = smp.latent
            if lat.ndim == 4:  # (C, T, h, w): a video clip; images keep the flat default
                tokens = max(tokens, lat.shape[1] * lat.shape[2] * lat.shape[3] // 4)
        return AUTO_OFFLOAD_HEADROOM + max(0, tokens - HEADROOM_TOKENS) * HEADROOM_PER_TOKEN

    def _auto_offload(self, patcher) -> None:
        """Offloading off means the whole base model sits in VRAM. When it can't (its weights
        alone leave too little room for the LoRA, optimizer and activations), turn offloading
        on rather than fail at step one: the run is slower, not broken."""
        m = self.spec.memory
        if m.offloading or not torch.cuda.is_available():
            return
        weights = patcher.model_size()
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        if weights > total - self.headroom:
            m.offloading = True
            gb = 1024 ** 3
            logging.info("[ZenTrainer] weight offloading on: the model is %.1f GB on a %.1f GB card", weights / gb, total / gb)
            self._status("loading", message=f"Offloading weights: the model ({weights / gb:.0f} GB) doesn't fit beside training in {total / gb:.0f} GB")

    def _offload_skip(self) -> None:
        """Kept (un-checkpointed) last blocks trade VRAM for speed, but with weights offloaded
        that VRAM holds weights instead, and is worth as much: measured on Krea 2 int8 at 1024 px,
        16 GB card, keeping 6 blocks cost 2 GB of weights on the card for no faster steps."""
        m = self.spec.memory
        if m.offloading and m.gradient_checkpointing and m.checkpoint_skip > 0:
            logging.info("[ZenTrainer] keep last blocks %d -> 0: with weights offloaded, that VRAM holds weights instead", m.checkpoint_skip)
            m.checkpoint_skip = 0

    def enter_training(self) -> None:
        """Load for training: everything else out of VRAM, model in train mode."""
        import comfy.model_management as mm

        patcher = self.models.patcher
        if self.spec.memory.offloading:
            from .memory.offload import watch_pins

            watch_pins(True)  # before loading: ComfyUI pins the offloaded weights as it loads them
            if not getattr(self, "_own_ram", False) and torch.cuda.is_available():
                from .memory.offload import own_ram

                self._own_ram = True  # once: what loads later is already in RAM
                n, size = own_ram(patcher.model.diffusion_model)
                if n:
                    logging.info("[ZenTrainer] the model was in a file mapping CUDA can't pin: %d weights (%.1f GB) "
                                 "copied to RAM, so ComfyUI pins the offloaded ones", n, size / 1024 ** 3)
            # everything else out, then as much of the model as fits beside training stays in
            # VRAM; only the rest streams. (Asking for "all the memory" here, as before, left a
            # weight budget of nothing: every layer crossed PCIe twice a step.)
            ours = [lm for lm in mm.current_loaded_models if lm.model is patcher]
            mm.free_memory(1e30, self.device, keep_loaded=ours)
            # back from samples, the cache is in the sampler's fragments: weights loaded between
            # them left too little room for the step that fit before (Krea 2 int8: out of VRAM on
            # the first step after every sample pass, until the retries ran out)
            torch.cuda.empty_cache()
            logging.info("[ZenTrainer] keeping %.1f GB of VRAM free for training beside the model", self.headroom / 1024 ** 3)
            mm.load_models_gpu([patcher], memory_required=self.headroom, force_full_load=False)
            self._fit_weights()
            self._pin_offloaded()
            if not getattr(self, "_first_resident", None):
                self._first_resident = patcher.loaded_size()  # the safe estimate's load, to return to
        else:
            mm.load_models_gpu([patcher], memory_required=1e20, force_full_load=True)
        patcher.model.requires_grad_(False)
        patcher.model.train()
        patcher.pre_run()
        self.network.inject(patcher)
        if hasattr(self, "optimizer"):
            optim.set_mode(self.optimizer, training=True)
        mm.in_training = True
        _comfy_compiler(False)
        if self.spec.memory.offloading:
            from .memory.offload import guard_backward, overlap

            overlap(True)
            guard_backward(True)
        if getattr(self, "_compiled", 0):
            torch.compiler.set_stance("default")
        self._recompile_if_moved()

    def enter_inference(self) -> None:
        import comfy.model_management as mm

        mm.in_training = False
        from .memory.offload import guard_backward, overlap, watch_pins

        overlap(False)
        guard_backward(False)
        watch_pins(False)  # samples load models of their own: ComfyUI's streams as they are
        # the compiler stays off for samples too: models that drive it from their own forward
        # (MiniMax H3's malloc graphs) abort the process on a partly offloaded, LoRA-wrapped
        # model. It's back to the launch setting when the run ends.
        optim.set_mode(self.optimizer, training=False)
        self.models.patcher.model.eval()
        if getattr(self, "_compiled", 0):
            # samples run the compiled blocks eager: no grads and other shapes would only add
            # graphs, and inference may take kernels dynamo can't trace (MOSS int8's Sage
            # attention: "RuntimeError when making fake tensor call", the run's first sample)
            torch.compiler.set_stance("force_eager")

    def exit_inference(self) -> None:
        # sampling may have loaded the VAE/text encoder and pushed the model out
        self.enter_training()

    # ---- the loop --------------------------------------------------------------

    def loop(self) -> str:
        total = self.spec.optim.steps
        self._peak = 0  # the most VRAM PyTorch held during training steps (callbacks reset its own counter)
        self._warm = 0
        self._times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        while self.step < total:
            self._handle_commands()
            t0 = time.perf_counter()
            if getattr(self, "_profile_next", False):
                self._profile_next = False
                self._profiled_step()
            else:
                final = getattr(self, "_final_stage", None) or (self.spec.dataset.resolution, self.dataset, self.batches)
                res, ds, batches = self._stage() if getattr(self, "_stages", None) else final
                if res > getattr(self, "_res_seen", 0):
                    if getattr(self, "_res_seen", 0):
                        self._new_resolution(res)
                    self._res_seen = res
                self.dataset, self.batches = ds, batches
                try:
                    ts = time.perf_counter()
                    self._train_step()
                    # (warm step, seconds, the step's shape): fits compare steps of the same shape
                    self._times = (getattr(self, "_times", []) +
                                   [(self._warm, time.perf_counter() - ts, getattr(self, "_step_key", None))])[-24:]
                    self._backoffs = 0  # the retries count in a row: a long run's sample passes don't use them up
                except torch.OutOfMemoryError as e:
                    if not self._back_off(e):
                        raise
                    continue  # that step again, with fewer weights on the card
                finally:
                    self.dataset = final[1]  # validation and samples use the run's own resolution
            if torch.cuda.is_available():
                self._peak = max(self._peak, torch.cuda.max_memory_reserved())
                self._warm += 1
                trial = getattr(self, "_trial", None)
                if trial and self._warm >= trial["at"] + TRIAL_STEPS[1]:
                    self._judge()
                if self._warm in TIGHTEN_AT:
                    self._tighten()
            self.last_step_seconds = time.perf_counter() - t0
            self.train_seconds += self.last_step_seconds
            self.step += 1
            self.progress.update_absolute(self.step)  # also raises on ComfyUI interrupt
            for cb in self.callbacks:
                cb.on_step(self)
        return "done"

    def _stage(self):
        """(resolution, dataset, batches) for this step: the stage the run is in, or while a change
        blends, the next one with a probability that rises across the blend (seeded by the step,
        so a resume picks the same)."""
        if not self._stages:
            return self._final_stage
        f = self.step / max(1, self.spec.optim.steps)
        bounds = [u for u, _, _, _ in self._stages]
        levels = [(r, ds, b) for _, r, ds, b in self._stages] + [self._final_stage]
        k = sum(1 for u in bounds if f >= u)
        blend = self.spec.dataset.stage_blend
        if k and blend > 0 and f < bounds[k - 1] + blend:
            if random.Random(self.spec.seed * 1000003 + self.step).random() >= (f - bounds[k - 1]) / blend:
                k -= 1
        return levels[k]

    def _new_resolution(self, res: int) -> None:
        """A higher resolution needs more room for activations: the weights go back to where the
        first load put them, and the VRAM fit measures again from there."""
        logging.info("[ZenTrainer] now training at %d px", res)
        if not self._auto_vram() or not getattr(self, "_first_resident", None):
            return
        self._trial = None
        self.__dict__.pop("_ceiling", None)
        self._resident = self._first_resident
        self._fit_weights()
        self._recompile_if_moved()
        self._resident = None
        self._warm = self._peak = 0
        self._times = []  # step times are by warm step, which starts again
        torch.cuda.reset_peak_memory_stats()

    def _recompile_if_moved(self) -> None:
        if getattr(self, "_compiled", 0):
            from .kernels.compile import reset_if_moved

            dm = self.models.patcher.model.diffusion_model
            self._residency = reset_if_moved(dm, getattr(self, "_residency", None))

    def _auto_vram(self) -> bool:
        m = self.spec.memory
        return m.offloading and m.vram_headroom <= 0 and torch.cuda.is_available()

    def _fit_weights(self) -> None:
        """Bring the weights on the card to the measured target (after a reload: ComfyUI sizes
        it by its own reserves, which _tighten has replaced)."""
        target = getattr(self, "_resident", None)
        if not target or not self._auto_vram():
            return
        patcher = self.models.patcher
        have = patcher.loaded_size()
        if have < target - 64 * 1024 ** 2:
            torch.cuda.empty_cache()  # into clean space, as _tighten
            patcher.partially_load(self.device, target - have)
        elif have > target + 64 * 1024 ** 2:
            patcher.partially_unload(patcher.offload_device, have - target)
            self._pin_offloaded()  # what came off the card is plain RAM again

    def _pin_offloaded(self) -> None:
        """Pinned copies of offloaded weights ComfyUI couldn't pin in place (see offload.py)."""
        if not self.spec.memory.offloading or not torch.cuda.is_available():
            return
        from .memory.offload import pin_offloaded

        n, size = pin_offloaded(self.models.patcher.model.diffusion_model)
        if n:
            logging.info("[ZenTrainer] pinned copies of %d offloaded weights (%.1f GB): they copy to the GPU without "
                         "blocking, under the previous layer's compute", n, size / 1024 ** 3)

    def _tighten(self) -> None:
        """With weights offloaded, everything training wasn't seen to use (bar a margin) holds
        model weights: fewer of them cross PCIe each step. The peak counts PyTorch's whole
        reservation (weights on the card, LoRA, optimizer, activations, and its cache): fitting on
        what tensors alone took left too little for the cache when shapes change step to step
        (Krea 2 int8, 16 GB card: two out-of-VRAM steps, settling barely higher). The rest of the
        card is the CUDA context, the desktop and other processes."""
        if not self._auto_vram():
            return
        torch.cuda.empty_cache()  # the weights go into clean space, not between cached fragments
        free, total = vram.mem_get_info()
        others = total - free - torch.cuda.memory_reserved()
        spare = total - others - self._peak - TIGHTEN_MARGIN
        patcher = self.models.patcher
        have = patcher.loaded_size()
        target = min(patcher.model_size(), have + spare, getattr(self, "_ceiling", 1 << 62))
        if target - have < 256 * 1024 ** 2:
            return
        gb = 1024 ** 3
        self._trial = {"at": self._warm, "prev": have, "before": list(getattr(self, "_times", [])),
                       "retries": torch.cuda.memory_stats().get("num_alloc_retries", 0)}
        self._resident = target
        self._fit_weights()
        self._recompile_if_moved()
        now = patcher.loaded_size()
        logging.info("[ZenTrainer] training peaked at %.1f GB of %.1f: model on the card %.1f -> %.1f of %.1f GB",
                     self._peak / gb, (total - others) / gb, have / gb, now / gb, patcher.model_size() / gb)
        self._peak = 0  # the next pass measures with these weights in
        torch.cuda.reset_peak_memory_stats()

    def _judge(self) -> None:
        """Keep the last fit only if it didn't slow the steps (see TRIAL_STEPS)."""
        trial = self._trial
        after = [t for t in getattr(self, "_times", []) if t[0] >= trial["at"] + TRIAL_STEPS[0]]
        ratio = trial_ratio(trial["before"], after)
        retries = torch.cuda.memory_stats().get("num_alloc_retries", 0) - trial["retries"]
        if ratio is None and not retries and self._warm < trial["at"] + TRIAL_LONGEST:
            return  # no step shaped like one before the fit yet: judge on a later step
        self._trial = None
        slower = ratio is not None and ratio > TRIAL_SLOWER
        gb = 1024 ** 3
        if not slower and not retries:
            if ratio is not None:
                logging.info("[ZenTrainer] with more of the model on the card: steps %+.0f%% (same-shaped steps)", (ratio - 1) * 100)
            return
        patcher = self.models.patcher
        have = patcher.loaded_size()
        self._resident = self._ceiling = trial["prev"]
        self._fit_weights()
        self._recompile_if_moved()
        why = f"{retries} allocator retries (the card ran too full)" if retries else f"same-shaped steps {(ratio - 1) * 100:+.0f}%"
        logging.info("[ZenTrainer] more of the model on the card was slower (%s): back to %.1f GB from %.1f, and no higher",
                     why, trial["prev"] / gb, have / gb)

    def _back_off(self, e: BaseException) -> bool:
        """A step ran out of VRAM (a bigger bucket than the first steps saw, say): take the
        margin's worth of weights off the card, never go above that again, and retry the step."""
        if not self._auto_vram() or getattr(self, "_backoffs", 0) >= TIGHTEN_RETRIES:
            return False
        import comfy.model_management as mm

        _release_frames(e)
        del e
        self.optimizer.zero_grad(set_to_none=True)
        self._backoffs = getattr(self, "_backoffs", 0) + 1
        patcher = self.models.patcher
        have = patcher.loaded_size()
        if have <= 0:
            return False  # nothing left on the card to move off
        self._resident = self._ceiling = max(0, have - TIGHTEN_MARGIN)
        self._fit_weights()
        self._recompile_if_moved()
        mm.soft_empty_cache()
        self._peak = 0
        torch.cuda.reset_peak_memory_stats()
        logging.warning("[ZenTrainer] out of VRAM: model on the card %.1f -> %.1f GB, retrying the step",
                        have / 1024 ** 3, patcher.loaded_size() / 1024 ** 3)
        return True

    def _profiled_step(self) -> None:
        """One training step under torch.profiler ("profile" command): a table of where the GPU
        time went (by op, and by op + input shapes) and a Chrome trace, in the run's folder."""
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            self._train_step()
            torch.cuda.synchronize()
        out = self.store.dir(self.run_id) if self.store else Path(".")
        ka = prof.key_averages()
        table = ka.table(sort_by="cuda_time_total", row_limit=60)
        shapes = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=40)
        (out / f"profile-step{self.step:06d}.txt").write_text(f"{table}\n\nby input shape\n{shapes}\n")
        prof.export_chrome_trace(str(out / f"profile-step{self.step:06d}.json"))
        logging.info("[ZenTrainer] profiled step %d: %s", self.step, out / f"profile-step{self.step:06d}.txt")
        from .kernels import int8_backward

        logging.info("[ZenTrainer] int8 backward paths so far: %s", dict(int8_backward.stats))

    def _train_step(self) -> None:
        s = self.spec
        ms = self.denoiser.model_sampling
        acc = s.optim.grad_accumulation_steps
        total_loss = torch.zeros((), device=self.device)
        # where the step's time goes: data, forward, backward, optimizer, shown on the run page.
        # Timing a phase means waiting for the GPU at its end, which leaves the GPU idle while the
        # next phase's work is queued (about 5% of a step), so only every PHASES_EVERY-th step does
        timed = self.step % PHASES_EVERY == 0
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        ph = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        mark = time.perf_counter()

        def lap(name):
            nonlocal mark
            if not timed:
                return
            sync()
            now = time.perf_counter()
            ph[name] += now - mark
            mark = now

        self.objective.start_step()
        keys = []
        for a in range(acc):
            idx = next(self.batches)
            x0, weight, shapes = self.batch(idx)
            keys.append((tuple(x0.shape), repr(shapes)))
            self._step_key = tuple(keys)  # what this step's time depends on (the auto VRAM fit's trials)
            conds = [self._cond(i, x0, shapes) for i in idx]

            noise = torch.randn(x0.shape, generator=self.generator).to(self.device)
            drawn = (self.step * acc + a) * s.optim.batch_size  # noise levels before these (spread)
            percents = sample_percents(s.objective, len(idx), self.generator, drawn, s.seed)
            sigma = sigmas_for(ms, percents, self._shift, self._schedule).to(self.device)
            x_t = self.objective.noisy(ms, sigma, noise, x0)
            lap("data")
            if s.slider.kind == "pairs":  # each picture at its own end, from the same noise
                total_loss += self._pair_loss(idx, x0, conds, weight, noise, sigma, x_t, percents, acc)
                lap("backward")
                continue
            if s.slider.kind:  # its own losses and backward passes (one per direction)
                total_loss += self._slider_loss(x0, shapes, x_t, sigma, acc)
                lap("backward")
                continue

            # the input needs grad only for ComfyUI's per-layer offload checkpoint (see memory.py);
            # otherwise it stays grad-free so the backward ends at the first trained layer
            with torch.autocast(self.device.type, dtype=self.train_dtype, enabled=self.train_dtype != torch.float32):
                x0_pred = self.denoiser(x_t.requires_grad_(getattr(self, "_input_grad", True)), sigma, conds,
                                        self._tread(idx))
            loss = self.objective.train_loss(x_t.detach(), x0, x0_pred, sigma, weight, percents).mean()
            lap("forward")
            scaled = loss / acc
            (self.scaler.scale(scaled) if self.scaler else scaled).backward()
            total_loss += loss.detach().float()
            lap("backward")

        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        params = self.network.trainable()
        for p in params:  # adapters may run in a lower precision than their grads
            if p.grad is not None and p.grad.dtype != p.dtype:
                p.grad = p.grad.to(p.dtype)
        clip = s.optim.max_grad_norm
        norm = torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf"))
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.step)
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        lap("optimizer")
        if timed:
            self.last_phases = {k: round(v, 4) for k, v in ph.items()}
        # the step's one wait for the GPU: its loss, grad norm and noise chart, read together
        loss_sum, grad_norm = self.objective.settle(total_loss, torch.as_tensor(norm))
        self.last_loss, self.last_grad_norm = loss_sum / acc, grad_norm

    # ---- sliders (harness/slider.py) ----------------------------------------------------------------

    def _data_spec(self, ds):
        """The dataset as data.build should make it: for an image slider, each picture's partner
        (slider.pair_set) encoded beside it at its own size, as channel inputs are (pixel for pixel),
        though the run itself isn't an edit run."""
        sl = self.spec.slider
        if sl.kind != "pairs":
            return ds
        from dataclasses import replace

        return replace(ds, edit=True, edit_method="concat", control_sets=[sl.pair_set], edit_grounded=False)

    def _keep_pairs(self, ds) -> None:
        """An image slider trains on pairs only: pictures without a partner leave the batches."""
        paired = [sm.has_refs if isinstance(sm, data.DiskSample) else bool(sm.refs and sm.refs[0] is not None)
                  for sm in ds.samples]
        if not any(paired):
            raise ValueError(f"no picture in {self.spec.dataset.folder} has a partner in _controls/{self.spec.slider.pair_set}/ "
                             "(same file names): an image slider trains on pairs")
        alone = len(paired) - sum(paired)
        if alone:
            logging.warning("[ZenTrainer] image slider: %d of %d pictures have no partner in _controls/%s/ and are left out",
                            alone, len(paired), self.spec.slider.pair_set)
        ds.buckets = {k: kept for k, v in ds.buckets.items() if (kept := [i for i in v if paired[i]])}

    def _pair_loss(self, idx, x0, conds, weight, noise, sigma, x_t, percents, acc: int) -> torch.Tensor:
        """An image slider's micro-batch: the dataset's pictures and their partners from the same
        noise at the same noise level, the LoRA at +1 towards one end and -1 towards the other, each
        the objective's own loss (a backward each, inside its strength). Returns the loss, detached,
        averaged over the two."""
        if self.streams.multi:
            raise ValueError("image sliders are for still-image models for now (this one trains several streams)")
        model = self.models.patcher.model
        partner = torch.stack([self.dataset.samples[i].refs[0] for i in idx]).to(self.device, torch.float32)
        x0_other = model.process_latent_in(partner)
        x_other = self.objective.noisy(self.denoiser.model_sampling, sigma, noise, x0_other)
        own = 1.0 if self.spec.slider.dataset_end == "positive" else -1.0
        total = torch.zeros((), device=self.device)
        for strength, xt, target in ((own, x_t, x0), (-own, x_other, x0_other)):
            with self.network.at_strength(strength):
                with torch.autocast(self.device.type, dtype=self.train_dtype, enabled=self.train_dtype != torch.float32):
                    pred = self.denoiser(xt.requires_grad_(getattr(self, "_input_grad", True)), sigma, conds, None)
                loss = self.objective.train_loss(xt.detach(), target, pred, sigma, weight, percents).mean()
                scaled = loss / (acc * 2)
                (self.scaler.scale(scaled) if self.scaler else scaled).backward()
            total += loss.detach().float() / 2
        return total

    def _slider_prompts(self) -> dict[str, str]:
        sl = self.spec.slider
        out = {"positive": sl.positive, "neutral": sl.neutral, "negative": sl.negative}
        out.update({f"anchor{k}": a for k, a in enumerate(a for a in sl.anchors if a.strip())})
        return out

    def _encode_texts(self, prompts: dict[str, str]) -> dict:
        """Prompts' text encodings, through the captions' text cache."""
        texts = data.TextCache(self.cache_dir, self.encoder_id)
        out = {}
        with torch.no_grad():
            for name, text in prompts.items():
                cond = texts.get(text)
                if cond is None:
                    clip = self.models.clip
                    cond = data._to_cpu(clip.encode_from_tokens_scheduled(clip.tokenize(text)))
                    texts.put(text, cond)
                out[name] = cond
        return out

    def _slider_cond(self, name: str, x0: torch.Tensor, shapes) -> dict:
        key = (name, tuple(x0.shape[1:]))
        if key not in self._slider_conds:
            with torch.no_grad():
                self._slider_conds[key] = self.denoiser.prepare(self._slider_text[name], x0[:1], self.device, shapes)
        return self._slider_conds[key]

    def _slider_loss(self, x0: torch.Tensor, shapes, x_t: torch.Tensor, sigma: torch.Tensor, acc: int) -> torch.Tensor:
        """A text slider's micro-batch: the base model's predictions (strength 0, no grad) make the
        targets; then the LoRA trains at +1 (and -1) on the neutral prompt, a backward each, inside
        its strength (gradient checkpointing recomputes the forward during the backward). Returns
        the loss, detached, averaged over the directions."""
        from . import slider

        sl, n = self.spec.slider, x0.shape[0]
        autocast = lambda: torch.autocast(self.device.type, dtype=self.train_dtype,  # noqa: E731
                                          enabled=self.train_dtype != torch.float32)
        cond = {k: [self._slider_cond(k, x0, shapes)] * n for k in self._slider_text}
        anchors = [k for k in cond if k.startswith("anchor")]
        with torch.no_grad(), self.network.at_strength(0.0), autocast():
            base = {k: self.denoiser(x_t, sigma, cond[k], None).float() for k in cond}
        plus, minus = slider.text_targets(self.spec.objective.target, x_t, sigma, base["positive"], base["neutral"],
                                          base["negative"], sl.guidance)
        directions = [(1.0, plus), (-1.0, minus)] if sl.directions == "both" else [(1.0, plus)]
        total = torch.zeros((), device=self.device)
        for strength, target in directions:
            with self.network.at_strength(strength):
                with autocast():
                    pred = self.denoiser(x_t.requires_grad_(getattr(self, "_input_grad", True)), sigma, cond["neutral"], None)
                loss = self.objective.loss(x_t.detach(), target, pred, sigma).mean()
                for a in anchors:  # what the slider should leave alone, at either end
                    with autocast():
                        held = self.denoiser(x_t, sigma, cond[a], None)
                    loss = loss + sl.anchor_weight * self.objective.loss(x_t.detach(), base[a], held, sigma).mean() / len(anchors)
                scaled = loss / (acc * len(directions))
                (self.scaler.scale(scaled) if self.scaler else scaled).backward()
            total += loss.detach().float() / len(directions)
        return total

    def _tread(self, idx: list[int]) -> dict | None:
        """TREAD's block replacements for this step: only on a video clip, only for models whose
        blocks take the replace hook with a sequence layout (MiniMax H3)."""
        share = self.spec.memory.tread
        if share <= 0 or any(self.dataset.samples[i].latent.ndim != 4 or self.dataset.samples[i].latent.shape[1] < 2 for i in idx):
            return None
        if self.spec.memory.tread_steps < 1.0 and torch.rand((), generator=self.generator).item() >= self.spec.memory.tread_steps:
            return None  # a whole step: keeps the LoRA working on the unrouted model it'll be used on
        dm = self.models.patcher.model.diffusion_model
        if "minimax" not in type(dm).__module__.lower() or not hasattr(dm, "blocks"):
            return None
        from .memory.tread import patches

        p = patches(len(dm.blocks), share, self.generator)
        return {"patches_replace": {"dit": p}} if p else None

    def batch(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        """The model-ready x0 for these samples, its loss weights (masks; for a multi-stream
        model also which part is the picture) and, for a multi-stream model, the stream shapes
        its conds need. See streams.py."""
        model = self.models.patcher.model
        latent = torch.stack([self.dataset.samples[i].latent for i in idx]).to(self.device, torch.float32)
        weight = self._mask_weight(idx, latent) if self.spec.dataset.masked_loss else None
        audio = [self.dataset.samples[i].audio for i in idx]
        packed = self.streams.pack(latent, weight, audio if any(a is not None for a in audio) else None)
        if packed is None:
            return model.process_latent_in(latent), weight, None
        model.latent_shapes = packed.shapes  # how the sampler leaves it: process_latent_in reads it
        return model.process_latent_in(packed.x0), packed.weight, packed.shapes

    def _mask_weight(self, idx: list[int], x0: torch.Tensor) -> torch.Tensor | None:
        """(B, h, w) loss weights: mask where present, 1 elsewhere; the unmasked area
        keeps `unmasked_weight` (0 = ignore it entirely)."""
        masks = [self.dataset.samples[i].mask for i in idx]
        if all(m is None for m in masks):
            return None
        floor = self.spec.dataset.unmasked_weight
        hw = x0.shape[-2:]
        w = torch.stack([m if m is not None else torch.ones(hw) for m in masks]).to(x0.device)
        return floor + (1.0 - floor) * w

    def fixed_cond(self, i: int, x0: torch.Tensor, shapes: list | None = None) -> dict:
        """A sample's own caption, prepared; never dropped out (validation uses this)."""
        sample = self.dataset.samples[i]
        return self._prepared(i, sample.caption, x0, shapes)

    def _prepared(self, i: int, cap: int, x0: torch.Tensor, shapes: list | None, keep: tuple | None = None) -> dict:
        """A caption's processed conds for sample i's shape; an edit pair's carry its own input
        picture, so those are kept per sample. `keep`: which channel inputs this step keeps (None:
        all)."""
        sample = self.dataset.samples[i]
        pairs = self.spec.slider.kind == "pairs"  # an image slider's partner is the other end, not an input
        own = (sample.refs and not pairs) or sample.cond is not None or self.concat_inputs
        key = (cap, tuple(sample.latent.shape), i if own else None, keep)
        # a disk dataset's pairs (thousands, each its own): made each step, never all kept on the GPU
        passing = own and isinstance(sample, data.DiskSample)
        if passing or key not in self._conds:
            cond = self.dataset.conditionings[cap]
            if sample.cond is not None:  # grounded: the encoder saw this image's inputs
                empty = cap == self.dataset.empty_caption and sample.cond_empty is not None
                cond = sample.cond_empty if empty else sample.cond
            if self.concat_inputs:  # channel inputs: every call carries them, zeros where missing or left out
                from .data.edit import with_concat

                cond = with_concat(cond, self._concat(sample, keep))
            elif sample.refs and not pairs:
                from .data.edit import with_refs

                cond = with_refs(cond, sample.refs, self.edit_method)
            with torch.no_grad():
                prepared = self.denoiser.prepare(cond, x0[:1], self.device, shapes)
            if passing:
                return prepared
            self._conds[key] = prepared
        return self._conds[key]

    def _cond(self, i: int, x0: torch.Tensor, shapes: list | None = None) -> dict:
        sample = self.dataset.samples[i]
        cap = random.choice(sample.variants) if len(sample.variants) > 1 else sample.caption
        if self.spec.dataset.caption_dropout and random.random() < self.spec.dataset.caption_dropout:
            cap = self.dataset.empty_caption  # an edit pair keeps its input picture
        keep = None
        if self.concat_inputs and self.spec.dataset.control_dropout:  # each input left out on its own
            keep = tuple(random.random() >= self.spec.dataset.control_dropout for _ in range(self.concat_inputs))
        return self._prepared(i, cap, x0, shapes, keep)

    def _concat(self, sample, keep: tuple | None) -> torch.Tensor:
        """A sample's channel inputs, processed, (1, C × inputs, h, w): zeros where it has no input
        in a set, or `keep` leaves it out."""
        model = self.models.patcher.model
        refs = sample.refs or [None] * self.concat_inputs
        parts = []
        for k, r in enumerate(refs):
            if r is None or (keep is not None and not keep[k]):
                parts.append(torch.zeros((1,) + tuple(sample.latent.shape)))
            else:
                parts.append(model.process_latent_in((r if r.ndim == 4 else r[None]).float()))
        return torch.cat(parts, dim=1)

    def _handle_commands(self) -> None:
        import comfy.model_management as mm

        self._apply(self.control.drain())
        if self.control.paused:
            self._status("paused")
            while not self.control.wait_resumed(0.5):
                mm.throw_exception_if_processing_interrupted()
                if _comfy_waiting():
                    self._park()
            self._apply(self.control.drain())  # whatever arrived while paused (set_lr, stop, ...)
            self._status("training")

    def _park(self) -> None:
        """Paused, and another ComfyUI prompt is waiting behind this run: let go of ComfyUI (its
        queue and the GPU) rather than hold them. The run stops as a stop does, saving its state,
        and the queue keeps it paused to continue from there (server/jobqueue.py, routes/runs.py
        parked). Without resume states it can't continue, so it stays paused and says so."""
        if self.spec.output.keep_states == "none":
            if not getattr(self, "_park_told", False):
                self._park_told = True
                self._status("paused", message="a ComfyUI workflow is waiting, but this run keeps no resume states "
                                               "(Output → Resume states), so pausing can't free ComfyUI: stop it, or resume")
            return
        if self.store:
            self.store.update(self.run_id, park=True)
        logging.info("[ZenTrainer] paused and a ComfyUI prompt is waiting: saving and letting go of ComfyUI")
        raise Stop("stopped")

    def _partial(self, name: str, what: str, kept: int, left_out: list) -> None:
        """Say so when a frozen adapter fits only part of this model (it was made for another build
        of it, e.g. the full H3 against a pruned one): what it does then is only partly there."""
        if not left_out:
            return
        share = len(left_out) / (kept + len(left_out))
        msg = (f"{name}: {len(left_out)} of {kept + len(left_out)} layers don't fit this model (made for another "
               f"build of it?), so {what} is only partly applied")
        logging.warning("[ZenTrainer] %s", msg)
        if share >= 0.05:
            self._status("loading", message=msg)

    def _apply(self, commands) -> None:
        tries: list[dict] = []
        for cmd in commands:
            if cmd.name == "stop":
                raise Stop("stopped")
            if cmd.name == "save":
                self.save(f"manual-step{self.step:06d}")
            elif cmd.name == "sample" and self._samples:
                if cmd.args.get("prompt"):
                    tries.append(cmd.args)  # "Try a prompt": batched below
                else:
                    self._samples.render(self)
            elif cmd.name == "profile":
                self._profile_next = True
            elif cmd.name == "set_lr":
                optim.set_lr(self.optimizer, self.scheduler, float(cmd.args["lr"]))
        if tries and self._samples:
            # manual tries in one session; ones queued while it renders join it, so the model and
            # encoder aren't swapped in and out per try. Other commands that arrive meanwhile
            # wait for the next step boundary.
            later = []

            def more():
                got = []
                for c in self.control.drain():
                    (got.append(c.args) if c.name == "sample" and c.args.get("prompt") else later.append(c))
                return got

            self._samples.render_manual(self, tries, more)
            if later:
                self._apply(later)

    # ---- outputs ---------------------------------------------------------------

    def save(self, tag: str, final: bool = False) -> str:
        import folder_paths
        import safetensors.torch

        optim.set_mode(self.optimizer, training=False)  # schedule-free: save the eval weights
        try:
            if self.ema is None:
                return self._save(tag, final, folder_paths, safetensors.torch)
            with self.ema.applied() as live:  # the LoRA file is the average; the state keeps the live weights
                return self._save(tag, final, folder_paths, safetensors.torch, live)
        finally:
            optim.set_mode(self.optimizer, training=True)

    def _save(self, tag, final, folder_paths, sft, live: dict | None = None) -> str:
        root, base = lora_root(self.spec.output.lora_dir)
        name = f"{slugify(self.spec.name)}_{self.run_id}" + ("" if final else f"_{tag}")
        folder = base / slugify(self.spec.name) if final else base / "checkpoints" / self.run_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.safetensors"
        save_as = _DTYPES[self.spec.output.save_dtype]
        weights = {k: v if k.endswith(".set_weight") else v.to(save_as) for k, v in self.network.state_dict().items()}
        if live is None and save_as.itemsize < self.network.dtype.itemsize:
            live = {k: v.detach() for k, v in self.network.params.items()}  # resume from the full weights, not the file's
        keep = self.spec.output.keep_states
        state = state_dir(self.run_id) / f"{name}.pt"
        _room_for(folder, sum(t.numel() * t.element_size() for t in weights.values()))
        # written beside it under a name ComfyUI doesn't list, then swapped in: a full disk or a
        # kill mid-save leaves no half-written LoRA in the loras folder
        part = folder / f".{name}.part"
        try:
            sft.save_file(weights, str(part), metadata=write_metadata(self.spec.to_dict(), self.step))
            # safetensors writes owner-only (0600); a LoRA is an output like ComfyUI's images, readable
            # by whatever else serves or syncs the folder (Docker, a share, another user's ComfyUI)
            os.chmod(part, 0o644)
            os.replace(part, path)
        finally:
            part.unlink(missing_ok=True)
        # resumable state (optimizer, schedule, step, RNG) goes outside the LoRA folders:
        # those are registered with ComfyUI, which would list a .pt there as a "LoRA"
        if keep != "none":
            state.parent.mkdir(parents=True, exist_ok=True)
            tmp_state = state.with_name(f".{state.name}.part")
            torch.save({
                "step": self.step, "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "rng": self.generator.get_state(), "batches_epoch": self.batches.epoch,
                "train_seconds": self.train_seconds,
                **({"balance": self.objective.balance.state_dict()} if self.objective.balance is not None else {}),
                **({"live": {k: v.cpu() for k, v in live.items()}} if live is not None else {}),
                **({"ema": self.ema.state_dict()} if self.ema is not None else {}),
            }, str(tmp_state))
            os.replace(tmp_state, state)
            if keep == "latest":  # resume only ever needs the newest
                for old in state.parent.glob("*.pt"):
                    if old != state:
                        old.unlink(missing_ok=True)
        row = {"run_id": self.run_id, "step": self.step, "path": str(path), "final": final,
               "lora_name": os.path.relpath(path, root), "state": str(state) if keep != "none" else None}
        self.emit(events.CHECKPOINT, row)
        if self.store:
            self.store.append(self.run_id, "checkpoints", row)
        return str(path)

    def _status(self, status: str, message: str | None = None, **fields: Any) -> None:
        if self.sample_job and status in ("loading", "training", "sampling"):
            # setup's progress, while sampling a finished run: still just sampling, said plainly
            status, message = "sampling", f"sampling step {self.sample_job['step']}"
        self.emit(events.STATUS, {"run_id": self.run_id, "status": status, "message": message, "step": self.step})
        self.emit(events.ZENKIT_JOB, {
            "id": f"zentrainer:{self.run_id}", "name": f"Train {self.spec.name}",
            "status": {"loading": "start", "training": "progress", "paused": "progress", "sampling": "progress",
                       "done": "done"}.get(
                status, "error"),
            "current": self.step, "total": self.spec.optim.steps, "message": message or status,
            "source": "ZenTrainer",
        })
        if self.store:
            self.store.update(self.run_id, status=status, message=message, **fields)


def _room_for(folder: Path, need: int) -> None:
    """Stop with a clear message instead of a half-written file when the disk is nearly full:
    room for this save twice over (the LoRA and its training state) plus a margin."""
    import shutil

    free = shutil.disk_usage(folder).free
    if free < need * 2 + (256 << 20):
        raise RuntimeError(f"not enough disk space to save a checkpoint in {folder}: {free / 1e9:.1f} GB free. "
                           "Free some up (System → Storage) and resume from the last checkpoint")

