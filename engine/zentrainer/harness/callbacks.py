"""Things that happen around steps: metrics, checkpoints, samples.

A callback gets the trainer at `on_start`, after every optimizer step, and at
`on_end`. New behaviour (EMA, validation loss, early stopping) is a new class here,
not a change to the loop.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .. import events, paths
from . import optim
from .data.streams import first_stream

# a slider's samples: each prompt at these strengths, one strip (spec.slider; harness/slider.py)
SLIDER_SWEEP = (-1.0, -0.5, 0.0, 0.5, 1.0)
SAMPLE_HEADROOM = 6 * 1024 ** 3  # extra VRAM kept free while an offloaded model samples (see render)

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    def on_start(self, t: Trainer) -> None: ...
    def on_step(self, t: Trainer) -> None: ...
    def on_end(self, t: Trainer, status: str) -> None: ...


# it/s (and so the ETA) over this many steps: while a run blends into a new resolution its steps
# come at both sizes, and a rate of the last few steps jumped between the two
RATE_STEPS = 50


def rate(times) -> float:
    """Steps a second over these step times: their count over their sum, leaving out any over 5x
    their median (a compile, or a recompile at a new shape), which isn't the run's pace."""
    ts = [t for t in times if t > 0]
    if not ts:
        return 0.0
    cap = 5 * sorted(ts)[len(ts) // 2]
    kept = [t for t in ts if t <= cap]
    return len(kept) / sum(kept)


class Metrics(Callback):
    """One row per optimizer step. `train_time` counts only optimizer steps; `elapsed` is
    wall-clock for this session; `it_s` is from the steps themselves (the last RATE_STEPS), so a
    sample render or a validation pass doesn't show up as a slowdown."""

    def __init__(self):
        self._t0 = 0.0
        self._times: collections.deque[float] = collections.deque(maxlen=RATE_STEPS)

    def on_start(self, t):
        self._t0 = time.monotonic()

    def on_step(self, t):
        now = time.monotonic()
        dt = t.last_step_seconds or 0.0
        self._times.append(dt)
        row = {
            "run_id": t.run_id,
            "step": t.step,
            "total": t.spec.optim.steps,
            "loss": t.last_loss,
            "lr": optim.current_lr(t.optimizer),
            "grad_norm": t.last_grad_norm,
            "epoch": t.epoch_at(t.step),
            "elapsed": round(now - self._t0, 3),
            "train_time": round(t.train_seconds, 3),
            "step_time": round(dt, 4),
            "it_s": round(rate(self._times), 3),
        }
        if getattr(t.objective, "drawn", None):  # each sample's [percent, σ, plain loss]: the noise chart
            row["drawn"] = [[round(p, 4), round(s, 5), round(v, 5)] for p, s, v in t.objective.drawn]
        if getattr(t, "last_phases", None):
            row["phases"] = t.last_phases  # seconds in data / forward / backward / optimizer
        if torch.cuda.is_available():
            # the step's peak VRAM (PyTorch's allocations: weights resident, activations, LoRA,
            # optimizer): what the offloading headroom has to cover
            row["vram_peak"] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t.emit(events.METRICS, row)
        if t.store:
            t.store.append(t.run_id, "metrics", row)


class Checkpoints(Callback):
    """LoRA files ComfyUI can load, plus a resumable training state alongside."""

    def __init__(self, every: int):
        self.every = every
        self.saved = None  # the step last saved by this run

    def on_step(self, t):
        if self.every and t.step % self.every == 0 and t.step < t.spec.optim.steps:
            t.save(f"step{t.step:06d}")
            self.saved = t.step

    def on_end(self, t, status):
        if t.step <= 0:
            return
        if status != "done" and self.saved == t.step:
            return  # stopped right after a save (often while it sampled): that file and its state resume
        t.save("final" if status == "done" else f"{status}-step{t.step:06d}", final=status == "done")


class _OptionsOnly:
    """Enough of a ModelPatcher for ComfyUI's guidance nodes to add their sampler options to:
    clone() is itself and the setters only touch a copy of the options."""

    def __init__(self, patcher):
        self._patcher = patcher
        self.model_options = patcher.model_options.copy()

    def clone(self):
        return self

    def get_model_object(self, name):
        return self._patcher.get_model_object(name)

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        import comfy.model_patcher

        self.model_options = comfy.model_patcher.set_model_options_post_cfg_function(self.model_options, fn, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, fn):
        self.model_options = {**self.model_options, "sampler_calc_cond_batch_function": fn}


class Samples(Callback):
    """Render the sample prompts with the in-progress weights, in-process.

    Uses ComfyUI's own sampler on the training patcher (adapters applied), then
    restores the training state. If this proves fragile on some families, the
    fallback is a separate sample job against the last checkpoint.
    """

    def __init__(self, every: int):
        self.every = every
        self._conds = None
        self._alt = None  # base patcher of the alternate sampling model, loaded once
        self._base_sd = None  # the base LoRA (network.base_lora), for the alternate model

    def _like(self, t, row: dict) -> None:
        """Hand a landed sample to the likeness worker, started on first use."""
        if not t.spec.samples.likeness or not t.store or row.get("sweep"):  # a slider's strip: five faces
            return
        if getattr(self, "_liker", None) is None:
            from ..datasets import faces

            from ..datasets import folders as datasets

            if not datasets.get_settings(t.spec.dataset.folder).get("faces"):
                self._liker = False  # likeness for datasets with faces on
                return
            self._liker = faces.LikenessWorker(t.store, t.run_id, t.spec.dataset.folder)
        if self._liker:
            self._liker.submit(row)

    def on_start(self, t):
        # encode even when unscheduled: a "sample now" command can still ask for them
        with torch.no_grad():
            self._encode(t, t.spec.samples)
        if t.spec.samples.baseline and self._conds and t.step == 0:
            self.render(t)  # the base model's look, before any training

    # ---- baselines: the untrained model's samples depend only on the model and the sample
    # settings, so a run that shares them with an earlier one copies its files instead of
    # rendering them again (output/zentrainer/samples/_baseline/<key>.<ext>)
    def _baseline_key(self, t, i: int) -> str:
        import hashlib

        m, s, it = t.spec.model, t.spec.samples, self.items[i]
        parts = [m.kind, m.ckpt_name, m.unet_name, m.unet_weight_dtype, m.clip_name, m.clip_type, m.vae_name,
                 getattr(m, "audio_vae_name", None), s.model_name, s.model_weight_dtype, t.spec.network.init_from,
                 self.prompts[i], s.negative, it.width, it.height, it.frames, s.fps, it.seed, s.steps, s.cfg,
                 s.sampler, s.scheduler, s.slg_mode, s.slg_layers, s.slg_double_layers, s.slg_scale, s.slg_start,
                 s.slg_end, s.lora_strength if t.spec.network.init_from else None, it.control]
        if s.sigmas:  # only then: baselines kept before custom sigmas existed still match
            parts.append(s.sigmas)
        # samples render with these on, so they make another baseline (added only when set, as sigmas)
        if t.spec.network.base_lora:
            parts.append(["base", t.spec.network.base_lora, t.spec.network.base_strength])
        if s.turbo_lora:
            parts.append(["turbo", s.turbo_lora, s.turbo_strength])
        return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:20]

    def _baseline_dir(self) -> Path:
        return paths.output_root() / "samples" / "_baseline"

    def _baseline_hit(self, t, i: int, out_dir: str) -> dict | None:
        import shutil

        key = self._baseline_key(t, i)
        for src in self._baseline_dir().glob(f"{key}.*"):
            name = f"step000000_{i:02d}{src.suffix}"
            shutil.copyfile(src, Path(out_dir) / name)
            return {"run_id": t.run_id, "step": 0, "index": i, "prompt": self.prompts[i], "filename": name,
                    "subfolder": f"zentrainer/samples/{t.run_id}", "type": "output", "model": t.spec.samples.model_name or None,
                    "reused": True}
        return None

    def _baseline_keep(self, t, i: int, path: Path) -> None:
        import shutil

        try:
            d = self._baseline_dir()
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, d / f"{self._baseline_key(t, i)}{path.suffix}")
        except OSError:
            pass

    @contextlib.contextmanager
    def _at_strength(self, t, model, strength: float):
        """The sampling model with the LoRA at `strength`: the training patcher's own adapters set to
        it, or the alternate model's LoRA patches rescaled in place. A patched copy per strength
        cost ComfyUI's dynamic loading 12.9 GB of RAM each on Krea 2 (it ran a 128 GB machine out);
        rescaled patches with a new patches_uuid are the same model, refreshed. A base LoRA's
        patches keep their own strength."""
        if not t.spec.samples.model_name:
            with t.network.at_strength(strength):
                yield model
            return
        import uuid

        ours, base = self._own_patches, t.spec.samples.lora_strength

        def scale(k: float) -> None:
            for key, lst in model.patches.items():
                model.patches[key] = [(base * k,) + p[1:] if id(p[1]) in ours else p for p in lst]
            model.patches_uuid = uuid.uuid4()

        scale(strength)
        try:
            yield model
        finally:
            scale(1.0)

    def _sampling_model(self, t):
        """The training patcher itself, or the alternate model with the current LoRA."""
        s = t.spec.samples
        if not s.model_name:
            return t.models.patcher
        import comfy.sd
        import folder_paths

        if self._alt is None:
            if t.spec.model.kind == "split":
                opts = {}
                if s.model_weight_dtype.startswith("fp8_e4m3fn"):
                    opts["dtype"] = torch.float8_e4m3fn
                elif s.model_weight_dtype == "fp8_e5m2":
                    opts["dtype"] = torch.float8_e5m2
                path = folder_paths.get_full_path_or_raise("diffusion_models", s.model_name)
                self._alt = comfy.sd.load_diffusion_model(path, model_options=opts)
            else:
                path = folder_paths.get_full_path_or_raise("checkpoints", s.model_name)
                self._alt = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=False)[0]
        weights = t.network.state_dict()
        wide = getattr(t.network, "wide", None)
        if wide is not None:  # the widened input layer: the same wrappers on this model (see wide_input.py)
            weights = {k: v for k, v in weights.items() if not k.endswith(".set_weight")}
        patched, _ = comfy.sd.load_lora_for_models(self._alt, None, weights, s.lora_strength, 0)
        self._own_patches = {id(p[1]) for lst in patched.patches.values() for p in lst}  # the trained LoRA's (a sweep rescales them)
        if wide is not None:
            patched.add_weight_wrapper(f"{t.network.wide_name}.weight", wide.weight)
            if wide.bias_delta is not None:
                patched.add_weight_wrapper(f"{t.network.wide_name}.bias", wide.bias)
        from .moss_refs import supports

        if getattr(t, "edit_method", None) and supports(patched.model):  # as the training model is
            from . import moss_refs

            moss_refs.apply(patched)
        base = t.spec.network.base_lora
        if base:  # the trained LoRA sits on it, so its samples do too
            if self._base_sd is None:
                import comfy.utils

                self._base_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", base), safe_load=True)
            patched, _ = comfy.sd.load_lora_for_models(patched, None, self._base_sd, t.spec.network.base_strength, 0)
        return patched

    def _guided_options(self, model, s) -> dict | None:
        """The sampling model's options with skip-layer guidance added, or None for none.
        Runs ComfyUI's own SkipLayerGuidanceDiT(Simple) on a stand-in that only collects
        options: cloning the training patcher would reload its weights for every sample."""
        if s.slg_mode == "off" or not (s.slg_layers or s.slg_double_layers):
            return None
        from comfy_extras import nodes_slg

        stub = _OptionsOnly(model)
        single, double = ", ".join(map(str, s.slg_layers)), ", ".join(map(str, s.slg_double_layers))
        if s.slg_mode == "uncond":
            nodes_slg.SkipLayerGuidanceDiTSimple.execute(stub, s.slg_start, s.slg_end, double, single)
        else:
            nodes_slg.SkipLayerGuidanceDiT.execute(stub, s.slg_scale, s.slg_start, s.slg_end, double, single)
        return stub.model_options

    def _encode(self, t, s):
        from .data import apply_trigger

        from ..spec import sample_prompts

        trig = t.spec.dataset.trigger
        self.items = sample_prompts(s)
        # samples only substitute [trigger] (and, on edit runs, take the instruction template)
        self.prompts = [_sample_text(t, p.prompt) for p in self.items]
        negative = apply_trigger(s.negative, trig, False)
        self._conds = self._encode_texts(t, self.prompts, negative, [_special(t, p) for p in self.items])

    def _encode_texts(self, t, prompts: list[str], negative: str, special: list | None = None) -> list:
        """[(cond, negative cond)] for these prompts; cached like captions, so a run with
        nothing new to encode never loads the text encoder. A prompt with a `special` encoder
        (an edit sample's input: grounded, or H3's references / first frame; see _special) is
        encoded by it, its stamp in the cache key."""
        special = special or [None] * len(prompts)
        stamp = [sp[0] if sp else None for sp in special]
        path = _prompt_cache(t, prompts + ([stamp] if any(stamp) else []), negative)
        if path is not None and path.is_file():
            try:
                cached = torch.load(path, weights_only=False)
                negs = cached.get("negatives") or [None] * len(prompts)
                return [(c, n if n is not None else cached["negative"]) for c, n in zip(cached["prompts"], negs)]
            except Exception:
                pass
        clip = t.models.clip
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
        conds = []
        for text, sp in zip(prompts, special):
            if sp is None:
                conds.append((clip.encode_from_tokens_scheduled(clip.tokenize(text)), neg))
            else:
                c, n = sp[1](clip, text, negative)
                conds.append((c, neg if n is None else n))
        if path is not None:
            from .data import _to_cpu

            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"prompts": [_to_cpu(c) for c, _ in conds], "negative": _to_cpu(neg),
                        "negatives": [_to_cpu(n) if sp is not None else None for (_, n), sp in zip(conds, special)]}, path)
        return conds

    def render_manual(self, t, tries: list[dict], more=None) -> list[dict]:
        """Manual tries ("Try a prompt"): rendered like the run's samples but kept apart (manual
        rows, their own files). A batch shares one text-encoder session (every new prompt
        encoded, then the encoder freed once) and one sampling session (the model switched to
        sampling once, back to training once); `more()` can hand over tries queued meanwhile,
        which join the same session. Each try: {prompt, width?, height?, frames?, seed?, base?,
        index?, sampler?, scheduler?, steps?, cfg?, sigmas?} (the last five: its own sampling,
        else the run's; see _try_sampling)."""
        from ..spec import SamplePrompt
        from .data import apply_trigger

        if self._conds is None:
            self.on_start(t)
        s, trig = t.spec.samples, t.spec.dataset.trigger
        negative = apply_trigger(s.negative, trig, False)
        rows: list[dict] = []
        saved = (self.items, self.prompts, self._conds)
        try:
            while tries:
                jobs = []
                with torch.no_grad():  # the encoder stays loaded across the batch's prompts
                    for a in tries:
                        text = _sample_text(t, str(a["prompt"]))
                        it = SamplePrompt(text, a.get("width") or s.width, a.get("height") or s.height,
                                          a.get("frames") or s.frames, 0, a.get("control") or None)
                        jobs.append((a, text, self._encode_texts(t, [text], negative, [_special(t, it)])))
                if hasattr(t, "_drop_text_encoder"):
                    t._drop_text_encoder()  # don't keep a big encoder beside training
                for a, text, conds in jobs:
                    # a new try gets the next number; "again at this step" reuses its try's (same card)
                    n = a["index"] if a.get("index") is not None else \
                        len({f.name.split("_")[1] for f in Path(_out_dir(t)).glob("manual_*")})
                    item = SamplePrompt(str(a["prompt"]), a.get("width") or s.width, a.get("height") or s.height,
                                        a.get("frames") or s.frames,
                                        int(time.time()) % 100000 if a.get("seed") is None else a["seed"],
                                        a.get("control") or None)
                    self.items, self.prompts, self._conds = [item], [text], conds
                    over = _try_sampling(s, a)
                    if a.get("base"):  # the same prompt and seed without the LoRA, to compare against
                        with t.network.disabled():
                            rows += self.render(t, manual=n, base=True, stay=True, over=over)
                    rows += self.render(t, manual=n, stay=True, over=over)
                tries = more() if more else []
        finally:
            self.items, self.prompts, self._conds = saved
            t.exit_inference()  # back to training once, after the whole batch
            mm_empty()
        return rows

    def on_step(self, t):
        if (self.every and t.step % self.every == 0) or _scores_saves(t, "likeness"):
            self.render(t)

    def on_end(self, t, status):
        liker = getattr(self, "_liker", None)
        if liker:
            # it finishes what's queued, then stops; when likeness picks the checkpoints, the end
            # waits for the last scores (Keep ranks them next)
            liker.close(wait=120 if t.spec.output.keep.removesuffix("_new_best") == "likeness" else 0)
        self._liker = None

    def render(self, t: Trainer, manual: int | None = None, base: bool = False, stay: bool = False,
               over=None) -> list[dict]:
        """The run's samples (or, from render_manual, one try). `stay` leaves the model set up
        for sampling afterwards: the caller renders more and switches back once. `over`: sample
        settings in place of the run's (a try's own sampler, steps...). With weight averaging
        on, samples show the average, as the checkpoints do."""
        ema = getattr(t, "ema", None)
        if ema is not None and not base:
            with ema.applied():
                return self._render(t, manual, base, stay, over)
        return self._render(t, manual, base, stay, over)

    def _render(self, t: Trainer, manual: int | None = None, base: bool = False, stay: bool = False,
                over=None) -> list[dict]:
        if self._conds is None:
            self.on_start(t)
        if not self._conds:
            t.emit(events.STATUS, {"run_id": t.run_id, "status": "training", "step": t.step,
                                   "message": "sample requested, but the run has no sample prompts"})
            return []
        import comfy.model_management as mm
        import comfy.sample
        from PIL import Image

        s = over or t.spec.samples
        sigmas = torch.tensor(s.sigmas, dtype=torch.float32) if s.sigmas else None  # in place of the scheduler
        steps = len(s.sigmas) - 1 if s.sigmas else s.steps
        vae = t.models.vae
        comp = vae.spacial_compression_encode()
        out_dir = _out_dir(t)
        rows = []

        if manual is None and t.step == 0 and all(any(self._baseline_dir().glob(f"{self._baseline_key(t, i)}.*")) for i in range(len(self._conds))):
            # every baseline rendered before with these exact settings: copy them, load nothing
            for i in range(len(self._conds)):
                row = self._baseline_hit(t, i, out_dir)
                rows.append(row)
                t.emit(events.SAMPLE, row)
                if t.store:
                    t.store.append(t.run_id, "samples", row)
                    self._like(t, row)
            return rows

        previewer = _previewer(t)
        count = len(self._conds)

        def progress(i, s_step, s_total, x0=None):
            row = {"run_id": t.run_id, "step": t.step, "phase": "sampling", "index": i, "count": count,
                   "prompt": self.prompts[i], "s_step": s_step, "s_total": s_total}
            if manual is not None:  # a "Try a prompt" render: the UI previews it in its own tab
                row.update(manual=True, manual_index=manual, base=base)
            if x0 is not None and previewer is not None:
                row["preview"] = _preview_url(previewer, x0)
            t.emit(events.SAMPLE_PROGRESS, row)

        guided = plain_options = model = None
        t.enter_inference()
        # ComfyUI sizes how much of the model to load for sampling by its own estimate, which
        # doesn't count our LoRA wrappers building each layer's full weight as it runs: with the
        # model offloaded, keep the same room free as training does
        reserved = mm.EXTRA_RESERVED_VRAM
        if t.spec.memory.offloading:
            mm.EXTRA_RESERVED_VRAM = reserved + SAMPLE_HEADROOM
        t.network.assist = False  # samples show the LoRA on the real model, not the training adapter
        try:
            with torch.no_grad():
                model = self._sampling_model(t)
                guided = self._guided_options(model, s)
                plain_options = model.model_options
                if guided is not None:
                    model.model_options = guided  # put back in finally
                for i, (pos, neg) in enumerate(self._conds):
                    reused = self._baseline_hit(t, i, out_dir) if t.step == 0 and manual is None else None
                    if reused is not None:
                        rows.append(reused)
                        t.emit(events.SAMPLE, reused)
                        if t.store:
                            t.store.append(t.run_id, "samples", reused)
                        continue
                    progress(i, 0, steps)
                    channels = model.model.latent_format.latent_channels
                    it = self.items[i]
                    concat = getattr(t, "concat_inputs", 0)
                    ref = None
                    if it.control and not getattr(t, "h3", False):  # an edit sample: the picture to edit as the reference, at its own shape
                        ref, it = _control(t, it, comp)
                        if not concat:
                            from .data.edit import with_refs

                            how = getattr(t, "edit_method", None)
                            pos, neg = with_refs(pos, [ref], how), with_refs(neg, [ref], how)
                    t_lat = _latent_frames(vae, it.frames)
                    latent = torch.zeros([1, channels] + ([t_lat] if t_lat else []) + [it.height // comp, it.width // comp])
                    latent = comfy.sample.fix_empty_latent_channels(model, latent)
                    if concat:  # channel inputs ride on every call: the sample's picture first, the rest empty
                        from .data.edit import with_concat

                        first = model.model.process_latent_in(ref.float()) if ref is not None else torch.zeros_like(latent)
                        c = torch.cat([first] + [torch.zeros_like(latent)] * (concat - 1), dim=1)
                        pos, neg = with_concat(pos, c), with_concat(neg, c)
                    noise = comfy.sample.prepare_noise(latent, it.seed)
                    def cb(step, x0, _x, total, _i=i):
                        progress(_i, step + 1, total, x0)

                    def draw(m):
                        # as in training: bypass adapters keep their own (fp32) weights beside the
                        # model's bf16 activations, and only autocast reconciles the two
                        with torch.autocast(t.device.type, dtype=t.train_dtype, enabled=t.train_dtype != torch.float32):
                            o = comfy.sample.sample(m, noise, steps, s.cfg, s.sampler, s.scheduler, pos, neg, latent,
                                                    sigmas=sigmas, seed=it.seed, disable_pbar=True, callback=cb)
                        img = vae.decode(first_stream(o))
                        return o, img.reshape(-1, *img.shape[-3:])

                    sweep = SLIDER_SWEEP if t.spec.slider.kind and manual is None and t.step > 0 and not t_lat else None
                    if sweep:  # a slider: the same sample at each strength, side by side
                        shots = []
                        for st in sweep:
                            with self._at_strength(t, model, st) as m:
                                shots.append(draw(m)[1][0])
                        strip = _strip(shots, sweep)
                    else:
                        out, img = draw(model)  # (frames, H, W, 3)
                    if t_lat:
                        from .data import video

                        name = f"manual_{manual:03d}_{'base' if base else f'step{t.step:06d}'}.mp4" if manual is not None else f"step{t.step:06d}_{i:02d}.mp4"
                        video.write(Path(out_dir) / name, img, s.fps, _sound(t, out))
                    else:
                        name = f"manual_{manual:03d}_{'base' if base else f'step{t.step:06d}'}.png" if manual is not None else f"step{t.step:06d}_{i:02d}.png"
                        pic = strip if sweep else Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
                        pic.save(os.path.join(out_dir, name))
                    if t.step == 0 and manual is None:
                        self._baseline_keep(t, i, Path(out_dir) / name)
                    row = {
                        "run_id": t.run_id, "step": t.step, "index": i, "prompt": self.prompts[i],
                        "filename": name, "subfolder": f"zentrainer/samples/{t.run_id}", "type": "output",
                        "model": s.model_name or None,
                    }
                    if sweep:
                        row["sweep"] = list(sweep)
                    if manual is not None:
                        row.update(manual=True, index=manual, width=self.items[i].width, height=self.items[i].height,
                                   seed=self.items[i].seed, frames=self.items[i].frames, base=base,
                                   sampling={"sampler": s.sampler, "scheduler": "custom" if s.sigmas else s.scheduler,
                                             "steps": steps, "cfg": s.cfg, **({"sigmas": list(s.sigmas)} if s.sigmas else {})})
                    # recorded as each one lands, so the UI swaps that tile now, not after the pass
                    rows.append(row)
                    t.emit(events.SAMPLE, row)
                    if t.store:
                        t.store.append(t.run_id, "samples", row)
                        self._like(t, row)
        finally:
            t.network.assist = True
            mm.EXTRA_RESERVED_VRAM = reserved
            if guided is not None:
                model.model_options = plain_options
            if not stay:
                t.exit_inference()
                mm.soft_empty_cache()
            t.emit(events.SAMPLE_PROGRESS, {"run_id": t.run_id, "step": t.step, "phase": "done", "count": count})

        return rows


class Validation(Callback):
    """Loss on a fixed set: the same images at the same noise levels with the same noise
    every time, so the curve moves only when the LoRA does. Runs before training too."""

    def __init__(self):
        self._set: list[tuple[int, list[float], list[torch.Tensor]]] = []

    def on_start(self, t):
        v = t.spec.validation
        if not v.every:
            return
        import random

        rng = random.Random(v.seed)
        held = getattr(t, "val_dataset", None)  # validation.dataset: pictures training never sees
        ds = held or t.dataset
        pool = list(range(len(ds.samples)))
        if held is not None:  # each picture once: a kohya "N_name" folder repeats them
            first: dict[int, int] = {}
            pool = [first.setdefault(id(sm), i) for i, sm in enumerate(ds.samples) if id(sm) not in first]
        picks = rng.sample(pool, min(v.images, len(pool)))
        gen = torch.Generator().manual_seed(v.seed)
        percents = [1.0 - (k + 0.5) / v.noise_levels for k in range(v.noise_levels)]  # spread over the schedule
        for i in picks:
            shape = t.streams.packed_shape(tuple(ds.samples[i].latent.shape))
            noises = [torch.randn(shape, generator=gen) for _ in percents]
            self._set.append((i, percents, noises))
        if t.step == 0:
            self.evaluate(t)

    def on_step(self, t):
        if self._set and (t.step % t.spec.validation.every == 0 or _scores_saves(t, "val_loss")):
            self.evaluate(t)

    def on_end(self, t, status):
        if self._set and status == "done" and t.step % t.spec.validation.every:
            self.evaluate(t)

    def evaluate(self, t) -> float | None:
        if not self._set:
            return None
        from . import optim

        import comfy.model_management as mm

        ms = t.denoiser.model_sampling
        losses, per_t = [], {}
        optim.set_mode(t.optimizer, training=False)
        # the model as training runs it, also for the run's last pass (on_end, after training mode is
        # off): ComfyUI's sampling paths compute a slightly different loss, and some don't run here
        # (Qwen-Image 2.1's prefix cache failed on a validation set's several pictures)
        was, mm.in_training = mm.in_training, True
        try:
            self._run(t, ms, losses, per_t)
        finally:
            mm.in_training = was
            optim.set_mode(t.optimizer, training=True)
        return self._report(t, losses, per_t)

    def _run(self, t, ms, losses, per_t) -> None:
        with torch.no_grad(), t.held_out():
            for i, percents, noises in self._set:
                x0, weight, shapes = t.batch([i])
                cond = t.fixed_cond(i, x0, shapes)
                for p, noise in zip(percents, noises):
                    sigma = torch.tensor([float(ms.percent_to_sigma(p))], device=t.device)
                    x_t = t.objective.noisy(ms, sigma, noise.to(t.device), x0)
                    with torch.autocast(t.device.type, dtype=t.train_dtype, enabled=t.train_dtype != torch.float32):
                        pred = t.denoiser(x_t, sigma, [cond])
                    loss = float(t.objective.loss(x_t, x0, pred, sigma, weight).mean())
                    losses.append(loss)
                    per_t.setdefault(round(1 - p, 3), []).append(loss)

    def _report(self, t, losses, per_t) -> float:
        val = sum(losses) / len(losses)
        row = {"run_id": t.run_id, "step": t.step, "val_loss": val, "held_out": bool(t.spec.validation.dataset),
               "by_noise": {str(k): sum(v) / len(v) for k, v in sorted(per_t.items())}}
        t.emit(events.VALIDATION, row)
        if t.store:
            t.store.append(t.run_id, "validation", row)
        return val


def _scores_saves(t, rule: str) -> bool:
    """A scheduled save this step whose score output.keep ranks it by: samples (likeness) or
    validation have to land on it too, whatever their own interval."""
    every = t.spec.output.save_every
    keep = t.spec.output.keep.removesuffix("_new_best")  # "likeness_new_best" scores saves as "likeness" does
    return keep == rule and bool(every) and t.step % every == 0 and t.step < t.spec.optim.steps


class Keep(Callback):
    """Removes the scheduled checkpoints output.keep doesn't keep (harness/keep.py), at each save
    and at the end. After Samples and Validation in the callbacks: their scores for a save land
    first."""

    def on_step(self, t):
        if t.spec.output.save_every and t.step % t.spec.output.save_every == 0:
            gone = self._prune(t)
            if gone:  # said on the run page; the run's status is the trainer's to set
                t.emit(events.STATUS, {"run_id": t.run_id, "status": "training", "step": t.step,
                                       "message": f"removed {len(gone)} checkpoint{'s' if len(gone) != 1 else ''}: "
                                                  f"{gone[-1][1]}"})

    def on_end(self, t, status):
        self._prune(t)

    def _prune(self, t) -> list:
        from . import keep
        from .trainer import state_dir

        out = t.spec.output
        if out.keep == "all" or not t.store:
            return []
        rows = t.store.read(t.run_id, "checkpoints")
        gone = keep.to_prune(rows, out.keep, out.keep_count, t.store.read(t.run_id, "validation"),
                             t.store.read(t.run_id, "likeness"))
        if not gone:
            return []
        removed = []
        for row, why in gone:
            freed = keep.remove(row, t.run_id, state_dir(t.run_id))
            if freed is None:
                continue
            removed.append((row, why))
            t.store.append(t.run_id, "pruned", {**row, "reason": why, "at_step": t.step})
            logging.info("[ZenTrainer] removed the step %d checkpoint (%.0f MB): %s", row["step"], freed / 1024 ** 2,
                         why)
        paths_gone = {row["path"] for row, _ in removed}
        t.store.rewrite(t.run_id, "checkpoints", [c for c in rows if c.get("path") not in paths_gone])
        return removed


def _prompt_cache(t, prompts: list[str], negative: str):
    import hashlib

    enc, root = getattr(t, "encoder_id", None), getattr(t, "cache_dir", None)
    if enc is None or root is None:
        return None
    key = hashlib.sha256(json.dumps([enc, prompts, negative]).encode()).hexdigest()[:20]
    return root / "prompts" / f"{key}.pt"


def _sample_text(t, prompt: str) -> str:
    """A sample prompt as the model gets it: [trigger] substituted, and on an edit run with an
    instruction template, wrapped the way the captions are."""
    trig = t.spec.dataset.trigger
    text = prompt.replace("[trigger]", trig) if trig else prompt.replace("[trigger]", "")
    d = t.spec.dataset
    if d.edit and d.edit_template:
        from .data.edit import instruction

        text = instruction(text, d.edit_template)
    return text


def _control_path(t, control: str) -> Path:
    import folder_paths

    inp = Path(folder_paths.get_input_directory())
    ds = paths.find_dataset(t.spec.dataset.folder)
    path = next((p for p in (ds / control, inp / control, Path(control)) if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"sample control picture {control!r} not found (in the dataset folder or ComfyUI's input)")
    return path


def _special(t, it):
    """How a sample with an input picture is encoded, as (cache stamp, fn(clip, text, negative) ->
    (cond, negative or None)); None for a plain prompt.
    - MiniMax H3: ComfyUI's own reference (edit runs) or image-to-video (keyframes) node, the
      sample's size and length; the negative stays plain.
    - grounded edits: the encoder sees the picture with the prompt and with the negative."""
    d = t.spec.dataset
    if not it.control or not (d.edit or (getattr(t, "h3", False) and d.video_keyframes)):
        return None
    path = _control_path(t, it.control)
    base = f"{path}|{path.stat().st_mtime_ns}"
    if getattr(t, "h3", False):
        from .data import edit

        size = (it.width, it.height, it.frames)
        if d.edit:
            def ref(clip, text, _neg):
                return edit.h3_reference_cond(clip, t.models.vae, t.models.audio_vae, text, [path], *size), None
            return f"h3ref|{base}|{size}", ref

        def first(clip, text, _neg):
            px = edit._image(path)
            return edit.h3_keyframe_cond(clip, t.models.vae, text, px, None, *size), None
        return f"h3first|{base}|{size}", first
    if d.edit_grounded and d.edit_method != "concat":
        from .data.edit import encode_grounded, grounding_pixels

        def grounded(clip, text, negative):
            px = [grounding_pixels(path, clip)]
            return encode_grounded(clip, text, px), encode_grounded(clip, negative, px)
        return f"grounded|{base}", grounded
    return None


def _control(t, it, comp: int):
    """An edit sample's input picture as a raw latent, and the sample resized to its aspect (the
    same area as the sample's size): edits come out the shape of what they edit."""
    import dataclasses

    from PIL import Image

    from .data import load_image, pixel_mode

    path = _control_path(t, it.control)
    with Image.open(path) as im:
        w0, h0 = im.size
    step = comp * 2
    area = it.width * it.height
    w = max(step, round((area * w0 / h0) ** 0.5 / step) * step)
    h = max(step, round((area * h0 / w0) ** 0.5 / step) * step)
    with torch.no_grad():
        ref = t.models.vae.encode(load_image(path, (w, h), pixel_mode(t.models.vae)).unsqueeze(0))[:1].detach()
    return ref, dataclasses.replace(it, width=w, height=h)


def _previewer(t):
    """The sharpest previewer installed for this model: its tiny VAE (models/vae_approx: taeh3,
    taef2, taesdxl...) when present, whatever ComfyUI's own --preview-method says (it launches
    with tiny-VAE previews off), else the cheap latent-to-RGB projection. Kept for the run."""
    cached = getattr(t, "_previewer", None)
    if cached is not None:
        return cached
    try:
        import latent_preview

        fmt = t.models.patcher.model.latent_format
        previewer = _tiny_previewer(t, fmt) or latent_preview.get_previewer(t.device, fmt)
        if previewer is None and fmt.latent_rgb_factors is not None:
            previewer = latent_preview.Latent2RGBPreviewer(
                fmt.latent_rgb_factors, fmt.latent_rgb_factors_bias, fmt.latent_rgb_factors_reshape)
        t._previewer = previewer
        return previewer
    except Exception:
        return None


def _video_tae_previewer(vae, device):
    """TAEHV previews decoded by the tiny model itself, kept on the GPU (21 MB) for the run.
    ComfyUI's VAE.decode goes through model management, which re-stages it on every sampler
    step because loading the sampled model back pushes it out."""
    import latent_preview

    model = vae.first_stage_model.to(device=device, dtype=vae.vae_dtype).eval()

    class Direct(latent_preview.LatentPreviewer):
        takes_video = True

        @torch.no_grad()
        def decode_latent_to_preview(self, x0):
            out = model.decode(x0[:1, :, :1].to(device=device, dtype=vae.vae_dtype))
            return latent_preview.preview_to_image(vae.process_output(out.float()).movedim(1, -1)[0][0], do_scale=False)

    return Direct()


def _tiny_previewer(t, fmt):
    import comfy.utils
    import folder_paths
    import latent_preview

    name = getattr(fmt, "taesd_decoder_name", None)
    if not name:
        return None
    file = next((f for f in folder_paths.get_filename_list("vae_approx") if f.startswith(name)), None)
    path = folder_paths.get_full_path("vae_approx", file) if file else None
    if not path:
        return None
    try:
        if name in latent_preview.VIDEO_TAES:
            from comfy.sd import VAE

            vae = VAE(comfy.utils.load_torch_file(path))
            vae.first_stage_model.show_progress_bar = False
            return _video_tae_previewer(vae, t.device)
        from comfy.taesd.taesd import TAESD

        return latent_preview.TAESDPreviewerImpl(TAESD(None, path, latent_channels=fmt.latent_channels).to(t.device))
    except Exception:
        logging.warning("[ZenTrainer] tiny VAE %s didn't load; previews use the latent colours", file, exc_info=True)
        return None


def _preview_url(previewer, x0) -> str | None:
    import base64
    import io

    try:
        x0 = first_stream(x0)[:1]
        # a video latent: the video tiny VAEs take it whole (and decode its first frame);
        # the rest want one frame
        if x0.ndim == 5 and not getattr(previewer, "takes_video", False):
            x0 = x0[:, :, 0]
        img = previewer.decode_latent_to_preview(x0)
        img.thumbnail((384, 384))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def write_metadata(spec_dict: dict, step: int) -> dict[str, str]:
    meta = {"zentrainer.spec": json.dumps(spec_dict), "zentrainer.step": str(step)}
    net = spec_dict.get("network") or {}
    if net.get("base_lora"):  # trained on top of it: it belongs beside this LoRA wherever it's used
        meta["zentrainer.base_lora"] = net["base_lora"]
        meta["zentrainer.base_strength"] = str(net.get("base_strength", 1.0))
    sl = spec_dict.get("slider") or {}
    if sl.get("kind"):  # a slider: what each end does, for whatever loads it to label them
        meta["zentrainer.slider"] = json.dumps({"kind": sl["kind"], "positive": sl.get("positive", ""),
                                                "negative": sl.get("negative", ""), "neutral": sl.get("neutral", ""),
                                                "range": [-1.0 if sl.get("directions", "both") == "both" else 0.0, 1.0]})
    return meta


def _latent_frames(vae, frames: int) -> int:
    """Latent frames for a `frames`-long sample from a video VAE, 0 for a still (or an image VAE)."""
    down = getattr(vae, "downscale_ratio", None)
    if frames <= 1 or not (isinstance(down, tuple) and callable(down[0])):
        return 0
    return int(down[0](frames))


def _sound(t, out):
    """A joint audio-video sample's soundtrack ((channels, samples), rate), when the run has the
    model's audio VAE; else None (the sample is saved silent)."""
    avae = getattr(t.models, "audio_vae", None)
    if avae is None or not getattr(out, "is_nested", False):
        return None
    streams = out.unbind()
    if len(streams) < 2:
        return None
    wave = avae.decode(streams[-1]).movedim(-1, 1)[:1]  # (1, channels, samples)
    std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0  # as ComfyUI's VAEDecodeAudio levels it
    wave = wave / std.clamp(min=1.0)
    rate = getattr(avae, "audio_sample_rate_output", getattr(avae, "audio_sample_rate", 44100))
    return wave[0].float().cpu(), int(rate)


def _out_dir(t) -> str:
    # rows still say subfolder zentrainer/samples/<run>: an address paths.resolve_view finds
    # wherever the ZenTrainer folder put it
    d = str(paths.output_root() / "samples" / t.run_id)
    os.makedirs(d, exist_ok=True)
    return d


def mm_empty() -> None:
    import comfy.model_management as mm

    mm.soft_empty_cache()


def _strip(shots: list, strengths) -> "Image.Image":
    """Pictures (H, W, 3 or 4 in 0-1) side by side, each under its strength: a slider's sample.
    Four-channel pictures keep their transparency (the labels and gaps stay solid)."""
    from PIL import Image, ImageDraw, ImageFont

    pics = [Image.fromarray((x.clamp(0, 1).cpu().numpy() * 255).astype("uint8")) for x in shots]
    w, h = pics[0].size
    band, gap = max(24, h // 16), max(4, w // 128)
    out = Image.new(pics[0].mode, (len(pics) * w + (len(pics) - 1) * gap, h + band), (18, 18, 20, 255))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=int(band * 0.6))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for k, (pic, st) in enumerate(zip(pics, strengths)):
        x = k * (w + gap)
        out.paste(pic, (x, band))
        label = "0" if st == 0 else f"{st:+g}"  # ASCII: the default font has no U+2212
        tw = draw.textlength(label, font=font)
        draw.text((x + (w - tw) / 2, band * 0.18), label, fill=(235, 235, 240), font=font)
    return out


def _try_sampling(s, a: dict):
    """A try's own sampler, scheduler, steps, cfg or sigmas laid over the run's sample settings
    (s), or None when it sets none. What ComfyUI doesn't have, or doesn't parse, is left as the
    run's, with a warning: a typo shouldn't cost the render."""
    import dataclasses

    import comfy.samplers

    from ..spec import SpecError, check_sigmas

    got = {}
    if a.get("sampler"):
        if a["sampler"] in comfy.samplers.KSampler.SAMPLERS:
            got["sampler"] = str(a["sampler"])
        else:
            logging.warning("[ZenTrainer] try: no sampler %r here, using the run's", a["sampler"])
    if a.get("scheduler") and a["scheduler"] != "custom":
        if a["scheduler"] in comfy.samplers.KSampler.SCHEDULERS:
            got["scheduler"] = str(a["scheduler"])
        else:
            logging.warning("[ZenTrainer] try: no scheduler %r here, using the run's", a["scheduler"])
    for key, kind, lo, hi in (("steps", int, 1, 200), ("cfg", float, 0.0, 30.0)):
        if a.get(key) is not None:
            try:
                got[key] = min(hi, max(lo, kind(a[key])))
            except (TypeError, ValueError):
                logging.warning("[ZenTrainer] try: %s %r isn't a number, using the run's", key, a[key])
    if a.get("sigmas") is not None:
        try:
            got["sigmas"] = check_sigmas(a["sigmas"])
        except SpecError as e:
            logging.warning("[ZenTrainer] try: %s; using the run's", e)
    elif got.get("scheduler"):  # a scheduler of its own replaces the run's custom sigmas
        got["sigmas"] = []
    return dataclasses.replace(s, **got) if got else None
