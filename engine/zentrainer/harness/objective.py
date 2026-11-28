"""What the model is trained to predict, at which noise levels, under which loss.

Family-agnostic by construction: everything goes through ComfyUI's `model_sampling`,
which already knows each model's noise schedule (eps / v / flow, shift, sigma range).

ComfyUI models always return a denoised estimate x0. For a forward-noised sample

    x_t = noise_scaling(sigma, noise, x0)        # x0 + σ·ε  or  (1-σ)·x0 + σ·ε

the residual r = (x_t - x0) / σ is ε for SD-style models and the velocity (ε - x0) for
flow models — each family's own training target. "native" compares residuals,
"x0" compares denoised images (ComfyUI's native trainer). They differ by a 1/σ²
per-sample weight, which matters most at low noise.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ..spec import ObjectiveSpec

LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mse": lambda a, b: F.mse_loss(a, b, reduction="none"),
    "l1": lambda a, b: F.l1_loss(a, b, reduction="none"),
    "huber": lambda a, b: F.huber_loss(a, b, reduction="none"),
    "smooth_l1": lambda a, b: F.smooth_l1_loss(a, b, reduction="none"),
}


GOLDEN = 0.6180339887498949


def biased(u: torch.Tensor, spec: ObjectiveSpec) -> torch.Tensor:
    """A uniform draw leaned toward one end, as a quantile of the noise distribution (1 = the
    most noise). AI Toolkit's content/style bias (from diffusers; T2I-Adapter §3.4) cubes it:
    "high_noise" keeps 1 - u³ (half the steps in the noisiest eighth), "low_noise" u³."""
    if spec.bias == "high_noise":
        return 1.0 - u ** spec.bias_power
    if spec.bias == "low_noise":
        return u ** spec.bias_power
    return u


def sample_percents(spec: ObjectiveSpec, n: int, generator: torch.Generator | None = None,
                    start: int | None = None, seed: int = 0) -> torch.Tensor:
    """Noise levels as "percent of the schedule" in (0, 1): 0 = full noise, 1 = clean.

    Each level is a quantile of the distribution (uniform, or logit-normal: AI Toolkit's
    "sigmoid"; "scheduler" is uniform over a scheduler's steps), so a bias reshapes any of them
    the same way. With `spec.spread` and `start` (how many
    were drawn before it, so a resume carries on the same sequence) the quantiles walk a
    golden-ratio sequence instead of random draws: consecutive steps land far apart, and even
    batch 1 covers the schedule evenly within a few steps rather than by luck."""
    if spec.spread and start is not None:
        k = start + torch.arange(n, dtype=torch.float64)
        u = ((seed * GOLDEN + k * GOLDEN) % 1.0).float()
    else:
        u = torch.rand(n, generator=generator)
    q = biased(u, spec).clamp(1e-6, 1.0 - 1e-6)
    if spec.timesteps == "logit_normal":
        t = torch.sigmoid(torch.special.ndtri(q) * spec.logit_std + spec.logit_mean)  # t = noise fraction
        if spec.uniform_mix > 0:
            pick = torch.rand(n, generator=generator) < spec.uniform_mix
            t = torch.where(pick, q, t)
    else:
        t = q
    # keep away from exact endpoints: σ=0 has no residual, σ=max can be inf for eps models
    return (1.0 - t).clamp(1e-3, 1.0 - 1e-3)


# AI Toolkit's "weighted" timesteps: a loss weight per noise level, measured on Flex.1 and seen on
# other flow models (ostris/ai-toolkit, toolkit/timestep_weighing, MIT). Its 1000 values averaged
# down to σ = 1, 0.975, ... 0; mean 1. Heaviest at σ ≈ 0.85 (1.5x), lightest near clean (0.45x).
FLEX_WEIGHTS = (
    0.986, 1.205, 1.369, 1.465, 1.503, 1.519, 1.509, 1.481, 1.443, 1.401, 1.357, 1.311, 1.265, 1.219,
    1.174, 1.128, 1.084, 1.042, 1.002, 0.963, 0.927, 0.893, 0.861, 0.832, 0.806, 0.783, 0.763, 0.746,
    0.732, 0.722, 0.716, 0.714, 0.716, 0.721, 0.728, 0.731, 0.716, 0.667, 0.586, 0.497, 0.449,
)


class Balance:
    """The "balanced" weighting, measured on the model being trained instead of copied: a running
    average of the loss in each of `BANDS` stretches of the schedule, and each step weighted by
    the average over the bands' over its own band's, so every noise level pulls on the LoRA about
    equally however hard it is. A table version of EDM2's learned loss weighting (Karras et al.
    2024, "Analyzing and Improving the Training Dynamics of Diffusion Models"). Bands with fewer
    than WARM losses weigh 1, so it starts as the plain loss."""

    BANDS, DECAY, WARM, LIMIT = 20, 0.95, 5, 4.0

    def __init__(self):
        self.avg = torch.zeros(self.BANDS, dtype=torch.float64)
        self.seen = torch.zeros(self.BANDS, dtype=torch.long)

    def __call__(self, percents: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        band = (percents.detach().float().cpu() * self.BANDS).long().clamp(0, self.BANDS - 1)
        warm = self.seen >= self.WARM
        ref = self.avg[warm].mean() if warm.any() else None
        w = torch.ones(len(band), dtype=torch.float64)
        for j, (b, value) in enumerate(zip(band.tolist(), loss.detach().float().cpu().tolist())):
            if ref is not None and warm[b]:
                w[j] = (ref / self.avg[b].clamp_min(1e-8)).clamp(1 / self.LIMIT, self.LIMIT)
            # a plain mean over the first few, then a running average
            k = self.seen[b].item()
            self.avg[b] = value if k == 0 else self.avg[b] + (value - self.avg[b]) * max(1 - self.DECAY, 1 / (k + 1))
            self.seen[b] += 1
        return w.float()

    def state_dict(self) -> dict:
        return {"avg": self.avg.clone(), "seen": self.seen.clone()}

    def load_state_dict(self, d: dict) -> None:
        self.avg, self.seen = d["avg"].clone(), d["seen"].clone()


def timestep_weight(spec: ObjectiveSpec, sigma: torch.Tensor) -> torch.Tensor | None:
    """Per-sample loss weight for each σ (a flow model's noise fraction) from a fixed curve, or None."""
    if spec.weighting != "flex":
        return None
    table = torch.tensor(FLEX_WEIGHTS)
    x = (1.0 - sigma.detach().float().cpu().clamp(0.0, 1.0)) * (len(table) - 1)
    i = x.floor().long().clamp(max=len(table) - 2)
    f = x - i
    return table[i] * (1 - f) + table[i + 1] * f


def scheduler_sigmas(model_sampling, name: str, steps: int) -> torch.Tensor:
    """The σ a ComfyUI scheduler visits sampling this model in `steps` steps, full noise first
    (the closing 0, which no step denoises from, left off)."""
    import comfy.samplers

    if name not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError(f"objective.scheduler is one of ComfyUI's: {', '.join(comfy.samplers.SCHEDULER_NAMES)}")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, name, steps).float().cpu()
    return sigmas[sigmas > 0]


def sigmas_for(model_sampling, percents: torch.Tensor, shift: float = 0.0,
               schedule: torch.Tensor | None = None) -> torch.Tensor:
    """σ for each percent: the model's own schedule, or for flow models with `shift` set, the
    flow schedule re-skewed by that shift instead of the sampling one. With `schedule` (a
    scheduler's σ, scheduler_sigmas), exactly the levels it samples at: percent 0 its first step."""
    if schedule is not None:
        i = (percents * len(schedule)).long().clamp(0, len(schedule) - 1)
        return schedule[i]
    if shift > 0:
        t = (1.0 - percents).clamp(0.0, 1.0)  # noise fraction
        return shift * t / (1 + (shift - 1) * t)  # comfy.model_sampling.time_snr_shift
    return torch.tensor([float(model_sampling.percent_to_sigma(float(p))) for p in percents])


def is_flow(model_sampling) -> bool:
    """σ in [0, 1] with x_t = (1-σ)x0 + σ·noise: where a training shift means something."""
    import comfy.model_sampling as cms

    # ModelSamplingFlux (Flux, Krea 2, Chroma) is flow matching too, just not a ModelSamplingDiscreteFlow
    flow = (cms.ModelSamplingDiscreteFlow, cms.ModelSamplingFlux)
    return isinstance(model_sampling, flow) and isinstance(model_sampling, cms.CONST)


class Objective:
    def __init__(self, spec: ObjectiveSpec):
        self.spec = spec
        self.loss_fn = LOSSES[spec.loss]
        self.balance = Balance() if spec.weighting == "balance" else None
        self.drawn: list[tuple[float, float, float]] = []  # (percent, σ, loss) since the trainer took them
        self._unread: list[tuple] = []  # this step's (percents, σ, loss), still on the GPU: see settle

    def start_step(self) -> None:
        self.drawn, self._unread = [], []

    def settle(self, *values: torch.Tensor) -> list[float]:
        """The step's numbers off the GPU in one read: `values` (one-element tensors, returned as
        floats) and what each sample drew and lost, into `drawn`. Reading any of them makes the CPU
        wait for the GPU to finish, so a step does it once, at its end."""
        unread, self._unread = self._unread, []
        parts = [v.detach().float().reshape(1) for v in values] + [t for _, s, l in unread for t in (s, l)]
        if not parts:
            return []
        flat = torch.cat([t.to(parts[0].device) for t in parts]).tolist()
        out, rest = flat[:len(values)], flat[len(values):]
        for p, s, _ in unread:
            n = len(s)
            self.drawn += zip(p.tolist(), rest[:n], rest[n:2 * n])
            rest = rest[2 * n:]
        return out

    def noisy(self, model_sampling, sigma: torch.Tensor, noise: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # noise_scaling reshapes a (B,) sigma itself, as in ComfyUI's samplers
        return model_sampling.noise_scaling(sigma.to(x0.device), noise, x0, False)

    def _err(self, x_t: torch.Tensor, x0: torch.Tensor, x0_pred: torch.Tensor,
             sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x0_pred = x0.float(), x0_pred.float()
        if self.spec.target == "native":
            s = _bcast(sigma, x0).float()
            return (x_t.float() - x0_pred) / s, (x_t.float() - x0) / s
        return x0_pred, x0

    def loss(self, x_t: torch.Tensor, x0: torch.Tensor, x0_pred: torch.Tensor, sigma: torch.Tensor,
             weight: torch.Tensor | None = None) -> torch.Tensor:
        """Per-sample loss, shape (B,). `weight` (B, h, w) weights latent positions (masked
        training); each sample's loss is normalised by its total weight so masked and
        unmasked samples stay on the same scale."""
        pred, target = self._err(x_t, x0, x0_pred, sigma)
        per_elem = self.loss_fn(pred, target)
        return _weighted_mean(per_elem, weight)

    def train_loss(self, x_t: torch.Tensor, x0: torch.Tensor, x0_pred: torch.Tensor, sigma: torch.Tensor,
                   weight: torch.Tensor | None = None, percents: torch.Tensor | None = None) -> torch.Tensor:
        """The loss a step trains on: `loss`, with optional 1-level 2D Haar high-frequency detail
        weighting (`haar_hf`) and per-noise-level weighting (`percents`: where each sample sits in
        the schedule, for "balance"). Validation keeps the plain loss, so its curve means the same
        with or without."""
        pred, target = self._err(x_t, x0, x0_pred, sigma)
        loss = _weighted_mean(self.loss_fn(pred, target), weight)
        if percents is not None:  # what each sample drew and lost, plain: the run page's noise chart
            # (read at the step's end, by settle: reading it here would stall the GPU before the backward)
            self._unread.append((percents.detach().float().cpu(), sigma.detach().float().reshape(-1),
                                 loss.detach().float().reshape(-1)))
        if self.spec.haar_hf > 0:
            hf = _haar_hf_loss(pred - target, weight)
            if hf is not None:
                loss = loss + self.spec.haar_hf * hf
        if self.balance is not None and percents is not None:
            return loss * self.balance(percents, loss).to(loss)
        w = timestep_weight(self.spec, sigma)
        return loss if w is None else loss * w.to(loss)


def _weighted_mean(per_elem: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return per_elem.reshape(per_elem.shape[0], -1).mean(dim=1)
    w = weight.to(per_elem)
    w = w.reshape(w.shape[0], *([1] * (per_elem.ndim - w.ndim)), *w.shape[1:])  # (B,1,[1,]h,w)
    w = w.expand_as(per_elem)
    b = per_elem.shape[0]
    return (per_elem * w).reshape(b, -1).sum(1) / w.reshape(b, -1).sum(1).clamp_min(1e-8)


def _haar_hf_loss(err: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor | None:
    """Per-sample 1-level 2D Haar high-frequency energy (LH² + HL² + HH²) on 4D/5D latents."""
    if err.ndim < 4:
        return None
    h, w = err.shape[-2] & ~1, err.shape[-1] & ~1
    if h < 2 or w < 2:
        return None
    e = err[..., :h, :w]
    e00, e01 = e[..., 0::2, 0::2], e[..., 0::2, 1::2]
    e10, e11 = e[..., 1::2, 0::2], e[..., 1::2, 1::2]
    lh = 0.5 * (e00 + e01 - e10 - e11)
    hl = 0.5 * (e00 - e01 + e10 - e11)
    hh = 0.5 * (e00 - e01 - e10 + e11)
    hf = lh.square() + hl.square() + hh.square()
    if weight is not None:
        wm = weight[..., :h, :w].to(hf)
        wm = 0.25 * (wm[..., 0::2, 0::2] + wm[..., 0::2, 1::2] + wm[..., 1::2, 0::2] + wm[..., 1::2, 1::2])
        return _weighted_mean(hf, wm)
    return hf.reshape(hf.shape[0], -1).mean(dim=1)


def _bcast(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return sigma.to(like.device).reshape(sigma.shape[0], *([1] * (like.ndim - 1)))
