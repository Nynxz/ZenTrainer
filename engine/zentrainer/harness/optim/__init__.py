"""Optimizers and LR schedules, as registries so a new one is one entry.

Built-ins need nothing extra; the rest import their package lazily and report as
unavailable (with the pip name) until it's installed (`zt extras optimizers`).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from ...spec import OptimSpec

AUTOMAGIC3_CEILING = 4.0  # Automagic v3's and v4's LR stays under this many times where it started


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): sign of an interpolated momentum. Memory of one moment;
    wants a ~3-10x lower LR than AdamW."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("exp_avg", torch.zeros_like(p))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-group["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def _stochastic_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Write fp32 `src` into bf16 `dst` with stochastic rounding: an update smaller than a
    bf16 step lands with the right probability instead of rounding to nothing."""
    noise = torch.randint_like(src, 0, 1 << 16, dtype=torch.int32)
    bits = src.view(torch.int32) + noise
    dst.copy_((bits & -65536).view(torch.float32))


class Automagic(torch.optim.Optimizer):
    """After ai-toolkit's Automagic: every element has its own learning rate. It starts at
    `lr`, grows by `lr_bump` while the element's update keeps the same sign step to step, and
    shrinks when it flips, within [min_lr, max_lr]: no LR to guess. Second moments are
    factored as in Adafactor (two vectors per matrix), updates are RMS-clipped, and bf16
    parameters get stochastic rounding. The scheduler's multiplier (warmup, cosine) still
    scales the steps."""

    def __init__(self, params, lr=1e-6, min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6, eps=1e-30, clip=1.0,
                 decay_rate=-0.8, weight_decay=0.0):
        super().__init__(params, {"lr": lr, "min_lr": min_lr, "max_lr": max_lr, "lr_bump": lr_bump, "eps": eps,
                                  "clip": clip, "decay_rate": decay_rate, "weight_decay": weight_decay, "initial_lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            scale = group["lr"] / group["initial_lr"]  # the schedule's multiplier
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["lr_mask"] = torch.full_like(grad, group["lr"])
                    st["last_sign"] = torch.zeros_like(grad, dtype=torch.bool)
                    if grad.dim() >= 2:
                        st["row"] = torch.zeros(grad.shape[:-1], device=grad.device)
                        st["col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device)
                    else:
                        st["sq"] = torch.zeros_like(grad)
                st["step"] += 1
                beta2 = 1.0 - st["step"] ** group["decay_rate"]
                g2 = grad * grad + group["eps"]
                if grad.dim() >= 2:
                    st["row"].mul_(beta2).add_(g2.mean(-1), alpha=1 - beta2)
                    st["col"].mul_(beta2).add_(g2.mean(-2), alpha=1 - beta2)
                    r = (st["row"] / st["row"].mean(-1, keepdim=True)).rsqrt().unsqueeze(-1)
                    update = grad * r * st["col"].unsqueeze(-2).rsqrt()
                else:
                    st["sq"].mul_(beta2).add_(g2, alpha=1 - beta2)
                    update = grad * st["sq"].rsqrt()
                update.div_((update.pow(2).mean().sqrt() / group["clip"]).clamp_(min=1.0))
                sign = update > 0
                if st["step"] > 1:  # same direction as last time: bigger steps; flipped: smaller
                    agree = torch.where(sign == st["last_sign"], group["lr_bump"], -group["lr_bump"])
                    st["lr_mask"].add_(agree).clamp_(group["min_lr"], group["max_lr"])
                st["last_sign"] = sign
                lr = st["lr_mask"] * scale
                target = p.float()
                if group["weight_decay"]:
                    target.mul_(1 - lr * group["weight_decay"])
                target.sub_(update * lr)
                if p.dtype == torch.bfloat16:
                    _stochastic_copy(p, target)
                else:
                    p.copy_(target)
        return loss

    def mean_lr(self) -> float:
        """The average per-element LR, for the metrics (the group LR is only the start)."""
        masks = [st["lr_mask"] for st in self.state.values() if "lr_mask" in st]
        if not masks:
            return self.param_groups[0]["lr"]
        scale = self.param_groups[0]["lr"] / self.param_groups[0]["initial_lr"]
        return float(sum(m.mean() for m in masks) / len(masks)) * scale


@dataclass(frozen=True)
class OptimizerDef:
    id: str
    label: str
    build: Callable[[list, OptimSpec], torch.optim.Optimizer]
    package: str | None = None  # import name; None = built in
    pip: str | None = None
    lr_hint: str = "1e-4"
    note: str = ""
    schedule_free: bool = False  # needs optimizer.train()/eval() around evaluation
    adaptive: bool = False  # steers its own LR: the schedule stays flat


# The optional packages are imported when picked, not at load: each is an extra (zt extras optimizers).

def _adamw8bit(p, s):
    from bitsandbytes.optim import AdamW8bit

    return AdamW8bit(p, lr=s.learning_rate, weight_decay=s.weight_decay)


def _lion8bit(p, s):
    from bitsandbytes.optim import Lion8bit

    return Lion8bit(p, lr=s.learning_rate, weight_decay=s.weight_decay)


def _automagic3(p, s):
    from .automagic3_batched import Automagic3Batched

    return Automagic3Batched(p, lr=s.learning_rate, weight_decay=s.weight_decay, fused=False,
                             **({"min_lr": s.learning_rate * s.lr_floor} if s.lr_floor > 0 else {}))


def _automagic4(p, s):
    from .automagic4_batched import Automagic4Batched

    return Automagic4Batched(p, lr=s.learning_rate, weight_decay=s.weight_decay, fused=False,
                             **({"min_lr": s.learning_rate * s.lr_floor} if s.lr_floor > 0 else {}))


def _prodigy(p, s):
    from prodigyopt import Prodigy

    return Prodigy(p, lr=s.learning_rate, weight_decay=s.weight_decay, safeguard_warmup=True, use_bias_correction=True)


def _schedule_free(p, s):
    from schedulefree import AdamWScheduleFree

    return AdamWScheduleFree(p, lr=s.learning_rate, weight_decay=s.weight_decay, warmup_steps=s.warmup_steps)


def _came(p, s):
    from came_pytorch import CAME

    return CAME(p, lr=s.learning_rate, weight_decay=s.weight_decay)


OPTIMIZERS: dict[str, OptimizerDef] = {d.id: d for d in [
    OptimizerDef("AdamW", "AdamW", lambda p, s: torch.optim.AdamW(p, lr=s.learning_rate, weight_decay=s.weight_decay),
                 note="The reliable default."),
    OptimizerDef("Adam", "Adam", lambda p, s: torch.optim.Adam(p, lr=s.learning_rate, weight_decay=s.weight_decay),
                 note="AdamW without decoupled weight decay; same LRs."),
    OptimizerDef("Adafactor", "Adafactor", lambda p, s: torch.optim.Adafactor(p, lr=s.learning_rate, weight_decay=s.weight_decay),
                 lr_hint="1e-4 – 1e-3", note="Factored second moments: least optimizer memory."),
    OptimizerDef("Lion", "Lion", lambda p, s: Lion(p, lr=s.learning_rate, weight_decay=s.weight_decay),
                 lr_hint="1e-5 – 3e-5", note="One moment; use a 3–10× lower LR than AdamW."),
    OptimizerDef("SGD", "SGD (momentum)", lambda p, s: torch.optim.SGD(p, lr=s.learning_rate, weight_decay=s.weight_decay, momentum=0.9),
                 lr_hint="1e-3", note="Plain momentum, no per-weight scaling: needs a much higher LR and patience."),
    OptimizerDef("RMSprop", "RMSprop", lambda p, s: torch.optim.RMSprop(p, lr=s.learning_rate, weight_decay=s.weight_decay),
                 note="Adam's scaling without momentum; rarely better for LoRAs."),
    OptimizerDef("AdamW8bit", "AdamW 8-bit", _adamw8bit, package="bitsandbytes", pip="bitsandbytes",
                 note="AdamW with 8-bit states: about a quarter of the optimizer memory."),
    OptimizerDef("Lion8bit", "Lion 8-bit", _lion8bit, package="bitsandbytes", pip="bitsandbytes",
                 lr_hint="1e-5 – 3e-5", note="Lion with an 8-bit moment: the least memory of the momentum ones."),
    OptimizerDef("Automagic", "Automagic (auto LR)",
                 lambda p, s: Automagic(p, lr=s.learning_rate, weight_decay=s.weight_decay),
                 lr_hint="1e-6 (start)",
                 note="Each weight tunes its own LR (1e-7 to 1e-3). Start at 1e-6, constant schedule."),
    OptimizerDef("Automagic3", "Automagic v3 (auto LR)",
                 _automagic3,
                 lr_hint="1e-4 (start)", adaptive=True,
                 note="Ostris's v3: one LR per group that climbs while updates agree and backs off when they "
                      "flip. It moves only on a steady trend, which small batches rarely give, so start near "
                      "a normal LoRA LR (1e-4); the schedule and warmup are ignored."),
    OptimizerDef("Automagic4", "Automagic v4 (auto LR, experimental)",
                 _automagic4,
                 lr_hint="1e-4 (start)", adaptive=True,
                 note="Ostris's v4, experimental: v3 whose LR settles at a level instead of sinking once "
                      "the trend is learned, and whose most sensitive weights take smaller steps. Start "
                      "near 1e-4; the schedule and warmup are ignored."),
    OptimizerDef("Prodigy", "Prodigy (auto LR)",
                 _prodigy,
                 package="prodigyopt", pip="prodigyopt", lr_hint="1.0",
                 note="Finds its own step size: set LR to 1.0 and leave it."),
    OptimizerDef("AdamWScheduleFree", "AdamW schedule-free",
                 _schedule_free,
                 package="schedulefree", pip="schedulefree", lr_hint="1e-4 – 5e-4",
                 note="No LR schedule needed (use constant).", schedule_free=True),
    OptimizerDef("CAME", "CAME",
                 _came,
                 package="came_pytorch", pip="came-pytorch", lr_hint="1e-4",
                 note="Adafactor-like memory with AdamW-like stability."),
]}


def available(d: OptimizerDef) -> bool:
    return d.package is None or importlib.util.find_spec(d.package) is not None


def catalog() -> list[dict]:
    return [{"id": d.id, "label": d.label, "available": available(d), "pip": d.pip,
             "lr_hint": d.lr_hint, "note": d.note} for d in OPTIMIZERS.values()]


def _schedule(spec: OptimSpec) -> Callable[[int], float]:
    """LR multiplier for optimizer step `i` (0-based)."""
    warm, total = spec.warmup_steps, spec.steps

    def decay(i: int) -> float:
        p = min(1.0, (i - warm) / max(1, total - warm))
        if spec.lr_scheduler == "linear":
            return 1.0 - p
        if spec.lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * p))
        return 1.0

    return lambda i: (i + 1) / warm if i < warm else decay(i)


def build(params: Iterable[torch.nn.Parameter], spec: OptimSpec):
    d = OPTIMIZERS.get(spec.optimizer)
    if d is None:
        raise ValueError(f"unknown optimizer {spec.optimizer!r}; have {', '.join(OPTIMIZERS)}")
    if not available(d):
        raise RuntimeError(f"{d.label} needs `{d.pip}`: run `scripts/zt extras optimizers` (or pip install {d.pip})")
    opt = d.build(list(params), spec)
    if d.adaptive and spec.lr_floor > 0:  # a group at a multiple of the LR floors at that multiple too
        for g in opt.param_groups:
            if "min_lr" in g:
                g["min_lr"] = spec.learning_rate * spec.lr_floor * g.get("lr_mult", 1.0)
    if d.id in ("Automagic3", "Automagic4"):
        # its own ceiling (1e3) never binds: a steady trend (an edit's new input columns, an input
        # dropped half the time) kept it climbing to hundreds and the run blew up
        for g in opt.param_groups:
            g["max_lr"] = spec.learning_rate * AUTOMAGIC3_CEILING * g.get("lr_mult", 1.0)
    if d.schedule_free or d.adaptive:  # its own warmup / LR control; the schedule must stay flat
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(spec))
    return opt, sched


def set_mode(opt: torch.optim.Optimizer, training: bool) -> None:
    """Schedule-free optimizers keep separate train/eval weights; others ignore this."""
    fn = getattr(opt, "train" if training else "eval", None)
    if callable(fn):
        fn()


def current_lr(opt: torch.optim.Optimizer) -> float:
    """The LR steps are taking now: an adaptive optimizer's own, else the first group's."""
    for name in ("mean_lr", "get_avg_learning_rate"):
        fn = getattr(opt, name, None)
        if callable(fn):
            return float(fn())
    return opt.param_groups[0]["lr"]


def set_lr(opt: torch.optim.Optimizer, sched, lr: float) -> None:
    """Change the base LR mid-run; the schedule keeps shaping it."""
    sched.base_lrs = [lr * g.get("lr_mult", 1.0) for g in opt.param_groups]  # LoRA+ keeps its ratio
    for group, base, f in zip(opt.param_groups, sched.base_lrs, sched.lr_lambdas):
        group["lr"] = base * f(sched.last_epoch)


def seek(opt: torch.optim.Optimizer, sched, step: int) -> None:
    """Put a fresh schedule at `step`, as if it had been stepped that many times (a resume with
    no saved scheduler state)."""
    sched.last_epoch = step
    for group, base, f in zip(opt.param_groups, sched.base_lrs, sched.lr_lambdas):
        group["lr"] = base * f(step)
    sched._last_lr = [g["lr"] for g in opt.param_groups]


REWARM_STEPS = 50  # a cold optimizer's ramp back up to the schedule (resume_cold)


def resume_cold(opt: torch.optim.Optimizer, sched, spec: OptimSpec, step: int) -> None:
    """A fresh optimizer and schedule going on at `step` (a resume whose state was lost): the
    schedule carries on where it was, ramped up over REWARM_STEPS, since a fresh optimizer's
    first steps are its largest. Optimizers that set their own LR keep their flat schedule."""
    d = OPTIMIZERS.get(spec.optimizer)
    if d is not None and not (d.adaptive or d.schedule_free):
        sched.lr_lambdas = [lambda i, f=f: f(i) * min(1.0, (i - step + 1) / REWARM_STEPS) for f in sched.lr_lambdas]
    seek(opt, sched, step)


def remap_state(state: dict, old_names: list[list[str]], new_names: list[list[str]]) -> dict:
    """An optimizer state_dict saved over one set of named tensors, renumbered for another: what
    the new set lacks is dropped (states saved while LoRA alphas still trained). Torch numbers a
    state's tensors across its groups in order; `old_names`/`new_names` name them, per group."""
    if [len(g) for g in old_names] == [len(g) for g in new_names] and old_names == new_names:
        return state
    old_idx = {n: i for i, n in enumerate(n for g in old_names for n in g)}
    new_idx = {n: i for i, n in enumerate(n for g in new_names for n in g)}
    per_param = {}
    for name, i in new_idx.items():
        o = old_idx.get(name)
        if o is not None and o in state["state"]:
            per_param[i] = state["state"][o]
    groups = []
    for saved, names in zip(state["param_groups"], new_names):
        groups.append({**saved, "params": [new_idx[n] for n in names]})
    return {"state": per_param, "param_groups": groups}
