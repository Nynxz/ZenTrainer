"""Automagic3Geo: Ostris's Automagic v3 + adapter-native geometry for diffusion LoRAs and LoKrs.

Derived from Ostris's Automagic v3 (ai-toolkit, MIT, Copyright (c) 2024 Ostris, LLC):
see THIRD_PARTY_NOTICES.md.

Builds on ZenTrainer's Triton-batched Automagic3Batched (Ostris's H=4 sign-polarity LR controller,
0.5 bytes/param state, no momentum lag breaking flip detection) and adds four adapter-geometry
upgrades:
1. Shape-batched exact r x r Gram polar orthogonalization on 2D updates with 2 <= min(r, c) <= 64
   (LoRA factors and small LoKr cores, sharing Lotus's eligibility rule): equalizes rank energy so
   updates don't collapse onto the dominant class direction. The polarity vote stays on the
   pre-orthogonalized update so orthogonal rotation never distorts sign-consistency detection; the
   fused Triton tail kernel applies the orthogonalized step in one pass.
2. Built-in LoRA+ asymmetry (default 2x step on lora_up / B) inside the same param group, so A and B
   share one unified polarity LR vote instead of fighting across separate groups.
3. Exact fp32 second moments for tiny high-leverage matrices (<= 1024 elems, e.g. lokr_w1) and
   in-place LoKr gauge balancing (w1 *= c, w2 /= c: preserves w1 kron w2 up to low-precision
   rounding with no persistent optimizer state and zero CPU sync).
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from .automagic3_batched import CHUNK, Automagic3Batched
from .lotus import EXACT_MAX_NUMEL, _balance_lokr_gauge, _gram_orth_batched, _orth_eligible


class Automagic3Geo(Automagic3Batched):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        min_lr: float = 1e-8,
        max_lr: float = 1e3,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 2.5,
        weight_decay: float = 0.0,
        polarity_history: int = 4,
        polar_orth: bool = True,
        lora_b_mult: float = 2.0,
        lokr_gauge_alpha: float = 0.25,
        vote_clip: float = 0.25,
        adapter_pairs: dict | None = None,
        fused: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            polarity_history=polarity_history,
            fused=False,
        )
        extra_defaults = dict(
            polar_orth=polar_orth,
            lora_b_mult=lora_b_mult,
            lokr_gauge_alpha=lokr_gauge_alpha,
            vote_clip=vote_clip,
        )
        self.defaults.update(extra_defaults)
        for group in self.param_groups:
            for k, v in extra_defaults.items():
                group.setdefault(k, v)
        self.set_adapter_pairs(adapter_pairs)

    def load_state_dict(self, state_dict):
        # Snapshot saved fp32 exact second moments before super() casts floating state to param.dtype
        # (an fp16 8x8 lokr_w1 second moment ~1e-11 would otherwise underflow to exact 0.0).
        saved_exact = {
            i: st["exp_avg_sq"].clone()
            for i, st in state_dict["state"].items()
            if isinstance(st.get("exp_avg_sq"), torch.Tensor)
        }
        super().load_state_dict(state_dict)
        ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for i, p in zip(ids, params):
            orig = saved_exact.get(i)
            if orig is not None and p.numel() <= EXACT_MAX_NUMEL:
                st = self.state.get(p)
                if st is not None:
                    st["exp_avg_sq"] = orig.to(device=p.device, dtype=torch.float32)

    def set_adapter_pairs(self, adapter_pairs: dict | None) -> None:
        self._adapter_pairs = adapter_pairs or {"lora_pairs": [], "lokr_pairs": []}
        self._lora_b_ids = {id(b) for _, b in self._adapter_pairs.get("lora_pairs", ())}
        self._lokr_pairs = list(self._adapter_pairs.get("lokr_pairs", ()))

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        super()._init_state(p, group)
        if p.dim() >= 2 and p.numel() <= EXACT_MAX_NUMEL:
            state = self.state[p]
            state.pop("exp_avg_sq_row", None)
            state.pop("exp_avg_sq_col", None)
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        stacks: dict[tuple, list] = defaultdict(list)
        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                accum = getattr(p, "_accum_grad", None)
                if accum is not None:
                    p.grad = accum
                    del p._accum_grad
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Automagic3Geo does not support sparse gradients.")
                if p.dim() < 2:
                    self._update_param(p, group)
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                exact = "exp_avg_sq" in state
                v_dtype = state["exp_avg_sq"].dtype if exact else state["exp_avg_sq_row"].dtype
                key = (
                    gi,
                    exact,
                    tuple(p.shape),
                    p.dtype,
                    p.grad.dtype,
                    v_dtype,
                    state["step"],
                    state["hist_idx"],
                    state["hist_fill"],
                    state["sign_history"].shape[0],
                    p.device,
                )
                stacks[key].append(p)
        for key, ps in stacks.items():
            group = self.param_groups[key[0]]
            exact = key[1]
            per = max(1, CHUNK // ps[0].numel())
            for i in range(0, len(ps), per):
                self._update_stack(ps[i : i + per], group, exact=exact)
        self._apply_group_votes()
        self._balance_lokr_gauge()
        return loss

    def _update_stack(self, ps: list, group: dict, *, exact: bool = False) -> None:
        n = len(ps)
        states = [self.state[p] for p in ps]
        st0 = states[0]
        beta2, eps, clip = group["beta2"], group["eps"], group["clip_threshold"]
        step = st0["step"] + 1
        bc2_sqrt = math.sqrt(1.0 - beta2 ** step)

        grad = torch.stack([p.grad for p in ps]).to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        sq = grad * grad

        if exact:
            vs = [s["exp_avg_sq"] for s in states]
            v = torch.stack(vs).to(torch.float32)
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            torch._foreach_copy_(vs, list(v.unbind(0)))
            update = v.add(eps).rsqrt().mul_(grad).mul_(bc2_sqrt)
            del sq, v
        else:
            rows = [s["exp_avg_sq_row"] for s in states]
            cols = [s["exp_avg_sq_col"] for s in states]
            row = torch.stack(rows).to(torch.float32)
            col = torch.stack(cols).to(torch.float32)
            row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            torch._foreach_copy_(rows, list(row.unbind(0)))
            torch._foreach_copy_(cols, list(col.unbind(0)))
            update = self._approx_sq_grad(row, col).mul_(grad).mul_(bc2_sqrt)
            del sq, row, col

        numel = ps[0].numel()
        dims = tuple(range(1, update.dim()))
        rms = torch.linalg.vector_norm(update, dim=dims, keepdim=True) / math.sqrt(numel)
        update.div_((rms / clip).clamp_(min=1.0))
        update.clamp_(-clip, clip)

        step_update: torch.Tensor | None = None
        if group.get("polar_orth", True) and update.dim() == 3 and _orth_eligible(ps[0]):
            step_update = _gram_orth_batched(update)

        hists = [s["sign_history"] for s in states]
        H, idx = hists[0].shape[0], st0["hist_idx"]
        new_idx = (idx + 1) % H
        fill = min(H, st0["hist_fill"] + 1)
        for s in states:
            s["hist_idx"], s["hist_fill"] = new_idx, fill
            s["step"] = step

        wd = group["weight_decay"]
        lr = torch.stack([s["lr"] for s in states])
        b_mult = float(group.get("lora_b_mult", 1.0))
        if b_mult != 1.0 and self._lora_b_ids and group.get("lr_mult", 1.0) == 1.0:
            mults = [b_mult if id(p) in self._lora_b_ids else 1.0 for p in ps]
            if any(m != 1.0 for m in mults):
                lr = lr * torch.tensor(mults, dtype=torch.float32, device=lr.device)

        gi = self._param_group_index.get(ps[0])
        do_vote = fill == H and gi is not None

        from . import _automagic_triton

        if _automagic_triton.supported(update, ps[0].dtype):
            num, den = _automagic_triton.tail_step(
                update,
                ps,
                hists,
                lr,
                wd,
                idx,
                new_idx,
                do_vote,
                is_v4=False,
                step_update=step_update,
            )
            if do_vote:
                if self._group_num[gi] is None:
                    self._group_num[gi], self._group_den[gi] = num, den
                else:
                    self._group_num[gi].add_(num.to(self._group_num[gi].device))
                    self._group_den[gi].add_(den.to(self._group_den[gi].device))
        else:
            weights, shifts = self._pack_consts(update.device)
            flat = update.reshape(n, -1).gt(0.0).to(torch.uint8)
            pad = (-numel) % 8
            if pad:
                flat = torch.cat([flat, flat.new_zeros(n, pad)], dim=1)
            packed = (flat.view(n, -1, 8) * weights).sum(-1, dtype=torch.uint8)
            torch._foreach_copy_([h[idx] for h in hists], list(packed.unbind(0)))

            if do_vote:
                chron = torch.roll(torch.stack(hists), -new_idx, dims=1)
                all_one = chron[:, 0].clone()
                all_zero = chron[:, 0].bitwise_xor(255)
                all_flip = torch.full_like(all_one, 255)
                prev = chron[:, 0]
                for t in range(1, H):
                    cur = chron[:, t]
                    all_one.bitwise_and_(cur)
                    all_zero.bitwise_and_(cur.bitwise_xor(255))
                    all_flip.bitwise_and_(cur.bitwise_xor(prev))
                    prev = cur
                up_byte = all_one.bitwise_or_(all_zero)
                up = ((up_byte.unsqueeze(-1) >> shifts).bitwise_and_(1)).view(n, -1)[:, :numel]
                down = ((all_flip.unsqueeze(-1) >> shifts).bitwise_and_(1)).view(n, -1)[:, :numel]
                w = update.abs().view(n, -1)
                num = (w * up).sum().sub_((w * down).sum())
                den = w.sum()
                if self._group_num[gi] is None:
                    self._group_num[gi], self._group_den[gi] = num, den
                else:
                    self._group_num[gi].add_(num.to(self._group_num[gi].device))
                    self._group_den[gi].add_(den.to(self._group_den[gi].device))
                del chron, all_one, all_zero, all_flip, up_byte, up, down, w

            applied = step_update if step_update is not None else update
            lr_b = lr.view((n,) + (1,) * (applied.dim() - 1))
            new_p = torch.stack(ps).to(torch.float32)
            if wd != 0.0:
                applied = applied + new_p * wd
            new_p.addcmul_(applied, lr_b, value=-1.0)
            if ps[0].dtype == torch.float32:
                torch._foreach_copy_(ps, list(new_p.unbind(0)))
            else:
                low = torch.empty((n,) + tuple(ps[0].shape), dtype=ps[0].dtype, device=new_p.device)
                self._stochastic_copy_(low, new_p)
                torch._foreach_copy_(ps, list(low.unbind(0)))
        for p in ps:
            p.grad = None

    def _apply_group_votes(self) -> None:
        for gi, group in enumerate(self.param_groups):
            num = self._group_num[gi]
            if num is None:
                continue
            den = self._group_den[gi]
            vclip = float(group.get("vote_clip", 0.25))
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-vclip, vclip)
            factor = torch.exp(signal)
            by_dev: dict[torch.device, list[torch.Tensor]] = defaultdict(list)
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and "lr" in st:
                    by_dev[st["lr"].device].append(st["lr"])
            lo, hi = group["min_lr"], group["max_lr"]
            for dev, lrs in by_dev.items():
                f = factor if factor.device == dev else factor.to(dev)
                torch._foreach_mul_(lrs, f)
                torch._foreach_clamp_min_(lrs, lo)
                torch._foreach_clamp_max_(lrs, hi)
            self._group_num[gi] = None
            self._group_den[gi] = None

    @torch.no_grad()
    def _balance_lokr_gauge(self) -> None:
        _balance_lokr_gauge(
            self._lokr_pairs,
            lambda w1: self.param_groups[gi].get("lokr_gauge_alpha", 0.0)
            if (gi := self._param_group_index.get(w1)) is not None
            else 0.0,
            sr_copy=self._stochastic_copy_,
        )
