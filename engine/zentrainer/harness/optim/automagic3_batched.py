"""Automagic v3 with the non-fused step batched across tensors of the same shape.

The vendored step (automagic3.py) updates one tensor at a time: ~50 small kernels each, so a
LoRA's ~800 tensors cost ~40k launches a step, measured at 146 ms on Krea 2 (10% of the step,
loaded or offloaded). A LoRA has only a handful of shapes, so tensors are stacked per shape and
the same math runs once per stack: a few hundred launches. The state (per tensor, same keys and
layout) is the vendored one, so states save and resume either way; tests check the two steps
agree.

Derived from Ostris's Automagic v3 (ai-toolkit, MIT, Copyright (c) 2024 Ostris, LLC):
see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from .automagic3 import Automagic3

# elements per stack: bounds the step's temporary memory (a few fp32 copies of a stack)
CHUNK = 1 << 24


class Automagic3Batched(Automagic3):
    @torch.no_grad()
    def step(self, closure=None):
        if self.fused:
            return super().step(closure)
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
                if p.grad.is_sparse or p.dim() < 2:
                    self._update_param(p, group)  # biases, norms: few, and 1-D state
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                key = (gi, tuple(p.shape), p.dtype, p.grad.dtype, state["exp_avg_sq_row"].dtype,
                       state["hist_idx"], state["hist_fill"], state["sign_history"].shape[0], p.device)
                stacks[key].append(p)
        for key, ps in stacks.items():
            group = self.param_groups[key[0]]
            per = max(1, CHUNK // ps[0].numel())
            for i in range(0, len(ps), per):
                self._update_stack(ps[i:i + per], group)
        self._apply_group_votes()
        return loss

    def _update_stack(self, ps: list, group: dict) -> None:
        """The vendored _update_param, on N same-shaped tensors stacked along a new first dim."""
        n = len(ps)
        states = [self.state[p] for p in ps]
        st0 = states[0]
        beta2, eps, clip = group["beta2"], group["eps"], group["clip_threshold"]

        grad = torch.stack([p.grad for p in ps]).to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        sq = grad * grad

        rows = [s["exp_avg_sq_row"] for s in states]
        cols = [s["exp_avg_sq_col"] for s in states]
        row = torch.stack(rows).to(torch.float32)
        col = torch.stack(cols).to(torch.float32)
        row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
        col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))
        update = self._approx_sq_grad(row, col).mul_(grad)
        del sq, row, col

        numel = ps[0].numel()
        dims = tuple(range(1, update.dim()))
        rms = torch.linalg.vector_norm(update, dim=dims, keepdim=True) / math.sqrt(numel)
        update.div_((rms / clip).clamp_(min=1.0))
        update.clamp_(-clip, clip)

        # the sign window, vote, WD, param update and stochastic rounding
        hists = [s["sign_history"] for s in states]
        H, idx = hists[0].shape[0], st0["hist_idx"]
        new_idx = (idx + 1) % H
        fill = min(H, st0["hist_fill"] + 1)
        for s in states:
            s["hist_idx"], s["hist_fill"] = new_idx, fill
            s["step"] += 1

        wd = group["weight_decay"]
        lr = torch.stack([s["lr"] for s in states])
        gi = self._param_group_index.get(ps[0])
        do_vote = fill == H and gi is not None

        from . import _automagic_triton

        if _automagic_triton.supported(update, ps[0].dtype):
            num, den = _automagic_triton.tail_step(
                update, ps, hists, lr, wd, idx, new_idx, do_vote, is_v4=False,
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
                # Bitwise agreement / alternation across all H planes without unpacking (n, H, numel)
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

            lr_b = lr.view((n,) + (1,) * (update.dim() - 1))
            new_p = torch.stack(ps).to(torch.float32)
            if wd != 0.0:
                update.add_(new_p, alpha=wd)
            new_p.addcmul_(update, lr_b, value=-1.0)
            if ps[0].dtype == torch.float32:
                torch._foreach_copy_(ps, list(new_p.unbind(0)))
            else:
                low = torch.empty((n,) + tuple(ps[0].shape), dtype=ps[0].dtype, device=new_p.device)
                self._stochastic_copy_(low, new_p)
                torch._foreach_copy_(ps, list(low.unbind(0)))
        for p in ps:
            p.grad = None

    def _apply_group_votes(self) -> None:
        """Batched group-LR update: avoids launching 2 tiny kernels per parameter (1,600+/step)."""
        for gi, group in enumerate(self.param_groups):
            num = self._group_num[gi]
            if num is None:
                continue
            den = self._group_den[gi]
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-1.0, 1.0)
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
