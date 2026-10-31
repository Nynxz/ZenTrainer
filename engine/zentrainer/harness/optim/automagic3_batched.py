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

        # the sign window: every tensor here shares hist_idx and hist_fill (part of the key)
        hists = [s["sign_history"] for s in states]
        H, idx = hists[0].shape[0], st0["hist_idx"]
        weights, shifts = self._pack_consts(update.device)
        flat = update.reshape(n, -1).gt(0.0).to(torch.uint8)
        pad = (-numel) % 8
        if pad:
            flat = torch.cat([flat, flat.new_zeros(n, pad)], dim=1)
        packed = (flat.view(n, -1, 8) * weights).sum(-1, dtype=torch.uint8)
        torch._foreach_copy_([h[idx] for h in hists], list(packed.unbind(0)))
        new_idx = (idx + 1) % H
        fill = min(H, st0["hist_fill"] + 1)
        for s in states:
            s["hist_idx"], s["hist_fill"] = new_idx, fill

        if fill == H:
            chron = torch.roll(torch.stack(hists), -new_idx, dims=1)
            bits = (chron.unsqueeze(-1) >> shifts).bitwise_and_(1).view(n, H, -1)[:, :, :numel]
            s1 = bits.sum(1, dtype=torch.int16)
            flips = (bits[:, 1:] ^ bits[:, :-1]).sum(1, dtype=torch.int16)
            up = s1.eq(H).logical_or_(s1.eq(0))
            down = flips.eq(H - 1)
            w = update.abs().view(n, -1)
            num = (w * up).sum().sub_((w * down).sum())
            den = w.sum()
            gi = self._param_group_index.get(ps[0])
            if gi is not None:
                if self._group_num[gi] is None:
                    self._group_num[gi], self._group_den[gi] = num, den
                else:
                    self._group_num[gi].add_(num.to(self._group_num[gi].device))
                    self._group_den[gi].add_(den.to(self._group_den[gi].device))
            del bits, s1, flips, up, down, w

        for s in states:
            s["step"] += 1

        wd = group["weight_decay"]
        lr = torch.stack([s["lr"] for s in states]).view((n,) + (1,) * (update.dim() - 1))
        new_p = torch.stack(ps).to(torch.float32)
        if wd != 0.0:
            update.add_(new_p, alpha=wd)
        new_p.addcmul_(update, lr, value=-1.0)
        if ps[0].dtype == torch.float32:
            torch._foreach_copy_(ps, list(new_p.unbind(0)))
        else:
            low = torch.empty((n,) + tuple(ps[0].shape), dtype=ps[0].dtype, device=new_p.device)
            self._stochastic_copy_(low, new_p)
            torch._foreach_copy_(ps, list(low.unbind(0)))
        for p in ps:
            p.grad = None
