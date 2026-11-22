"""Lotus: a LoRA-native, memory-efficient optimizer for diffusion training.

Combines four ideas tailored specifically to low-rank diffusion adapters:
1. Scale-normalized Riemannian LoRA preconditioning on paired (lora_down, lora_up) weights:
   whitening each factor's update by the tiny r x r Gram matrix of its partner
   (A A^T / s_A + damp * I)^-1 and (B^T B / s_B + damp * I)^-1 prevents rank collapse and
   eliminates A/B scale fighting while keeping the user's LR on AdamW's intuitive 1e-4 scale.
2. Smoothed momentum stored in bf16 (with stochastic rounding) + Adafactor-factored second
   moments: denoises batch-size-1 random-timestep diffusion gradients like AdamW at ~2-3x less
   optimizer VRAM, and trains direct bf16/fp16 LoRAs cleanly via Philox stochastic rounding.
3. Pre-momentum update-RMS trust region + norm-preserving cautious gating: stops outlier
   timestep spikes from poisoning momentum and damps momentum overshoot without amplifying noise.
4. Bounded self-regulating LR multiplier in [lr_min_mult, lr_max_mult] driven by the group's
   smoothed cosine alignment between previous momentum and current normalized gradient, with an
   asymmetric deadzone: honors warmup/cosine schedules, climbs on steady trends, backs off on
   overshoot, and holds steady on pure noise (zero Automagic3 decay stall, zero Automagic4 runaway).

Batched across same-shaped tensors and accelerated with fused Triton kernels on CUDA and ROCm.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

CHUNK = 1 << 24  # max elements per shape stack (bounds peak transient memory)


def _sr_truncate_(dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
    """Stochastically round fp32 `src_fp32` into bf16/fp16 `dst` in place (mutates `src_fp32`)."""
    if dst.dtype == torch.bfloat16:
        drop = 16
    elif dst.dtype == torch.float16:
        drop = 13
    else:
        dst.copy_(src_fp32)
        return
    as_i = src_fp32.view(torch.int32)
    as_i.add_(torch.randint_like(as_i, 1 << drop))
    as_i.bitwise_and_(-(1 << drop))
    dst.copy_(src_fp32)


def _discover_pairs(param_groups: list[dict], explicit_pairs=None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Find (lora_down_A, lora_up_B) pairs from explicit pairs or Network._own weakref tags."""
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if explicit_pairs:
        for a, b in explicit_pairs:
            key = (id(a), id(b))
            if key not in seen and a.ndim == 2 and b.ndim == 2 and a.shape[0] == b.shape[1]:
                seen.add(key)
                pairs.append((a, b))
    all_params = {id(p): p for g in param_groups for p in g["params"]}
    for p in all_params.values():
        if getattr(p, "_zt_lora_side", None) != "down":
            continue
        ref = getattr(p, "_zt_lora_partner", None)
        partner = ref() if callable(ref) else ref
        if partner is None or id(partner) not in all_params:
            continue
        key = (id(p), id(partner))
        if key not in seen and p.ndim == 2 and partner.ndim == 2 and p.shape[0] == partner.shape[1]:
            seen.add(key)
            pairs.append((p, partner))
    return pairs


def _gram_preconditioner(G: torch.Tensor, damp: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Frobenius-normalized (G / ||G||_F + damp * I)^-1 with unit spectral RMS.

    Normalizing by ||G||_F keeps the largest eigenvalue of G / ||G||_F <= 1 regardless of rank r,
    so the condition number is bounded by (1 + damp) / damp (~5x at damp=0.25) even at rank 64-128
    when B is still near rank-1 early in training. Returns (P, valid_mask) where valid_mask is
    (N, 1, 1) True when ||G||_F > 1e-10 (False at zero-initialized lora_up on step 1 -> identity).
    """
    n, r, _ = G.shape
    eye = torch.eye(r, device=G.device, dtype=torch.float32).unsqueeze(0)
    gnorm = G.flatten(1).norm(dim=1)  # (N,)
    valid = gnorm > 1e-10
    scale = gnorm.clamp_min(1e-10).view(n, 1, 1)
    Gn = torch.where(valid.view(n, 1, 1), G / scale + damp * eye, eye)
    L, info = torch.linalg.cholesky_ex(Gn)
    ok = valid & (info == 0)
    # Fallback cleanly to identity on any numerical degeneracy
    L_safe = torch.where(ok.view(n, 1, 1), L, eye)
    P = torch.cholesky_inverse(L_safe)
    rms = (P.flatten(1).norm(dim=1) / math.sqrt(r)).clamp_min(1e-6).view(n, 1, 1)
    P = P / rms
    return P, ok.view(n, 1, 1)


class Lotus(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        cautious_min: float = 0.25,
        precond_damp: float = 0.25,
        auto_lr: bool = True,
        lr_min_mult: float = 0.5,
        lr_max_mult: float = 2.5,
        lr_beta: float = 0.85,
        lr_rate: float = 0.20,
        lora_pairs=None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            cautious_min=cautious_min,
            precond_damp=precond_damp,
            auto_lr=auto_lr,
            lr_min_mult=lr_min_mult,
            lr_max_mult=lr_max_mult,
            lr_beta=lr_beta,
            lr_rate=lr_rate,
        )
        super().__init__(params, defaults)
        self._explicit_pairs = lora_pairs
        self._pairs = _discover_pairs(self.param_groups, lora_pairs)
        self._group_mult: list[torch.Tensor | None] = [None] * len(self.param_groups)
        self._group_cos: list[torch.Tensor | None] = [None] * len(self.param_groups)

    def _init_state(self, p: torch.Tensor) -> dict:
        st = self.state[p]
        st["step"] = 0
        # Store momentum in bf16 on CUDA/CPU (with stochastic rounding) to halve optimizer VRAM;
        # fp16 weights still keep bf16 momentum (same exponent range as fp32, no underflow).
        m_dtype = torch.bfloat16 if p.dim() >= 2 else torch.float32
        st["exp_avg"] = torch.zeros_like(p, dtype=m_dtype)
        if p.dim() >= 2:
            st["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=torch.float32, device=p.device)
            st["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device)
        else:
            st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        return st

    def _ensure_group_ctrl(self, gi: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._group_mult[gi] is None:
            # Recover from loaded state if resumed from a checkpoint
            g = self.param_groups[gi]
            m0 = float(g.get("_lotus_lr_mult", 1.0))
            c0 = float(g.get("_lotus_ema_cos", 0.0))
            self._group_mult[gi] = torch.tensor(m0, dtype=torch.float32, device=device)
            self._group_cos[gi] = torch.tensor(c0, dtype=torch.float32, device=device)
        return self._group_mult[gi], self._group_cos[gi]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        from . import _lotus_triton

        # 1. Bucket active parameters by (group, shape, p.dtype, g.dtype, step, device)
        stacks: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        ones: list[tuple[int, torch.Tensor]] = []
        align_bufs: dict[int, torch.Tensor] = {}
        group_of: dict[torch.Tensor, int] = {}

        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lotus does not support sparse gradients.")
                st = self.state[p]
                if not st:
                    st = self._init_state(p)
                group_of[p] = gi
                if gi not in align_bufs:
                    self._ensure_group_ctrl(gi, p.device)
                    align_bufs[gi] = torch.zeros((3,), dtype=torch.float32, device=p.device)
                if p.dim() < 2:
                    ones.append((gi, p))
                else:
                    key = (gi, tuple(p.shape), p.dtype, p.grad.dtype, st["exp_avg"].dtype, st["step"], p.device)
                    stacks[key].append(p)

        # 2. Compute momentum updates for 2D+ stacks and 1D params (pre-Riemannian)
        # Keep stacked views alive so _precondition_pairs and apply_step reuse them without extra stacks.
        updates: dict[torch.Tensor, torch.Tensor] = {}
        chunk_records: list[tuple[int, list[torch.Tensor], torch.Tensor]] = []
        for key, ps in stacks.items():
            gi = key[0]
            group = self.param_groups[gi]
            per = max(1, CHUNK // ps[0].numel())
            for i in range(0, len(ps), per):
                chunk = ps[i:i + per]
                u_stack = self._momentum_stack(chunk, group, align_bufs[gi], _lotus_triton)
                chunk_records.append((gi, chunk, u_stack))
                for p, u in zip(chunk, u_stack.unbind(0)):
                    updates[p] = u

        for gi, p in ones:
            updates[p] = self._momentum_1d(p, self.param_groups[gi], align_bufs[gi])

        # 3. Update bounded group LR multiplier on-device (zero GPU->CPU sync)
        for gi, abuf in align_bufs.items():
            group = self.param_groups[gi]
            if not group["auto_lr"]:
                continue
            mult, ema_cos = self._group_mult[gi], self._group_cos[gi]
            dot, m2, g2 = abuf[0], abuf[1], abuf[2]
            valid = (m2 > 1e-24) & (g2 > 1e-24)
            cos = dot / ((m2 * g2).clamp_min(1e-24).sqrt() + 1e-12)
            beta_c = group["lr_beta"]
            new_cos = torch.where(valid, ema_cos * beta_c + cos * (1.0 - beta_c), ema_cos)
            ema_cos.copy_(new_cos)
            # Asymmetric deadzone: backs off cleanly on overshoot (< -0.02), climbs on real trend (> +0.06),
            # holds rock steady on zero-mean diffusion noise.
            adj = torch.where(new_cos > 0.06, new_cos - 0.06,
                              torch.where(new_cos < -0.02, (new_cos + 0.02) * 2.0, torch.zeros_like(new_cos)))
            new_mult = (mult * torch.exp(group["lr_rate"] * adj)).clamp(group["lr_min_mult"], group["lr_max_mult"])
            mult.copy_(torch.where(valid, new_mult, mult))

        # 4. Batched Riemannian LoRA preconditioning across (A=down, B=up) pairs (updates u_stack in place)
        p_stacks: dict[int, torch.Tensor] = {}
        self._precondition_pairs(updates, group_of, chunk_records, p_stacks)

        # 5. Apply decoupled weight decay, effective LR, and stochastic rounding
        for ci, (gi, chunk, u_stack) in enumerate(chunk_records):
            group = self.param_groups[gi]
            lr_base, wd = group["lr"], group["weight_decay"]
            dev, p_dtype = chunk[0].device, chunk[0].dtype
            mult = self._group_mult[gi]
            if mult.device != dev:
                mult = mult.to(dev)
            p_stack = p_stacks.pop(ci, None)
            if _lotus_triton.supported(u_stack, p_dtype):
                _lotus_triton.apply_step(u_stack, chunk, mult, lr_base, wd, p_stack=p_stack)
            else:
                lr_eff = mult * lr_base
                new_p = (p_stack if p_stack is not None else torch.stack(chunk)).to(torch.float32)
                if wd != 0.0:
                    u_stack.add_(new_p, alpha=wd)
                new_p.addcmul_(u_stack, lr_eff, value=-1.0)
                if p_dtype == torch.float32:
                    torch._foreach_copy_(chunk, list(new_p.unbind(0)))
                else:
                    low = torch.empty((len(chunk),) + tuple(chunk[0].shape), dtype=p_dtype, device=dev)
                    _sr_truncate_(low, new_p)
                    torch._foreach_copy_(chunk, list(low.unbind(0)))

        for gi, p in ones:
            group = self.param_groups[gi]
            lr_base, wd = group["lr"], group["weight_decay"]
            mult = self._group_mult[gi]
            if mult.device != p.device:
                mult = mult.to(p.device)
            u = updates[p]
            lr_eff = mult * lr_base
            if p.dtype == torch.float32:
                if wd != 0.0:
                    u.add_(p, alpha=wd)
                p.addcmul_(u, lr_eff, value=-1.0)
            else:
                pf = p.to(torch.float32)
                if wd != 0.0:
                    u.add_(pf, alpha=wd)
                pf.addcmul_(u, lr_eff, value=-1.0)
                _sr_truncate_(p, pf)

        for p in updates:
            p.grad = None
        return loss

    def _momentum_stack(self, ps: list[torch.Tensor], group: dict, align_buf: torch.Tensor, triton_mod) -> torch.Tensor:
        n = len(ps)
        states = [self.state[p] for p in ps]
        b1, b2 = group["betas"]
        eps, clip, caut = group["eps"], group["clip_threshold"], group["cautious_min"]

        grad = torch.stack([p.grad for p in ps]).to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        sq = grad * grad

        rows = [s["exp_avg_sq_row"] for s in states]
        cols = [s["exp_avg_sq_col"] for s in states]
        row = torch.stack(rows)
        col = torch.stack(cols)
        row.mul_(b2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - b2)
        col.mul_(b2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - b2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))
        del sq

        for s in states:
            s["step"] += 1
        step = states[0]["step"]
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step

        r = (row / row.mean(dim=-1, keepdim=True).clamp_min(eps)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        gh = grad.mul_(r * c).mul_(math.sqrt(bc2))
        del row, col, r, c

        # Pre-momentum spike immunity: clip RMS and element extremes before updating momentum
        numel = ps[0].numel()
        dims = tuple(range(1, gh.dim()))
        rms = torch.linalg.vector_norm(gh, dim=dims, keepdim=True) / math.sqrt(numel)
        gh.div_((rms / clip).clamp_(min=1.0))
        gh.clamp_(-3.0 * clip, 3.0 * clip)

        ms = [s["exp_avg"] for s in states]
        do_align = step > 1
        abuf = align_buf if align_buf.device == gh.device else torch.zeros((3,), dtype=torch.float32, device=gh.device)
        if triton_mod.supported(gh, ps[0].dtype):
            u = triton_mod.momentum_step(gh, ms, abuf, b1, 1.0 / bc1, caut, do_align)
        else:
            mp = torch.stack(ms).to(torch.float32)
            if do_align:
                abuf[0].add_((mp * gh).sum())
                abuf[1].add_((mp * mp).sum())
                abuf[2].add_((gh * gh).sum())
            mn = mp.mul_(b1).add_(gh, alpha=1.0 - b1)
            if ms[0].dtype == torch.float32:
                torch._foreach_copy_(ms, list(mn.unbind(0)))
            else:
                low = torch.empty((n,) + tuple(ps[0].shape), dtype=ms[0].dtype, device=mn.device)
                _sr_truncate_(low, mn.clone())
                torch._foreach_copy_(ms, list(low.unbind(0)))
            u = mn.mul_(1.0 / bc1)
            if caut < 1.0:
                cm = torch.where(u * gh > 0.0, 1.0, caut)
                u.mul_(cm).mul_((cm * cm).mean(dim=dims, keepdim=True).clamp_min(0.1).rsqrt())
        if abuf is not align_buf:
            align_buf.add_(abuf.to(align_buf.device))
        return u

    def _momentum_1d(self, p: torch.Tensor, group: dict, align_buf: torch.Tensor) -> torch.Tensor:
        st = self.state[p]
        b1, b2 = group["betas"]
        eps, clip, caut = group["eps"], group["clip_threshold"], group["cautious_min"]
        st["step"] += 1
        step = st["step"]
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        grad = p.grad.to(torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        st["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        gh = grad.mul_((st["exp_avg_sq"] / bc2 + eps).rsqrt())
        rms = gh.norm() / math.sqrt(p.numel())
        gh.div_((rms / clip).clamp_(min=1.0)).clamp_(-3.0 * clip, 3.0 * clip)
        mp = st["exp_avg"].to(torch.float32)
        if step > 1:
            dev_a = align_buf.device
            align_buf[0].add_((mp * gh).sum().to(dev_a))
            align_buf[1].add_((mp * mp).sum().to(dev_a))
            align_buf[2].add_((gh * gh).sum().to(dev_a))
        mn = mp.mul_(b1).add_(gh, alpha=1.0 - b1)
        st["exp_avg"].copy_(mn)
        u = mn / bc1
        if caut < 1.0:
            cm = torch.where(u * gh > 0.0, 1.0, caut)
            u = u * cm * (cm * cm).mean().clamp_min(0.1).rsqrt()
        return u

    def _precondition_pairs(
        self,
        updates: dict[torch.Tensor, torch.Tensor],
        group_of: dict[torch.Tensor, int],
        chunk_records: list[tuple[int, list[torch.Tensor], torch.Tensor]],
        p_stacks: dict[int, torch.Tensor],
    ) -> None:
        if not self._pairs:
            return
        # Map each param to its (chunk_idx, slot) so we can stack params once and update u_stack in place
        loc: dict[torch.Tensor, tuple[int, int]] = {
            p: (ci, j) for ci, (_, chunk, _) in enumerate(chunk_records) for j, p in enumerate(chunk)
        }
        buckets: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for p_a, p_b in self._pairs:
            if p_a not in updates and p_b not in updates:
                continue
            ga = self.param_groups[group_of.get(p_a, group_of[p_b])]
            damp = ga["precond_damp"]
            if damp <= 0.0:
                continue
            clip = ga["clip_threshold"]
            buckets[(p_a.device, p_a.shape[0], p_a.shape[1], p_b.shape[0], damp, clip)].append((p_a, p_b))

        def get_p_stack(ci: int) -> torch.Tensor:
            st = p_stacks.get(ci)
            if st is None:
                st = p_stacks[ci] = torch.stack(chunk_records[ci][1])
            return st

        for (dev, r, _in, _out, damp, clip), pairs in buckets.items():
            # Fast path: when all pairs' As live in one chunk in order (and Bs in one chunk in order),
            # reuse p_stack and u_stack directly with zero extra stack/copy!
            bs = [(a, b) for a, b in pairs if b in updates]
            if bs:
                a_locs = [loc.get(a) for a, _ in bs]
                b_locs = [loc.get(b) for _, b in bs]
                if (all(x is not None for x in a_locs) and len({x[0] for x in a_locs}) == 1
                        and len(a_locs) == len(chunk_records[a_locs[0][0]][1])
                        and [x[1] for x in a_locs] == list(range(len(a_locs)))):
                    A = get_p_stack(a_locs[0][0]).to(torch.float32)
                else:
                    A = torch.stack([a.detach().to(torch.float32) for a, _ in bs])
                P_A, ok = _gram_preconditioner(torch.bmm(A, A.transpose(1, 2)), damp)
                b_fast = (all(x is not None for x in b_locs) and len({x[0] for x in b_locs}) == 1
                          and len(b_locs) == len(chunk_records[b_locs[0][0]][1])
                          and [x[1] for x in b_locs] == list(range(len(b_locs))))
                ub = chunk_records[b_locs[0][0]][2] if b_fast else torch.stack([updates[b] for _, b in bs])
                ub_p = torch.where(ok, torch.bmm(ub, P_A), ub)
                rms = torch.linalg.vector_norm(ub_p, dim=(1, 2), keepdim=True) / math.sqrt(ub_p[0].numel())
                ub_p.div_((rms / clip).clamp_(min=1.0))
                if b_fast:
                    ub.copy_(ub_p)
                else:
                    for (_, b), u in zip(bs, ub_p.unbind(0)):
                        updates[b].copy_(u)

            as_ = [(a, b) for a, b in pairs if a in updates]
            if as_:
                a_locs = [loc.get(a) for a, _ in as_]
                b_locs = [loc.get(b) for _, b in as_]
                if (all(x is not None for x in b_locs) and len({x[0] for x in b_locs}) == 1
                        and len(b_locs) == len(chunk_records[b_locs[0][0]][1])
                        and [x[1] for x in b_locs] == list(range(len(b_locs)))):
                    B = get_p_stack(b_locs[0][0]).to(torch.float32)
                else:
                    B = torch.stack([b.detach().to(torch.float32) for _, b in as_])
                P_B, ok = _gram_preconditioner(torch.bmm(B.transpose(1, 2), B), damp)
                a_fast = (all(x is not None for x in a_locs) and len({x[0] for x in a_locs}) == 1
                          and len(a_locs) == len(chunk_records[a_locs[0][0]][1])
                          and [x[1] for x in a_locs] == list(range(len(a_locs))))
                ua = chunk_records[a_locs[0][0]][2] if a_fast else torch.stack([updates[a] for a, _ in as_])
                ua_p = torch.where(ok, torch.bmm(P_B, ua), ua)
                rms = torch.linalg.vector_norm(ua_p, dim=(1, 2), keepdim=True) / math.sqrt(ua_p[0].numel())
                ua_p.div_((rms / clip).clamp_(min=1.0))
                if a_fast:
                    ua.copy_(ua_p)
                else:
                    for (a, _), u in zip(as_, ua_p.unbind(0)):
                        updates[a].copy_(u)

    def get_avg_learning_rate(self) -> float:
        """Effective learning rate (scheduled group LR * bounded self-regulating multiplier)."""
        vals = []
        for gi, g in enumerate(self.param_groups):
            m = self._group_mult[gi]
            mult = float(m) if m is not None else float(g.get("_lotus_lr_mult", 1.0))
            vals.append(float(g["lr"]) * mult)
        return sum(vals) / len(vals) if vals else 0.0

    def state_dict(self) -> dict:
        # Persist each group's adaptive multiplier and alignment EMA inside param_groups
        for gi, g in enumerate(self.param_groups):
            if self._group_mult[gi] is not None:
                g["_lotus_lr_mult"] = float(self._group_mult[gi].detach().cpu())
            if self._group_cos[gi] is not None:
                g["_lotus_ema_cos"] = float(self._group_cos[gi].detach().cpu())
        return super().state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # Restore state dtypes (parent casts floating state to param.dtype)
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                if "exp_avg" in st:
                    st["exp_avg"] = st["exp_avg"].to(torch.bfloat16 if p.dim() >= 2 else torch.float32)
                for k in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                    if k in st:
                        st[k] = st[k].to(torch.float32)
        self._pairs = _discover_pairs(self.param_groups, self._explicit_pairs)
        self._group_mult = [None] * len(self.param_groups)
        self._group_cos = [None] * len(self.param_groups)
