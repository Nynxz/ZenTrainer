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
EXACT_MAX_NUMEL = 1024  # tiny high-leverage matrices (e.g. lokr_w1 4x4..32x32) keep exact fp32 moments


def _is_exact(p: torch.Tensor) -> bool:
    return p.dim() < 2 or p.numel() <= EXACT_MAX_NUMEL


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


def _discover_kron_pairs(param_groups: list[dict], explicit_kron_pairs=None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Find direct Kronecker (lokr_w1, lokr_w2) pairs from explicit pairs or Network._own weakref tags."""
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if explicit_kron_pairs:
        for w1, w2 in explicit_kron_pairs:
            key = (id(w1), id(w2))
            if key not in seen:
                seen.add(key)
                pairs.append((w1, w2))
    all_params = {id(p): p for g in param_groups for p in g["params"]}
    for p in all_params.values():
        if getattr(p, "_zt_kron_side", None) != "w1":
            continue
        ref = getattr(p, "_zt_kron_partner", None)
        partner = ref() if callable(ref) else ref
        if partner is None or id(partner) not in all_params:
            continue
        key = (id(p), id(partner))
        if key not in seen:
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
        auto_lr: bool = False,
        lr_min_mult: float = 0.25,
        lr_max_mult: float = 4.0,
        lr_beta: float = 0.90,
        lr_rate: float = 0.03,
        lora_pairs=None,
        kron_pairs=None,
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
        self._explicit_kron_pairs = kron_pairs
        self._pairs = _discover_pairs(self.param_groups, lora_pairs)
        self._kron_pairs = _discover_kron_pairs(self.param_groups, kron_pairs)

    def _init_state(self, p: torch.Tensor) -> dict:
        st = self.state[p]
        st["step"] = 0
        # Store momentum in bf16 (with stochastic rounding) + factored second moments for large 2D+
        # weights; 1D and tiny high-leverage matrices (e.g. lokr_w1) keep exact fp32 moments.
        exact = _is_exact(p)
        st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32 if exact else torch.bfloat16)
        if exact:
            st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        else:
            st["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=torch.float32, device=p.device)
            st["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device)
        return st

    def _ensure_lr_ctrl(self, p: torch.Tensor) -> torch.Tensor:
        """Per-tensor on-device [lr_mult, ema_cos, ema_var] state (12 bytes) for auto_lr mode."""
        st = self.state[p]
        ctrl = st.get("lr_ctrl")
        if ctrl is None:
            ctrl = st["lr_ctrl"] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=p.device)
        return ctrl

    @staticmethod
    def _update_lr_ctrl(
        ctrls: list[torch.Tensor],
        align_rows: torch.Tensor,
        group: dict,
        step: int,
    ) -> torch.Tensor:
        """Per-tensor SNR-normalized adaptive LR update (zero GPU->CPU sync).

        Tracks smoothed alignment c_ema and its variance s2_ema per tensor; the dimensionless
        z-score z = c_ema / sqrt(s2_ema + 1e-6) works identically on noisy batch-size-1 diffusion
        and large batches. Climbs on real trends (z > 0.35), brakes 1.5x faster on overshoot
        (z < -0.25), gently relaxes mult > 1 toward 1.0 when trends fade, and holds steady >= 1.0
        on pure noise (no Automagic3 decay stall, no Automagic4 stuck-at-ceiling jitter).
        """
        ctrl = torch.stack(ctrls)  # (N, 3)
        dot, m2, g2 = align_rows[:, 0], align_rows[:, 1], align_rows[:, 2]
        valid = (m2 > 1e-24) & (g2 > 1e-24)
        cos = dot / ((m2 * g2).clamp_min(1e-24).sqrt() + 1e-12)
        mult, c_prev, s2_prev = ctrl[:, 0], ctrl[:, 1], ctrl[:, 2]
        bc = group["lr_beta"]
        diff = cos - c_prev
        c_new = torch.where(valid, c_prev * bc + cos * (1.0 - bc), c_prev)
        s2_new = torch.where(valid, s2_prev * bc + diff * diff * (1.0 - bc), s2_prev)
        if step >= 10:  # let variance EMA warm up before moving multipliers
            z = c_new / (s2_new + 1e-6).sqrt()
            rate = group["lr_rate"]
            up = mult * torch.exp(rate * (z - 0.35).clamp(0.0, 2.0))
            down = mult * torch.exp(rate * ((z + 0.25) * 1.5).clamp(-2.0, 0.0))
            relax = torch.where(mult > 1.0, 1.0 + (mult - 1.0) * 0.992, mult)
            m_new = torch.where(z > 0.35, up, torch.where(z < -0.25, down, relax)).clamp(
                group["lr_min_mult"], group["lr_max_mult"]
            )
            mult_new = torch.where(valid, m_new, mult)
        else:
            mult_new = mult
        new_ctrl = torch.stack((mult_new, c_new, s2_new), dim=1)
        torch._foreach_copy_(ctrls, list(new_ctrl.unbind(0)))
        return mult_new

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
                if group["auto_lr"]:
                    self._ensure_lr_ctrl(p)
                group_of[p] = gi
                if "exp_avg_sq" in st:
                    ones.append((gi, p))
                else:
                    key = (gi, tuple(p.shape), p.dtype, p.grad.dtype, st["exp_avg"].dtype, st["step"], p.device)
                    stacks[key].append(p)

        # 2. Compute momentum updates and per-tensor LR multipliers (when auto_lr is on)
        updates: dict[torch.Tensor, torch.Tensor] = {}
        chunk_records: list[tuple[int, list[torch.Tensor], torch.Tensor]] = []
        chunk_mults: list[torch.Tensor | None] = []
        one_mults: list[torch.Tensor | None] = []

        for key, ps in stacks.items():
            gi = key[0]
            group = self.param_groups[gi]
            per = max(1, CHUNK // ps[0].numel())
            for i in range(0, len(ps), per):
                chunk = ps[i:i + per]
                u_stack, align_rows = self._momentum_stack(chunk, group, _lotus_triton)
                if group["auto_lr"]:
                    ctrls = [self._ensure_lr_ctrl(p) for p in chunk]
                    if align_rows is not None:
                        mults = self._update_lr_ctrl(ctrls, align_rows, group, self.state[chunk[0]]["step"])
                    else:
                        mults = torch.stack(ctrls)[:, 0]
                else:
                    mults = None
                chunk_records.append((gi, chunk, u_stack))
                chunk_mults.append(mults)
                for p, u in zip(chunk, u_stack.unbind(0)):
                    updates[p] = u

        for gi, p in ones:
            group = self.param_groups[gi]
            u, align_row = self._momentum_1d(p, group)
            if group["auto_lr"]:
                ctrl = self._ensure_lr_ctrl(p)
                if align_row is not None:
                    mult = self._update_lr_ctrl([ctrl], align_row, group, self.state[p]["step"])[0]
                else:
                    mult = ctrl[0]
            else:
                mult = None
            updates[p] = u
            one_mults.append(mult)

        # 3. Batched Riemannian preconditioning: Gram whitening on (A, B) pairs and scale balancing on (w1, w2) Kronecker pairs
        p_stacks: dict[int, torch.Tensor] = {}
        self._precondition_pairs(updates, group_of, chunk_records, p_stacks)
        self._precondition_kron_pairs(updates, group_of)

        # 4. Apply decoupled weight decay, effective LR (scheduled or per-tensor auto), and stochastic rounding
        for ci, (gi, chunk, u_stack) in enumerate(chunk_records):
            group = self.param_groups[gi]
            lr_base, wd = group["lr"], group["weight_decay"]
            dev, p_dtype = chunk[0].device, chunk[0].dtype
            mults = chunk_mults[ci]
            p_stack = p_stacks.pop(ci, None)
            if _lotus_triton.supported(u_stack, p_dtype):
                _lotus_triton.apply_step(u_stack, chunk, mults, lr_base, wd, p_stack=p_stack)
            else:
                new_p = (p_stack if p_stack is not None else torch.stack(chunk)).to(torch.float32)
                if wd != 0.0:
                    u_stack.add_(new_p, alpha=wd)
                if mults is not None:
                    lr_eff = (mults * lr_base).view(-1, *([1] * (u_stack.dim() - 1)))
                    new_p.addcmul_(u_stack, lr_eff, value=-1.0)
                else:
                    new_p.add_(u_stack, alpha=-lr_base)
                if p_dtype == torch.float32:
                    torch._foreach_copy_(chunk, list(new_p.unbind(0)))
                else:
                    low = torch.empty((len(chunk),) + tuple(chunk[0].shape), dtype=p_dtype, device=dev)
                    _sr_truncate_(low, new_p)
                    torch._foreach_copy_(chunk, list(low.unbind(0)))

        for (gi, p), mult in zip(ones, one_mults):
            group = self.param_groups[gi]
            lr_base, wd = group["lr"], group["weight_decay"]
            u = updates[p]
            if p.dtype == torch.float32:
                if wd != 0.0:
                    u.add_(p, alpha=wd)
                if mult is not None:
                    p.addcmul_(u, mult * lr_base, value=-1.0)
                else:
                    p.add_(u, alpha=-lr_base)
            else:
                pf = p.to(torch.float32)
                if wd != 0.0:
                    u.add_(pf, alpha=wd)
                if mult is not None:
                    pf.addcmul_(u, mult * lr_base, value=-1.0)
                else:
                    pf.add_(u, alpha=-lr_base)
                _sr_truncate_(p, pf)

        for p in updates:
            p.grad = None
        return loss

    def _momentum_stack(self, ps: list[torch.Tensor], group: dict, triton_mod) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        do_align = bool(group["auto_lr"]) and step > 1
        if triton_mod.supported(gh, ps[0].dtype):
            return triton_mod.momentum_step(gh, ms, b1, 1.0 / bc1, caut, do_align)

        mp = torch.stack(ms).to(torch.float32)
        align_rows = None
        if do_align:
            align_rows = torch.stack(
                ((mp * gh).sum(dim=dims), (mp * mp).sum(dim=dims), (gh * gh).sum(dim=dims)),
                dim=1,
            )
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
        return u, align_rows

    def _momentum_1d(self, p: torch.Tensor, group: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        align_row = None
        if group["auto_lr"] and step > 1:
            align_row = torch.stack(((mp * gh).sum(), (mp * mp).sum(), (gh * gh).sum())).unsqueeze(0)
        mn = mp.mul_(b1).add_(gh, alpha=1.0 - b1)
        st["exp_avg"].copy_(mn)
        u = mn / bc1
        if caut < 1.0:
            cm = torch.where(u * gh > 0.0, 1.0, caut)
            u = u * cm * (cm * cm).mean().clamp_min(0.1).rsqrt()
        return u, align_row

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

    def _precondition_kron_pairs(
        self,
        updates: dict[torch.Tensor, torch.Tensor],
        group_of: dict[torch.Tensor, int],
    ) -> None:
        """Bounded Riemannian scale balancing for direct Kronecker (lokr_w1, lokr_w2) pairs.

        For dW = W1 kron W2, ||u1 kron W2||_F / ||W1 kron u2||_F = rms(W2) / rms(W1) regardless of
        tensor shapes. Scaling u1 by s1 = clamp(sqrt(rms(W1) / rms(W2)), 0.5, 2.0) and u2 by 1/s1
        preserves geometric-mean LR (s1 * s2 = 1) while letting the large expressive W2 factor
        adapt immediately instead of lagging behind zero-initialized W1 early in training.
        """
        if not self._kron_pairs:
            return
        buckets: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for w1, w2 in self._kron_pairs:
            if w1 not in updates or w2 not in updates:
                continue
            if self.param_groups[group_of[w1]]["precond_damp"] <= 0.0:
                continue
            buckets[(w1.device, tuple(w1.shape), tuple(w2.shape))].append((w1, w2))
        for (dev, s1_shape, s2_shape), pairs in buckets.items():
            W1 = torch.stack([w1.detach() for w1, _ in pairs]).to(torch.float32)
            W2 = torch.stack([w2.detach() for _, w2 in pairs]).to(torch.float32)
            r1 = torch.linalg.vector_norm(W1.flatten(1), dim=1) / math.sqrt(W1[0].numel())
            r2 = torch.linalg.vector_norm(W2.flatten(1), dim=1) / math.sqrt(W2[0].numel())
            valid = (r1 > 1e-6) & (r2 > 1e-6)
            sc1 = torch.where(valid, (r1 / r2.clamp_min(1e-12)).sqrt().clamp(0.5, 2.0), torch.ones_like(r1))
            sc2 = 1.0 / sc1
            torch._foreach_mul_([updates[w1] for w1, _ in pairs], list(sc1.view(-1, *([1] * len(s1_shape))).unbind(0)))
            torch._foreach_mul_([updates[w2] for _, w2 in pairs], list(sc2.view(-1, *([1] * len(s2_shape))).unbind(0)))

    def get_avg_learning_rate(self) -> float:
        """Mean effective learning rate across trainable tensors (scheduled LR, or scaled by each
        tensor's SNR-adaptive multiplier when auto_lr is on)."""
        vals: list[float] = []
        for g in self.param_groups:
            lr = float(g["lr"])
            if not g["auto_lr"]:
                vals.append(lr)
                continue
            ms = [float(self.state[p]["lr_ctrl"][0]) for p in g["params"] if "lr_ctrl" in self.state.get(p, {})]
            vals.append(lr * (sum(ms) / len(ms)) if ms else lr)
        return sum(vals) / len(vals) if vals else 0.0

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # Restore state dtypes (parent casts floating state to param.dtype)
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                if "exp_avg" in st:
                    st["exp_avg"] = st["exp_avg"].to(torch.float32 if "exp_avg_sq" in st else torch.bfloat16)
                for k in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq", "lr_ctrl"):
                    if k in st:
                        st[k] = st[k].to(torch.float32)
        self._pairs = _discover_pairs(self.param_groups, self._explicit_pairs)
        self._kron_pairs = _discover_kron_pairs(self.param_groups, self._explicit_kron_pairs)
