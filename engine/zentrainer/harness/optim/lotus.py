"""Lotus: an adapter-native, memory-efficient optimizer for diffusion training.

Combines four ideas tailored specifically to low-rank diffusion adapters (LoRA, LoKr, LoHa):
1. Smoothed momentum stored in bf16 (with stochastic rounding) + Adafactor-factored second
   moments (beta2=0.999): denoises random-timestep diffusion gradients like AdamW at ~3x less
   optimizer VRAM, while tiny high-leverage matrices (e.g. lokr_w1, <= 1024 elems) and 1D
   parameters keep exact fp32 moments. Direct bf16/fp16 adapters train cleanly via Philox
   stochastic rounding.
2. Pre-momentum update-RMS trust region (clip=2.5) + norm-preserving cautious gating (0.25):
   stops outlier timestep spikes from poisoning momentum without choking normal diffusion
   timestep swings, and damps momentum overshoot without amplifying noise.
3. Bounded Riemannian Kronecker scale balancing on direct (lokr_w1, lokr_w2) pairs: scales
   w1 and w2 updates symmetrically (s1 * s2 = 1, clamped to [0.5, 2.0]) so the large expressive
   w2 factor adapts immediately instead of lagging behind zero-initialized w1.
4. Optional group-level hypergradient adaptive LR (auto_lr, bounded in [0.05x, 20x]): pools
   scale- and size-invariant per-tensor cosine alignment cos(m_{t-1}, gh_t) across the group
   (20 bytes/group, zero CPU sync), climbs on low LR, brakes on overshoot, pauses climbing
   during linear warmup and the final cosine decay tail, and holds steady on pure noise.

Batched across same-shaped tensors and accelerated with zero-copy in-place Triton kernels on
CUDA and AMD ROCm.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

CHUNK = 1 << 24  # max elements per shape stack (fallback for 3D/4D conv or non-contiguous tensors)
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


class Lotus(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-30,
        clip_threshold: float = 2.5,
        weight_decay: float = 0.0,
        cautious_min: float = 0.25,
        kron_balance: bool = True,
        auto_lr: bool = False,
        lr_min_mult: float = 0.05,
        lr_max_mult: float = 20.0,
        lr_beta: float = 0.90,
        lr_rate: float = 0.03,
        lr_floor: float = 0.04,
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
            kron_balance=kron_balance,
            auto_lr=auto_lr,
            lr_min_mult=lr_min_mult,
            lr_max_mult=lr_max_mult,
            lr_beta=lr_beta,
            lr_rate=lr_rate,
            lr_floor=lr_floor,
        )
        super().__init__(params, defaults)
        self._explicit_kron_pairs = kron_pairs
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

    def _ensure_group_lr_ctrl(self, group: dict) -> tuple[torch.Tensor, dict]:
        """Group-level on-device [mult, ema_cos, ema_var, prev_lr, peak_lr] state (20 bytes) for auto_lr.

        Stored on the group's first parameter so state_dict/load_state_dict persists it cleanly across
        devices; migrates old per-tensor `lr_ctrl` checkpoints by taking their mean multiplier.
        """
        p0 = group["params"][0]
        st = self.state[p0]
        ctrl = st.get("group_lr_ctrl")
        if ctrl is None:
            init_m = 1.0
            old = [self.state[p]["lr_ctrl"] for p in group["params"] if "lr_ctrl" in self.state.get(p, {})]
            if old:
                init_m = float(torch.stack([c.to(old[0].device) for c in old])[:, 0].mean().item())
            ctrl = st["group_lr_ctrl"] = torch.tensor(
                [init_m, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=p0.device
            )
            st.setdefault("group_lr_step", 0)
        return ctrl, st

    @staticmethod
    def _accum_align(align_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale- and size-invariant per-tensor cosine sum and valid count (zero GPU->CPU sync)."""
        dot, m2, g2 = align_rows[:, 0], align_rows[:, 1], align_rows[:, 2]
        valid = (m2 > 1e-24) & (g2 > 1e-24)
        cos = torch.where(valid, dot / ((m2 * g2).clamp_min(1e-24).sqrt() + 1e-12), torch.zeros_like(dot))
        return cos.sum(), valid.to(torch.float32).sum()

    @staticmethod
    def _update_group_lr_ctrl(
        ctrl: torch.Tensor,
        host_state: dict,
        sum_cos: torch.Tensor,
        count_valid: torch.Tensor,
        group: dict,
    ) -> None:
        """Group-level hypergradient adaptive LR update (zero GPU->CPU sync).

        Pools per-tensor cosines cos(m_{t-1}, gh_t) with equal weight per tensor so tiny high-leverage
        matrices (e.g. lokr_w1) have equal vote with large projections and coupled pairs (A/B, Q/K)
        never fight. Uses the true EMA standard error sqrt(v_ema * (1 - bc) / (1 + bc) + floor^2):
        rock-solid on pure noise, climbs on low LR, brakes immediately on overshoot, pauses climbing
        during linear warmup and the final cosine decay tail.
        """
        host_state["group_lr_step"] = int(host_state.get("group_lr_step", 0)) + 1
        step = host_state["group_lr_step"]
        base_lr = float(group["lr"])
        bc = float(group["lr_beta"])
        rate = float(group["lr_rate"])
        floor = float(group["lr_floor"])
        se_scale = (1.0 - bc) / (1.0 + bc)

        mult, c_prev, v_prev, prev_lr, peak_lr = ctrl.unbind(0)
        has_valid = count_valid > 0.0
        cos = sum_cos / count_valid.clamp_min(1.0)
        if step > 2:
            diff = cos - c_prev
            c_new = torch.where(has_valid, c_prev * bc + cos * (1.0 - bc), c_prev)
            v_new = torch.where(has_valid, v_prev * bc + diff * diff * (1.0 - bc), v_prev)
        else:
            c_new, v_new = c_prev, v_prev

        base_t = torch.tensor(base_lr, dtype=torch.float32, device=ctrl.device)
        if step >= 8:
            denom = (v_new * se_scale + floor * floor).sqrt()
            z = (c_new / denom).clamp(-2.0, 2.0)
            allow_up = (base_t <= prev_lr + 1e-15) & (base_t >= 0.5 * peak_lr)
            z = torch.where((z > 0.0) & ~allow_up, torch.zeros_like(z), z)
            m_cand = (mult * torch.exp(rate * z)).clamp(group["lr_min_mult"], group["lr_max_mult"])
            mult_new = torch.where(has_valid, m_cand, mult)
        else:
            mult_new = mult

        peak_new = torch.maximum(peak_lr, base_t)
        ctrl.copy_(torch.stack((mult_new, c_new, v_new, base_t, peak_new)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        from . import _lotus_triton

        # 1. Bucket active parameters by (group, shape, p.dtype, g.dtype, m.dtype, step, device)
        stacks: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        exact: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        group_of: dict[torch.Tensor, int] = {}

        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lotus does not support sparse gradients.")
                st = self.state[p]
                if "exp_avg" not in st:
                    st = self._init_state(p)
                group_of[p] = gi
                key = (gi, tuple(p.shape), p.dtype, p.grad.dtype, st["exp_avg"].dtype, st["step"], p.device)
                (exact if "exp_avg_sq" in st else stacks)[key].append(p)

        # Snapshot each active auto_lr group's multiplier once (1-step lag: zero cross-bucket sync)
        group_ctrls: dict[int, tuple[torch.Tensor, dict]] = {}
        group_mults: dict[int, torch.Tensor] = {}
        align_acc: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for gi in set(group_of.values()):
            group = self.param_groups[gi]
            if group["auto_lr"] and group["params"]:
                ctrl, host_st = self._ensure_group_lr_ctrl(group)
                group_ctrls[gi] = (ctrl, host_st)
                group_mults[gi] = ctrl[0].clone()
                align_acc[gi] = (
                    torch.zeros((), dtype=torch.float32, device=ctrl.device),
                    torch.zeros((), dtype=torch.float32, device=ctrl.device),
                )

        # 1b. Pre-compute Kronecker scalar balancing scales (w1, w2) so LoKr w2 stays on the
        # zero-copy pointer-array Triton fast path.
        kron_scale_map = self._compute_kron_scales(group_of)

        # 2. Compute momentum updates and accumulate group cosine alignment (or finish in place via Triton)
        updates: dict[torch.Tensor, torch.Tensor] = {}
        chunk_records: list[tuple[int, list[torch.Tensor], torch.Tensor]] = []
        chunk_mults: list[torch.Tensor | None] = []
        done_params: list[torch.Tensor] = []

        for key, ps, is_exact in [(k, v, False) for k, v in stacks.items()] + [(k, v, True) for k, v in exact.items()]:
            gi = key[0]
            group = self.param_groups[gi]
            g_mult = group_mults.get(gi)

            # Zero-copy pointer-array Triton fast path for contiguous 2D factored buckets
            if not is_exact:
                states = [self.state[p] for p in ps]
                ms = [s["exp_avg"] for s in states]
                if _lotus_triton.can_fused_inplace(ps, ms):
                    b1, b2 = group["betas"]
                    for s in states:
                        s["step"] += 1
                    step = states[0]["step"]
                    bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
                    rows = [s["exp_avg_sq_row"] for s in states]
                    cols = [s["exp_avg_sq_col"] for s in states]
                    do_align = bool(group["auto_lr"]) and step > 1

                    ks = None
                    if kron_scale_map and any(p in kron_scale_map for p in ps):
                        one = torch.ones((), dtype=torch.float32, device=ps[0].device)
                        ks = torch.stack([kron_scale_map.get(p, one) for p in ps])
                    align_rows = _lotus_triton.fused_inplace_step(
                        ps, ms, rows, cols,
                        b1=b1, b2=b2, bc1=bc1, bc2=bc2,
                        eps=group["eps"], clip=group["clip_threshold"],
                        cautious_min=group["cautious_min"], wd=group["weight_decay"],
                        lr_base=group["lr"], do_align=do_align,
                        lr_mult=g_mult, kron_scales=ks,
                    )
                    if align_rows is not None and gi in align_acc:
                        sc, cnt = self._accum_align(align_rows)
                        acc_s, acc_c = align_acc[gi]
                        acc_s.add_(sc.to(acc_s.device))
                        acc_c.add_(cnt.to(acc_c.device))
                    done_params.extend(ps)
                    continue

            per = max(1, CHUNK // max(1, ps[0].numel()))
            for i in range(0, len(ps), per):
                chunk = ps[i:i + per]
                if is_exact:
                    u_stack, align_rows = self._momentum_exact(chunk, group)
                else:
                    u_stack, align_rows = self._momentum_stack(chunk, group, _lotus_triton)
                if align_rows is not None and gi in align_acc:
                    sc, cnt = self._accum_align(align_rows)
                    acc_s, acc_c = align_acc[gi]
                    acc_s.add_(sc.to(acc_s.device))
                    acc_c.add_(cnt.to(acc_c.device))
                mults = g_mult.to(chunk[0].device).expand(len(chunk)).contiguous() if g_mult is not None else None
                chunk_records.append((gi, chunk, u_stack))
                chunk_mults.append(mults)
                for p, u in zip(chunk, u_stack.unbind(0)):
                    updates[p] = u

        # 3. Apply Kronecker balancing scales to remaining stacked/exact updates (e.g. lokr_w1)
        if kron_scale_map and updates:
            ks_ps = [p for p in updates if p in kron_scale_map]
            if ks_ps:
                torch._foreach_mul_(
                    [updates[p] for p in ks_ps],
                    [kron_scale_map[p].view(*([1] * p.dim())) for p in ks_ps],
                )

        # 4. Apply decoupled weight decay, effective LR, and stochastic rounding to fallback chunks
        for ci, (gi, chunk, u_stack) in enumerate(chunk_records):
            group = self.param_groups[gi]
            lr_base, wd = group["lr"], group["weight_decay"]
            dev, p_dtype = chunk[0].device, chunk[0].dtype
            mults = chunk_mults[ci]
            if _lotus_triton.supported(u_stack, p_dtype):
                _lotus_triton.apply_step(u_stack, chunk, mults, lr_base, wd)
            else:
                new_p = torch.stack(chunk).to(torch.float32)
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

        # 5. Update group-level hypergradient LR controller once per active auto_lr group
        for gi, (ctrl, host_st) in group_ctrls.items():
            acc_s, acc_c = align_acc[gi]
            self._update_group_lr_ctrl(ctrl, host_st, acc_s, acc_c, self.param_groups[gi])

        for p in done_params:
            p.grad = None
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
        inv_s = (r * c).mul_(math.sqrt(bc2))
        del row, col, r, c

        numel = ps[0].numel()
        dims = tuple(range(1, grad.dim()))
        ms = [s["exp_avg"] for s in states]
        do_align = bool(group["auto_lr"]) and step > 1

        gh = grad.mul_(inv_s)
        # Pre-momentum spike immunity: clip RMS and element extremes before updating momentum
        if clip < 1e8:
            rms = torch.linalg.vector_norm(gh, dim=dims, keepdim=True) / math.sqrt(numel)
            gh.div_((rms / clip).clamp_(min=1.0))
            gh.clamp_(-3.0 * clip, 3.0 * clip)

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

    def _momentum_exact(self, ps: list[torch.Tensor], group: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """_momentum_stack for 1D and tiny tensors: full fp32 moments, same-shaped ones in one stack."""
        states = [self.state[p] for p in ps]
        b1, b2 = group["betas"]
        eps, clip, caut = group["eps"], group["clip_threshold"], group["cautious_min"]
        for s in states:
            s["step"] += 1
        step = states[0]["step"]
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        n, lead = len(ps), (len(ps),) + (1,) * ps[0].dim()
        grad = torch.stack([p.grad for p in ps]).to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        vs = [s["exp_avg_sq"] for s in states]
        v = torch.stack(vs).mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        torch._foreach_copy_(vs, list(v.unbind(0)))
        inv_s = v.div_(bc2).add_(eps).rsqrt_()
        ms = [s["exp_avg"] for s in states]
        mp = torch.stack(ms)

        gh = grad.mul_(inv_s)
        if clip < 1e8:
            rms = torch.linalg.vector_norm(gh.reshape(n, -1), dim=1) / math.sqrt(ps[0].numel())
            gh.div_((rms / clip).clamp_(min=1.0).view(lead)).clamp_(-3.0 * clip, 3.0 * clip)
        align_rows = None
        if group["auto_lr"] and step > 1:
            g2, m2 = gh.reshape(n, -1), mp.reshape(n, -1)
            align_rows = torch.stack(((m2 * g2).sum(dim=1), (m2 * m2).sum(dim=1), (g2 * g2).sum(dim=1)), dim=1)
        mn = mp.mul_(b1).add_(gh, alpha=1.0 - b1)
        torch._foreach_copy_(ms, list(mn.unbind(0)))
        u = mn / bc1
        if caut < 1.0:
            cm = torch.where(u * gh > 0.0, 1.0, caut)
            u.mul_(cm).mul_((cm * cm).reshape(n, -1).mean(dim=1).clamp_min(0.1).rsqrt().view(lead))
        return u, align_rows

    def _compute_kron_scales(
        self,
        group_of: dict[torch.Tensor, int],
    ) -> dict[torch.Tensor, torch.Tensor]:
        """Bounded Riemannian scale balancing for direct Kronecker (lokr_w1, lokr_w2) pairs.

        For dW = W1 kron W2, ||u1 kron W2||_F / ||W1 kron u2||_F = rms(W2) / rms(W1) regardless of
        tensor shapes. Scaling u1 by s1 = clamp(sqrt(rms(W1) / rms(W2)), 0.5, 2.0) and u2 by 1/s1
        preserves geometric-mean LR (s1 * s2 = 1) while letting the large expressive W2 factor
        adapt immediately instead of lagging behind zero-initialized W1 early in training.
        """
        out: dict[torch.Tensor, torch.Tensor] = {}
        if not self._kron_pairs:
            return out
        buckets: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for w1, w2 in self._kron_pairs:
            if w1 not in group_of or w2 not in group_of:
                continue
            if not self.param_groups[group_of[w1]]["kron_balance"]:
                continue
            buckets[(w1.device, tuple(w1.shape), tuple(w2.shape))].append((w1, w2))
        for (dev, _s1_shape, _s2_shape), pairs in buckets.items():
            W1 = torch.stack([w1.detach() for w1, _ in pairs]).to(torch.float32)
            W2 = torch.stack([w2.detach() for _, w2 in pairs]).to(torch.float32)
            r1 = torch.linalg.vector_norm(W1.flatten(1), dim=1) / math.sqrt(W1[0].numel())
            r2 = torch.linalg.vector_norm(W2.flatten(1), dim=1) / math.sqrt(W2[0].numel())
            valid = (r1 > 1e-6) & (r2 > 1e-6)
            sc1 = torch.where(valid, (r1 / r2.clamp_min(1e-12)).sqrt().clamp(0.5, 2.0), torch.ones_like(r1))
            sc2 = 1.0 / sc1
            for (w1, w2), s1, s2 in zip(pairs, sc1.unbind(0), sc2.unbind(0)):
                out[w1] = s1
                out[w2] = s2
        return out

    def get_avg_learning_rate(self) -> float:
        """Mean effective learning rate across trainable tensors (scheduled LR, or scaled by each
        group's hypergradient multiplier when auto_lr is on)."""
        total, count = 0.0, 0
        for g in self.param_groups:
            lr, n = float(g["lr"]), len(g["params"])
            if not n:
                continue
            mult = 1.0
            if g["auto_lr"]:
                st0 = self.state.get(g["params"][0], {})
                if "group_lr_ctrl" in st0:
                    mult = float(st0["group_lr_ctrl"][0].item())
                else:
                    old = [self.state[p]["lr_ctrl"] for p in g["params"] if "lr_ctrl" in self.state.get(p, {})]
                    if old:
                        mult = float(torch.stack([c.to(old[0].device) for c in old])[:, 0].mean().item())
            total += lr * mult * n
            count += n
        return total / count if count else 0.0

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # The parent casts floating state to the param's dtype: an fp16 adapter's second moments
        # (~1e-11) round to zero there. Take each tensor from the saved state at its own dtype.
        saved = state_dict["state"]
        ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for i, p in zip(ids, params):
            for k, v in saved.get(i, {}).items():
                if torch.is_tensor(v):
                    self.state[p][k] = v.to(p.device).clone()
        self._kron_pairs = _discover_kron_pairs(self.param_groups, self._explicit_kron_pairs)
