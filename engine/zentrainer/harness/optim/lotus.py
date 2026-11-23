"""Lotus: an adapter-native, memory-efficient optimizer for diffusion training.

Combines three ideas tailored specifically to low-rank diffusion adapters (LoRA, LoKr, LoHa):
1. Smoothed momentum stored in bf16 (with stochastic rounding) + Adafactor-factored second
   moments (beta2=0.999): denoises random-timestep diffusion gradients like AdamW at ~3x less
   optimizer VRAM, while tiny high-leverage matrices (e.g. lokr_w1, <= 1024 elems) and 1D
   parameters keep exact fp32 moments. Direct bf16/fp16 adapters train cleanly via Philox
   stochastic rounding.
2. Pre-momentum update-RMS trust region (clip=2.5) + norm-preserving cautious gating (0.25):
   stops outlier timestep spikes from poisoning momentum without choking normal diffusion
   timestep swings, and damps momentum overshoot without amplifying noise.
3. Optional group-level hypergradient adaptive LR (auto_lr, bounded in [0.05x, 20x]): pools
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
        auto_lr: bool = False,
        lr_min_mult: float = 0.05,
        lr_max_mult: float = 20.0,
        lr_beta: float = 0.90,
        lr_rate: float = 0.03,
        lr_floor: float = 0.04,
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
            auto_lr=auto_lr,
            lr_min_mult=lr_min_mult,
            lr_max_mult=lr_max_mult,
            lr_beta=lr_beta,
            lr_rate=lr_rate,
            lr_floor=lr_floor,
        )
        super().__init__(params, defaults)

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
        denom = torch.where(valid, (m2 * g2).sqrt(), torch.ones_like(dot))
        cos = torch.where(valid, dot / denom, torch.zeros_like(dot))
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

        base_t = torch.full((), base_lr, dtype=torch.float32, device=ctrl.device)
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

        # 2. Compute momentum updates, apply immediately per chunk, and accumulate group cosine alignment
        stepped_params: list[torch.Tensor] = []

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

                    align_rows = _lotus_triton.fused_inplace_step(
                        ps, ms, rows, cols,
                        b1=b1, b2=b2, bc1=bc1, bc2=bc2,
                        eps=group["eps"], clip=group["clip_threshold"],
                        cautious_min=group["cautious_min"], wd=group["weight_decay"],
                        lr_base=group["lr"], do_align=do_align,
                        lr_mult=g_mult,
                    )
                    if align_rows is not None and gi in align_acc:
                        sc, cnt = self._accum_align(align_rows)
                        acc_s, acc_c = align_acc[gi]
                        acc_s.add_(sc.to(acc_s.device))
                        acc_c.add_(cnt.to(acc_c.device))
                    stepped_params.extend(ps)
                    continue

            per = max(1, CHUNK // max(1, ps[0].numel()))
            lr_base, wd = group["lr"], group["weight_decay"]
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
                dev, p_dtype = chunk[0].device, chunk[0].dtype
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
                stepped_params.extend(chunk)

        # 3. Update group-level hypergradient LR controller once per active auto_lr group
        for gi, (ctrl, host_st) in group_ctrls.items():
            acc_s, acc_c = align_acc[gi]
            self._update_group_lr_ctrl(ctrl, host_st, acc_s, acc_c, self.param_groups[gi])

        for p in stepped_params:
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
        # Preserve original state dtypes before super() casts floating state to param dtype
        # (an fp16 adapter's second moments ~1e-11 would otherwise round to zero).
        saved = {
            i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
            for i, st in state_dict["state"].items()
        }
        super().load_state_dict(state_dict)
        for g in self.param_groups:
            for k, v in self.defaults.items():
                g.setdefault(k, v)
        ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for i, p in zip(ids, params):
            for k, v in saved.get(i, {}).items():
                if torch.is_tensor(v):
                    self.state[p][k] = v.to(p.device)
