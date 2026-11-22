"""Fused Triton kernels for the Lotus LoRA-native optimizer (CUDA and AMD ROCm).

Two lightweight kernels per shape stack replace ~15 eager PyTorch ops:
1. `_lotus_momentum_kernel`: updates bf16/fp32 EMA momentum with Philox stochastic rounding,
   computes norm-preserving cautious gating, and accumulates group momentum-gradient alignment
   (dot, ||m||^2, ||g_hat||^2) for zero-sync bounded LR self-regulation.
2. `_lotus_apply_kernel`: applies decoupled weight decay, effective LR (`group_lr * lr_mult`
   read on-device with zero GPU->CPU sync), and Philox bf16/fp16 stochastic rounding in one pass.
"""

from __future__ import annotations

import torch

_AVAILABLE: bool | None = None
_MOMENTUM_KERNEL = None
_APPLY_KERNEL = None
_FACTOR_SQ_KERNEL = None
_PRESTATS_KERNEL = None
_INPLACE_KERNEL = None
_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import triton  # noqa: F401

            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def supported(t: torch.Tensor, param_dtype: torch.dtype) -> bool:
    return t.is_cuda and param_dtype in _DTYPE_CODE and available()


def _get_momentum_kernel():
    global _MOMENTUM_KERNEL
    if _MOMENTUM_KERNEL is not None:
        return _MOMENTUM_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("seed", "b1", "inv_bc1", "cautious_min"))
    def _lotus_momentum_kernel(
        gh_ptr,        # (N, numel) fp32 normalized & clipped gradient
        m_ptr,         # (N, numel) bf16 or fp32 momentum (updated in place)
        u_ptr,         # (N, numel) fp32 output update (pre-cautious-norm)
        align_ptr,     # (N, 3) fp32 per-row [dot, m2, g2] atomically accumulated when DO_ALIGN
        msq_ptr,       # (N,) fp32 sum(mask^2) per row when DO_CAUTIOUS
        b1,            # float32 beta1
        inv_bc1,       # float32 1 / (1 - beta1^step)
        cautious_min,  # float32 floor for opposing-sign coordinates (e.g. 0.25)
        numel,         # int32 elements per tensor
        seed,          # int32 Philox seed for bf16 momentum stochastic rounding
        DO_ALIGN: tl.constexpr,
        DO_CAUTIOUS: tl.constexpr,
        M_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        base = row * numel + offs

        gh = tl.load(gh_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mp = tl.load(m_ptr + base, mask=mask, other=0.0).to(tl.float32)

        if DO_ALIGN:
            tl.atomic_add(align_ptr + row * 3 + 0, tl.sum(mp * gh))
            tl.atomic_add(align_ptr + row * 3 + 1, tl.sum(mp * mp))
            tl.atomic_add(align_ptr + row * 3 + 2, tl.sum(gh * gh))

        mn = mp * b1 + gh * (1.0 - b1)
        if M_BF16:
            r = tl.randint(seed, base).to(tl.int32, bitcast=True)
            as_i = mn.to(tl.int32, bitcast=True)
            as_i = (as_i + (r & 0xFFFF)) & (-65536)
            tl.store(m_ptr + base, as_i.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
        else:
            tl.store(m_ptr + base, mn, mask=mask)

        u = mn * inv_bc1
        if DO_CAUTIOUS:
            cm = tl.where(u * gh > 0.0, 1.0, cautious_min)
            cm = tl.where(mask, cm, 0.0)
            tl.atomic_add(msq_ptr + row, tl.sum(cm * cm))
            u = u * cm
        tl.store(u_ptr + base, u, mask=mask)

    _MOMENTUM_KERNEL = _lotus_momentum_kernel
    return _MOMENTUM_KERNEL


def _get_apply_kernel():
    global _APPLY_KERNEL
    if _APPLY_KERNEL is not None:
        return _APPLY_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("seed", "lr_base", "wd"))
    def _lotus_apply_kernel(
        u_ptr,         # (N, numel) fp32 final update
        p_ptr,         # (N, numel) fp32/bf16/fp16 params (updated in place)
        mult_ptr,      # (N,) fp32 on-device per-tensor lr_mult when HAS_MULT (zero GPU->CPU sync)
        lr_base,       # float32 scheduled group["lr"]
        wd,            # float32 decoupled weight decay
        numel,         # int32 elements per tensor
        seed,          # int32 Philox seed for bf16/fp16 stochastic rounding
        HAS_MULT: tl.constexpr,
        HAS_WD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,  # 0=fp32, 1=bf16, 2=fp16
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        base = row * numel + offs

        u = tl.load(u_ptr + base, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + base, mask=mask, other=0.0).to(tl.float32)
        lr = lr_base * tl.load(mult_ptr + row).to(tl.float32) if HAS_MULT else lr_base
        if HAS_WD:
            u = u + wd * p
        new_p = p - lr * u

        if OUT_DTYPE == 0:
            tl.store(p_ptr + base, new_p, mask=mask)
        else:
            drop_bits: tl.constexpr = 16 if OUT_DTYPE == 1 else 13
            low_mask: tl.constexpr = (1 << drop_bits) - 1
            hi_mask: tl.constexpr = -(1 << drop_bits)
            r = tl.randint(seed, base).to(tl.int32, bitcast=True)
            as_i = new_p.to(tl.int32, bitcast=True)
            as_i = (as_i + (r & low_mask)) & hi_mask
            out_t: tl.constexpr = tl.bfloat16 if OUT_DTYPE == 1 else tl.float16
            tl.store(p_ptr + base, as_i.to(tl.float32, bitcast=True).to(out_t), mask=mask)

    _APPLY_KERNEL = _lotus_apply_kernel
    return _APPLY_KERNEL


def momentum_step(
    gh: torch.Tensor,
    ms: list[torch.Tensor],
    b1: float,
    inv_bc1: float,
    cautious_min: float,
    do_align: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Update stacked momentum tensors in place and return ((N, *shape) fp32 update, (N, 3) align or None)."""
    import triton

    n = len(ms)
    shape = gh.shape[1:]
    numel = gh[0].numel()
    dev = gh.device
    gh2 = gh.reshape(n, numel).contiguous()
    m_stack = torch.stack(ms).reshape(n, numel).contiguous()
    u2 = torch.empty((n, numel), dtype=torch.float32, device=dev)
    do_caut = cautious_min < 1.0
    align_buf = torch.zeros((n, 3), dtype=torch.float32, device=dev) if do_align else u2
    msq = torch.zeros((n,), dtype=torch.float32, device=dev) if do_caut else u2
    m_bf16 = ms[0].dtype == torch.bfloat16
    seed = int(torch.randint(0, 1 << 30, (), device="cpu").item()) if m_bf16 else 0

    BLOCK = 1024
    grid = (n, triton.cdiv(numel, BLOCK))
    _get_momentum_kernel()[grid](
        gh2, m_stack, u2, align_buf, msq,
        float(b1), float(inv_bc1), float(cautious_min), numel, seed,
        DO_ALIGN=do_align,
        DO_CAUTIOUS=do_caut,
        M_BF16=m_bf16,
        BLOCK=BLOCK,
        num_warps=4,
    )
    torch._foreach_copy_(ms, list(m_stack.view((n,) + shape).unbind(0)))
    if do_caut:
        # Preserve L2 step norm without amplifying noise on surviving coordinates
        u2.mul_((msq / numel).clamp_min(0.1).rsqrt().unsqueeze(1))
    return u2.view((n,) + shape), (align_buf if do_align else None)


def apply_step(
    u: torch.Tensor,
    ps: list[torch.Tensor],
    lr_mults: torch.Tensor | None,
    lr_base: float,
    wd: float,
    p_stack: torch.Tensor | None = None,
) -> None:
    """Apply decoupled weight decay, effective LR (optionally per-tensor), and stochastic rounding."""
    import triton

    n = len(ps)
    shape = ps[0].shape
    numel = ps[0].numel()
    u2 = u.reshape(n, numel).contiguous()
    p_buf = (p_stack if p_stack is not None else torch.stack(ps)).reshape(n, numel).contiguous()
    out_dtype = _DTYPE_CODE[ps[0].dtype]
    seed = int(torch.randint(0, 1 << 30, (), device="cpu").item()) if out_dtype != 0 else 0
    has_mult = lr_mults is not None
    mult_buf = lr_mults.reshape(n).contiguous() if has_mult else u2

    BLOCK = 1024
    grid = (n, triton.cdiv(numel, BLOCK))
    _get_apply_kernel()[grid](
        u2, p_buf, mult_buf, float(lr_base), float(wd), numel, seed,
        HAS_MULT=has_mult,
        HAS_WD=(wd != 0.0),
        OUT_DTYPE=out_dtype,
        BLOCK=BLOCK,
        num_warps=4,
    )
    torch._foreach_copy_(ps, list(p_buf.view((n,) + shape).unbind(0)))


# ---- Zero-copy in-place pointer-array fused fast path (2D factored tensors) ----------------------------
# Avoids torch.stack(grads), torch.stack(ms), torch.stack(ps), and all torch._foreach_copy_ round-trips:
# reads/writes each parameter and momentum tensor directly via an (N,) int64 pointer table on CUDA & ROCm.


def _get_factor_sq_kernel():
    global _FACTOR_SQ_KERNEL
    if _FACTOR_SQ_KERNEL is not None:
        return _FACTOR_SQ_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("R", "C", "inv_R", "inv_C"))
    def _lotus_factor_sq_kernel(
        g_ptrs,       # (N,) int64 pointers to (R, C) contiguous grad tensors
        sq_row_ptr,   # (N, R) fp32 zeroed output: mean_c(g^2)
        sq_col_ptr,   # (N, C) fp32 zeroed output: mean_r(g^2)
        R,            # int32 rows
        C,            # int32 cols
        inv_R,        # float32 1 / R
        inv_C,        # float32 1 / C
        G_DTYPE: tl.constexpr,  # 0=fp32, 1=bf16, 2=fp16
        BR: tl.constexpr,
        BC: tl.constexpr,
    ):
        tid = tl.program_id(0)
        pr = tl.program_id(1)
        pc = tl.program_id(2)
        r_offs = pr * BR + tl.arange(0, BR)
        c_offs = pc * BC + tl.arange(0, BC)
        r_mask = r_offs < R
        c_mask = c_offs < C
        mask = r_mask[:, None] & c_mask[None, :]

        g_t: tl.constexpr = tl.float32 if G_DTYPE == 0 else (tl.bfloat16 if G_DTYPE == 1 else tl.float16)
        g_base = tl.load(g_ptrs + tid).to(tl.pointer_type(g_t))
        g = tl.load(g_base + r_offs[:, None] * C + c_offs[None, :], mask=mask, other=0.0).to(tl.float32)
        g = tl.where((g == g) & (tl.abs(g) < 3.4e38), g, 0.0)
        sq = g * g

        r_sum = tl.sum(sq, axis=1) * inv_C
        c_sum = tl.sum(sq, axis=0) * inv_R
        tl.atomic_add(sq_row_ptr + tid * R + r_offs, r_sum, mask=r_mask)
        tl.atomic_add(sq_col_ptr + tid * C + c_offs, c_sum, mask=c_mask)

    _FACTOR_SQ_KERNEL = _lotus_factor_sq_kernel
    return _FACTOR_SQ_KERNEL


def _get_prestats_kernel():
    global _PRESTATS_KERNEL
    if _PRESTATS_KERNEL is not None:
        return _PRESTATS_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("R", "C", "numel", "b1", "inv_bc1", "clip", "cautious_min"))
    def _lotus_prestats_kernel(
        g_ptrs,        # (N,) int64 pointers to (R, C) grad tensors
        m_ptrs,        # (N,) int64 pointers to (R, C) momentum tensors
        r_scale_ptr,   # (N, R) fp32 rsqrt(row / mean_row) * sqrt(bc2)
        c_scale_ptr,   # (N, C) fp32 rsqrt(col)
        sq_sum_ptr,    # (N,) fp32 sum(gh^2) or sum(u^2) when DO_CLIP
        msq_ptr,       # (N,) fp32 sum(cm^2) when DO_CAUTIOUS
        align_ptr,     # (N, 3) fp32 [dot, m2, g2] when DO_ALIGN
        R,
        C,
        numel,
        b1,
        inv_bc1,
        clip,
        cautious_min,
        G_DTYPE: tl.constexpr,
        M_BF16: tl.constexpr,
        DO_CLIP: tl.constexpr,
        DO_CAUTIOUS: tl.constexpr,
        DO_ALIGN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tid = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        r_idx = offs // C
        c_idx = offs - r_idx * C

        g_t: tl.constexpr = tl.float32 if G_DTYPE == 0 else (tl.bfloat16 if G_DTYPE == 1 else tl.float16)
        g_base = tl.load(g_ptrs + tid).to(tl.pointer_type(g_t))
        g = tl.load(g_base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.where((g == g) & (tl.abs(g) < 3.4e38), g, 0.0)

        rs = tl.load(r_scale_ptr + tid * R + r_idx, mask=mask, other=0.0)
        cs = tl.load(c_scale_ptr + tid * C + c_idx, mask=mask, other=0.0)
        gh = g * (rs * cs)
        if DO_CLIP:
            tl.atomic_add(sq_sum_ptr + tid, tl.sum(gh * gh))
        if DO_CAUTIOUS or DO_ALIGN:
            m_t: tl.constexpr = tl.bfloat16 if M_BF16 else tl.float32
            m_base = tl.load(m_ptrs + tid).to(tl.pointer_type(m_t))
            mp = tl.load(m_base + offs, mask=mask, other=0.0).to(tl.float32)
            if DO_ALIGN:
                # cos(mp, gh) is scale-invariant under positive rms scaling
                tl.atomic_add(align_ptr + tid * 3 + 0, tl.sum(mp * gh))
                tl.atomic_add(align_ptr + tid * 3 + 1, tl.sum(mp * mp))
                tl.atomic_add(align_ptr + tid * 3 + 2, tl.sum(gh * gh))
            if DO_CAUTIOUS:
                gh_c = tl.maximum(tl.minimum(gh, 3.0 * clip), -3.0 * clip) if DO_CLIP else gh
                mn = mp * b1 + gh_c * (1.0 - b1)
                cm = tl.where(mn * gh_c > 0.0, 1.0, cautious_min)
                cm = tl.where(mask, cm, 0.0)
                tl.atomic_add(msq_ptr + tid, tl.sum(cm * cm))

    _PRESTATS_KERNEL = _lotus_prestats_kernel
    return _PRESTATS_KERNEL


def _get_inplace_kernel():
    global _INPLACE_KERNEL
    if _INPLACE_KERNEL is not None:
        return _INPLACE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("R", "C", "numel", "b1", "inv_bc1", "clip", "cautious_min", "wd", "m_seed", "p_seed"))
    def _lotus_fused_inplace_kernel(
        p_ptrs,        # (N,) int64 pointers to (R, C) param tensors (updated in place)
        g_ptrs,        # (N,) int64 pointers to (R, C) grad tensors
        m_ptrs,        # (N,) int64 pointers to (R, C) momentum tensors (updated in place)
        r_scale_ptr,   # (N, R) fp32
        c_scale_ptr,   # (N, C) fp32
        rms_scale_ptr, # (N,) fp32 1 / max(1, rms / clip)
        caut_scale_ptr,# (N,) fp32 rsqrt(max(0.1, msq / numel))
        u_scale_ptr,   # (N,) fp32 per-tensor update scale (e.g. Kronecker balancing) when HAS_USCALE
        eff_lr_ptr,    # (N,) fp32 lr_base * auto_mult
        R,
        C,
        numel,
        b1,
        inv_bc1,
        clip,
        cautious_min,
        wd,
        m_seed,
        p_seed,
        P_DTYPE: tl.constexpr,
        G_DTYPE: tl.constexpr,
        M_BF16: tl.constexpr,
        DO_CLIP: tl.constexpr,
        DO_CAUTIOUS: tl.constexpr,
        HAS_USCALE: tl.constexpr,
        HAS_WD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tid = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        r_idx = offs // C
        c_idx = offs - r_idx * C

        p_t: tl.constexpr = tl.float32 if P_DTYPE == 0 else (tl.bfloat16 if P_DTYPE == 1 else tl.float16)
        g_t: tl.constexpr = tl.float32 if G_DTYPE == 0 else (tl.bfloat16 if G_DTYPE == 1 else tl.float16)
        m_t: tl.constexpr = tl.bfloat16 if M_BF16 else tl.float32

        p_base = tl.load(p_ptrs + tid).to(tl.pointer_type(p_t))
        g_base = tl.load(g_ptrs + tid).to(tl.pointer_type(g_t))
        m_base = tl.load(m_ptrs + tid).to(tl.pointer_type(m_t))

        p = tl.load(p_base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.where((g == g) & (tl.abs(g) < 3.4e38), g, 0.0)
        mp = tl.load(m_base + offs, mask=mask, other=0.0).to(tl.float32)

        rs = tl.load(r_scale_ptr + tid * R + r_idx, mask=mask, other=0.0)
        cs = tl.load(c_scale_ptr + tid * C + c_idx, mask=mask, other=0.0)
        gh = g * (rs * cs)
        if DO_CLIP:
            gh = gh * tl.load(rms_scale_ptr + tid)
            gh = tl.maximum(tl.minimum(gh, 3.0 * clip), -3.0 * clip)
        mn = mp * b1 + gh * (1.0 - b1)
        u = mn * inv_bc1

        # Store updated momentum in place
        if M_BF16:
            rm = tl.randint(m_seed, tid * numel + offs).to(tl.int32, bitcast=True)
            mi = mn.to(tl.int32, bitcast=True)
            mi = (mi + (rm & 0xFFFF)) & (-65536)
            tl.store(m_base + offs, mi.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
        else:
            tl.store(m_base + offs, mn, mask=mask)

        if DO_CAUTIOUS:
            cm = tl.where(u * gh > 0.0, 1.0, cautious_min)
            u = u * cm * tl.load(caut_scale_ptr + tid)
        if HAS_USCALE:
            u = u * tl.load(u_scale_ptr + tid)

        lr = tl.load(eff_lr_ptr + tid)
        if HAS_WD:
            u = u + wd * p
        new_p = p - lr * u

        if P_DTYPE == 0:
            tl.store(p_base + offs, new_p, mask=mask)
        else:
            drop_bits: tl.constexpr = 16 if P_DTYPE == 1 else 13
            low_mask: tl.constexpr = (1 << drop_bits) - 1
            hi_mask: tl.constexpr = -(1 << drop_bits)
            rp = tl.randint(p_seed, tid * numel + offs).to(tl.int32, bitcast=True)
            pi = new_p.to(tl.int32, bitcast=True)
            pi = (pi + (rp & low_mask)) & hi_mask
            tl.store(p_base + offs, pi.to(tl.float32, bitcast=True).to(p_t), mask=mask)

    _INPLACE_KERNEL = _lotus_fused_inplace_kernel
    return _INPLACE_KERNEL


def can_fused_inplace(ps: list[torch.Tensor], ms: list[torch.Tensor]) -> bool:
    """Whether this shape bucket can run the zero-copy pointer-array Triton fast path."""
    if not ps or not available():
        return False
    p0 = ps[0]
    if not (p0.is_cuda and p0.dim() == 2 and p0.dtype in _DTYPE_CODE and p0.grad.dtype in _DTYPE_CODE):
        return False
    return all(p.is_contiguous() and p.grad.is_contiguous() and m.is_contiguous() for p, m in zip(ps, ms))


def fused_inplace_step(
    ps: list[torch.Tensor],
    ms: list[torch.Tensor],
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    *,
    b1: float,
    b2: float,
    bc1: float,
    bc2: float,
    eps: float,
    clip: float,
    cautious_min: float,
    wd: float,
    lr_base: float,
    do_align: bool,
    lr_mult: torch.Tensor | None = None,
    kron_scales: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Full Lotus step on a 2D shape bucket in place via pointer tables (0 tensor stacks/copies)."""
    import math
    import triton

    n = len(ps)
    R, C = ps[0].shape
    numel = R * C
    dev = ps[0].device
    p_code = _DTYPE_CODE[ps[0].dtype]
    g_code = _DTYPE_CODE[ps[0].grad.dtype]
    m_bf16 = ms[0].dtype == torch.bfloat16

    p_ptrs = torch.tensor([p.data_ptr() for p in ps], dtype=torch.int64, device=dev)
    g_ptrs = torch.tensor([p.grad.data_ptr() for p in ps], dtype=torch.int64, device=dev)
    m_ptrs = torch.tensor([m.data_ptr() for m in ms], dtype=torch.int64, device=dev)

    # 1. Factored row/col squared-gradient reduction directly from g_ptrs
    sq_row = torch.zeros((n, R), dtype=torch.float32, device=dev)
    sq_col = torch.zeros((n, C), dtype=torch.float32, device=dev)
    if R <= 32 and C > 32:
        BR, BC = 16, 64
    elif C <= 32 and R > 32:
        BR, BC = 64, 16
    else:
        BR, BC = 32, 32
    _get_factor_sq_kernel()[(n, triton.cdiv(R, BR), triton.cdiv(C, BC))](
        g_ptrs, sq_row, sq_col, R, C, 1.0 / R, 1.0 / C,
        G_DTYPE=g_code, BR=BR, BC=BC, num_warps=4,
    )
    row = torch.stack(rows).mul_(b2).add_(sq_row.add_(eps), alpha=1.0 - b2)
    col = torch.stack(cols).mul_(b2).add_(sq_col.add_(eps), alpha=1.0 - b2)
    torch._foreach_copy_(rows, list(row.unbind(0)))
    torch._foreach_copy_(cols, list(col.unbind(0)))
    r_scale = (row / row.mean(dim=-1, keepdim=True).clamp_min(eps)).rsqrt_().mul_(math.sqrt(bc2)).contiguous()
    c_scale = col.rsqrt().contiguous()

    # 2. Optional pre-stats pass (rms clipping, cautious norm factor, per-tensor alignment)
    do_clip = clip < 1e8
    do_caut = cautious_min < 1.0
    BLOCK = 1024
    grid = (n, triton.cdiv(numel, BLOCK))
    sq_sum = torch.zeros((n,), dtype=torch.float32, device=dev) if do_clip else r_scale
    msq = torch.zeros((n,), dtype=torch.float32, device=dev) if do_caut else r_scale
    align_buf = torch.zeros((n, 3), dtype=torch.float32, device=dev) if do_align else r_scale
    if do_clip or do_caut or do_align:
        _get_prestats_kernel()[grid](
            g_ptrs, m_ptrs, r_scale, c_scale, sq_sum, msq, align_buf,
            R, C, numel, float(b1), 1.0 / bc1, float(clip), float(cautious_min),
            G_DTYPE=g_code, M_BF16=m_bf16,
            DO_CLIP=do_clip, DO_CAUTIOUS=do_caut, DO_ALIGN=do_align,
            BLOCK=BLOCK, num_warps=4,
        )

    rms_scale = (clip / (sq_sum / numel).sqrt().clamp_min(clip)).contiguous() if do_clip else r_scale
    caut_scale = (msq / numel).clamp_min(0.1).rsqrt().contiguous() if do_caut else r_scale
    if lr_mult is not None:
        eff_lr = lr_mult.to(device=dev, dtype=torch.float32).mul(float(lr_base)).expand(n).contiguous()
    else:
        eff_lr = torch.full((n,), float(lr_base), dtype=torch.float32, device=dev)
    has_uscale = kron_scales is not None
    u_scale = kron_scales.contiguous() if has_uscale else r_scale

    # 3. Single in-place pass: update momentum + parameter with Philox stochastic rounding
    m_seed = int(torch.randint(0, 1 << 30, (), device="cpu").item()) if m_bf16 else 0
    p_seed = int(torch.randint(0, 1 << 30, (), device="cpu").item()) if p_code != 0 else 0
    _get_inplace_kernel()[grid](
        p_ptrs, g_ptrs, m_ptrs, r_scale, c_scale, rms_scale, caut_scale, u_scale, eff_lr,
        R, C, numel, float(b1), 1.0 / bc1, float(clip), float(cautious_min), float(wd),
        m_seed, p_seed,
        P_DTYPE=p_code, G_DTYPE=g_code, M_BF16=m_bf16,
        DO_CLIP=do_clip, DO_CAUTIOUS=do_caut, HAS_USCALE=has_uscale, HAS_WD=(wd != 0.0),
        BLOCK=BLOCK, num_warps=4,
    )
    return align_buf if do_align else None
