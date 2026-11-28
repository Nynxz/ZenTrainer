"""Fused Triton kernels for the Lotus LoRA-native optimizer (CUDA and AMD ROCm).

Two lightweight kernels per shape stack replace ~15 eager PyTorch ops:
1. `_lotus_momentum_kernel`: updates int8/bf16/fp32 EMA momentum with Philox stochastic rounding,
   computes norm-preserving cautious gating, and accumulates group momentum-gradient alignment
   (dot, ||m||^2, ||g_hat||^2) for zero-sync bounded LR self-regulation.
2. `_lotus_apply_kernel`: applies decoupled weight decay, effective LR (`group_lr * lr_mult`
   read on-device with zero GPU->CPU sync), and Philox bf16/fp16 stochastic rounding (including
   unbiased subnormal rounding for fp16) in one pass.
"""

from __future__ import annotations

import random

import torch

_AVAILABLE: bool | None = None
triton = None
tl = None
_MOMENTUM_KERNEL = None
_APPLY_KERNEL = None
_FACTOR_SQ_KERNEL = None
_RMS_KERNEL = None
_PRESTATS_KERNEL = None
_INPLACE_KERNEL = None
_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_M_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.int8: 2}
_PTR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def available() -> bool:
    global _AVAILABLE, triton, tl
    if _AVAILABLE is None:
        try:
            import triton as _triton
            import triton.language as _tl

            triton = _triton
            tl = _tl
            _AVAILABLE = True
        except (ImportError, OSError, RuntimeError):
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
        m_ptr,         # (N, numel) int8, bf16, or fp32 momentum (updated in place)
        u_ptr,         # (N, numel) fp32 output update (pre-cautious-norm)
        align_ptr,     # (N, 3) fp32 per-row [dot, m2, g2] atomically accumulated when DO_ALIGN
        msq_ptr,       # (N,) fp32 sum(mask^2) per row when DO_CAUTIOUS
        b1,            # float32 beta1
        inv_bc1,       # float32 1 / (1 - beta1^step)
        cautious_min,  # float32 floor for opposing-sign coordinates (e.g. 0.25)
        numel,         # int32 elements per tensor
        seed,          # int32 Philox seed for int8/bf16 momentum stochastic rounding
        DO_ALIGN: tl.constexpr,
        DO_CAUTIOUS: tl.constexpr,
        M_MODE: tl.constexpr,  # 0=fp32, 1=bf16, 2=int8 quadratic-companded SR
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        base = row * numel + offs

        gh = tl.load(gh_ptr + base, mask=mask, other=0.0).to(tl.float32)
        if M_MODE == 2:
            qf = tl.load(m_ptr + base, mask=mask, other=0).to(tl.float32)
            t = qf * 0.007874015748031496
            mp = tl.where(qf < 0.0, -4.0, 4.0) * (t * t)
        else:
            mp = tl.load(m_ptr + base, mask=mask, other=0.0).to(tl.float32)

        if DO_ALIGN:
            tl.atomic_add(align_ptr + row * 3 + 0, tl.sum(mp * gh))
            tl.atomic_add(align_ptr + row * 3 + 1, tl.sum(mp * mp))
            tl.atomic_add(align_ptr + row * 3 + 2, tl.sum(gh * gh))

        mn = mp * b1 + gh * (1.0 - b1)
        if M_MODE == 1:
            r = tl.randint(seed, base).to(tl.int32, bitcast=True)
            as_i = mn.to(tl.int32, bitcast=True)
            as_i = (as_i + (r & 0xFFFF)) & (-65536)
            tl.store(m_ptr + base, as_i.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
        elif M_MODE == 2:
            ax = tl.minimum(tl.abs(mn), 4.0)
            k = tl.minimum(tl.floor(tl.sqrt(ax * 0.25) * 127.0), 126.0)
            lo_t = k * 0.007874015748031496
            lo = lo_t * lo_t * 4.0
            hi_t = (k + 1.0) * 0.007874015748031496
            hi = hi_t * hi_t * 4.0
            p_up = (ax - lo) / tl.maximum(hi - lo, 1e-12)
            rm = (tl.randint(seed, base).to(tl.int32, bitcast=True) & 0xFFFFFF).to(tl.float32) * 5.960464477539063e-8
            q_abs = k + tl.where(rm < p_up, 1.0, 0.0)
            q = tl.where(mn < 0.0, -q_abs, q_abs).to(tl.int8)
            tl.store(m_ptr + base, q, mask=mask)
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
        elif OUT_DTYPE == 1:
            r = tl.randint(seed, base).to(tl.int32, bitcast=True)
            as_i = new_p.to(tl.int32, bitcast=True)
            as_i = (as_i + (r & 0xFFFF)) & (-65536)
            tl.store(p_ptr + base, as_i.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
        else:
            x = tl.maximum(tl.minimum(new_p, 65504.0), -65504.0)
            r = tl.randint(seed, base).to(tl.int32, bitcast=True)
            norm_i = (x.to(tl.int32, bitcast=True) + (r & 0x1FFF)) & (-8192)
            norm_f = norm_i.to(tl.float32, bitcast=True)
            u = (r & 0xFFFFFF).to(tl.float32) * 5.960464477539063e-8
            sub_f = tl.floor(x * 16777216.0 + u) * 5.960464477539063e-8
            out_f = tl.where(tl.abs(x) < 6.103515625e-5, sub_f, norm_f)
            tl.store(p_ptr + base, out_f.to(tl.float16), mask=mask)

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
    m_mode = _M_DTYPE_CODE[ms[0].dtype]
    seed = random.getrandbits(30) if m_mode != 0 else 0

    BLOCK = 1024
    grid = (n, triton.cdiv(numel, BLOCK))
    _get_momentum_kernel()[grid](
        gh2, m_stack, u2, align_buf, msq,
        float(b1), float(inv_bc1), float(cautious_min), numel, seed,
        DO_ALIGN=do_align,
        DO_CAUTIOUS=do_caut,
        M_MODE=m_mode,
        BLOCK=BLOCK,
        num_warps=4,
    )
    torch._foreach_copy_(ms, list(m_stack.view((n,) + shape).unbind(0)))
    if do_caut:
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
    seed = random.getrandbits(30) if out_dtype != 0 else 0
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


def _get_rms_kernel():
    global _RMS_KERNEL
    if _RMS_KERNEL is not None:
        return _RMS_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("R", "C", "numel"))
    def _lotus_rms_kernel(
        g_ptrs,        # (N,) int64 pointers to (R, C) grad tensors
        r_scale_ptr,   # (N, R) fp32 rsqrt(row / mean_row) * sqrt(bc2)
        c_scale_ptr,   # (N, C) fp32 rsqrt(col)
        sq_sum_ptr,    # (N,) fp32 sum(gh^2)
        R,
        C,
        numel,
        G_DTYPE: tl.constexpr,
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
        tl.atomic_add(sq_sum_ptr + tid, tl.sum(gh * gh))

    _RMS_KERNEL = _lotus_rms_kernel
    return _RMS_KERNEL


def _get_prestats_kernel():
    global _PRESTATS_KERNEL
    if _PRESTATS_KERNEL is not None:
        return _PRESTATS_KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("R", "C", "numel", "b1", "clip", "cautious_min"))
    def _lotus_prestats_kernel(
        g_ptrs,        # (N,) int64 pointers to (R, C) grad tensors
        m_ptrs,        # (N,) int64 pointers to (R, C) momentum tensors
        r_scale_ptr,   # (N, R) fp32 rsqrt(row / mean_row) * sqrt(bc2)
        c_scale_ptr,   # (N, C) fp32 rsqrt(col)
        rms_scale_ptr, # (N,) fp32 1 / max(1, rms / clip) when DO_CLIP
        msq_ptr,       # (N,) fp32 sum(cm^2) when DO_CAUTIOUS
        align_ptr,     # (N, 3) fp32 [dot, m2, g2] when DO_ALIGN
        R,
        C,
        numel,
        b1,
        clip,
        cautious_min,
        G_DTYPE: tl.constexpr,
        M_MODE: tl.constexpr,  # 0=fp32, 1=bf16, 2=int8 quadratic-companded SR
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
            gh = gh * tl.load(rms_scale_ptr + tid)
            gh = tl.maximum(tl.minimum(gh, 3.0 * clip), -3.0 * clip)

        m_t: tl.constexpr = tl.float32 if M_MODE == 0 else (tl.bfloat16 if M_MODE == 1 else tl.int8)
        m_base = tl.load(m_ptrs + tid).to(tl.pointer_type(m_t))
        if M_MODE == 2:
            qf = tl.load(m_base + offs, mask=mask, other=0).to(tl.float32)
            t = qf * 0.007874015748031496
            mp = tl.where(qf < 0.0, -4.0, 4.0) * (t * t)
        else:
            mp = tl.load(m_base + offs, mask=mask, other=0.0).to(tl.float32)
        if DO_ALIGN:
            tl.atomic_add(align_ptr + tid * 3 + 0, tl.sum(mp * gh))
            tl.atomic_add(align_ptr + tid * 3 + 1, tl.sum(mp * mp))
            tl.atomic_add(align_ptr + tid * 3 + 2, tl.sum(gh * gh))
        if DO_CAUTIOUS:
            mn = mp * b1 + gh * (1.0 - b1)
            cm = tl.where(mn * gh > 0.0, 1.0, cautious_min)
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
        eff_lr_ptr,    # (N,) fp32 lr_base * auto_mult
        align_ptr,     # (N, 3) fp32 [dot, m2, g2] when DO_ALIGN
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
        M_MODE: tl.constexpr,  # 0=fp32, 1=bf16, 2=int8 quadratic-companded SR
        DO_CLIP: tl.constexpr,
        DO_CAUTIOUS: tl.constexpr,
        DO_ALIGN: tl.constexpr,
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
        m_t: tl.constexpr = tl.float32 if M_MODE == 0 else (tl.bfloat16 if M_MODE == 1 else tl.int8)

        p_base = tl.load(p_ptrs + tid).to(tl.pointer_type(p_t))
        g_base = tl.load(g_ptrs + tid).to(tl.pointer_type(g_t))
        m_base = tl.load(m_ptrs + tid).to(tl.pointer_type(m_t))

        p = tl.load(p_base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.where((g == g) & (tl.abs(g) < 3.4e38), g, 0.0)
        if M_MODE == 2:
            qf = tl.load(m_base + offs, mask=mask, other=0).to(tl.float32)
            t = qf * 0.007874015748031496
            mp = tl.where(qf < 0.0, -4.0, 4.0) * (t * t)
        else:
            mp = tl.load(m_base + offs, mask=mask, other=0.0).to(tl.float32)

        rs = tl.load(r_scale_ptr + tid * R + r_idx, mask=mask, other=0.0)
        cs = tl.load(c_scale_ptr + tid * C + c_idx, mask=mask, other=0.0)
        gh = g * (rs * cs)
        if DO_CLIP:
            gh = gh * tl.load(rms_scale_ptr + tid)
            gh = tl.maximum(tl.minimum(gh, 3.0 * clip), -3.0 * clip)
        if DO_ALIGN:
            tl.atomic_add(align_ptr + tid * 3 + 0, tl.sum(mp * gh))
            tl.atomic_add(align_ptr + tid * 3 + 1, tl.sum(mp * mp))
            tl.atomic_add(align_ptr + tid * 3 + 2, tl.sum(gh * gh))

        mn = mp * b1 + gh * (1.0 - b1)
        u = mn * inv_bc1

        # Store updated momentum in place
        if M_MODE == 1:
            rm = tl.randint(m_seed, tid * numel + offs).to(tl.int32, bitcast=True)
            mi = mn.to(tl.int32, bitcast=True)
            mi = (mi + (rm & 0xFFFF)) & (-65536)
            tl.store(m_base + offs, mi.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
        elif M_MODE == 2:
            ax = tl.minimum(tl.abs(mn), 4.0)
            k = tl.minimum(tl.floor(tl.sqrt(ax * 0.25) * 127.0), 126.0)
            lo_t = k * 0.007874015748031496
            lo = lo_t * lo_t * 4.0
            hi_t = (k + 1.0) * 0.007874015748031496
            hi = hi_t * hi_t * 4.0
            p_up = (ax - lo) / tl.maximum(hi - lo, 1e-12)
            rm = (tl.randint(m_seed, tid * numel + offs).to(tl.int32, bitcast=True) & 0xFFFFFF).to(tl.float32) * 5.960464477539063e-8
            q_abs = k + tl.where(rm < p_up, 1.0, 0.0)
            q = tl.where(mn < 0.0, -q_abs, q_abs).to(tl.int8)
            tl.store(m_base + offs, q, mask=mask)
        else:
            tl.store(m_base + offs, mn, mask=mask)

        if DO_CAUTIOUS:
            cm = tl.where(u * gh > 0.0, 1.0, cautious_min)
            u = u * cm * tl.load(caut_scale_ptr + tid)

        lr = tl.load(eff_lr_ptr + tid)
        if HAS_WD:
            u = u + wd * p
        new_p = p - lr * u

        if P_DTYPE == 0:
            tl.store(p_base + offs, new_p, mask=mask)
        elif P_DTYPE == 1:
            rp = tl.randint(p_seed, tid * numel + offs).to(tl.int32, bitcast=True)
            pi = new_p.to(tl.int32, bitcast=True)
            pi = (pi + (rp & 0xFFFF)) & (-65536)
            tl.store(p_base + offs, pi.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
        else:
            x = tl.maximum(tl.minimum(new_p, 65504.0), -65504.0)
            rp = tl.randint(p_seed, tid * numel + offs).to(tl.int32, bitcast=True)
            norm_i = (x.to(tl.int32, bitcast=True) + (rp & 0x1FFF)) & (-8192)
            norm_f = norm_i.to(tl.float32, bitcast=True)
            u = (rp & 0xFFFFFF).to(tl.float32) * 5.960464477539063e-8
            sub_f = tl.floor(x * 16777216.0 + u) * 5.960464477539063e-8
            out_f = tl.where(tl.abs(x) < 6.103515625e-5, sub_f, norm_f)
            tl.store(p_base + offs, out_f.to(tl.float16), mask=mask)

    _INPLACE_KERNEL = _lotus_fused_inplace_kernel
    return _INPLACE_KERNEL


def can_fused_inplace(ps: list[torch.Tensor], ms: list[torch.Tensor]) -> bool:
    """Whether this shape bucket can run the zero-copy pointer-array Triton fast path."""
    if not ps or not available():
        return False
    p0 = ps[0]
    if not (
        p0.is_cuda
        and p0.dim() == 2
        and p0.dtype in _DTYPE_CODE
        and p0.grad.dtype in _DTYPE_CODE
        and ms[0].dtype in _M_DTYPE_CODE
    ):
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
    m_mode = _M_DTYPE_CODE[ms[0].dtype]

    p_key = (dev, tuple(p.data_ptr() for p in ps), tuple(m.data_ptr() for m in ms))
    cached = _PTR_CACHE.get(p_key)
    if cached is None:
        if len(_PTR_CACHE) >= 256:
            _PTR_CACHE.clear()
        pm = torch.tensor([list(p_key[1]), list(p_key[2])], dtype=torch.int64, device=dev)
        cached = (pm[0], pm[1])
        _PTR_CACHE[p_key] = cached
    p_ptrs, m_ptrs = cached
    g_ptrs = torch.tensor([p.grad.data_ptr() for p in ps], dtype=torch.int64, device=dev)

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

    # 2. Optional RMS reduction, then clipped cautious/alignment stats (exact match with eager)
    do_clip = clip < 1e8
    do_caut = cautious_min < 1.0
    BLOCK = 1024
    grid = (n, triton.cdiv(numel, BLOCK))
    if do_clip:
        sq_sum = torch.zeros((n,), dtype=torch.float32, device=dev)
        _get_rms_kernel()[grid](
            g_ptrs, r_scale, c_scale, sq_sum, R, C, numel,
            G_DTYPE=g_code, BLOCK=BLOCK, num_warps=4,
        )
        rms_scale = (clip / (sq_sum / numel).sqrt().clamp_min(clip)).contiguous()
    else:
        rms_scale = r_scale

    msq = torch.zeros((n,), dtype=torch.float32, device=dev) if do_caut else r_scale
    align_buf = torch.zeros((n, 3), dtype=torch.float32, device=dev) if do_align else r_scale
    # When cautious gating is off (default), fold DO_ALIGN directly into the final in-place pass
    # instead of launching a separate _lotus_prestats_kernel over grad + momentum.
    if do_caut:
        _get_prestats_kernel()[grid](
            g_ptrs, m_ptrs, r_scale, c_scale, rms_scale, msq, align_buf,
            R, C, numel, float(b1), float(clip), float(cautious_min),
            G_DTYPE=g_code, M_MODE=m_mode,
            DO_CLIP=do_clip, DO_CAUTIOUS=True, DO_ALIGN=do_align,
            BLOCK=BLOCK, num_warps=4,
        )
    caut_scale = (msq / numel).clamp_min(0.1).rsqrt().contiguous() if do_caut else r_scale
    if lr_mult is not None:
        eff_lr = lr_mult.to(device=dev, dtype=torch.float32).mul(float(lr_base)).expand(n).contiguous()
    else:
        eff_lr = torch.full((n,), float(lr_base), dtype=torch.float32, device=dev)

    # 3. Single in-place pass: update momentum + parameter with Philox stochastic rounding
    m_seed = random.getrandbits(30) if m_mode != 0 else 0
    p_seed = random.getrandbits(30) if p_code != 0 else 0
    _get_inplace_kernel()[grid](
        p_ptrs, g_ptrs, m_ptrs, r_scale, c_scale, rms_scale, caut_scale, eff_lr, align_buf,
        R, C, numel, float(b1), 1.0 / bc1, float(clip), float(cautious_min), float(wd),
        m_seed, p_seed,
        P_DTYPE=p_code, G_DTYPE=g_code, M_MODE=m_mode,
        DO_CLIP=do_clip, DO_CAUTIOUS=do_caut, DO_ALIGN=(do_align and not do_caut),
        HAS_WD=(wd != 0.0),
        BLOCK=BLOCK, num_warps=4,
    )
    return align_buf if do_align else None
