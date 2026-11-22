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
