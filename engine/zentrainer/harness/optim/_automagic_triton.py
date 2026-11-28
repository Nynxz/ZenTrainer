"""Fused Triton tail kernel for Automagic3Batched and Automagic4Batched.

Replaces ~15-20 eager PyTorch ops per shape stack (sign bit packing, history stack roll,
8x-larger (n, H, numel) uint8 bit unpack, sum/XOR/mask reductions, decoupled weight decay,
parameter update, and bf16/fp16 stochastic rounding) with a single kernel pass over the stacked
update and parameter tensors. Works on both CUDA and AMD ROCm via Triton.
"""

from __future__ import annotations

import torch

_AVAILABLE: bool | None = None
_KERNEL = None


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import triton  # noqa: F401

            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=("idx", "new_idx", "H", "seed"))
    def _automagic_tail_kernel(
        u_ptr,          # (N, numel) fp32 update used for sign history & vote (contiguous)
        u_step_ptr,     # (N, numel) fp32 update applied to weights (aliases u_ptr when SEPARATE_STEP_U=False)
        p_ptr,          # (N, numel) fp32/bf16/fp16 params (contiguous, updated in place)
        lr_ptr,         # (N,) fp32 learning rates
        hist_ptr,       # (N, H, width) uint8 sign history (unused when DO_VOTE=False)
        packed_ptr,     # (N, width) uint8 newly packed sign plane
        vote_ptr,       # (2,) fp32 [num_or_wflips, den] atomically accumulated
        wd,             # float32 weight decay
        numel,          # int32 elements per param tensor
        width,          # int32 bytes per sign plane = ceil(numel / 8)
        H,              # int32 history window length
        idx,            # int32 slot overwritten this step
        new_idx,        # int32 oldest slot after overwrite = (idx + 1) % H
        seed,           # int32 Philox seed for bf16/fp16 stochastic rounding
        DO_VOTE: tl.constexpr,
        IS_V4: tl.constexpr,
        HAS_WD: tl.constexpr,
        SEPARATE_STEP_U: tl.constexpr,
        OUT_DTYPE: tl.constexpr,  # 0=fp32, 1=bf16 (drop 16), 2=fp16 (drop 13)
        BLOCK_B: tl.constexpr,    # bytes per program (BLOCK_B * 8 elements)
    ):
        row = tl.program_id(0)
        pid_b = tl.program_id(1)
        b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < width
        shifts = tl.arange(0, 8).to(tl.uint8)
        e_offs = b_offs[:, None] * 8 + tl.arange(0, 8)[None, :]
        e_mask = e_offs < numel

        base_e = row * numel + e_offs
        u = tl.load(u_ptr + base_e, mask=e_mask, other=0.0).to(tl.float32)

        # Pack sign bits (u > 0.0) into uint8 bytes: bit k = element 8*b + k
        cur_bits = (u > 0.0).to(tl.uint8)
        cur_byte = tl.sum(cur_bits << shifts[None, :], axis=1).to(tl.uint8)
        tl.store(packed_ptr + row * width + b_offs, cur_byte, mask=b_mask)

        if DO_VOTE:
            au = tl.abs(u)  # 0.0 where e_offs >= numel, so padding bits never vote
            hist_row = hist_ptr + row * H * width
            if IS_V4:
                flips = tl.zeros([BLOCK_B, 8], dtype=tl.int32)
                prev_b = tl.zeros([BLOCK_B], dtype=tl.uint8)
                for t in range(H):
                    slot = (new_idx + t) % H
                    if slot == idx:
                        b = cur_byte
                    else:
                        b = tl.load(hist_row + slot * width + b_offs, mask=b_mask, other=0)
                    if t > 0:
                        diff = b ^ prev_b
                        flips += ((diff[:, None] >> shifts[None, :]) & 1).to(tl.int32)
                    prev_b = b
                b_wflips = tl.sum(au * flips.to(tl.float32))
                b_den = tl.sum(au)
                tl.atomic_add(vote_ptr + 0, b_wflips)
                tl.atomic_add(vote_ptr + 1, b_den)
            else:
                all_one = tl.full([BLOCK_B], 255, dtype=tl.uint8)
                all_zero = tl.full([BLOCK_B], 255, dtype=tl.uint8)
                all_flip = tl.full([BLOCK_B], 255, dtype=tl.uint8)
                prev_b = tl.zeros([BLOCK_B], dtype=tl.uint8)
                for t in range(H):
                    slot = (new_idx + t) % H
                    if slot == idx:
                        b = cur_byte
                    else:
                        b = tl.load(hist_row + slot * width + b_offs, mask=b_mask, other=0)
                    all_one = all_one & b
                    all_zero = all_zero & (b ^ 255)
                    if t > 0:
                        all_flip = all_flip & (b ^ prev_b)
                    prev_b = b
                up_byte = all_one | all_zero
                up = ((up_byte[:, None] >> shifts[None, :]) & 1).to(tl.float32)
                down = ((all_flip[:, None] >> shifts[None, :]) & 1).to(tl.float32)
                b_num = tl.sum(au * (up - down))
                b_den = tl.sum(au)
                tl.atomic_add(vote_ptr + 0, b_num)
                tl.atomic_add(vote_ptr + 1, b_den)

        # Decoupled weight decay + lr step + optional stochastic rounding in the same pass
        p_val = tl.load(p_ptr + base_e, mask=e_mask, other=0.0).to(tl.float32)
        lr_val = tl.load(lr_ptr + row).to(tl.float32)
        u_step = tl.load(u_step_ptr + base_e, mask=e_mask, other=0.0).to(tl.float32) if SEPARATE_STEP_U else u
        if HAS_WD:
            u_step = u_step + wd * p_val
        new_p = p_val - lr_val * u_step

        if OUT_DTYPE == 0:
            tl.store(p_ptr + base_e, new_p, mask=e_mask)
        else:
            drop_bits: tl.constexpr = 16 if OUT_DTYPE == 1 else 13
            low_mask: tl.constexpr = (1 << drop_bits) - 1
            hi_mask: tl.constexpr = -(1 << drop_bits)
            r = tl.randint(seed, base_e).to(tl.int32, bitcast=True)
            as_i = new_p.to(tl.int32, bitcast=True)
            as_i = (as_i + (r & low_mask)) & hi_mask
            rounded = as_i.to(tl.float32, bitcast=True)
            out_t: tl.constexpr = tl.bfloat16 if OUT_DTYPE == 1 else tl.float16
            tl.store(p_ptr + base_e, rounded.to(out_t), mask=e_mask)

    _KERNEL = _automagic_tail_kernel
    return _KERNEL


_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def supported(update: torch.Tensor, param_dtype: torch.dtype) -> bool:
    return update.is_cuda and param_dtype in _DTYPE_CODE and available()


def tail_step(
    update: torch.Tensor,
    ps: list[torch.Tensor],
    hists: list[torch.Tensor],
    lrs: torch.Tensor,
    wd: float,
    idx: int,
    new_idx: int,
    do_vote: bool,
    *,
    is_v4: bool = False,
    step_update: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run sign-history pack, optional vote, WD, param update, and stochastic rounding in one kernel.

    When `step_update` is provided, sign-history packing and voting still read `update` while the
    parameter step applies `step_update` (e.g. polar-orthogonalized LoRA updates).

    Returns (vote_primary, den) where vote_primary is `num` for v3 and `weighted_flips` for v4,
    or (None, None) when `do_vote` is False. Updates `hists[i][idx]` and `ps[i]` in place.
    """
    import triton

    n = len(ps)
    numel = ps[0].numel()
    H, width = hists[0].shape
    dev = update.device
    u2 = update.reshape(n, numel).contiguous()
    separate = step_update is not None
    us2 = step_update.reshape(n, numel).contiguous() if separate else u2
    p_stack = torch.stack(ps).reshape(n, numel).contiguous()
    lr1 = lrs.reshape(n).to(torch.float32).contiguous()
    packed = torch.empty((n, width), dtype=torch.uint8, device=dev)
    hist_stack = torch.stack(hists).contiguous() if do_vote else packed
    vote = torch.zeros((2,), dtype=torch.float32, device=dev) if do_vote else lr1
    out_dtype = _DTYPE_CODE[ps[0].dtype]
    seed = int(torch.randint(0, 1 << 30, (), device="cpu").item()) if out_dtype != 0 else 0

    BLOCK_B = 128
    grid = (n, triton.cdiv(width, BLOCK_B))
    _get_kernel()[grid](
        u2, us2, p_stack, lr1, hist_stack, packed, vote,
        float(wd), numel, width, H, idx, new_idx, seed,
        DO_VOTE=do_vote,
        IS_V4=is_v4,
        HAS_WD=(wd != 0.0),
        SEPARATE_STEP_U=separate,
        OUT_DTYPE=out_dtype,
        BLOCK_B=BLOCK_B,
        num_warps=4,
    )
    torch._foreach_copy_([h[idx] for h in hists], list(packed.unbind(0)))
    torch._foreach_copy_(ps, list(p_stack.view((n,) + tuple(ps[0].shape)).unbind(0)))
    if not do_vote:
        return None, None
    return vote[0], vote[1]
