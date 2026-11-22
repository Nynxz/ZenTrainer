"""Fused W8A16 Triton backward GEMM for int8 ConvRot weights (MiniMax H3, Krea 2 int8 builds).

Ported from Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0); the kernel, its
autotune configs and wrapper are by @mabseyuk (int64-offset variant), with Fizgig's change of
keying the autotune on the power-of-2 bucket of the token count. Changed here: this header, an
optional fp32 output (out_dtype) so the ConvRot rotation after it rounds to bf16 once, L2 launch
swizzling, even-dimension load paths, and AMD-safe tile configs. Licensed under the Apache License,
Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

What it computes: for a frozen linear layer whose weight is int8 with a scale per output row,
the input gradient (grad_out * row_scale) @ int8_weight, scaling in fp32 and casting the int8
tiles inside the kernel, so the full bf16 weight ComfyUI's backward dequantizes per layer per
step is never built. ConvRot's Hadamard rotation is applied after (harness/kernels/int8_backward.py).
Fizgig measured it within one bf16 ulp of the eager path, 1.2-1.4x per GEMM at clip-sized token
counts (4k+), about even below ~1.7k tokens; it always saves the transient bf16 weight.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    _NVIDIA_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
    ]
    _AMD_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
    ]

    @triton.autotune(
        configs=_AMD_CONFIGS if torch.version.hip else _NVIDIA_CONFIGS,
        # M_KEY, not M: keyed on the exact token count, a training run re-tunes for every
        # caption length (measured: 1,488 config benchmarks in one 46-step epoch, 160 more in
        # the next as caption dropout shifted lengths — 3x the epoch). Keyed on the power-of-2
        # bucket of M there are ~6 keys per shape and the tuning is done in the first steps.
        key=["M_KEY", "N", "K", "EVEN_N", "EVEN_K"],
    )
    @triton.jit
    def _w8a16_backward_kernel(
        G_ptr, W_ptr, S_ptr, O_ptr,
        M, N, K, M_KEY,
        stride_gm, stride_gn,
        stride_wn, stride_wk,
        stride_om, stride_ok,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, GROUP: tl.constexpr,
        EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_m, n_k = tl.cdiv(M, BLOCK_M), tl.cdiv(K, BLOCK_K)
        group = GROUP * n_k
        first_m = (pid // group) * GROUP
        rows = tl.minimum(n_m - first_m, GROUP)
        pid_m = first_m + (pid % group) % rows
        pid_k = (pid % group) // rows

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_k = offs_k < K
        offs_m_i64 = offs_m.to(tl.int64)          # #89 v2 hardening: M*N and M*K can exceed 2^31
        accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        offs_n0 = tl.arange(0, BLOCK_N)
        g_ptrs = G_ptr + offs_m_i64[:, None] * stride_gm + offs_n0[None, :] * stride_gn
        w_ptrs = W_ptr + offs_n0[:, None] * stride_wn + offs_k[None, :] * stride_wk
        s_ptrs = S_ptr + offs_n0

        # N is the forward layer's output dimension. Scaling occurs in fp32,
        # followed by the same BF16 rounding as the eager ``.to(ctx.dt)``.
        for n0 in range(0, N, BLOCK_N):
            if EVEN_N:
                grad = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)
                scale = tl.load(s_ptrs).to(tl.float32)
                if EVEN_K:
                    weight = tl.load(w_ptrs)
                else:
                    weight = tl.load(w_ptrs, mask=mask_k[None, :], other=0)
            else:
                offs_n = n0 + offs_n0
                mask_n = offs_n < N
                grad = tl.load(g_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
                scale = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.float32)
                weight = tl.load(w_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0)
            grad_scaled = (grad.to(tl.float32) * scale[None, :]).to(tl.bfloat16)
            accumulator += tl.dot(grad_scaled, weight.to(tl.bfloat16), out_dtype=tl.float32)
            g_ptrs += BLOCK_N * stride_gn
            w_ptrs += BLOCK_N * stride_wn
            s_ptrs += BLOCK_N

        out_ptrs = O_ptr + offs_m_i64[:, None] * stride_om + offs_k[None, :] * stride_ok
        if EVEN_K:
            tl.store(out_ptrs, accumulator.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])
        else:
            tl.store(out_ptrs, accumulator.to(O_ptr.dtype.element_ty),
                     mask=(mask_m[:, None] & mask_k[None, :]))


    def fused_w8a16_input_grad(grad_out, qdata, wscale, out_dtype=torch.bfloat16):
        """Return the input gradient for a frozen W8A16 linear layer (out_dtype: bf16, or fp32
        to round once later; a ZenTrainer addition)."""
        original_shape = grad_out.shape[:-1]
        grad2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        qdata = qdata.contiguous()
        scale = wscale.reshape(-1).contiguous()
        M, N = grad2d.shape
        weight_n, K = qdata.shape

        if not grad2d.is_cuda:
            raise RuntimeError("W8A16 backward kernel requires CUDA")
        if grad2d.dtype != torch.bfloat16:
            raise RuntimeError("W8A16 backward kernel expects BF16 gradients")
        if qdata.dtype != torch.int8:
            raise RuntimeError("W8A16 backward kernel expects INT8 qdata")
        if weight_n != N or scale.numel() != N:
            raise RuntimeError(
                f"W8A16 backward shape mismatch: grad={tuple(grad2d.shape)}, "
                f"weight={tuple(qdata.shape)}, scale={scale.numel()}")

        out = torch.empty((M, K), device=grad2d.device, dtype=out_dtype)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),)
        _w8a16_backward_kernel[grid](
            grad2d, qdata, scale, out,
            M, N, K, 1 << max(0, M - 1).bit_length(),      # M_KEY: the power-of-2 bucket of M
            grad2d.stride(0), grad2d.stride(1),
            qdata.stride(0), qdata.stride(1),
            out.stride(0), out.stride(1),
            GROUP=8, EVEN_N=(N % 64 == 0), EVEN_K=(K % 256 == 0),
        )
        return out.reshape(*original_shape, K)

else:

    def fused_w8a16_input_grad(grad_out, qdata, wscale):
        raise RuntimeError("Triton is required for fused W8A16 backward")
