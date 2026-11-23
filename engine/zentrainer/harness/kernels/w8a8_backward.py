"""Triton kernels for the W8A8 input gradient of a frozen int8 ConvRot linear (see
int8_backward.py): the gradient quantized to int8 per row, then multiplied by the int8 weight on
int8 tensor cores, with optional fused residual add and fused ConvRot Hadamard-256 + residual.

    quantize_rows:        gs = grad * s (s: the weight's per-output-row scale), each row's step =
                          max|gs| / 127, gq = floor(gs / step + U[0, 1)) (stochastic rounding: the
                          expectation is exact, so the error averages out over steps)
    int8_matmul:          (gq @ q) * step (+ residual when unrotated), int32 accumulation, bf16 out
    hadamard256_residual: ConvRot's H256 = H16 (x) H16 in fp32 + optional residual, bf16 out
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


_h16_cache: dict[torch.device, torch.Tensor] = {}


def _hadamard16(device: torch.device) -> torch.Tensor:
    """Normalised regular H16 = H4 (x) H4 / 4 in fp32 (symmetric: H16 == H16.T), cached per device."""
    h = _h16_cache.get(device)
    if h is None:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = (torch.kron(h4, h4) / 4.0).to(device=device).contiguous()
        _h16_cache[device] = h
    return h


if TRITON_AVAILABLE:

    # Tile configs, per vendor. NVIDIA keeps the set these kernels were measured with. The AMD set
    # was swept on an RX 7800 XT (gfx1101, RDNA3, ROCm 10.3 / Triton 3.8): 24 tiles x MOSS's four
    # ConvRot shapes (9216x3072, 3072x3072, 16384x3072, 3072x8192) at 2304, 4096 and 4352 tokens.
    # 128x128x64 w4 s3 won 9 of 12 shapes and 128x128x128 w8 s3 the other 3; the NVIDIA set's
    # 128x256x64 and 64x256x64 were 5-12% slower here. Kept generic rather than RDNA3-only: tiles
    # that fit RDNA3's 64 KB LDS also fit CDNA's, a config that doesn't fit is skipped by the
    # autotuner (OutOfResources, triton/runtime/autotuner.py), and 256x128x64 is the tile CDNA
    # usually prefers even though it never wins on RDNA3. Nothing here is a guess for gfx11; MI
    # cards are untested and should be swept the same way before their numbers are quoted.
    _NVIDIA_TILES = [
        (128, 128, 64, 8, 4), (128, 256, 64, 8, 3), (64, 256, 64, 4, 4),
        (128, 128, 128, 8, 3), (256, 128, 64, 8, 3), (64, 128, 128, 4, 4),
    ]
    _AMD_TILES = [
        (128, 128, 64, 4, 3), (128, 128, 128, 8, 3), (64, 128, 64, 4, 3), (256, 128, 64, 8, 3),
    ]

    def _tile_configs() -> list:
        """triton.Configs for this vendor. `torch.version.hip` is a constant of the build, so no
        device is touched at import (the engine imports before ComfyUI picks a device)."""
        tiles = _AMD_TILES if torch.version.hip else _NVIDIA_TILES
        return [triton.Config({"BM": bm, "BK": bk, "BN": bn}, num_stages=st, num_warps=w)
                for bm, bk, bn, w, st in tiles]

    @triton.jit
    def _quantize_rows_1pass_kernel(G, S, Q, R, N, stride_g, stride_q, SEED, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        seed = tl.load(SEED)  # read on the device: no host round trip, so torch.compile traces through
        g_row = G + row.to(tl.int64) * stride_g
        q_row = Q + row.to(tl.int64) * stride_q
        offs = tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(g_row + offs, mask=m, other=0.0).to(tl.float32) * tl.load(S + offs, mask=m, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        step = tl.maximum(amax, 1e-30) / 127.0
        inv = 1.0 / step
        noise = tl.rand(seed, row * N + offs)
        v = tl.minimum(tl.maximum(tl.floor(x * inv + noise), -127.0), 127.0)
        tl.store(q_row + offs, v.to(tl.int8), mask=m)
        tl.store(R + row, step)

    @triton.jit
    def _quantize_rows_kernel(G, S, Q, R, N, stride_g, stride_q, SEED, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        seed = tl.load(SEED)  # read on the device: no host round trip, so torch.compile traces through
        g_row = G + row.to(tl.int64) * stride_g
        q_row = Q + row.to(tl.int64) * stride_q
        amax = tl.zeros((), dtype=tl.float32)
        for n0 in range(0, N, BLOCK):
            offs = n0 + tl.arange(0, BLOCK)
            m = offs < N
            x = tl.load(g_row + offs, mask=m, other=0.0).to(tl.float32) * tl.load(S + offs, mask=m, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
        step = tl.maximum(amax, 1e-30) / 127.0
        inv = 1.0 / step
        for n0 in range(0, N, BLOCK):
            offs = n0 + tl.arange(0, BLOCK)
            m = offs < N
            x = tl.load(g_row + offs, mask=m, other=0.0).to(tl.float32) * tl.load(S + offs, mask=m, other=0.0).to(tl.float32)
            noise = tl.rand(seed, row * N + offs)
            v = tl.minimum(tl.maximum(tl.floor(x * inv + noise), -127.0), 127.0)
            tl.store(q_row + offs, v.to(tl.int8), mask=m)
        tl.store(R + row, step)

    @triton.autotune(
        configs=_tile_configs(),
        # the token count bucketed to its power of 2 (as Fizgig's W8A16 kernel): a run tunes once
        # per shape bucket, not once per caption length
        key=["M_KEY", "N", "K", "EVEN_N", "EVEN_K", "HAS_RES"],
    )
    @triton.jit
    def _int8_matmul_kernel(A, B, R, RES, O, M, N, K, M_KEY, s_am, s_an, s_bn, s_bk, s_rm, s_rk, s_om, s_ok,
                            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
                            EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, HAS_RES: tl.constexpr):
        # grouped launch order: neighbouring programs share rows of A in L2
        pid = tl.program_id(0)
        n_m, n_k = tl.cdiv(M, BM), tl.cdiv(K, BK)
        group = GROUP * n_k
        first_m = (pid // group) * GROUP
        rows = tl.minimum(n_m - first_m, GROUP)
        pid_m = first_m + (pid % group) % rows
        pid_k = (pid % group) // rows

        offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        offs_k = pid_k * BK + tl.arange(0, BK)
        mask_m, mask_k = offs_m < M, offs_k < K
        tl.multiple_of(offs_k, 16)
        acc = tl.zeros((BM, BK), dtype=tl.int32)
        a_ptrs = A + offs_m[:, None] * s_am + tl.arange(0, BN)[None, :] * s_an
        b_ptrs = B + tl.arange(0, BN)[:, None] * s_bn + offs_k[None, :] * s_bk
        for n0 in range(0, N, BN):
            if EVEN_N:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
                if EVEN_K:
                    b = tl.load(b_ptrs)
                else:
                    b = tl.load(b_ptrs, mask=mask_k[None, :], other=0)
            else:
                offs_n = n0 + tl.arange(0, BN)
                mask_n = offs_n < N
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0)
                b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0)
            acc += tl.dot(a, b, out_dtype=tl.int32)
            a_ptrs += BN * s_an
            b_ptrs += BN * s_bn
        step = tl.load(R + offs_m, mask=mask_m, other=0.0)
        out = acc.to(tl.float32) * step[:, None]
        if HAS_RES:
            if EVEN_K:
                res = tl.load(RES + offs_m[:, None] * s_rm + offs_k[None, :] * s_rk, mask=mask_m[:, None], other=0.0).to(tl.float32)
            else:
                res = tl.load(RES + offs_m[:, None] * s_rm + offs_k[None, :] * s_rk, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            out = out + res
        if EVEN_K:
            tl.store(O + offs_m[:, None] * s_om + offs_k[None, :] * s_ok, out.to(tl.bfloat16), mask=mask_m[:, None])
        else:
            tl.store(O + offs_m[:, None] * s_om + offs_k[None, :] * s_ok, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])

    @triton.jit
    def _hadamard256_res_kernel(X, H16, RES, OUT, N_GROUPS, HAS_RES: tl.constexpr, BG: tl.constexpr):
        # Fast ConvRot H256 = H16 (x) H16 in fp32 with optional residual add, rounded to bf16 once.
        # H16 is symmetric (H16 == H16^T): X (BG, 16, 16) -> X @ H16 -> (X @ H16)^T @ H16 -> transpose back.
        pid = tl.program_id(0)
        g0 = pid * BG
        i16 = tl.arange(0, 16)
        h16 = tl.load(H16 + i16[:, None] * 16 + i16[None, :]).to(tl.float32)
        r = tl.arange(0, BG * 16)
        gr = g0 + (r // 16)
        mr = gr < N_GROUPS
        offs = gr.to(tl.int64)[:, None] * 256 + (r % 16)[:, None] * 16 + i16[None, :]
        x = tl.load(X + offs, mask=mr[:, None], other=0.0).to(tl.float32)
        y = tl.dot(x, h16, allow_tf32=False)
        y = tl.trans(tl.reshape(y, (BG, 16, 16)), (0, 2, 1))
        y = tl.reshape(y, (BG * 16, 16))
        y = tl.dot(y, h16, allow_tf32=False)
        y = tl.trans(tl.reshape(y, (BG, 16, 16)), (0, 2, 1))
        y = tl.reshape(y, (BG * 16, 16))
        if HAS_RES:
            y = y + tl.load(RES + offs, mask=mr[:, None], other=0.0).to(tl.float32)
        tl.store(OUT + offs, y.to(tl.bfloat16), mask=mr[:, None])


def quantize_rows(grad: torch.Tensor, scale: torch.Tensor, seed: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """grad (M, N) any float, scale (N,): int8 (M, N) of grad * scale, and each row's step (M,).
    `seed`: the rounding noise's seed, an int or a one-element int32 tensor on grad's device (a
    tensor keeps it off the host, so a compiled caller needn't break its graph to read it)."""
    if grad.stride(1) != 1:
        grad = grad.contiguous()
    if not isinstance(seed, torch.Tensor):
        seed = torch.tensor([seed], device=grad.device, dtype=torch.int32)
    m, n = grad.shape
    q = torch.empty((m, n), device=grad.device, dtype=torch.int8)
    steps = torch.empty((m,), device=grad.device, dtype=torch.float32)
    sc = scale.reshape(-1).contiguous()
    if n <= 2048:
        _quantize_rows_1pass_kernel[(m,)](grad, sc, q, steps, n, grad.stride(0), q.stride(0), seed,
                                          BLOCK=triton.next_power_of_2(max(16, n)), num_warps=4)
    elif n >= 8192:
        _quantize_rows_kernel[(m,)](grad, sc, q, steps, n, grad.stride(0), q.stride(0), seed,
                                    BLOCK=2048, num_warps=8)
    else:
        _quantize_rows_kernel[(m,)](grad, sc, q, steps, n, grad.stride(0), q.stride(0), seed,
                                    BLOCK=1024, num_warps=4)
    return q, steps


def int8_matmul(a: torch.Tensor, b: torch.Tensor, steps: torch.Tensor,
                residual: torch.Tensor | None = None) -> torch.Tensor:
    """(a @ b) * steps[:, None] (+ residual) as bf16: a (M, N) int8, b (N, K) int8 in any layout."""
    m, n = a.shape
    k = b.shape[1]
    out = torch.empty((m, k), device=a.device, dtype=torch.bfloat16)
    has_res = residual is not None
    res = residual.reshape(m, k) if has_res else out
    if has_res and res.stride(1) != 1:
        res = res.contiguous()
    grid = lambda META: (triton.cdiv(m, META["BM"]) * triton.cdiv(k, META["BK"]),)
    _int8_matmul_kernel[grid](a, b, steps, res, out, m, n, k, 1 << max(0, m - 1).bit_length(),
                              a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                              res.stride(0), res.stride(1), out.stride(0), out.stride(1),
                              GROUP=8, EVEN_N=(n % 128 == 0), EVEN_K=(k % 256 == 0), HAS_RES=has_res)
    return out


def hadamard256_residual(x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
    """ConvRot's 256-wide Hadamard rotation on `x` (M, K) + optional `residual` (M, K) in one fused
    fp32-internal pass, written as bf16."""
    m, k = x.shape
    x = x.contiguous()
    has_res = residual is not None
    res = residual.reshape(m, k).contiguous() if has_res else x
    out = torch.empty((m, k), device=x.device, dtype=torch.bfloat16)
    n_groups = m * (k // 256)
    _hadamard256_res_kernel[(triton.cdiv(n_groups, 4),)](x, _hadamard16(x.device), res, out, n_groups,
                                                         HAS_RES=has_res, BG=4, num_warps=4)
    return out
