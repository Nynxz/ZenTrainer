"""Per-head QK RMSNorm and RoPE as one op that trains (memory.fused_qk_rope): Krea 2's, MOSS's and
MiniMax H3's attention run their q/k norms and then the rope as separate ops while training
(comfy_kitchen's fused rms_rope has no backward), each a pass over q and k plus casts to fp32 and back.

    forward:   y = rope(n * w),  n = x * r,  r = rsqrt(mean(x²) + eps)
               comfy_kitchen's rms_rope / rms_rope_split_half: fp32 inside, rounded once
    backward:  g_z = Mᵀ g_y      each pair rotated back (M: the rope's 2x2 per pair, from freqs)
               g_n = g_z * w
               g_x = r * (g_n - n * mean(g_n * n))
               one Triton kernel, a program per row; plain torch without Triton or off CUDA

Tensors come in the model's own layout: (B, H, L, D) with freqs (B or 1, 1, L, D/2, 2, 2) (Krea 2),
or seq_first (B, L, H, D) with freqs (B or 1, L, 1, D/2, 2, 2) (MOSS); any strides. A pair is
elements 2i and 2i+1 (interleaved, Krea 2) or i and i + D/2 (split_half, MOSS). H3 rotates only
the first `rot` dims of each head (split-half pairs i and i + rot/2, freqs (1, L, 1, rot/2, 2, 2));
the rest are normalized and passed through; rot 0 = all of D. w is the norm's
weight (D,) in fp32 (Krea 2's 1 + scale: in bf16 it would round to steps of 1/128). The weights and
freqs get no gradient: callers use the separate ops when a norm's weight trains. Registered as
custom ops, so torch.compile keeps the fused kernels whole instead of tracing into them.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


def _rotated(t: torch.Tensor, rot: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The dims the rope turns, and the rest (None when it turns them all)."""
    return (t[..., :rot], t[..., rot:]) if rot and rot < t.shape[-1] else (t, None)


def _join(a: torch.Tensor, rest: torch.Tensor | None) -> torch.Tensor:
    return a if rest is None else torch.cat((a, rest), -1)


def _pairs(t: torch.Tensor, split_half: bool) -> tuple[torch.Tensor, torch.Tensor]:
    d = t.shape[-1] // 2
    return (t[..., :d], t[..., d:]) if split_half else (t[..., 0::2], t[..., 1::2])


def _unpairs(a: torch.Tensor, b: torch.Tensor, split_half: bool) -> torch.Tensor:
    return torch.cat((a, b), -1) if split_half else torch.stack((a, b), -1).flatten(-2)


def _freqs(freqs: torch.Tensor, x: torch.Tensor, seq_first: bool, dt: torch.dtype) -> torch.Tensor:
    """freqs cut to x's length (as ComfyUI's rope does), in dt."""
    seq = 1 if seq_first else 2
    if freqs.shape[seq] != x.shape[seq]:
        freqs = freqs.narrow(seq, 0, x.shape[seq])
    return freqs.to(dt)


def reference(x: torch.Tensor, w: torch.Tensor, freqs: torch.Tensor, eps: float, split_half: bool = False,
              seq_first: bool = False, rot: int = 0) -> torch.Tensor:
    """The maths in plain torch at x's precision (at least fp32): what the op computes."""
    dt = torch.promote_types(x.dtype, torch.float32)
    n, rest = _rotated(F.rms_norm(x.to(dt), (x.shape[-1],), weight=w.to(dt), eps=eps), rot)
    n0, n1 = _pairs(n, split_half)
    f = _freqs(freqs, x, seq_first, dt)  # f[..., i, j] = M[i][j]
    y = _unpairs(f[..., 0, 0] * n0 + f[..., 0, 1] * n1, f[..., 1, 0] * n0 + f[..., 1, 1] * n1, split_half)
    return _join(y, rest).to(x.dtype)


def _backward_torch(g, x, w, freqs, eps, split_half, seq_first, rot=0):
    dt = torch.promote_types(x.dtype, torch.float32)
    f = _freqs(freqs, x, seq_first, dt)
    g, rest = _rotated(g.to(dt), rot)
    g0, g1 = _pairs(g, split_half)
    gz = _join(_unpairs(f[..., 0, 0] * g0 + f[..., 1, 0] * g1, f[..., 0, 1] * g0 + f[..., 1, 1] * g1, split_half), rest)
    gn = gz * w.to(dt)
    xf = x.to(dt)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    n = xf * r
    return (r * (gn - n * (gn * n).mean(-1, keepdim=True))).to(x.dtype)


if TRITON_AVAILABLE:

    @triton.jit
    def _backward_kernel(G, X, W, FR, OUT, H, L, sgb, sgh, sgl, sxb, sxh, sxl, sob, soh, sol, sfb, eps,
                         D: tl.constexpr, ROT: tl.constexpr, HP: tl.constexpr, PP: tl.constexpr,
                         SPLIT: tl.constexpr, BH: tl.constexpr):
        # ROT: the dims the rope turns (D for all), as ROT/2 pairs padded to HP; the D - ROT after
        # them (padded to PP, 0 when none) are only normalized. One program handles BH heads at a
        # single (b, t), reusing the loaded freqs[b, t] and W across all BH heads instead of reading
        # them H times per token.
        pid_bt = tl.program_id(0)
        pid_hg = tl.program_id(1)
        b = (pid_bt // L).to(tl.int64)
        t = (pid_bt % L).to(tl.int64)
        k = tl.arange(0, HP)
        km = k < ROT // 2
        if SPLIT:  # pair k: elements k and k + ROT/2
            e, o = k, k + ROT // 2
        else:  # pair k: elements 2k and 2k + 1
            e, o = 2 * k, 2 * k + 1
        f = FR + b * sfb + t * (ROT * 2) + k * 4
        f00 = tl.load(f, mask=km, other=0.0)
        f01 = tl.load(f + 1, mask=km, other=0.0)
        f10 = tl.load(f + 2, mask=km, other=0.0)
        f11 = tl.load(f + 3, mask=km, other=0.0)
        we = tl.load(W + e, mask=km, other=0.0)
        wo = tl.load(W + o, mask=km, other=0.0)
        if PP > 0:
            j = ROT + tl.arange(0, PP)
            jm = j < D
            wp = tl.load(W + j, mask=jm, other=0.0)
        h0 = pid_hg * BH
        for dh in range(BH):
            hi = h0 + dh
            if hi < H:
                h = hi.to(tl.int64)
                g = G + b * sgb + h * sgh + t * sgl
                x = X + b * sxb + h * sxh + t * sxl
                out = OUT + b * sob + h * soh + t * sol
                xe = tl.load(x + e, mask=km, other=0.0).to(tl.float32)
                xo = tl.load(x + o, mask=km, other=0.0).to(tl.float32)
                ge = tl.load(g + e, mask=km, other=0.0).to(tl.float32)
                go = tl.load(g + o, mask=km, other=0.0).to(tl.float32)
                sq = tl.sum(xe * xe, 0) + tl.sum(xo * xo, 0)
                if PP > 0:
                    xp = tl.load(x + j, mask=jm, other=0.0).to(tl.float32)
                    gp = tl.load(g + j, mask=jm, other=0.0).to(tl.float32)
                    sq += tl.sum(xp * xp, 0)
                r = tl.rsqrt(sq / D + eps)
                ne = xe * r
                no = xo * r
                gne = (f00 * ge + f10 * go) * we
                gno = (f01 * ge + f11 * go) * wo
                dot = tl.sum(gne * ne, 0) + tl.sum(gno * no, 0)
                if PP > 0:
                    npp = xp * r
                    gnp = gp * wp
                    dot += tl.sum(gnp * npp, 0)
                dot = dot / D
                tl.store(out + e, (r * (gne - ne * dot)).to(OUT.dtype.element_ty), mask=km)
                tl.store(out + o, (r * (gno - no * dot)).to(OUT.dtype.element_ty), mask=km)
                if PP > 0:
                    tl.store(out + j, (r * (gnp - npp * dot)).to(OUT.dtype.element_ty), mask=jm)


_triton_failed = False  # once a launch fails (an unusual GPU or Triton build): plain torch from then on


def _prep_freqs(freqs: torch.Tensor, x: torch.Tensor, seq_first: bool) -> torch.Tensor:
    f = _freqs(freqs, x, seq_first, torch.float32)
    return (f[:, :, 0] if seq_first else f[:, 0]).contiguous()  # (B or 1, L, rot/2, 2, 2)


def _backward(g, x, w, freqs, eps, split_half, seq_first, rot=0, _prepped_freqs=None):
    global _triton_failed
    D = x.shape[-1]
    rot = rot if rot and rot < D else D
    if _triton_failed or not (TRITON_AVAILABLE and x.is_cuda and D in (64, 128, 256) and rot % 2 == 0
                              and x.dtype in (torch.bfloat16, torch.float16, torch.float32)):
        return _backward_torch(g, x, w, freqs, eps, split_half, seq_first, rot)
    try:
        return _backward_triton(g, x, w, freqs, eps, split_half, seq_first, rot, _prepped_freqs)
    except Exception as e:  # the torch backward is the same maths
        _triton_failed = True
        logging.warning("[ZenTrainer] fused q/k rope: the Triton backward failed (%s); plain torch from here", e)
        return _backward_torch(g, x, w, freqs, eps, split_half, seq_first, rot)


def _backward_triton(g, x, w, freqs, eps, split_half, seq_first, rot, prepped_freqs=None):
    D = x.shape[-1]
    hd, sd = (2, 1) if seq_first else (1, 2)  # which dims hold the heads and the sequence
    B, H, L = x.shape[0], x.shape[hd], x.shape[sd]
    f = prepped_freqs if prepped_freqs is not None else _prep_freqs(freqs, x, seq_first)
    g, x = (t if t.stride(-1) == 1 else t.contiguous() for t in (g, x))
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    bh = 8 if H >= 8 else (4 if H >= 4 else (2 if H >= 2 else 1))
    _backward_kernel[(B * L, triton.cdiv(H, bh))](
        g, x, w.float().contiguous(), f, out, H, L,
        g.stride(0), g.stride(hd), g.stride(sd), x.stride(0), x.stride(hd), x.stride(sd),
        out.stride(0), out.stride(hd), out.stride(sd), f.stride(0) if f.shape[0] > 1 else 0,
        eps, D=D, ROT=rot, HP=triton.next_power_of_2(rot // 2),
        PP=triton.next_power_of_2(D - rot) if rot < D else 0, SPLIT=split_half, BH=bh,
        num_warps=1 if D <= 64 else 2)
    return out


@torch.library.custom_op("zentrainer::qk_norm_rope", mutates_args=())
def qk_norm_rope(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, freqs: torch.Tensor,
                 eps: float, split_half: bool = False, seq_first: bool = False,
                 rot: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    partial = bool(rot) and rot < q.shape[-1]
    if q.is_cuda and q.dtype != torch.float64 and (split_half or not partial):
        try:
            import comfy.quant_ops

            ck = comfy.quant_ops.ck
            if partial:  # H3's: kitchen fuses a partial split-half rope too
                oq, ok = ck.rms_rope_split_half(q, k, freqs, wq.float(), wk.float(), epsilon=eps, rot_dim=rot)
            else:
                oq, ok = (ck.rms_rope_split_half if split_half else ck.rms_rope)(q, k, freqs, wq.float(), wk.float(), eps)
            return oq.contiguous(), ok.contiguous()
        except (ImportError, AttributeError, TypeError):  # TypeError: a kitchen without rot_dim
            pass
    return (reference(q, wq, freqs, eps, split_half, seq_first, rot).contiguous(),
            reference(k, wk, freqs, eps, split_half, seq_first, rot).contiguous())


@qk_norm_rope.register_fake
def _(q, k, wq, wk, freqs, eps, split_half=False, seq_first=False, rot=0):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("zentrainer::qk_norm_rope_backward", mutates_args=())
def qk_norm_rope_backward(gq: torch.Tensor, gk: torch.Tensor, q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor,
                          wk: torch.Tensor, freqs: torch.Tensor, eps: float, split_half: bool,
                          seq_first: bool, rot: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    # prepare freqs once for both q and k when Triton runs (avoids a duplicate cast/copy per layer)
    pf = (_prep_freqs(freqs, q, seq_first) if (TRITON_AVAILABLE and not _triton_failed and q.is_cuda
                                               and q.shape[-1] in (64, 128, 256)) else None)
    return (_backward(gq, q, wq, freqs, eps, split_half, seq_first, rot, pf),
            _backward(gk, k, wk, freqs, eps, split_half, seq_first, rot, pf))


@qk_norm_rope_backward.register_fake
def _(gq, gk, q, k, wq, wk, freqs, eps, split_half, seq_first, rot=0):
    return q.new_empty(q.shape), k.new_empty(k.shape)


def _setup(ctx, inputs, output):
    q, k, wq, wk, freqs, eps, split_half, seq_first, rot = inputs
    ctx.save_for_backward(q, k, wq, wk, freqs)
    ctx.eps, ctx.split_half, ctx.seq_first, ctx.rot = eps, split_half, seq_first, rot


def _grad(ctx, gq, gk):
    q, k, wq, wk, freqs = ctx.saved_tensors
    dq, dk = qk_norm_rope_backward(gq, gk, q, k, wq, wk, freqs, ctx.eps, ctx.split_half, ctx.seq_first, ctx.rot)
    return dq, dk, None, None, None, None, None, None, None


qk_norm_rope.register_autograd(_grad, setup_context=_setup)
