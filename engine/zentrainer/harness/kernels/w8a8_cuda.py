"""The W8A8 input gradient of a frozen int8 ConvRot linear in raw CUDA (w8a8_cuda.cu): the gradient
quantized to int8 per row (stochastically rounded), multiplied by the int8 weight as it's stored
on int8 tensor cores, scaled, and ConvRot's Hadamard rotation applied, all in two kernels. The
Triton path (w8a8_backward.py) takes three: its matmul, then the rotation as a bf16 matmul, which
also rounds twice.

Compiled at first use with NVRTC, which PyTorch ships (torch.cuda._compile_kernel's machinery),
so nothing needs building or a CUDA toolkit; about 1.5 s once per process. Needs an NVIDIA card
of compute capability 8.0 or newer. Anything else, any shape it doesn't take, a failure to
compile, or ZENTRAINER_CUDA_KERNELS=0 in the environment (to compare): the caller uses the Triton
path. A custom op, so torch.compile doesn't trace into the launch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

SMEM = 3 * (128 * 64 + 64 * 256)  # BM=128 3-stage pipeline: 72 KB (works on all sm_80+)
SMEM4 = 4 * (128 * 64 + 64 * 256)  # BM=128 4-stage pipeline: 96 KB (sm_80/86/89/90/120 with >= 96 KB opt-in)
SMEM64 = 2 * (64 * 64 + 64 * 256)  # BM=64 2-stage pipeline: 40 KB (2 CTAs/SM)
_NAMES = (
    "zt_quantize_rows", "zt_quantize_rows_2k", "zt_quantize_rows_4k", "zt_quantize_rows_8k", "zt_quantize_rows_16k",
    "zt_w8a8_gemm_rot", "zt_w8a8_gemm", "zt_w8a8_gemm_s4_rot", "zt_w8a8_gemm_s4",
    "zt_w8a8_gemm_sk_rot", "zt_w8a8_gemm_sk", "zt_w8a8_gemm_s4_sk_rot", "zt_w8a8_gemm_s4_sk",
    "zt_w8a8_gemm64_rot", "zt_w8a8_gemm64",
)
_kernels: dict[int, dict | None] = {}  # by device: its kernels, or None if they couldn't load
_sms: dict[int, int] = {}  # SM count per device, to schedule Stream-K and BM=64
_has_s4: dict[int, bool] = {}  # whether device opt-in shared memory fits SMEM4 (96 KB)
_sk_done: dict[tuple[int, int], torch.Tensor] = {}  # per-(device, stream) zeroed arrival counters (reset in-kernel)


def _load(device: torch.device) -> dict | None:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _kernels:
        return _kernels[idx]
    _kernels[idx] = None
    if os.environ.get("ZENTRAINER_CUDA_KERNELS", "1") == "0":
        logging.info("[ZenTrainer] int8 backward: CUDA kernels off (ZENTRAINER_CUDA_KERNELS=0)")
        return None
    try:
        if torch.version.hip or torch.cuda.get_device_capability(idx) < (8, 0):
            return None
        from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

        src = Path(__file__).with_name("w8a8_cuda.cu").read_text()
        major, minor = torch.cuda.get_device_capability(idx)
        props = torch.cuda.get_device_properties(idx)
        optin = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
        can_s4 = optin >= SMEM4 + 256  # leaves headroom for static __shared__ (s_arrived) in Stream-K kernels
        with torch.cuda.device(idx):
            ptx, _ = _nvrtc_compile(src, _NAMES[0], f"{major}{minor}", None, ["-std=c++17"])
            ks = _cuda_load_module(ptx, list(_NAMES))
            for n in ("zt_w8a8_gemm_rot", "zt_w8a8_gemm", "zt_w8a8_gemm_sk_rot", "zt_w8a8_gemm_sk"):
                ks[n].set_shared_memory_config(SMEM)
            if can_s4:
                for n in ("zt_w8a8_gemm_s4_rot", "zt_w8a8_gemm_s4", "zt_w8a8_gemm_s4_sk_rot", "zt_w8a8_gemm_s4_sk"):
                    ks[n].set_shared_memory_config(SMEM4)
            for n in ("zt_w8a8_gemm64_rot", "zt_w8a8_gemm64"):
                ks[n].set_shared_memory_config(SMEM64)
        _sms[idx] = props.multi_processor_count
        _has_s4[idx] = can_s4
        _kernels[idx] = ks
        logging.info("[ZenTrainer] int8 backward: CUDA kernels compiled for sm_%d%d (%d SMs, 4-stage=%s)",
                     major, minor, _sms[idx], "on" if can_s4 else "off")
    except Exception:
        logging.warning("[ZenTrainer] int8 backward's CUDA kernels didn't load; using Triton's", exc_info=True)
    return _kernels[idx]


def _done_buf(device: torch.device, idx: int, stream_id: int, need: int) -> torch.Tensor:
    """Per-stream zero-initialized Stream-K arrival counters (reset back to 0 in-kernel by each last finisher)."""
    key = (idx, stream_id)
    buf = _sk_done.get(key)
    if buf is None or buf.shape[0] < need:
        buf = torch.zeros(max(need, 512), device=device, dtype=torch.int32)
        _sk_done[key] = buf
    return buf


def supported(grad_2d: torch.Tensor, q: torch.Tensor, group: int) -> bool:
    """Whether these kernels take this gradient (M, out) and int8 weight (out, in). Compiled
    blocks trace the backward with fake tensors, which have no address (a failed data_ptr() check
    put the dequantized bf16 path into the compiled graph) and whose storage offset dynamo can't
    trace: there the alignment is left to the op, which falls back when a real tensor isn't."""
    m, n = grad_2d.shape
    k = q.shape[1]
    return (grad_2d.is_cuda and grad_2d.dtype == torch.bfloat16 and q.dtype == torch.int8 and q.is_contiguous()
            and grad_2d.stride(1) == 1 and grad_2d.stride(0) % 8 == 0
            and (torch.compiler.is_compiling() or _aligned(grad_2d, q))
            and m > 0 and n % 64 == 0 and k % 256 == 0 and group in (0, 256)
            and (_loaded(grad_2d.device) if torch.compiler.is_compiling() else _load(grad_2d.device) is not None))


def _loaded(device: torch.device) -> bool:
    """Read-only, for a trace (it can't write the cache): load() ran when the int8 backward was
    enabled (int8_backward.enable)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _kernels.get(idx) is not None


def _aligned(grad_2d: torch.Tensor, q: torch.Tensor) -> bool:
    return grad_2d.data_ptr() % 16 == 0 and q.data_ptr() % 16 == 0


@torch.library.custom_op("zentrainer::w8a8_input_grad", mutates_args=())
def input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, seed: torch.Tensor,
               rotate: bool, residual: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 (M, in): (quantized grad_2d * scale) @ q, rescaled, rotated by ConvRot's 256-wide
    Hadamard when `rotate`, plus `residual` (M, in) when given. grad_2d (M, out) bf16, q (out, in)
    int8, scale (out,) the weight's per row, seed a one-element int32 tensor on the device.
    See supported()."""
    ks = _load(grad_2d.device)
    m, n = grad_2d.shape
    k = q.shape[1]
    if ks is None or not _aligned(grad_2d, q):  # decided while tracing: the real tensors don't suit
        from .int8_backward import _w8a8_input_grad

        g = _w8a8_input_grad(grad_2d, q, scale, 256 if rotate else 0)
        return g if residual is None else g + residual.to(g.dtype)
    idx = grad_2d.device.index if grad_2d.device.index is not None else torch.cuda.current_device()
    if scale.dtype != torch.float32 or not scale.is_contiguous():
        scale = scale.reshape(-1).float().contiguous()
    stream = torch.cuda.current_stream(grad_2d.device)
    stream_id = stream.cuda_stream
    gq = torch.empty((m, n), device=grad_2d.device, dtype=torch.int8)
    steps = torch.empty((m,), device=grad_2d.device, dtype=torch.float32)
    out = torch.empty((m, k), device=grad_2d.device, dtype=torch.bfloat16)
    res = residual is not None
    if res:
        if residual.dtype != torch.bfloat16 or not residual.is_contiguous():
            residual = residual.to(torch.bfloat16).reshape(m, k).contiguous()
        if residual.data_ptr() % 16 != 0:
            residual = residual.clone()  # uint2/uint4 vector loads require 16-byte alignment
    if n <= 2048:
        qname, qblock = "zt_quantize_rows_2k", 128
    elif n <= 4096:
        qname, qblock = "zt_quantize_rows_4k", 256
    elif n <= 8192:
        qname, qblock = "zt_quantize_rows_8k", 256
    elif n <= 16384:
        qname, qblock = "zt_quantize_rows_16k", 512
    else:
        qname, qblock = "zt_quantize_rows", 256
    ks[qname](grid=(m, 1, 1), block=(qblock, 1, 1), args=[grad_2d, scale, gq, steps, seed, n, grad_2d.stride(0)],
              stream=stream)

    sms = _sms.get(idx, 64)
    n_k = k // 256
    ctas_128 = n_k * ((m + 127) // 128)
    kt = n // 64
    waves = (ctas_128 + sms - 1) // sms
    eff = ctas_128 / (waves * sms)
    use_s4 = _has_s4.get(idx, False) and n >= 2048
    # Two-phase Stream-K eliminates tail-wave quantization (e.g. 1.34 / 2.40 waves on 170-SM GPUs)
    # whenever wave efficiency is low and the reduction dimension is deep enough to amortize partial sums.
    dp_waves = max(0, (ctas_128 // sms) - 1)
    dp_tiles = dp_waves * sms
    sk_tiles = ctas_128 - dp_tiles
    if eff < 0.88 and kt >= 48 and sk_tiles * kt >= sms and m >= 512:
        try:
            partials = torch.empty((sms * 2 * 32768,), device=grad_2d.device, dtype=torch.int32)
            done = _done_buf(grad_2d.device, idx, stream_id, sk_tiles)
        except torch.OutOfMemoryError:
            partials = None  # fall back to the workspace-free ordinary GEMM instead of aborting the step
        if partials is not None:
            gname = (("zt_w8a8_gemm_s4_sk_rot" if rotate else "zt_w8a8_gemm_s4_sk") if use_s4
                     else ("zt_w8a8_gemm_sk_rot" if rotate else "zt_w8a8_gemm_sk"))
            ks[gname](grid=(sms, 1, 1), block=(256, 1, 1),
                      args=[gq, q, steps, out, residual if res else out, int(res), m, n, k, partials, done, dp_tiles],
                      shared_mem=SMEM4 if use_s4 else SMEM, stream=stream)
            return out
    if ctas_128 < sms * 2 and m <= 2048:
        # BM=64 (128 threads, 40 KB SMEM -> 2 CTAs/SM) fills SMs on small-M / short-N shapes
        gname = "zt_w8a8_gemm64_rot" if rotate else "zt_w8a8_gemm64"
        ks[gname](grid=(n_k * ((m + 63) // 64), 1, 1), block=(128, 1, 1),
                  args=[gq, q, steps, out, residual if res else out, int(res), m, n, k],
                  shared_mem=SMEM64, stream=stream)
    elif use_s4:
        gname = "zt_w8a8_gemm_s4_rot" if rotate else "zt_w8a8_gemm_s4"
        ks[gname](grid=(ctas_128, 1, 1), block=(256, 1, 1),
                  args=[gq, q, steps, out, residual if res else out, int(res), m, n, k],
                  shared_mem=SMEM4, stream=stream)
    else:
        gname = "zt_w8a8_gemm_rot" if rotate else "zt_w8a8_gemm"
        ks[gname](grid=(ctas_128, 1, 1), block=(256, 1, 1),
                  args=[gq, q, steps, out, residual if res else out, int(res), m, n, k],
                  shared_mem=SMEM, stream=stream)
    return out


@input_grad.register_fake
def _(grad_2d, q, scale, seed, rotate, residual=None):
    return grad_2d.new_empty((grad_2d.shape[0], q.shape[1]), dtype=torch.bfloat16)
