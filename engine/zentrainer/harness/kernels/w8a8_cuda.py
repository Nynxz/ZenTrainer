"""The W8A8 input gradient of a frozen int8 ConvRot linear in raw CUDA (w8a8_cuda.cu): the gradient
quantized to int8 per row (stochastically rounded), multiplied by the int8 weight as it's stored
on int8 tensor cores, scaled, and ConvRot's Hadamard rotation applied, all in two kernels. The
Triton path (w8a8_backward.py) takes three: its matmul, then the rotation as a bf16 matmul, which
also rounds twice.

Measured on an RTX 5090 at 4352 tokens (Krea 2's and MOSS's layer shapes), matmul and rotation:
6144 -> 6144 512 us against Triton's 734, 16384 -> 6144 1328 against 1875, 3072 -> 3072 166
against 223 (600-660 int8 TOPS on the big ones, cuBLAS's speed without its transposed weight
copy); the quantize 23 us against 60. One rounding to bf16 at the end: 1.66e-3 relative error
against the exact product, the Triton path's 2.35e-3.

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

SMEM = 3 * (128 * 64 + 64 * 256)  # the GEMM's pipeline: three stages of a 128 x 64 and a 64 x 256 tile
_NAMES = ("zt_quantize_rows", "zt_w8a8_gemm_rot", "zt_w8a8_gemm")
_kernels: dict[int, dict | None] = {}  # by device: its kernels, or None if they couldn't load


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
        with torch.cuda.device(idx):
            ptx, _ = _nvrtc_compile(src, _NAMES[0], f"{major}{minor}", None, ["-std=c++17"])
            ks = _cuda_load_module(ptx, list(_NAMES))
            for n in _NAMES[1:]:
                ks[n].set_shared_memory_config(SMEM)
        _kernels[idx] = ks
        logging.info("[ZenTrainer] int8 backward: CUDA kernels compiled for sm_%d%d", major, minor)
    except Exception:
        logging.warning("[ZenTrainer] int8 backward's CUDA kernels didn't load; using Triton's", exc_info=True)
    return _kernels[idx]


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
               rotate: bool) -> torch.Tensor:
    """bf16 (M, in): (quantized grad_2d * scale) @ q, rescaled, rotated by ConvRot's 256-wide
    Hadamard when `rotate`. grad_2d (M, out) bf16, q (out, in) int8, scale (out,) the weight's per
    row, seed a one-element int32 tensor on the device (the rounding noise's). See supported()."""
    ks = _load(grad_2d.device)
    m, n = grad_2d.shape
    k = q.shape[1]
    if ks is None or not _aligned(grad_2d, q):  # decided while tracing: the real tensors don't suit
        from .int8_backward import _w8a8_input_grad

        return _w8a8_input_grad(grad_2d, q, scale, 256 if rotate else 0)
    scale = scale.reshape(-1).float().contiguous()
    gq = torch.empty((m, n), device=grad_2d.device, dtype=torch.int8)
    steps = torch.empty((m,), device=grad_2d.device, dtype=torch.float32)
    out = torch.empty((m, k), device=grad_2d.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(grad_2d.device)
    ks["zt_quantize_rows"](grid=(m, 1, 1), block=(256, 1, 1), args=[grad_2d, scale, gq, steps, seed, n, grad_2d.stride(0)],
                           stream=stream)
    ks["zt_w8a8_gemm_rot" if rotate else "zt_w8a8_gemm"](grid=((k // 256) * ((m + 127) // 128), 1, 1), block=(256, 1, 1),
                                                         args=[gq, q, steps, out, m, n, k], shared_mem=SMEM, stream=stream)
    return out


@input_grad.register_fake
def _(grad_2d, q, scale, seed, rotate):
    return grad_2d.new_empty((grad_2d.shape[0], q.shape[1]), dtype=torch.bfloat16)
