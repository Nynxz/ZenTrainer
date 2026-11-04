"""Overlapped weight streaming while an offloaded model trains.

ComfyUI casts an offloaded layer's weights on one of two offload streams, and picking a stream
(comfy.model_management.get_offload_stream) makes it wait for everything already queued on the
compute stream. So layer N+1's copy can't start until layer N has finished: every copy sits
between two layers instead of under one. Profiled on Krea 2 int8, 16 GB: the step's extra time
over a fully loaded one was almost exactly its host-to-device copies (410 ms).

That wait guards the stream's cast buffer, and uncast_bias_weight already guards it precisely:
after a layer's last use of its weights the stream waits for that layer. With two streams in
turn, layer N+1's copy waits only for layer N-1 (its buffer's last user) and runs while layer N
computes. Every training-time caller releases its stream that way (ComfyUI's layers through
CastBiasWeightContext, as does int8_backward). The dynamic-VRAM path (ops.cast_modules_with_vbar)
serves sampling models, so this is installed for training steps only and taken off for samples.
Checked: identical first-step loss with and without, steps 1.74 -> 1.57 s.
"""

from __future__ import annotations

import logging

_original = None


def _stream(device):
    import torch

    import comfy.model_management as mm

    if mm.NUM_STREAMS == 0 or torch.compiler.is_compiling() or device not in mm.STREAMS:
        return _original(device)  # its own setup (first use) and its compile-time None
    ss = mm.STREAMS[device]
    c = (mm.stream_counters.get(device, 0) + 1) % len(ss)
    mm.stream_counters[device] = c
    return ss[c]


def overlap(on: bool) -> None:
    """Weight copies overlap the previous layer's compute (on), or ComfyUI's own (off)."""
    global _original
    import comfy.model_management as mm

    if on and _original is None:
        if mm.NUM_STREAMS == 0:
            return
        _original, mm.get_offload_stream = mm.get_offload_stream, _stream
        logging.info("[ZenTrainer] offloaded weights stream under the previous layer's compute")
    elif not on and _original is not None:
        mm.get_offload_stream, _original = _original, None


# With cuda_malloc (ComfyUI's default) a streamed layer's weights are fresh allocations on its
# offload stream, and uncast_bias_weight orders their free after the layer's forward only. Training
# keeps them for the backward too (autograd saves the weight), and cudaMallocAsync frees in the
# offload stream's order: the backward GEMM still queued on the compute stream can find them reused,
# or unmapped by the next empty_cache. A CUDA coredump caught it as a Warp MMU Fault in a bf16
# cuBLAS GEMM, ~1 run in 3 on Krea 2 offloaded with Automagic3 + EMA (their long step tail lets the
# CPU run far ahead of the GPU) right after the weights moved; never with CUDA_LAUNCH_BLOCKING.
# So while grads are on, a streamed weight the layer uses as it is gets copied once on the compute
# stream: autograd keeps the copy, freed in the compute stream's order. (record_stream on the
# original also orders it, but holds every block's weights until the GPU catches up: out of VRAM.)
_cast = None


def _own(t):
    import torch

    qdata = getattr(t, "_qdata", None)
    if isinstance(qdata, torch.Tensor) and hasattr(t, "_copy_with"):
        return t._copy_with(qdata=qdata.clone())  # a quantized weight: its scales are cloned too
    return t.clone()


def _guarded_cast(*args, **kwargs):
    import torch

    out = _cast(*args, **kwargs)
    if len(out) == 3 and out[2] is not None and out[2][0] is not None and torch.is_grad_enabled() \
            and not torch.compiler.is_compiling():
        weight, bias, state = out
        _, weight_a, bias_a = state
        if weight is not None and weight is weight_a:
            weight = _own(weight)
        if bias is not None and bias is bias_a:
            bias = _own(bias)
        out = (weight, bias, state)
    return out


def guard_backward(on: bool) -> None:
    """Streamed weights the backward keeps are the compute stream's own (on), or ComfyUI's (off)."""
    global _cast
    import comfy.ops

    if on and _cast is None:
        _cast, comfy.ops.cast_bias_weight = comfy.ops.cast_bias_weight, _guarded_cast
    elif not on and _cast is not None:
        comfy.ops.cast_bias_weight, _cast = _cast, None


# ComfyUI logs a bare "Pin error." when it can't pin an offloaded weight (and drops CUDA's code).
# Unpinned weights copy to the GPU synchronously, so offloaded steps slow down and can't overlap.
# While training, every pin it tries is counted, and a summary with the CUDA codes is logged when
# training pauses for samples or ends.
_register = None
_pins: dict = {}


# torch's binding names few of the codes this returns
_CODES = {1: "invalid value", 2: "out of memory", 712: "already registered", 713: "not registered", 801: "not supported"}


def _counted_register(ptr, size, flags):
    r = _register(ptr, size, flags)
    code = int(r)
    key = "ok" if code == 0 else f"CUDA error {code}" + (f": {_CODES[code]}" if code in _CODES else "")
    _pins[key] = _pins.get(key, 0) + 1
    return r


def watch_pins(on: bool) -> None:
    """Count ComfyUI's pinning while training (on); log what failed and stop counting (off)."""
    global _register
    import torch

    if not torch.cuda.is_available():
        return
    rt = torch.cuda.cudart()
    if on and _register is None:
        _register, rt.cudaHostRegister = rt.cudaHostRegister, _counted_register
    elif not on and _register is not None:
        rt.cudaHostRegister, _register = _register, None
        failed = {k: v for k, v in _pins.items() if k != "ok"}
        if failed:
            logging.warning("[ZenTrainer] ComfyUI pinned %d offloaded weights and couldn't pin %d (%s): those copy to the GPU "
                            "without overlapping, so offloaded steps are slower", _pins.get("ok", 0), sum(failed.values()),
                            ", ".join(f"{k} ×{v}" for k, v in failed.items()))
        elif _pins:
            logging.info("[ZenTrainer] ComfyUI pinned all %d offloaded weights it tried", _pins.get("ok", 0))
        _pins.clear()


# ComfyUI's server maps model files through comfy-aimdo (comfy.utils.load_safetensors), and CUDA
# can't pin that memory in place: cudaHostRegister answers 1 "invalid value" (801 "not supported"
# with the read-only flag), so every offloaded weight copied to the GPU synchronously. A pinned
# copy (pin_memory) of the same bytes works. Offloaded weights still unpinned get one when training
# loads them; ComfyUI then sees them pinned and leaves them be. Costs RAM equal to the offloaded
# part, so it's skipped when that would leave less than this free.
PIN_RAM_SPARE = 8 * 1024 ** 3


def _unpinned(t):
    """The CPU tensor holding a weight's bytes when it isn't pinned (an int8 weight's storage is its
    QuantizedTensor's _qdata), else None."""
    import torch

    if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
        return None
    inner = getattr(t, "_qdata", None)
    data = inner if isinstance(inner, torch.Tensor) else t
    return None if data.numel() == 0 or data.is_pinned() else data


def pin_offloaded(model) -> tuple[int, int]:
    """Pinned copies of the offloaded (CPU) weights not pinned yet; (tensors, bytes) pinned."""
    import torch

    todo = []
    for m in model.modules():
        for name in ("weight", "bias"):
            t = getattr(m, name, None)
            if _unpinned(t) is not None:
                todo.append((m, name, t))
    need = sum(_unpinned(t).nbytes for _, _, t in todo)
    if not todo:
        return 0, 0
    try:
        import comfy.model_management as mm

        free = mm.get_free_memory(torch.device("cpu"))
    except Exception:
        free = None
    if free is not None and free - need < PIN_RAM_SPARE:
        logging.warning("[ZenTrainer] offloaded weights left unpinned: pinning %.1f GB would leave under %.0f GB of RAM free",
                        need / 1024 ** 3, PIN_RAM_SPARE / 1024 ** 3)
        return 0, 0
    done = 0
    for m, name, t in todo:
        inner = getattr(t, "_qdata", None)
        if isinstance(inner, torch.Tensor):
            t._qdata = inner.pin_memory()
        else:
            t.data = t.data.pin_memory()
        done += 1
    _pins.clear()  # ComfyUI's failed attempts on these are moot now: they're pinned
    return done, need


def pin_state(model) -> tuple[int, int]:
    """(bytes of weights offloaded to RAM, bytes of them not pinned): unpinned ones copy to the GPU
    synchronously and the system can swap them out."""
    import torch

    offloaded = loose = 0
    for m in model.modules():
        for name in ("weight", "bias"):
            t = getattr(m, name, None)
            if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
                continue
            inner = getattr(t, "_qdata", None)
            data = inner if isinstance(inner, torch.Tensor) else t
            offloaded += data.nbytes
            if _unpinned(t) is not None:
                loose += data.nbytes
    return offloaded, loose


def _cpu_weights(model):
    """(module, attribute, tensor holding the bytes) for every CPU weight and bias not pinned."""
    out = []
    for m in model.modules():
        for name in ("weight", "bias"):
            t = getattr(m, name, None)
            data = _unpinned(t)
            if data is not None:
                out.append((m, name, data))
    return out


def _clear_cuda_error() -> None:
    """Swallow the error a failed cudaHostRegister leaves queued (as ComfyUI does after its own
    failed pins): left there, the next CUDA call fails with "invalid argument"."""
    import torch

    try:
        a = torch.ones(1, dtype=torch.uint8, device="cuda")
        _ = a + a
        torch.cuda.synchronize()
    except Exception:
        pass


def own_ram(model) -> tuple[int, int]:
    """Before ComfyUI loads the model for training: weights still in memory CUDA can't pin in
    place (the model file as comfy-aimdo maps it) move to ordinary RAM, which ComfyUI then pins
    itself as it offloads them, with no errors. One weight is tried first; where that pins (the
    usual case without aimdo) nothing is copied. Returns (tensors, bytes) copied."""
    import torch

    weights = _cpu_weights(model)
    if not weights:
        return 0, 0
    rt = torch.cuda.cudart()
    probe = weights[0][2]
    if int(rt.cudaHostRegister(probe.data_ptr(), probe.nbytes, 1)) == 0:
        rt.cudaHostUnregister(probe.data_ptr())
        return 0, 0
    _clear_cuda_error()  # a failed register also queues an error the next CUDA call would raise
    need = sum(d.nbytes for _, _, d in weights)
    try:
        import comfy.model_management as mm

        free = mm.get_free_memory(torch.device("cpu"))
    except Exception:
        free = None
    if free is not None and free - need < PIN_RAM_SPARE:
        logging.warning("[ZenTrainer] the model stays in its file mapping, which CUDA can't pin: copying it to RAM "
                        "(%.1f GB) would leave under %.0f GB free, so offloaded weights copy to the GPU slower",
                        need / 1024 ** 3, PIN_RAM_SPARE / 1024 ** 3)
        return 0, 0
    for m, name, data in weights:
        t = getattr(m, name)
        if isinstance(getattr(t, "_qdata", None), torch.Tensor):
            t._qdata = data.clone()
        else:
            t.data = data.clone()
    return len(weights), need
