"""The MOSS attention patch (harness/compat/moss.py) on a ComfyUI whose attention takes more."""

import sys
import types

import torch

from zentrainer.harness import compat
from zentrainer.harness.compat import moss as moss_patch


def _install(monkeypatch, in_training):
    calls = []

    def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options,
                  ref_kv=None, kv_out=None, mask=None):  # as ComfyUI after #17296 / #17320
        calls.append((ref_kv, kv_out, mask))
        return "comfy"

    moss = types.ModuleType("comfy.ldm.moss_image.model")
    moss.attention = attention
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = in_training
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.moss_image", types.ModuleType("comfy.ldm.moss_image")),
                      ("comfy.ldm.moss_image.model", moss), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    monkeypatch.setattr(compat, "fused_qk_rope", True)
    moss_patch.apply()
    return moss, calls


def test_new_arguments_reach_comfyui(monkeypatch):
    moss, calls = _install(monkeypatch, in_training=True)
    norm = torch.nn.RMSNorm(4).requires_grad_(False)
    args = (torch.zeros(1, 2, 8),) * 3 + (norm, norm, None, 2, None, {})
    mask = torch.zeros(1, 2)
    assert moss.attention(*args, mask=mask) == "comfy"  # by name (TextBlock)
    assert moss.attention(*args, "ref", None, mask) == "comfy"  # positionally (the image blocks)
    assert calls[0][2] is mask and calls[1] == ("ref", None, mask)


def test_sampling_goes_to_comfyui(monkeypatch):
    moss, calls = _install(monkeypatch, in_training=False)
    norm = torch.nn.RMSNorm(4).requires_grad_(False)
    assert moss.attention(*(torch.zeros(1, 2, 8),) * 3, norm, norm, None, 2, None, {}) == "comfy"
    assert calls == [(None, None, None)]


def test_modulated_rms_norm_keeps_input_dtype_in_training(monkeypatch):
    sampling_calls = []

    def sampling_mod(x, norm, scale, shift):
        sampling_calls.append(1)
        return "sampling"

    def sampling_gate(x, y, gate):
        sampling_calls.append(2)
        return "sampling_gate"

    moss = types.ModuleType("comfy.ldm.moss_image.model")
    moss.attention = lambda *a, **kw: "comfy"
    moss.modulated_rms_norm = sampling_mod
    moss.gated_residual = sampling_gate
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = True
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.moss_image", types.ModuleType("comfy.ldm.moss_image")),
                      ("comfy.ldm.moss_image.model", moss), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    monkeypatch.setattr(compat, "in_run", True)
    moss_patch.apply()

    norm = torch.nn.RMSNorm(8, dtype=torch.bfloat16)
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    y = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 8, dtype=torch.float32)
    shift = torch.randn(1, 1, 8, dtype=torch.float32)
    out = moss.modulated_rms_norm(x, norm, scale, shift)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out, (norm(x) * (1 + scale) + shift).to(torch.bfloat16))
    res = moss.gated_residual(x, y, scale)
    assert res.dtype == torch.bfloat16
    assert torch.allclose(res, torch.addcmul(x, y, scale).to(torch.bfloat16))
    # Outside a ZenTrainer run (e.g. a subsequent stock ComfyUI TrainLoraNode run): delegates untouched
    compat.in_run = False
    assert moss.modulated_rms_norm(x, norm, scale, shift) == "sampling"
    assert moss.gated_residual(x, y, scale) == "sampling_gate"
    # Sampling inside a run: also delegates untouched
    compat.in_run = True
    mm.in_training = False
    assert moss.modulated_rms_norm(x, norm, scale, shift) == "sampling"
    assert moss.gated_residual(x, y, scale) == "sampling_gate" and sampling_calls == [1, 2, 1, 2]
