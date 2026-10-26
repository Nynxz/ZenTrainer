"""Qwen-Image 2.1 (harness/compat/qwen_image21.py): the prefix cache off and the residuals out of
place while training, ComfyUI's own code while sampling."""

import sys
import types

import torch

from zentrainer.harness import compat
from zentrainer.harness.compat import qwen_image21 as patch


def _gated_residual(x, y, gate, prefix_len):  # ComfyUI's, in place
    g_prefix, g_target = gate
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x


def _install(monkeypatch):
    class Model:
        def select_prefix_cache(self, key, cache_bytes, device, options):
            return "cache", True

    qi21 = types.ModuleType("comfy.ldm.qwen_image21.model")
    qi21.QwenImage21Transformer2DModel = Model
    qi21._gated_residual = _gated_residual
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = False
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.qwen_image21", types.ModuleType("comfy.ldm.qwen_image21")),
                      ("comfy.ldm.qwen_image21.model", qi21), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    patch.apply()
    return qi21, Model(), mm


def test_training_recomputes_the_prefix_and_sampling_keeps_the_cache(monkeypatch):
    _, model, mm = _install(monkeypatch)
    assert model.select_prefix_cache("k", 1, "cuda", {}) == ("cache", True)  # sampling: ComfyUI's
    mm.in_training = True
    assert model.select_prefix_cache("k", 1, "cuda", {}) == (None, False)  # training: no cache


def test_training_residuals_are_out_of_place_with_the_same_values(monkeypatch):
    qi21, _, mm = _install(monkeypatch)
    x, y = torch.randn(1, 6, 4), torch.randn(1, 6, 4)
    gate = (torch.randn(1, 1, 4), torch.randn(1, 1, 4))
    want = _gated_residual(x.clone(), y, gate, 2)
    mm.in_training = True
    for prefix in (2, 0):
        before = x.clone()
        out = qi21._gated_residual(x, y, gate, prefix)
        assert torch.equal(x, before)  # the block's input is left as it was
        assert torch.allclose(out, _gated_residual(x.clone(), y, gate, prefix))
    assert torch.allclose(qi21._gated_residual(x, y, gate, 2), want)
    mm.in_training = False
    assert qi21._gated_residual(x, y, gate, 2) is x  # sampling: ComfyUI's, in place


def test_the_cache_finds_a_slot_among_keys_of_other_lengths(monkeypatch):
    class PoseBranchCache:  # ComfyUI's select, as far as the slots go
        def __init__(self):
            self.slots, self.slot = [], None

        def select(self, k, create=True):
            for s in self.slots:
                if s["key"].shape == k.shape and torch.equal(s["key"], k):
                    self.slots.remove(s)
                    self.slots.append(s)
                    self.slot = s
                    return True
            return False

    animate2 = types.ModuleType("comfy.ldm.wan.model_animate2")
    animate2.PoseBranchCache = PoseBranchCache
    for name, mod in (("comfy.ldm.wan", types.ModuleType("comfy.ldm.wan")), ("comfy.ldm.wan.model_animate2", animate2)):
        monkeypatch.setitem(sys.modules, name, mod)
    _install(monkeypatch)
    cache = PoseBranchCache()
    cond, uncond = {"key": torch.randn(1, 9)}, {"key": torch.randn(1, 7)}  # CFG > 1, prompts of other lengths
    cache.slots = [cond, uncond]
    assert cache.select(uncond["key"].clone(), create=False) and cache.slot is uncond  # by value: a size error
    assert [s is uncond for s in cache.slots] == [False, True]  # the one found, moved last
    assert cache.select(cond["key"].clone(), create=False) and [s is cond for s in cache.slots] == [False, True]
