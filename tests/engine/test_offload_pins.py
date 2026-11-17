"""What an offloaded run holds in RAM: pinned weights (offload.pin_state) and the alternate
sampling model, let go between sample passes when training needs the RAM (Samples._let_sampler_go)."""

import sys
import types

import pytest
import torch

from zentrainer.harness.callbacks import Samples
from zentrainer.harness.memory import offload


def test_pin_state_counts_offloaded_and_unpinned_bytes():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2, bias=False))
    offloaded, loose = offload.pin_state(model)
    assert offloaded == loose == (16 + 4 + 8) * 4  # float32, nothing pinned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_pin_state_counts_pinned_weights_as_held():
    lin = torch.nn.Linear(4, 4)
    lin.weight.data = lin.weight.data.pin_memory()
    offloaded, loose = offload.pin_state(lin)
    assert offloaded == (16 + 4) * 4 and loose == 4 * 4  # only the bias is loose


class _Alt:
    def model_size(self):
        return 3 * 1024 ** 3


@pytest.fixture
def mm(monkeypatch):
    state = {"free": 100 * 1024 ** 3, "calls": []}
    fake = types.SimpleNamespace(get_free_memory=lambda _d: state["free"],
                                 cleanup_models_gc=lambda: state["calls"].append("gc"))
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=fake))
    return state


def _run(offloading):
    return types.SimpleNamespace(
        spec=types.SimpleNamespace(memory=types.SimpleNamespace(offloading=offloading),
                                   samples=types.SimpleNamespace(model_name="turbo")),
        models=types.SimpleNamespace(patcher=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=None))))


def test_the_sampling_model_stays_while_ram_has_room(mm, monkeypatch):
    monkeypatch.setattr(offload, "pin_state", lambda _m: (4 * 1024 ** 3, 0))
    s = Samples(1)
    s._alt = alt = _Alt()
    s._let_sampler_go(_run(True))  # offloaded, nothing left to pin, 100 GB free
    assert s._alt is alt and mm["calls"] == []


def test_it_goes_when_ram_is_short(mm):
    mm["free"] = 10 * 1024 ** 3  # under its 3 GB plus the 8 GB spare
    s = Samples(1)
    s._alt = _Alt()
    s._let_sampler_go(_run(False))
    assert s._alt is None and mm["calls"] == ["gc"]


def test_it_goes_when_training_still_has_weights_to_pin(mm, monkeypatch):
    mm["free"] = 14 * 1024 ** 3  # room for its 3 GB plus the 8 GB spare, not beside 4 GB still to pin
    monkeypatch.setattr(offload, "pin_state", lambda _m: (4 * 1024 ** 3, 4 * 1024 ** 3))
    s = Samples(1)
    s._alt = _Alt()
    s._let_sampler_go(_run(True))
    assert s._alt is None
