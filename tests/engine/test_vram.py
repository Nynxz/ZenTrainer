"""Whole-card VRAM on Windows ROCm (vram.py), and torch's own numbers when its counter can't be had."""

import logging
import types

import torch

from zentrainer import vram


def windows_rocm(monkeypatch, counter):
    monkeypatch.setattr(vram, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (12, 16))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: types.SimpleNamespace(pci_bus_id=3))
    monkeypatch.setattr(vram, "_counters", {})
    monkeypatch.setattr(vram, "_Counter", counter)


def test_other_processes_count_as_used(monkeypatch):
    windows_rocm(monkeypatch, lambda bus: types.SimpleNamespace(read=lambda: 10))
    assert vram.mem_get_info() == (6, 16)


def test_a_counter_that_fails_any_way_falls_back_to_torch(monkeypatch, caplog):
    def broken(bus):
        raise AttributeError("D3DKMTEnumAdapters2")
    windows_rocm(monkeypatch, broken)
    with caplog.at_level(logging.WARNING):
        assert vram.mem_get_info() == (12, 16)
        assert vram.mem_get_info() == (12, 16)
    assert len(caplog.records) == 1 and "PCI bus 3" in caplog.text
