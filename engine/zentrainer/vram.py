"""The GPU's free and total memory, counting every process on it.

torch.cuda.mem_get_info asks the driver about the whole card, except with ROCm on Windows, where it
only counts this process: the desktop and other programs don't show, so a VRAM fit would plan with
memory that's already taken. There the used memory comes from Windows' own counter (the dedicated
GPU memory Task Manager shows), for the adapter on the GPU's PCI bus.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading

import torch

_lock = threading.Lock()
_counters: dict[int, object] = {}


def mem_get_info(device=None) -> tuple[int, int]:
    free, total = torch.cuda.mem_get_info(device)
    if os.name != "nt" or not torch.version.hip:
        return free, total
    bus = torch.cuda.get_device_properties(device).pci_bus_id
    with _lock:
        if bus not in _counters:
            try:
                _counters[bus] = _Counter(bus)
            except Exception as e:  # any Windows quirk here only costs the other processes' share
                logging.warning("[ZenTrainer] no Windows GPU memory counter for PCI bus %d (%s): VRAM shows this "
                                "process only", bus, e)
                _counters[bus] = None
        counter = _counters[bus]
        used = counter.read() if counter else 0
    return min(free, total - used), total


class _Counter:
    """PDH's "GPU Adapter Memory" counter for the adapter on a PCI bus, found through D3DKMT (which
    names adapters by LUID, as the counter's instances do)."""

    def __init__(self, bus: int):
        from ctypes import wintypes

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class ADAPTERINFO(ctypes.Structure):
            _fields_ = [("hAdapter", wintypes.UINT), ("AdapterLuid", LUID), ("NumOfSources", wintypes.ULONG),
                        ("bPrecisePresentRegionsPreferred", wintypes.BOOL)]

        class ENUMADAPTERS2(ctypes.Structure):
            _fields_ = [("NumAdapters", wintypes.ULONG), ("pAdapters", ctypes.POINTER(ADAPTERINFO))]

        class QUERYADAPTERINFO(ctypes.Structure):
            _fields_ = [("hAdapter", wintypes.UINT), ("Type", ctypes.c_int), ("pPrivateDriverData", ctypes.c_void_p),
                        ("PrivateDriverDataSize", wintypes.UINT)]

        class ADAPTERADDRESS(ctypes.Structure):
            _fields_ = [("BusNumber", wintypes.UINT), ("DeviceNumber", wintypes.UINT), ("FunctionNumber", wintypes.UINT)]

        gdi = ctypes.WinDLL("gdi32")
        found = ENUMADAPTERS2()
        if gdi.D3DKMTEnumAdapters2(ctypes.byref(found)):  # a first call with no array asks how many
            raise OSError("D3DKMTEnumAdapters2 failed")
        adapters = (ADAPTERINFO * found.NumAdapters)()
        found.pAdapters = adapters
        if gdi.D3DKMTEnumAdapters2(ctypes.byref(found)):
            raise OSError("D3DKMTEnumAdapters2 failed")
        luid = None
        for a in adapters[:found.NumAdapters]:
            addr = ADAPTERADDRESS()
            query = QUERYADAPTERINFO(a.hAdapter, 6, ctypes.cast(ctypes.byref(addr), ctypes.c_void_p),  # ADAPTERADDRESS
                                     ctypes.sizeof(addr))
            if not gdi.D3DKMTQueryAdapterInfo(ctypes.byref(query)) and addr.BusNumber == bus and luid is None:
                luid = a.AdapterLuid
            gdi.D3DKMTCloseAdapter(ctypes.byref(wintypes.UINT(a.hAdapter)))
        if luid is None:
            raise OSError("no display adapter on that bus")

        self.pdh = ctypes.WinDLL("pdh")
        self.query, self.counter = ctypes.c_void_p(), ctypes.c_void_p()
        path = r"\GPU Adapter Memory(luid_0x%08X_0x%08X_phys_0)\Dedicated Usage" % (luid.HighPart & 0xFFFFFFFF,
                                                                                    luid.LowPart)
        if (self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query))
                or self.pdh.PdhAddEnglishCounterW(self.query, path, None, ctypes.byref(self.counter))):
            raise OSError(f"no counter {path}")

        class VALUE(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

        self.value = VALUE()

    def read(self) -> int:
        if (self.pdh.PdhCollectQueryData(self.query)
                or self.pdh.PdhGetFormattedCounterValue(self.counter, 0x400, None, ctypes.byref(self.value))):  # PDH_FMT_LARGE
            return 0
        return self.value.largeValue
