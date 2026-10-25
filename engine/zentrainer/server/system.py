"""Memory telemetry: VRAM and RAM sampled every second into a 15-minute ring buffer, so
a chart opened mid-run still has history.

VRAM  device used/total (all processes, from the driver), PyTorch allocated/reserved
      in this process, and comfy_aimdo's own usage when dynamic VRAM is active.
RAM   system used/total, and this ComfyUI process's resident size.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

INTERVAL = 1.0
_samples: deque[dict] = deque(maxlen=900)


def sample() -> dict:
    s: dict = {"t": round(time.time(), 2)}
    try:
        import psutil

        vm = psutil.virtual_memory()
        s["ram_total"] = vm.total
        s["ram_used"] = vm.total - vm.available
        s["proc_rss"] = psutil.Process().memory_info().rss
    except Exception:
        pass
    try:
        import torch

        from .. import vram

        if torch.cuda.is_available():
            free, total = vram.mem_get_info()
            s["vram_total"] = total
            s["vram_used"] = total - free
            s["torch_allocated"] = torch.cuda.memory_allocated()
            s["torch_reserved"] = torch.cuda.memory_reserved()
    except Exception:
        pass
    try:
        import comfy_aimdo.control as aimdo

        used = aimdo.get_total_vram_usage()
        if isinstance(used, (int, float)) and used > 0:
            s["aimdo_used"] = int(used)
    except Exception:
        pass
    return s


def history(since: float = 0.0) -> list[dict]:
    return [x for x in list(_samples) if x["t"] > since]


async def sampler() -> None:
    while True:
        try:
            _samples.append(await asyncio.to_thread(sample))
        except Exception:
            logging.debug("[ZenTrainer] memory sample failed", exc_info=True)
        await asyncio.sleep(INTERVAL)
