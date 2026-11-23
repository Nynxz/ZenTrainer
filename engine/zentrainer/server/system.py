"""Memory telemetry: VRAM and RAM sampled every second into a 15-minute ring buffer, so
a chart opened mid-run still has history.

VRAM  device used/total (all processes, from the driver), PyTorch allocated/reserved
      in this process, and comfy_aimdo's own usage when dynamic VRAM is active.
RAM   system used/total, and this ComfyUI process's resident size.

Tabs that ask (`watch`) are sent each sample, and the loaded models when they change, on ComfyUI's
socket rather than polling for them. Only those: every other socket (ComfyUI's own tabs) would get
a message a second it has no use for.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

INTERVAL = 1.0
_samples: deque[dict] = deque(maxlen=900)
_watchers: set[str] = set()  # websocket client ids
_loaded: list[dict] | None = None  # the models last sent


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


def watch(sid: str) -> None:
    """Send this tab the samples and loaded models until its socket closes. It asks again when it
    reconnects, so a restart (which forgets) costs it nothing."""
    global _loaded
    # ComfyUI's status, on which tabs ask, also comes with every prompt queued or finished: only a
    # tab new to the list needs the models sent again
    if sid not in _watchers:
        _watchers.add(sid)
        _loaded = None


async def sampler() -> None:
    from server import PromptServer

    from .. import events
    from ..models import residency

    global _loaded
    seen = None  # residency.fingerprint() when the models were last read
    while True:
        try:
            s = await asyncio.to_thread(sample)
            _samples.append(s)
            _watchers.intersection_update(PromptServer.instance.sockets)
            for sid in _watchers:
                events.emit(events.SYSTEM, s, sid)
            # the full report walks every layer of every model: only when the cheap fingerprint
            # says something loaded, unloaded or streamed (or a tab is new)
            if _watchers:
                now = await asyncio.to_thread(residency.fingerprint)
                if now != seen or _loaded is None:
                    models = await asyncio.to_thread(residency.report)
                    # only cache the fingerprint when the report is complete and state didn't move under it;
                    # otherwise clear `seen` so the next tick retries even if residency settles back to its prior value
                    if not any("error" in m for m in models) and await asyncio.to_thread(residency.fingerprint) == now:
                        seen = now
                    else:
                        seen = None
                    if models != _loaded:
                        _loaded = models
                        for sid in _watchers:
                            events.emit(events.LOADED, {"models": models}, sid)
        except Exception:
            logging.debug("[ZenTrainer] memory sample failed", exc_info=True)
        await asyncio.sleep(INTERVAL)
