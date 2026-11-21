"""Memory samples and loaded models pushed to the tabs that asked, and only those (server/system.py)."""

import asyncio
import sys
import types


def test_samples_go_to_watching_tabs_only_and_models_when_they_change(monkeypatch):
    from zentrainer.models import residency
    from zentrainer.server import system

    sent = []
    sockets = {"tab": object(), "comfyui": object()}  # ComfyUI's own tab never asks
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
        sockets=sockets, send_sync=lambda e, d, sid=None: sent.append((e, sid))))
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setattr(system, "sample", lambda: {"t": 1.0})
    reports = []
    monkeypatch.setattr(residency, "report", lambda: reports.append(1) or [{"name": "m", "bytes": 1, "vram": 1}])
    monkeypatch.setattr(residency, "fingerprint", lambda: ("same",))
    monkeypatch.setattr(system, "_watchers", set())
    monkeypatch.setattr(system, "_loaded", None)

    async def run(ticks):
        left = [ticks]

        async def sleep(_s):
            left[0] -= 1
            if not left[0]:
                raise asyncio.CancelledError

        monkeypatch.setattr(system.asyncio, "sleep", sleep)
        try:
            await system.sampler()
        except asyncio.CancelledError:
            pass

    system.watch("tab")
    system.watch("gone")  # asked, then its socket closed: dropped
    asyncio.run(run(3))
    assert system._watchers == {"tab"}
    assert sent.count(("zentrainer.system", "tab")) == 3 and sent.count(("zentrainer.loaded", "tab")) == 1
    assert all(sid == "tab" for _, sid in sent)
    assert len(reports) == 1  # nothing moved: the layers weren't walked again

    # asking again (ComfyUI's status comes with every prompt) doesn't send the same models again
    sent.clear()
    system.watch("tab")
    asyncio.run(run(2))
    assert ("zentrainer.loaded", "tab") not in sent
