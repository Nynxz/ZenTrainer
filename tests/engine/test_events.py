import json
import math
import sys
import types


def test_finite_makes_json_a_browser_can_read():
    from zentrainer.events import finite

    row = {"step": 3, "loss": 0.25, "grad_norm": math.inf, "by_noise": {"0.5": math.nan}, "rows": [1.5, -math.inf, (math.nan,)]}
    out = finite(row)
    assert out == {"step": 3, "loss": 0.25, "grad_norm": None, "by_noise": {"0.5": None}, "rows": [1.5, None, [None]]}
    json.dumps(out, allow_nan=False)  # what JSON.parse takes: no NaN or Infinity


def test_emit_sends_no_infinity(monkeypatch):
    from zentrainer import events

    sent = []
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(send_sync=lambda e, d: sent.append((e, d))))
    monkeypatch.setitem(sys.modules, "server", server)
    events.emit(events.METRICS, {"run_id": "r", "step": 1, "grad_norm": math.inf})
    assert sent == [(events.METRICS, {"run_id": "r", "step": 1, "grad_norm": None})]
