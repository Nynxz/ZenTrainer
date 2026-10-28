"""A long run's metrics thinned for the run page (runs/thin.py)."""

from zentrainer.runs.thin import history, thin


def _rows(n, start=1, elapsed0=0.0):
    return [{"step": s, "loss": float(s % 7), "elapsed": elapsed0 + s} for s in range(start, start + n)]


def test_short_runs_are_left_alone():
    rows = _rows(300)
    out, every = thin(rows)
    assert out == rows and every == 1


def test_a_long_run_keeps_its_curve_and_its_latest_steps():
    rows = _rows(150_000)
    out, every = thin(rows, keep=4000, tail=400)
    assert len(out) <= 4000 and every > 1
    assert out[-400:] == rows[-400:]  # the latest steps exact
    assert [r["step"] for r in out] == sorted(r["step"] for r in out)
    first = rows[:every]  # a stretch's row carries the stretch's mean loss, at its last step
    assert out[0]["step"] == first[-1]["step"]
    assert abs(out[0]["loss"] - sum(r["loss"] for r in first) / len(first)) < 1e-9
    mean_all = sum(r["loss"] for r in rows[:-400]) / len(rows[:-400])
    mean_out = sum(r["loss"] for r in out[:-400]) / len(out[:-400])
    assert abs(mean_all - mean_out) < 0.01  # the same curve, fewer points



def test_a_nan_step_is_left_out_of_its_stretch_not_spread_over_it():
    rows = _rows(10_000)
    rows[5]["loss"] = float("nan")
    rows[6]["loss"] = float("inf")
    out, every = thin(rows, keep=1000, tail=100)
    first = [r["loss"] for r in rows[:every] if r["loss"] == r["loss"] and r["loss"] != float("inf")]
    assert abs(out[0]["loss"] - sum(first) / len(first)) < 1e-9

def test_abandoned_steps_give_way_to_their_redo():
    rows = _rows(10) + _rows(5, start=6, elapsed0=100.0)  # resumed from step 5's checkpoint
    kept = history(rows)
    assert [r["step"] for r in kept] == list(range(1, 11)) and kept[-1]["elapsed"] == 110.0
