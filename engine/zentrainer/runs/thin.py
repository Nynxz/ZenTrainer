"""A long run's metrics, thinned for a page: a 150k-step run is 150k rows (62 MB of JSON), and the run
page fetching, holding and redrawing them all every step lagged the UI, then stopped it drawing at
all. Thinned, each row stands for a stretch of steps: the stretch's last row, carrying the mean
loss of all of them, so the curve is the same curve. The latest steps stay exact. Before thinning,
steps a resume from an earlier checkpoint redid replace the abandoned ones (web lib/resumed.ts
does the same to what the page gets live), so a stretch never mixes the two.
"""

from __future__ import annotations

import math

KEEP = 4000  # rows a page gets, at most
TAIL = 400  # of them, the latest, unthinned


def history(rows: list[dict]) -> list[dict]:
    """Rows in step order, a redone step replacing the rows from there on (they were abandoned)."""
    out: list[dict] = []
    for r in rows:
        while out and out[-1].get("step", 0) >= r.get("step", 0):
            out.pop()
        out.append(r)
    return out


def thin(rows: list[dict], keep: int = KEEP, tail: int = TAIL) -> tuple[list[dict], int]:
    """(rows, steps per row). Unchanged (1) when there are no more than `keep`."""
    rows = history(rows)
    if len(rows) <= keep:
        return rows, 1
    head, last = rows[:-tail], rows[-tail:]
    per = math.ceil(len(head) / (keep - tail))
    out = []
    for i in range(0, len(head), per):
        part = head[i:i + per]
        row = dict(part[-1])
        # a NaN or inf step is left out, not averaged: one would make the whole bucket null
        losses = [r["loss"] for r in part if isinstance(r.get("loss"), (int, float)) and math.isfinite(r["loss"])]
        if losses:
            row["loss"] = sum(losses) / len(losses)
        out.append(row)
    return out + last, per
