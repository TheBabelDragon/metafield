"""Closed-loop synthetic MetaField ↔ WaveBridge test (no hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

_META = Path(__file__).resolve().parents[1]
_WB = Path("/home/workdir/artifacts/wavebridge")
for p in (_META, _WB):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.path.insert(0, str(_META / "examples"))


def test_closed_loop_runs():
    from wavebridge_closed_loop import run_loop

    report = run_loop(steps=4, dim=24, noise_rms=0.0003, seed=1)
    assert report["steps"] == 4
    assert report["store_size"] == 4
    assert len(report["history"]) == 4
    for h in report["history"]:
        assert h["mae"] < 0.1
    assert report["suggestion"] is not None
    assert "source_id" in report["suggestion"]
