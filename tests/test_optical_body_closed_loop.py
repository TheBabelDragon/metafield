"""Optical body closed-loop coverage and memory tests."""

from __future__ import annotations

import sys
from pathlib import Path

_META = Path(__file__).resolve().parents[1]
_WB = Path("/home/workdir/artifacts/wavebridge")
for p in (_META, _WB):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(_META / "examples"))


def test_optical_body_covers_all_lasers():
    from optical_body_closed_loop import run_loop

    report = run_loop(steps=16, n_lasers=8, n_detectors=12, seed=3, noise_rms=0.005)
    assert report["store_size"] == 16
    assert report["coverage"] == 1.0
    assert len(report["lasers_seen"]) == 8
    assert report["transfer_matrix_shape"] == [12, 8]
    assert report["final_suggestion"]["action"] == "excite"


def test_optical_body_records_laser_in_extras():
    from optical_body_closed_loop import run_loop

    report = run_loop(steps=4, n_lasers=4, n_detectors=6, seed=0)
    assert all("laser_id" in h for h in report["history"])
