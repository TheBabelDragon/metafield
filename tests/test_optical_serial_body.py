"""OpticalSerialBody dry-run tests (no hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

_META = Path(__file__).resolve().parents[1]
_WB = Path("/home/workdir/artifacts/wavebridge")
for p in (_META, _WB):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_serial_body_dry_run():
    from optical_serial_body import OpticalSerialBody

    with OpticalSerialBody(dry_run=True, n_lasers=6, n_detectors=10) as body:
        assert body.ping() is True
        rec = body.excite(2)
        assert rec.ok
        assert rec.detector_response.shape == (10,)
        T = body.estimate_transfer_matrix()
        assert T.shape[1] == 6


def test_hardware_probe_dry_run(tmp_path: Path):
    sys.path.insert(0, str(_META / "examples"))
    from optical_hardware_probe import run_probe

    report = run_probe(
        "/dev/null",
        n_lasers=4,
        n_detectors=8,
        dry_run=True,
        persist_root=tmp_path / "p",
    )
    assert report["ping"] is True
    assert report["n_ok"] == 4
    assert report["n_fail"] == 0
