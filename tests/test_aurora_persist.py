"""AuroraObjectStore content-addressed persistence tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_META = Path(__file__).resolve().parents[1]
_WB = Path("/home/workdir/artifacts/wavebridge")
for p in (_META, _WB):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_put_get_roundtrip(tmp_path: Path):
    from aurora_persist import AuroraObjectStore

    store = AuroraObjectStore(root=tmp_path / "obj", publish=False)
    sha = store.put("observation", {"laser_id": 3, "peak": 0.8}, ref="test/obs")
    assert len(sha) == 64
    body = store.get(sha)
    assert body is not None
    assert body["kind"] == "observation"
    assert body["data"]["laser_id"] == 3
    resolved = store.resolve("test/obs")
    assert resolved is not None
    assert resolved["data"]["peak"] == 0.8


def test_put_transfer_matrix(tmp_path: Path):
    from aurora_persist import AuroraObjectStore

    store = AuroraObjectStore(root=tmp_path / "obj2", publish=False)
    T = np.random.default_rng(0).random((10, 6)).astype(np.float32)
    sha = store.put_transfer_matrix(T, body_id="b1")
    body = store.get(sha)
    assert body["kind"] == "transfer_matrix"
    assert body["meta"]["shape"] == [10, 6]


def test_persist_loop_smoke(tmp_path: Path):
    from examples.optical_body_closed_loop import run_loop
    from aurora_persist import AuroraObjectStore

    report = run_loop(steps=4, n_lasers=4, n_detectors=6, seed=1)
    store = AuroraObjectStore(root=tmp_path / "loop", publish=False)
    for h in report["history"]:
        store.put("observation", h)
    store.put_probe_history(report["history"], ref="loop/hist")
    assert store.resolve("loop/hist") is not None
    assert len(store.list_index()) >= 5
