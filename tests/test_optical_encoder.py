"""OpticalEncoder unit + latent-loop smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_META = Path(__file__).resolve().parents[1]
_WB = Path("/home/workdir/artifacts/wavebridge")
for p in (_META, _WB):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_encoder_fit_encode_decode():
    from optical_encoder import OpticalEncoder

    rng = np.random.default_rng(0)
    n_det = 16
    samples = [rng.random(n_det).astype(np.float32) for _ in range(40)]
    enc = OpticalEncoder(n_det, latent_dim=4, seed=0)
    stats = enc.fit(samples, epochs=25)
    assert stats.n_samples == 40
    z = enc.encode(samples[0])
    assert z.shape == (4,)
    x_hat = enc.decode(z)
    assert x_hat.shape == (n_det,)
    assert float(np.min(x_hat)) >= -1e-5
    assert float(np.max(x_hat)) <= 1.0 + 1e-5
    err = enc.reconstruction_error(samples[0])
    assert err >= 0.0


def test_latent_loop_runs():
    sys.path.insert(0, str(_META / "examples"))
    from optical_latent_loop import run_loop

    report = run_loop(
        steps=12, n_lasers=6, n_detectors=10, latent_dim=4, seed=1, epochs=15
    )
    assert report["store_size"] == 12
    assert report["coverage"] == 1.0
    assert report["encoder_stats"]["n_samples"] > 0
    assert "latent_novelty" in report["history"][0]
