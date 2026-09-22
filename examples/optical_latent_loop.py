#!/usr/bin/env python3
"""
optical_latent_loop.py

Phase 13 software reference:

  OpticalBodySimulator
        ↓ detector vector
  OpticalEncoder (fit on buffer)
        ↓ latent z
  latent-space novelty → active probe
        ↓ next laser

The encoder interprets the physical substrate; the body remains the source of truth.

Usage:
  PYTHONPATH=../wavebridge:. python examples/optical_latent_loop.py
  PYTHONPATH=../wavebridge:. python examples/optical_latent_loop.py --steps 30 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry
from optical_encoder import OpticalEncoder
from examples.optical_body_closed_loop import _make_body, suggest_next_laser


def _latent_novelty(z: np.ndarray, prior_latents: List[np.ndarray]) -> float:
    if not prior_latents:
        return 1.0
    stack = np.stack(prior_latents, axis=0)
    dists = np.linalg.norm(stack - z.reshape(1, -1), axis=1)
    return float(min(1.0, float(np.min(dists)) * 2.0))


def run_loop(
    steps: int = 24,
    n_lasers: int = 12,
    n_detectors: int = 20,
    latent_dim: int = 6,
    seed: int = 0,
    fit_every: int = 6,
    epochs: int = 30,
) -> Dict[str, Any]:
    body = _make_body(n_lasers, n_detectors, seed, noise_rms=0.01)
    enc = OpticalEncoder(n_detectors, latent_dim=latent_dim, seed=seed)
    store = FieldMemoryStore(soft_capacity=512)

    responses: List[np.ndarray] = []
    latents: List[np.ndarray] = []
    history: List[Dict[str, Any]] = []
    last_laser: Optional[int] = None
    next_laser = 0

    for step in range(steps):
        if step > 0:
            suggestion = suggest_next_laser(
                [item["entry"] for item in store.buffer],
                n_lasers,
                last_laser=last_laser,
            )
            next_laser = int(suggestion["source_id"])
        else:
            suggestion = {
                "action": "excite",
                "source_id": 0,
                "reason": "bootstrap",
                "curiosity_score": 1.0,
            }

        rec = body.excite(next_laser, drive_level=0.85)
        response = np.asarray(rec.detector_response, dtype=np.float32)
        responses.append(response)

        if step > 0 and step % fit_every == 0 and len(responses) >= latent_dim:
            enc.fit(responses, epochs=epochs)

        if enc.stats.n_samples == 0 and len(responses) >= max(4, latent_dim):
            enc.fit(responses, epochs=epochs)

        if enc.stats.n_samples > 0:
            z = enc.encode(response)
            recon_err = enc.reconstruction_error(response)
        else:
            z = np.zeros(latent_dim, dtype=np.float32)
            recon_err = 1.0

        novelty = _latent_novelty(z, latents)
        latents.append(z)
        anomaly = float(min(1.0, 0.5 * novelty + 0.5 * min(1.0, recon_err * 10.0)))

        entry = FieldMemoryEntry(
            body_id="optical-latent-sim",
            excitation_id=next_laser,
            observed_response=response.tolist(),
            confidence=0.9 if recon_err < 0.05 else 0.6,
            anomaly=anomaly,
            extras={
                "laser_id": next_laser,
                "latent": z.tolist(),
                "recon_mse": recon_err,
                "latent_novelty": novelty,
                "encoder_backend": enc.backend,
            },
        )
        store.add(entry)
        last_laser = next_laser

        history.append(
            {
                "step": step,
                "laser_id": next_laser,
                "anomaly": round(anomaly, 4),
                "recon_mse": round(recon_err, 5),
                "latent_novelty": round(novelty, 4),
                "latent_norm": round(float(np.linalg.norm(z)), 4),
                "reason": suggestion.get("reason"),
            }
        )

    return {
        "steps": steps,
        "n_lasers": n_lasers,
        "n_detectors": n_detectors,
        "latent_dim": latent_dim,
        "encoder_backend": enc.backend,
        "encoder_stats": {
            "n_samples": enc.stats.n_samples,
            "last_recon_mse": enc.stats.last_recon_mse,
        },
        "store_size": len(store),
        "coverage": len({h["laser_id"] for h in history}) / max(n_lasers, 1),
        "history": history,
        "final_suggestion": suggest_next_laser(
            [item["entry"] for item in store.buffer],
            n_lasers,
            last_laser=last_laser,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical latent closed loop")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lasers", type=int, default=12)
    parser.add_argument("--detectors", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_loop(
        steps=args.steps,
        n_lasers=args.lasers,
        n_detectors=args.detectors,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"[optical_latent] backend={report['encoder_backend']} "
            f"coverage={report['coverage']:.0%} "
            f"recon_mse={report['encoder_stats']['last_recon_mse']:.5f}"
        )
        for h in report["history"][:10]:
            print(
                f"  step {h['step']:02d}: laser={h['laser_id']:02d} "
                f"anom={h['anomaly']:.3f} recon={h['recon_mse']:.4f} "
                f"nov={h['latent_novelty']:.3f}"
            )
        if len(report["history"]) > 10:
            print(f"  ... ({len(report['history']) - 10} more)")
        fs = report["final_suggestion"]
        print(f"[probe] next laser={fs['source_id']} reason={fs.get('reason')}")


if __name__ == "__main__":
    main()
