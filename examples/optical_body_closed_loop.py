#!/usr/bin/env python3
"""
optical_body_closed_loop.py

Phase 11-12 software reference: multi-laser optical body + MetaField memory.

  laser_id (stimulus)
        |
  OpticalBodySimulator.excite
        |
  BPW34 detector vector
        |
  FieldMemoryStore
        |
  active_probe (curiosity / unexplored lasers)
        |
  next laser_id

No hardware. Uses WaveBridge OpticalBodySimulator when available;
falls back to a tiny local coupling model if wavebridge is missing.

Usage:
  PYTHONPATH=../wavebridge:. python examples/optical_body_closed_loop.py
  PYTHONPATH=../wavebridge:. python examples/optical_body_closed_loop.py --steps 24 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry
from active_probe import score_sources, suggest_next


def _make_body(n_lasers: int, n_detectors: int, seed: int, noise_rms: float):
    try:
        from wavebridge import OpticalBodySimulator

        return OpticalBodySimulator(
            n_lasers=n_lasers,
            n_detectors=n_detectors,
            seed=seed,
            noise_rms=noise_rms,
        )
    except ImportError:
        class _Fallback:
            def __init__(self):
                self.n_lasers = n_lasers
                self.n_detectors = n_detectors
                rng = np.random.default_rng(seed)
                self.coupling = rng.random((n_detectors, n_lasers)).astype(np.float32)
                self._rng = np.random.default_rng(seed + 1)
                self.noise_rms = noise_rms
                self.ambient = 0.02

            def excite(self, laser_id, drive_level=1.0, geometry_state="nominal"):
                from types import SimpleNamespace

                power = float(np.tanh(drive_level * 1.5))
                base = self.coupling[:, laser_id] * power
                noise = self._rng.normal(0, self.noise_rms, size=self.n_detectors)
                resp = np.clip(base + self.ambient + noise, 0, 1).astype(np.float32)
                return SimpleNamespace(
                    laser_id=laser_id,
                    drive_level=drive_level,
                    detector_response=resp,
                    ambient=self.ambient,
                    geometry_state=geometry_state,
                )

            def estimate_transfer_matrix(self, drive_level=1.0):
                cols = [
                    self.excite(L, drive_level).detector_response
                    for L in range(self.n_lasers)
                ]
                return np.column_stack(cols).astype(np.float32)

        return _Fallback()


def _anomaly_vs_prior(
    response: np.ndarray,
    priors: List[np.ndarray],
) -> float:
    if not priors:
        return 0.5
    mean = np.mean(np.stack(priors, axis=0), axis=0)
    mae = float(np.mean(np.abs(response - mean)))
    return float(min(1.0, mae * 3.0))


def suggest_next_laser(
    entries: List[FieldMemoryEntry],
    n_lasers: int,
    *,
    last_laser: Optional[int] = None,
) -> Dict[str, Any]:
    """Curiosity + exploration: prefer unexplored lasers, then anomaly scores."""
    seen: Set[int] = set()
    for e in entries:
        lid = None
        if e.extras and "laser_id" in e.extras:
            try:
                lid = int(e.extras["laser_id"])
            except (TypeError, ValueError):
                pass
        if lid is None and e.excitation_id is not None:
            lid = int(e.excitation_id)
        if lid is not None and 0 <= lid < n_lasers:
            seen.add(lid)

    unexplored = [L for L in range(n_lasers) if L not in seen]
    if unexplored:
        candidates = [L for L in unexplored if L != last_laser] or unexplored
        choice = candidates[0]
        return {
            "action": "excite",
            "source_id": choice,
            "curiosity_score": 1.0,
            "reason": "unexplored laser",
            "unexplored": unexplored,
        }

    base = suggest_next(entries, exclude=last_laser)
    if base is not None:
        return base
    nxt = 0 if last_laser is None else (last_laser + 1) % n_lasers
    return {
        "action": "excite",
        "source_id": nxt,
        "curiosity_score": 0.0,
        "reason": "cycle",
    }


def run_loop(
    steps: int = 20,
    n_lasers: int = 12,
    n_detectors: int = 20,
    seed: int = 0,
    noise_rms: float = 0.01,
    drive_level: float = 0.85,
) -> Dict[str, Any]:
    body = _make_body(n_lasers, n_detectors, seed, noise_rms)
    store = FieldMemoryStore(soft_capacity=512)
    history: List[Dict[str, Any]] = []
    priors_by_laser: Dict[int, List[np.ndarray]] = {L: [] for L in range(n_lasers)}
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
                "curiosity_score": 1.0,
                "reason": "bootstrap",
            }

        rec = body.excite(next_laser, drive_level=drive_level)
        response = np.asarray(rec.detector_response, dtype=np.float32)
        anom = _anomaly_vs_prior(response, priors_by_laser[next_laser])
        priors_by_laser[next_laser].append(response)

        conf = 0.95 if float(np.max(response)) > 0.05 else 0.55
        entry = FieldMemoryEntry(
            body_id="optical-body-sim",
            excitation_id=next_laser,
            observed_response=response.tolist(),
            confidence=conf,
            anomaly=anom,
            extras={
                "laser_id": next_laser,
                "drive_level": float(drive_level),
                "n_detectors": int(n_detectors),
                "geometry_state": getattr(rec, "geometry_state", "nominal"),
            },
        )
        store.add(entry)
        last_laser = next_laser

        history.append(
            {
                "step": step,
                "laser_id": next_laser,
                "anomaly": round(anom, 4),
                "confidence": conf,
                "peak_response": round(float(np.max(response)), 4),
                "reason": suggestion.get("reason"),
            }
        )

    T = body.estimate_transfer_matrix(drive_level=1.0)
    entries = [item["entry"] for item in store.buffer]
    scores = score_sources(entries)

    return {
        "steps": steps,
        "n_lasers": n_lasers,
        "n_detectors": n_detectors,
        "store_size": len(store),
        "lasers_seen": sorted({h["laser_id"] for h in history}),
        "coverage": len({h["laser_id"] for h in history}) / max(n_lasers, 1),
        "curiosity_scores": {str(k): round(v, 4) for k, v in scores.items()},
        "transfer_matrix_shape": list(T.shape),
        "transfer_matrix_frobenius": float(np.linalg.norm(T)),
        "history": history,
        "store_stats": store.get_stats(),
        "final_suggestion": suggest_next_laser(entries, n_lasers, last_laser=last_laser),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optical body closed loop (simulator -> FieldMemoryStore -> probe)"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lasers", type=int, default=12)
    parser.add_argument("--detectors", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-rms", type=float, default=0.01)
    parser.add_argument("--drive", type=float, default=0.85)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_loop(
        steps=args.steps,
        n_lasers=args.lasers,
        n_detectors=args.detectors,
        seed=args.seed,
        noise_rms=args.noise_rms,
        drive_level=args.drive,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"[optical_body] steps={report['steps']} "
            f"coverage={report['coverage']:.0%} "
            f"lasers_seen={report['lasers_seen']}"
        )
        for h in report["history"][:12]:
            print(
                f"  step {h['step']:02d}: laser={h['laser_id']:02d} "
                f"anom={h['anomaly']:.3f} peak={h['peak_response']:.3f} "
                f"({h['reason']})"
            )
        if len(report["history"]) > 12:
            print(f"  ... ({len(report['history']) - 12} more steps)")
        fs = report["final_suggestion"]
        print(
            f"[active_probe] next laser={fs['source_id']} "
            f"score={fs.get('curiosity_score')} reason={fs.get('reason')}"
        )
        print(
            f"[transfer] shape={report['transfer_matrix_shape']} "
            f"||T||_F={report['transfer_matrix_frobenius']:.3f}"
        )


if __name__ == "__main__":
    main()
