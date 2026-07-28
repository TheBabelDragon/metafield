#!/usr/bin/env python3
"""
optical_body_stub.py

Phase 0 (Passive observability) stub for the optical dodecahedral body.

This is intentionally boring and hardware-free.
It simulates firing lasers 0..N-1, produces FieldObservation packets,
writes a replayable JSONL log, and can re-play the exact sequence.

Use this to exercise the shared schema, logging, and Aurora-side
consumption while the real BPW34 + laser firmware is being built.

See PHYSICAL_FIELD_SUBSTRATE.md and issue #1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Optional

from schemas.field_observation import (
    FieldRegion,
    FieldObservation,
    optical_observation,
    validate_observation,
)


# ---------------------------------------------------------------------------
# Synthetic optical model (just enough to be interesting)
# ---------------------------------------------------------------------------

NUM_LASERS = 12          # one per face of a dodecahedron (approx)
NUM_DETECTORS = 20       # BPW34-style photodiodes
NOISE_FLOOR = 0.015


def _synthetic_response(laser_idx: int, detector_idx: int, seed: int) -> float:
    """
    Deterministic-ish response with a little structured coupling.
    Same (laser, detector, seed) always yields the same base value.
    """
    # Simple geometric-ish coupling: stronger when indices are "close"
    dist = abs(laser_idx - (detector_idx % NUM_LASERS))
    coupling = math.exp(-0.35 * dist)
    # Add a stable per-pair bias from the seed
    rng = random.Random((seed * 10007 + laser_idx * 97 + detector_idx) % (2**31))
    bias = 0.15 + 0.7 * rng.random()
    base = coupling * bias
    # Tiny noise (still seeded for replay)
    noise = (rng.random() - 0.5) * 2 * NOISE_FLOOR
    return max(0.0, min(1.0, base + noise))


def make_regions(laser_idx: int, seed: int) -> List[FieldRegion]:
    regions = []
    for d in range(NUM_DETECTORS):
        observed = _synthetic_response(laser_idx, d, seed)
        # In Phase 0 we have no "expected" yet — that comes with the transfer matrix
        regions.append(
            FieldRegion(
                region=f"detector_{d:02d}",
                observed=round(observed, 4),
                expected=None,
                confidence=0.95 if observed > NOISE_FLOOR * 3 else 0.6,
                anomaly=0.0,
            )
        )
    return regions


# ---------------------------------------------------------------------------
# Passive loop + replay
# ---------------------------------------------------------------------------

def run_passive_sequence(
    n_excitations: int = NUM_LASERS,
    body_id: str = "optical-stub-01",
    log_path: Path = Path("/tmp/metafield/optical_phase0.jsonl"),
    seed: int = 42,
    sleep_s: float = 0.0,
) -> List[FieldObservation]:
    """
    Fire lasers 0 .. n_excitations-1, emit FieldObservation packets,
    append them to a JSONL log that can be replayed exactly.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    observations: List[FieldObservation] = []

    print(f"[optical_stub] Phase 0 passive sequence — {n_excitations} excitations")
    print(f"[optical_stub] body_id={body_id}  log={log_path}")

    with log_path.open("a", encoding="utf-8") as fh:
        for i in range(n_excitations):
            laser = i % NUM_LASERS
            regions = make_regions(laser, seed)
            obs = optical_observation(
                body_id=body_id,
                excitation_id=i,
                regions=regions,
                geometry_state="uncalibrated",  # Phase 0 has no transfer map yet
                transfer_matrix_hint={
                    "laser": laser,
                    "num_detectors": NUM_DETECTORS,
                    "note": "synthetic Phase-0 stub",
                },
            )
            problems = validate_observation(obs)
            if problems:
                print(f"[optical_stub] VALIDATION FAILED for excitation {i}: {problems}")
                obs.health = "error"

            fh.write(obs.to_json() + "\n")
            fh.flush()
            observations.append(obs)

            n_active = sum(1 for r in regions if (r.observed or 0) > NOISE_FLOOR * 3)
            print(
                f"  excitation={i:03d} laser={laser:02d} "
                f"active_detectors≈{n_active}/{NUM_DETECTORS} "
                f"health={obs.health}"
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"[optical_stub] wrote {len(observations)} observations → {log_path}")
    return observations


def replay_log(log_path: Path) -> List[FieldObservation]:
    """Re-read a Phase-0 JSONL log and re-validate every packet."""
    if not log_path.exists():
        print(f"[optical_stub] no log at {log_path}")
        return []

    observations: List[FieldObservation] = []
    with log_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                obs = FieldObservation.from_dict(data)
                problems = validate_observation(obs)
                if problems:
                    print(f"[replay] line {line_no} problems: {problems}")
                observations.append(obs)
            except Exception as e:
                print(f"[replay] line {line_no} parse error: {e}")

    print(f"[optical_stub] replayed {len(observations)} observations from {log_path}")
    return observations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optical body Phase-0 stub")
    parser.add_argument("--excitations", type=int, default=NUM_LASERS,
                        help="Number of laser firings to simulate")
    parser.add_argument("--body-id", default="optical-stub-01")
    parser.add_argument("--log", type=Path,
                        default=Path("/tmp/metafield/optical_phase0.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds between excitations (for live demos)")
    parser.add_argument("--replay-only", action="store_true",
                        help="Only replay an existing log, do not generate new data")
    parser.add_argument("--clear-log", action="store_true",
                        help="Truncate the log before writing")
    args = parser.parse_args()

    if args.clear_log and args.log.exists():
        args.log.unlink()
        print(f"[optical_stub] cleared {args.log}")

    if args.replay_only:
        replay_log(args.log)
        return

    run_passive_sequence(
        n_excitations=args.excitations,
        body_id=args.body_id,
        log_path=args.log,
        seed=args.seed,
        sleep_s=args.sleep,
    )
    # Always show a quick replay confirmation
    print()
    replay_log(args.log)


if __name__ == "__main__":
    main()
