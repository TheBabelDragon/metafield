#!/usr/bin/env python3
"""
zvs_body_stub.py

Phase 0 stub for the ZVS resonant / HV power node.

Simulates a healthy (or mildly stressed) isolated power stage and emits
FieldObservation packets that MetaField can already consume.

This lets the intelligence layer develop against realistic telemetry
(current, voltage, temperatures, E-STOP state) while the real ESP32-S3
TWAI + ADM3053 firmware is being written.

See PHYSICAL_FIELD_SUBSTRATE.md and the private zvs-node repo.
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
    zvs_observation,
    validate_observation,
)


# ---------------------------------------------------------------------------
# Synthetic power-stage model
# ---------------------------------------------------------------------------

def _synthetic_telemetry(cycle: int, seed: int, stress: float = 0.0) -> dict:
    """
    Deterministic-ish power-stage health.
    stress ∈ [0, 1] gently raises temperatures and current noise.
    """
    rng = random.Random((seed * 10007 + cycle * 97) % (2**31))

    # Base values for a healthy 48 V ZVS node under light load
    bus_v = 47.8 + (rng.random() - 0.5) * 0.4
    bus_i = 1.2 + stress * 2.5 + (rng.random() - 0.5) * 0.15
    t_mosfet = 38.0 + stress * 25.0 + (rng.random() - 0.5) * 2.0
    t_flyback = 42.0 + stress * 30.0 + (rng.random() - 0.5) * 2.5
    t_ambient = 28.0 + (rng.random() - 0.5) * 1.5

    # Simple resonant “quality” proxy (higher is better)
    quality = max(0.0, min(1.0, 0.92 - stress * 0.35 + (rng.random() - 0.5) * 0.04))

    e_stop_ok = True if stress < 0.85 else (rng.random() > 0.3)

    return {
        "bus_voltage": round(bus_v, 2),
        "bus_current": round(bus_i, 3),
        "t_mosfet": round(t_mosfet, 1),
        "t_flyback": round(t_flyback, 1),
        "t_ambient": round(t_ambient, 1),
        "resonant_quality": round(quality, 3),
        "e_stop_ok": e_stop_ok,
        "fan_pwm": min(1.0, 0.3 + stress * 0.6),
    }


def make_regions(telemetry: dict) -> List[FieldRegion]:
    """Map key telemetry into FieldRegion objects."""
    regions = []

    # Normalize a few channels into 0..1 “observed” for the shared schema
    # (MetaField cares about relative structure more than absolute units)

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))

    regions.append(FieldRegion(
        region="bus_current",
        observed=round(norm(telemetry["bus_current"], 0.0, 6.0), 4),
        confidence=0.95,
        anomaly=0.0 if telemetry["bus_current"] < 4.5 else 0.4,
        extras={"raw_A": telemetry["bus_current"]},
    ))
    regions.append(FieldRegion(
        region="bus_voltage",
        observed=round(norm(telemetry["bus_voltage"], 40.0, 52.0), 4),
        confidence=0.97,
        anomaly=0.0 if 46.0 < telemetry["bus_voltage"] < 50.0 else 0.3,
        extras={"raw_V": telemetry["bus_voltage"]},
    ))
    regions.append(FieldRegion(
        region="t_mosfet",
        observed=round(norm(telemetry["t_mosfet"], 20.0, 90.0), 4),
        confidence=0.9,
        anomaly=0.0 if telemetry["t_mosfet"] < 70.0 else 0.5,
        extras={"raw_C": telemetry["t_mosfet"]},
    ))
    regions.append(FieldRegion(
        region="t_flyback",
        observed=round(norm(telemetry["t_flyback"], 20.0, 100.0), 4),
        confidence=0.9,
        anomaly=0.0 if telemetry["t_flyback"] < 80.0 else 0.6,
        extras={"raw_C": telemetry["t_flyback"]},
    ))
    regions.append(FieldRegion(
        region="resonant_quality",
        observed=telemetry["resonant_quality"],
        confidence=0.85,
        anomaly=0.0 if telemetry["resonant_quality"] > 0.7 else 0.45,
    ))
    regions.append(FieldRegion(
        region="e_stop",
        observed=1.0 if telemetry["e_stop_ok"] else 0.0,
        confidence=1.0,
        anomaly=0.0 if telemetry["e_stop_ok"] else 1.0,
    ))

    return regions


# ---------------------------------------------------------------------------
# Passive loop + replay
# ---------------------------------------------------------------------------

def run_cycle_sequence(
    n_cycles: int = 20,
    body_id: str = "zvs-stub-01",
    log_path: Path = Path("/tmp/metafield/zvs_phase0.jsonl"),
    seed: int = 42,
    stress: float = 0.15,
    sleep_s: float = 0.0,
) -> List[FieldObservation]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    observations: List[FieldObservation] = []

    print(f"[zvs_stub] Phase 0 telemetry sequence — {n_cycles} cycles")
    print(f"[zvs_stub] body_id={body_id}  stress={stress:.2f}  log={log_path}")

    with log_path.open("a", encoding="utf-8") as fh:
        for i in range(n_cycles):
            # Gentle stress ramp for interest
            cycle_stress = min(1.0, stress + 0.01 * i)
            telem = _synthetic_telemetry(i, seed, cycle_stress)
            regions = make_regions(telem)

            obs = zvs_observation(
                body_id=body_id,
                excitation_id=i,
                regions=regions,
                geometry_state="calibrated",
                power_extras={
                    "note": "synthetic Phase-0 ZVS stub",
                    "raw": telem,
                },
            )

            # Mark health from E-STOP and temperature
            if not telem["e_stop_ok"]:
                obs.health = "error"
            elif telem["t_mosfet"] > 75 or telem["t_flyback"] > 85:
                obs.health = "partial"

            problems = validate_observation(obs)
            if problems:
                print(f"[zvs_stub] VALIDATION FAILED for cycle {i}: {problems}")
                obs.health = "error"

            fh.write(obs.to_json() + "\n")
            fh.flush()
            observations.append(obs)

            print(
                f"  cycle={i:03d}  I={telem['bus_current']:.2f}A  "
                f"Tmos={telem['t_mosfet']:.0f}C  Q={telem['resonant_quality']:.2f}  "
                f"E-STOP={'OK' if telem['e_stop_ok'] else 'OPEN'}  health={obs.health}"
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"[zvs_stub] wrote {len(observations)} observations → {log_path}")
    return observations


def replay_log(log_path: Path) -> List[FieldObservation]:
    if not log_path.exists():
        print(f"[zvs_stub] no log at {log_path}")
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

    print(f"[zvs_stub] replayed {len(observations)} observations from {log_path}")
    return observations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ZVS body Phase-0 stub")
    parser.add_argument("--cycles", type=int, default=20,
                        help="Number of telemetry cycles to simulate")
    parser.add_argument("--body-id", default="zvs-stub-01")
    parser.add_argument("--log", type=Path,
                        default=Path("/tmp/metafield/zvs_phase0.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stress", type=float, default=0.15,
                        help="Base thermal/current stress 0..1")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds between cycles")
    parser.add_argument("--replay-only", action="store_true",
                        help="Only replay an existing log")
    parser.add_argument("--clear-log", action="store_true",
                        help="Truncate the log before writing")
    args = parser.parse_args()

    if args.clear_log and args.log.exists():
        args.log.unlink()
        print(f"[zvs_stub] cleared {args.log}")

    if args.replay_only:
        replay_log(args.log)
        return

    run_cycle_sequence(
        n_cycles=args.cycles,
        body_id=args.body_id,
        log_path=args.log,
        seed=args.seed,
        stress=args.stress,
        sleep_s=args.sleep,
    )
    print()
    replay_log(args.log)


if __name__ == "__main__":
    main()
