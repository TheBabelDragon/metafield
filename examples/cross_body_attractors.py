#!/usr/bin/env python3
"""
examples/cross_body_attractors.py

Tiny experiment: treat ZVS resonant health as another sense modality
and look for joint structure with the optical body.

What it does
------------
1. Run the optical Phase-0 stub → synthetic laser/detector responses.
2. Run the ZVS Phase-0 stub → synthetic power-stage telemetry
   (current, voltage, MOSFET/flyback temps, resonant quality, E-STOP).
3. Align the two streams by excitation / cycle index.
4. Build joint feature vectors:
      [optical mean intensity, optical mean anomaly, optical mean confidence,
       zvs bus_current, zvs t_mosfet, zvs resonant_quality, zvs e_stop]
5. Feed the joint vectors into AttractorDynamics.
6. Print whether attractors form, and a simple correlation snapshot
   (does higher ZVS thermal stress travel with higher optical anomaly?).

This is deliberately lightweight. It is not a full geometry + HMC run;
it is a first “does the shared schema + attractors already see
cross-body structure?” smoke experiment.

Usage
-----
  python examples/cross_body_attractors.py
  python examples/cross_body_attractors.py --optical-excitations 16 --zvs-cycles 24 --stress 0.35
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_body_stub import run_passive_sequence
from zvs_body_stub import run_cycle_sequence
from attractors import AttractorDynamics
from schemas.field_observation import FieldObservation


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def optical_summary(obs: FieldObservation) -> Tuple[float, float, float]:
    """mean intensity, mean anomaly, mean confidence across detectors."""
    intens, anoms, confs = [], [], []
    for r in obs.field_regions:
        if r.observed is not None:
            intens.append(float(r.observed))
        anoms.append(float(r.anomaly))
        confs.append(float(r.confidence))
    mean_i = sum(intens) / len(intens) if intens else 0.0
    mean_a = sum(anoms) / len(anoms) if anoms else 0.0
    mean_c = sum(confs) / len(confs) if confs else 0.0
    return mean_i, mean_a, mean_c


def zvs_summary(obs: FieldObservation) -> Tuple[float, float, float, float]:
    """
    Pull the key health channels that the stub already normalises
    into field_regions (bus_current, t_mosfet, resonant_quality, e_stop).
    Fall back to modality["zvs"]["raw"] if present.
    """
    by_name = {r.region: r for r in obs.field_regions}

    def get(name: str, default: float = 0.0) -> float:
        r = by_name.get(name)
        if r is not None and r.observed is not None:
            return float(r.observed)
        # try raw
        raw = (obs.modality or {}).get("zvs", {}).get("raw", {})
        if name == "bus_current" and "bus_current" in raw:
            return min(1.0, float(raw["bus_current"]) / 6.0)
        if name == "t_mosfet" and "t_mosfet" in raw:
            return min(1.0, (float(raw["t_mosfet"]) - 20.0) / 70.0)
        if name == "resonant_quality" and "resonant_quality" in raw:
            return float(raw["resonant_quality"])
        if name == "e_stop" and "e_stop_ok" in raw:
            return 1.0 if raw["e_stop_ok"] else 0.0
        return default

    return (
        get("bus_current"),
        get("t_mosfet"),
        get("resonant_quality"),
        get("e_stop"),
    )


def joint_vector(opt: FieldObservation, zvs: FieldObservation) -> torch.Tensor:
    oi, oa, oc = optical_summary(opt)
    zi, zt, zq, ze = zvs_summary(zvs)
    return torch.tensor([oi, oa, oc, zi, zt, zq, ze], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Simple correlation helper (no scipy)
# ---------------------------------------------------------------------------

def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx < 1e-12 or deny < 1e-12:
        return 0.0
    return num / (denx * deny)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-body attractor smoke experiment")
    parser.add_argument("--optical-excitations", type=int, default=16)
    parser.add_argument("--zvs-cycles", type=int, default=24)
    parser.add_argument("--stress", type=float, default=0.25,
                        help="Base ZVS thermal/current stress 0..1")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    opt_log = Path("/tmp/metafield/cross_optical.jsonl")
    zvs_log = Path("/tmp/metafield/cross_zvs.jsonl")
    for p in (opt_log, zvs_log):
        if p.exists():
            p.unlink()

    print("=== 1. Optical Phase-0 stub ===")
    optical_obs = run_passive_sequence(
        n_excitations=args.optical_excitations,
        body_id="optical-cross-01",
        log_path=opt_log,
        seed=args.seed,
    )

    print("\n=== 2. ZVS Phase-0 stub ===")
    zvs_obs = run_cycle_sequence(
        n_cycles=args.zvs_cycles,
        body_id="zvs-cross-01",
        log_path=zvs_log,
        seed=args.seed + 1,
        stress=args.stress,
    )

    # Align by index (truncate to the shorter stream)
    n = min(len(optical_obs), len(zvs_obs))
    optical_obs = optical_obs[:n]
    zvs_obs = zvs_obs[:n]
    print(f"\n=== 3. Aligned {n} joint observations ===")

    # Build joint vectors + collect series for correlation
    vectors: List[torch.Tensor] = []
    opt_anomaly: List[float] = []
    zvs_temp: List[float] = []
    zvs_quality: List[float] = []
    opt_conf: List[float] = []

    for o, z in zip(optical_obs, zvs_obs):
        v = joint_vector(o, z)
        vectors.append(v)
        _, oa, oc = optical_summary(o)
        _, zt, zq, _ = zvs_summary(z)
        opt_anomaly.append(oa)
        opt_conf.append(oc)
        zvs_temp.append(zt)
        zvs_quality.append(zq)

    print("\n=== 4. Correlation snapshot ===")
    r_temp_anom = pearson(zvs_temp, opt_anomaly)
    r_qual_conf = pearson(zvs_quality, opt_conf)
    r_temp_qual = pearson(zvs_temp, zvs_quality)
    print(f"    corr(ZVS t_mosfet, optical anomaly)     = {r_temp_anom:+.3f}")
    print(f"    corr(ZVS resonant_quality, opt conf)    = {r_qual_conf:+.3f}")
    print(f"    corr(ZVS t_mosfet, ZVS quality)         = {r_temp_qual:+.3f}")
    print("    (synthetic data — expect mild structure from the shared stress ramp)")

    print("\n=== 5. Joint attractor dynamics ===")
    dyn = AttractorDynamics(
        soft_attractor_target=12,
        hard_cap=20,
        energy_budget=30.0,
        merge_tolerance=0.25,
    )

    for i, v in enumerate(vectors):
        # interestingness: higher when optical anomaly or ZVS stress is elevated
        interest = 0.5 + 0.5 * (opt_anomaly[i] + (1.0 - zvs_quality[i])) / 2.0
        dyn.reinforce_from_latent(v, interestingness=interest)
        if (i + 1) % 4 == 0:
            dyn.step()

    # a few free evolution steps
    for _ in range(6):
        dyn.step()

    stats = dyn.get_stats()
    print(f"    num_attractors   = {stats['num_attractors']}")
    print(f"    avg_strength     = {stats['avg_strength']:.3f}")
    print(f"    max_strength     = {stats['max_strength']:.3f}")
    print(f"    avg_radius       = {stats['avg_radius']:.3f}")
    print(f"    avg_consistency  = {stats['avg_consistency']:.3f}")
    print(f"    total_energy     = {stats['total_energy']:.2f} / {stats['energy_budget']:.1f}")

    if stats["num_attractors"] > 0:
        print("\n    Top attractors (position ≈ [opt_I, opt_A, opt_C, zvs_I, zvs_T, zvs_Q, zvs_E]):")
        landscape = sorted(dyn.get_landscape(), key=lambda x: x[1], reverse=True)
        for i, (pos, strength) in enumerate(landscape[:5]):
            p = [f"{x:.2f}" for x in pos.tolist()]
            print(f"      [{i}] strength={strength:.2f}  pos=[{', '.join(p)}]")

    print("\n=== Done ===")
    print("Joint optical + ZVS features successfully drove attractor dynamics.")
    print("Next real step: replace the stubs with live serial/CAN streams.")


if __name__ == "__main__":
    main()
