#!/usr/bin/env python3
"""
wavebridge_closed_loop.py

Deterministic software reference loop (no laser, ADC, ESP32, serial, audio):

  MetaField state
        ↓
  wavebridge_adapter.field_state_to_wavebridge
        ↓
  FieldPacket → framed waveform
        ↓
  SimulatedChannel
        ↓
  extract packet → Observation
        ↓
  wavebridge_adapter → FieldMemoryStore
        ↓
  active_probe.suggest_next
        ↓
  next stimulus

Usage:
  PYTHONPATH=../wavebridge:. python examples/wavebridge_closed_loop.py
  PYTHONPATH=../wavebridge:. python examples/wavebridge_closed_loop.py --steps 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from field_memory_store import FieldMemoryStore
from active_probe import suggest_next
from wavebridge_adapter import (
    field_state_to_wavebridge,
    observation_to_memory_entry,
    wavebridge_observation_to_field,
)


def _require_wavebridge():
    try:
        from wavebridge import SimulatedChannel, packet_roundtrip
        from wavebridge.channel import ChannelSpec
    except ImportError as exc:
        raise SystemExit(
            "wavebridge required on PYTHONPATH "
            "(clone TheBabelDragon/wavebridge and export PYTHONPATH)"
        ) from exc
    return SimulatedChannel, packet_roundtrip, ChannelSpec


def make_stimulus(step: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Deterministic excitation vector for step index."""
    rng = np.random.default_rng(seed + step * 17)
    t = np.linspace(0, 1, dim, dtype=np.float32)
    base = 0.6 * np.sin(2 * np.pi * (3 + step % 5) * t)
    noise = 0.15 * rng.standard_normal(dim).astype(np.float32)
    return (base + noise).astype(np.float32)


def run_loop(
    steps: int = 6,
    dim: int = 32,
    body_id: str = "sim-optical-01",
    noise_rms: float = 0.0005,
    seed: int = 0,
) -> dict:
    SimulatedChannel, packet_roundtrip, ChannelSpec = _require_wavebridge()

    store = FieldMemoryStore(soft_capacity=256)
    channel = SimulatedChannel(
        spec=ChannelSpec(gain=0.98, noise_rms=noise_rms, seed=seed)
    )

    history = []
    for step in range(steps):
        state = make_stimulus(step, dim=dim, seed=seed)
        packet = field_state_to_wavebridge(
            state,
            source="metafield",
            metadata={"body_id": body_id, "excitation_id": step},
        )
        obs = packet_roundtrip(
            packet,
            channel,
            leading_silence=100 + 10 * step,
            trailing_silence=50,
        )
        field_dict = wavebridge_observation_to_field(
            obs,
            body_id=body_id,
            body_type="optical",
            excitation_id=step,
        )
        recovered = np.asarray(field_dict["observed_response"], dtype=np.float32)
        err = float(np.mean(np.abs(recovered - state)))
        field_dict["anomaly"] = min(1.0, err * 5.0)
        entry = observation_to_memory_entry(field_dict)
        store.add(entry)

        history.append(
            {
                "step": step,
                "excitation_id": step,
                "n_samples": int(recovered.size),
                "mae": err,
                "confidence": field_dict["confidence"],
                "anomaly": field_dict["anomaly"],
            }
        )

    entries = [item["entry"] for item in store.buffer]
    suggestion = suggest_next(entries)

    return {
        "steps": steps,
        "store_size": len(store),
        "store_stats": store.get_stats(),
        "history": history,
        "suggestion": suggestion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic WaveBridge ↔ MetaField loop")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--noise-rms", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args()

    report = run_loop(
        steps=args.steps,
        dim=args.dim,
        noise_rms=args.noise_rms,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[closed_loop] steps={report['steps']} store={report['store_size']}")
        for h in report["history"]:
            print(
                f"  step {h['step']}: mae={h['mae']:.5f} "
                f"conf={h['confidence']:.2f} anom={h['anomaly']:.4f}"
            )
        if report["suggestion"]:
            print(
                f"[active_probe] next source_id={report['suggestion']['source_id']} "
                f"score={report['suggestion']['curiosity_score']}"
            )
        else:
            print("[active_probe] no suggestion")


if __name__ == "__main__":
    main()
