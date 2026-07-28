#!/usr/bin/env python3
"""
examples/optical_memory_smoke.py

End-to-end Phase-0 smoke test (no hardware required):

  1. Run the synthetic optical body stub → JSONL observations
  2. Promote each packet to FieldMemoryEntry
  3. Load into FieldMemoryStore
  4. Print stats + a small prioritized sample

Usage:
  python examples/optical_memory_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_body_stub import run_passive_sequence
from optical_serial_consumer import process_packet
from field_memory_store import FieldMemoryStore


def main() -> None:
    log = Path("/tmp/metafield/optical_smoke.jsonl")
    if log.exists():
        log.unlink()

    print("=== 1. Synthetic passive sequence ===")
    observations = run_passive_sequence(
        n_excitations=12,
        body_id="optical-smoke-01",
        log_path=log,
        seed=7,
    )
    print(f"    wrote {len(observations)} FieldObservation packets → {log}")

    print("\n=== 2. Promote → FieldMemoryEntry + FieldMemoryStore ===")
    store = FieldMemoryStore(soft_capacity=256)
    for obs in observations:
        entry = process_packet(obs.to_dict())
        if entry is not None:
            store.add(entry)

    stats = store.get_stats()
    print(f"    store size          = {stats['size']}")
    print(f"    total_added         = {stats['total_added']}")
    print(f"    avg_priority        = {stats['avg_priority']:.3f}")
    print(f"    avg_anomaly         = {stats['avg_anomaly']:.3f}")
    print(f"    avg_confidence      = {stats['avg_confidence']:.3f}")

    print("\n=== 3. Prioritized sample (n=4) ===")
    sample = store.sample(4)
    for i, e in enumerate(sample):
        print(
            f"    [{i}] body={e.body_id}  exc={e.excitation_id}  "
            f"conf={e.confidence:.2f}  anom={e.anomaly:.3f}"
        )

    out = Path("/tmp/metafield/field_memory_smoke.jsonl")
    n = store.save_jsonl(out)
    print(f"\n=== 4. Saved {n} entries → {out} ===")
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
