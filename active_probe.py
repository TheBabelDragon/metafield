#!/usr/bin/env python3
"""
active_probe.py

Closes the circle a little:

  observe → remember → decide what light would reveal the most → excite again

This is not full Phase-3 intelligence. It is a simple, honest curiosity heuristic:

  Prefer sources / regions with high anomaly or low confidence in recent
  FieldMemoryEntry samples. Emit an excitation *request* the body can run.

Usage:
  python active_probe.py --from-store /tmp/metafield/field_memory_smoke.jsonl
  python active_probe.py --from-store ... --emit-command

The body (optical-body-s3) can accept:
  EXCITE <laser_id>
on Serial (see firmware command path).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry


def score_sources(entries: List[FieldMemoryEntry]) -> Dict[int, float]:
    """
    Aggregate curiosity score per excitation source.
    Higher = more worth illuminating next.

    score ≈ mean(anomaly) + 0.5 * (1 - mean(confidence))
    """
    buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for e in entries:
        if e.excitation_id is None:
            continue
        # excitation_id may be a counter; prefer modality laser if present in extras
        src = e.excitation_id
        if e.extras and "laser_id" in e.extras:
            try:
                src = int(e.extras["laser_id"])
            except (TypeError, ValueError):
                pass
        buckets[int(src)].append((float(e.anomaly), float(e.confidence)))

    scores: Dict[int, float] = {}
    for src, pairs in buckets.items():
        if not pairs:
            continue
        mean_anom = sum(a for a, _ in pairs) / len(pairs)
        mean_conf = sum(c for _, c in pairs) / len(pairs)
        scores[src] = mean_anom + 0.5 * (1.0 - mean_conf)
    return scores


def suggest_next(entries: List[FieldMemoryEntry], exclude: Optional[int] = None) -> Optional[Dict[str, Any]]:
    scores = score_sources(entries)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for src, score in ranked:
        if exclude is not None and src == exclude:
            continue
        return {
            "action": "excite",
            "source_id": src,
            "curiosity_score": round(score, 4),
            "reason": "high anomaly / low confidence in recent field memory",
            "ranked": [{"source_id": s, "score": round(sc, 4)} for s, sc in ranked[:5]],
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest next optical excitation (active probe)")
    parser.add_argument("--from-store", type=Path, required=True, help="FieldMemoryEntry JSONL")
    parser.add_argument("--exclude", type=int, default=None, help="Skip this source_id")
    parser.add_argument("--emit-command", action="store_true",
                        help="Print firmware Serial command: EXCITE <id>")
    args = parser.parse_args()

    store = FieldMemoryStore()
    n = store.load_jsonl(args.from_store)
    if n == 0:
        print("No entries loaded.")
        return

    entries = [item["entry"] for item in store.buffer]
    suggestion = suggest_next(entries, exclude=args.exclude)
    if not suggestion:
        print("No scored sources — need observations with excitation_id / anomaly.")
        return

    print(json.dumps(suggestion, indent=2))
    if args.emit_command:
        print(f"EXCITE {suggestion['source_id']}")


if __name__ == "__main__":
    main()
