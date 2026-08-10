#!/usr/bin/env python3
"""
echo_close_loop.py

Finish the Echo → MetaField loop:

  residuals / surprise JSONL  →  FieldMemoryStore  →  prioritized sample

Uses the same FieldMemoryStore optical Phase-0 already uses. High residual
(anomaly) entries rank higher for replay.

Usage:

  python examples/echo_close_loop.py

  python examples/echo_close_loop.py \
    --surprise /tmp/metafield/echo_surprise_memory.jsonl \
    --also /tmp/metafield/field_memory.jsonl \
    --save-store /tmp/metafield/echo_store.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry
from schemas.field_observation import FieldObservation


def _load_surprise_mod():
    path = ROOT / "examples" / "echo_surprise_memory.py"
    spec = importlib.util.spec_from_file_location("echo_surprise_memory", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_entries(path: Path) -> List[FieldMemoryEntry]:
    if not path.exists():
        return []
    out: List[FieldMemoryEntry] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[loop] skip {path.name}:{line_no}: {e}", file=sys.stderr)
                continue
            out.append(
                FieldMemoryEntry(
                    body_id=data.get("body_id", "unknown"),
                    excitation_id=data.get("excitation_id"),
                    location=data.get("location"),
                    expected_response=data.get("expected_response"),
                    observed_response=data.get("observed_response"),
                    confidence=float(data.get("confidence", 0.0)),
                    anomaly=float(data.get("anomaly", 0.0)),
                    attractor_id=data.get("attractor_id"),
                    timestamp=data.get("timestamp", ""),
                    extras=data.get("extras") or {},
                )
            )
    return out


def ensure_surprise(args: argparse.Namespace) -> Path:
    surprise = Path(args.surprise)
    if surprise.exists() and surprise.stat().st_size > 0:
        return surprise

    print(f"[loop] no surprise file at {surprise}; building…")
    sm = _load_surprise_mod()

    echo = Path(args.echo)
    if not echo.exists():
        print(f"[loop] missing echo log: {echo}", file=sys.stderr)
        sys.exit(1)

    residuals_path = sm.find_residual_file(Path(args.residuals) if args.residuals else None)
    echo_rows = sm.load_jsonl(echo)
    res_rows = sm.load_jsonl(residuals_path)
    if not res_rows:
        print("[loop] empty residuals — run predictor --score first", file=sys.stderr)
        sys.exit(1)

    abs_vals = sorted(float(r.get("abs_residual_motion", 0.0)) for r in res_rows)
    p90 = abs_vals[int(len(abs_vals) * 0.9)] if abs_vals else 0.3
    threshold = float(args.threshold) if args.threshold is not None else p90

    surprise.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with surprise.open("w", encoding="utf-8") as out:
        for r in res_rows:
            abs_r = float(r.get("abs_residual_motion", 0.0))
            if abs_r < threshold:
                continue
            idx = int(r.get("i", -1))
            if idx < 0 or idx >= len(echo_rows):
                continue
            try:
                obs = FieldObservation.from_dict(echo_rows[idx])
            except Exception:
                continue
            entry = FieldMemoryEntry.from_observation(obs.to_dict())
            entry.anomaly = min(1.0, abs_r)
            entry.extras = dict(entry.extras or {})
            entry.extras.update({
                "surprise": True,
                "pred_motion": r.get("pred_motion"),
                "actual_motion": r.get("actual_motion"),
                "residual_motion": r.get("residual_motion"),
                "abs_residual_motion": abs_r,
                "source_index": idx,
            })
            out.write(entry.to_json() + "\n")
            n += 1
    print(f"[loop] built {n} surprise entries → {surprise} (threshold={threshold:.4f})")
    return surprise


def main() -> None:
    p = argparse.ArgumentParser(description="Echo surprise → FieldMemoryStore (close the loop)")
    p.add_argument("--echo", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--residuals", type=Path, default=None)
    p.add_argument(
        "--surprise",
        type=Path,
        default=Path("/tmp/metafield/echo_surprise_memory.jsonl"),
    )
    p.add_argument(
        "--also",
        type=Path,
        default=Path("/tmp/metafield/field_memory.jsonl"),
        help="Optional full consumer log to merge (lower anomaly bulk)",
    )
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--sample", type=int, default=8)
    p.add_argument(
        "--save-store",
        type=Path,
        default=Path("/tmp/metafield/echo_store.jsonl"),
    )
    p.add_argument("--capacity", type=int, default=2048)
    args = p.parse_args()

    surprise_path = ensure_surprise(args)
    surprise_entries = load_entries(surprise_path)
    bulk_entries: List[FieldMemoryEntry] = []
    if args.also and args.also.exists():
        bulk_entries = load_entries(args.also)

    store = FieldMemoryStore(soft_capacity=args.capacity)

    for e in bulk_entries:
        store.add(e)
    for e in surprise_entries:
        store.add(e)

    stats = store.get_stats()
    print()
    print("=== FieldMemoryStore ===")
    print(f"  size={stats['size']}  total_added={stats['total_added']}  capacity={stats['soft_capacity']}")
    print(f"  avg_priority={stats['avg_priority']:.3f}")
    print(f"  avg_anomaly={stats['avg_anomaly']:.3f}")
    print(f"  avg_confidence={stats['avg_confidence']:.3f}")
    print(f"  from_surprise={len(surprise_entries)}  from_bulk={len(bulk_entries)}")

    if stats["size"] == 0:
        print("[loop] store empty — no surprise/bulk entries found", file=sys.stderr)
        sys.exit(1)

    sampled = store.sample(n=args.sample)
    print()
    print(f"=== prioritized sample (n={len(sampled)}) ===")
    for i, e in enumerate(sampled, 1):
        surprise = bool((e.extras or {}).get("surprise"))
        abs_r = (e.extras or {}).get("abs_residual_motion")
        tag = "SURPRISE" if surprise else "bulk"
        extra = f"  |r|={abs_r:.3f}" if isinstance(abs_r, (int, float)) else ""
        print(
            f"  {i:2d}. [{tag}] body={e.body_id}  exc={e.excitation_id}  "
            f"anom={e.anomaly:.3f}  conf={e.confidence:.2f}{extra}"
        )

    n = store.save_jsonl(args.save_store)
    print()
    print(f"[loop] store saved ({n} entries) → {args.save_store}")
    print("[loop] closed.")


if __name__ == "__main__":
    main()
