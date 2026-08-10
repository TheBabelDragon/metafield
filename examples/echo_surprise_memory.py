#!/usr/bin/env python3
"""
echo_surprise_memory.py

Turn motion-prediction residuals into MetaField episodic memory.

High |residual_motion| = field behavior the head did not expect.
Those frames become FieldMemoryEntry rows with anomaly set from the residual.

Usage (after echo_field_predictor --score):

  python examples/echo_surprise_memory.py \
    --echo /tmp/metafield/echo.jsonl \
    --residuals /tmp/metafield/echo_residuals.jsonl \
    --save /tmp/metafield/echo_surprise_memory.jsonl \
    --threshold 0.30

If --residuals is omitted, searches /tmp/metafield for *residual*.jsonl
(including common typos like echo_risiduals.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schemas.field_memory import FieldMemoryEntry
from schemas.field_observation import FieldObservation, validate_observation


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[surprise] skip {path.name}:{line_no}: {e}", file=sys.stderr)
    return rows


def find_residual_file(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        print(f"[surprise] not found: {explicit}", file=sys.stderr)

    search_dirs = [Path("/tmp/metafield"), Path.cwd(), ROOT / "tmp"]
    candidates: List[Path] = []
    patterns = ("*residual*.jsonl", "*risidual*.jsonl", "*resid*.jsonl")
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pat in patterns:
            candidates.extend(sorted(d.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True))

    # de-dupe preserving order
    seen = set()
    unique = []
    for c in candidates:
        rp = c.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(c)

    if not unique:
        print("[surprise] no residual JSONL found under /tmp/metafield", file=sys.stderr)
        print("  expected something like:", file=sys.stderr)
        print("    /tmp/metafield/echo_residuals.jsonl", file=sys.stderr)
        print("  recreate with:", file=sys.stderr)
        print(
            "    python examples/echo_field_predictor.py "
            "--file /tmp/metafield/echo.jsonl "
            "--load-model /tmp/metafield/echo_head.pt "
            "--score /tmp/metafield/echo_residuals.jsonl",
            file=sys.stderr,
        )
        sys.exit(1)

    chosen = unique[0]
    if len(unique) > 1:
        print(f"[surprise] multiple residual files; using newest: {chosen}")
        for u in unique[1:5]:
            print(f"            also saw: {u}")
    else:
        print(f"[surprise] using residuals: {chosen}")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Promote high residual Echo frames → FieldMemory")
    p.add_argument("--echo", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument(
        "--residuals",
        type=Path,
        default=None,
        help="Residual JSONL from --score (auto-find if omitted)",
    )
    p.add_argument("--save", type=Path, default=Path("/tmp/metafield/echo_surprise_memory.jsonl"))
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Min abs residual_motion to keep (default: residual p90)",
    )
    p.add_argument("--top", type=int, default=15, help="Print this many strongest surprises")
    args = p.parse_args()

    if not args.echo.exists():
        print(f"[surprise] missing echo log: {args.echo}", file=sys.stderr)
        sys.exit(1)

    residuals_path = find_residual_file(args.residuals)

    echo_rows = load_jsonl(args.echo)
    res_rows = load_jsonl(residuals_path)
    if not res_rows:
        print("[surprise] empty residuals", file=sys.stderr)
        sys.exit(1)

    # need abs_residual_motion keys
    if "abs_residual_motion" not in res_rows[0]:
        print(
            f"[surprise] {residuals_path} does not look like a score file "
            f"(missing abs_residual_motion)",
            file=sys.stderr,
        )
        sys.exit(1)

    abs_vals = sorted(float(r.get("abs_residual_motion", 0.0)) for r in res_rows)
    p90 = abs_vals[int(len(abs_vals) * 0.9)] if abs_vals else 0.3
    threshold = float(args.threshold) if args.threshold is not None else p90
    print(
        f"[surprise] echo={len(echo_rows)} residual_rows={len(res_rows)} "
        f"threshold={threshold:.4f} (p90={p90:.4f})"
    )

    surprises: List[tuple] = []
    args.save.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.save.open("w", encoding="utf-8") as out:
        for r in res_rows:
            abs_r = float(r.get("abs_residual_motion", 0.0))
            if abs_r < threshold:
                continue
            idx = int(r.get("i", -1))
            if idx < 0 or idx >= len(echo_rows):
                continue
            obs_dict = echo_rows[idx]
            try:
                obs = FieldObservation.from_dict(obs_dict)
            except Exception as e:
                print(f"[surprise] bad obs i={idx}: {e}", file=sys.stderr)
                continue
            problems = validate_observation(obs)
            if problems:
                print(f"[surprise] validation i={idx}: {problems}", file=sys.stderr)

            anomaly = min(1.0, abs_r)
            entry = FieldMemoryEntry.from_observation(obs.to_dict())
            entry.anomaly = anomaly
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
            written += 1
            surprises.append((abs_r, idx, r.get("actual_motion"), r.get("pred_motion")))

    surprises.sort(reverse=True)
    print(f"[surprise] wrote {written} FieldMemoryEntry → {args.save}")
    print()
    print(f"=== top {min(args.top, len(surprises))} surprises ===")
    for abs_r, idx, actual, pred in surprises[: args.top]:
        print(f"  i={idx:4d}  |r|={abs_r:.3f}  actual={actual}  pred={pred}")


if __name__ == "__main__":
    main()
