"""
export.py — walk a MetaField JSONL run and emit axiom_notes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .mapping import (
    DEFAULT_TICKS_PER_STEP,
    map_record_to_notes,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _infer_ranges(records: Sequence[Dict[str, Any]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Auto-scale plaquette and color channels from the run itself."""
    plaquettes = [float(r.get("plaquette", 0.5)) for r in records if "plaquette" in r]
    colors: List[float] = []
    for r in records:
        cf = r.get("color_fields") or []
        colors.extend(float(c) for c in cf)

    def span(vals: List[float], fallback: Tuple[float, float]) -> Tuple[float, float]:
        if not vals:
            return fallback
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return (lo - 0.05, hi + 0.05)
        pad = 0.05 * (hi - lo)
        return (lo - pad, hi + pad)

    return span(plaquettes, (0.0, 1.0)), span(colors, (0.0, 1.0))


def export_jsonl_to_axiom(
    jsonl_path: str | Path,
    out_path: str | Path,
    *,
    ticks_per_step: int = DEFAULT_TICKS_PER_STEP,
    note_dur: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read MetaField observables JSONL → write AXIOM-native notes JSON.

    Returns the payload that was written (for testing / live reuse).
    """
    jsonl_path = Path(jsonl_path)
    out_path = Path(out_path)
    records = _load_jsonl(jsonl_path)
    if not records:
        payload = {"notes": []}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    plaq_range, color_range = _infer_ranges(records)
    dur = note_dur if note_dur is not None else max(2, ticks_per_step - 2)

    notes: List[Dict[str, Any]] = []
    tick = 0
    for i, rec in enumerate(records):
        step_notes = map_record_to_notes(
            rec,
            tick,
            ticks_per_step=ticks_per_step,
            plaquette_range=plaq_range,
            color_range=color_range,
            note_dur=dur,
            reset_topo_baseline=(i == 0),
        )
        notes.extend(step_notes)
        tick += ticks_per_step

    payload = {
        "notes": notes,
        "meta": {
            "source": str(jsonl_path),
            "n_trajectories": len(records),
            "ticks_per_step": ticks_per_step,
            "plaquette_range": list(plaq_range),
            "color_range": list(color_range),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="MetaField JSONL → AXIOM notes")
    p.add_argument("jsonl", type=Path, help="path to observables .jsonl")
    p.add_argument("-o", "--out", type=Path, default=Path("axiom_notes.json"))
    p.add_argument("--ticks-per-step", type=int, default=DEFAULT_TICKS_PER_STEP)
    p.add_argument("--note-dur", type=int, default=None)
    args = p.parse_args(argv)

    payload = export_jsonl_to_axiom(
        args.jsonl,
        args.out,
        ticks_per_step=args.ticks_per_step,
        note_dur=args.note_dur,
    )
    print(f"wrote {len(payload['notes'])} notes → {args.out}")


if __name__ == "__main__":
    main()
