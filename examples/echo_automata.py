#!/usr/bin/env python3
"""
echo_automata.py — Multi-level field automata

Level 0 (atoms)
  residual_high, fuse_agreed, has_tracks, high_motion, quiet

Level 1 (AND over atoms)
  SURPRISE_CONFIRMED   = residual_high ∧ fuse_agreed
  TRACKED_SURPRISE     = residual_high ∧ has_tracks
  HIGH_MOTION_SURPRISE = residual_high ∧ high_motion
  QUIET_ANOMALY        = residual_high ∧ quiet

Level 2 (composition of L1)
  CONFIRMED_TRACK = SURPRISE_CONFIRMED ∧ TRACKED_SURPRISE
  ACTIVE_FIELD    = HIGH_MOTION_SURPRISE ∧ (SURPRISE_CONFIRMED ∨ TRACKED_SURPRISE)
  STEALTH_BREAK   = QUIET_ANOMALY ∧ has_tracks
  FULL_LOCK       = CONFIRMED_TRACK ∧ HIGH_MOTION_SURPRISE

Usage:

  python examples/echo_automata.py
  python examples/echo_automata.py --follow --threshold 0.30
  python examples/echo_automata.py --follow --min-level 2
  python examples/echo_automata.py --no-follow   # score existing file once
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Set

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize_argv(argv: List[str]) -> List[str]:
    """Map unicode dashes (— – −) to ASCII '-' so pasted flags work."""
    out = []
    for a in argv:
        for ch in ("\u2014", "\u2013", "\u2212"):  # em, en, minus
            a = a.replace(ch, "-")
        out.append(a)
    return out


def _load_predictor():
    path = ROOT / "examples" / "echo_field_predictor.py"
    spec = importlib.util.spec_from_file_location("echo_field_predictor", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_model(path: Path, pred_mod):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = pred_mod.TinyFieldHead(in_dim=int(ckpt["in_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    window = int(ckpt.get("window", 8))
    print(f"[automata] head {path}  window={window}")
    return model, window


def follow_jsonl(path: Path, poll_s: float = 0.2):
    print(f"[automata] waiting for {path} …")
    while not path.exists():
        time.sleep(poll_s)
    print(f"[automata] following {path}")
    with path.open(encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:
                time.sleep(poll_s)
                try:
                    if path.stat().st_size < fh.tell():
                        fh.seek(0)
                except OSError:
                    pass
                continue
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def region_map(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for r in obj.get("field_regions") or []:
        out[str(r.get("region", ""))] = r
    return out


def eval_atoms(
    abs_r: float,
    threshold: float,
    motion: float,
    fuse_agreed: bool,
    n_tracks: int,
) -> Dict[str, bool]:
    return {
        "residual_high": abs_r >= threshold,
        "fuse_agreed": bool(fuse_agreed),
        "has_tracks": n_tracks >= 1,
        "high_motion": motion >= 0.6,
        "quiet": motion < 0.25,
    }


def eval_l1(atoms: Dict[str, bool]) -> List[str]:
    if not atoms["residual_high"]:
        return []
    gates = []
    if atoms["fuse_agreed"]:
        gates.append("SURPRISE_CONFIRMED")
    if atoms["has_tracks"]:
        gates.append("TRACKED_SURPRISE")
    if atoms["high_motion"]:
        gates.append("HIGH_MOTION_SURPRISE")
    if atoms["quiet"]:
        gates.append("QUIET_ANOMALY")
    if not gates:
        gates.append("SURPRISE_RAW")
    return gates


def eval_l2(l1: Set[str], atoms: Dict[str, bool]) -> List[str]:
    gates = []
    if "SURPRISE_CONFIRMED" in l1 and "TRACKED_SURPRISE" in l1:
        gates.append("CONFIRMED_TRACK")
    if "HIGH_MOTION_SURPRISE" in l1 and (
        "SURPRISE_CONFIRMED" in l1 or "TRACKED_SURPRISE" in l1
    ):
        gates.append("ACTIVE_FIELD")
    if "QUIET_ANOMALY" in l1 and atoms["has_tracks"]:
        gates.append("STEALTH_BREAK")
    if "CONFIRMED_TRACK" in gates and "HIGH_MOTION_SURPRISE" in l1:
        gates.append("FULL_LOCK")
    if "FULL_LOCK" in gates:
        for g in ("CONFIRMED_TRACK", "ACTIVE_FIELD"):
            if g not in gates:
                gates.append(g)
    return gates


def main() -> None:
    sys.argv = [sys.argv[0]] + _normalize_argv(sys.argv[1:])

    p = argparse.ArgumentParser(description="Multi-level Echo field automata")
    p.add_argument("--file", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--model", type=Path, default=Path("/tmp/metafield/echo_head.pt"))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--events", type=Path, default=Path("/tmp/metafield/echo_events.jsonl"))
    p.add_argument(
        "--min-level",
        type=int,
        default=1,
        choices=(1, 2),
        help="Only emit events that reach at least this gate level (1=L1, 2=L2)",
    )
    p.add_argument(
        "--follow",
        action="store_true",
        default=True,
        help="Tail the echo JSONL (default)",
    )
    p.add_argument(
        "--no-follow",
        action="store_true",
        help="Score existing file once, then exit",
    )
    args = p.parse_args()
    follow = not args.no_follow

    if not args.model.exists():
        print(f"[automata] missing model: {args.model}", file=sys.stderr)
        print(
            "  train: python examples/echo_field_predictor.py "
            "--file /tmp/metafield/echo.jsonl --save-model /tmp/metafield/echo_head.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    pred_mod = _load_predictor()
    model, window = load_model(args.model, pred_mod)
    history: Deque[List[float]] = deque(maxlen=window)

    args.events.parent.mkdir(parents=True, exist_ok=True)
    if not args.events.exists():
        args.events.touch()
    ev_fh = args.events.open("a", encoding="utf-8")

    frames = 0
    events = 0
    print(f"[automata] threshold={args.threshold:.3f}  min_level={args.min_level}  events→{args.events}")
    print("[automata] L1: SURPRISE_CONFIRMED | TRACKED_SURPRISE | HIGH_MOTION_SURPRISE | QUIET_ANOMALY")
    print("[automata] L2: CONFIRMED_TRACK | ACTIVE_FIELD | STEALTH_BREAK | FULL_LOCK")

    def handle(obj: Dict[str, Any]) -> None:
        nonlocal frames, events
        feats = pred_mod.features_from_observation(obj)
        if feats is None:
            return

        if len(history) < window:
            history.append(feats)
            frames += 1
            return

        x = torch.tensor([[v for row in history for v in row]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x)[0]
        pred_m = float(pred[0])
        actual_m = float(feats[0])
        abs_r = abs(actual_m - pred_m)
        history.append(feats)
        frames += 1

        by = region_map(obj)
        fuse = by.get("fuse") or {}
        fuse_agreed = bool((fuse.get("extras") or {}).get("agreed", False))
        n_tracks = sum(1 for k in by if k.startswith("track_"))
        motion = actual_m

        atoms = eval_atoms(abs_r, args.threshold, motion, fuse_agreed, n_tracks)
        l1 = eval_l1(atoms)
        l2 = eval_l2(set(l1), atoms)

        if args.min_level >= 2 and not l2:
            if frames % 50 == 0:
                print(f"[automata] frames={frames} events={events} |r|={abs_r:.3f} (no L2)")
            return
        if args.min_level < 2 and not l1:
            if frames % 50 == 0:
                print(f"[automata] frames={frames} events={events} |r|={abs_r:.3f}")
            return

        level = 2 if l2 else 1
        gates = list(dict.fromkeys(l2 + l1))

        event = {
            "schema_version": 2,
            "kind": "field_automata_event",
            "level": level,
            "gates": gates,
            "l1": l1,
            "l2": l2,
            "atoms": atoms,
            "body_id": obj.get("body_id", "echo-grid-01"),
            "abs_residual_motion": abs_r,
            "pred_motion": pred_m,
            "actual_motion": actual_m,
            "motion": motion,
            "fuse_agreed": fuse_agreed,
            "n_tracks": n_tracks,
            "threshold": args.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "excitation_id": obj.get("excitation_id"),
        }
        ev_fh.write(json.dumps(event) + "\n")
        ev_fh.flush()
        events += 1

        shown = l2 if l2 else l1
        print(
            f"[GATE L{level}] #{events}  {','.join(shown)}  "
            f"|r|={abs_r:.3f}  motion={motion:.2f}  "
            f"fuse={'Y' if fuse_agreed else 'n'}  tracks={n_tracks}"
        )

    try:
        if not follow:
            if not args.file.exists():
                print(f"[automata] missing {args.file}", file=sys.stderr)
                sys.exit(1)
            with args.file.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            handle(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        else:
            for obj in follow_jsonl(args.file):
                handle(obj)
    except KeyboardInterrupt:
        print("\n[automata] stopped")
    finally:
        ev_fh.close()
        print(f"[automata] frames={frames}  events={events}  → {args.events}")


if __name__ == "__main__":
    main()
