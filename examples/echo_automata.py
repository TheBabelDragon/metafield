#!/usr/bin/env python3
"""
echo_automata.py — Field automata v0 (AND / threshold gates)

Reads live Echo FieldObservation JSONL, scores residual with the trained
head, evaluates simple gates, writes control events.

Gates (v0):

  SURPRISE_CONFIRMED  = |r| >= T  AND  fuse.agreed
  TRACKED_SURPRISE    = |r| >= T  AND  n_tracks >= 1
  HIGH_MOTION_SURPRISE= |r| >= T  AND  motion >= 0.6
  QUIET_ANOMALY       = |r| >= T  AND  motion < 0.25

Usage:

  python examples/echo_automata.py --follow
  python examples/echo_automata.py --follow --threshold 0.30

See docs/ECHO_STACK.md for the full three-terminal setup.
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
from typing import Any, Deque, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        name = str(r.get("region", ""))
        out[name] = r
    return out


def eval_gates(
    abs_r: float,
    threshold: float,
    motion: float,
    fuse_agreed: bool,
    n_tracks: int,
) -> List[str]:
    if abs_r < threshold:
        return []
    fired = []
    if fuse_agreed:
        fired.append("SURPRISE_CONFIRMED")
    if n_tracks >= 1:
        fired.append("TRACKED_SURPRISE")
    if motion >= 0.6:
        fired.append("HIGH_MOTION_SURPRISE")
    if motion < 0.25:
        fired.append("QUIET_ANOMALY")
    if not fired:
        fired.append("SURPRISE_RAW")
    return fired


def main() -> None:
    p = argparse.ArgumentParser(description="Echo field automata v0 (AND gates)")
    p.add_argument("--file", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--model", type=Path, default=Path("/tmp/metafield/echo_head.pt"))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--events", type=Path, default=Path("/tmp/metafield/echo_events.jsonl"))
    p.add_argument("--no-follow", action="store_true")
    args = p.parse_args()

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
    print(f"[automata] threshold={args.threshold:.3f}  events→{args.events}")
    print("[automata] gates: SURPRISE_CONFIRMED | TRACKED_SURPRISE | HIGH_MOTION_SURPRISE | QUIET_ANOMALY")

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

        gates = eval_gates(abs_r, args.threshold, motion, fuse_agreed, n_tracks)
        if not gates:
            if frames % 50 == 0:
                print(f"[automata] frames={frames} events={events} |r|={abs_r:.3f}")
            return

        event = {
            "schema_version": 1,
            "kind": "field_automata_event",
            "gates": gates,
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
        print(
            f"[GATE] #{events}  {','.join(gates)}  "
            f"|r|={abs_r:.3f}  motion={motion:.2f}  "
            f"fuse={'Y' if fuse_agreed else 'n'}  tracks={n_tracks}"
        )

    try:
        if args.no_follow:
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
