#!/usr/bin/env python3
"""
echo_continuous.py

Live close of the Echo → MetaField loop.

Follows FieldObservation JSONL from Echo (--metafield-log), scores each new
frame with the trained motion head, and on high residual:

  • prints a SURPRISE line
  • appends FieldMemoryEntry to surprise JSONL
  • adds into FieldMemoryStore (priority by anomaly)
  • periodically checkpoints the store

Usage:

  python examples/echo_continuous.py

  python examples/echo_continuous.py \
    --model /tmp/metafield/echo_head.pt \
    --threshold 0.30 \
    --save-surprise /tmp/metafield/echo_surprise_memory.jsonl \
    --save-store /tmp/metafield/echo_store.jsonl

Ctrl+C stops and flushes the store.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry
from schemas.field_observation import FieldObservation


def _load_predictor():
    path = ROOT / "examples" / "echo_field_predictor.py"
    spec = importlib.util.spec_from_file_location("echo_field_predictor", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_model(path: Path, pred_mod):
    if not path.exists():
        print(f"[continuous] model not found: {path}", file=sys.stderr)
        print(
            "  train first:\n"
            "    python examples/echo_field_predictor.py "
            "--file /tmp/metafield/echo.jsonl "
            "--save-model /tmp/metafield/echo_head.pt",
            file=sys.stderr,
        )
        sys.exit(1)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    window = int(ckpt.get("window", 8))
    in_dim = int(ckpt["in_dim"])
    model = pred_mod.TinyFieldHead(in_dim=in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[continuous] loaded {path}  window={window}  in_dim={in_dim}")
    if "stats" in ckpt:
        s = ckpt["stats"]
        print(
            f"  ckpt val motion={s.get('val_mse_motion', float('nan')):.4f} "
            f"vs persistence={s.get('baseline_mse_motion', float('nan')):.4f}"
        )
    return model, window


def features_for(obj: Dict[str, Any], pred_mod) -> Optional[List[float]]:
    if "field_regions" in obj:
        return pred_mod.features_from_observation(obj)
    return pred_mod.features_from_memory_entry(obj)


def follow_jsonl(path: Path, poll_s: float = 0.2):
    print(f"[continuous] waiting for {path} …")
    while not path.exists():
        time.sleep(poll_s)
    print(f"[continuous] following {path}")
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
            if not text or not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def main() -> None:
    p = argparse.ArgumentParser(description="Live Echo residual → FieldMemoryStore")
    p.add_argument("--file", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--model", type=Path, default=Path("/tmp/metafield/echo_head.pt"))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument(
        "--save-surprise",
        type=Path,
        default=Path("/tmp/metafield/echo_surprise_memory.jsonl"),
    )
    p.add_argument(
        "--save-store",
        type=Path,
        default=Path("/tmp/metafield/echo_store.jsonl"),
    )
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--capacity", type=int, default=2048)
    p.add_argument("--no-follow", action="store_true", help="Score existing file once, then exit")
    args = p.parse_args()

    pred_mod = _load_predictor()
    model, window = load_model(args.model, pred_mod)

    store = FieldMemoryStore(soft_capacity=args.capacity)
    for warm in (args.save_store, args.save_surprise):
        if warm.exists():
            n = store.load_jsonl(warm)
            if n:
                print(f"[continuous] warm-loaded {n} from {warm}")

    args.save_surprise.parent.mkdir(parents=True, exist_ok=True)
    surprise_fh = args.save_surprise.open("a", encoding="utf-8")

    history: Deque[List[float]] = deque(maxlen=window)
    frames = 0
    surprises = 0

    print(f"[continuous] threshold={args.threshold:.3f}  store_capacity={args.capacity}")
    print("[continuous] running (Ctrl+C to stop)")

    def handle(obj: Dict[str, Any]) -> None:
        nonlocal frames, surprises
        feats = features_for(obj, pred_mod)
        if feats is None:
            return

        if len(history) < window:
            history.append(feats)
            frames += 1
            if frames % 10 == 0 or len(history) == window:
                print(f"[continuous] warming {len(history)}/{window}")
            return

        x = torch.tensor([[v for row in history for v in row]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x)[0]
        pred_m = float(pred[0])
        actual_m = float(feats[0])
        residual = actual_m - pred_m
        abs_r = abs(residual)

        frames += 1
        history.append(feats)

        if frames % 40 == 0:
            print(
                f"[continuous] frames={frames}  surprises={surprises}  "
                f"motion={actual_m:.3f} pred={pred_m:.3f} |r|={abs_r:.3f}  store={len(store)}"
            )

        if abs_r < args.threshold:
            return

        try:
            obs = FieldObservation.from_dict(obj)
            entry = FieldMemoryEntry.from_observation(obs.to_dict())
        except Exception as e:
            print(f"[continuous] obs error: {e}", file=sys.stderr)
            return

        entry.anomaly = min(1.0, abs_r)
        entry.extras = dict(entry.extras or {})
        entry.extras.update({
            "surprise": True,
            "pred_motion": pred_m,
            "actual_motion": actual_m,
            "residual_motion": residual,
            "abs_residual_motion": abs_r,
            "live": True,
        })

        store.add(entry)
        surprise_fh.write(entry.to_json() + "\n")
        surprise_fh.flush()
        surprises += 1

        print(
            f"[SURPRISE] #{surprises}  |r|={abs_r:.3f}  "
            f"actual={actual_m:.3f} pred={pred_m:.3f}  "
            f"conf={entry.confidence:.2f}  store={len(store)}"
        )

        if surprises % max(1, args.checkpoint_every) == 0:
            n = store.save_jsonl(args.save_store)
            print(f"[continuous] checkpoint store ({n}) → {args.save_store}")

    try:
        if args.no_follow:
            if not args.file.exists():
                print(f"[continuous] missing {args.file}", file=sys.stderr)
                sys.exit(1)
            with args.file.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        handle(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            for obj in follow_jsonl(args.file):
                handle(obj)
    except KeyboardInterrupt:
        print("\n[continuous] stopped")
    finally:
        surprise_fh.close()
        n = store.save_jsonl(args.save_store)
        print(f"[continuous] final store ({n} entries) → {args.save_store}")
        print(f"[continuous] frames={frames}  surprises={surprises}")


if __name__ == "__main__":
    main()
