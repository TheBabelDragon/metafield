#!/usr/bin/env python3
"""
echo_retrain_loop.py — keep the motion head current

Watches /tmp/metafield/echo.jsonl, retrains when enough *new* samples
arrive (or on a time interval), writes echo_head.pt atomically.

Echo / automata that hot-reload the checkpoint pick up new weights
without a process restart.

Usage:

  python examples/echo_retrain_loop.py
  python examples/echo_retrain_loop.py --interval 300 --min-new 80 --epochs 40

Leave this as Terminal 3 overnight with Echo + automata.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

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


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("{"):
                    n += 1
    except OSError:
        return 0
    return n


def atomic_save(ckpt: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def retrain_once(
    pred_mod,
    data: Path,
    out: Path,
    *,
    window: int,
    epochs: int,
    lr: float,
    val_frac: float,
    seed: int,
) -> dict:
    series = pred_mod.load_feature_series(data)
    if len(series) < window + 20:
        return {"ok": False, "reason": f"samples={len(series)} need>={window + 20}"}

    X, Y = pred_mod.make_windows(series, window)
    print(f"[retrain] samples={len(series)} windows={X.shape[0]} in_dim={X.shape[1]}")
    print("[retrain] training…")
    stats, model = pred_mod.train_eval(X, Y, epochs, lr, val_frac, seed)

    better = stats["val_mse_motion"] < stats["baseline_mse_motion"]
    print(
        f"[retrain] val motion={stats['val_mse_motion']:.5f} "
        f"vs persistence={stats['baseline_mse_motion']:.5f} "
        f"beats={'yes' if better else 'no'}"
    )

    ckpt = {
        "state_dict": model.state_dict(),
        "feature_names": pred_mod.FEATURE_NAMES,
        "window": window,
        "in_dim": int(X.shape[1]),
        "stats": stats,
        "trained_at": time.time(),
        "n_samples": len(series),
    }
    atomic_save(ckpt, out)
    print(f"[retrain] saved → {out}")
    return {"ok": True, "stats": stats, "n_samples": len(series), "better": better}


def main() -> None:
    p = argparse.ArgumentParser(description="Continuous Echo head retrain loop")
    p.add_argument("--file", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--save-model", type=Path, default=Path("/tmp/metafield/echo_head.pt"))
    p.add_argument("--interval", type=float, default=300.0, help="Seconds between checks (default 300)")
    p.add_argument("--min-new", type=int, default=80, help="Min new JSONL lines since last train")
    p.add_argument("--min-total", type=int, default=120, help="Min total samples before first train")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--once", action="store_true", help="Single retrain then exit")
    args = p.parse_args()

    pred_mod = _load_predictor()
    last_trained_lines = 0
    if args.save_model.exists():
        try:
            ckpt = torch.load(args.save_model, map_location="cpu", weights_only=False)
            last_trained_lines = int(ckpt.get("n_samples") or 0)
            print(f"[retrain] existing head n_samples={last_trained_lines}")
        except Exception as e:
            print(f"[retrain] could not read existing head ({e})")

    print(
        f"[retrain] watching {args.file}  interval={args.interval:.0f}s  "
        f"min_new={args.min_new}  → {args.save_model}"
    )

    while True:
        n = count_jsonl_lines(args.file)
        new = n - last_trained_lines
        due = n >= args.min_total and (last_trained_lines == 0 or new >= args.min_new)

        if due:
            print(f"[retrain] trigger lines={n} new={new}")
            result = retrain_once(
                pred_mod,
                args.file,
                args.save_model,
                window=args.window,
                epochs=args.epochs,
                lr=args.lr,
                val_frac=args.val_frac,
                seed=args.seed,
            )
            if result.get("ok"):
                last_trained_lines = int(result["n_samples"])
            else:
                print(f"[retrain] skip: {result.get('reason')}")
        else:
            print(
                f"[retrain] idle lines={n} new={new} "
                f"(need +{max(0, args.min_new - new)} or total>={args.min_total})"
            )

        if args.once:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[retrain] stopped")
            break


if __name__ == "__main__":
    main()
