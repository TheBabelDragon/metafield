#!/usr/bin/env python3
"""
echo_field_predictor.py

Tiny learned head on Echo Grid sessions.

Reads FieldObservation JSONL (preferred: /tmp/metafield/echo.jsonl) or
FieldMemoryEntry JSONL, builds fixed feature windows, trains a small MLP
to predict next-step motion + entropy.

Usage:

  # train + optional save
  python examples/echo_field_predictor.py --file /tmp/metafield/echo.jsonl \
    --save-model /tmp/metafield/echo_head.pt

  # score residuals with a saved head
  python examples/echo_field_predictor.py --file /tmp/metafield/echo.jsonl \
    --load-model /tmp/metafield/echo_head.pt --score /tmp/metafield/echo_residuals.jsonl

Motion is the useful target on real CSI; entropy is often near-constant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


FEATURE_NAMES = [
    "motion", "entropy", "df_max", "drive", "fuse",
    "n_tracks", "track_energy", "track_conf",
]


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def features_from_observation(obj: Dict[str, Any]) -> Optional[List[float]]:
    regions = obj.get("field_regions") or []
    by_name: Dict[str, Dict[str, Any]] = {}
    tracks = []
    for r in regions:
        name = str(r.get("region", ""))
        if name.startswith("track_"):
            tracks.append(r)
        else:
            by_name[name] = r

    motion = _f((by_name.get("motion") or {}).get("observed"))
    entropy = _f((by_name.get("entropy") or {}).get("observed"))
    df_max = _f((by_name.get("df_max") or {}).get("observed"))
    drive = _f((by_name.get("drive") or {}).get("observed"))
    fuse = _f((by_name.get("fuse") or {}).get("observed"))

    n_tracks = float(len(tracks))
    if tracks:
        te = sum(_f(t.get("observed")) for t in tracks) / len(tracks)
        tc = sum(_f(t.get("confidence"), 0.5) for t in tracks) / len(tracks)
    else:
        te, tc = 0.0, 0.0

    mod = (obj.get("modality") or {}).get("echo") or {}
    if motion == 0.0 and "motion" in mod:
        motion = _f(mod.get("motion"))
    if entropy == 0.0 and "entropy_raw" in mod:
        entropy = min(1.0, _f(mod.get("entropy_raw")) / 1.5)

    return [motion, entropy, df_max, drive, fuse, n_tracks / 6.0, te, tc]


def features_from_memory_entry(obj: Dict[str, Any]) -> Optional[List[float]]:
    obs = obj.get("observed_response") or []
    if not obs:
        return None
    core = [_f(v) for v in obs[:5]]
    while len(core) < 5:
        core.append(0.0)
    extra = obs[5:]
    n_tracks = float(len(extra))
    te = sum(_f(v) for v in extra) / len(extra) if extra else 0.0
    tc = _f(obj.get("confidence"), 0.5)
    return core + [n_tracks / 6.0, te, tc]


def load_feature_series(path: Path) -> List[List[float]]:
    series: List[List[float]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[predictor] skip line {line_no}: {e}", file=sys.stderr)
                continue

            feats = None
            if "field_regions" in obj:
                feats = features_from_observation(obj)
            elif "observed_response" in obj or "body_id" in obj:
                feats = features_from_memory_entry(obj)

            if feats is None or len(feats) != len(FEATURE_NAMES):
                continue
            series.append(feats)
    return series


def make_windows(
    series: List[List[float]],
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(series) < window + 1:
        raise SystemExit(
            f"[predictor] need at least {window + 1} samples, got {len(series)}"
        )
    xs, ys = [], []
    for i in range(window, len(series)):
        past = series[i - window : i]
        nxt = series[i]
        xs.append([v for row in past for v in row])
        ys.append([nxt[0], nxt[1]])  # motion, entropy
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


class TinyFieldHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval(
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int,
    lr: float,
    val_frac: float,
    seed: int,
) -> Tuple[Dict[str, float], TinyFieldHead]:
    n = X.shape[0]
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    if n_train < 8:
        raise SystemExit(f"[predictor] too few train samples ({n_train})")

    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xva, Yva = X[n_train:], Y[n_train:]

    torch.manual_seed(seed)
    model = TinyFieldHead(in_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    last_tr = last_va = float("nan")

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        loss.backward()
        opt.step()
        last_tr = float(loss.item())

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xva), Yva).item())
        last_va = vloss
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"  epoch {ep:4d}/{epochs}  train_mse={last_tr:.5f}  val_mse={last_va:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(Xva)
        mse_motion = float(((pred[:, 0] - Yva[:, 0]) ** 2).mean().item())
        mse_entropy = float(((pred[:, 1] - Yva[:, 1]) ** 2).mean().item())
        F = len(FEATURE_NAMES)
        window_idx = (X.shape[1] // F - 1) * F
        last_motion = Xva[:, window_idx + 0]
        last_entropy = Xva[:, window_idx + 1]
        base_m = float(((last_motion - Yva[:, 0]) ** 2).mean().item())
        base_e = float(((last_entropy - Yva[:, 1]) ** 2).mean().item())

    stats = {
        "n_total": float(n),
        "n_train": float(n_train),
        "n_val": float(n_val),
        "train_mse": last_tr,
        "val_mse": best_val,
        "val_mse_motion": mse_motion,
        "val_mse_entropy": mse_entropy,
        "baseline_mse_motion": base_m,
        "baseline_mse_entropy": base_e,
    }
    return stats, model


def score_series(
    series: List[List[float]],
    model: TinyFieldHead,
    window: int,
    out_path: Path,
) -> None:
    """Write residual JSONL: pred vs actual motion (primary signal)."""
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = []
    with out_path.open("w", encoding="utf-8") as fh, torch.no_grad():
        for i in range(window, len(series)):
            past = series[i - window : i]
            x = torch.tensor([[v for row in past for v in row]], dtype=torch.float32)
            pred = model(x)[0]
            actual_m, actual_e = series[i][0], series[i][1]
            pred_m, pred_e = float(pred[0]), float(pred[1])
            res_m = actual_m - pred_m
            row = {
                "i": i,
                "actual_motion": actual_m,
                "pred_motion": pred_m,
                "residual_motion": res_m,
                "abs_residual_motion": abs(res_m),
                "actual_entropy": actual_e,
                "pred_entropy": pred_e,
            }
            residuals.append(abs(res_m))
            fh.write(json.dumps(row) + "\n")

    if not residuals:
        print("[predictor] no score rows")
        return
    residuals.sort()
    n = len(residuals)
    mean_r = sum(residuals) / n
    p50 = residuals[n // 2]
    p90 = residuals[int(n * 0.9)]
    p99 = residuals[min(n - 1, int(n * 0.99))]
    print()
    print("=== score (motion residual) ===")
    print(f"  rows={n}  mean|r|={mean_r:.4f}  p50={p50:.4f}  p90={p90:.4f}  p99={p99:.4f}")
    print(f"  wrote → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny Echo field predictor")
    p.add_argument("--file", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", type=Path, default=None)
    p.add_argument("--load-model", type=Path, default=None)
    p.add_argument(
        "--score",
        type=Path,
        default=None,
        help="Write residual JSONL here (trains or uses --load-model)",
    )
    args = p.parse_args()

    if not args.file.exists():
        print(f"[predictor] file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"[predictor] loading {args.file}")
    series = load_feature_series(args.file)
    print(f"[predictor] samples={len(series)}  features={FEATURE_NAMES}")
    if len(series) < args.window + 5:
        print(f"[predictor] not enough data ({len(series)})", file=sys.stderr)
        sys.exit(1)

    X, Y = make_windows(series, args.window)
    print(f"[predictor] windows={X.shape[0]}  in_dim={X.shape[1]}  targets=motion,entropy")

    model: TinyFieldHead
    stats: Optional[Dict[str, float]] = None

    if args.load_model:
        ckpt = torch.load(args.load_model, map_location="cpu", weights_only=False)
        model = TinyFieldHead(in_dim=int(ckpt["in_dim"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[predictor] loaded {args.load_model}")
        if "stats" in ckpt:
            s = ckpt["stats"]
            print(
                f"  (ckpt val motion {s.get('val_mse_motion', float('nan')):.5f} "
                f"vs persistence {s.get('baseline_mse_motion', float('nan')):.5f})"
            )
    else:
        print("[predictor] training…")
        stats, model = train_eval(X, Y, args.epochs, args.lr, args.val_frac, args.seed)
        print()
        print("=== baseline result ===")
        print(f"  samples     train={int(stats['n_train'])}  val={int(stats['n_val'])}")
        print(f"  val MSE     {stats['val_mse']:.5f}  (best)")
        print(
            f"  val motion  {stats['val_mse_motion']:.5f}  "
            f"vs persistence {stats['baseline_mse_motion']:.5f}"
        )
        print(
            f"  val entropy {stats['val_mse_entropy']:.5f}  "
            f"vs persistence {stats['baseline_mse_entropy']:.5f}"
        )
        better_m = stats["val_mse_motion"] < stats["baseline_mse_motion"]
        better_e = stats["val_mse_entropy"] < stats["baseline_mse_entropy"]
        print(
            f"  beats persistence?  motion={'yes' if better_m else 'no'}  "
            f"entropy={'yes' if better_e else 'no'}"
        )

        if args.save_model:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_names": FEATURE_NAMES,
                    "window": args.window,
                    "in_dim": X.shape[1],
                    "stats": stats,
                },
                args.save_model,
            )
            print(f"[predictor] saved model → {args.save_model}")

    if args.score:
        score_series(series, model, args.window, args.score)


if __name__ == "__main__":
    main()
