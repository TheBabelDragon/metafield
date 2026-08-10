#!/usr/bin/env python3
"""
echo_field_predictor.py

Tiny learned head on Echo Grid sessions.

Reads FieldObservation JSONL (preferred: /tmp/metafield/echo.jsonl) or
FieldMemoryEntry JSONL, builds fixed feature windows, trains a small MLP
to predict next-step motion + entropy.

Usage:

  python examples/echo_field_predictor.py --file /tmp/metafield/echo.jsonl
  python examples/echo_field_predictor.py --file /tmp/metafield/field_memory.jsonl --epochs 80

No automata. Baseline only — measurable train/val MSE before anything deeper.
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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

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
    """Named regions from Echo FieldObservation packets."""
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

    # modality fallback if regions sparse
    mod = (obj.get("modality") or {}).get("echo") or {}
    if motion == 0.0 and "motion" in mod:
        motion = _f(mod.get("motion"))
    if entropy == 0.0 and "entropy_raw" in mod:
        entropy = min(1.0, _f(mod.get("entropy_raw")) / 1.5)

    return [motion, entropy, df_max, drive, fuse, n_tracks / 6.0, te, tc]


def features_from_memory_entry(obj: Dict[str, Any]) -> Optional[List[float]]:
    """Flattened FieldMemoryEntry (order depends on emission)."""
    obs = obj.get("observed_response") or []
    if not obs:
        return None
    # pad/truncate to 5 core slots: motion, entropy, df_max, drive, fuse
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

            if feats is None:
                continue
            if len(feats) != len(FEATURE_NAMES):
                continue
            series.append(feats)
    return series


def make_windows(
    series: List[List[float]],
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [N, window * F] past features
    Y: [N, 2] next motion, next entropy
    """
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
    X = torch.tensor(xs, dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TinyFieldHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),  # motion/entropy in [0,1]
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
) -> Dict[str, float]:
    n = X.shape[0]
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    if n_train < 8:
        raise SystemExit(f"[predictor] too few train samples ({n_train})")

    # temporal split (no shuffle) — respect causality
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
        # per-target
        mse_motion = float(((pred[:, 0] - Yva[:, 0]) ** 2).mean().item())
        mse_entropy = float(((pred[:, 1] - Yva[:, 1]) ** 2).mean().item())
        # naive baseline: predict last window's motion/entropy (persistence)
        # last frame in window is features [window-1]
        F = len(FEATURE_NAMES)
        last_motion = Xva[:, (window_idx := (X.shape[1] // F - 1) * F) + 0]
        last_entropy = Xva[:, window_idx + 1]
        base_m = float(((last_motion - Yva[:, 0]) ** 2).mean().item())
        base_e = float(((last_entropy - Yva[:, 1]) ** 2).mean().item())

    return {
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


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny Echo field predictor baseline")
    p.add_argument(
        "--file",
        type=Path,
        default=Path("/tmp/metafield/echo.jsonl"),
        help="FieldObservation or FieldMemoryEntry JSONL",
    )
    p.add_argument("--window", type=int, default=8, help="Past frames in input")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to save state_dict",
    )
    args = p.parse_args()

    if not args.file.exists():
        print(f"[predictor] file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"[predictor] loading {args.file}")
    series = load_feature_series(args.file)
    print(f"[predictor] samples={len(series)}  features={FEATURE_NAMES}")
    if len(series) < args.window + 5:
        print(
            f"[predictor] not enough data ({len(series)}). "
            f"Collect more with Echo --metafield-log then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    X, Y = make_windows(series, args.window)
    print(f"[predictor] windows={X.shape[0]}  in_dim={X.shape[1]}  targets=motion,entropy")
    print("[predictor] training…")
    stats = train_eval(X, Y, args.epochs, args.lr, args.val_frac, args.seed)

    print()
    print("=== baseline result ===")
    print(f"  samples     train={int(stats['n_train'])}  val={int(stats['n_val'])}")
    print(f"  val MSE     {stats['val_mse']:.5f}  (best)")
    print(f"  val motion  {stats['val_mse_motion']:.5f}  vs persistence {stats['baseline_mse_motion']:.5f}")
    print(f"  val entropy {stats['val_mse_entropy']:.5f}  vs persistence {stats['baseline_mse_entropy']:.5f}")
    better_m = stats["val_mse_motion"] < stats["baseline_mse_motion"]
    better_e = stats["val_mse_entropy"] < stats["baseline_mse_entropy"]
    print(f"  beats persistence?  motion={'yes' if better_m else 'no'}  entropy={'yes' if better_e else 'no'}")

    if args.save_model:
        # retrain path already has best weights inside train_eval only in-memory;
        # quick re-fit for save
        model = TinyFieldHead(in_dim=X.shape[1])
        # reload best by re-running is heavy; save last trained via second pass
        n = X.shape[0]
        n_val = max(1, int(n * args.val_frac))
        n_train = n - n_val
        Xtr, Ytr = X[:n_train], Y[:n_train]
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        for _ in range(args.epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(Xtr), Ytr)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(X[n_train:]), Y[n_train:]).item())
            if v < best_val:
                best_val = v
                best_state = {k: v2.detach().clone() for k, v2 in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
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


if __name__ == "__main__":
    main()
