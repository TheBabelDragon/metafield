#!/usr/bin/env python3
"""
aurora_persist_loop.py

Phase 14: run optical body loop and persist artifacts as content-addressed objects.

Usage:
  PYTHONPATH=../wavebridge:. python examples/aurora_persist_loop.py
  PYTHONPATH=../wavebridge:. python examples/aurora_persist_loop.py --root /tmp/mf_aurora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aurora_persist import AuroraObjectStore
from examples.optical_body_closed_loop import run_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist optical loop to Aurora store")
    parser.add_argument("--root", type=Path, default=Path("/tmp/metafield/aurora_objects"))
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lasers", type=int, default=8)
    parser.add_argument("--detectors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_loop(
        steps=args.steps,
        n_lasers=args.lasers,
        n_detectors=args.detectors,
        seed=args.seed,
    )
    store = AuroraObjectStore(root=args.root)

    shas = []
    for h in report["history"]:
        sha = store.put(
            "observation",
            h,
            meta={"body_id": "optical-body-sim", "source": "aurora_persist_loop"},
        )
        shas.append(sha)

    hist_sha = store.put_probe_history(
        report["history"], ref="optical-body-sim/latest_probe_history"
    )
    try:
        from examples.optical_body_closed_loop import _make_body

        body = _make_body(args.lasers, args.detectors, args.seed, 0.01)
        T = body.estimate_transfer_matrix(drive_level=1.0)
        tm_sha = store.put_transfer_matrix(
            T, body_id="optical-body-sim", ref="optical-body-sim/transfer_matrix"
        )
    except Exception:
        tm_sha = None

    summary = {
        "root": str(args.root),
        "n_observations": len(shas),
        "first_sha": shas[0] if shas else None,
        "last_sha": shas[-1] if shas else None,
        "probe_history_sha": hist_sha,
        "transfer_matrix_sha": tm_sha,
        "coverage": report["coverage"],
        "index_tail": store.list_index(limit=5),
    }
    store.put("run_summary", summary, ref="optical-body-sim/latest_run")

    resolved = store.resolve("optical-body-sim/latest_probe_history")
    summary["resolved_history_steps"] = (
        len(resolved["data"]) if resolved else 0
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"[aurora_persist] root={args.root}")
        print(f"  observations={summary['n_observations']}")
        print(f"  probe_history={hist_sha[:12]}…")
        if tm_sha:
            print(f"  transfer_matrix={tm_sha[:12]}…")
        print(f"  resolved_history_steps={summary['resolved_history_steps']}")
        print(f"  coverage={summary['coverage']:.0%}")


if __name__ == "__main__":
    main()
