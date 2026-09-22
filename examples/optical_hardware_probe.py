#!/usr/bin/env python3
"""
optical_hardware_probe.py

Hardware test harness for optical body session.

Usage:
  PYTHONPATH=../wavebridge:. python examples/optical_hardware_probe.py \\
      --port /dev/ttyUSB0 --lasers 12

  PYTHONPATH=../wavebridge:. python examples/optical_hardware_probe.py --dry-run

  PYTHONPATH=../wavebridge:. python examples/optical_hardware_probe.py \\
      --dry-run --persist /tmp/mf_hw_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from optical_serial_body import OpticalSerialBody
from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry


def run_probe(
    port: str,
    *,
    baud: int = 115200,
    n_lasers: int = 12,
    n_detectors: int = 20,
    dry_run: bool = False,
    laser_list: list | None = None,
    persist_root: Path | None = None,
) -> dict:
    lasers = laser_list or list(range(n_lasers))
    store = FieldMemoryStore(soft_capacity=512)
    history = []

    with OpticalSerialBody(
        port=port,
        baud=baud,
        n_lasers=n_lasers,
        n_detectors=n_detectors,
        dry_run=dry_run,
    ) as body:
        alive = body.ping()
        for laser_id in lasers:
            rec = body.excite(laser_id, drive_level=0.9)
            peak = float(np.max(rec.detector_response)) if rec.detector_response.size else 0.0
            entry = FieldMemoryEntry(
                body_id="optical-hardware",
                excitation_id=laser_id,
                observed_response=rec.detector_response.tolist(),
                confidence=0.9 if rec.ok and peak > 0.05 else 0.4,
                anomaly=0.0 if rec.ok else 1.0,
                extras={
                    "laser_id": laser_id,
                    "ok": rec.ok,
                    "error": rec.error,
                    "peak": peak,
                    "geometry_state": rec.geometry_state,
                },
            )
            store.add(entry)
            history.append(
                {
                    "laser_id": laser_id,
                    "ok": rec.ok,
                    "peak": round(peak, 4),
                    "n_detectors": int(rec.detector_response.size),
                    "error": rec.error,
                }
            )
        try:
            T = body.estimate_transfer_matrix(drive_level=0.9)
            T_shape = list(T.shape)
            T_norm = float(np.linalg.norm(T))
        except Exception:
            T = None
            T_shape = None
            T_norm = None

    report = {
        "port": port,
        "dry_run": dry_run,
        "ping": alive,
        "n_ok": sum(1 for h in history if h["ok"]),
        "n_fail": sum(1 for h in history if not h["ok"]),
        "history": history,
        "transfer_matrix_shape": T_shape,
        "transfer_matrix_frobenius": T_norm,
        "store_stats": store.get_stats(),
    }

    if persist_root is not None:
        try:
            from aurora_persist import AuroraObjectStore

            aos = AuroraObjectStore(root=persist_root, publish=False)
            aos.put_probe_history(history, ref="hardware/probe_history")
            if T is not None:
                aos.put_transfer_matrix(T, body_id="optical-hardware")
            for h in history:
                aos.put("observation", h, meta={"source": "optical_hardware_probe"})
            report["persist_root"] = str(persist_root)
        except Exception as exc:
            report["persist_error"] = str(exc)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical body hardware probe")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--lasers", type=int, default=12)
    parser.add_argument("--detectors", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=int, nargs="*", help="Subset of laser ids")
    parser.add_argument("--persist", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_probe(
        args.port,
        baud=args.baud,
        n_lasers=args.lasers,
        n_detectors=args.detectors,
        dry_run=args.dry_run,
        laser_list=args.only,
        persist_root=args.persist,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[hardware_probe] port={report['port']} dry_run={report['dry_run']}")
        print(f"  ping={'OK' if report['ping'] else 'FAIL'}")
        print(f"  ok={report['n_ok']} fail={report['n_fail']}")
        for h in report["history"]:
            status = "ok" if h["ok"] else f"ERR {h['error'][:40]}"
            print(
                f"  laser {h['laser_id']:02d}: peak={h['peak']:.3f} "
                f"det={h['n_detectors']} {status}"
            )
        if report.get("transfer_matrix_shape"):
            print(
                f"  transfer {report['transfer_matrix_shape']} "
                f"||T||={report['transfer_matrix_frobenius']:.3f}"
            )
        if report.get("persist_root"):
            print(f"  persisted → {report['persist_root']}")


if __name__ == "__main__":
    main()
