#!/usr/bin/env python3
"""No-hardware check for the scarcity clock. Run from repo root:

    python examples/clock_smoke.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schemas.scarcity_clock import resolve_clock, ScarcityClock
from schemas.field_observation import optical_observation, FieldRegion, FieldObservation
from security import write_local_stats, read_local_stats


def main() -> None:
    print("1. default resolve (no tip) → unanchored")
    for key in ("METAFIELD_BTC_HEIGHT", "METAFIELD_BTC_BLOCK_HASH", "METAFIELD_BTC_WORK", "METAFIELD_CLOCK_PATH"):
        os.environ.pop(key, None)
    clock = resolve_clock(allow_network=False)
    print("  ", clock.to_dict())
    assert not clock.is_anchored

    print("2. env tip → included, not confirmed")
    os.environ["METAFIELD_BTC_HEIGHT"] = "900001"
    os.environ["METAFIELD_BTC_BLOCK_HASH"] = "ab" * 16
    clock = resolve_clock(allow_network=False)
    print("  ", clock.to_dict())
    assert clock.btc_height == 900001
    assert clock.confidence == "included"
    assert not clock.is_authoritative
    os.environ.pop("METAFIELD_BTC_HEIGHT")
    os.environ.pop("METAFIELD_BTC_BLOCK_HASH")

    print("3. file tip")
    tip = Path(tempfile.gettempdir()) / "metafield-clock-smoke.json"
    tip.write_text(json.dumps({"btc_height": 880000, "btc_work": "ff", "btc_block_hash": "11"}))
    os.environ["METAFIELD_CLOCK_PATH"] = str(tip)
    clock = resolve_clock(allow_network=False)
    print("  ", clock.to_dict())
    assert clock.btc_height == 880000
    os.environ.pop("METAFIELD_CLOCK_PATH")
    tip.unlink(missing_ok=True)

    print("4. observation packet carries clock, timestamp is observed_at")
    obs = optical_observation("optical-01", 3, [FieldRegion(region="r0", observed=0.4)])
    obs.clock = ScarcityClock.from_tip({"height": 880000, "hash": "11"}, source="explicit")
    packet = obs.to_dict()
    print("   excitation_id=", packet["excitation_id"], "height=", packet["clock"]["btc_height"])
    back = FieldObservation.from_dict(packet)
    assert back.resolved_clock().btc_height == 880000

    print("5. stats.json stamp (unanchored unless env/file set)")
    runtime = Path(tempfile.mkdtemp(prefix="mf-clock-"))
    os.environ["METAFIELD_RUNTIME_DIR"] = str(runtime)
    stats_path = runtime / "stats.json"
    write_local_stats({"traj": 12, "live": True, "health": "ok"}, stats_path)
    stamped = read_local_stats(stats_path)
    print("  ", stamped["clock"])
    assert stamped["clock"]["confidence"] == "none"
    assert stamped["clock"]["btc_height"] is None

    print("ok — clock is testable. Pull branch scarcity-clock / PR #5.")


if __name__ == "__main__":
    main()
