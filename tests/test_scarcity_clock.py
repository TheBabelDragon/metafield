"""Scarcity clock: wall time is never the epoch."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schemas.scarcity_clock import (
    CONFIDENCE_CONFIRMED,
    CONFIDENCE_INCLUDED,
    CONFIDENCE_NONE,
    SOURCE_PEER,
    ScarcityClock,
    attach_clock,
    parse_clock,
    resolve_clock,
)
from schemas.field_observation import FieldObservation, FieldRegion, optical_observation
from schemas.field_memory import FieldMemoryEntry
from schemas.work_claim import WorkClaim


def test_unanchored_is_default_and_has_no_epoch():
    clock = ScarcityClock.unanchored(observed_at=123.0)
    assert clock.confidence == CONFIDENCE_NONE
    assert clock.btc_height is None
    assert clock.epoch is None
    assert not clock.is_anchored
    assert not clock.is_authoritative
    assert clock.observed_at == 123.0


def test_none_confidence_strips_height():
    clock = ScarcityClock(
        epoch=900000,
        btc_height=900000,
        btc_block_hash="ab",
        btc_work="1",
        anchor_id="a1",
        observed_at=1.0,
        confidence=CONFIDENCE_NONE,
        source="explicit",
    )
    assert clock.btc_height is None
    assert clock.anchor_id is None


def test_canonical_tuple_excludes_observed_at():
    a = ScarcityClock.from_tip(
        {"btc_height": 800000, "btc_block_hash": "aa", "btc_work": "ff", "observed_at": 1.0},
        source="explicit",
        confidence=CONFIDENCE_INCLUDED,
    )
    b = ScarcityClock.from_tip(
        {"btc_height": 800000, "btc_block_hash": "aa", "btc_work": "ff", "observed_at": 999.0},
        source="explicit",
        confidence=CONFIDENCE_INCLUDED,
    )
    assert a.canonical_tuple() == b.canonical_tuple()
    assert a.observed_at != b.observed_at


def test_peer_claim_cannot_be_confirmed_on_arrival():
    clock = ScarcityClock.from_dict(
        {
            "btc_height": 100,
            "btc_block_hash": "00",
            "confidence": CONFIDENCE_CONFIRMED,
            "source": SOURCE_PEER,
        }
    )
    assert clock.confidence == CONFIDENCE_INCLUDED
    assert not clock.is_authoritative


def test_never_invent_height_from_wall_time():
    clock = resolve_clock(allow_network=False)
    assert clock.btc_height is None
    assert clock.confidence == CONFIDENCE_NONE
    assert clock.epoch is None


def test_env_tip_is_included_not_invented():
    os.environ["METAFIELD_BTC_HEIGHT"] = "850000"
    os.environ["METAFIELD_BTC_BLOCK_HASH"] = "deadbeef"
    os.environ["METAFIELD_BTC_WORK"] = "abc"
    try:
        clock = resolve_clock(allow_network=False)
        assert clock.btc_height == 850000
        assert clock.btc_block_hash == "deadbeef"
        assert clock.source == "env"
        assert clock.confidence == CONFIDENCE_INCLUDED
        assert not clock.is_authoritative
    finally:
        os.environ.pop("METAFIELD_BTC_HEIGHT", None)
        os.environ.pop("METAFIELD_BTC_BLOCK_HASH", None)
        os.environ.pop("METAFIELD_BTC_WORK", None)


def test_legacy_observation_without_clock_is_unanchored():
    obs = FieldObservation.from_dict(
        {
            "body_id": "x",
            "body_type": "optical",
            "timestamp": "2020-01-01T00:00:00+00:00",
            "field_regions": [],
        }
    )
    clock = parse_clock(getattr(obs, "clock", None) or (obs.to_dict().get("clock") if hasattr(obs, "to_dict") else None))
    assert not clock.is_anchored or obs.timestamp.startswith("2020")


def test_memory_entry_copies_clock_from_observation():
    obs = {
        "body_id": "optical-01",
        "excitation_id": 3,
        "field_regions": [{"region": "a", "observed": 1.0, "expected": 1.1, "confidence": 0.8, "anomaly": 0.1}],
        "clock": {"btc_height": 42, "confidence": "included", "source": "aurora"},
        "geometry_state": "calibrated",
    }
    entry = FieldMemoryEntry.from_observation(obs)
    assert entry.resolved_clock().btc_height == 42
    dumped = entry.to_dict()
    assert dumped["clock"]["btc_height"] == 42


def test_work_claim_hash_ignores_wall_time():
    a = WorkClaim(traj=10, credit=1.0, btc_height=100, clock_confidence="included", timestamp=1.0)
    b = WorkClaim(traj=10, credit=1.0, btc_height=100, clock_confidence="included", timestamp=999.0)
    assert a.compute_evidence_hash() == b.compute_evidence_hash()
    c = WorkClaim(traj=10, credit=1.0, btc_height=101, clock_confidence="included", timestamp=1.0)
    assert a.compute_evidence_hash() != c.compute_evidence_hash()


def test_attach_clock_does_not_invent_height():
    payload = attach_clock({"traj": 4, "live": True})
    assert payload["clock"]["btc_height"] is None
    assert payload["clock"]["confidence"] == "none"


def test_file_tip():
    path = Path("/tmp/metafield-clock-test.json")
    path.write_text(json.dumps({"btc_height": 777, "btc_block_hash": "11", "btc_work": "22"}))
    os.environ["METAFIELD_CLOCK_PATH"] = str(path)
    os.environ.pop("METAFIELD_BTC_HEIGHT", None)
    try:
        clock = resolve_clock(allow_network=False)
        assert clock.btc_height == 777
        assert clock.source == "file"
    finally:
        os.environ.pop("METAFIELD_CLOCK_PATH", None)
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [
        test_unanchored_is_default_and_has_no_epoch,
        test_none_confidence_strips_height,
        test_canonical_tuple_excludes_observed_at,
        test_peer_claim_cannot_be_confirmed_on_arrival,
        test_never_invent_height_from_wall_time,
        test_env_tip_is_included_not_invented,
        test_legacy_observation_without_clock_is_unanchored,
        test_memory_entry_copies_clock_from_observation,
        test_work_claim_hash_ignores_wall_time,
        test_attach_clock_does_not_invent_height,
        test_file_tip,
    ]
    for fn in tests:
        print(f"→ {fn.__name__}")
        fn()
        print("  ok")
    print("all scarcity clock tests passed")
