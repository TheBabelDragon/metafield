#!/usr/bin/env python3
"""
scarcity_clock.py

Bitcoin supplies the scarce, externally verifiable temporal coordinate.
Local wall time is observation metadata only — never the epoch.

    process time     = traj / excitation_id / HMC step
    btc_height       = ordered epoch
    cumulative work  = scarcity / security weight
    observed_at      = informational

Never invent height from time.time() or datetime.now().
Peer-supplied height/hash is evidence, not truth.
Aligned with aurora-swarm-btc mods/asset_fabric/artifact_clock.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CLOCK_VERSION = 1
CONFIDENCE_NONE = "none"
CONFIDENCE_PENDING = "pending"
CONFIDENCE_INCLUDED = "included"
CONFIDENCE_CONFIRMED = "confirmed"
CONFIDENCE_REORGED = "reorged"
_CONFIDENCE = frozenset(
    {CONFIDENCE_NONE, CONFIDENCE_PENDING, CONFIDENCE_INCLUDED, CONFIDENCE_CONFIRMED, CONFIDENCE_REORGED}
)
SOURCE_NONE = "none"
SOURCE_ENV = "env"
SOURCE_FILE = "file"
SOURCE_AURORA = "aurora"
SOURCE_PEER = "peer_claim"
SOURCE_EXPLICIT = "explicit"


def confidence_from_status(status: Optional[str], *, confirmations: int = 0, depth: int = 6) -> str:
    if not status:
        return CONFIDENCE_NONE
    s = str(status).upper().replace("-", "_")
    if s in ("UNANCHORED", "NONE"):
        return CONFIDENCE_NONE
    if s in ("REORGED", "RE_ANCHOR_REQUIRED", "REANCHOR_REQUIRED"):
        return CONFIDENCE_REORGED
    if s == "CONFIRMED":
        return CONFIDENCE_CONFIRMED
    if s == "INCLUDED":
        return CONFIDENCE_CONFIRMED if confirmations >= depth else CONFIDENCE_INCLUDED
    if s in (
        "COMMITMENT_PENDING", "BROADCAST", "RECORDED", "PENDING_BROADCAST", "SUBMITTED", "PENDING"
    ):
        return CONFIDENCE_PENDING
    return CONFIDENCE_NONE


def _opt_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


@dataclass(frozen=True)
class ScarcityClock:
    epoch: Optional[int]
    btc_height: Optional[int]
    btc_block_hash: Optional[str]
    btc_work: Optional[str]
    anchor_id: Optional[str]
    observed_at: float
    confidence: str
    source: str = SOURCE_NONE

    def __post_init__(self) -> None:
        conf = self.confidence if self.confidence in _CONFIDENCE else CONFIDENCE_NONE
        if conf != self.confidence:
            object.__setattr__(self, "confidence", conf)
        if conf == CONFIDENCE_NONE:
            object.__setattr__(self, "epoch", None)
            object.__setattr__(self, "btc_height", None)
            object.__setattr__(self, "btc_block_hash", None)
            object.__setattr__(self, "btc_work", None)
            object.__setattr__(self, "anchor_id", None)

    @property
    def clock_version(self) -> int:
        return CLOCK_VERSION

    @property
    def is_authoritative(self) -> bool:
        return self.confidence == CONFIDENCE_CONFIRMED and self.epoch is not None

    @property
    def is_anchored(self) -> bool:
        return self.btc_height is not None and self.confidence not in (CONFIDENCE_NONE, CONFIDENCE_REORGED)

    def canonical_tuple(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str], Optional[str], str]:
        return (
            self.epoch,
            self.btc_height,
            self.btc_block_hash,
            self.btc_work,
            self.anchor_id,
            self.confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clock_version": CLOCK_VERSION,
            "epoch": self.epoch,
            "btc_height": self.btc_height,
            "btc_block_hash": self.btc_block_hash,
            "btc_work": self.btc_work,
            "anchor_id": self.anchor_id,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "source": self.source,
            "authoritative": self.is_authoritative,
            "anchored": self.is_anchored,
        }

    @classmethod
    def unanchored(cls, *, observed_at: float = 0.0) -> "ScarcityClock":
        return cls(
            epoch=None,
            btc_height=None,
            btc_block_hash=None,
            btc_work=None,
            anchor_id=None,
            observed_at=float(observed_at or 0.0),
            confidence=CONFIDENCE_NONE,
            source=SOURCE_NONE,
        )

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ScarcityClock":
        if not d or not isinstance(d, dict):
            return cls.unanchored()
        confidence = str(d.get("confidence") or CONFIDENCE_NONE)
        source = str(d.get("source") or SOURCE_NONE)
        if source == SOURCE_PEER and confidence == CONFIDENCE_CONFIRMED:
            confidence = CONFIDENCE_INCLUDED
        return cls(
            epoch=_opt_int(d.get("epoch") if d.get("epoch") is not None else d.get("btc_height")),
            btc_height=_opt_int(d.get("btc_height")),
            btc_block_hash=_opt_str(d.get("btc_block_hash")),
            btc_work=_opt_str(d.get("btc_work")),
            anchor_id=_opt_str(d.get("anchor_id")),
            observed_at=float(d.get("observed_at") or 0.0),
            confidence=confidence,
            source=source,
        )

    @classmethod
    def from_tip(
        cls,
        tip: Dict[str, Any],
        *,
        source: str = SOURCE_EXPLICIT,
        observed_at: float = 0.0,
        confidence: Optional[str] = None,
    ) -> "ScarcityClock":
        height = _opt_int(tip.get("btc_height") if tip.get("btc_height") is not None else tip.get("height"))
        if height is None:
            return cls.unanchored(observed_at=observed_at)
        work = _opt_str(tip.get("btc_work") if tip.get("btc_work") is not None else tip.get("work"))
        block_hash = _opt_str(
            tip.get("btc_block_hash")
            if tip.get("btc_block_hash") is not None
            else tip.get("block_hash") or tip.get("hash")
        )
        conf = confidence or str(tip.get("confidence") or CONFIDENCE_INCLUDED)
        if source == SOURCE_PEER and conf == CONFIDENCE_CONFIRMED:
            conf = CONFIDENCE_INCLUDED
        epoch = _opt_int(tip.get("epoch"))
        if epoch is None:
            epoch = height
        obs = tip.get("observed_at")
        if obs is None:
            obs = observed_at
        return cls(
            epoch=epoch,
            btc_height=height,
            btc_block_hash=block_hash,
            btc_work=work,
            anchor_id=_opt_str(tip.get("anchor_id")),
            observed_at=float(obs or 0.0),
            confidence=conf,
            source=source,
        )


def parse_clock(value: Any) -> ScarcityClock:
    if isinstance(value, ScarcityClock):
        return value
    if isinstance(value, dict):
        if "confidence" in value or "source" in value:
            return ScarcityClock.from_dict(value)
        if "btc_height" in value or "height" in value or "epoch" in value:
            return ScarcityClock.from_tip(value, source=str(value.get("source") or SOURCE_EXPLICIT))
        return ScarcityClock.from_dict(value)
    return ScarcityClock.unanchored()


def _clock_from_env(*, observed_at: float = 0.0) -> Optional[ScarcityClock]:
    height = _opt_int(os.environ.get("METAFIELD_BTC_HEIGHT"))
    if height is None:
        return None
    return ScarcityClock.from_tip(
        {
            "btc_height": height,
            "btc_block_hash": os.environ.get("METAFIELD_BTC_BLOCK_HASH"),
            "btc_work": os.environ.get("METAFIELD_BTC_WORK"),
            "confidence": os.environ.get("METAFIELD_BTC_CONFIDENCE") or CONFIDENCE_INCLUDED,
            "observed_at": observed_at,
        },
        source=SOURCE_ENV,
        observed_at=observed_at,
    )


def _clock_from_file(*, observed_at: float = 0.0) -> Optional[ScarcityClock]:
    raw = os.environ.get("METAFIELD_CLOCK_PATH", "").strip()
    candidates = [Path(raw)] if raw else []
    runtime = os.environ.get("METAFIELD_RUNTIME_DIR", "").strip()
    if runtime:
        candidates.append(Path(runtime) / "btc_clock.json")
    xdg = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg:
        candidates.append(Path(xdg) / "metafield" / "btc_clock.json")
    candidates.append(Path("/tmp/metafield/btc_clock.json"))
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("clock"), dict):
            data = data["clock"]
        clock = ScarcityClock.from_tip(data, source=SOURCE_FILE, observed_at=observed_at)
        if clock.btc_height is not None or clock.is_anchored:
            return clock
        parsed = ScarcityClock.from_dict(data)
        if parsed.is_anchored:
            return parsed
    return None


def _clock_from_redis(*, observed_at: float = 0.0) -> Optional[ScarcityClock]:
    url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    keys = ("aurora:btc:clock", "aurora:clock", "aurora:asset:clock:tip", "aurora:sensing:context")
    try:
        import redis  # type: ignore

        r = redis.from_url(url, decode_responses=True, socket_connect_timeout=0.4)
        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            data = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(data, dict):
                continue
            nested = data.get("clock") or data.get("btc_clock")
            if not isinstance(nested, dict):
                nested = (data.get("metafield") or {}).get("clock")
            payload = nested if isinstance(nested, dict) else data
            if payload.get("btc_height", payload.get("height")) is None:
                continue
            return ScarcityClock.from_tip(payload, source=SOURCE_AURORA, observed_at=observed_at)
    except Exception:
        return None
    return None


def resolve_clock(explicit: Any = None, *, observed_at: float = 0.0, allow_network: bool = True) -> ScarcityClock:
    """Best-effort local clock. Fail open to unanchored. Never invent height."""
    if explicit is not None:
        clock = parse_clock(explicit)
        if clock.is_anchored or clock.confidence != CONFIDENCE_NONE:
            if clock.source == SOURCE_NONE:
                return ScarcityClock(
                    epoch=clock.epoch,
                    btc_height=clock.btc_height,
                    btc_block_hash=clock.btc_block_hash,
                    btc_work=clock.btc_work,
                    anchor_id=clock.anchor_id,
                    observed_at=clock.observed_at or observed_at,
                    confidence=clock.confidence,
                    source=SOURCE_EXPLICIT,
                )
            return clock
    env_clock = _clock_from_env(observed_at=observed_at)
    if env_clock is not None:
        return env_clock
    file_clock = _clock_from_file(observed_at=observed_at)
    if file_clock is not None:
        return file_clock
    if allow_network:
        redis_clock = _clock_from_redis(observed_at=observed_at)
        if redis_clock is not None:
            return redis_clock
    return ScarcityClock.unanchored(observed_at=observed_at)


def attach_clock(payload: Dict[str, Any], clock: Optional[ScarcityClock] = None) -> Dict[str, Any]:
    existing = payload.get("clock")
    if isinstance(existing, dict) and (
        existing.get("btc_height") is not None or existing.get("confidence") == CONFIDENCE_NONE
    ):
        if "clock_version" not in existing:
            payload["clock"] = ScarcityClock.from_dict(existing).to_dict()
        return payload
    payload["clock"] = (clock or resolve_clock(allow_network=False)).to_dict()
    return payload
