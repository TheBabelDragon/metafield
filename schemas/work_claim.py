#!/usr/bin/env python3
"""
WorkClaim — evidence that regulated MetaField work occurred.

Internal swarm credit claim from the Python runtime path.
Integrity: evidence_hash (content) + mac (HMAC-SHA256 with control token).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time
import hashlib
import hmac
import json


SCHEMA_VERSION = 3


@dataclass
class WorkClaim:
    schema_version: int = SCHEMA_VERSION
    claim_id: str = ""
    node_id: str = "metafield_local"
    traj: int = 0
    acceptance_rate: float = 0.0
    recent_abs_dh: float = 0.0
    health: str = "unknown"
    live: bool = False
    memory_size: int = 0
    num_attractors: int = 0
    credit: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=lambda: time.time())  # observed_at only
    btc_height: Optional[int] = None
    btc_work: Optional[str] = None
    clock_confidence: str = "none"
    evidence_hash: str = ""
    mac: str = ""  # HMAC-SHA256 hex over evidence_hash with control token
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkClaim":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})

    def _canonical_payload(self) -> str:
        # Wall timestamp is not part of claim identity.
        payload = {
            "node_id": self.node_id,
            "traj": self.traj,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "recent_abs_dh": round(self.recent_abs_dh, 6),
            "health": self.health,
            "memory_size": self.memory_size,
            "num_attractors": self.num_attractors,
            "credit": round(self.credit, 6),
            "btc_height": self.btc_height,
            "btc_work": self.btc_work,
            "clock_confidence": self.clock_confidence,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def compute_evidence_hash(self) -> str:
        return hashlib.sha256(self._canonical_payload().encode()).hexdigest()[:32]

    def seal(self, token: Optional[str] = None) -> "WorkClaim":
        if not self.claim_id:
            height = self.btc_height if self.btc_height is not None else "u"
            self.claim_id = f"wc_{height}_{self.traj}"
        self.evidence_hash = self.compute_evidence_hash()
        if token:
            self.mac = hmac.new(
                token.encode("utf-8"),
                self.evidence_hash.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        else:
            self.mac = ""
        return self

    def verify_mac(self, token: str) -> bool:
        if not token or not self.mac or not self.evidence_hash:
            return False
        expected = hmac.new(
            token.encode("utf-8"),
            self.evidence_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, self.mac)
