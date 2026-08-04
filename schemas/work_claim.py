#!/usr/bin/env python3
"""
WorkClaim — evidence that regulated MetaField work occurred.

This is *not* a blockchain transaction and *not* external currency.
It is an internal swarm credit claim derived from the Python runtime path
(stats / HMC / continuity), gated by METAFIELD_CONTROL_TOKEN at mint time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time
import hashlib
import json


SCHEMA_VERSION = 1


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
    timestamp: float = field(default_factory=lambda: time.time())
    evidence_hash: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkClaim":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})

    def compute_evidence_hash(self) -> str:
        payload = {
            "node_id": self.node_id,
            "traj": self.traj,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "recent_abs_dh": round(self.recent_abs_dh, 6),
            "health": self.health,
            "memory_size": self.memory_size,
            "num_attractors": self.num_attractors,
            "credit": round(self.credit, 6),
            "timestamp": self.timestamp,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def seal(self) -> "WorkClaim":
        if not self.claim_id:
            self.claim_id = f"wc_{int(self.timestamp)}_{self.traj}"
        self.evidence_hash = self.compute_evidence_hash()
        return self
