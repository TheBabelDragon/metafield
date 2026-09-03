#!/usr/bin/env python3
"""
field_memory.py

Episodic memory entry for physical field bodies.

This is the MetaField-side structure that turns a stream of
FieldObservation packets into "I remember what this response means."

It is deliberately separate from the body's FRAM (identity) and
from the body's working RAM (current thought).

See MEMORY_ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json

from schemas.scarcity_clock import ScarcityClock, parse_clock


@dataclass
class FieldMemoryEntry:
    """
    One remembered field experience.

    Links an observation back to an attractor and (optionally)
    a spatial / region key so MetaField can say:

        “At this excitation, on this body, at this chain
         position (or unanchored), the field looked like this.”
    """
    body_id: str
    excitation_id: Optional[int] = None

    # Optional spatial / region key (body-defined or MetaField-assigned)
    location: Optional[Dict[str, Any]] = None   # e.g. {"region": "..."} or {"x":..,"y":..,"z":..}

    expected_response: Optional[List[float]] = None
    observed_response: Optional[List[float]] = None

    confidence: float = 0.0
    anomaly: float = 0.0

    attractor_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    clock: Optional[ScarcityClock] = None

    # Free-form context from the observation or from MetaField processing
    extras: Dict[str, Any] = field(default_factory=dict)

    def resolved_clock(self) -> ScarcityClock:
        if self.clock is None:
            return ScarcityClock.unanchored()
        return self.clock if isinstance(self.clock, ScarcityClock) else parse_clock(self.clock)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clock"] = self.resolved_clock().to_dict()
        if not d["extras"]:
            del d["extras"]
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_observation(
        cls,
        obs_dict: Dict[str, Any],
        attractor_id: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
    ) -> "FieldMemoryEntry":
        """Convenience: build an entry from a FieldObservation-style dict."""
        regions = obs_dict.get("field_regions") or []
        observed = [r.get("observed") for r in regions if r.get("observed") is not None]
        expected = [r.get("expected") for r in regions if r.get("expected") is not None]
        confs = [r.get("confidence", 0.0) for r in regions]
        anoms = [r.get("anomaly", 0.0) for r in regions]

        return cls(
            body_id=obs_dict.get("body_id", "unknown"),
            excitation_id=obs_dict.get("excitation_id"),
            location=location,
            expected_response=expected or None,
            observed_response=observed or None,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            anomaly=max(anoms) if anoms else 0.0,
            attractor_id=attractor_id,
            timestamp=obs_dict.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            clock=parse_clock(obs_dict.get("clock")),
            extras={"geometry_state": obs_dict.get("geometry_state")},
        )
