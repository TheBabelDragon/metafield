#!/usr/bin/env python3
"""
field_observation.py

Canonical, body-agnostic observation schema for MetaField.

Any physical or simulated body (lattice HMC, optical dodecahedron,
WiFi CSI, ultrasonic, ZVS resonant/HV, …) should publish observations that conform
to this shape. MetaField’s memory, geometry, prediction, and
attractor layers consume this interface and remain unaware of the
underlying sensor modality.

See PHYSICAL_FIELD_SUBSTRATE.md for the architectural rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json

from schemas.scarcity_clock import ScarcityClock, parse_clock


@dataclass
class FieldRegion:
    """One coherent region of the observed field."""
    region: str
    expected: Optional[float] = None
    observed: Optional[float] = None
    confidence: float = 0.0
    anomaly: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["extras"]:
            del d["extras"]
        return d


@dataclass
class FieldObservation:
    body_id: str
    body_type: str
    excitation_id: Optional[int] = None
    field_regions: List[FieldRegion] = field(default_factory=list)
    geometry_state: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    clock: Optional[ScarcityClock] = None
    modality: Dict[str, Any] = field(default_factory=dict)
    health: str = "ok"
    schema_version: int = 2

    def resolved_clock(self) -> ScarcityClock:
        if self.clock is None:
            return ScarcityClock.unanchored()
        return self.clock if isinstance(self.clock, ScarcityClock) else parse_clock(self.clock)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_id": self.body_id,
            "body_type": self.body_type,
            "excitation_id": self.excitation_id,
            "field_regions": [r.to_dict() for r in self.field_regions],
            "geometry_state": self.geometry_state,
            "timestamp": self.timestamp,
            "clock": self.resolved_clock().to_dict(),
            "modality": self.modality or None,
            "health": self.health,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldObservation":
        regions = [
            FieldRegion(**{k: v for k, v in r.items() if k in (
                "region", "expected", "observed", "confidence", "anomaly", "extras"
            )}) if isinstance(r, dict) else r
            for r in data.get("field_regions", [])
        ]
        raw_clock = data.get("clock")
        clock = parse_clock(raw_clock) if raw_clock is not None else ScarcityClock.unanchored()
        return cls(
            body_id=data["body_id"],
            body_type=data["body_type"],
            excitation_id=data.get("excitation_id"),
            field_regions=regions,
            geometry_state=data.get("geometry_state", "unknown"),
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            clock=clock,
            modality=data.get("modality") or {},
            health=data.get("health", "ok"),
            schema_version=data.get("schema_version", 2),
        )


def lattice_observation(body_id, traj, regions, geometry_state="calibrated", hmc_extras=None):
    return FieldObservation(
        body_id=body_id, body_type="lattice", excitation_id=traj,
        field_regions=regions, geometry_state=geometry_state,
        modality={"hmc": hmc_extras or {}},
    )


def optical_observation(body_id, excitation_id, regions, geometry_state="uncalibrated", transfer_matrix_hint=None):
    return FieldObservation(
        body_id=body_id, body_type="optical", excitation_id=excitation_id,
        field_regions=regions, geometry_state=geometry_state,
        modality={"optical": transfer_matrix_hint or {}},
    )


def ultrasonic_observation(body_id, excitation_id, regions, geometry_state="calibrated", echo_extras=None):
    return FieldObservation(
        body_id=body_id, body_type="ultrasonic", excitation_id=excitation_id,
        field_regions=regions, geometry_state=geometry_state,
        modality={"echo": echo_extras or {}},
    )


def zvs_observation(body_id, excitation_id, regions, geometry_state="calibrated", power_extras=None):
    return FieldObservation(
        body_id=body_id, body_type="zvs", excitation_id=excitation_id,
        field_regions=regions, geometry_state=geometry_state,
        modality={"zvs": power_extras or {}},
    )


def validate_observation(obs: FieldObservation):
    problems = []
    if not obs.body_id:
        problems.append("body_id is required")
    if obs.body_type not in ("optical", "lattice", "wifi_csi", "ultrasonic", "zvs", "sim", "other"):
        problems.append(f"unknown body_type: {obs.body_type}")
    if not (0.0 <= obs.schema_version):
        problems.append("schema_version must be >= 0")
    for i, r in enumerate(obs.field_regions):
        if not r.region:
            problems.append(f"field_regions[{i}].region is empty")
        if not (0.0 <= r.confidence <= 1.0):
            problems.append(f"field_regions[{i}].confidence out of range")
        if not (0.0 <= r.anomaly <= 1.0):
            problems.append(f"field_regions[{i}].anomaly out of range")
    return problems
