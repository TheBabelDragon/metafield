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


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class FieldRegion:
    """One coherent region of the observed field."""
    region: str                     # stable identifier, e.g. "bottom_plane_cluster_3"
    expected: Optional[float] = None
    observed: Optional[float] = None
    confidence: float = 0.0         # 0..1
    anomaly: float = 0.0            # 0..1, higher = more anomalous
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # drop empty extras for cleaner JSON
        if not d["extras"]:
            del d["extras"]
        return d


@dataclass
class FieldObservation:
    """
    Single observation packet emitted by any body.

    Required fields keep the interface minimal; optional fields let
    specific bodies (optical transfer matrix, HMC trajectory id, etc.)
    attach modality-specific context without polluting the core.
    """
    # Identity
    body_id: str                    # e.g. "optical-dodeca-01", "lattice-sim-0", "zvs-node-03"
    body_type: str                  # "optical" | "lattice" | "wifi_csi" | "ultrasonic" | "zvs" | "sim"
    excitation_id: Optional[int] = None   # sequential or hash of the stimulus

    # The actual field state
    field_regions: List[FieldRegion] = field(default_factory=list)

    # Geometry / calibration status of the body itself
    geometry_state: str = "unknown"  # "uncalibrated" | "calibrating" | "calibrated" | "degraded"

    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional modality-specific payload (kept opaque to MetaField core)
    modality: Dict[str, Any] = field(default_factory=dict)

    # Health of the observation itself
    health: str = "ok"              # "ok" | "partial" | "stale" | "error"
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_id": self.body_id,
            "body_type": self.body_type,
            "excitation_id": self.excitation_id,
            "field_regions": [r.to_dict() for r in self.field_regions],
            "geometry_state": self.geometry_state,
            "timestamp": self.timestamp,
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
        return cls(
            body_id=data["body_id"],
            body_type=data["body_type"],
            excitation_id=data.get("excitation_id"),
            field_regions=regions,
            geometry_state=data.get("geometry_state", "unknown"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            modality=data.get("modality") or {},
            health=data.get("health", "ok"),
            schema_version=data.get("schema_version", 1),
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def lattice_observation(
    body_id: str,
    traj: int,
    regions: List[FieldRegion],
    geometry_state: str = "calibrated",
    hmc_extras: Optional[Dict[str, Any]] = None,
) -> FieldObservation:
    """Helper for the existing lattice/HMC body."""
    return FieldObservation(
        body_id=body_id,
        body_type="lattice",
        excitation_id=traj,
        field_regions=regions,
        geometry_state=geometry_state,
        modality={"hmc": hmc_extras or {}},
    )


def optical_observation(
    body_id: str,
    excitation_id: int,
    regions: List[FieldRegion],
    geometry_state: str = "uncalibrated",
    transfer_matrix_hint: Optional[Dict[str, Any]] = None,
) -> FieldObservation:
    """Helper for the optical dodecahedral body (Phase 0+)."""
    return FieldObservation(
        body_id=body_id,
        body_type="optical",
        excitation_id=excitation_id,
        field_regions=regions,
        geometry_state=geometry_state,
        modality={"optical": transfer_matrix_hint or {}},
    )


def ultrasonic_observation(
    body_id: str,
    excitation_id: Optional[int],
    regions: List[FieldRegion],
    geometry_state: str = "calibrated",
    echo_extras: Optional[Dict[str, Any]] = None,
) -> FieldObservation:
    """Helper for Echo Grid / ultrasonic field body."""
    return FieldObservation(
        body_id=body_id,
        body_type="ultrasonic",
        excitation_id=excitation_id,
        field_regions=regions,
        geometry_state=geometry_state,
        modality={"echo": echo_extras or {}},
    )


def zvs_observation(
    body_id: str,
    excitation_id: Optional[int],
    regions: List[FieldRegion],
    geometry_state: str = "calibrated",
    power_extras: Optional[Dict[str, Any]] = None,
) -> FieldObservation:
    """Helper for the ZVS resonant / HV power body."""
    return FieldObservation(
        body_id=body_id,
        body_type="zvs",
        excitation_id=excitation_id,
        field_regions=regions,
        geometry_state=geometry_state,
        modality={"zvs": power_extras or {}},
    )


# ---------------------------------------------------------------------------
# Minimal validation (fail-closed)
# ---------------------------------------------------------------------------

def validate_observation(obs: FieldObservation) -> List[str]:
    """Return a list of problems; empty list means the observation is acceptable."""
    problems: List[str] = []
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
