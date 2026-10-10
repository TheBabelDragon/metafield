"""Shared evidence lineage for FieldObservation.

Maps the CANtheon / CANgate boundary envelope into MetaField without
importing CANtheon. All conversion is dict-based and fail-closed.

Observation kinds distinguish:
  RAW          — direct measurement from a body / transport
  DERIVED      — computed from one or more parents (same substrate)
  PREDICTION   — model-produced expectation
  ASSERTION    — admitted claim / constraint (not a measurement)

Quality is never silently rewritten to a "good" value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceKind(str, Enum):
    RAW = "RAW"
    DERIVED = "DERIVED"
    PREDICTION = "PREDICTION"
    ASSERTION = "ASSERTION"


class EvidenceQuality(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    STALE = "STALE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    DECODE_ERROR = "DECODE_ERROR"
    UNKNOWN = "UNKNOWN"


_QUALITY_MAP = {
    "VALID": EvidenceQuality.VALID,
    "INVALID": EvidenceQuality.INVALID,
    "STALE": EvidenceQuality.STALE,
    "OUT_OF_RANGE": EvidenceQuality.OUT_OF_RANGE,
    "DECODE_ERROR": EvidenceQuality.DECODE_ERROR,
}


def parse_quality(value: Any) -> EvidenceQuality:
    if value is None:
        return EvidenceQuality.UNKNOWN
    if isinstance(value, EvidenceQuality):
        return value
    s = str(value).upper()
    return _QUALITY_MAP.get(s, EvidenceQuality.UNKNOWN)


@dataclass
class EvidenceLineage:
    """Provenance / parent lineage attached to a FieldObservation.

    Optional on every observation. Existing consumers ignore it.
    """

    kind: EvidenceKind = EvidenceKind.RAW
    quality: EvidenceQuality = EvidenceQuality.UNKNOWN
    source_id: str = ""
    event_id: str = ""
    sequence: Optional[int] = None
    source_timestamp_ns: Optional[int] = None
    ingest_timestamp_ns: Optional[int] = None
    parent_event_ids: List[str] = field(default_factory=list)
    transformation_id: str = ""
    transformation_version: str = ""
    unit: str = ""
    schema_id: str = ""
    schema_version: str = ""
    boundary_schema_id: str = ""
    boundary_schema_version: str = ""
    raw_identity: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": self.kind.value if isinstance(self.kind, EvidenceKind) else str(self.kind),
            "quality": self.quality.value if isinstance(self.quality, EvidenceQuality) else str(self.quality),
            "source_id": self.source_id,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "source_timestamp_ns": self.source_timestamp_ns,
            "ingest_timestamp_ns": self.ingest_timestamp_ns,
            "parent_event_ids": list(self.parent_event_ids),
            "transformation_id": self.transformation_id,
            "transformation_version": self.transformation_version,
            "unit": self.unit,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "boundary_schema_id": self.boundary_schema_id,
            "boundary_schema_version": self.boundary_schema_version,
        }
        if self.raw_identity:
            d["raw_identity"] = dict(self.raw_identity)
        if self.extras:
            d["extras"] = dict(self.extras)
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["EvidenceLineage"]:
        if not data:
            return None
        kind_raw = data.get("kind", "RAW")
        try:
            kind = EvidenceKind(str(kind_raw).upper())
        except ValueError:
            kind = EvidenceKind.RAW
        return cls(
            kind=kind,
            quality=parse_quality(data.get("quality")),
            source_id=str(data.get("source_id") or data.get("node_id") or ""),
            event_id=str(data.get("event_id") or ""),
            sequence=data.get("sequence"),
            source_timestamp_ns=data.get("source_timestamp_ns"),
            ingest_timestamp_ns=data.get("ingest_timestamp_ns"),
            parent_event_ids=list(data.get("parent_event_ids") or []),
            transformation_id=str(data.get("transformation_id") or ""),
            transformation_version=str(data.get("transformation_version") or ""),
            unit=str(data.get("unit") or ""),
            schema_id=str(data.get("schema_id") or ""),
            schema_version=str(data.get("schema_version") or ""),
            boundary_schema_id=str(data.get("boundary_schema_id") or ""),
            boundary_schema_version=str(data.get("boundary_schema_version") or ""),
            raw_identity=dict(data.get("raw_identity") or {}),
            extras=dict(data.get("extras") or {}),
        )
