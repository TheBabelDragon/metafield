"""CANtheon / CANgate boundary envelope → FieldObservation.

Pure, dependency-free conversion. Accepts the JSON-serializable envelope
produced by cantheon.adapters.metafield_contract (or any compatible dict).

Does not import cantheon.
"""

from __future__ import annotations

from typing import Any, Dict, List

from schemas.field_observation import FieldObservation, FieldRegion
from schemas.evidence import (
    EvidenceLineage,
    EvidenceKind,
    EvidenceQuality,
    parse_quality,
)
from schemas.scarcity_clock import ScarcityClock


SUPPORTED_BOUNDARY_SCHEMA_ID = "cantheon.metafield_boundary"
SUPPORTED_BOUNDARY_MAJOR = 0


class BoundaryAdapterError(ValueError):
    """Raised when the envelope is malformed or schema-incompatible."""


def check_boundary_schema(envelope: Dict[str, Any]) -> None:
    """Fail-closed schema check for the CANtheon boundary contract."""
    sid = envelope.get("boundary_schema_id") or envelope.get("schema_id") or ""
    sver = str(envelope.get("boundary_schema_version") or envelope.get("schema_version") or "")
    if sid and sid != SUPPORTED_BOUNDARY_SCHEMA_ID:
        if "kind" in envelope and envelope.get("kind") == "observation":
            pass
        else:
            raise BoundaryAdapterError(
                f"unsupported boundary_schema_id: {sid!r} "
                f"(expected {SUPPORTED_BOUNDARY_SCHEMA_ID!r})"
            )
    if sver:
        try:
            major = int(str(sver).split(".")[0])
        except ValueError:
            raise BoundaryAdapterError(f"unparseable boundary_schema_version: {sver!r}")
        if major != SUPPORTED_BOUNDARY_MAJOR:
            raise BoundaryAdapterError(
                f"incompatible boundary major version: {sver} "
                f"(supported major={SUPPORTED_BOUNDARY_MAJOR})"
            )


def _quality_to_health(q: EvidenceQuality) -> str:
    if q == EvidenceQuality.VALID:
        return "ok"
    if q == EvidenceQuality.STALE:
        return "degraded"
    if q in (EvidenceQuality.INVALID, EvidenceQuality.OUT_OF_RANGE, EvidenceQuality.DECODE_ERROR):
        return "error"
    return "unknown"


def _confidence_from_quality(q: EvidenceQuality) -> float:
    if q == EvidenceQuality.VALID:
        return 1.0
    if q == EvidenceQuality.STALE:
        return 0.3
    if q == EvidenceQuality.OUT_OF_RANGE:
        return 0.1
    return 0.0


def boundary_to_field_observation(envelope: Dict[str, Any]) -> FieldObservation:
    """Convert a CANtheon boundary envelope dict into FieldObservation."""
    if not isinstance(envelope, dict):
        raise BoundaryAdapterError("envelope must be a dict")

    check_boundary_schema(envelope)

    kind = str(envelope.get("kind") or "observation")
    if kind not in ("observation", "diagnostic", "health"):
        raise BoundaryAdapterError(f"unsupported envelope kind: {kind!r}")

    node_id = str(envelope.get("node_id") or "")
    if not node_id and kind == "observation":
        raise BoundaryAdapterError("node_id is required for observation envelopes")

    payload = envelope.get("payload") or {}
    if not isinstance(payload, dict):
        raise BoundaryAdapterError("payload must be a dict")

    quality = parse_quality(envelope.get("quality"))
    sequence = envelope.get("sequence")
    source_ts = envelope.get("source_timestamp_ns")
    ingest_ts = envelope.get("ingest_timestamp_ns")

    signal_id = str(payload.get("signal_id") or payload.get("message_name") or "value")
    unit = str(payload.get("unit") or "")
    value = payload.get("value")
    region = FieldRegion(
        region=signal_id,
        observed=float(value) if value is not None else None,
        confidence=_confidence_from_quality(quality),
        anomaly=0.0 if quality == EvidenceQuality.VALID else 1.0,
        unit=unit,
        extras={
            k: payload[k]
            for k in ("raw_value", "message_id", "message_name", "signal_name", "source")
            if k in payload
        },
    )

    prov = envelope.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}
    parent_ids: List[str] = []
    if prov.get("parent_observation_id"):
        parent_ids.append(str(prov["parent_observation_id"]))
    event_id = ""
    if prov.get("node_id") and prov.get("signal_id") is not None and prov.get("sequence") is not None:
        event_id = (
            f"{prov.get('node_id')}|0x{int(prov.get('can_message_id') or 0):03X}|"
            f"{prov.get('signal_id')}|{prov.get('sequence')}"
        )

    evidence_kind = EvidenceKind.RAW
    if kind in ("diagnostic", "health"):
        evidence_kind = EvidenceKind.ASSERTION

    lineage = EvidenceLineage(
        kind=evidence_kind,
        quality=quality,
        source_id=node_id,
        event_id=event_id,
        sequence=int(sequence) if sequence is not None else None,
        source_timestamp_ns=int(source_ts) if source_ts is not None else None,
        ingest_timestamp_ns=int(ingest_ts) if ingest_ts is not None else None,
        parent_event_ids=parent_ids,
        transformation_id="cantheon.metafield_boundary",
        transformation_version=str(envelope.get("boundary_schema_version") or "0.3.0"),
        unit=unit,
        schema_id=str(envelope.get("schema_id") or ""),
        schema_version=str(envelope.get("schema_version") or ""),
        boundary_schema_id=str(envelope.get("boundary_schema_id") or SUPPORTED_BOUNDARY_SCHEMA_ID),
        boundary_schema_version=str(envelope.get("boundary_schema_version") or "0.3.0"),
        raw_identity={
            k: prov[k]
            for k in ("raw_data_hex", "can_message_id", "can_message_id_hex", "signal_id")
            if k in prov
        },
    )

    body_type = "cantheon"
    frame_format = (payload.get("frame_format") or "").upper()
    if "CAN_FD" in frame_format:
        body_type = "can_fd"
    elif "CAN" in frame_format:
        body_type = "can"

    modality: Dict[str, Any] = {
        "cantheon": {
            "kind": kind,
            "message_id": payload.get("message_id"),
            "signal_id": payload.get("signal_id"),
        }
    }

    ts_ns = source_ts if source_ts is not None else ingest_ts
    timestamp = f"ns:{int(ts_ns)}" if ts_ns is not None else ""

    return FieldObservation(
        body_id=node_id or "unknown",
        body_type=body_type,
        excitation_id=int(sequence) if sequence is not None else None,
        field_regions=[region],
        geometry_state="calibrated" if quality == EvidenceQuality.VALID else "unknown",
        timestamp=timestamp or datetime_fallback(),
        clock=ScarcityClock.unanchored(),
        modality=modality,
        health=_quality_to_health(quality),
        schema_version=2,
        evidence=lineage,
    )


def datetime_fallback() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
