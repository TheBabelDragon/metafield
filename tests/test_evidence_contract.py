"""Phase 3: MetaField semantic mapping of the shared evidence contract."""

from schemas.field_observation import (
    FieldObservation,
    FieldRegion,
    validate_observation,
)
from schemas.evidence import (
    EvidenceKind,
    EvidenceQuality,
    EvidenceLineage,
    parse_quality,
)
from schemas.field_memory import FieldMemoryEntry
from adapters.cantheon_boundary import (
    boundary_to_field_observation,
    BoundaryAdapterError,
)


def _sample_envelope(**overrides):
    base = {
        "kind": "observation",
        "boundary_schema_id": "cantheon.metafield_boundary",
        "boundary_schema_version": "0.3.0",
        "node_id": "temp_sensor_01",
        "payload": {
            "message_id": 256,
            "signal_id": "temperature",
            "value": 25.0,
            "unit": "\u00b0C",
            "raw_value": 250,
            "message_name": "TemperatureStatus",
            "signal_name": "Temperature",
            "source": "can",
        },
        "quality": "VALID",
        "sequence": 1,
        "ingest_timestamp_ns": 1000,
        "source_timestamp_ns": 500,
        "timestamp_ns": 1000,
        "schema_id": "temp.v1",
        "schema_version": "1.0",
        "provenance": {
            "node_id": "temp_sensor_01",
            "can_message_id": 256,
            "can_message_id_hex": "0x100",
            "signal_id": "temperature",
            "sequence": 1,
            "timestamp_ns": 1000,
            "schema_version": "1.0",
            "schema_id": "temp.v1",
            "raw_data_hex": "fa00",
        },
    }
    base.update(overrides)
    return base


def test_parse_quality():
    assert parse_quality("VALID") == EvidenceQuality.VALID
    assert parse_quality("DECODE_ERROR") == EvidenceQuality.DECODE_ERROR
    assert parse_quality(None) == EvidenceQuality.UNKNOWN
    assert parse_quality("nope") == EvidenceQuality.UNKNOWN


def test_evidence_roundtrip():
    e = EvidenceLineage(
        kind=EvidenceKind.RAW,
        quality=EvidenceQuality.VALID,
        source_id="n1",
        sequence=3,
        unit="kPa",
        parent_event_ids=["p1"],
    )
    d = e.to_dict()
    e2 = EvidenceLineage.from_dict(d)
    assert e2 is not None
    assert e2.kind == EvidenceKind.RAW
    assert e2.quality == EvidenceQuality.VALID
    assert e2.source_id == "n1"
    assert e2.sequence == 3
    assert e2.parent_event_ids == ["p1"]


def test_field_observation_without_evidence_still_works():
    obs = FieldObservation(
        body_id="optical_01",
        body_type="optical",
        field_regions=[FieldRegion(region="center", observed=1.0, confidence=0.9)],
    )
    d = obs.to_dict()
    assert "evidence" not in d
    obs2 = FieldObservation.from_dict(d)
    assert obs2.evidence is None
    assert validate_observation(obs2) == []


def test_field_observation_with_evidence():
    obs = FieldObservation(
        body_id="temp_sensor_01",
        body_type="cantheon",
        field_regions=[FieldRegion(region="temperature", observed=25.0, unit="\u00b0C")],
        evidence=EvidenceLineage(
            kind=EvidenceKind.RAW,
            quality=EvidenceQuality.VALID,
            source_id="temp_sensor_01",
            sequence=1,
            unit="\u00b0C",
        ),
    )
    d = obs.to_dict()
    assert "evidence" in d
    obs2 = FieldObservation.from_dict(d)
    assert obs2.evidence is not None
    assert obs2.evidence.quality == EvidenceQuality.VALID


def test_boundary_to_observation_preserves_lineage():
    env = _sample_envelope()
    obs = boundary_to_field_observation(env)
    assert obs.body_id == "temp_sensor_01"
    assert abs(obs.field_regions[0].observed - 25.0) < 1e-9
    assert obs.evidence is not None
    assert obs.evidence.kind == EvidenceKind.RAW
    assert obs.evidence.quality == EvidenceQuality.VALID
    assert obs.evidence.sequence == 1
    assert obs.evidence.source_timestamp_ns == 500
    assert obs.evidence.ingest_timestamp_ns == 1000
    assert obs.evidence.raw_identity.get("raw_data_hex") == "fa00"
    assert validate_observation(obs) == []


def test_boundary_rejects_bad_major_version():
    env = _sample_envelope(boundary_schema_version="1.0.0")
    try:
        boundary_to_field_observation(env)
        raised = False
    except BoundaryAdapterError:
        raised = True
    assert raised


def test_boundary_rejects_missing_node_id():
    env = _sample_envelope()
    del env["node_id"]
    try:
        boundary_to_field_observation(env)
        raised = False
    except BoundaryAdapterError:
        raised = True
    assert raised


def test_invalid_quality_not_silently_valid():
    env = _sample_envelope(quality="DECODE_ERROR")
    obs = boundary_to_field_observation(env)
    assert obs.evidence.quality == EvidenceQuality.DECODE_ERROR
    assert obs.health == "error"


def test_memory_entry_from_observation_with_evidence():
    env = _sample_envelope()
    obs = boundary_to_field_observation(env)
    entry = FieldMemoryEntry.from_observation(obs.to_dict())
    assert entry.body_id == "temp_sensor_01"
    assert abs(entry.observed_response[0] - 25.0) < 1e-9


def test_parent_lineage():
    env = _sample_envelope()
    env["provenance"]["parent_observation_id"] = "parent-42"
    obs = boundary_to_field_observation(env)
    assert "parent-42" in obs.evidence.parent_event_ids


def test_kinds_distinguished():
    raw = boundary_to_field_observation(_sample_envelope())
    assert raw.evidence.kind == EvidenceKind.RAW
    diag = boundary_to_field_observation(
        _sample_envelope(kind="diagnostic", quality="INVALID")
    )
    assert diag.evidence.kind == EvidenceKind.ASSERTION
