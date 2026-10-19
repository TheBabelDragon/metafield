"""MetaField observation API discovery.

Consumes CANtheon boundary envelopes via adapters.cantheon_boundary.
Does not enable Redis publication or physical control as a side effect.
"""

from __future__ import annotations

from typing import Any, Optional

API_SCHEMA_ID = "metafield.observation_api"
API_SCHEMA_VERSION = "1.0.0"


def health() -> dict[str, Any]:
    return {
        "schema_id": API_SCHEMA_ID,
        "schema_version": API_SCHEMA_VERSION,
        "healthy": True,
        "capabilities": [
            "cantheon_boundary_consume",
            "field_observation",
            "evidence_lineage",
            "local_file_export",
        ],
        "non_goals": [
            "redis_publish_by_default",
            "remote_commands",
            "physical_control",
            "web_server_for_sensing",
        ],
        "expected_upstream": {
            "schema_id": "cantheon.metafield_boundary",
            "schema_version": "0.3.0",
        },
    }


def convert_envelope(envelope: dict[str, Any]) -> Any:
    """Convert a CANtheon boundary envelope to FieldObservation(s).

    Raises on incompatible schema. Preserves quality and timestamps.
    """
    from adapters.cantheon_boundary import envelope_to_field_observations

    return envelope_to_field_observations(envelope)


def status_snapshot(*, schema_version: Optional[str] = None) -> dict[str, Any]:
    return {
        "schema_id": API_SCHEMA_ID,
        "schema_version": schema_version or API_SCHEMA_VERSION,
        "health": health(),
    }
