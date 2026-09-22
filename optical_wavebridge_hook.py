#!/usr/bin/env python3
"""
optical_wavebridge_hook.py

Thin integration hook that wires the optical-body path through WaveBridge
using the physical packet boundary (SimulatedChannel or any PhysicalChannel).

  MetaField state
      ↓  encode_excitation
  FieldPacket
      ↓  packet_roundtrip(channel)
  Observation
      ↓  ingest_wavebridge_observation
  FieldMemoryStore
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from wavebridge_adapter import (
    field_state_to_wavebridge,
    observation_to_memory_entry,
    wavebridge_observation_to_field,
)


def encode_excitation(
    state: Any,
    *,
    body_id: str = "optical-stub-01",
    excitation_id: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    """Encode a MetaField excitation state into a WaveBridge packet."""
    meta = dict(metadata or {})
    meta.setdefault("body_id", body_id)
    if excitation_id is not None:
        meta.setdefault("excitation_id", excitation_id)
    return field_state_to_wavebridge(state, source="metafield", metadata=meta)


def transmit_excitation(
    state: Any,
    channel,
    *,
    body_id: str = "optical-stub-01",
    excitation_id: Optional[int] = None,
    leading_silence: int = 100,
    trailing_silence: int = 50,
):
    """
    Encode state and run packet_roundtrip on the given PhysicalChannel.

    Returns a WaveBridge Observation.
    """
    from wavebridge import packet_roundtrip

    packet = encode_excitation(
        state, body_id=body_id, excitation_id=excitation_id
    )
    return packet_roundtrip(
        packet,
        channel,
        leading_silence=leading_silence,
        trailing_silence=trailing_silence,
    )


def ingest_wavebridge_observation(
    observation,
    store,
    *,
    body_id: str = "optical-stub-01",
    excitation_id: Optional[int] = None,
):
    """Convert Observation → FieldMemoryEntry → store.add."""
    field_dict = wavebridge_observation_to_field(
        observation,
        body_id=body_id,
        body_type="optical",
        excitation_id=excitation_id,
    )
    entry = observation_to_memory_entry(field_dict)
    store.add(entry)
    return entry


def ingest_optical_observation(
    values: Any,
    store,
    *,
    body_id: str = "optical-stub-01",
    excitation_id: Optional[int] = None,
    source: str = "optical",
):
    from wavebridge import decode_field_observation

    obs = decode_field_observation(values, source=source)
    return ingest_wavebridge_observation(
        obs, store, body_id=body_id, excitation_id=excitation_id
    )
