#!/usr/bin/env python3
"""
optical_wavebridge_hook.py

Thin integration hook that wires the existing optical-body path
through WaveBridge without pulling hardware or WaveBridge internals
into MetaField core.

Intended closed loop (synthetic or physical):

  MetaField state
      ↓
  WaveBridge encode  (via wavebridge_adapter)
      ↓
  physical excitation  (SimulatedChannel | AudioChannel | OpticalChannel)
      ↓
  optical body / BPW34 observation
      ↓
  WaveBridge decode
      ↓
  MetaField observation → FieldMemoryStore
      ↓
  learned geometry / predictor / active_probe
      ↓
  next excitation

Hardware (ESP32, laser, PAM, TIA, serial) stays behind a future
PhysicalChannel interface and is intentionally absent from this module.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

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


def ingest_optical_observation(
    values: Any,
    store,
    *,
    body_id: str = "optical-stub-01",
    excitation_id: Optional[int] = None,
    source: str = "optical",
):
    """
    Decode (or wrap) optical samples, convert to FieldMemoryEntry, add to store.

    `values` may be a raw array, a WaveBridge Observation, or a FieldPacket.
    """
    from wavebridge import decode_field_observation

    obs = decode_field_observation(values, source=source)
    field_dict = wavebridge_observation_to_field(
        obs,
        body_id=body_id,
        body_type="optical",
        excitation_id=excitation_id,
    )
    entry = observation_to_memory_entry(field_dict)
    store.add(entry)
    return entry
