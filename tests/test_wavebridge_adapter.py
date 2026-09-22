"""Unit tests for the MetaField ↔ WaveBridge adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow local wavebridge clone
_WAVEBRIDGE_ROOT = Path("/home/workdir/artifacts/wavebridge")
if _WAVEBRIDGE_ROOT.exists() and str(_WAVEBRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAVEBRIDGE_ROOT))

# Metafield root
_META_ROOT = Path(__file__).resolve().parents[1]
if str(_META_ROOT) not in sys.path:
    sys.path.insert(0, str(_META_ROOT))


def test_field_state_to_wavebridge():
    from wavebridge_adapter import field_state_to_wavebridge
    from wavebridge.bridge import FieldPacket

    values = np.array([0.1, -0.3, 0.7], dtype=np.float32)
    packet = field_state_to_wavebridge(
        values, source="metafield", metadata={"body_id": "stub-01"}
    )
    assert isinstance(packet, FieldPacket)
    assert packet.weight_count == 3
    assert packet.metadata.get("source") == "metafield"
    assert packet.metadata.get("body_id") == "stub-01"


def test_wavebridge_observation_to_field():
    from wavebridge import encode_field_state, decode_field_observation
    from wavebridge_adapter import wavebridge_observation_to_field

    state = np.linspace(-1, 1, 20, dtype=np.float32)
    packet = encode_field_state(state, source="metafield")
    obs = decode_field_observation(packet, source="optical")
    field_dict = wavebridge_observation_to_field(
        obs, body_id="optical-01", excitation_id=3
    )
    assert field_dict["body_id"] == "optical-01"
    assert field_dict["excitation_id"] == 3
    assert len(field_dict["observed_response"]) == 20
    assert 0.0 <= field_dict["confidence"] <= 1.0


def test_observation_to_memory_entry():
    from wavebridge_adapter import (
        field_state_to_wavebridge,
        observation_to_memory_entry,
        wavebridge_observation_to_field,
    )
    from wavebridge import decode_field_observation

    state = np.ones(8, dtype=np.float32) * 0.5
    packet = field_state_to_wavebridge(state)
    obs = decode_field_observation(packet, source="sim")
    field_dict = wavebridge_observation_to_field(obs, body_id="sim-body")
    entry = observation_to_memory_entry(field_dict)
    assert entry.body_id == "sim-body"
    assert entry.observed_response is not None
    assert len(entry.observed_response) == 8
