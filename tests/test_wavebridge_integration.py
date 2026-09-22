"""
End-to-end synthetic integration test (no hardware):

  field_state
    → wavebridge encode
    → synthetic optical observation
    → field observation
    → FieldMemoryStore
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_WAVEBRIDGE_ROOT = Path("/home/workdir/artifacts/wavebridge")
if _WAVEBRIDGE_ROOT.exists() and str(_WAVEBRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAVEBRIDGE_ROOT))

_META_ROOT = Path(__file__).resolve().parents[1]
if str(_META_ROOT) not in sys.path:
    sys.path.insert(0, str(_META_ROOT))


def test_field_state_to_memory_store():
    from wavebridge.channel import ChannelSpec, simulate_channel
    from wavebridge.codec import decode_pcm16, encode_pcm16
    from wavebridge import decode_field_observation

    from wavebridge_adapter import (
        field_state_to_wavebridge,
        observation_to_memory_entry,
        wavebridge_observation_to_field,
    )
    from field_memory_store import FieldMemoryStore

    rng = np.random.default_rng(0)
    field_state = rng.standard_normal(64).astype(np.float32)

    # Encode via adapter
    packet = field_state_to_wavebridge(
        field_state,
        source="metafield",
        metadata={"body_id": "optical-stub-01", "excitation_id": 0},
    )

    # Simulate optical channel on normalized samples
    pcm = np.frombuffer(packet.payload, dtype="<i2")
    samples = decode_pcm16(pcm)
    captured = simulate_channel(
        samples,
        sample_rate=packet.sample_rate,
        spec=ChannelSpec(gain=0.95, noise_rms=0.001, seed=2),
    )
    # Reconstruct scaled values as the "optical observation"
    optical_values = captured * packet.peak_scale

    obs = decode_field_observation(
        optical_values, source="optical", channel="sim-bpw34"
    )
    field_dict = wavebridge_observation_to_field(
        obs,
        body_id="optical-stub-01",
        body_type="optical",
        excitation_id=0,
    )
    entry = observation_to_memory_entry(field_dict)

    store = FieldMemoryStore(soft_capacity=32)
    store.add(entry)
    assert len(store) == 1
    stats = store.get_stats()
    assert stats["size"] == 1
    assert stats["total_added"] == 1

    # Recovered response should be close to original state
    recovered = np.asarray(entry.observed_response, dtype=np.float32)
    scale = float(np.max(np.abs(field_state))) or 1.0
    rel = float(np.max(np.abs(recovered - field_state)) / scale)
    assert rel < 0.05
