#!/usr/bin/env python3
"""
wavebridge_adapter.py

MetaField-side adapter for WaveBridge.

Owns the MetaField-specific translation.  WaveBridge itself knows
nothing about FieldObservation, FieldMemoryStore, or lattice geometry.

Conceptual flow:

  field_state
      ↓
  MetaField adapter
      ↓
  WaveBridge (encode_field_state)
      ↓
  physical waveform

  physical waveform
      ↓
  WaveBridge (decode)
      ↓
  MetaField adapter
      ↓
  FieldMemoryStore / learned geometry
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

# WaveBridge is an optional peer dependency.  Import lazily so MetaField
# still loads when wavebridge is not installed (e.g. lattice-only runs).
try:
    from wavebridge import (
        Observation as WBObservation,
        decode_field_observation,
        encode_field_state,
    )
    from wavebridge.bridge import FieldPacket

    _HAS_WAVEBRIDGE = True
except ImportError:  # pragma: no cover
    _HAS_WAVEBRIDGE = False
    FieldPacket = Any  # type: ignore
    WBObservation = Any  # type: ignore


def _require_wavebridge() -> None:
    if not _HAS_WAVEBRIDGE:
        raise ImportError(
            "wavebridge is required for wavebridge_adapter. "
            "Install/clone TheBabelDragon/wavebridge and ensure it is on PYTHONPATH."
        )


def field_state_to_wavebridge(
    values: Any,
    *,
    source: str = "metafield",
    metadata: Optional[Mapping[str, Any]] = None,
    sample_rate: int = 44100,
) -> FieldPacket:
    """
    Translate a MetaField field-state array into a WaveBridge packet.

    Parameters
    ----------
    values :
        Array-like field state (e.g. excitation amplitudes, latent vector).
    source :
        Provenance tag (default "metafield").
    metadata :
        Optional MetaField metadata (body_id, excitation_id, …).
    sample_rate :
        Logical sample rate for the eventual waveform.
    """
    _require_wavebridge()
    meta = dict(metadata or {})
    return encode_field_state(
        values,
        source=source,
        metadata=meta,
        sample_rate=sample_rate,
    )


def wavebridge_observation_to_field(
    observation: Union[WBObservation, Any],
    *,
    body_id: str = "optical-wavebridge",
    body_type: str = "optical",
    excitation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Translate a WaveBridge Observation into a MetaField-friendly dict
    that can feed FieldMemoryStore / FieldObservation construction.

    Returns a plain dict with:
      body_id, body_type, excitation_id,
      observed_response (list[float]),
      confidence, anomaly, modality, source, metadata
    """
    _require_wavebridge()

    if not isinstance(observation, WBObservation):
        # Accept raw array as a convenience
        values = np.asarray(observation, dtype=np.float32)
        source = "optical"
        channel = None
        meta: Dict[str, Any] = {}
    else:
        values = np.asarray(observation.values, dtype=np.float32)
        source = observation.source
        channel = observation.channel
        meta = dict(observation.metadata or {})

    flat = values.reshape(-1)
    # Simple confidence heuristic from dynamic range
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    confidence = 0.95 if peak > 0.05 else 0.6
    anomaly = 0.0

    return {
        "body_id": body_id,
        "body_type": body_type,
        "excitation_id": excitation_id,
        "observed_response": flat.tolist(),
        "confidence": confidence,
        "anomaly": anomaly,
        "modality": {
            "source": source,
            "channel": channel,
            "n_samples": int(flat.size),
            "peak": peak,
        },
        "metadata": meta,
    }


def observation_to_memory_entry(
    field_dict: Mapping[str, Any],
):
    """
    Build a FieldMemoryEntry from the adapter output dict.
    Imports schemas only when called so the adapter stays light.
    """
    from schemas.field_memory import FieldMemoryEntry

    return FieldMemoryEntry(
        body_id=str(field_dict.get("body_id", "unknown")),
        excitation_id=field_dict.get("excitation_id"),
        observed_response=field_dict.get("observed_response"),
        confidence=float(field_dict.get("confidence", 0.0)),
        anomaly=float(field_dict.get("anomaly", 0.0)),
        extras={
            "modality": field_dict.get("modality"),
            "metadata": field_dict.get("metadata"),
            "body_type": field_dict.get("body_type"),
        },
    )
