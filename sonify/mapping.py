"""
mapping.py — translate one MetaField observable record into AXIOM note dicts.

Default mapping (tune freely):
  accepted              → drum hit (kick=36 ACC, snare=38 REJ)
  plaquette             → bass pitch (D-minor pentatonic, slow order parameter)
  color_fields[0..2]    → three melodic voices (piano / pluck / strings)
  fisher_curvature      → velocity (louder near phase structure) + optional BPM bias
  topological_charge    → crash (49) + octave accent when spike vs rolling baseline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import math

# ---------------------------------------------------------------------------
# Musical constants (edit these, not the plumbing)
# ---------------------------------------------------------------------------

# D minor pentatonic, MIDI note numbers (D3–D5 range for bass; higher for melody)
D_MIN_PENT = [38, 41, 43, 46, 48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72]

# Track names must match what AXIOM expects after the 7th-track patch
TRACK_DRUMS = "drums"
TRACK_BASS = "bass"
TRACK_COLOR0 = "color0"   # piano / pluck
TRACK_COLOR1 = "color1"
TRACK_COLOR2 = "color2"
TRACK_PHYSICS = "physics" # strings accent / crash layer

# Drum map (General MIDI)
KICK = 36
SNARE = 38
CRASH = 49

# Defaults
DEFAULT_TICKS_PER_STEP = 8
DEFAULT_NOTE_DUR = 6          # ticks
DEFAULT_VEL = 80
VEL_MIN, VEL_MAX = 40, 127

# Topological spike detection
TOPO_SPIKE_SIGMA = 2.5        # standard deviations above rolling mean


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _quantize_to_scale(value: float, scale: Sequence[int],
                       vmin: float, vmax: float) -> int:
    """Map a continuous observable into a scale degree."""
    if vmax <= vmin:
        idx = len(scale) // 2
    else:
        t = _clamp((value - vmin) / (vmax - vmin), 0.0, 1.0)
        idx = int(round(t * (len(scale) - 1)))
    return scale[idx]


def _vel_from_curvature(curv: Optional[float],
                        baseline: float = 0.0,
                        scale: float = 5.0) -> int:
    """Louder near large |curvature| (phase-transition proxy)."""
    if curv is None or not math.isfinite(curv):
        return DEFAULT_VEL
    # soft sigmoid around baseline
    x = abs(curv - baseline) / max(1e-6, scale)
    v = VEL_MIN + (VEL_MAX - VEL_MIN) * (1.0 - math.exp(-x))
    return int(_clamp(v, VEL_MIN, VEL_MAX))


class TopoBaseline:
    """Rolling mean / std for topological charge spike detection."""

    def __init__(self, window: int = 32):
        self.window = window
        self._buf: List[float] = []

    def update(self, q: float) -> bool:
        """Return True if this q is a spike relative to history."""
        if not math.isfinite(q):
            return False
        is_spike = False
        if len(self._buf) >= 8:
            mean = sum(self._buf) / len(self._buf)
            var = sum((x - mean) ** 2 for x in self._buf) / len(self._buf)
            std = math.sqrt(var) + 1e-9
            if abs(q - mean) > TOPO_SPIKE_SIGMA * std:
                is_spike = True
        self._buf.append(q)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        return is_spike


# Module-level baseline so consecutive records share history
_topo_baseline = TopoBaseline()


def map_record_to_notes(
    record: Dict[str, Any],
    tick: int,
    *,
    ticks_per_step: int = DEFAULT_TICKS_PER_STEP,
    plaquette_range: tuple = (0.0, 1.0),
    color_range: tuple = (0.0, 1.0),
    note_dur: int = DEFAULT_NOTE_DUR,
    reset_topo_baseline: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert one JSONL observable record into a list of AXIOM note dicts.

    Each note:
      {"midi": int, "start": int, "dur": int, "vel": int, "track": str}
    """
    if reset_topo_baseline:
        global _topo_baseline
        _topo_baseline = TopoBaseline()

    notes: List[Dict[str, Any]] = []
    accepted = bool(record.get("accepted", False))
    plaquette = float(record.get("plaquette", 0.5))
    topo = float(record.get("topological_charge", 0.0))
    curv = record.get("fisher_curvature")
    curv_f = float(curv) if curv is not None else None
    colors = record.get("color_fields") or [0.0, 0.0, 0.0]
    if len(colors) < 3:
        colors = list(colors) + [0.0] * (3 - len(colors))

    vel = _vel_from_curvature(curv_f)

    # --- drums: acceptance rhythm ---
    drum_midi = KICK if accepted else SNARE
    notes.append({
        "midi": drum_midi,
        "start": tick,
        "dur": max(2, note_dur // 2),
        "vel": vel,
        "track": TRACK_DRUMS,
    })

    # --- bass: plaquette (order parameter) ---
    bass_midi = _quantize_to_scale(
        plaquette, D_MIN_PENT[:8],  # lower octave half
        plaquette_range[0], plaquette_range[1],
    )
    notes.append({
        "midi": bass_midi,
        "start": tick,
        "dur": note_dur,
        "vel": vel,
        "track": TRACK_BASS,
    })

    # --- three color voices ---
    color_tracks = [TRACK_COLOR0, TRACK_COLOR1, TRACK_COLOR2]
    # stagger the three voices slightly in pitch register
    register_offsets = [4, 6, 8]  # indices into D_MIN_PENT
    for i, (cval, trk, off) in enumerate(zip(colors, color_tracks, register_offsets)):
        scale_slice = D_MIN_PENT[off:off + 7]
        if not scale_slice:
            scale_slice = D_MIN_PENT
        midi = _quantize_to_scale(
            float(cval), scale_slice, color_range[0], color_range[1],
        )
        notes.append({
            "midi": midi,
            "start": tick,
            "dur": note_dur,
            "vel": max(VEL_MIN, vel - 10 * i),
            "track": trk,
        })

    # --- topological charge spike → crash + accent ---
    if _topo_baseline.update(topo):
        notes.append({
            "midi": CRASH,
            "start": tick,
            "dur": note_dur * 2,
            "vel": min(VEL_MAX, vel + 20),
            "track": TRACK_DRUMS,
        })
        # octave spike on the physics track
        accent = bass_midi + 12
        notes.append({
            "midi": _clamp(accent, 24, 96),
            "start": tick,
            "dur": note_dur,
            "vel": min(VEL_MAX, vel + 15),
            "track": TRACK_PHYSICS,
        })

    return notes
