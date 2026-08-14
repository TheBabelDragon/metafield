"""
observables.py — extract the sonify JSONL record from a live MetaField state.

Kept separate so the HMC loop only needs one call site and so unit tests
can feed synthetic gauge tensors without spinning up full HMC.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def mean_plaquette(gauge) -> float:
    """Average ReTr(U_p)/N over all plaquettes — confinement order parameter."""
    traces = gauge.plaquette_traces()
    return float(traces.mean().real.item())


def color_field_means(U: "torch.Tensor") -> List[float]:
    """
    Mean |U| magnitude per SU(N) color index (diagonal-ish proxy).

    U shape: lattice + (n_dims, N, N). We average |U[..., c, c]| over
    lattice and directions for each color c. Cheap and musically useful.
    """
    n = U.shape[-1]
    out = []
    for c in range(n):
        diag = U[..., c, c]
        out.append(float(torch.abs(diag).mean().real.item()))
    return out


def topological_charge_proxy(U: "torch.Tensor") -> float:
    """
    Cheap proxy — not the integer topological charge.

    Production codes use the clover (or Lüscher) definition; that needs
    careful lattice continuum matching. Here we use a global phase-ish
    scalar so the spike detector has signal. Replace with a real Q once ready.
    """
    phase = torch.angle(torch.diagonal(U, dim1=-2, dim2=-1).sum(-1)).mean()
    return float(phase.real.item()) if torch.is_complex(phase) else float(phase.item())


def dirac_eigmin_proxy(cg_residual: float, cg_iters: int) -> float:
    """
    Chiral-condensate stand-in until a real lowest-eigenvalue solve exists.

    Small residual + few iters → well-conditioned Dirac → larger effective
    mass gap proxy. Inverted and clamped for a musically useful 0–1 range.
    """
    if not math.isfinite(cg_residual):
        return float("nan")
    score = -math.log10(max(cg_residual, 1e-16)) / 16.0
    score *= 50.0 / max(cg_iters, 1)
    return max(0.0, min(1.0, score))


def build_record(
    *,
    step: int,
    traj_id: int,
    accepted: bool,
    gauge,
    fisher_curvature: Optional[float] = None,
    cg_residual: float = float("nan"),
    cg_iters: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one JSONL-ready observable dict."""
    U = gauge.U
    rec: Dict[str, Any] = {
        "step": int(step),
        "traj_id": int(traj_id),
        "accepted": bool(accepted),
        "plaquette": mean_plaquette(gauge),
        "topological_charge": topological_charge_proxy(U),
        "fisher_curvature": (
            float(fisher_curvature)
            if fisher_curvature is not None and math.isfinite(float(fisher_curvature))
            else None
        ),
        "color_fields": color_field_means(U),
        "dirac_eigmin": dirac_eigmin_proxy(cg_residual, cg_iters),
    }
    if extra:
        rec.update(extra)
    return rec


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one record as a single JSON line."""
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
