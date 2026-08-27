from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any, List

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "meta_field_sim_torch.py requires PyTorch. Install it with:\n"
        "    pip install torch\n"
        "(GPU build if you have CUDA: see https://pytorch.org/get-started/locally/)"
    ) from e


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConfigV2:
    L: int = 4                    # keep small by default -- Wilson-Dirac + CG is much
    n_dims: int = 4                 # heavier per step than the NumPy toy version
    color_dim: int = 3
    spinor_dim: int = 4             # 4-component Euclidean Dirac spinor

    mass: float = 0.1               # bare fermion mass
    wilson_r: float = 1.0           # Wilson parameter (r=1 standard, lifts doublers)

    beta: float = 5.5               # inverse gauge coupling (Wilson gauge action)

    hmc_n_leapfrog: int = 10
    hmc_step_size: float = 0.05
    hmc_trajectories: int = 20

    include_fermions: bool = False   # False = quenched (gauge-only) HMC, as before.
                                       # True = dynamical fermions via pseudofermion heatbath.
    cg_tol: float = 1e-8
    cg_maxiter: int = 200
    # Production HMC codes typically use a looser CG tolerance during the
    # molecular-dynamics force evaluations (called many times per trajectory,
    # speed matters) and a tight tolerance for the Metropolis energy check
    # (accuracy matters, called only twice per trajectory). Mirrored here.
    cg_tol_md: float = 1e-6
    cg_tol_action: float = 1e-10

    seed: int = 0
    device: str = "cpu"             # set to "cuda" if available
    dtype: torch.dtype = torch.complex128
