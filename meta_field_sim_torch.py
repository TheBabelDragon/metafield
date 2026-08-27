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


# ---------------------------------------------------------------------------
# su(N) algebra helpers
# ---------------------------------------------------------------------------

def dagger(M: "torch.Tensor") -> "torch.Tensor":
    return M.conj().transpose(-1, -2)


def project_traceless_antihermitian(M: "torch.Tensor") -> "torch.Tensor":
    """Project onto su(N): traceless, anti-Hermitian."""
    n = M.shape[-1]
    A = 0.5 * (M - dagger(M))
    tr = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    A = A - (tr / n)[..., None, None] * eye
    return A


def project_traceless_hermitian(M: "torch.Tensor") -> "torch.Tensor":
    """Project onto traceless Hermitian matrices (used for HMC momenta)."""
    n = M.shape[-1]
    H = 0.5 * (M + dagger(M))
    tr = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    H = H - (tr / n)[..., None, None] * eye
    return H


def expm_anti_hermitian(X: "torch.Tensor") -> "torch.Tensor":
    """
    Exact exponential of a batch of anti-Hermitian matrices via a
    Hermitian eigendecomposition: H = i X is Hermitian, so
    exp(X) = exp(-i H) = V diag(exp(-i lambda)) V^dagger with lambda
    real. Result is exactly unitary regardless of step size.
    """
    H = 1j * X
    eigvals, eigvecs = torch.linalg.eigh(H)              # eigvals real, (..., N)
    phase = torch.exp(-1j * eigvals.to(X.dtype))
    Vh = dagger(eigvecs)
    scaled_Vh = phase[..., :, None] * Vh
    return eigvecs @ scaled_Vh


def random_su_n_hermitian(shape: Tuple[int, ...], n: int, dtype, device,
                           generator: "torch.Generator") -> "torch.Tensor":
    """Random traceless-Hermitian matrix batch (for HMC momenta / noise)."""
    real = torch.randn(shape, generator=generator, dtype=torch.float64, device=device)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64, device=device)
    A = (real + 1j * imag).to(dtype)
    H = 0.5 * (A + dagger(A))
    tr = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n, dtype=dtype, device=device)
    H = H - (tr / n)[..., None, None] * eye
    return H
