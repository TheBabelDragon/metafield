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


# ---------------------------------------------------------------------------
# Lattice geometry (torch)
# ---------------------------------------------------------------------------

class LatticeV2:
    def __init__(self, config: ConfigV2):
        self.L = config.L
        self.n_dims = config.n_dims
        self.shape = tuple([self.L] * self.n_dims)
        self.volume = self.L ** self.n_dims

    def shift(self, field: "torch.Tensor", axis: int, direction: int) -> "torch.Tensor":
        """shift(f, mu, +1)(x) == f(x + e_mu); shift(f, mu, -1)(x) == f(x - e_mu)."""
        return torch.roll(field, shifts=-direction, dims=axis)


# ---------------------------------------------------------------------------
# Gamma matrices (Euclidean, Degrand-DeTar chiral-like convention)
# ---------------------------------------------------------------------------

def euclidean_gamma_matrices(dtype, device) -> "torch.Tensor":
    """
    Returns a (4, 4, 4) tensor: gamma[mu] is a 4x4 Hermitian matrix,
    satisfying the Euclidean Clifford algebra {gamma_mu, gamma_nu} =
    2 delta_{mu nu} I. Convention follows Degrand & DeTar, \"Lattice
    Methods for Quantum Chromodynamics\".
    """
    i = 1j
    g1 = torch.tensor([[0, 0, 0, -i],
                        [0, 0, -i, 0],
                        [0, i, 0, 0],
                        [i, 0, 0, 0]], dtype=dtype, device=device)
    g2 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0]], dtype=dtype, device=device)
    g3 = torch.tensor([[0, 0, -i, 0],
                        [0, 0, 0, i],
                        [i, 0, 0, 0],
                        [0, -i, 0, 0]], dtype=dtype, device=device)
    g4 = torch.tensor([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]], dtype=dtype, device=device)
    return torch.stack([g1, g2, g3, g4], dim=0)


def gamma5(gammas: "torch.Tensor") -> "torch.Tensor":
    g5 = gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
    return g5


# ---------------------------------------------------------------------------
# Gauge field: SU(N) links, real Wilson action, autograd-derived force
# ---------------------------------------------------------------------------

class GaugeFieldV2:
    def __init__(self, lattice: LatticeV2, config: ConfigV2, generator: "torch.Generator"):
        self.lattice = lattice
        self.config = config
        self.n = config.color_dim
        self.device = torch.device(config.device)
        shape = lattice.shape + (config.n_dims, self.n, self.n)

        eye = torch.eye(self.n, dtype=config.dtype, device=self.device)
        eye = eye.expand(shape).clone()
        noise = random_su_n_hermitian(shape, self.n, config.dtype, self.device, generator)
        X = 1j * 0.1 * project_traceless_hermitian(noise)  # small random cold-ish start
        self.U = expm_anti_hermitian(project_traceless_antihermitian(X)) @ eye

    def plaquette_traces(self, U: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Re Tr(U_plaquette)/N for every site and every mu<nu plane."""
        U = self.U if U is None else U
        lat = self.lattice
        traces = []
        for mu in range(lat.n_dims):
            for nu in range(mu + 1, lat.n_dims):
                U_mu = U[..., mu, :, :]
                U_nu = U[..., nu, :, :]
                U_nu_xpm = lat.shift(U_nu, mu, +1)
                U_mu_xpn = lat.shift(U_mu, nu, +1)
                plaq = U_mu @ U_nu_xpm @ dagger(U_mu_xpn) @ dagger(U_nu)
                tr = torch.diagonal(plaq, dim1=-2, dim2=-1).sum(-1).real / self.n
                traces.append(tr)
        return torch.stack(traces, dim=-1)

    def wilson_action(self, U: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Scalar Wilson gauge action S_G = beta * sum_plaquettes (1 - ReTr(U_p)/N)."""
        traces = self.plaquette_traces(U)
        return self.config.beta * torch.sum(1.0 - traces)

    def force(self) -> "torch.Tensor":
        """
        dS/dU via automatic differentiation, then projected onto the
        su(N) algebra so it can be used directly as an HMC force. This
        is the \"differentiable physics\" piece: instead of hand-deriving
        a staple formula, we let autograd differentiate the real
        Wilson action.
        """
        U = self.U.detach().clone().requires_grad_(True)
        S = self.wilson_action(U)
        (grad,) = torch.autograd.grad(S, U)
        # grad here is dS/dU* in PyTorch's Wirtinger convention; project
        # U @ grad^dagger onto su(N) to get a valid algebra-valued force,
        # mirroring the analytic staple-force construction in the NumPy version.
        raw = U.detach() @ dagger(grad.detach())
        return project_traceless_antihermitian(raw)
