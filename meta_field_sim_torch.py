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
    2 delta_{mu nu} I. Convention follows Degrand & DeTar, "Lattice
    Methods for Quantum Chromodynamics".
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
        is the "differentiable physics" piece: instead of hand-deriving
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


# ---------------------------------------------------------------------------
# Wilson-Dirac fermion field
# ---------------------------------------------------------------------------

class WilsonDiracOperator:
    """
    The real Wilson-Dirac operator:

        (D psi)(x) = (m + 4r) psi(x)
            - (1/2) sum_mu [
                  (r*I - gamma_mu) (x) U_mu(x)   psi(x+mu)
                + (r*I + gamma_mu) (x) U_mu(x-mu)^dagger psi(x-mu)
              ]

    acting on a field psi(x) with shape lattice + (spinor=4, color=N).
    Uses the gamma5-Hermiticity identity D^dagger = gamma5 D gamma5
    (standard for Wilson fermions) instead of a separately coded
    adjoint operator.
    """

    def __init__(self, lattice: LatticeV2, config: ConfigV2):
        self.lattice = lattice
        self.config = config
        self.gammas = euclidean_gamma_matrices(config.dtype, torch.device(config.device))
        self.g5 = gamma5(self.gammas)
        eye4 = torch.eye(4, dtype=config.dtype, device=torch.device(config.device))
        self.r_plus = [config.wilson_r * eye4 + g for g in self.gammas]
        self.r_minus = [config.wilson_r * eye4 - g for g in self.gammas]

    def apply(self, psi: "torch.Tensor", U: "torch.Tensor") -> "torch.Tensor":
        cfg = self.config
        lat = self.lattice
        out = (cfg.mass + cfg.n_dims * cfg.wilson_r) * psi
        for mu in range(lat.n_dims):
            U_mu = U[..., mu, :, :]
            U_mu_back = lat.shift(U_mu, mu, -1)

            psi_fwd = lat.shift(psi, mu, +1)          # psi(x+mu), (..., 4, N)
            psi_back = lat.shift(psi, mu, -1)          # psi(x-mu)

            transported_fwd = torch.einsum('...ij,...sj->...si', U_mu, psi_fwd)
            transported_back = torch.einsum('...ij,...sj->...si', dagger(U_mu_back), psi_back)

            term_fwd = torch.einsum('st,...ti->...si', self.r_minus[mu], transported_fwd)
            term_back = torch.einsum('st,...ti->...si', self.r_plus[mu], transported_back)

            out = out - 0.5 * (term_fwd + term_back)
        return out

    def apply_dagger(self, psi: "torch.Tensor", U: "torch.Tensor") -> "torch.Tensor":
        """D^dagger psi = gamma5 D (gamma5 psi), the standard identity."""
        g5psi = torch.einsum('st,...ti->...si', self.g5, psi)
        Dg5psi = self.apply(g5psi, U)
        return torch.einsum('st,...ti->...si', self.g5, Dg5psi)

    def normal_op(self, psi: "torch.Tensor", U: "torch.Tensor") -> "torch.Tensor":
        """Q = D^dagger D, Hermitian positive-definite -- what CG solves."""
        return self.apply_dagger(self.apply(psi, U), U)


# ---------------------------------------------------------------------------
# Complex conjugate-gradient solver
# ---------------------------------------------------------------------------

def cg_solve(matvec: Callable[["torch.Tensor"], "torch.Tensor"],
             b: "torch.Tensor",
             x0: Optional["torch.Tensor"] = None,
             tol: float = 1e-8,
             maxiter: int = 200) -> Tuple["torch.Tensor", int, float]:
    """
    Standard conjugate-gradient solve of Q x = b for a Hermitian
    positive-definite complex linear operator `matvec`, using the
    Hermitian inner product <a, b> = sum(conj(a) * b). Returns
    (x, iterations_used, final_residual_norm).
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.sum(r.conj() * r).real

    b_norm = torch.sqrt(torch.sum(b.conj() * b).real).clamp_min(1e-30)

    for it in range(maxiter):
        Ap = matvec(p)
        alpha = rs_old / torch.sum(p.conj() * Ap).real.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r.conj() * r).real
        resid = torch.sqrt(rs_new) / b_norm
        if resid < tol:
            return x, it + 1, float(resid)
        p = r + (rs_new / rs_old.clamp_min(1e-30)) * p
        rs_old = rs_new
    return x, maxiter, float(torch.sqrt(rs_old) / b_norm)


# ---------------------------------------------------------------------------
# Pseudofermion field (dynamical-fermion HMC via heatbath)
# ---------------------------------------------------------------------------

class PseudofermionField:
    """
    Represents the fermion determinant det(D^dagger D) as a bosonic
    (pseudofermion) field phi, via the standard Gaussian-integral
    identity:

        det(Q) proportional-to  integral d(phi*) d(phi) exp(-phi^dagger Q^-1 phi),   Q = D^dagger D

    HEATBATH REFRESH (the piece you asked for): to sample phi from
    exp(-phi^dagger Q^-1 phi) exactly, draw eta from a plain complex
    Gaussian (mean 0, unit variance per component) and set

        phi = D^dagger eta

    Proof sketch: with D = U Sigma V^dagger (SVD), Q^-1 = V Sigma^-2
    V^dagger, and D (Q^-1) D^dagger = U Sigma V^dagger V Sigma^-2
    V^dagger V Sigma U^dagger = U U^dagger = I, so
    phi^dagger Q^-1 phi = eta^dagger D (Q^-1) D^dagger eta = eta^dagger eta,
    i.e. phi's distribution is exactly the target one whenever eta is
    a plain unit Gaussian. This is the textbook pseudofermion heatbath
    used in every dynamical-fermion lattice QCD code -- no MCMC
    sub-loop needed, it's an exact one-shot sample.

    ACTION AND FORCE: given the current phi, the pseudofermion action
    is S_pf(U) = phi^dagger Q(U)^-1 phi. Writing x = Q(U)^-1 phi
    (obtained via CG, reusing the solver already built for the
    diagnostic D x = b solve), the identity phi^dagger x = x^dagger Q x
    = ||D(U) x||^2 lets us evaluate the action cheaply once x is known.

    For the force (needed by the HMC leapfrog integrator), we use the
    same trick production codes use to avoid differentiating through
    the CG iterations at all: because x = Q(U)^-1 phi sits exactly at
    the minimum of the quadratic form phi^dagger x - x^dagger Q(U) x
    for fixed phi, the total derivative of S_pf with respect to U
    equals the *partial* derivative of ||D(U) x||^2 with respect to U
    *holding x fixed* (all the terms coming from dx/dU cancel exactly
    at the stationary point -- this is the same identity behind the
    adjoint-state method in PDE-constrained optimization). So: solve
    for x once via CG, detach it, then let autograd differentiate the
    simple expression ||D(U) x||^2 with respect to U. No CG solve
    needs to appear inside the autograd graph.
    """

    def __init__(self, lattice: "LatticeV2", dirac: "WilsonDiracOperator",
                 config: "ConfigV2", generator: "torch.Generator"):
        self.lattice = lattice
        self.dirac = dirac
        self.config = config
        self.generator = generator
        self.field_shape = lattice.shape + (config.spinor_dim, config.color_dim)
        self.phi: Optional["torch.Tensor"] = None

    def refresh(self, U: "torch.Tensor") -> None:
        """Heatbath refresh: draw a fresh phi from the exact target distribution."""
        real = torch.randn(self.field_shape, generator=self.generator, dtype=torch.float64)
        imag = torch.randn(self.field_shape, generator=self.generator, dtype=torch.float64)
        eta = (real + 1j * imag).to(self.config.dtype)
        self.phi = self.dirac.apply_dagger(eta, U)

    def solve(self, U: "torch.Tensor", tol: float,
              x0: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", int, float]:
        """x = Q(U)^-1 phi via CG on the Hermitian positive-definite normal operator."""
        def matvec(v):
            return self.dirac.normal_op(v, U)
        return cg_solve(matvec, self.phi, x0=x0, tol=tol, maxiter=self.config.cg_maxiter)

    def action(self, x: "torch.Tensor", U: "torch.Tensor") -> "torch.Tensor":
        """S_pf = ||D(U) x||^2, given x = Q(U)^-1 phi already solved."""
        Dx = self.dirac.apply(x, U)
        return torch.sum((Dx.conj() * Dx).real)

    def force(self, x: "torch.Tensor", U: "torch.Tensor") -> "torch.Tensor":
        """
        su(N)-projected force from the pseudofermion action, with x
        treated as fixed (see class docstring for why that's exact).
        Mirrors GaugeField.force()'s construction so the two forces
        combine consistently in the HMC leapfrog step.
        """
        U_req = U.detach().clone().requires_grad_(True)
        S = self.action(x.detach(), U_req)
        (grad,) = torch.autograd.grad(S, U_req)
        raw = U_req.detach() @ dagger(grad.detach())
        return project_traceless_antihermitian(raw)


# ---------------------------------------------------------------------------
# Hybrid Monte Carlo (quenched / pure-gauge, or dynamical with pseudofermions)
# ---------------------------------------------------------------------------

class HMC:
    """
    Standard HMC. Two modes, selected by whether a PseudofermionField
    is passed in:

    QUENCHED (pseudofermion=None): pure-gauge Wilson action only.
        1. refresh momenta P ~ Gaussian, traceless Hermitian
        2. compute H = (1/2) sum Tr(P^2) + S_gauge(U)
        3. leapfrog-integrate (U, P) for n_leapfrog steps
        4. Metropolis accept/reject on Delta H

    DYNAMICAL (pseudofermion given): full lattice-QCD-style HMC with a
    fermion determinant included via a pseudofermion field.
        0. HEATBATH: refresh phi = D(U)^dagger eta, eta ~ Gaussian
           (exact one-shot sample of the pseudofermion action, see
           PseudofermionField.refresh)
        1. refresh momenta P ~ Gaussian, traceless Hermitian
        2. compute H = (1/2) sum Tr(P^2) + S_gauge(U) + phi^dagger Q(U)^-1 phi
           (the last term evaluated via a tight-tolerance CG solve)
        3. leapfrog-integrate (U, P), with the force at every step now
           F_gauge(U) + F_pseudofermion(U) -- the fermion force needs
           one CG solve (at the MD tolerance, looser than the action
           tolerance -- see ConfigV2.cg_tol_md/cg_tol_action) per
           leapfrog step, which is exactly why dynamical-fermion HMC
           is dominated by CG solves in real lattice QCD codes
        4. Metropolis accept/reject on Delta H (using a fresh,
           tight-tolerance CG solve for the final pseudofermion action)

    Either way, this samples the correct equilibrium distribution
    (exp(-S_gauge(U)) or exp(-S_gauge(U) - S_pf(U)) respectively),
    exactly, for any step size -- unlike plain noisy-Euler/Langevin
    integration, which only samples the right distribution in the
    eps -> 0 limit.

    KNOWN SIMPLIFICATION: this uses the plain fermion determinant
    det(Q), which corresponds to 4 fermion flavors (since Q = D^dagger
    D naturally represents 2 degenerate flavors already via the
    pseudofermion trick, and this isn't further rooted for a single
    flavor). Realistic 2-flavor or 1-flavor simulations use rational
    approximations (RHMC) to represent det(Q)^{1/2} or det(Q)^{1/4} --
    a real additional piece of machinery (multi-shift CG + a rational
    approximation to x^{-1/2}) that is not implemented here.
    """

    def __init__(self, gauge: GaugeFieldV2, config: ConfigV2, generator: "torch.Generator",
                 pseudofermion: Optional[PseudofermionField] = None):
        self.gauge = gauge
        self.config = config
        self.generator = generator
        self.pseudofermion = pseudofermion
        self.n_accepted = 0
        self.n_total = 0
        self.last_cg_stats: Dict[str, Any] = {}

    def _hamiltonian(self, U: "torch.Tensor", P: "torch.Tensor",
                      pf_x0: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        cfg = self.config
        kinetic = 0.5 * torch.sum((P @ P).diagonal(dim1=-2, dim2=-1).sum(-1).real)
        potential = self.gauge.wilson_action(U)
        pf_x = None
        if self.pseudofermion is not None:
            pf_x, iters, resid = self.pseudofermion.solve(U, tol=cfg.cg_tol_action, x0=pf_x0)
            self.last_cg_stats = {"action_cg_iters": iters, "action_cg_resid": resid}
            potential = potential + self.pseudofermion.action(pf_x, U)
        return kinetic + potential, pf_x

    def _force_at(self, U: "torch.Tensor", pf_x0: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        cfg = self.config
        U_req = U.detach().clone().requires_grad_(True)
        S = self.gauge.wilson_action(U_req)
        (grad,) = torch.autograd.grad(S, U_req)
        raw = U_req.detach() @ dagger(grad.detach())
        F = project_traceless_antihermitian(raw)

        pf_x = None
        if self.pseudofermion is not None:
            pf_x, iters, resid = self.pseudofermion.solve(U, tol=cfg.cg_tol_md, x0=pf_x0)
            self.last_cg_stats = {"md_cg_iters": iters, "md_cg_resid": resid}
            F = F + self.pseudofermion.force(pf_x, U)
        return F, pf_x

    def trajectory(self) -> Dict[str, Any]:
        cfg = self.config
        U0 = self.gauge.U.clone()
        shape = U0.shape
        P0 = random_su_n_hermitian(shape, self.gauge.n, cfg.dtype, U0.device, self.generator)

        # heatbath refresh of the pseudofermion field, if dynamical fermions are on
        if self.pseudofermion is not None:
            self.pseudofermion.refresh(U0)

        H0, pf_x = self._hamiltonian(U0, P0)

        U, P = U0.clone(), P0.clone()
        eps = cfg.hmc_step_size

        # leapfrog integration, warm-starting each CG solve from the previous one
        F, pf_x = self._force_at(U, pf_x0=pf_x)
        P = P + 0.5 * eps * (1j * F)  # F is anti-Hermitian; i*F is Hermitian, matches P's algebra
        for step in range(cfg.hmc_n_leapfrog):
            U = expm_anti_hermitian(eps * (-1j * P)) @ U
            F, pf_x = self._force_at(U, pf_x0=pf_x)
            coeff = eps if step < cfg.hmc_n_leapfrog - 1 else 0.5 * eps
            P = P + coeff * (1j * F)

        H1, pf_x_final = self._hamiltonian(U, P, pf_x0=pf_x)
        delta_h = float((H1 - H0).real)

        accept_prob = min(1.0, math.exp(-delta_h)) if delta_h < 700 else 0.0
        u = torch.rand((), generator=self.generator, dtype=torch.float64).item()
        accepted = u < accept_prob

        self.n_total += 1
        if accepted:
            self.n_accepted += 1
            self.gauge.U = U

        result = {
            "delta_h": delta_h,
            "accept_prob": accept_prob,
            "accepted": accepted,
            "acceptance_rate": self.n_accepted / self.n_total,
        }
        # expose the final pseudofermion solution (physically meaningful psi
        # sample) so callers can e.g. feed it to LearnedInformationGeometry
        result["pseudofermion_x"] = pf_x_final.detach() if pf_x_final is not None else None
        result.update(self.last_cg_stats)
        return result



# ---------------------------------------------------------------------------
# Unimplemented roadmap seams -- documented, not faked
# ---------------------------------------------------------------------------

class NeuralEffectiveAction:
    """
    TODO (not implemented): a graph neural network mapping gauge-link
    configurations to a learned scalar effective action S(U), trained
    so that its induced HMC force approximates (or improves on) the
    analytic Wilson action -- e.g. a message-passing GNN over the
    lattice graph with links as edge features. Needs a training
    dataset of configurations, a loss (e.g. matching force fields or
    matching observable expectation values), and a training loop.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "NeuralEffectiveAction is a documented roadmap seam, not an implementation. "
            "See the class docstring for what it would need."
        )


class FourierNeuralOperator:
    """
    TODO (not implemented): learn psi(t) -> psi(t + dt) directly in
    Fourier space (spectral convolution layers) as a fast surrogate
    for explicit lattice evolution. Needs training trajectories
    generated by the explicit solver above as supervision.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FourierNeuralOperator is a documented roadmap seam, not an implementation."
        )


class _MLP(nn.Module):
    """Small fully-connected network; shared by the encoder and decoder."""

    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class LearnedInformationGeometry:
    """
    A real Riemannian-geometry-on-a-learned-latent-space pipeline:

        psi (complex lattice field)
            -> flatten to a real vector
            -> encoder (MLP)      -> latent z
            -> decoder (MLP)      -> reconstruction of psi
        Gaussian observation model p(psi | z) = N(decoder(z), sigma^2 I)
            -> Fisher information metric  G(z) = (1/sigma^2) J(z)^T J(z)
               where J(z) = d(decoder)/dz, computed exactly via
               torch.func.jacrev (this is the standard pullback metric
               used in Riemannian-VAE / "latent space oddity"-style work)
            -> Christoffel symbols Gamma^k_{ij}(z) from G via jacrev
            -> Riemann tensor, Ricci tensor, and scalar curvature R(z)
               from the Christoffel symbols via central finite
               differences (a deliberate, documented choice: composing
               three nested jacrev calls through a matrix inverse is
               fragile and slow; a single layer of finite differences
               on top of an exact, autodiff-derived Jacobian is the
               standard practical compromise and is still "real"
               numerical differential geometry, not a proxy statistic).

    Training data comes for free from the physics already being
    computed: MetaFieldSimulationV2 feeds it the CG solutions x (from
    D x = b, solved against each HMC gauge background) as psi samples
    to autoencode -- no synthetic or external dataset required.

    Keep `latent_dim` small (3-4). The Riemann tensor has
    O(latent_dim^4) components and every curvature evaluation costs
    O(latent_dim^2) additional Jacobian evaluations, so this is meant
    for a compact learned summary of the field, not a high-dimensional
    embedding.
    """

    def __init__(self, lattice: "LatticeV2", config: "ConfigV2",
                 latent_dim: int = 3, hidden_dims: Tuple[int, ...] = (256, 64),
                 sigma: float = 1.0, lr: float = 1e-3):
        self.lattice = lattice
        self.config = config
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.field_shape = lattice.shape + (config.spinor_dim, config.color_dim)
        self.input_dim = 2 * int(torch.tensor(self.field_shape).prod().item())

        enc_dims = [self.input_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), self.input_dim]
        self.encoder = _MLP(enc_dims).to(dtype=torch.float64)
        self.decoder = _MLP(dec_dims).to(dtype=torch.float64)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.last_loss: Optional[float] = None

    # -- flatten / unflatten between complex lattice fields and real vectors --

    def _flatten(self, psi: "torch.Tensor") -> "torch.Tensor":
        flat = psi.reshape(-1)
        return torch.cat([flat.real, flat.imag]).to(torch.float64)

    def _unflatten(self, vec: "torch.Tensor") -> "torch.Tensor":
        n = vec.shape[-1] // 2
        real, imag = vec[..., :n], vec[..., n:]
        flat_complex = (real + 1j * imag).to(self.config.dtype)
        return flat_complex.reshape(self.field_shape)

    # -- encoder / decoder ------------------------------------------------

    def encode(self, psi: "torch.Tensor") -> "torch.Tensor":
        x = self._flatten(psi)
        return self.encoder(x)

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        """Returns the flat real reconstruction vector (not unflattened)."""
        return self.decoder(z)

    def train_on_batch(self, psi_samples: List["torch.Tensor"]) -> float:
        """One Adam step of reconstruction (autoencoding) on a batch of
        psi field configurations. Returns the MSE loss."""
        x = torch.stack([self._flatten(p) for p in psi_samples])
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.mean((x_hat - x) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.item())
        return self.last_loss

    # -- Riemannian geometry on the latent manifold ------------------------

    def fisher_metric(self, z: "torch.Tensor") -> "torch.Tensor":
        """
        G(z) = (1/sigma^2) J(z)^T J(z), the Fisher information metric
        of the Gaussian decoder model p(psi | z) = N(decoder(z), sigma^2 I),
        i.e. the pullback of the ambient Euclidean metric through the
        decoder. Uses torch.func.jacrev for an exact Jacobian.
        """
        J = torch.func.jacrev(self.decoder)(z)          # (output_dim, latent_dim)
        G = (J.transpose(-1, -2) @ J) / (self.sigma ** 2)
        return G

    def christoffel_symbols(self, z: "torch.Tensor", eps: float = 1e-4) -> "torch.Tensor":
        """
        Gamma^k_{ij} = 1/2 g^{kl} (d_i g_{jl} + d_j g_{il} - d_l g_{ij}),
        with d_i g_{jl}(z) obtained via torch.func.jacrev on the exact
        Fisher metric above. Returns a (dim, dim, dim) tensor indexed
        [k, i, j].
        """
        dim = self.latent_dim
        G = self.fisher_metric(z)
        eye = torch.eye(dim, dtype=G.dtype)
        G_inv = torch.linalg.inv(G + eps * eye)

        # dG[i, j, k] = d(g_{ij}) / d(z_k)
        dG = torch.func.jacrev(self.fisher_metric)(z)

        # Gamma[k, i, j] = 0.5 * sum_l G_inv[k,l] * (dG[j,l,i] + dG[i,l,j] - dG[i,j,l])
        term = (torch.einsum('jli->ijl', dG) + torch.einsum('ilj->ijl', dG) - dG)
        Gamma = 0.5 * torch.einsum('kl,ijl->kij', G_inv, term)
        return Gamma

    def curvature(self, z: "torch.Tensor", fd_eps: float = 1e-3) -> Dict[str, "torch.Tensor"]:
        """
        Full Riemann tensor, Ricci tensor, and scalar curvature at z.
        First derivatives (Christoffel symbols) are exact via autodiff;
        the one additional derivative needed for the Riemann tensor is
        taken by central finite differences on christoffel_symbols(z)
        (documented tradeoff -- see class docstring).
        """
        dim = self.latent_dim
        Gamma0 = self.christoffel_symbols(z)

        # dGamma[k, i, j, m] = d(Gamma^k_{ij}) / d(z_m), via central differences
        dGamma = torch.zeros(dim, dim, dim, dim, dtype=Gamma0.dtype)
        for m in range(dim):
            step = torch.zeros(dim, dtype=z.dtype)
            step[m] = fd_eps
            G_plus = self.christoffel_symbols(z + step)
            G_minus = self.christoffel_symbols(z - step)
            dGamma[:, :, :, m] = (G_plus - G_minus) / (2 * fd_eps)

        # Riemann^l_{ijk} = d_i Gamma^l_{jk} - d_j Gamma^l_{ik}
        #                 + Gamma^l_{im} Gamma^m_{jk} - Gamma^l_{jm} Gamma^m_{ik}
        Riemann = torch.zeros(dim, dim, dim, dim, dtype=Gamma0.dtype)  # [l,i,j,k]
        for l in range(dim):
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        term1 = dGamma[l, j, k, i]
                        term2 = dGamma[l, i, k, j]
                        term3 = sum(Gamma0[l, i, m] * Gamma0[m, j, k] for m in range(dim))
                        term4 = sum(Gamma0[l, j, m] * Gamma0[m, i, k] for m in range(dim))
                        Riemann[l, i, j, k] = term1 - term2 + term3 - term4

        # Ricci_{jk} = sum_i Riemann^i_{ijk}
        Ricci = torch.zeros(dim, dim, dtype=Gamma0.dtype)
        for j in range(dim):
            for k in range(dim):
                Ricci[j, k] = sum(Riemann[i, i, j, k] for i in range(dim))

        G = self.fisher_metric(z)
        eye = torch.eye(dim, dtype=G.dtype)
        G_inv = torch.linalg.inv(G + 1e-4 * eye)
        scalar_curvature = torch.einsum('jk,jk->', G_inv, Ricci)

        return {
            "metric": G,
            "christoffel": Gamma0,
            "riemann": Riemann,
            "ricci": Ricci,
            "scalar_curvature": scalar_curvature,
        }


class DistributedLattice:
    """
    Implemented -- but as a separate, standalone script rather than a
    class you instantiate from here.

    Domain-decomposed simulation only means something once you have
    more than one process/machine actually running it; that can't be
    exercised from a single-process import in this file. See
    `meta_field_distributed.py` (shipped alongside this file), which
    implements 1D domain decomposition along lattice axis 0 across N
    ranks using torch.distributed with the Gloo (CPU, TCP) backend --
    the right choice for your two networked machines, since Gloo
    doesn't need CUDA/NCCL. It exchanges only the boundary ("halo")
    slices with each neighbor rank every step, exactly as production
    lattice codes do.

    See the header of that file for exact launch commands for your two
    nodes.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DistributedLattice's real implementation lives in the standalone "
            "script meta_field_distributed.py -- see this class's docstring."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MetaFieldSimulationV2:
    """
    Drives HMC on the gauge field -- quenched (gauge-only) by default,
    or fully dynamical (gauge + pseudofermion-represented fermion
    determinant) when config.include_fermions is True.

    In dynamical mode, each trajectory's own pseudofermion CG solution
    (already computed as part of the physics, not a throwaway
    diagnostic) is what feeds LearnedInformationGeometry when enabled.
    In quenched mode, we still solve D x = b for a random source each
    trajectory purely as a diagnostic / smoke test of the Wilson-Dirac
    operator and CG solver, and use *that* as the geometry's training
    signal instead.
    """

    def __init__(self, config: Optional[ConfigV2] = None,
                 use_learned_geometry: bool = False,
                 geometry_latent_dim: int = 3,
                 geometry_batch_size: int = 4,
                 geometry_report_every: int = 5):
        self.config = config or ConfigV2()
        torch.manual_seed(self.config.seed)
        self.generator = torch.Generator().manual_seed(self.config.seed)

        self.lattice = LatticeV2(self.config)
        self.gauge = GaugeFieldV2(self.lattice, self.config, self.generator)
        self.dirac = WilsonDiracOperator(self.lattice, self.config)

        self.pseudofermion: Optional[PseudofermionField] = None
        if self.config.include_fermions:
            self.pseudofermion = PseudofermionField(self.lattice, self.dirac, self.config, self.generator)

        self.hmc = HMC(self.gauge, self.config, self.generator, pseudofermion=self.pseudofermion)

        self.history: List[Dict[str, Any]] = []

        # Optional learned information geometry, trained on physically
        # meaningful psi samples produced during run() -- no synthetic
        # data needed.
        self.use_learned_geometry = use_learned_geometry
        self.geometry_batch_size = geometry_batch_size
        self.geometry_report_every = geometry_report_every
        self.geometry: Optional[LearnedInformationGeometry] = None
        self._psi_buffer: List["torch.Tensor"] = []
        if use_learned_geometry:
            self.geometry = LearnedInformationGeometry(
                self.lattice, self.config, latent_dim=geometry_latent_dim
            )

    def run(self) -> List[Dict[str, Any]]:
        cfg = self.config
        mode = "dynamical (pseudofermion heatbath)" if cfg.include_fermions else "quenched (gauge-only)"
        print(f"\n=== MetaField v2 (PyTorch): {mode} HMC + Wilson-Dirac + CG ===")
        print(f"Lattice: {self.lattice.shape} | beta={cfg.beta} | "
              f"leapfrog={cfg.hmc_n_leapfrog} step={cfg.hmc_step_size}\n")

        for traj in range(cfg.hmc_trajectories):
            hmc_result = self.hmc.trajectory()

            if cfg.include_fermions and hmc_result["pseudofermion_x"] is not None:
                # use the trajectory's own physical CG solution
                x = hmc_result["pseudofermion_x"]
                iters = hmc_result.get("action_cg_iters", hmc_result.get("md_cg_iters", -1))
                resid = hmc_result.get("action_cg_resid", hmc_result.get("md_cg_resid", float("nan")))
            else:
                # quenched mode: solve D x = b for a random source as a
                # diagnostic / smoke test of the operator and solver
                shape = self.lattice.shape + (cfg.spinor_dim, cfg.color_dim)
                b = (torch.randn(shape, generator=self.generator, dtype=torch.float64)
                     + 1j * torch.randn(shape, generator=self.generator, dtype=torch.float64)).to(cfg.dtype)
                rhs = self.dirac.apply_dagger(b, self.gauge.U)

                def matvec(v):
                    return self.dirac.normal_op(v, self.gauge.U)

                x, iters, resid = cg_solve(matvec, rhs, tol=cfg.cg_tol, maxiter=cfg.cg_maxiter)

            action_val = float(self.gauge.wilson_action().real)
            record = {
                "trajectory": traj,
                "wilson_action": action_val,
                "delta_h": hmc_result["delta_h"],
                "accepted": hmc_result["accepted"],
                "acceptance_rate": hmc_result["acceptance_rate"],
                "cg_iters": iters,
                "cg_residual": resid,
            }

            geom_line = ""
            if self.use_learned_geometry:
                self._psi_buffer.append(x.detach())
                if len(self._psi_buffer) >= self.geometry_batch_size:
                    loss = self.geometry.train_on_batch(self._psi_buffer)
                    self._psi_buffer = []
                    record["geometry_loss"] = loss
                    geom_line = f" | AE loss={loss:.4e}"

                    if traj % self.geometry_report_every == 0:
                        with torch.no_grad():
                            z = self.geometry.encode(x.detach())
                        curv = self.geometry.curvature(z)
                        R = float(curv["scalar_curvature"].real)
                        record["scalar_curvature"] = R
                        geom_line += f" | latent R={R:.4e}"

            self.history.append(record)
            print(
                f"traj {traj:3d} | S={action_val:.4f} | dH={hmc_result['delta_h']:+.4f} | "
                f"{'ACC' if hmc_result['accepted'] else 'rej'} "
                f"(rate={hmc_result['acceptance_rate']:.2f}) | "
                f"CG: {iters} iters, resid={resid:.2e}{geom_line}"
            )

        return self.history


if __name__ == "__main__":
    config = ConfigV2(
        L=4,                 # keep tiny for a first CPU smoke test; Wilson-Dirac + CG
        beta=5.5,             # cost grows fast with L^4
        hmc_n_leapfrog=10,
        hmc_step_size=0.05,
        hmc_trajectories=10,  # dynamical fermions do a CG solve per leapfrog step,
        include_fermions=True,  # so this is noticeably slower than the quenched default;
        seed=42,               # drop to include_fermions=False for a faster gauge-only run.
    )
    sim = MetaFieldSimulationV2(
        config,
        use_learned_geometry=True,
        geometry_latent_dim=3,
        geometry_batch_size=4,
        geometry_report_every=5,
    )
    sim.run()
