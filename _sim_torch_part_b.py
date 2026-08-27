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
