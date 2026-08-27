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
