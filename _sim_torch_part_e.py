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
    doesn't need CUDA/NCCL. It exchanges only the boundary (\"halo\")
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
                x = hmc_result["pseudofermion_x"]
                iters = hmc_result.get("action_cg_iters", hmc_result.get("md_cg_iters", -1))
                resid = hmc_result.get("action_cg_resid", hmc_result.get("md_cg_resid", float("nan")))
            else:
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
        L=4,
        beta=5.5,
        hmc_n_leapfrog=10,
        hmc_step_size=0.05,
        hmc_trajectories=10,
        include_fermions=True,
        seed=42,
    )
    sim = MetaFieldSimulationV2(
        config,
        use_learned_geometry=True,
        geometry_latent_dim=3,
        geometry_batch_size=4,
        geometry_report_every=5,
    )
    sim.run()
