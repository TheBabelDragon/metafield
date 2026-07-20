# MetaField v2

**PyTorch backend for lattice field theory simulation** — built around algorithms genuinely used in lattice QCD research (not toy stand-ins).

## Highlights

- **GaugeFieldV2**: SU(N) links, real Wilson plaquette action, **autograd-computed force** (dS/dU via `torch.autograd`, not hand-derived staples), exact group exponential update.
- **WilsonDiracOperator**: The actual Wilson-Dirac operator with real 4×4 Euclidean gamma matrices and Wilson term to lift fermion doublers — the standard operator used in lattice QCD.
- **cg_solve**: Complex-valued conjugate gradient for D x = b, solved via the normal equations using the γ5-Hermiticity identity.
- **HMC**: Quenched (pure-gauge) *and* dynamical-fermion Hybrid Monte Carlo with Gaussian momentum refresh, leapfrog integration, and Metropolis accept/reject. Samples the correct equilibrium distribution.
- **PseudofermionField**: Exact one-shot heatbath refresh + efficient force computation (stationary-point trick so CG is not inside autograd).
- **LearnedInformationGeometry**: Autoencoder trained on real CG solutions from HMC trajectories → Fisher metric, Christoffel symbols, Riemann/Ricci tensor, and scalar curvature on the latent manifold of physical fields.

## Important Caveat

> This file was written in a sandbox with no PyTorch installed and no network access, so it could not be executed or debugged against a real interpreter before being handed over. The physics and the algorithms are standard and were transcribed carefully, but **treat first execution as a debugging pass**, not a guaranteed clean run.

Shape mismatches, dtype issues, or numerical problems are the most likely failure modes. Please report anything that breaks.

## Installation

```bash
pip install torch
# GPU: https://pytorch.org/get-started/locally/
```

## Run

```bash
python meta_field_sim_torch.py
```

The `__main__` block runs a small dynamical HMC example on a 4⁴ lattice with learned geometry enabled. It will be noticeably slower with `include_fermions=True` because of the CG solves inside the leapfrog integrator.

Tweak `ConfigV2` at the bottom of the script for different lattice sizes, β, trajectory counts, or to run quenched (faster) mode.

## Physics & Design Choices

- Uses **automatic differentiation** for the gauge force — a deliberate "differentiable physics" design choice instead of the traditional analytic staple force.
- Wilson-Dirac operator follows the standard Euclidean formulation (Degrand & DeTar convention).
- Dynamical fermions use the textbook pseudofermion heatbath (exact, no inner MCMC loop).
- The learned geometry component is trained for free on the CG solutions that the simulation already computes.

## Future Work (Documented Stubs)

The following are intentionally left as documented roadmap items rather than fake implementations:

- `NeuralEffectiveAction` (GNN effective action)
- `FourierNeuralOperator`
- `DistributedLattice` (see companion script `meta_field_distributed.py` for torch.distributed + Gloo domain decomposition)
- Multigrid solvers, RHMC, etc.

Each is its own substantial research project.

## References

- T. DeGrand & C. DeTar, *Lattice Methods for Quantum Chromodynamics* (World Scientific, 2006)
- Standard lattice QCD literature on HMC, Wilson fermions, and pseudofermions.

## License

MIT — see the [LICENSE](LICENSE) file for details.

---

*Built as part of the MetaField project exploring lattice field theory, geometry, and machine learning.*