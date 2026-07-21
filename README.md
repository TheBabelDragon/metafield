# MetaField

**A lattice gauge theory simulator with a growing memory and prediction layer.**

MetaField is a distributed Hybrid Monte Carlo (HMC) engine for SU(3) lattice gauge theory (quenched and dynamical fermions) combined with a learned geometric representation of field configurations. It is designed as a foundation for systems that accumulate experience and form expectations inside a mathematically consistent simulated universe.

## Current Capabilities (v1.19)

- Stable distributed HMC (single or multi-node) with high acceptance rates
- Wilson gauge action + Wilson-Dirac operator with pseudofermions
- Learned Information Geometry: autoencoder on field configurations with Fisher metric and curvature estimation
- Episodic memory system that stores contextual experiences (latent states + observables)
- Latent predictor that begins forming expectations about future behavior
- Prioritized replay based on reconstruction error, curvature, and prediction difficulty
- Rich diagnostics and visualizations (latent space, reconstruction error, action history)

## Philosophy

Rather than treating this as a physics simulator with AI components attached, the project explores what it means to build an intelligence that *grows up inside* a simulated physical universe. The simulation is the environment. Memory, prediction, and eventual agency emerge from interacting with that world.

## Quick Start (Single Machine)

```bash
# Clone
git clone https://github.com/TheBabelDragon/metafield.git
cd metafield

# Create environment
python -m venv ../.venv
source ../.venv/bin/activate
pip install torch matplotlib scikit-learn

# Run with full diagnostics and memory/prediction layer
python meta_field_distributed.py --world-size 1 --include-fermions true --diagnostic
```

This will run HMC trajectories while training the geometry model and the latent predictor, and will save diagnostic plots (`latent_space.png`, `reconstruction_error.png`).

## Key Components

| Component                    | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `DistributedHMC`            | Core Hybrid Monte Carlo engine (gauge + fermions)                           |
| `LearnedInformationGeometry`| Autoencoder + Riemannian geometry on field configurations                   |
| `EpisodicMemory`            | Stores contextual experiences with prioritization                           |
| `LatentPredictor`           | Learns to predict future observables from latent state                      |
| `DistributedLattice`        | Domain-decomposed lattice with halo exchange (Gloo backend)                 |

## Running on Multiple Machines

The code supports distributed execution via `torch.distributed` (Gloo over TCP). Example for a 2-node run:

```bash
# On control node (rank 0)
python meta_field_distributed.py --role control --world-size 2

# On worker node
python meta_field_distributed.py --role worker --master-addr <CONTROL-IP> --world-size 2
```

## Output & Diagnostics

When running with `--diagnostic`, the system produces:

- `latent_space.png` — 2D projection of the learned latent manifold
- `reconstruction_error.png` — Autoencoder reconstruction quality
- Console output showing prediction loss and memory buffer statistics

## Long-term Direction

The project is evolving from a world generator toward an agent that:

- Accumulates episodic memory of interesting physical configurations
- Develops expectations about its own dynamics
- Prioritizes surprising or informative experiences
- Eventually builds internal world models and active experimentation

The simulation is not just data — it is the environment in which reasoning can emerge.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- (Optional) matplotlib + scikit-learn for visualizations

## License

MIT License

---

*This is an active research project exploring the intersection of lattice gauge theory, geometric deep learning, and agentic systems in simulated physical environments.*
