# MetaField

**A lattice gauge theory simulator with memory, prediction, and a growing path toward distributed swarm intelligence.**

MetaField combines a stable Hybrid Monte Carlo engine for SU(3) lattice gauge theory with a learned geometric representation of field configurations and an episodic memory + prediction layer. It is designed as the foundation for an intelligence that grows inside a mathematically consistent simulated universe.

---

## Current Capabilities

- Stable Hybrid Monte Carlo (HMC) with high acceptance rates
- Wilson gauge action + Wilson-Dirac operator
- Learned Information Geometry (autoencoder + Fisher metric + curvature estimation)
- Episodic memory with prioritized replay
- Latent predictor that forms expectations about future behavior
- Dynamic geometry training (number of epochs scales with the amount of data collected)
- Efficient continuous mode (`--continuous`) with reduced predictor update frequency

---

## Long-term Vision: The Aurora + MetaField Super Hybrid

We are building toward a deep integration with [Aurora Swarm BTC](https://github.com/TheBabelDragon/aurora-swarm-btc) — turning Aurora’s distributed swarm infrastructure into the coordination and community compute layer for MetaField.

### What the Hybrid Enables

Instead of treating distributed compute as a dumb resource pool, the hybrid treats the **swarm itself as an environment** in which intelligence can grow:

- Aurora handles node discovery, scheduling, communication, resilience, and community compute across many machines.
- MetaField provides the physics simulation (lattice QCD), learned geometry, episodic memory, and prediction systems.
- Over time, MetaField’s internal signals (curvature, prediction difficulty, interesting configurations) can influence how Aurora allocates compute.
- The result is a distributed system that runs physics as its native substrate and gradually develops memory, expectations, curiosity, and eventually agency.

This is not "MetaField running on Aurora" or "Aurora with some physics mods." It is a true co-evolution where the swarm infrastructure and the physics-based intelligence layer strengthen each other.

See `HYBRID_VISION.md` and `INTEGRATION_PLAN.md` for the detailed architecture and phased roadmap.

---

## Quick Start (Single Machine)

```bash
git clone https://github.com/TheBabelDragon/metafield.git
cd metafield

python -m venv ../.venv
source ../.venv/bin/activate
pip install torch matplotlib scikit-learn

# Recommended for development
python meta_field_distributed.py --world-size 1 --diagnostic --continuous
```

Press `Ctrl+C` to stop cleanly. Long-running sessions are well supported.

---

## Key Improvements (Recent)

- Geometry training epochs now scale dynamically with the number of samples collected
- Predictor training frequency reduced in continuous mode for better performance during long runs
- Cleaner, less noisy logging in continuous mode

---

## Key Components

| Component                    | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `DistributedHMC`            | Core Hybrid Monte Carlo engine                        |
| `LearnedInformationGeometry`| Autoencoder + Riemannian geometry on field configurations |
| `EpisodicMemory`            | Stores contextual experiences with prioritization     |
| `LatentPredictor`           | Learns to predict future observables from latent state|

---

## Continuous Mode

```bash
python meta_field_distributed.py --world-size 1 --diagnostic --continuous
```

This is currently the recommended way to run MetaField. The system is designed for extended sessions where memory and prediction can meaningfully develop.

---

## Multi-Machine

Multi-machine support exists but remains fragile due to Gloo + common Linux hostname resolution issues (`127.0.1.1` in `/etc/hosts`). The code prints clear guidance when this problem is detected.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- matplotlib + scikit-learn (for visualizations and geometry)

## License

MIT License

---

*Actively evolving toward a distributed physics-based intelligence swarm.*
