# MetaField

**A lattice gauge theory simulator with memory, prediction, and a growing path toward distributed swarm intelligence.**

MetaField combines a stable Hybrid Monte Carlo engine for SU(3) lattice gauge theory (quenched and dynamical fermions) with a learned geometric representation and an episodic memory + prediction layer. It is designed as the foundation for an intelligence that grows inside a mathematically consistent simulated universe.

---

## Current State (v1.28)

- Stable HMC (single or multi-node) with high acceptance
- Wilson gauge action + Wilson-Dirac operator with pseudofermions
- Learned Information Geometry with Fisher metric and curvature estimation
- Episodic memory with prioritized replay
- Latent predictor that forms expectations about future behavior
- Dynamic geometry training (epochs now scale with the amount of data collected)
- Clean continuous mode (`--continuous`) with efficient predictor updates
- More forgiving argument parsing

---

## Long-term Vision: Aurora + MetaField Super Hybrid

We are actively working toward a deep integration with [Aurora Swarm BTC](https://github.com/TheBabelDragon/aurora-swarm-btc), turning Aurora’s distributed swarm infrastructure into the coordination layer for MetaField.

**Goal**: A distributed swarm that runs physics simulations as its native environment, accumulates structured memory across machines, and gradually develops curiosity, goals, and reasoning.

See:
- `HYBRID_VISION.md` — High-level architecture and philosophy
- `INTEGRATION_PLAN.md` — Concrete phased roadmap

---

## Quick Start (Single Machine)

```bash
git clone https://github.com/TheBabelDragon/metafield.git
cd metafield

python -m venv ../.venv
source ../.venv/bin/activate
pip install torch matplotlib scikit-learn

# Recommended: long-running continuous mode
python meta_field_distributed.py --world-size 1 --include-fermions --diagnostic --continuous
```

**Note**: `--include-fermions` now accepts many formats (`true`, `false`, `1`, `0`, `yes`, `no`, or just the flag itself).

---

## Key Improvements in Recent Versions

- **Dynamic geometry training** — Autoencoder epochs now scale with the number of samples collected (better for long runs).
- **Optimized predictor training** — In continuous mode, the predictor is updated less frequently for better performance.
- **More forgiving CLI** — `--include-fermions` is now much easier to use.
- **Cleaner continuous mode logging** — Less noise, more useful periodic summaries.

---

## Key Components

| Component                    | Description                                      |
|-----------------------------|--------------------------------------------------|
| `DistributedHMC`            | Core Hybrid Monte Carlo engine                   |
| `LearnedInformationGeometry`| Autoencoder + Riemannian geometry on fields      |
| `EpisodicMemory`            | Stores contextual experiences with prioritization|
| `LatentPredictor`           | Learns expectations from latent state            |

---

## Continuous Mode (Recommended)

```bash
python meta_field_distributed.py --world-size 1 --include-fermions --diagnostic --continuous
```

Press `Ctrl+C` to stop cleanly. The system now handles long runs more efficiently.

---

## Multi-Machine / Distributed

Still experimental due to Gloo + system configuration issues on some machines (common `127.0.1.1` in `/etc/hosts` problem). See the error messages in the code for the recommended fix.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- matplotlib + scikit-learn (for visualizations)

## License

MIT License

---

*Active development toward a distributed physics-based intelligence swarm.*
