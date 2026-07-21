# MetaField

**A lattice gauge theory simulator with memory, prediction, and a growing path toward distributed swarm intelligence.**

MetaField combines a stable Hybrid Monte Carlo engine for SU(3) lattice gauge theory with a learned geometric representation of field configurations and an episodic memory + prediction layer. It is designed as the foundation for an intelligence that grows inside a mathematically consistent simulated universe.

---

## Current Capabilities

- Stable Hybrid Monte Carlo (HMC) with high acceptance rates
- Wilson gauge action + Wilson-Dirac operator
- Learned Information Geometry (autoencoder + Fisher metric + curvature estimation)
- Episodic memory with prioritized replay + `get_stats()` for observability
- Latent predictor that forms expectations about future behavior
- Dynamic geometry training (epochs scale with data volume)
- Efficient continuous mode (`--continuous`) with configurable system summaries

---

## Aurora + MetaField Super Hybrid (In Progress)

We have begun building toward a deep integration with [Aurora Swarm BTC](https://github.com/TheBabelDragon/aurora-swarm-btc).

Aurora provides the distributed swarm infrastructure (node coordination, scheduling, community compute, and a clean mod/hook system). MetaField provides the physics simulation + growing intelligence layer (memory, prediction, geometry).

### Current Integration Work

- Created `aurora_mods/metafield_sensing/` — a proposed first mod that exposes MetaField memory, prediction, and geometry signals into Aurora’s sensing layer.
- MetaField core components (`memory.py`, `prediction.py`, `geometry.py`) have been extracted and cleaned up for easier integration.
- Strong emphasis on observability (`get_stats()`, periodic summaries) to support future Aurora sensing.

See `aurora_mods/metafield_sensing/` for the current proposal and `INTEGRATION_PLAN.md` + `HYBRID_VISION.md` for the broader roadmap.

---

## Quick Start

```bash
git clone https://github.com/TheBabelDragon/metafield.git
cd metafield

python -m venv ../.venv
source ../.venv/bin/activate
pip install torch matplotlib scikit-learn

# Recommended
python meta_field_distributed.py --world-size 1 --diagnostic --continuous --summary-interval 30
```

Press `Ctrl+C` to stop cleanly.

---

## Key Recent Improvements

- System summary interval is now configurable via `--summary-interval`
- Periodic high-level health + memory + prediction summaries in continuous mode
- Major modularization (memory, prediction, and geometry extracted to their own files)
- Aurora integration work started (`metafield_sensing` mod proposal)

---

## Key Components

| Component                    | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `DistributedHMC`            | Core Hybrid Monte Carlo engine                        |
| `EpisodicMemory`            | Prioritized episodic memory with `get_stats()`        |
| `LatentPredictor`           | Predicts future values from latent representations    |
| `LearnedInformationGeometry`| Autoencoder + Riemannian geometry on fields           |

---

## Continuous Mode (Recommended)

```bash
python meta_field_distributed.py --world-size 1 --diagnostic --continuous --summary-interval 30
```

Long-running sessions are well supported and now include periodic system summaries.

---

## Multi-Machine

Multi-machine support exists but is still sensitive to system configuration (common `127.0.1.1` hostname issue). The code prints clear guidance when this problem is detected.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- matplotlib + scikit-learn

## License

MIT License

---

*Actively evolving toward a distributed physics-based intelligence swarm.*
