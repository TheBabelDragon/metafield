# MetaField

**A lattice gauge theory simulator with memory, prediction, and a growing path toward distributed swarm intelligence — now generalized to multiple physical field substrates.**

MetaField combines a stable Hybrid Monte Carlo engine for SU(3) lattice gauge theory with a learned geometric representation of field configurations and an episodic memory + prediction layer. It is designed as the foundation for an intelligence that grows inside a mathematically consistent simulated universe *and* real physical field bodies.

---

## Quick Start (Arch Linux)

```bash
# system packages
sudo pacman -S --needed python python-pip python-virtualenv git

# clone
git clone https://github.com/TheBabelDragon/metafield.git
cd metafield

# venv (keep it outside the repo if you prefer)
python -m venv .venv
source .venv/bin/activate

# deps
pip install -U pip
pip install -r requirements.txt
# live ESP32 consumer (optional):
pip install pyserial

# lattice continuous run
python meta_field_distributed.py --world-size 1 --diagnostic --continuous --summary-interval 30
```

Press `Ctrl+C` to stop cleanly.

### Optical body path (no hardware required)

```bash
# synthetic passive sequence → JSONL
python optical_body_stub.py --clear-log --excitations 12

# promote observations → FieldMemoryEntry store
python optical_serial_consumer.py \
  --file /tmp/metafield/optical_phase0.jsonl \
  --save /tmp/metafield/field_memory.jsonl

# end-to-end smoke
python examples/optical_memory_smoke.py

# suggest next light (curiosity heuristic)
python active_probe.py --from-store /tmp/metafield/field_memory.jsonl --emit-command
```

Live board (after firmware is flashed):

```bash
# find the port (often /dev/ttyACM0 or /dev/ttyUSB0 on Arch)
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null

# dialout group so you don't need root for serial
sudo usermod -aG dialout $USER
# log out/in once after that

python optical_serial_consumer.py --port /dev/ttyACM0 --save /tmp/metafield/field_memory.jsonl
```

Optional flags for lattice run:
- `--export-stats` — write local stats.json for the sensing mod
- `--aurora-feed` — enable read-only drive force from Aurora (requires Redis)

---

## Current Capabilities

- Stable Hybrid Monte Carlo (HMC) with high acceptance rates
- Wilson gauge action + Wilson-Dirac operator
- Learned Information Geometry (autoencoder + Fisher metric + curvature estimation)
- Episodic memory with prioritized replay + `get_stats()` for observability
- Latent predictor that forms expectations about future behavior
- Dynamic geometry training (epochs scale with data volume)
- Efficient continuous mode (`--continuous`) with configurable system summaries
- Aurora environment feed (read-only drive force from swarm sensing)
- Physical Field Substrate (optical body schema, FieldMemoryStore, active_probe)

---

## Aurora + MetaField Super Hybrid (In Progress)

We have begun building toward a deep integration with [Aurora Swarm BTC](https://github.com/TheBabelDragon/aurora-swarm-btc).

Aurora provides the distributed swarm infrastructure **and** acts as the experimental nervous system for physical bodies. MetaField provides the physics simulation + growing intelligence layer.

See `aurora_mods/metafield_sensing/`, `INTEGRATION_PLAN.md`, `HYBRID_VISION.md`, `PHYSICAL_FIELD_SUBSTRATE.md`, `MEMORY_ARCHITECTURE.md`, `WHAT_WE_BUILT.md`.

---

## Key Components

| Component | Description |
|-----------|-------------|
| `DistributedHMC` | Core Hybrid Monte Carlo engine |
| `EpisodicMemory` | Lattice prioritized episodic memory |
| `FieldMemoryStore` | Field-body episodic buffer (optical, etc.) |
| `LatentPredictor` | Predicts future values from latents |
| `LearnedInformationGeometry` | Autoencoder + Riemannian geometry |
| `AttractorDynamics` | Persistent basins on the latent manifold |
| `AuroraFeed` | Read-only environment drive from the swarm |
| `active_probe` | Suggest next excitation from field memory |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- matplotlib / scikit-learn (optional)
- pyserial (optional, live ESP32)

## License

MIT License

---

*Actively evolving toward a distributed physics-based intelligence swarm with multiple physical bodies.*
