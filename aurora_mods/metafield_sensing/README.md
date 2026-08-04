# metafield_sensing

MetaField sensing integration mod for Aurora Swarm.

**Current version: 0.3.0** (schema v5)

## Purpose

Expose useful internal signals from MetaField into Aurora's sensing and monitoring systems.

Signals (schema v5):
- Episodic memory statistics
- Attractor landscape
- Prediction loss
- Geometry / Fisher / scalar curvature
- HMC health (acceptance, |ΔH|)
- Aurora drive force mode
- **Swarm credit mint** (`total_credit`, last claim id) from `mint_state.json`
- Overall health string

## Status

- **v0.2** — File-based stats.json, schema v4
- **v0.3** (current) — Read-only mint totals alongside stats; still no Redis

## Usage

```bash
python meta_field_distributed.py --diagnostic --continuous --export-stats --summary-interval 30

# optional mint
export METAFIELD_CONTROL_TOKEN=…
python credit_mint.py --watch

# visual tick
python -m aurora_mods.metafield_sensing.entrypoint
```

Still strictly **read-only / file-based**. Fail-closed if no export.
