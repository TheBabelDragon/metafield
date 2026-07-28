# metafield_sensing

MetaField sensing integration mod for Aurora Swarm.

**Current version: 0.2.3**

## Purpose

Expose useful internal signals from MetaField into Aurora's sensing and monitoring systems.

Signals (schema v4):
- Episodic memory statistics (size, priority, exploration rate)
- Attractor landscape (count, total energy, budget, consistency)
- Prediction loss
- Geometry / reconstruction error + Fisher diagnostics + scalar curvature
- HMC health (acceptance rate, recent |ΔH|)
- Aurora drive force mode (when feed is active)
- Overall health string

## Why Start With Sensing?

This is the lowest-risk, highest-value first integration:
- Immediate value: Aurora operators can see what MetaField is doing
- Does not require modifying Aurora's core scheduling logic yet
- Builds the data foundation needed for smarter integrations later

## Current Status

- **v0.1** — Basic sensing hook + placeholder stats
- **v0.2** (current) — Real connection to MetaField via local `stats.json` export, richer schema (HMC + geometry + versioning), improved tick output

Still strictly **read-only / file-based**. No Redis publish.

## Multi-body direction

Future physical bodies (optical dodecahedron, WiFi CSI, ultrasonic, …) will publish through the shared `FieldObservation` schema defined in `schemas/field_observation.py`.  
This mod will eventually consume that common interface so the sensing surface stays body-agnostic. See `PHYSICAL_FIELD_SUBSTRATE.md`.

## Usage

On the MetaField side (continuous run with export):

```bash
python meta_field_distributed.py --world-size 1 --diagnostic --continuous --export-stats --summary-interval 30
```

The mod reads from the path defined by `METAFIELD_RUNTIME_DIR` (default `/tmp/metafield/stats.json`).

## Proposed Evolution

1. **v0.1** — Basic sensing hook + placeholder stats
2. **v0.2** (current) — Real file-based connection + richer schema
3. **v0.3** — Curvature and "interesting configuration" signals
4. **v0.4+** — Authenticated publish + task scheduling influence (requires control token)
5. **Later** — Consume `FieldObservation` packets from any body (optical Phase 0+, etc.)

## Integration Notes

- This mod is intentionally read-only at first (sensing only).
- Future versions can register additional hooks or task types to allow Aurora to schedule MetaField workloads.
- Fail-closed: if no stats file is present the mod reports `health=no_export` and continues.

---

*Part of the MetaField + Aurora super hybrid effort.*
