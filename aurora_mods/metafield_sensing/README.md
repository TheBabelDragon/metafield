# metafield_sensing

MetaField sensing integration mod for Aurora Swarm.

## Purpose

Expose useful internal signals from MetaField into Aurora's sensing and monitoring systems.

Initial signals:
- Episodic memory statistics (size, priority distribution)
- Prediction loss
- Geometry / autoencoder loss
- Later: curvature and "interesting configuration" signals

## Why Start With Sensing?

This is the lowest-risk, highest-value first integration:
- Immediate value: Aurora operators can see what MetaField is doing
- Does not require modifying Aurora's core scheduling logic yet
- Builds the data foundation needed for smarter integrations later

## Current Status

This is an **early implementation / proposal**.

The structure follows Aurora's mod conventions (`manifest.json` + `entrypoint.py` + hooks).

`get_metafield_stats()` currently returns placeholder data. The next step is to implement a real connection to a running MetaField process.

## Proposed Evolution

1. **v0.1** (current) — Basic sensing hook + placeholder stats
2. **v0.2** — Real connection to MetaField (shared memory / file / API)
3. **v0.3** — Richer signals (curvature, reconstruction quality, memory priority distribution)
4. **v0.4+** — Move beyond sensing into task scheduling and curvature-aware compute allocation

## Integration Notes

- This mod is intentionally read-only at first (sensing only).
- Future versions can register additional hooks or task types to allow Aurora to schedule MetaField workloads.

## Next Immediate Work

- Decide on the best way for this mod to read live data from MetaField
- Implement real `get_metafield_stats()`
- Test hook registration once we have access to Aurora's runtime

---

*Part of the MetaField + Aurora super hybrid effort.*
