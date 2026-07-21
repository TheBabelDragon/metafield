# metafield_sensing

Initial proposal for a MetaField sensing integration mod for Aurora Swarm.

## Goal

Expose useful signals from a running MetaField instance into Aurora’s sensing and monitoring layer.

Initial signals:
- Episodic memory size and priority distribution
- Recent prediction loss
- Geometry (autoencoder) loss
- Later: curvature / interesting configuration signals

## Why This First?

This is a low-risk, high-value first integration:
- Gives immediate observability of MetaField workloads inside Aurora
- Does not require scheduling or modifying core Aurora behavior yet
- Builds the foundation for deeper integration later (task scheduling, curvature-aware allocation, etc.)

## Next Steps (Proposed)

1. Flesh out `get_metafield_stats()` to actually read from a running MetaField process.
2. Integrate with Aurora’s real sensing / hook system once we have access to the Aurora codebase.
3. Expand to support scheduling MetaField continuous runs as Aurora tasks.
4. Add curvature / interestingness signals so Aurora can prioritize interesting physics regions.

## Status

This is currently a **skeleton / proposal**. The structure follows Aurora’s documented mod conventions.

We can iterate on this together before moving it into the main Aurora repository.
