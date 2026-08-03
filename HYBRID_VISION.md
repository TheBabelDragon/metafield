# MetaField + Aurora Swarm — Super Hybrid Vision

**Goal**: Combine Aurora’s distributed swarm infrastructure with MetaField’s physics simulation + memory/prediction layer into one cohesive long-term system.

An intelligence that grows up inside a mathematically consistent simulated universe *and* real physical field substrates, running across many machines through a coordinated swarm.

---

## Core Philosophy

- **Aurora** provides the **swarm substrate** (and experimental nervous system): node discovery, coordination, scheduling, resilience, community compute, modular extensibility, experiment orchestration, and replay.
- **MetaField** provides the **physics + intelligence core**: lattice gauge theory simulation, learned geometry, episodic memory, prediction, and eventual agency (curiosity, goals, active experimentation, world modeling).
- Over time, MetaField’s internal state (memory, predictions, curvature, interesting configurations) can influence Aurora’s scheduling and resource allocation.

This is not "MetaField on top of Aurora" or "Aurora with some MetaField mods". It is a true hybrid where both systems evolve together.

**Important generalization (2026-07 / 2026-08):** MetaField is no longer tied exclusively to the simulated lattice. The lattice is one *body*. Optical, ultrasonic, WiFi-CSI, **ZVS resonant/HV**, and future physical substrates are additional bodies that feed the same intelligence layer. See `PHYSICAL_FIELD_SUBSTRATE.md`.

---

## High-Level Architecture (Target State)

```
                    Community / Distributed Machines
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                        Aurora Swarm Layer                     │
│  • Node discovery & coordination                              │
│  • Scheduler + Overlord                                       │
│  • Comms layer (incl. future CAN / TWAI bridges)              │
│  • Sensing / Monitoring                                       │
│  • Modular extension system (mods/hooks)                      │
│  • Community compute distribution                             │
│  • Experimental nervous system (excitation sequences, replay) │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    MetaField Intelligence Core                │
│  • Lattice QCD Simulation (HMC + Wilson-Dirac)  ← one body    │
│  • Optical / Ultrasonic / ZVS / other physical substrates     │
│  • Learned Information Geometry (autoencoder + curvature)     │
│  • Episodic Memory + Prioritized Replay                       │
│  • Latent Predictor / Expectation Formation                   │
│  • Future: Curiosity, Goals, Active Experimentation           │
│  • Future: Internal World Model                               │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    Emergent Agency & Reasoning
```

---

## Phased Roadmap

### Phase 0 — Foundations (Current)
- Stabilize single-machine MetaField with `--continuous` mode
- Strengthen episodic memory + prediction layer
- Improve diagnostics and robustness
- Begin light modularization of MetaField components
- Define the Physical Field Interface abstraction (see `PHYSICAL_FIELD_SUBSTRATE.md`)
- Capture ZVS node architecture + Phase-0 stub

### Phase 1 — First Integration
- Make MetaField simulation + memory loop runnable as an Aurora worker/mod
- Use Aurora’s scheduling instead of manual `--continuous` runs
- Basic task distribution across machines via Aurora
- Optical body: Phase 0 (passive observability) + Phase 1 (self-calibration / transfer map)
- ZVS body: live TWAI telemetry as `FieldObservation`s

### Phase 2 — Deeper Integration
- MetaField components become first-class Aurora mods (simulation, memory, prediction, geometry)
- Aurora’s sensing layer feeds into MetaField memory
- MetaField prediction/curvature signals begin influencing Aurora scheduling
- Optical body: predictive model + first adaptive probing
- Multi-body experiments (optical + ZVS coordinated via shared protocol)

### Phase 3 — Agency Emergence
- Curiosity-driven exploration
- Goal-directed behavior
- Active experimentation ("What if we change β?" / "Which laser pattern yields most information?" / "How does resonant health correlate with field response?")
- Internal world modeling
- MetaField memory/prediction systems start shaping long-term swarm behavior
- Physical reconfiguration of bodies becomes possible

---

## Key Integration Points (Future)

| Aurora Concept       | MetaField Mapping                          | Integration Type      |
|----------------------|--------------------------------------------|-----------------------|
| Worker               | HMC trajectory runner + memory updater     | Mod / Task            |
| Scheduler            | Decides which configurations / excitations to explore | Influenced by curvature/prediction |
| Sensing              | Hardware + simulation state monitoring     | Feeds MetaField memory |
| Mod/Hook System      | Plug in new MetaField behaviors + new bodies | Primary extension point |
| Overlord / Control   | High-level swarm coordination              | Shared governance     |
| Experimental nervous system | Schedule excitation sequences, replay | Optical / ZVS / other bodies |
| CAN / TWAI bridge    | Direct path from isolated power nodes      | Future transport      |

---

## Why This Hybrid Matters

Most distributed ML systems treat compute as a dumb resource pool.

This hybrid treats the **swarm itself as an environment** in which an intelligence can grow — with real physics, structured memory, and eventually its own goals and reasoning.

It combines:
- The reliability and coordination of a production swarm system (Aurora)
- The richness of a mathematically grounded physical simulation *and* real physical field substrates (MetaField)
- The long-term possibility of emergent agency

The optical dodecahedron, the ultrasonic array, and the ZVS resonant nodes are not “the AI.” They are persistent physical environments that give MetaField something to learn against.

---

## Current Status (as of 2026-08)

- Strong single-machine HMC + geometry + episodic memory + prediction
- Continuous mode (`--continuous`) supported
- Memory prioritization and online prediction working
- Distributed multi-machine still fragile (Gloo + localhost issues)
- Physical Field Substrate abstraction documented and extended to ZVS
- Optical and ZVS Phase-0 stubs available for parallel development

Next focus: Stabilize single-machine experience, finish clean integration points with Aurora’s mod system, begin Phase 0 passive observability for the first optical body, and bring ZVS telemetry online.

---

*This document is living. It will evolve as we build.*
