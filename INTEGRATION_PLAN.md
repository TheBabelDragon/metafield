# MetaField + Aurora Integration Plan

**Goal**: Turn MetaField into a first-class citizen inside the Aurora swarm so it can benefit from distributed coordination, community compute, and Aurora’s modular architecture — while keeping MetaField’s physics + intelligence core intact.

This is a phased, pragmatic plan.

---

## Guiding Principles

- Start small and concrete. Avoid big rewrites early.
- Make MetaField components easy to plug into Aurora’s existing `mods/` + hook system.
- Keep the single-machine experience excellent as the foundation.
- Let Aurora handle distribution, scheduling, and node management.
- Allow MetaField’s memory/prediction/curvature signals to eventually influence Aurora’s decisions.

---

## Phase 1: Preparation (Current Focus)

### 1.1 Make MetaField more modular (Internal)

**Goal**: Make it easy to extract and reuse core pieces.

- Separate concerns clearly:
  - `simulation/` — HMC engine + lattice operations
  - `memory/` — EpisodicMemory + prioritization
  - `prediction/` — LatentPredictor
  - `geometry/` — LearnedInformationGeometry
  - `core/` — Shared utilities and config
- Add clear interfaces / abstract base classes where helpful.
- Improve continuous mode so it’s observable and resumable.

### 1.2 Improve diagnostics and observability

- Better structured logging during long runs.
- Expose key signals (memory size, prediction loss, average priority, curvature) in a machine-readable way.
- These signals will later be useful for Aurora’s sensing layer.

### 1.3 Stabilize single-machine continuous experience

- Make `--continuous` the primary development mode.
- Clean shutdown + state saving (future).
- Periodic summaries that are useful for both humans and future schedulers.

---

## Phase 2: First Integration Points

### 2.1 Run MetaField as an Aurora Worker / Mod

**Goal**: Make the current HMC + memory + prediction loop schedulable by Aurora.

Possible approaches:

- Wrap the main training/simulation loop as a simple Aurora mod.
- Use Aurora’s existing worker infrastructure instead of manual `--continuous`.
- Start with single-node execution inside Aurora, then expand to multi-node.

**First concrete deliverable**:
- A minimal wrapper that lets Aurora start a MetaField continuous run as a task.
- Basic reporting back of memory/prediction stats to Aurora’s sensing layer.

### 2.2 Map MetaField concepts to Aurora concepts

| MetaField Component       | Aurora Equivalent          | Integration Type      | Priority |
|---------------------------|----------------------------|-----------------------|----------|
| EpisodicMemory            | Mod state / memory store   | Mod internal state    | High     |
| LatentPredictor           | Prediction hook            | Mod                   | Medium   |
| LearnedInformationGeometry| Sensing + feature extraction | Sensing mod        | Medium   |
| HMC trajectory loop       | Scheduled task / workload  | Worker task           | High     |
| Curvature / interesting configs | Scheduling signal     | Influencer            | Medium   |

---

## Phase 3: Deeper Integration

- Allow Aurora’s scheduler to use MetaField signals (e.g. high curvature regions → prioritize those configurations).
- Make memory replay and prediction part of Aurora’s sensing/mod system.
- Explore bidirectional influence (Aurora scheduling decisions affect MetaField memory, and vice versa).

---

## Open Questions

- Should MetaField become a set of Aurora mods, or should it remain a mostly separate codebase that Aurora schedules?
- How much of Aurora’s existing Bitcoin-mining-oriented design needs to be generalized?
- What is the minimal viable integration that already provides value?

---

## Current Status

- `HYBRID_VISION.md` created
- `INTEGRATION_PLAN.md` created
- MetaField has working episodic memory + prediction in continuous mode
- Distributed multi-machine still needs work (Gloo issues)

Next: Focus on modularity + continuous mode improvements while we design the first Aurora integration points.

---

*This is a living document. Feedback and adjustments welcome.*
