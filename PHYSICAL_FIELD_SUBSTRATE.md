# Physical Field Substrate

**Living document.** The optical dodecahedral system is the first concrete body; the abstraction is deliberately broader.

---

## Core Insight

The optical system should **not** become a special-case project inside MetaField.  
It should become **another embodiment of the same abstraction**.

```
MetaField Core (physics / inference engine)
        │
        │  same intelligence layer
        │
Physical Field Interface
        │
 ┌──────┴──────┬──────────────┬──────────────┐
 │ Optical     │ WiFi CSI     │ Ultrasonic   │ Simulation
 │ body        │ body         │ body         │ (lattice)
 └─────────────┴──────────────┴──────────────┘
```

- **MetaField** stays the physics/inference engine.  
  It already possesses state evolution, geometry representation, energy/error concepts, episodic memory + prioritized replay, attractor dynamics, and prediction.  
  Those capabilities remain valuable regardless of the source of states.

- The optical substrate (dodecahedral plexi volume + laser excitation + BPW34 projections) is simply **another source of states**.

Instead of only:

```
lattice state → MetaField → prediction
```

you also have:

```
optical field state → MetaField → prediction
```

Same intelligence layer, different body.

---

## The Important Interface Is Not Raw Sensors

Avoid exposing low-level values such as:

```json
{
  "bpw34_17": 0.62,
  "bpw34_18": 0.71
}
```

directly into MetaField.

The optical node (and every future body) should publish a **higher-level observation schema**.

**Formal definition lives in** `schemas/field_observation.py` (`FieldObservation` + `FieldRegion`).

Example payload:

```json
{
  "schema_version": 1,
  "body_id": "optical-dodeca-01",
  "body_type": "optical",
  "excitation_id": 142,
  "field_regions": [
    {
      "region": "bottom_plane_cluster_3",
      "expected": 0.82,
      "observed": 0.76,
      "confidence": 0.91,
      "anomaly": 0.05
    }
  ],
  "geometry_state": "calibrated",
  "timestamp": "2026-07-28T04:00:00+00:00",
  "modality": {},
  "health": "ok"
}
```

This keeps MetaField agnostic.  
A future ultrasonic node, WiFi CSI node, optical node, or the existing simulated lattice can all feed the **same conceptual interface**.

The “field” that MetaField models is therefore not physical coordinates alone; it is the learned relationship between states.

---

## Memory layers

Observation is not enough. The body needs memory of experience.

Three distinct layers (see `MEMORY_ARCHITECTURE.md` for the full design):

| Layer | Owner | Purpose |
|-------|--------|--------|
| Fast working memory (RAM/PSRAM) | ESP32-S3 | current laser state, detector frame, temporary geometry |
| Persistent calibration (FRAM) | optical-body-s3 | “Who am I?” — identity + expected signatures |
| Field / episodic memory | MetaField | “I remember what this response means.” |
| Experience archive (MicroSD) | optical-body-s3 | long-term JSONL experiment log |

---

## Optical Structure as Physical Measurement Substrate

The relationship between the optical structure and MetaField is where the hardware becomes interesting.  
The optical device is not “a sensor attached to MetaField”; it becomes a **physical measurement substrate** for MetaField.

Useful abstraction:

```
Physical optical field
        ↓
Measurement layer
        ↓
MetaField representation
        ↓
Prediction / inference
        ↓
New excitation pattern
        ↺
```

The dodecahedral plexi volume provides:

- geometry
- symmetry
- transformations
- constraints
- nonlinear optical behavior

It becomes a physical function:

```
O = f(L, G, E)
```

where:

- `L` = laser excitation pattern
- `G` = geometry of the enclosure
- `E` = environment / object state
- `O` = observed BPW34 projection

MetaField’s job is **not** to recreate the optics perfectly.  
It learns the useful relationship between excitation and observation.

### How MetaField would see it

Instead of storing raw sensor readings it builds something closer to:

```
Optical state:
  Region A: confidence 0.96, expected response X
  Region B: confidence 0.71, changing
  Region C: anomaly detected
```

---

## Aurora’s Role

Aurora is **not** the intelligence.

Aurora becomes the **experimental nervous system**:

- schedule excitation sequences
- manage nodes
- coordinate experiments
- handle memory / replay
- decide when to probe

MetaField decides:

- what the observations mean
- what latent state exists
- what is surprising
- what prediction is likely

That division is clean and matches the existing HYBRID_VISION separation (Aurora = swarm substrate, MetaField = physics + intelligence core).

---

## Adaptation Path for the Structure

The structure needs to adapt in three progressive ways.

### 1. Self-calibration (“Understand myself”)

The system runs a probe sequence:

```
Laser 1 → measure
Laser 2 → measure
Laser 3 → measure
...
```

It builds the optical transfer map / matrix:

```
M_ij = response of detector j to emitter i
```

This replaces hand-calculating every reflection and refraction.  
The dodecahedron is no longer a container; it is a **learned transformation operator**.  
That matrix is the optical fingerprint of the structure and lives in FRAM.

### 2. Adaptive probing

Eventually the system stops blindly scanning.

```
Current belief: “I am uncertain about this region.”
        ↓
Activate specific lasers
        ↓
Collect more information
        ↓
Update model
```

The structure becomes an active participant.

### 3. Physical reconfiguration (later)

Possible future additions:

- movable laser mounts
- adjustable mirrors
- shutters
- variable filters
- different internal optical materials

Then MetaField is not just learning a fixed environment; it is learning how changing the environment changes the field.

---

## Recommended Roadmap (Optical Body)

**Phase 0 — Passive observability**  
Make the optical node boring first.  
It must reliably answer:

- What laser fired?
- What did every detector see?
- What changed?
- Can I replay the exact experiment?

No learning yet. Stable perception before curiosity.

Software side (already available):

```bash
python optical_body_stub.py --clear-log --excitations 12
python optical_body_stub.py --replay-only
```

Firmware side (ESP32-S3 physical node):

- New repo: [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3)
- Real BPW34 → ADS1115 path preferred from day one
- Emits the same conceptual `FieldObservation` packets
- Self-map on boot creates the first geometry fingerprint (intended for FRAM)
- FRAM + MicroSD memory layers scaffolded

See issue #1.

**Phase 1 — Self-calibration / Optical identity**  
Learn the transfer map `M_ij`.  
“What does my own body look like?”

**Phase 2 — Predictive model**  
“Given laser pattern X, I expect detector pattern Y.”

**Phase 3 — Active exploration**  
“Which laser pattern gives me the most information?”

This progression mirrors biological development: stable perception → self-model → prediction → curiosity-driven probing.

---

## Architectural Shift

A normal computer:

```
data → computation → output
```

This system:

```
stimulus → physical transformation → observation → learned interpretation → new stimulus
```

The dodecahedron is effectively a physical layer of computation — not because the plexiglass is thinking, but because its geometry transforms information before the software ever sees it.

The adaptation sequence is therefore not “make the AI smarter first.” It is:

1. Make the optical field repeatable (Phase 0).
2. Let MetaField learn the transfer behavior (Phase 1).
3. Introduce controlled changes (Phase 2–3).
4. Let the model learn which changes matter.

That gives MetaField something it normally lacks: a **persistent, physically grounded environment** to model.

---

## Relation to Existing Work

- The current lattice QCD path remains a first-class body (simulation body).
- The existing `metafield_sensing` mod, stats schema, Aurora feed, and episodic memory already provide the scaffolding needed for any new body.
- New bodies only need to speak the higher-level observation schema (`schemas/field_observation.py`); the rest of MetaField stays unchanged.

See also:

- `MEMORY_ARCHITECTURE.md` — three-tier memory (RAM / FRAM / Field Memory / SD archive)
- `schemas/field_observation.py` — formal `FieldObservation` / `FieldRegion` types + helpers
- `schemas/field_memory.py` — `FieldMemoryEntry` for episodic field experience
- `optical_body_stub.py` — Phase-0 synthetic passive loop + replayable JSONL log
- [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3) — ESP32-S3 firmware (real BPW34 path preferred)
- `HYBRID_VISION.md` — overall MetaField + Aurora architecture
- `INTEGRATION_PLAN.md` — current integration status and next gated steps
- `aurora_mods/metafield_sensing/` — the sensing surface that future bodies will feed
- Issue #1 — Optical body Phase 0 tracking

---

*This document is living. It will evolve as the first optical body is built and as additional bodies appear.*
