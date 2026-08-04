# Physical Field Substrate

**Living document.** The optical dodecahedral system is the first concrete body; the abstraction is deliberately broader.

---

## Core Insight

Any physical or simulated system should **not** become a special-case project inside MetaField.  
It should become **another embodiment of the same abstraction**.

```
MetaField Core (physics / inference engine)
        │
        │  same intelligence layer
        │
Physical Field Interface
        │
 ┌──────┴──────┬──────────────┬──────────────┬──────────────┐
 │ Optical     │ WiFi CSI     │ Ultrasonic   │ ZVS / HV     │ Simulation
 │ body        │ body         │ body         │ resonant     │ (lattice)
 └─────────────┴──────────────┴──────────────┴──────────────┘
```

- **MetaField** stays the physics/inference engine.
- Every substrate is simply **another source of states** that publishes `FieldObservation` packets.

---

## Bodies (current)

| Body | Repo | Status | Primary sense / act |
|------|------|--------|---------------------|
| Lattice (HMC) | this repo | live | simulated gauge field |
| Optical | [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3) | Phase 0 software + hardware architecture | laser excitation + BPW34 |
| Ultrasonic | [echo-grid-ultrasonic-os](https://github.com/TheBabelDragon/echo-grid-ultrasonic-os) | firmware + field kernel | 40 kHz transducers |
| **ZVS / Resonant HV** | [zvs-node](https://github.com/TheBabelDragon/zvs-node) (private) | architecture captured | isolated ESP32-S3 TWAI/CAN + ZVS power stage + flyback HV |
| WiFi CSI | [wifi-sensing-system](https://github.com/TheBabelDragon/wifi-sensing-system) | live | CSI spatial intelligence |

---

## First milestone (optical body)

> **The device can reboot, identify its optical body, and reproduce the same field response.**

Architecture and BOM are complete (see [optical-body-s3 BOM_AND_MILESTONE.md](https://github.com/TheBabelDragon/optical-body-s3/blob/main/BOM_AND_MILESTONE.md)).  
**Next deliverable is the first calibration dataset**, not more ICs.

Stack:

```
ESP32-S3 RAM → FRAM (identity) → MicroSD (archive) → MetaField memory
```

---

## Light body — Dodecabox optical aperture mask (v1)

The physical enclosure is **not** “paint some edges black.”  
It is a defined **optical aperture mask** on each of the 12 identical pentagonal modules.

Full specification: **[DODECABOX_OPTICAL_MASK.md](DODECABOX_OPTICAL_MASK.md)**

Summary of the controlled optical graph:

| Feature | Dimension |
|---------|-----------|
| Outer pentagon side | 37.103 mm |
| Inner pentagon side | 26.714 mm |
| Face separation | 11.56 mm |
| Outer black border | 3.00 mm |
| Inner cavity coating | 100 % matte black |
| Edge optical gate width | 1.50 mm |
| Vertex optical node | 2.00 mm |
| Bond line | 0.10–0.20 mm |
| Optional photon seam | 0.75 mm |

Across 12 faces this yields:

- **60** edge gates
- **60** vertex optical nodes
- **12** illumination / sensing surfaces

The result is a black optical cavity with five controlled transmission channels per face — a defined optical graph instead of an uncontrolled glowing object.

---

## ZVS Node (new)

Isolated high-power resonant body:

- ESP32-S3 + ADM3053 (TWAI/CAN) on the logic side (GND1)
- Resonant Royer/ZVS power stage + flyback HV output on the bus/HV side (GND2)
- Hardware E-STOP that cannot be defeated by firmware
- Telemetry: isolated current (ACS781), voltage (AMC1301), temperatures, E-STOP state, fan

It can act as:
1. High-efficiency resonant driver for ultrasonic (or other) transducers, or
2. Experimental high-voltage field generator via the replaceable flyback cartridge.

Because it speaks (or will speak) the same `FieldObservation` contract, MetaField does not need to know whether the “body” is optical photons, acoustic pressure, or resonant power health.

See the private [zvs-node](https://github.com/TheBabelDragon/zvs-node) repo for full isolation rules, placement, and mechanical stack.

---

## Dual detector streams (optical)

```
BPW34 → LM393  → "event happened"     (reflex)
BPW34 → ADS1115 → "how much happened" (perception)
```

Details: optical-body-s3 `DETECTOR_ARCHITECTURE.md`.

---

## The Important Interface Is Not Raw Sensors

Bodies publish **FieldObservation** (`schemas/field_observation.py`).  
MetaField stores **FieldMemoryEntry** (`schemas/field_memory.py` + `field_memory_store.py`).

---

## Memory layers

See `MEMORY_ARCHITECTURE.md`:

| Layer | Owner | Role |
|-------|--------|------|
| RAM/PSRAM | ESP32-S3 | current thought |
| FRAM | optical-body-s3 | “Who am I?” |
| MicroSD | optical-body-s3 | experience archive |
| Field Memory | MetaField | meaning / replay |

---

## Clean calibration (whatsinthebox)

1. Dark frame (all OFF) → `D`
2. `ExcitationSequence` one-hot → `R_corrected = R − D`
3. `OpticalFingerprint` → FRAM

GPIO pin map last; abstractions stay `fire` / `readAll` / `readMask`.

---

## Aurora’s Role

Experimental nervous system (schedule, nodes, replay).  
MetaField decides meaning, surprise, prediction.

---

## Roadmap (optical)

- **Phase 0** — passive observability + clean calibration + identity on reboot ✅ software
- **Phase 1** — trusted transfer matrix / drift detection
- **Phase 2** — predictive model
- **Phase 3** — active exploration

## Roadmap (ZVS)

- **Phase 0** — architecture + isolation rules + telemetry schema (current)
- **Phase 1** — ESP32-S3 TWAI firmware emitting health `FieldObservation`s
- **Phase 2** — closed-loop with MetaField (temperature / current as additional sense modalities)

---

## See also

- `DODECABOX_OPTICAL_MASK.md` — light-body aperture mask (v1)
- `MEMORY_ARCHITECTURE.md`
- `schemas/field_observation.py` / `schemas/field_memory.py`
- `field_memory_store.py` / `optical_body_stub.py` / `zvs_body_stub.py` / `optical_serial_consumer.py`
- [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3)
- [zvs-node](https://github.com/TheBabelDragon/zvs-node) (private)
- Issue #1

---

*Architecture complete for optical. ZVS architecture captured. The next thing is measurement.*
