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
- The optical substrate is simply **another source of states**.

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

## Dual detector streams

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

---

## See also

- `MEMORY_ARCHITECTURE.md`
- `schemas/field_observation.py` / `schemas/field_memory.py`
- `field_memory_store.py` / `optical_body_stub.py` / `optical_serial_consumer.py`
- [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3)
- Issue #1

---

*Architecture complete. The next thing is measurement.*
