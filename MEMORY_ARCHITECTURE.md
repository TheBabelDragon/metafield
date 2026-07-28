# Memory Architecture for Physical Field Bodies

**Living document.**  
The optical substrate is moving from “I can observe a response” to “I remember what this response means.”

That requires three distinct memory layers. They are not interchangeable.

```
┌─────────────────────────────────────────────────────────────┐
│  Field Memory (episodic)          ← MetaField               │
│  “Yesterday, when this pattern occurred, the field looked   │
│   like this.”                                               │
│  Attractors · confidence · replay · meaning                 │
│  Implementation: FieldMemoryStore + FieldMemoryEntry         │
└──────────────────────────▲──────────────────────────────────┘
                           │ FieldObservation packets
┌──────────────────────────┴──────────────────────────────────┐
│  Persistent Calibration Memory (FRAM)  ← optical-body-s3    │
│  “Who am I?”                                                │
│  Optical identity · expected signatures · gain/offset       │
│  Survives power loss                                        │
└──────────────────────────▲──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Fast Working Memory (RAM / PSRAM)     ← ESP32-S3           │
│  Current laser state · current detector frame               │
│  Temporary geometry · active voxel estimates                │
└─────────────────────────────────────────────────────────────┘
                           │
                    MicroSD (experience archive)
                    JSONL logs · calibration history
                    thousands of optical experiments
```

---

## 1. Fast Working Memory (RAM equivalent)

**Owner:** ESP32-S3 (optical-body-s3)

**Purpose:** the current thought.

Holds:

- current laser excitation pattern
- current BPW34 / detector frame
- temporary geometry calculations
- active region / voxel estimates
- in-flight observation being assembled

The stock ESP32-S3 SRAM is sufficient for Phase 0–1 control loops.  
For larger field maps or longer sensor history, prefer an **ESP32-S3 with PSRAM**.

This layer is ephemeral. Power cycle → gone. That is correct.

---

## 2. Persistent Calibration Memory (FRAM)

**Owner:** optical-body-s3  
**Hardware:** MB85RC256V (I²C FRAM) — already selected

**Purpose:** “Who am I?”

Stores the optical identity that must survive power loss:

```
Optical identity
  Laser 0: expected detector signature
  Laser 1: expected detector signature
  …
  Detector 73: gain correction, offset
  geometry_version / fingerprint hash
  node_id
```

This is the transfer matrix / geometry fingerprint produced by the self-map, plus per-detector calibration constants.

FRAM is ideal: non-volatile, high endurance, fast enough for occasional writes after a calibration run.

---

## 3. Field Memory (episodic)

**Owner:** MetaField core

**Purpose:** “I remember what this response means.”

This is **not** normal RAM. It is episodic memory over field states.

**Implementation (Phase 0):**

- `schemas/field_memory.py` → `FieldMemoryEntry`
- `field_memory_store.py` → prioritized buffer (anomaly / low-confidence preferred)
- Parallel to lattice `EpisodicMemory` in `memory.py` (does not replace it)

Conceptual entry:

```python
FieldMemoryEntry:
    location:          # optional spatial / region key
    expected_response:
    observed_response:
    confidence:
    anomaly:
    timestamp:
    attractor_id:      # link into the existing attractor dynamics
    excitation_id:     # link back to the body packet
    body_id:
```

Example meaning:

> “Yesterday, when this laser pattern occurred, the field looked like this.”

Smoke test (no hardware):

```bash
python examples/optical_memory_smoke.py
```

The body only emits `FieldObservation` packets.  
MetaField decides what is worth remembering and how it links to attractors.

---

## Experience Archive (MicroSD)

**Owner:** optical-body-s3 (or a host that scrapes Serial/UDP)

Not fast. Not working memory.  
It is the long-term experiment log:

- thousands / millions of optical experiments
- JSONL observation streams
- calibration history
- replay datasets for MetaField offline training

Architecture wants:

```
FRAM     → identity + calibration
SD card  → experience archive
RAM      → current thought
```

MicroSD breakout is the next most useful hardware addition after FRAM.

---

## Hardware priority (memory-related)

| Priority | Item                         | Status / Note                          |
|----------|------------------------------|----------------------------------------|
| 1        | MB85RC256V FRAM              | ✅ already solved                      |
| 2        | MicroSD breakout             | ⭐ next most useful                    |
| 3        | ESP32-S3 with PSRAM (2nd node) | useful for larger field maps         |
| 4        | More RAM-like compute        | only after measurements prove the need |

Right now the nervous system exists.  
The next thing is giving it a memory of experience.

---

## Optical RAM (future, not now)

The 100 BPW34 + laser array can eventually become a form of physical memory, but not initially.

First version:

```
laser pattern
     ↓
optical transformation
     ↓
detector pattern
     ↓
memory lookup / model
```

Later (requires adaptive materials or active optical elements):

```
stimulus
   ↓
physical medium changes
   ↓
response changes
```

That is Phase 3+ territory. Do not build it yet.

---

## Relation to existing work

- `schemas/field_observation.py` — packet the body emits
- `schemas/field_memory.py` — `FieldMemoryEntry`
- `field_memory_store.py` — prioritized episodic store for field bodies
- `examples/optical_memory_smoke.py` — end-to-end Phase-0 smoke test
- `optical_body_stub.py` / `optical_serial_consumer.py` — synthetic body + host bridge
- [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3) — ESP32-S3 firmware (FRAM + SD)
- `memory.py` — lattice episodic memory (parallel path, not replaced)
- `PHYSICAL_FIELD_SUBSTRATE.md` — overall body abstraction

---

*Living document. Memory is what turns observation into identity and experience.*
