# What we built

MetaField is no longer only a simulated lattice.

It has a **physical body interface** — and the first body is an optical field instrument, not a sensor bolted on as an afterthought.

---

## The split that matters

```
MetaField     → interpretation (geometry, attractors, confidence, curiosity)
Aurora        → coordination (experiments, nodes, replay)
optical-body  → the body (excite, measure, remember who it is)
```

Same intelligence layer. Different embodiments. Optical is the first real one.

---

## The body does not “run the AI”

It does this:

```
dark frame
  → isolate every electrical source (one-hot)
  → dark-corrected transfer matrix
  → OpticalFingerprint → FRAM
  → on reboot: probe → “still me” or “drift” → remap only if needed
```

That is identity. Not a blink demo.

---

## Closing the circle (shaped light)

```
observe → FieldMemoryStore → active_probe (curiosity)
                ↓
         EXCITE <id> on Serial
                ↓
         body.exciteOnce → new observation
```

The body can be *told* which light to shape. MetaField can prefer sources with high anomaly / low confidence. Not full Phase 3 — the hinge is there.

---

## Two views of the same light

```
BPW34 → LM393   → what changed?     (reflex)
BPW34 → ADS1115 → how much?         (perception)
```

---

## Memory is three things, not one

| Layer | Question |
|-------|----------|
| RAM | what am I thinking right now? |
| FRAM | who am I? |
| MicroSD | what have I lived? |
| MetaField FieldMemoryStore | what does it mean? |

---

## First milestone

> The device can reboot, identify its optical body, and reproduce the same field response.

When that passes on hardware, MetaField has a persistent physical environment to learn from — not a simulation pretending to be one.

---

## Repos

- **This one** — schemas, FieldMemoryStore, `active_probe.py`, architecture docs  
- **[optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3)** — ESP32-S3 firmware (`EXCITE` / `MAP` / `VERIFY`)

BOM is frozen. Next work is measurement.

---

*We didn’t make the AI smarter first. We gave it a body that can recognize itself in the dark — and a way to choose the next light.*
