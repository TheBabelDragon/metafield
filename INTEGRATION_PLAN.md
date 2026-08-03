# MetaField + Aurora Integration Plan

**Updated for v1.50 + Physical Field Substrate + Phase-0 stub + ZVS body**

---

## Current status (v1.50+)

### Done
- Soft expandable episodic memory + force-based attractors + homeostasis + adaptive basins
- Continuous singleton lock (duplicate continuous prohibited)
  - Acquire recovers from dead / corrupt / unreadable locks automatically
  - **No manual file deletion required**
  - Escape hatch: `METAFIELD_FORCE_UNLOCK=1` (still no `rm` needed)
  - Cleaner runtime path selection (`METAFIELD_RUNTIME_DIR` → `XDG_RUNTIME_DIR` → `/tmp`)
  - Release on clean exit is intentional and correct
  - Clean release also writes `health="stopped"` so sensing is not left with zombie data
- Control surface fail-closed (`METAFIELD_CONTROL_TOKEN`)
- Local stats export (`--export-stats`) — file only, no Redis publish
- **Aurora environment feed (read-only)**
  - Start prompt from live sensing context
  - Drive force → exploration scale, energy budget scale, interest gate bias
  - Degrades gracefully if Redis unavailable
- **Richer local sensing surface (schema v3/v4)**
  - HMC acceptance rate + recent |ΔH|
  - Geometry reconstruction error + train loss
  - Occasional scalar curvature probe
  - Clear health string + explicit `live` / `stopped` signals
  - Versioned schema + improved `metafield_sensing` mod (0.2.3)
- **Attractor → geometry deformation loop is active**
  - High-interestingness experiences reinforce attractors
  - Attractors are passed into geometry training so the manifold begins to deform around persistent basins
  - Attractor influence weight = 0.4
- **Physical Field Substrate abstraction documented** (`PHYSICAL_FIELD_SUBSTRATE.md`)
  - MetaField remains the intelligence layer; lattice / optical / WiFi / ultrasonic / **ZVS** are interchangeable bodies
  - Higher-level observation schema preferred over raw sensor values
  - Formal `FieldObservation` types in `schemas/field_observation.py`
- **Optical Phase-0 software stub** (`optical_body_stub.py`)
  - Synthetic passive excitation loop
  - Emits valid `FieldObservation` packets
  - Replayable JSONL log
  - Lets Aurora / MetaField side develop in parallel with real hardware
- **ZVS Phase-0 architecture + stub** (`zvs_body_stub.py`)
  - Private hardware repo: [zvs-node](https://github.com/TheBabelDragon/zvs-node)
  - Synthetic health / telemetry observations (current, voltage, temps, E-STOP)
  - Same observation contract so MetaField can treat resonant-power health as just another modality

### Not yet (still gated)
- Redis *publish* from MetaField into Aurora channels
- Overlord remote commands (requires control token + explicit design)
- Scheduler task wrapper for HMC continuous runs
- Real optical hardware/firmware path (BPW34 + lasers)
- Optical body Phase 1 (transfer-matrix self-calibration)
- Real ZVS firmware (ESP32-S3 TWAI + isolated sensors)

---

## Continuous lock design note

**Releasing the lock on clean exit is correct.**  
Leaving it held would block the next continuous run.

**Manual deletion is never required.** The acquire path:
1. Detects dead PIDs and cleans automatically
2. Treats corrupt / unreadable lock files as stale and cleans them
3. Offers `METAFIELD_FORCE_UNLOCK=1` as an escape hatch if you are certain a process is gone (still no `rm`)

Runtime path priority:
1. `METAFIELD_RUNTIME_DIR` (explicit)
2. `$XDG_RUNTIME_DIR/metafield` (Linux user runtime)
3. `/tmp/metafield` (fallback)

On clean release we also write a final stats snapshot with `health="stopped"` so sensing consumers do not keep reporting stale live data.

---

## Aurora feed → MetaField drive force

| Aurora environment | MetaField effect |
|--------------------|------------------|
| Empty / scale_up | exploration ↑, energy budget ↑ |
| High occupancy / scale_down | exploration ↓, energy budget ↓ |
| Anomaly / security | interest gate lowered, exploration moderated |
| Unavailable | neutral (local-only dynamics) |

Enable with:

```bash
export REDIS_URL=redis://127.0.0.1:6379/0   # optional; auto-detects
python meta_field_distributed.py --world-size 1 --diagnostic --continuous --aurora-feed --export-stats
```

Without Redis, MetaField still runs; feed reports unavailable and drive force stays neutral.

---

## Local sensing (current foundation)

```bash
python meta_field_distributed.py --world-size 1 --diagnostic --continuous --export-stats --summary-interval 30
```

Writes a versioned `stats.json` that the `metafield_sensing` mod consumes. On clean exit the same file is updated to `health="stopped"`.

Future bodies (optical, ZVS, etc.) should publish into the same higher-level observation schema so the sensing surface remains body-agnostic. See `PHYSICAL_FIELD_SUBSTRATE.md` and `schemas/field_observation.py`.

Optical Phase-0 stub for parallel development:

```bash
python optical_body_stub.py --clear-log --excitations 12
python optical_body_stub.py --replay-only
```

ZVS Phase-0 stub:

```bash
python zvs_body_stub.py --clear-log --cycles 20
python zvs_body_stub.py --replay-only
```

---

## Next integration steps (when security allows)

1. Authenticated publish of MetaField stats onto `aurora:sensing:*` (token required)
2. Aurora mod registration for live dashboard
3. Scheduler influence (prefer nodes when MetaField interest is high)
4. Overlord start/stop of continuous runs under singleton lock
5. Real optical hardware path replacing the synthetic stub (issue #1)
6. Optical Phase 1 — transfer-matrix self-calibration
7. ZVS firmware that emits live telemetry as `FieldObservation`s over CAN or serial

---

*Living document.*
