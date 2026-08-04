# Process pipeline — continuity, security, JSON, sensing

How the pieces copy data to each other without opening the network early.

```
┌─────────────────────────────────────────────────────────────┐
│  meta_field_distributed.py  (--continuous)                  │
│                                                             │
│  ContinuousLock  ──►  continuous.lock   (singleton ownership)│
│        │                                                    │
│        │  every --summary-interval                          │
│        ▼                                                    │
│  --export-stats  ──►  stats.json        (snapshot, 0600)    │
│        │                                                    │
│        │  Ctrl+C / clean exit                               │
│        ▼                                                    │
│  release lock + write stats live=false, health=stopped      │
└──────────────────────────────┬──────────────────────────────┘
                               │  read-only file
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  aurora_mods/metafield_sensing  (visual / operator path)    │
│                                                             │
│  read_local_stats()  ←  stats.json                          │
│  on_sensing_tick()   →  human-readable line / Aurora hook   │
│                                                             │
│  NO Redis yet. NO control commands. Fail closed.            │
└─────────────────────────────────────────────────────────────┘
```

Paths (same runtime dir for both):

| File | Role |
|------|------|
| `continuous.lock` | Who owns the continuous process |
| `stats.json` | What the process is doing (visual + sensing) |

Location order: `METAFIELD_RUNTIME_DIR` → `$XDG_RUNTIME_DIR/metafield` → `/tmp/metafield`.

---

## 1. Continuity

`--continuous` means: run until interrupt, one brain on this machine.

- Acquires **ContinuousLock** before the HMC loop.
- Second continuous start → `ContinuousLockError` (duplicate path prohibited).
- Clean exit / Ctrl+C → lock released; optional final stats marked `stopped`.
- Stale lock (dead PID / corrupt) → auto-cleaned on next acquire.
- Escape hatch: `METAFIELD_FORCE_UNLOCK=1` (still no manual `rm`).

Continuity is **process identity**, not the visual stream.

---

## 2. JSON save (the copy)

`--export-stats` is the only writer of `stats.json` during the run.

Each summary tick (when diagnostic + continuous + export):

```json
{
  "schema_version": 4,
  "version": "…",
  "traj": 30,
  "health": "Good",
  "live": true,
  "hmc": { "acceptance_rate": 0.55, "recent_abs_dh": 0.8, … },
  "memory": { … },
  "attractors": { … },
  "prediction": { … },
  "geometry": { … },
  "aurora": { … },
  "control_enabled": false
}
```

Write path is atomic (`*.tmp` → replace) and mode `0600` when the OS allows.

**This is the copy boundary:** process memory → one JSON file → any reader.
No second continuous process, no shared memory, no Redis until auth exists.

On clean shutdown the lock release path sets `live: false`, `health: "stopped"` so sensing does not keep showing a live run.

---

## 3. Security

| Surface | Default | Gate |
|---------|---------|------|
| Continuous singleton | on with `--continuous` | file lock + PID liveness |
| Stats export | off unless `--export-stats` | local file only |
| Control / overlord | **disabled** | `METAFIELD_CONTROL_TOKEN` must be set + match |
| Sensing mod | read-only | never writes lock or stats |

Rules of thumb:

- **Lock** = “is a continuous MetaField allowed to start?”
- **Token** = “is anyone allowed to *command* MetaField?”
- **JSON** = “what may be *observed* without commanding?”

Sensing never needs the control token. Future write/schedule hooks do.

---

## 4. Visual pipeline (operator path)

There is no separate GUI process yet. The visual path **is** the sensing path:

```bash
# terminal A — brain + export
python meta_field_distributed.py \
  --diagnostic --continuous --export-stats --summary-interval 30

# terminal B — visual / tick consumer
python -m aurora_mods.metafield_sensing.entrypoint
# or: from entrypoint import on_sensing_tick; on_sensing_tick()
```

`on_sensing_tick()` prints one line:

```text
[metafield_sensing] traj=30 | health=Good | mem=… | accept=0.55 | |dH|=0.80 | …
```

If export is off or process never started → `health=no_export`.  
If process stopped cleanly → `health=stopped`.

That is the finished **process → file → view** loop. Aurora can later hang the same `get_metafield_stats()` off a real dashboard without changing MetaField’s writer.

---

## 5. What does *not* cross the boundary yet

| Not yet | Why |
|---------|-----|
| Redis publish | unauthenticated network surface |
| Remote control | needs token + explicit design |
| Optical FieldObservation in stats.json | body pipeline is separate (JSONL / serial); merge later as optional `bodies` block |
| Second continuous process | lock forbids it by design |

---

## Quick reference

```bash
# full local pipeline
export METAFIELD_RUNTIME_DIR=${METAFIELD_RUNTIME_DIR:-$XDG_RUNTIME_DIR/metafield}

python meta_field_distributed.py \
  --diagnostic --continuous --export-stats --summary-interval 30

# elsewhere
python -c "from aurora_mods.metafield_sensing.entrypoint import on_sensing_tick; on_sensing_tick()"

# stuck lock after a kill -9
METAFIELD_FORCE_UNLOCK=1 python meta_field_distributed.py --diagnostic --continuous --export-stats
```

**Copy rule:** only the continuous owner writes `stats.json`; everyone else reads. Lock and token stay out of the visual path.
