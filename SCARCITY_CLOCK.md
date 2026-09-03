# Scarcity clock

**Living document.** Wall-clock time is not MetaField's epoch.

This matches [aurora-swarm-btc](https://github.com/TheBabelDragon/aurora-swarm-btc) `mods/asset_fabric/artifact_clock.py`.

---

## Rule

Bitcoin is not the field store. Bitcoin is not observation identity.
Bitcoin supplies an independently verifiable temporal / scarcity anchor.

```
process time     traj / excitation_id / HMC step
scarcity time    btc_height + cumulative work     ← only authoritative epoch
observed_at      wall / monotonic                 ← debug and liveness only
```

Local wall time is never the observation epoch.
An observation can live without Bitcoin.
Unanchored packets are valid. They have no authoritative Bitcoin epoch.
Peer-supplied height/hash is evidence, not truth.
Never invent height from `time.time()` or `datetime.now()`.

---

## Packet

```python
from schemas.scarcity_clock import ScarcityClock, resolve_clock

ScarcityClock.unanchored()          # default
resolve_clock()                     # env → file → Redis tip; else unanchored
```

| Field | Role |
|-------|------|
| `epoch` / `btc_height` | coarse temporal coordinate |
| `btc_work` | scarcity weight (string, usually hex) |
| `btc_block_hash` | cryptographic tip identity |
| `anchor_id` | optional Aurora commitment id |
| `confidence` | `none` `pending` `included` `confirmed` `reorged` |
| `source` | `none` `env` `file` `aurora` `peer_claim` `explicit` |
| `observed_at` | informational |

`authoritative` is only `confirmed` + a real epoch. A `peer_claim` is never confirmed on arrival.

---

## Sources (fail open)

1. Explicit clock on the packet
2. `METAFIELD_BTC_HEIGHT` (+ optional `METAFIELD_BTC_BLOCK_HASH`, `METAFIELD_BTC_WORK`, `METAFIELD_BTC_CONFIDENCE`)
3. `METAFIELD_CLOCK_PATH` or `$METAFIELD_RUNTIME_DIR/btc_clock.json`
4. Redis: `aurora:btc:clock`, `aurora:clock`, `aurora:asset:clock:tip`
5. Else unanchored

Env and file tips are **included**, not confirmed. Confirmation is Aurora / `btc_anchor` work.

---

## Where it attaches

- `FieldObservation.clock` — schema v2
- `FieldMemoryEntry.clock`
- `WorkClaim` v3 — `btc_height` / `btc_work` / `clock_confidence` are in the evidence hash; wall `timestamp` is not
- `stats.json` via `write_local_stats`

`timestamp` fields remain as `observed_at` so existing JSONL replay still loads.

---

## Mint

Process liveness still uses live continuous PID + stats mtime. That is “is the writer alive”, not “when did this happen”.

Cooldown:

- always `Δtraj`
- anchored → `Δheight ≥ METAFIELD_MINT_COOLDOWN_BLOCKS` (default 1)
- unanchored → residual wall `METAFIELD_MINT_COOLDOWN_SEC` as anti-spam only

---

*Real-time is a bug. Scarcity is the clock.*
