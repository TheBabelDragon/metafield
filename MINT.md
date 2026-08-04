# Credit mint (internal swarm credits)

**Python route, not Git. Capability token, not a wallet.**

This mints *internal* regulation credits from observed MetaField work
(`stats.json`), only when `METAFIELD_CONTROL_TOKEN` is set and matches.

It does **not**:

- touch Git history as proof of work
- issue chain assets / BTC / external currency
- open network publish (still local JSONL)

---

## Flow

```
Aurora-style regulation (env caps)
        │
METAFIELD_CONTROL_TOKEN  (capability — fail closed)
        │
stats.json  (Python continuous + --export-stats)
        │
CreditMint.try_mint_from_stats()
        │
work_claims.jsonl  +  mint_state.json
```

Same runtime dir as the lock and stats:
`METAFIELD_RUNTIME_DIR` → `$XDG_RUNTIME_DIR/metafield` → `/tmp/metafield`.

---

## Enable

```bash
export METAFIELD_CONTROL_TOKEN="$(openssl rand -hex 16)"

# terminal A — produce evidence
python meta_field_distributed.py \
  --diagnostic --continuous --export-stats --summary-interval 30

# terminal B — mint when regulation says yes
python credit_mint.py --watch --interval 20
```

One-shot:

```bash
python credit_mint.py
```

Without the token:

```text
[mint] control surface disabled.
```

---

## Regulation (env-overridable)

| Env | Default | Meaning |
|-----|---------|--------|
| `METAFIELD_MINT_MIN_TRAJ_DELTA` | 10 | min new trajectories since last mint |
| `METAFIELD_MINT_MIN_ACCEPT` | 0.35 | acceptance floor |
| `METAFIELD_MINT_MAX_ABS_DH` | 3.0 | reject unhealthy integrator |
| `METAFIELD_MINT_COOLDOWN_SEC` | 60 | min seconds between mints |
| `METAFIELD_MINT_BASE_CREDIT` | 1.0 | base unit before quality multiplier |

Credit scales with acceptance quality and inverse `|ΔH|` when gates pass.

---

## Files

| Path | Role |
|------|------|
| `stats.json` | evidence (writer: continuous + `--export-stats`) |
| `work_claims.jsonl` | append-only sealed claims |
| `mint_state.json` | last traj, cooldown, total_credit |
| `continuous.lock` | continuity ownership (separate concern) |

Claims carry `evidence_hash` over a canonical subset of fields (integrity aid, not a blockchain).

---

## Principles checklist

| Principle | How |
|-----------|-----|
| Fail closed | no token → no mint |
| Python route | stats from live process only |
| Not Git | commits never call the mint |
| Self-regulated | traj delta, accept, \|dH\|, cooldown |
| Observable | JSONL + state file, mode 0600 when possible |
| Separated | token ≠ credit; credit ≠ external coin |

Aurora can later consume `work_claims.jsonl` under the same rules without MetaField ever speaking to a chain.
