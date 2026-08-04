# Credit mint (internal swarm credits)

**Python route, not Git. Capability token, not a wallet.**

Mints *internal* regulation credits from observed MetaField work (`stats.json`),
only when `METAFIELD_CONTROL_TOKEN` is set.

---

## Flow

```
continuous.lock (live PID required)
        +
stats.json (fresh mtime required)
        +
METAFIELD_CONTROL_TOKEN
        │
CreditMint (flock + regulation + HMAC seal)
        │
work_claims.jsonl
```

---

## Enable

```bash
export METAFIELD_CONTROL_TOKEN="$(openssl rand -hex 16)"

python meta_field_distributed.py \
  --diagnostic --continuous --export-stats --summary-interval 30

python credit_mint.py --watch --interval 20
```

---

## Residual controls (final)

| Control | Default |
|---------|--------|
| Live continuous owner | required (`METAFIELD_MINT_REQUIRE_CONTINUOUS=1`) |
| Stats freshness | ≤ 120s (`METAFIELD_MINT_MAX_STATS_AGE`) |
| Runtime dir | reject world-writable; optional `METAFIELD_STRICT_RUNTIME=1` |
| Claim MAC | HMAC-SHA256(token, evidence_hash) |
| Mint flock | exclusive `mint.lock` |
| Metric clamps | finite only, accept∈[0,1], abs(\|dH\|) |
| Credit caps | per-claim + total + claims file size |
| Env bombs | all regulation env vars clamped |
| Token surface | **env only** (no `--token` argv) |

---

## Regulation env

| Env | Default |
|-----|--------|
| `METAFIELD_MINT_MIN_TRAJ_DELTA` | 10 |
| `METAFIELD_MINT_MIN_ACCEPT` | 0.35 |
| `METAFIELD_MINT_MAX_ABS_DH` | 3.0 |
| `METAFIELD_MINT_COOLDOWN_SEC` | 60 |
| `METAFIELD_MINT_BASE_CREDIT` | 1.0 |
| `METAFIELD_MINT_MAX_CREDIT` | 10 |
| `METAFIELD_MINT_MAX_TOTAL` | 1e6 |
| `METAFIELD_MINT_MAX_STATS_AGE` | 120 |

---

## Still residual (honest)

1. **Same-uid attacker** who can write `stats.json` *and* hold the control token can still shape evidence. Token secrecy + private runtime dir are the boundary.
2. **No Redis** on this path by design. Future Aurora publish must re-validate MAC + caps — never mint from raw Redis blobs.
3. **PID reuse** edge case on continuous.lock after very long uptime; freshness + live flag reduce impact.

---

## Not this mint

- Git commits as proof of work
- Chain assets / external currency
- Unauthenticated network publish
