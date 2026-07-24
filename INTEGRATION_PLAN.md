# MetaField + Aurora Integration Plan

**Updated for v1.46**

---

## Current status (v1.46)

### Done
- Soft expandable episodic memory + force-based attractors + homeostasis + adaptive basins
- Continuous singleton lock (duplicate continuous prohibited)
- Control surface fail-closed (`METAFIELD_CONTROL_TOKEN`)
- Local stats export (`--export-stats`) — file only, no Redis publish
- **Aurora environment feed (read-only)**
  - Start prompt from live sensing context
  - Drive force → exploration scale, energy budget scale, interest gate bias
  - Degrades gracefully if Redis unavailable

### Not yet (still gated)
- Redis *publish* from MetaField into Aurora channels
- Overlord remote commands (requires control token + explicit design)
- Scheduler task wrapper for HMC continuous runs

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

## Next integration steps (when security allows)

1. Authenticated publish of MetaField stats onto `aurora:sensing:*` (token required)
2. Aurora mod registration for live dashboard
3. Scheduler influence (prefer nodes when MetaField interest is high)
4. Overlord start/stop of continuous runs under singleton lock

---

*Living document.*
