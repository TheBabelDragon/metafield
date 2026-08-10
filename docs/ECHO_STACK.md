# Echo ↔ MetaField stack (closed loop)

## Pipeline

```
CSI / φ  →  FieldObservation JSONL
         →  motion head (echo_head.pt)
         →  residual |r|
         →  AND-gates (automata)
         →  events JSONL
         →  Echo HUD + φ boost
         →  surprise → FieldMemoryStore
```

## One-time setup

```bash
# train head (once you have echo log)
cd ~/metafield-work/metafield && source .venv/bin/activate
python examples/echo_field_predictor.py \
  --file /tmp/metafield/echo.jsonl \
  --save-model /tmp/metafield/echo_head.pt \
  --score /tmp/metafield/echo_residuals.jsonl
```

Echo side needs torch once to load the `.pt`:

```bash
cd ~/echo-grid-ultrasonic-os && source .venv/bin/activate
pip install torch -q
```

## Live (3 terminals)

**Terminal 1 — Echo**
```bash
cd ~/echo-grid-ultrasonic-os && source .venv/bin/activate
python visualization/dashboard.py --csi --metafield-log --field-head --automata-events
```

**Terminal 2 — Automata gates**
```bash
cd ~/metafield-work/metafield && source .venv/bin/activate
python examples/echo_automata.py --follow --threshold 0.30
```

**Terminal 3 — Continuous surprise memory (optional)**
```bash
cd ~/metafield-work/metafield && source .venv/bin/activate
python examples/echo_continuous.py --threshold 0.30
```

## Artifacts under `/tmp/metafield/`

| File | Role |
|------|------|
| `echo.jsonl` | live FieldObservation stream |
| `echo_head.pt` | trained motion head |
| `echo_residuals.jsonl` | offline residual score |
| `echo_events.jsonl` | automata gate events |
| `echo_surprise_memory.jsonl` | high-\|r\| memories |
| `echo_store.jsonl` | FieldMemoryStore dump |
| `field_memory.jsonl` | bulk consumer log |

## Gates (v0)

| Gate | Rule |
|------|------|
| `SURPRISE_CONFIRMED` | \|r\| ≥ T AND fuse agreed |
| `TRACKED_SURPRISE` | \|r\| ≥ T AND tracks ≥ 1 |
| `HIGH_MOTION_SURPRISE` | \|r\| ≥ T AND motion ≥ 0.6 |
| `QUIET_ANOMALY` | \|r\| ≥ T AND motion < 0.25 |

Default threshold `T = 0.30` (near residual p90 from your sessions).

## Offline close-loop check

```bash
python examples/echo_close_loop.py
python examples/echo_automata.py --no-follow --threshold 0.30
```
