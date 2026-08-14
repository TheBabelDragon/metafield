# sonify

MetaField → AXIOM bridge via plain JSONL.

**Decouples lattice QCD from MIDI.** The sim never imports musical code; AXIOM never imports lattice code.

## Status (2026-08-14)

Package is complete on `main`. If `meta_field_sim_torch.py` was clobbered during integration, restore it first:

```bash
git checkout c46d1f4096eaa7be687da1c24873666b47c95102 -- meta_field_sim_torch.py
patch -p1 < sonify/instrument_sim.patch
```

See issue #2 for details.

## Contract

After each HMC trajectory the sim appends one line to a `.jsonl` file:

```json
{
  "step": 0,
  "traj_id": 0,
  "accepted": true,
  "plaquette": 0.72,
  "topological_charge": 0.01,
  "fisher_curvature": 1.2,
  "color_fields": [0.9, 0.88, 0.91],
  "dirac_eigmin": 0.4
}
```

Enable emission with:

```python
sim = MetaFieldSimulationV2(..., observables_jsonl="/tmp/metafield/observables.jsonl")
sim.run()
```

## Offline export

```bash
python -m sonify.export /tmp/metafield/observables.jsonl -o axiom_notes.json --ticks-per-step 8
```

## Mapping (edit `mapping.py`)

| Observable | Sound |
|---|---|
| `accepted` | kick (36) / snare (38) |
| `plaquette` | bass (D-min pentatonic) |
| `color_fields[0..2]` | three melodic voices |
| `fisher_curvature` | note velocity |
| `topological_charge` spike | crash (49) + accent |

## AXIOM side (still local / private)

1. Add track `Track('physics', 'Metafield', 'strings')` + `TCOLORS['physics']`
2. `_import_metafield_run(path)` reading `axiom_notes.json`
3. "Import Run" button → `dialogs.pick_document()`

## Stretch — live sonification

Sim writes JSON lines to a TCP socket; `PhysicsBridge` thread in `DAWView` calls `trk.add_note()` in real time.
