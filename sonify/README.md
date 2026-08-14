# sonify

MetaField → AXIOM bridge via plain JSONL.

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

## Offline export

```bash
python -m sonify.export run.jsonl -o axiom_notes.json --ticks-per-step 8
```

## Mapping (edit `mapping.py`)

| Observable | Sound |
|---|---|
| `accepted` | kick (36) / snare (38) |
| `plaquette` | bass (D-min pentatonic) |
| `color_fields[0..2]` | three melodic voices |
| `fisher_curvature` | note velocity |
| `topological_charge` spike | crash (49) + accent |

AXIOM imports `axiom_notes.json` via `_import_metafield_run`.
