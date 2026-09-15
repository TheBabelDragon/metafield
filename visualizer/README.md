# MetaField Visualizer

Interactive Three.js presentation and observation layer for the MetaField optical-body / field systems.

This is **not** a replacement for the Python MetaField engine, `FieldMemoryStore`, `LatentPredictor`, `active_probe`, or Aurora integration. It is a visual instrument that consumes a small, stable observation contract.

## Status

- **v0.1** — isolated Vite + Three.js app under `visualizer/`
- Uses **deterministic synthetic demo data** only (`DEMO / HEURISTIC`)
- Does **not** claim live hardware, live predictor, or live MetaField coupling
- Architected so `demo-data.js` can later be swapped for a real adapter

## Quick start

```bash
cd visualizer
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`).

Production build:

```bash
npm run build
npm run preview   # optional local static serve of dist/
```

Tests (adapter smoke):

```bash
npm test
```

No Python and no optical hardware are required.

## Architecture

```
visualizer/
├── index.html
├── package.json
├── vite.config.js
├── README.md
└── src/
    ├── main.js          # app init, HUD, interaction
    ├── scene.js         # Three.js scene / camera / renderer / loop
    ├── body.js          # dodecahedral body, face IDs, face state
    ├── field.js         # field shell, pred/obs markers, residual haze
    ├── probes.js        # active probe beam + excitation animation
    ├── timeline.js      # history, play/pause, scrub, demo loop
    ├── demo-data.js     # deterministic synthetic observation stream
    ├── adapter.js       # normalize + validate observation contract
    └── adapter.test.js  # smoke tests for adapter
```

### Observation contract (renderer-facing)

```js
{
  tick: 0,
  timestamp: "2026-01-01T00:00:00.000Z",
  probe: {
    face: 0,
    wavelength_nm: 850,
    intensity: 1.0
  },
  observations: [
    {
      face: 0,
      value: 0.72,
      confidence: 0.91,
      anomaly: 0.04
    }
    // … one entry per face (0..11)
  ],
  prediction: {
    face: 0,
    value: 0.68
  },
  error: {
    face: 0,
    value: 0.04
  },
  nextProbe: {
    face: 3
  },
  phase: "compare",   // idle | select | probe | observe | predict | compare
  _demo: true
}
```

The renderer only consumes this shape via `adapter.normalizeObservation()`. It does not know whether data came from `demo-data.js` or a future live source.

## Demo loop (≈30 s readable cycle)

Press **DEMO**. The instrument cycles:

1. **Select** next probe (labeled DEMO / HEURISTIC)
2. **Probe** — excitation beam travels to the face
3. **Observe** — face field updates, observation marker appears
4. **Predict** — predicted state glyph appears
5. **Compare** — residual / anomaly becomes visible
6. **Select** next probe and repeat

Controls:

| Control | Action |
|--------|--------|
| DEMO | Auto-cycle the full probe → observe → predict → compare loop |
| PLAY / PAUSE | Scrub-playback of recorded history |
| Timeline slider | Jump to any tick |
| PROBE SELECTED | Fire a manual excitation at the selected face |
| RESET | Return to tick 0 |
| Drag / scroll | Orbit and zoom |
| Click face | Select face |

## Connecting real MetaField data later

Intended path (not implemented in this PR):

```
MetaField core
  → FieldObservation / FieldMemoryStore
  → visualizer adapter (replace or extend demo-data.js)
  → Three.js renderer
```

Future streaming path:

```
MetaField
  → WebSocket / HTTP / JSONL
  → visualizer adapter
  → Three.js
```

Implementation notes for the next PR:

1. Keep the observation contract stable.
2. Replace `generateDemoSequence()` with a source that pushes normalized observations into `timeline.load()` or a live buffer.
3. Map optical-body face IDs 1:1 (already 0..11).
4. Do not move prediction logic into the visualizer; display what MetaField / `LatentPredictor` / `active_probe` already compute.
5. Label the HUD source as `LIVE` when `_demo` is false.

## Visual design

Dark scientific instrument aesthetic. Geometry carries state:

- Face color / emissive → field magnitude, selection, anomaly
- Purple diamond → prediction
- Green diamond → observation
- Beam + impact ring → active probe
- Particle haze → residual / anomaly energy

UI overlays are minimal and numeric only where useful.

## Definition of done (v0.1)

- [x] `npm install` / `npm run dev` / `npm run build` succeed
- [x] No Python or hardware dependency
- [x] Deterministic demo data
- [x] Adapter smoke test
- [x] Interactable dodecahedral body with stable face IDs
- [x] Visible probe → observe → predict → compare → next-probe loop
- [x] Timeline scrub + demo autoplay
- [x] Explicit DEMO / HEURISTIC labeling (no false live claims)

## License

Same as the parent MetaField repository.
