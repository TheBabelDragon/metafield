/**
 * Deterministic synthetic observation stream for the MetaField visualizer.
 * This is DEMO / HEURISTIC data only — not live MetaField or hardware.
 *
 * Contract shape (renderer-facing):
 * {
 *   tick: number,
 *   timestamp: string,
 *   probe: { face: number, wavelength_nm: number, intensity: number } | null,
 *   observations: [{ face: number, value: number, confidence: number, anomaly: number }],
 *   prediction: { face: number, value: number } | null,
 *   error: { face: number, value: number } | null,
 *   nextProbe: { face: number } | null,
 *   phase: 'idle' | 'probe' | 'observe' | 'predict' | 'compare' | 'select'
 * }
 */

const NUM_FACES = 12;
const WAVELENGTHS = [850, 780, 650, 940];

/** Simple seeded PRNG (mulberry32) for determinism */
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Generate a full deterministic demo sequence of observations.
 * @param {number} length - number of ticks
 * @param {number} [seed=42]
 * @returns {Array<object>}
 */
export function generateDemoSequence(length = 48, seed = 42) {
  const rng = mulberry32(seed);
  const sequence = [];
  const field = new Float32Array(NUM_FACES);
  for (let i = 0; i < NUM_FACES; i++) field[i] = 0.15 + rng() * 0.1;

  let probeFace = Math.floor(rng() * NUM_FACES);
  let nextFace = (probeFace + 3) % NUM_FACES;

  for (let tick = 0; tick < length; tick++) {
    const phaseIndex = tick % 6;
    const phases = ['select', 'probe', 'observe', 'predict', 'compare', 'select'];
    const phase = phases[phaseIndex];

    // Advance probe on select steps
    if (phase === 'select' && tick > 0) {
      probeFace = nextFace;
      // Deterministic heuristic: prefer faces with higher recent anomaly / lower confidence
      // Here we just walk a fixed pattern for clarity.
      nextFace = (probeFace + 1 + Math.floor(rng() * 3)) % NUM_FACES;
    }

    const intensity = 0.7 + rng() * 0.3;
    const wavelength = WAVELENGTHS[probeFace % WAVELENGTHS.length];

    // Inject excitation into field on probe/observe
    if (phase === 'probe' || phase === 'observe') {
      field[probeFace] = clamp(field[probeFace] + intensity * 0.35, 0, 1.2);
      // Soft neighbor bleed
      const left = (probeFace + NUM_FACES - 1) % NUM_FACES;
      const right = (probeFace + 1) % NUM_FACES;
      field[left] = clamp(field[left] + intensity * 0.08, 0, 1.2);
      field[right] = clamp(field[right] + intensity * 0.08, 0, 1.2);
    }

    // Mild global decay
    for (let i = 0; i < NUM_FACES; i++) {
      field[i] = clamp(field[i] * 0.97 + 0.01, 0, 1.2);
    }

    const observedValue = clamp(field[probeFace] + (rng() - 0.5) * 0.06, 0, 1.2);
    // Prediction slightly lags / biased
    const predictedValue = clamp(
      field[probeFace] * 0.92 + 0.05 + (rng() - 0.5) * 0.04,
      0,
      1.2
    );
    const err = Math.abs(observedValue - predictedValue);
    const confidence = clamp(1.0 - err * 2.2 - rng() * 0.05, 0.35, 0.98);
    const anomaly = clamp(err * 1.4 + (rng() > 0.85 ? 0.15 : 0), 0, 1);

    const observations = [];
    for (let f = 0; f < NUM_FACES; f++) {
      const isActive = f === probeFace;
      observations.push({
        face: f,
        value: isActive ? observedValue : clamp(field[f] + (rng() - 0.5) * 0.02, 0, 1.2),
        confidence: isActive ? confidence : clamp(0.6 + rng() * 0.25, 0.4, 0.95),
        anomaly: isActive ? anomaly : clamp(rng() * 0.08, 0, 0.3),
      });
    }

    sequence.push({
      tick,
      timestamp: new Date(Date.UTC(2026, 0, 1, 0, 0, tick)).toISOString(),
      probe:
        phase === 'probe' || phase === 'observe' || phase === 'predict' || phase === 'compare'
          ? { face: probeFace, wavelength_nm: wavelength, intensity }
          : null,
      observations,
      prediction:
        phase === 'predict' || phase === 'compare'
          ? { face: probeFace, value: predictedValue }
          : null,
      error:
        phase === 'compare'
          ? { face: probeFace, value: err }
          : null,
      nextProbe: { face: nextFace },
      phase,
      _demo: true,
    });
  }

  return sequence;
}

export { NUM_FACES };
