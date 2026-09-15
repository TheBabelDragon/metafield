/**
 * Observation adapter boundary.
 *
 * Isolates the renderer from the data source.
 * Today: demo-data.js
 * Tomorrow: MetaField → FieldObservation / FieldMemoryStore → this adapter
 * Future:   WebSocket / HTTP / JSONL stream → this adapter
 */

/**
 * Normalize any incoming observation-like object into the renderer contract.
 * @param {object} raw
 * @returns {object} normalized observation
 */
export function normalizeObservation(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new TypeError('Observation must be an object');
  }

  const tick = Number(raw.tick ?? 0);
  const timestamp = String(raw.timestamp ?? new Date().toISOString());

  const probe = raw.probe
    ? {
        face: Number(raw.probe.face),
        wavelength_nm: Number(raw.probe.wavelength_nm ?? 850),
        intensity: Number(raw.probe.intensity ?? 1),
      }
    : null;

  const observations = Array.isArray(raw.observations)
    ? raw.observations.map((o) => ({
        face: Number(o.face),
        value: Number(o.value ?? 0),
        confidence: Number(o.confidence ?? 0.5),
        anomaly: Number(o.anomaly ?? 0),
      }))
    : [];

  const prediction = raw.prediction
    ? {
        face: Number(raw.prediction.face),
        value: Number(raw.prediction.value ?? 0),
      }
    : null;

  const error = raw.error
    ? {
        face: Number(raw.error.face),
        value: Number(raw.error.value ?? 0),
      }
    : null;

  const nextProbe = raw.nextProbe
    ? { face: Number(raw.nextProbe.face) }
    : null;

  const phase = String(raw.phase ?? 'idle');

  return {
    tick,
    timestamp,
    probe,
    observations,
    prediction,
    error,
    nextProbe,
    phase,
    _demo: Boolean(raw._demo),
  };
}

/**
 * Validate a normalized observation (smoke-level).
 * @param {object} obs
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validateObservation(obs) {
  const errors = [];
  if (typeof obs.tick !== 'number' || Number.isNaN(obs.tick)) {
    errors.push('tick must be a number');
  }
  if (!Array.isArray(obs.observations)) {
    errors.push('observations must be an array');
  } else {
    for (const o of obs.observations) {
      if (typeof o.face !== 'number') errors.push('observation.face must be number');
      if (typeof o.value !== 'number') errors.push('observation.value must be number');
    }
  }
  return { ok: errors.length === 0, errors };
}
