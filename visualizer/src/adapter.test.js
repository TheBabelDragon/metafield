/**
 * Smoke tests for the observation adapter.
 * Run with: npm test  (node --test)
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeObservation, validateObservation } from './adapter.js';
import { generateDemoSequence } from './demo-data.js';

describe('adapter', () => {
  it('normalizes a minimal observation', () => {
    const raw = {
      tick: 3,
      observations: [{ face: 0, value: 0.5, confidence: 0.9, anomaly: 0.1 }],
    };
    const n = normalizeObservation(raw);
    assert.equal(n.tick, 3);
    assert.equal(n.observations.length, 1);
    assert.equal(n.observations[0].face, 0);
    assert.equal(n.probe, null);
    assert.equal(n.prediction, null);
  });

  it('validates demo sequence entries', () => {
    const seq = generateDemoSequence(12, 7);
    assert.ok(seq.length === 12);
    for (const raw of seq) {
      const n = normalizeObservation(raw);
      const v = validateObservation(n);
      assert.equal(v.ok, true, v.errors.join('; '));
      assert.ok(n.observations.length === 12);
    }
  });

  it('rejects non-object', () => {
    assert.throws(() => normalizeObservation(null), TypeError);
  });
});
