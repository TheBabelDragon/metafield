/**
 * MetaField Visualizer — entry point
 * DEMO / HEURISTIC only — not live MetaField or hardware.
 */
import * as THREE from 'three';
import { createScene } from './scene.js';
import { createFieldBody } from './body.js';
import { createFieldLayer } from './field.js';
import { createProbeSystem } from './probes.js';
import { createTimeline } from './timeline.js';
import { generateDemoSequence } from './demo-data.js';
import { normalizeObservation } from './adapter.js';

const container = document.getElementById('canvas-container');
if (!container) {
  throw new Error('Missing #canvas-container');
}

const ctx = createScene(container);
const body = createFieldBody(ctx.scene);
const field = createFieldLayer(ctx.scene, body);
const probes = createProbeSystem(ctx.scene, body);

const el = {
  tick: document.getElementById('v-tick'),
  mode: document.getElementById('v-mode'),
  probe: document.getElementById('v-probe'),
  wave: document.getElementById('v-wave'),
  pred: document.getElementById('v-pred'),
  obs: document.getElementById('v-obs'),
  err: document.getElementById('v-err'),
  conf: document.getElementById('v-conf'),
  next: document.getElementById('v-next'),
  source: document.getElementById('v-source'),
  timeline: document.getElementById('timeline'),
  tlMin: document.getElementById('tl-min'),
  tlCur: document.getElementById('tl-cur'),
  tlMax: document.getElementById('tl-max'),
  btnDemo: document.getElementById('btn-demo'),
  btnPlay: document.getElementById('btn-play'),
  btnPause: document.getElementById('btn-pause'),
  btnProbe: document.getElementById('btn-probe'),
  btnReset: document.getElementById('btn-reset'),
};

function fmt(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return '—';
  return Number(n).toFixed(digits);
}

function updateHUD(obs) {
  if (!obs) {
    el.tick.textContent = '—';
    el.mode.textContent = 'IDLE';
    el.probe.textContent = '—';
    el.wave.textContent = '—';
    el.pred.textContent = '—';
    el.obs.textContent = '—';
    el.err.textContent = '—';
    el.conf.textContent = '—';
    el.next.textContent = '—';
    return;
  }

  el.tick.textContent = String(obs.tick);
  el.mode.textContent = (obs.phase || 'idle').toUpperCase();

  if (obs.probe) {
    el.probe.textContent = `face ${obs.probe.face}`;
    el.wave.textContent = `${obs.probe.wavelength_nm} nm`;
  } else {
    el.probe.textContent = '—';
    el.wave.textContent = '—';
  }

  el.pred.textContent = obs.prediction
    ? `f${obs.prediction.face} = ${fmt(obs.prediction.value)}`
    : '—';

  const activeObs = obs.probe
    ? obs.observations.find((o) => o.face === obs.probe.face)
    : null;
  if (activeObs) {
    el.obs.textContent = `f${activeObs.face} = ${fmt(activeObs.value)}`;
    el.conf.textContent = fmt(activeObs.confidence);
  } else {
    el.obs.textContent = '—';
    el.conf.textContent = '—';
  }

  el.err.textContent = obs.error
    ? `f${obs.error.face} = ${fmt(obs.error.value)}`
    : '—';

  el.next.textContent = obs.nextProbe
    ? `face ${obs.nextProbe.face} (DEMO)`
    : '—';

  el.source.textContent = obs._demo ? 'DEMO / HEURISTIC' : 'LIVE';
}

function applyFrame(raw) {
  if (!raw) {
    updateHUD(null);
    return;
  }
  const obs = normalizeObservation(raw);
  body.applyObservation(obs);
  field.applyObservation(obs);
  probes.applyObservation(obs);
  updateHUD(obs);
  body.setAutoRotate(!(obs.phase === 'probe' || obs.phase === 'observe'));
}

const timeline = createTimeline({
  onFrame(obs) {
    applyFrame(obs);
  },
  onIndexChange(index, total) {
    el.timeline.max = Math.max(0, total - 1);
    el.timeline.value = String(index);
    el.tlMin.textContent = '0';
    el.tlMax.textContent = String(Math.max(0, total - 1));
    el.tlCur.textContent = total ? `tick ${index} / ${total - 1}` : 'history';
  },
});

const sequence = generateDemoSequence(60, 42);
timeline.load(sequence);

function on(node, event, fn) {
  if (node) node.addEventListener(event, fn);
}

on(el.timeline, 'input', () => {
  timeline.pause();
  el.btnDemo?.classList.remove('active');
  timeline.setIndex(Number(el.timeline.value));
});

on(el.btnDemo, 'click', () => {
  const onDemo = !timeline.isDemo();
  timeline.toggleDemo(onDemo);
  el.btnDemo.classList.toggle('active', onDemo);
  if (onDemo) body.setAutoRotate(true);
});

on(el.btnPlay, 'click', () => {
  timeline.play();
  el.btnDemo?.classList.remove('active');
});

on(el.btnPause, 'click', () => {
  timeline.pause();
  el.btnDemo?.classList.remove('active');
});

on(el.btnReset, 'click', () => {
  timeline.reset();
  probes.clearProbe();
  el.btnDemo?.classList.remove('active');
  body.setAutoRotate(true);
});

on(el.btnProbe, 'click', () => {
  const face = body.state.selectedFace ?? 0;
  probes.triggerProbe(face, 1.0, 850);
  body.setSelectedFace(face);
  const snap = sequence[timeline.getIndex()] || sequence[0];
  if (snap) {
    const obs = normalizeObservation({
      ...snap,
      phase: 'probe',
      probe: { face, wavelength_nm: 850, intensity: 1.0 },
    });
    body.applyObservation(obs);
    field.applyObservation(obs);
    updateHUD(obs);
  }
});

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

ctx.domElement.addEventListener('pointerdown', (event) => {
  const rect = ctx.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, ctx.camera);
  const hits = raycaster.intersectObjects(body.getPickables(), false);
  if (hits.length > 0) {
    const faceId = hits[0].object.userData.faceId;
    if (faceId != null) {
      body.setSelectedFace(faceId);
      const raw = sequence[timeline.getIndex()];
      if (raw) applyFrame(raw);
    }
  }
});

ctx.addUpdatable((dt, t) => {
  body.update(dt);
  field.update(dt, t);
  probes.update(dt);
  timeline.update(dt);
});

applyFrame(sequence[0]);
timeline.toggleDemo(true);
el.btnDemo?.classList.add('active');

console.info(
  '[MetaField Visualizer] DEMO / HEURISTIC mode — not live MetaField or optical hardware.'
);
