/**
 * MetaField Visualizer — entry point
 * Presentation / observation layer over the MetaField systems.
 * First version uses deterministic demo data only.
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
  el.mode.className = 'value accent';

  if (obs.probe) {
    el.probe.textContent = `face ${obs.probe.face}`;
    el.wave.textContent = `${obs.probe.wavelength_nm} nm`;
  } else {
    el.probe.textContent = '—';
    el.wave.textContent = '—';
  }

  if (obs.prediction) {
    el.pred.textContent = `f${obs.prediction.face} = ${fmt(obs.prediction.value)}`;
  } else {
    el.pred.textContent = '—';
  }

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

  if (obs.error) {
    el.err.textContent = `f${obs.error.face} = ${fmt(obs.error.value)}`;
  } else {
    el.err.textContent = '—';
  }

  if (obs.nextProbe) {
    el.next.textContent = `face ${obs.nextProbe.face} (DEMO)`;
  } else {
    el.next.textContent = '—';
  }

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
    el.timeline.value = index;
    el.tlMin.textContent = '0';
    el.tlMax.textContent = String(Math.max(0, total - 1));
    el.tlCur.textContent = total ? `tick ${index} / ${total - 1}` : 'history';
  },
});

const sequence = generateDemoSequence(60, 42);
timeline.load(sequence);

el.timeline.addEventListener('input', () => {
  timeline.pause();
  el.btnDemo.classList.remove('active');
  timeline.setIndex(Number(el.timeline.value));
});

el.btnDemo.addEventListener('click', () => {
  const on = !timeline.isDemo();
  timeline.toggleDemo(on);
  el.btnDemo.classList.toggle('active', on);
  if (on) body.setAutoRotate(true);
});

el.btnPlay.addEventListener('click', () => {
  timeline.play();
  el.btnDemo.classList.remove('active');
});

el.btnPause.addEventListener('click', () => {
  timeline.pause();
  el.btnDemo.classList.remove('active');
});

el.btnReset.addEventListener('click', () => {
  timeline.reset();
  probes.clearProbe();
  el.btnDemo.classList.remove('active');
  body.setAutoRotate(true);
});

el.btnProbe.addEventListener('click', () => {
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

function onPointerDown(event) {
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
}
ctx.domElement.addEventListener('pointerdown', onPointerDown);

ctx.addUpdatable((dt, t) => {
  body.update(dt);
  field.update(dt, t);
  probes.update(dt);
  timeline.update(dt);
});

applyFrame(sequence[0]);

console.info(
  '[MetaField Visualizer] DEMO / HEURISTIC mode. Data from demo-data.js — not live MetaField predictor or optical hardware.'
);
