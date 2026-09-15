import * as THREE from 'three';
import { NUM_FACES } from './demo-data.js';

/**
 * Field visualization layer: ambient field shell, confidence haze,
 * predicted vs observed residual glyphs.
 */
export function createFieldLayer(scene, body) {
  const group = new THREE.Group();
  group.name = 'fieldLayer';
  scene.add(group);

  // Soft outer field shell
  const shellGeo = new THREE.IcosahedronGeometry(2.1, 2);
  const shellMat = new THREE.MeshBasicMaterial({
    color: 0x0c1a30,
    transparent: true,
    opacity: 0.12,
    side: THREE.BackSide,
    depthWrite: false,
  });
  const shell = new THREE.Mesh(shellGeo, shellMat);
  group.add(shell);

  // Prediction markers (small diamonds floating outside faces)
  const predMarkers = [];
  const obsMarkers = [];

  const diamondGeo = new THREE.OctahedronGeometry(0.07, 0);
  for (let i = 0; i < NUM_FACES; i++) {
    const predMat = new THREE.MeshBasicMaterial({
      color: 0xa78bfa,
      transparent: true,
      opacity: 0,
    });
    const pred = new THREE.Mesh(diamondGeo, predMat);
    pred.visible = false;
    group.add(pred);
    predMarkers.push(pred);

    const obsMat = new THREE.MeshBasicMaterial({
      color: 0x3dd68c,
      transparent: true,
      opacity: 0,
    });
    const obs = new THREE.Mesh(diamondGeo.clone(), obsMat);
    obs.visible = false;
    group.add(obs);
    obsMarkers.push(obs);
  }

  // Particle-ish residual field (simple points)
  const particleCount = 120;
  const pGeo = new THREE.BufferGeometry();
  const pPos = new Float32Array(particleCount * 3);
  const pCol = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    const r = 1.6 + Math.random() * 0.7;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    pPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    pPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    pPos[i * 3 + 2] = r * Math.cos(phi);
    pCol[i * 3] = 0.15;
    pCol[i * 3 + 1] = 0.35;
    pCol[i * 3 + 2] = 0.7;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
  const pMat = new THREE.PointsMaterial({
    size: 0.03,
    vertexColors: true,
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
  });
  const points = new THREE.Points(pGeo, pMat);
  group.add(points);

  function applyObservation(obs) {
    // Reset markers
    for (let i = 0; i < NUM_FACES; i++) {
      predMarkers[i].visible = false;
      predMarkers[i].material.opacity = 0;
      obsMarkers[i].visible = false;
      obsMarkers[i].material.opacity = 0;
    }

    if (!obs) return;

    if (obs.prediction) {
      const f = obs.prediction.face;
      const center = body.faceCenters[f];
      const normal = body.faceNormals[f];
      if (center && normal) {
        const m = predMarkers[f];
        m.position.copy(center).addScaledVector(normal, 0.28 + obs.prediction.value * 0.15);
        m.visible = true;
        m.material.opacity = 0.9;
        m.scale.setScalar(0.8 + obs.prediction.value);
      }
    }

    if (obs.probe && (obs.phase === 'observe' || obs.phase === 'compare')) {
      const f = obs.probe.face;
      const o = obs.observations.find((x) => x.face === f);
      const center = body.faceCenters[f];
      const normal = body.faceNormals[f];
      if (center && normal && o) {
        const m = obsMarkers[f];
        m.position.copy(center).addScaledVector(normal, 0.22 + o.value * 0.12);
        m.visible = true;
        m.material.opacity = 0.95;
        m.scale.setScalar(0.7 + o.value);
      }
    }

    // Modulate particle colors by global anomaly energy
    let anomalyEnergy = 0;
    if (obs.observations) {
      for (const o of obs.observations) anomalyEnergy += o.anomaly || 0;
    }
    anomalyEnergy = Math.min(1, anomalyEnergy / 3);
    const cols = points.geometry.attributes.color;
    for (let i = 0; i < particleCount; i++) {
      cols.setXYZ(
        i,
        0.12 + anomalyEnergy * 0.7,
        0.3 - anomalyEnergy * 0.15,
        0.65 - anomalyEnergy * 0.3
      );
    }
    cols.needsUpdate = true;
    shell.material.opacity = 0.1 + anomalyEnergy * 0.12;
  }

  function update(dt, t) {
    points.rotation.y += dt * 0.04;
    shell.rotation.y -= dt * 0.02;
    // Gentle pulse on prediction markers
    for (const m of predMarkers) {
      if (m.visible) {
        m.rotation.y += dt * 2;
        m.rotation.x += dt * 1.2;
      }
    }
  }

  return {
    group,
    applyObservation,
    update,
  };
}
