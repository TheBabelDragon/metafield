import * as THREE from 'three';
import { NUM_FACES } from './demo-data.js';

/**
 * Dodecahedral field body with 12 addressable faces (IDs 0..11).
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const INV_PHI = 1 / PHI;

function dodecahedronFaceNormals() {
  const normals = [];
  const bases = [
    [0, INV_PHI, PHI],
    [0, -INV_PHI, PHI],
    [0, INV_PHI, -PHI],
    [0, -INV_PHI, -PHI],
    [PHI, 0, INV_PHI],
    [-PHI, 0, INV_PHI],
    [PHI, 0, -INV_PHI],
    [-PHI, 0, -INV_PHI],
    [INV_PHI, PHI, 0],
    [-INV_PHI, PHI, 0],
    [INV_PHI, -PHI, 0],
    [-INV_PHI, -PHI, 0],
  ];
  for (const b of bases) {
    normals.push(new THREE.Vector3(b[0], b[1], b[2]).normalize());
  }
  return normals;
}

function makePentagonGeometry(radius) {
  const verts = [];
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2 - Math.PI / 2;
    verts.push(new THREE.Vector3(Math.cos(a) * radius, Math.sin(a) * radius, 0));
  }
  const positions = [0, 0, 0];
  for (const v of verts) positions.push(v.x, v.y, v.z);
  const indices = [];
  for (let i = 0; i < 5; i++) {
    indices.push(0, i + 1, ((i + 1) % 5) + 1);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

export function createFieldBody(scene) {
  const group = new THREE.Group();
  group.name = 'fieldBody';
  const radius = 1.35;

  const shellGeo = new THREE.DodecahedronGeometry(radius, 0);
  const shell = new THREE.Mesh(
    shellGeo,
    new THREE.MeshBasicMaterial({
      color: 0x3a6aaa,
      wireframe: true,
      transparent: true,
      opacity: 0.7,
    })
  );
  group.add(shell);

  const core = new THREE.Mesh(
    shellGeo.clone(),
    new THREE.MeshStandardMaterial({
      color: 0x0c1a30,
      metalness: 0.3,
      roughness: 0.55,
      transparent: true,
      opacity: 0.45,
      side: THREE.DoubleSide,
    })
  );
  group.add(core);

  const normals = dodecahedronFaceNormals();
  const faceMeshes = [];
  const faceCenters = [];
  const faceNormals = [];
  const panelRadius = 0.55;

  for (let faceId = 0; faceId < NUM_FACES; faceId++) {
    const normal = normals[faceId];
    const center = normal.clone().multiplyScalar(radius * 0.92);

    const geo = makePentagonGeometry(panelRadius);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x1a3a5c),
      emissive: new THREE.Color(0x102030),
      emissiveIntensity: 0.35,
      metalness: 0.15,
      roughness: 0.4,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(center);
    mesh.lookAt(center.clone().add(normal));
    mesh.userData.faceId = faceId;
    mesh.userData.baseColor = new THREE.Color(0x1a3a5c);
    mesh.name = `face_${faceId}`;
    group.add(mesh);

    faceMeshes.push(mesh);
    faceCenters.push(center.clone());
    faceNormals.push(normal.clone());

    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.04, 10, 10),
      new THREE.MeshBasicMaterial({ color: 0x5ab0ff })
    );
    marker.position.copy(center).addScaledVector(normal, 0.08);
    group.add(marker);
  }

  scene.add(group);

  let autoRotate = true;
  const state = {
    selectedFace: null,
    values: new Float32Array(NUM_FACES),
    confidence: new Float32Array(NUM_FACES).fill(0.5),
    anomaly: new Float32Array(NUM_FACES),
  };

  function setFaceVisual(faceId, { value, confidence, anomaly, highlight, pred, observed }) {
    const mesh = faceMeshes[faceId];
    if (!mesh) return;

    const v = value ?? state.values[faceId] ?? 0;
    const a = anomaly ?? state.anomaly[faceId] ?? 0;
    const conf = confidence ?? state.confidence[faceId] ?? 0.5;

    const color = new THREE.Color(0x0a1525).lerp(new THREE.Color(0x2a90ff), THREE.MathUtils.clamp(v, 0, 1));
    if (a > 0.05) {
      color.lerp(new THREE.Color(0xff5533), THREE.MathUtils.clamp(a * 1.2, 0, 1));
    }
    mesh.material.color.copy(color);

    let emColor = 0x102030;
    let emIntensity = 0.25 + v * 0.4;
    if (highlight) {
      emColor = 0x3aa0ff;
      emIntensity = 0.85;
    }
    if (pred) {
      emColor = 0x9b6bff;
      emIntensity = 0.9;
    }
    if (observed) {
      emColor = 0x2ecc7a;
      emIntensity = 0.9;
    }
    if (a > 0.25) {
      emColor = 0xff4422;
      emIntensity = 1.0;
    }
    mesh.material.emissive = new THREE.Color(emColor);
    mesh.material.emissiveIntensity = emIntensity;
    mesh.material.opacity = 0.7 + conf * 0.28;
    mesh.scale.setScalar(1 + v * 0.12);
  }

  function applyObservation(obs) {
    if (!obs || !obs.observations) return;
    for (const o of obs.observations) {
      if (o.face < 0 || o.face >= NUM_FACES) continue;
      state.values[o.face] = o.value;
      state.confidence[o.face] = o.confidence;
      state.anomaly[o.face] = o.anomaly;
    }

    for (let f = 0; f < NUM_FACES; f++) {
      const isProbe = obs.probe && obs.probe.face === f;
      const isPred = obs.prediction && obs.prediction.face === f;
      const isErr = obs.error && obs.error.face === f;
      const isSelected = state.selectedFace === f;
      setFaceVisual(f, {
        value: state.values[f],
        confidence: state.confidence[f],
        anomaly: isErr ? Math.max(state.anomaly[f], obs.error.value) : state.anomaly[f],
        highlight: isProbe || isSelected,
        pred: isPred && (obs.phase === 'predict' || obs.phase === 'compare'),
        observed: isProbe && (obs.phase === 'observe' || obs.phase === 'compare'),
      });
    }
  }

  function setSelectedFace(id) {
    state.selectedFace = id;
  }

  function getPickables() {
    return faceMeshes.filter(Boolean);
  }

  function update(dt) {
    if (autoRotate) group.rotation.y += dt * 0.12;
  }

  return {
    group,
    faceMeshes,
    faceCenters,
    faceNormals,
    state,
    applyObservation,
    setSelectedFace,
    getFaceWorldCenter(faceId) {
      const local = faceCenters[faceId];
      if (!local) return new THREE.Vector3();
      return local.clone().applyMatrix4(group.matrixWorld);
    },
    getFaceNormal(faceId) {
      return faceNormals[faceId]?.clone() ?? new THREE.Vector3(0, 1, 0);
    },
    getPickables,
    update,
    setAutoRotate(v) {
      autoRotate = v;
    },
  };
}
