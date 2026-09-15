import * as THREE from 'three';
import { NUM_FACES } from './demo-data.js';

/**
 * Dodecahedral field body with addressable faces.
 * Face IDs 0..11 are stable and map 1:1 to MetaField optical-body cells.
 */

function createDodecahedronGeometry(radius = 1.35) {
  const geo = new THREE.DodecahedronGeometry(radius, 0);
  return geo;
}

/**
 * Build a mesh group where each of the 12 faces is a selectable Mesh.
 */
export function createFieldBody(scene) {
  const group = new THREE.Group();
  group.name = 'fieldBody';

  const shellGeo = new THREE.DodecahedronGeometry(1.36, 0);
  const shellMat = new THREE.MeshBasicMaterial({
    color: 0x1a3050,
    wireframe: true,
    transparent: true,
    opacity: 0.55,
  });
  const shell = new THREE.Mesh(shellGeo, shellMat);
  group.add(shell);

  const bodyMat = new THREE.MeshPhysicalMaterial({
    color: 0x0a1525,
    metalness: 0.15,
    roughness: 0.55,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
    transmission: 0.15,
    thickness: 0.6,
  });
  const bodyMesh = new THREE.Mesh(shellGeo.clone(), bodyMat);
  group.add(bodyMesh);

  const faceMeshes = [];
  const faceCenters = [];
  const faceNormals = [];

  const pos = shellGeo.attributes.position;
  const index = shellGeo.index;

  const tmpA = new THREE.Vector3();
  const tmpB = new THREE.Vector3();
  const tmpC = new THREE.Vector3();
  const tmpN = new THREE.Vector3();
  const tmpCentroid = new THREE.Vector3();

  const buckets = new Map();
  const normalKey = (n) =>
    `${n.x.toFixed(4)},${n.y.toFixed(4)},${n.z.toFixed(4)}`;

  for (let i = 0; i < index.count; i += 3) {
    const ia = index.getX(i);
    const ib = index.getX(i + 1);
    const ic = index.getX(i + 2);
    tmpA.fromBufferAttribute(pos, ia);
    tmpB.fromBufferAttribute(pos, ib);
    tmpC.fromBufferAttribute(pos, ic);
    tmpN.subVectors(tmpB, tmpA).cross(tmpC.clone().sub(tmpA)).normalize();
    const key = normalKey(tmpN);
    if (!buckets.has(key)) {
      buckets.set(key, {
        normal: tmpN.clone(),
        vertices: [],
        indices: [],
      });
    }
    const b = buckets.get(key);
    b.vertices.push(tmpA.clone(), tmpB.clone(), tmpC.clone());
    b.indices.push(ia, ib, ic);
  }

  let faceId = 0;
  for (const [, bucket] of buckets) {
    if (faceId >= NUM_FACES) break;

    tmpCentroid.set(0, 0, 0);
    for (const v of bucket.vertices) tmpCentroid.add(v);
    tmpCentroid.divideScalar(bucket.vertices.length);

    const ring = [];
    const seen = new Set();
    for (const v of bucket.vertices) {
      const k = `${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
      if (!seen.has(k)) {
        seen.add(k);
        ring.push(v.clone().addScaledVector(bucket.normal, 0.025));
      }
    }
    const origin = tmpCentroid.clone().addScaledVector(bucket.normal, 0.025);
    const ref = new THREE.Vector3();
    if (Math.abs(bucket.normal.y) < 0.9) ref.set(0, 1, 0);
    else ref.set(1, 0, 0);
    const tangent = new THREE.Vector3().crossVectors(bucket.normal, ref).normalize();
    const bitangent = new THREE.Vector3().crossVectors(bucket.normal, tangent).normalize();
    ring.sort((a, b) => {
      const va = a.clone().sub(origin);
      const vb = b.clone().sub(origin);
      const angA = Math.atan2(va.dot(bitangent), va.dot(tangent));
      const angB = Math.atan2(vb.dot(bitangent), vb.dot(tangent));
      return angA - angB;
    });

    const panelPositions = [];
    const panelIndices = [];
    panelPositions.push(origin.x, origin.y, origin.z);
    for (const v of ring) {
      panelPositions.push(v.x, v.y, v.z);
    }
    for (let i = 0; i < ring.length; i++) {
      const a = 0;
      const b = i + 1;
      const c = ((i + 1) % ring.length) + 1;
      panelIndices.push(a, b, c);
    }

    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(panelPositions, 3)
    );
    pGeo.setIndex(panelIndices);
    pGeo.computeVertexNormals();

    const baseColor = new THREE.Color(0x122033);
    const mat = new THREE.MeshStandardMaterial({
      color: baseColor.clone(),
      emissive: new THREE.Color(0x000000),
      metalness: 0.2,
      roughness: 0.45,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(pGeo, mat);
    mesh.userData.faceId = faceId;
    mesh.userData.baseColor = baseColor.clone();
    mesh.name = `face_${faceId}`;
    group.add(mesh);
    faceMeshes.push(mesh);
    faceCenters.push(origin.clone());
    faceNormals.push(bucket.normal.clone());

    const markerGeo = new THREE.SphereGeometry(0.035, 8, 8);
    const markerMat = new THREE.MeshBasicMaterial({
      color: 0x3aa0ff,
      transparent: true,
      opacity: 0.7,
    });
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.position.copy(origin).addScaledVector(bucket.normal, 0.06);
    marker.userData.faceId = faceId;
    group.add(marker);

    faceId++;
  }

  while (faceMeshes.length < NUM_FACES) {
    const i = faceMeshes.length;
    const phi = (i / NUM_FACES) * Math.PI * 2;
    const c = new THREE.Vector3(Math.cos(phi), Math.sin(phi) * 0.3, Math.sin(phi)).normalize().multiplyScalar(1.2);
    faceCenters.push(c);
    faceNormals.push(c.clone().normalize());
    faceMeshes.push(null);
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

    const base = mesh.userData.baseColor;
    const color = base.clone();

    const v = value ?? state.values[faceId] ?? 0;
    color.lerp(new THREE.Color(0x1a6aaa), THREE.MathUtils.clamp(v, 0, 1));

    const a = anomaly ?? state.anomaly[faceId] ?? 0;
    if (a > 0.05) {
      color.lerp(new THREE.Color(0xff6b4a), THREE.MathUtils.clamp(a, 0, 1));
    }

    mesh.material.color.copy(color);

    let em = 0x000000;
    if (highlight) em = 0x3aa0ff;
    if (pred) em = 0x6b4dff;
    if (observed) em = 0x2ecc7a;
    if (a > 0.25) em = 0xff4422;

    mesh.material.emissive = new THREE.Color(em);
    mesh.material.emissiveIntensity = highlight || pred || observed || a > 0.2 ? 0.55 : 0.08;

    const conf = confidence ?? state.confidence[faceId] ?? 0.5;
    mesh.material.opacity = 0.55 + conf * 0.4;
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
        pred: isPred && obs.phase === 'predict',
        observed: isProbe && (obs.phase === 'observe' || obs.phase === 'compare'),
      });
    }
  }

  function setSelectedFace(id) {
    state.selectedFace = id;
  }

  function getFaceWorldCenter(faceId) {
    const local = faceCenters[faceId];
    if (!local) return new THREE.Vector3();
    return local.clone().applyMatrix4(group.matrixWorld);
  }

  function getFaceNormal(faceId) {
    return faceNormals[faceId]?.clone() ?? new THREE.Vector3(0, 1, 0);
  }

  function getPickables() {
    return faceMeshes.filter(Boolean);
  }

  function update(dt) {
    if (autoRotate) {
      group.rotation.y += dt * 0.08;
    }
  }

  return {
    group,
    faceMeshes,
    faceCenters,
    faceNormals,
    state,
    applyObservation,
    setSelectedFace,
    getFaceWorldCenter,
    getFaceNormal,
    getPickables,
    update,
    setAutoRotate(v) {
      autoRotate = v;
    },
  };
}
