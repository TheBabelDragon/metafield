import * as THREE from 'three';

/**
 * Active probe visualization: beam / excitation traveling to a face.
 */
export function createProbeSystem(scene, body) {
  const group = new THREE.Group();
  group.name = 'probes';
  scene.add(group);

  // Beam line
  const beamMat = new THREE.LineBasicMaterial({
    color: 0x3aa0ff,
    transparent: true,
    opacity: 0.85,
  });
  const beamGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(),
    new THREE.Vector3(),
  ]);
  const beam = new THREE.Line(beamGeo, beamMat);
  beam.visible = false;
  group.add(beam);

  // Excitation core at impact
  const coreGeo = new THREE.SphereGeometry(0.08, 16, 16);
  const coreMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
  });
  const core = new THREE.Mesh(coreGeo, coreMat);
  core.visible = false;
  group.add(core);

  // Expanding ring on face
  const ringGeo = new THREE.RingGeometry(0.05, 0.12, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x3aa0ff,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.visible = false;
  group.add(ring);

  // Origin point (instrument emitter)
  const origin = new THREE.Vector3(0, 2.6, 0);
  const originMarker = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.06, 0),
    new THREE.MeshBasicMaterial({ color: 0x3aa0ff })
  );
  originMarker.position.copy(origin);
  group.add(originMarker);

  let anim = null; // { face, t, duration, intensity, wavelength }

  function triggerProbe(faceId, intensity = 1, wavelength = 850) {
    anim = {
      face: faceId,
      t: 0,
      duration: 0.85,
      intensity,
      wavelength,
    };
    beam.visible = true;
    core.visible = true;
    ring.visible = true;

    // Wavelength → color tint (IR-ish)
    const hue = wavelength > 900 ? 0.95 : wavelength > 800 ? 0.58 : 0.05;
    const color = new THREE.Color().setHSL(hue, 0.85, 0.6);
    beamMat.color.copy(color);
    coreMat.color.copy(color);
    ringMat.color.copy(color);
  }

  function clearProbe() {
    anim = null;
    beam.visible = false;
    core.visible = false;
    ring.visible = false;
  }

  function applyObservation(obs) {
    if (obs?.probe && (obs.phase === 'probe' || obs.phase === 'observe')) {
      if (!anim || anim.face !== obs.probe.face) {
        triggerProbe(obs.probe.face, obs.probe.intensity, obs.probe.wavelength_nm);
      }
    } else if (obs?.phase === 'select' || obs?.phase === 'idle') {
      // keep residual glow briefly; clear after compare
    }
    if (obs?.phase === 'select' && anim && anim.t > anim.duration) {
      clearProbe();
    }
  }

  function update(dt) {
    originMarker.rotation.y += dt * 1.5;
    originMarker.rotation.x += dt * 0.8;

    if (!anim) return;

    anim.t += dt;
    const u = Math.min(1, anim.t / anim.duration);
    // Ease out
    const e = 1 - Math.pow(1 - u, 3);

    const center = body.faceCenters[anim.face];
    const normal = body.faceNormals[anim.face];
    if (!center || !normal) return;

    const target = center.clone().addScaledVector(normal, 0.08);
    const current = origin.clone().lerp(target, e);

    // Update beam
    const positions = beam.geometry.attributes.position;
    positions.setXYZ(0, origin.x, origin.y, origin.z);
    positions.setXYZ(1, current.x, current.y, current.z);
    positions.needsUpdate = true;

    core.position.copy(current);
    core.scale.setScalar(0.6 + Math.sin(u * Math.PI) * 1.4 * anim.intensity);

    // Ring expands on impact
    if (u > 0.55) {
      const impact = (u - 0.55) / 0.45;
      ring.position.copy(target);
      // Orient ring to face normal
      ring.lookAt(target.clone().add(normal));
      const s = 0.5 + impact * 3.5;
      ring.scale.set(s, s, s);
      ring.material.opacity = 0.85 * (1 - impact);
    } else {
      ring.material.opacity = 0;
    }

    beam.material.opacity = 0.9 * (1 - u * 0.5);
    core.material.opacity = 0.95 * (1 - Math.max(0, u - 0.7) / 0.3);

    if (u >= 1) {
      // Hold residual then fade on next phases
      beam.material.opacity = 0.25;
    }
  }

  return {
    group,
    triggerProbe,
    clearProbe,
    applyObservation,
    update,
  };
}
