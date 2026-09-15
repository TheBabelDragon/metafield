import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070c);
  scene.fog = new THREE.FogExp2(0x05070c, 0.028);

  const w = Math.max(container.clientWidth || window.innerWidth, 1);
  const h = Math.max(container.clientHeight || window.innerHeight, 1);

  const camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
  camera.position.set(3.6, 2.2, 4.4);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 2.0;
  controls.maxDistance = 14;
  controls.target.set(0, 0, 0);
  controls.update();

  scene.add(new THREE.AmbientLight(0x445566, 0.85));

  const key = new THREE.DirectionalLight(0xd0e8ff, 1.35);
  key.position.set(4, 6, 3);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x6688cc, 0.55);
  fill.position.set(-3, 1, -4);
  scene.add(fill);

  const rim = new THREE.PointLight(0x3aa0ff, 1.1, 24);
  rim.position.set(-2, -1, 3);
  scene.add(rim);

  const grid = new THREE.GridHelper(12, 24, 0x1a2a40, 0x0e1520);
  grid.position.y = -1.8;
  grid.material.opacity = 0.5;
  grid.material.transparent = true;
  scene.add(grid);

  const clock = new THREE.Clock();
  const updatables = [];

  function onResize() {
    const rw = Math.max(container.clientWidth || window.innerWidth, 1);
    const rh = Math.max(container.clientHeight || window.innerHeight, 1);
    camera.aspect = rw / rh;
    camera.updateProjectionMatrix();
    renderer.setSize(rw, rh);
  }
  window.addEventListener('resize', onResize);

  let raf = 0;
  function animate() {
    raf = requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    controls.update();
    for (const u of updatables) u(dt, clock.elapsedTime);
    renderer.render(scene, camera);
  }
  animate();

  return {
    scene,
    camera,
    renderer,
    controls,
    addUpdatable(fn) {
      updatables.push(fn);
    },
    dispose() {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    },
    domElement: renderer.domElement,
  };
}
