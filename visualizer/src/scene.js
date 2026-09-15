import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Core Three.js scene, camera, renderer, lights, animation loop.
 */
export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070c);
  scene.fog = new THREE.FogExp2(0x05070c, 0.035);

  const camera = new THREE.PerspectiveCamera(
    42,
    container.clientWidth / container.clientHeight,
    0.1,
    100
  );
  camera.position.set(3.8, 2.4, 4.6);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 2.2;
  controls.maxDistance = 14;
  controls.target.set(0, 0, 0);
  controls.update();

  const ambient = new THREE.AmbientLight(0x334466, 0.55);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xaaccff, 1.1);
  key.position.set(4, 6, 3);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x4466aa, 0.35);
  fill.position.set(-3, 1, -4);
  scene.add(fill);

  const rim = new THREE.PointLight(0x3aa0ff, 0.6, 20);
  rim.position.set(-2, -1, 3);
  scene.add(rim);

  const grid = new THREE.GridHelper(12, 24, 0x1a2a40, 0x0e1520);
  grid.position.y = -1.8;
  grid.material.opacity = 0.45;
  grid.material.transparent = true;
  scene.add(grid);

  const clock = new THREE.Clock();
  const updatables = [];

  function onResize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener('resize', onResize);

  let raf = 0;
  function animate() {
    raf = requestAnimationFrame(animate);
    const dt = clock.getDelta();
    controls.update();
    for (const u of updatables) u(dt, clock.elapsedTime);
    renderer.render(scene, camera);
  }
  animate();

  function addUpdatable(fn) {
    updatables.push(fn);
  }

  function dispose() {
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', onResize);
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
  }

  return {
    scene,
    camera,
    renderer,
    controls,
    addUpdatable,
    dispose,
    domElement: renderer.domElement,
  };
}
