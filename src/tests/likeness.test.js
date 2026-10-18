import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from '../../vendor/three.core.js';
import { readBundledGeometry } from './headFixture.js';
import { buildAnatomicalRig } from '../anatomy.js';
import { CONFIG as TRACKER } from '../tracker.js';
import {
  FACE_OVAL, FIT_LANDMARKS, normalizeFaceLandmarks, validatePortrait,
  makeIdentityControls, makeFaceWarp, deformIdentityGeometry,
  createTriangleSampler, LatestFaceSource,
} from '../likeness.js';
import { samplePortraitPixel, sampleSkinTone, mapPortraitSurface, bakePortraitAtlas } from '../portraitTexture.js';

function landmarks() {
  const lm = Array.from({ length: 478 }, (_, i) => ({ x: 0.30 + i % 20 * 0.02, y: 0.25 + Math.floor(i / 20) * 0.018, z: Math.sin(i) * 0.01 }));
  for (const [i, x, y] of [[33, 0.34, 0.42], [133, 0.43, 0.42], [263, 0.66, 0.42], [362, 0.57, 0.42]]) lm[i] = { x, y, z: 0 };
  for (let j = 0; j < FACE_OVAL.length; j++) {
    const angle = j / FACE_OVAL.length * Math.PI * 2 - Math.PI / 2;
    lm[FACE_OVAL[j]] = { x: 0.5 + Math.cos(angle) * 0.27, y: 0.5 + Math.sin(angle) * 0.34, z: 0.01 };
  }
  return lm;
}
const near = (a, b, epsilon = 1e-5) => assert.ok(Math.abs(a - b) < epsilon, `${a} ≈ ${b}`);
const nearPoints = (a, b, epsilon) => a.forEach((p, i) => p.forEach((v, k) => near(v, b[i][k], epsilon)));
const identityResult = (lm = landmarks()) => ({ faceLandmarks: [lm] });

function controls(delta = [0.045, -0.025, 0.018]) {
  return [[-0.4, -0.5, 0.8], [0.4, -0.5, 0.8], [-0.4, 0.5, 0.8], [0.4, 0.5, 0.8], [0, 0, 0.9]]
    .map((from) => ({ from, delta }));
}
const warp = makeFaceWarp(controls());

function solidPixels(color, size = 8) {
  return Uint8ClampedArray.from({ length: size * size * 4 }, (_, i) => i % 4 === 3 ? 255 : color[i % 4]);
}
function quad(alpha = 1) {
  return {
    indices: [0, 1, 2, 0, 2, 3], atlasUV: [0, 0, 1, 0, 1, 1, 0, 1],
    photoUV: [0, 1, 1, 1, 1, 0, 0, 0], alpha: [alpha, alpha, alpha, alpha],
    positions: [-0.3, -0.3, 0.8, 0.3, -0.3, 0.8, 0.3, 0.3, 0.8, -0.3, 0.3, 0.8],
  };
}

test('likeness normalization ignores image resolution, crop translation and subject distance', () => {
  const lm = landmarks(), base = normalizeFaceLandmarks(lm, 640, 480);
  nearPoints(base, normalizeFaceLandmarks(lm, 1280, 960));
  nearPoints(base, normalizeFaceLandmarks(lm.map((p) => ({ ...p, x: p.x + 0.1, y: p.y - 0.08 })), 640, 480));
  const scaled = lm.map((p) => ({ x: (p.x - 0.5) * 0.7 + 0.5, y: (p.y - 0.5) * 0.7 + 0.5, z: p.z * 0.7 }));
  nearPoints(base, normalizeFaceLandmarks(scaled, 640, 480));
});

test('landmark normalization handles non-square images and camera roll', () => {
  const lm = landmarks(), angle = 0.28, c = Math.cos(angle), s = Math.sin(angle);
  const rotated = lm.map((p) => {
    const x = (p.x - 0.5) * 640, y = (p.y - 0.5) * 480;
    return { x: (x * c - y * s) / 640 + 0.5, y: (x * s + y * c) / 480 + 0.5, z: p.z };
  });
  nearPoints(normalizeFaceLandmarks(lm, 640, 480), normalizeFaceLandmarks(rotated, 640, 480));
});

test('identity fitting removes estimated yaw/pitch instead of baking the photo pose into the face', () => {
  const lm = landmarks(), rotation = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0.12, 0.22, -0.08));
  const transformed = lm.map((p) => {
    const v = new THREE.Vector3((p.x - 0.5) * 640, (0.5 - p.y) * 480, -p.z * 640).applyMatrix4(rotation);
    return { x: 0.5 + v.x / 640, y: 0.5 - v.y / 480, z: -v.z / 640 };
  });
  nearPoints(normalizeFaceLandmarks(lm, 640, 480), normalizeFaceLandmarks(transformed, 640, 480, rotation.elements));
});

test('same face at another image size produces zero identity deltas', () => {
  const lm = landmarks();
  const reference = { landmarks: lm, width: 640, height: 480, anchors: FIT_LANDMARKS.map((index) => ({ index, point: [0, 0, 0.8] })) };
  for (const c of makeIdentityControls(reference, lm, 1280, 960)) for (const d of c.delta) near(d, 0);
});

test('portrait validation rejects missing/ambiguous/partial/nonfinite/small faces before changing identity', () => {
  assert.throws(() => validatePortrait({ faceLandmarks: [] }, 640, 480), /No face/);
  assert.throws(() => validatePortrait({ faceLandmarks: [landmarks(), landmarks()] }, 640, 480), /just one/);
  const clipped = landmarks(); clipped[152].y = 1.05;
  assert.throws(() => validatePortrait(identityResult(clipped), 640, 480), /whole face/);
  const invalid = landmarks(); invalid[1].z = NaN;
  assert.throws(() => validatePortrait(identityResult(invalid), 640, 480), /Invalid/);
  assert.throws(() => validatePortrait(identityResult(), 32, 32), /too small/);
  assert.throws(() => normalizeFaceLandmarks([], 640, 480), /complete/);
});

test('portrait profiles are rejected, while clear front-facing portraits are accepted', () => {
  const matrix = new THREE.Matrix4().makeRotationY(0.7).elements;
  assert.throws(() => validatePortrait({ ...identityResult(), facialTransformationMatrixes: [{ data: matrix }] }, 640, 480), /front-facing/);
  assert.equal(validatePortrait(identityResult(), 640, 480).length, 478);
});

test('smooth identity field fits facial controls but keeps ears, rear skull and neck rigid', () => {
  const out = new Float64Array(3);
  warp(0, 0, 0.8, out); nearPoints([Array.from(out)], [[0.045, -0.025, 0.818]], 1e-4);
  for (const point of [[0.68, 0, 0.1], [-0.68, 0, 0.1], [0, 0, -0.7], [0, -1.12, 0.8]]) {
    warp(...point, out); assert.deepEqual(Array.from(out), point);
  }
});

test('single-view fitting cannot extrapolate a forehead ridge onto the unseen round cranium', () => {
  const point = [0.1, 0.95, 0.6], out = new Float64Array(3);
  warp(...point, out);
  assert.deepEqual(Array.from(out), point);
});

test('extreme source proportions cannot produce unbounded identity displacements', () => {
  const extreme = makeFaceWarp(controls([8, -6, 10])), p = new Float64Array(3);
  extreme(0, 0, 0.8, p);
  assert.ok(Math.abs(p[0]) <= 0.16001 && Math.abs(p[1]) <= 0.22001 && Math.abs(p[2] - 0.8) <= 0.12001);
  assert.throws(() => makeFaceWarp([]), /valid/);
});

function triangleGeometry() {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([-0.1, 0, 0.8, 0.1, 0, 0.8, 0, 0.2, 0.8], 3));
  geo.setIndex([0, 1, 2]); geo.computeVertexNormals();
  geo.morphTargetsRelative = true;
  geo.morphAttributes.position = [new THREE.Float32BufferAttribute([0.02, -0.01, 0.01, 0.01, 0, 0, 0, 0.02, 0.01], 3), new THREE.Float32BufferAttribute(new Float32Array(9), 3)];
  geo.morphAttributes.normal = [new THREE.Float32BufferAttribute(new Float32Array(9), 3), new THREE.Float32BufferAttribute(new Float32Array(9), 3)];
  return geo;
}

test('identity deformation rebases expression deltas instead of detaching morphs from the fitted face', () => {
  const source = triangleGeometry(), before = source.attributes.position.array.slice();
  const fitted = deformIdentityGeometry(source, warp), result = new Float64Array(3);
  for (let i = 0; i < 3; i++) {
    const rest = new THREE.Vector3().fromBufferAttribute(source.attributes.position, i);
    const delta = new THREE.Vector3().fromBufferAttribute(source.morphAttributes.position[0], i);
    warp(rest.x + delta.x, rest.y + delta.y, rest.z + delta.z, result);
    for (let k = 0; k < 3; k++) near(fitted.attributes.position.array[i * 3 + k] + fitted.morphAttributes.position[0].array[i * 3 + k], result[k]);
  }
  assert.deepEqual(source.attributes.position.array, before);
  assert.ok(fitted.morphAttributes.position[1].array.every((v) => v === 0));
  fitted.dispose(); source.dispose();
});

test('replacing a likeness always starts from pristine geometry, never accumulates the previous fit', () => {
  const source = triangleGeometry();
  const a = deformIdentityGeometry(source, warp), b = deformIdentityGeometry(source, makeFaceWarp(controls([-0.03, 0.02, 0])));
  const again = deformIdentityGeometry(source, warp);
  assert.deepEqual(a.attributes.position.array, again.attributes.position.array);
  assert.notDeepEqual(a.attributes.position.array, b.attributes.position.array);
  for (const geo of [source, a, b, again]) geo.dispose();
});

test('eyeball geometry and its local gaze pivot use the same identity warp as the lids', () => {
  const source = new THREE.SphereGeometry(0.12, 8, 8), offset = [0.28, 0.105, 0.665], next = new Float64Array(3);
  warp(...offset, next);
  const fitted = deformIdentityGeometry(source, warp, offset, next), point = new Float64Array(3);
  for (let i = 0; i < source.attributes.position.count; i++) {
    warp(source.attributes.position.getX(i) + offset[0], source.attributes.position.getY(i) + offset[1], source.attributes.position.getZ(i) + offset[2], point);
    for (let k = 0; k < 3; k++) near(fitted.attributes.position.array[i * 3 + k] + next[k], point[k]);
  }
  fitted.dispose(); source.dispose();
});

test('actual anatomical ears stay bit-identical after likeness fitting across all 52 position/normal morphs', () => {
  const source = readBundledGeometry(), asset = buildAnatomicalRig(source, new THREE.MeshPhysicalMaterial(), new THREE.Texture());
  const original = asset.meshes[0].geometry, fitted = deformIdentityGeometry(original, warp), p = original.attributes.position;
  let ears = 0, changedFace = 0;
  for (let i = 0; i < p.count; i++) {
    if (Math.abs(p.getX(i)) > 0.615 && Math.abs(p.getY(i)) < 0.30 && p.getZ(i) < 0.15 && p.getZ(i) > -0.17) {
      ears++;
      for (const attr of ['position', 'normal']) for (let k = 0; k < 3; k++) assert.equal(fitted.attributes[attr].array[i * 3 + k], original.attributes[attr].array[i * 3 + k]);
      for (const attr of ['position', 'normal']) for (const morph of fitted.morphAttributes[attr]) for (let k = 0; k < 3; k++) assert.equal(morph.array[i * 3 + k], 0);
    }
    if (Math.abs(p.getX(i)) < 0.4 && p.getZ(i) > 0.7 && Math.abs(p.getY(i)) < 0.5
      && fitted.attributes.position.getX(i) !== p.getX(i)) changedFace++;
  }
  assert.ok(ears > 200 && changedFace > 100);
  assert.equal(fitted.morphAttributes.position.length, 52);
  for (const attr of ['position', 'normal']) for (const morph of fitted.morphAttributes[attr]) assert.ok(morph.array.every(Number.isFinite));
  fitted.dispose(); asset.root.traverse((o) => { if (o.isMesh) o.geometry.dispose(); }); source.dispose();
});

test('barycentric texture mapping preserves landmark correspondence rather than averaging facial features', () => {
  const sampler = createTriangleSampler([{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }], [0, 1, 2]);
  const hit = sampler(0.2, 0.3);
  nearPoints([hit.weights], [[0.5, 0.2, 0.3]]);
  assert.equal(sampler(0.9, 0.9), null);
  assert.equal(sampler(-0.1, 0.2), null);
});

test('local canonical topology is complete, triangular and indexed to the 468 face landmarks', () => {
  const topology = JSON.parse(readFileSync(new URL('../../assets/face-topology.json', import.meta.url)));
  assert.equal(topology.length, 898 * 3);
  assert.ok(topology.every((i) => Number.isInteger(i) && i >= 0 && i < 468));
});

test('bilinear photo sampling clamps edges and interpolates RGB without flipping the portrait', () => {
  const pixels = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
  assert.deepEqual(Array.from(samplePortraitPixel(pixels, 2, 2, 0, 0, [0, 0, 0])), [255, 0, 0]);
  assert.deepEqual(Array.from(samplePortraitPixel(pixels, 2, 2, 1, 1, [0, 0, 0])), [255, 255, 255]);
  assert.deepEqual(Array.from(samplePortraitPixel(pixels, 2, 2, 0.5, 0.5, [0, 0, 0])), [127.5, 127.5, 127.5]);
});

test('skin tone is measured from the current portrait, not hardcoded or inherited from the stock head', () => {
  assert.deepEqual(sampleSkinTone(solidPixels([95, 64, 48]), 8, 8, landmarks()), [95, 64, 48]);
  assert.deepEqual(sampleSkinTone(solidPixels([223, 185, 162]), 8, 8, landmarks()), [223, 185, 162]);
});

test('portrait mapping never projects the photograph onto an ear or the rear skull', () => {
  const lm = landmarks(); lm[0] = { x: 0.25, y: 0.25 }; lm[1] = { x: 0.75, y: 0.25 }; lm[2] = { x: 0.5, y: 0.75 };
  const surface = { projected: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], positions: [0.68, 0, 0.1, 0, 0, -0.7, 0, 0, 0.8], normals: [0, 0, 1, 0, 0, -1, 0, 0, 1] };
  const mapped = mapPortraitSurface(surface, { landmarks: lm, width: 640, height: 480 }, lm, [0, 1, 2], 640, 480);
  assert.equal(mapped.alpha[0], 0); assert.equal(mapped.alpha[1], 0); assert.equal(mapped.alpha[2], 1);
});

test('baked atlas contains the active photo and excludes stock texture data', () => {
  const atlas = bakePortraitAtlas([quad()], solidPixels([174, 70, 35]), 8, 8, [100, 80, 65], undefined, 32);
  assert.deepEqual(Array.from(atlas.data.slice((16 * 32 + 16) * 4, (16 * 32 + 16) * 4 + 4)), [174, 70, 35, 255]);
  assert.ok(atlas.photoPixels > 500);
});

test('arbitrary neck-cap UVs cannot paint a flat skin-colored patch over the portrait chin', () => {
  const atlas = bakePortraitAtlas([quad(), quad(0)], solidPixels([174, 70, 35]), 8, 8, [100, 80, 65], undefined, 32);
  assert.deepEqual(Array.from(atlas.data.slice((16 * 32 + 16) * 4, (16 * 32 + 16) * 4 + 4)), [174, 70, 35, 255]);
});

test('latest face source wins, queued stale images never run and reset cancels pending work', async () => {
  const source = new LatestFaceSource(), committed = [];
  let release, begin;
  const started = new Promise((resolve) => { begin = resolve; });
  const first = source.run(async (current) => {
    begin();
    await new Promise((resolve) => { release = resolve; });
    if (current()) committed.push('old');
  });
  await started;
  const second = source.run(async () => committed.push('middle'));
  const third = source.run(async (current) => { if (current()) committed.push('new'); });
  release(); await Promise.all([first, second, third]);
  assert.deepEqual(committed, ['new']);
  const cancelled = source.run(async () => committed.push('cancelled')); source.cancel(); await cancelled;
  assert.deepEqual(committed, ['new']);
});

test('a failed face load does not poison the source queue', async () => {
  const source = new LatestFaceSource();
  await assert.rejects(source.run(async () => { throw new Error('bad image'); }), /bad image/);
  assert.equal(await source.run(async () => 'next face'), 'next face');
});

test('tracker and photo detector use module-relative local MediaPipe assets, including under a URL prefix', () => {
  for (const [key, suffix] of [['visionModule', '/vendor/tasks-vision/vision_bundle.mjs'], ['wasmLocal', '/vendor/tasks-vision/wasm'], ['modelLocal', '/assets/face_landmarker.task']]) {
    assert.ok(TRACKER[key].endsWith(suffix));
    assert.ok(new URL(TRACKER[key]).protocol === 'file:'); // Node; http(s) in the browser
    assert.ok(!TRACKER[key].includes('/src/vendor/'));
  }
});

test('vendored MediaPipe telemetry is disabled so closing an image detector cannot transmit usage logs', () => {
  const runtime = readFileSync(new URL('../../vendor/tasks-vision/vision_bundle.mjs', import.meta.url), 'utf8');
  assert.ok(runtime.includes('Usage telemetry disabled by FaceMirror: local-only processing'));
  assert.ok(!runtime.includes('this.h=setInterval(()=>{this.flush()},6e4)'));
});
