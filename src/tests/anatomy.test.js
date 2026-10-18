/**
 * Anatomical ear regressions: actual bundled vertices, all 52 morphs,
 * tracked/dragged poses, depth occlusion, and the asset-free closed fallback.
 * Uses the vendored three.js core; no npm dependencies, camera, or browser.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readBundledGeometry } from './headFixture.js';
import * as THREE from '../../vendor/three.core.js';
import { CHANNELS, CH } from '../rig.js';
import { PoseTracker } from '../pose.js';
import { ANATOMY, anatomicalMorphDelta, buildAnatomicalRig, clipNeck, eyelidPoint } from '../anatomy.js';
import { buildFallbackEars } from '../ears.js';

const source = readBundledGeometry();
const asset = buildAnatomicalRig(source, new THREE.MeshPhysicalMaterial(), new THREE.Texture());
const head = asset.meshes[0];
const position = head.geometry.getAttribute('position');
const earIndices = [-1, 1].map((side) => Array.from({ length: position.count }, (_, i) => i).filter((i) => (
  position.getX(i) * side > 0.615 && position.getY(i) > -0.35 && position.getY(i) < 0.25
  && position.getZ(i) > -0.17 && position.getZ(i) < 0.15
)));
const vertex = (i) => new THREE.Vector3().fromBufferAttribute(position, i);

function earVertex(side) {
  const target = new THREE.Vector3(side * 0.68, -0.03, -0.02);
  return earIndices[side < 0 ? 0 : 1].reduce((a, b) => vertex(a).distanceToSquared(target) < vertex(b).distanceToSquared(target) ? a : b);
}

test('bundled head has anatomical ears in the skull mesh, with no separate floating ear nodes', () => {
  for (const vertices of earIndices) assert.ok(vertices.length > 100, `${vertices.length} real ear vertices`);
  assert.equal(asset.root.getObjectByName('anatomicalHeadAndEars'), head);
  assert.equal(head.parent, asset.root);
  assert.equal(head.material.transparent, false);
  assert.equal(head.material.depthTest, true);
  assert.equal(head.material.depthWrite, true);
  assert.equal(head.material.side, THREE.FrontSide);
});

test('all 52 expression fields leave BOTH actual ears rigid, including jaw and cheek extremes', () => {
  const delta = new Float32Array(3);
  for (const name of CHANNELS) {
    for (const ids of earIndices) {
      for (const i of ids) {
        anatomicalMorphDelta(name, position.getX(i), position.getY(i), position.getZ(i), delta);
        assert.deepEqual([...delta], [0, 0, 0], `${name} must not drag ear vertex ${i}`);
      }
    }
  }
  anatomicalMorphDelta('mouthSmileLeft', -0.30, -0.30, 0.72, delta);
  assert.ok(delta[1] > 0.05, 'smiling still moves facial tissue');
  anatomicalMorphDelta('jawForward', 0, -0.60, 0.72, delta);
  assert.ok(delta[2] > 0.03, 'jaw channels still work');
});

test('stored morph positions AND normals keep ear lighting stationary under expressions', () => {
  for (const name of CHANNELS) {
    const k = CH[name];
    for (const attr of ['position', 'normal']) {
      const morph = head.geometry.morphAttributes[attr][k];
      for (const ids of earIndices) for (const i of ids) {
        assert.deepEqual([morph.getX(i), morph.getY(i), morph.getZ(i)], [0, 0, 0], `${name}: ${attr} at ear ${i}`);
      }
    }
  }
});

test('ears follow the complete tracked yaw/pitch/roll and manual drag transform exactly', () => {
  head.morphTargetInfluences.fill(0.7);
  const quat = new Float32Array(4);
  for (const [yaw, pitch, roll] of [[1.2, 0.3, 0.2], [-1.2, -0.3, -0.2], [0, 0.6, -0.4]]) {
    const pose = new PoseTracker();
    pose.neckLagSec = 0;
    const matrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));
    pose.setFromMatrix(matrix.elements, 0);
    pose.dragBy(55, -20);
    for (let i = 0; i < 90; i++) pose.update(1 / 60, i / 60);
    pose.getQuaternion(quat);
    asset.root.quaternion.fromArray(quat);
    asset.root.position.set(0.12, -0.05, 0.03);
    asset.root.updateMatrixWorld(true);
    for (const side of [-1, 1]) {
      const i = earVertex(side);
      const expected = vertex(i).applyMatrix4(asset.root.matrixWorld);
      const actual = head.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(head.matrixWorld);
      assert.ok(actual.distanceTo(expected) < 1e-7, `ear stays attached at yaw ${yaw}`);
    }
  }
  head.morphTargetInfluences.fill(0);
  asset.root.position.set(0, 0, 0);
  asset.root.rotation.set(0, 0, 0);
});

test('profile turns naturally occlude the far ear with the skull in BOTH directions', () => {
  const camera = new THREE.Vector3(0, 0.06, 5.1);
  for (const yaw of [-70, 70, -85, 85]) {
    asset.root.rotation.set(0, yaw * Math.PI / 180, 0);
    asset.root.updateMatrixWorld(true);
    const nearSide = yaw > 0 ? -1 : 1;
    for (const side of [-1, 1]) {
      const target = vertex(earVertex(side)).applyMatrix4(head.matrixWorld);
      const distance = target.distanceTo(camera);
      const ray = new THREE.Raycaster(camera, target.clone().sub(camera).normalize());
      const hit = ray.intersectObject(head, false)[0];
      assert.ok(hit, 'the opaque head intersects the ear sightline');
      const localHit = head.worldToLocal(hit.point.clone());
      if (side === nearSide) {
        // The helix may occlude a vertex down inside the concha. The first
        // visible surface must still be the near ear, not the cranium.
        assert.ok(localHit.x * side > 0.615 && localHit.z < 0.15,
          `near ear visible at ${yaw}°: ${localHit.toArray()}`);
      } else assert.ok(hit.distance < distance - 0.10 && localHit.x * side < 0.615, `far ear behind skull at ${yaw}°`);
    }
  }
  asset.root.rotation.set(0, 0, 0);
});

test('anatomical rig still exposes every ARKit channel and independent curved blink targets', () => {
  assert.deepEqual(Object.keys(asset.morphMap).sort(), [...CHANNELS].sort());
  assert.equal(asset.missingChannels.length, 0);
  assert.equal(asset.hasTeeth, false, 'no-teeth jaw cap remains honest');
  for (const mesh of asset.meshes) {
    assert.equal(mesh.geometry.morphAttributes.position.length, 52);
    assert.equal(mesh.geometry.morphAttributes.normal.length, 52);
  }
  const left = asset.meshes[1], right = asset.meshes[2];
  const own = left.geometry.morphAttributes.position[CH.eyeBlinkLeft];
  const other = right.geometry.morphAttributes.position[CH.eyeBlinkLeft];
  assert.ok(own.array.some((v) => Math.abs(v) > 0.02), 'left blink closes a real 3D lid');
  assert.ok(other.array.every((v) => v === 0), 'left wink does not move the right lid');
  const upper = new Float32Array(3), lower = new Float32Array(3);
  eyelidPoint(-1, Math.PI / 2, 0, 1, 0.8, upper);
  eyelidPoint(-1, Math.PI * 1.5, 0, 1, 0.8, lower);
  assert.ok(Math.hypot(...upper.map((v, i) => v - lower[i])) < 1e-6, 'fully closed edges meet without an eye gap');
});

test('blink geometry keeps the fixed socket boundary sealed', () => {
  for (const mesh of asset.meshes) {
    for (const name of ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeWideLeft', 'eyeSquintRight']) {
      const morph = mesh.geometry.morphAttributes.position[CH[name]];
      const start = mesh === head ? 0 : ANATOMY.lidRings * (ANATOMY.lidSegments + 1);
      for (let i = start; i < morph.count; i++) {
        assert.deepEqual([morph.getX(i), morph.getY(i), morph.getZ(i)], [0, 0, 0], `${name}: fixed outer socket`);
      }
    }
  }
});

test('combined blink + expression squint cannot invert the lids', () => {
  const weights = new Float32Array(52);
  weights[CH.eyeBlinkLeft] = 1;
  weights[CH.eyeSquintLeft] = 1;
  weights[CH.eyeWideRight] = 1;
  asset.updateLids(weights);
  assert.equal(asset.meshes[1].morphTargetInfluences[CH.eyeBlinkLeft], 1);
  assert.equal(asset.meshes[1].morphTargetInfluences[CH.eyeSquintLeft], 0);
  assert.equal(asset.meshes[2].morphTargetInfluences[CH.eyeBlinkRight], -0.30);
  for (const mesh of asset.meshes) mesh.morphTargetInfluences.fill(0);
});

test('neck clipping removes shoulder geometry, caps the cut, and keeps finite attributes', () => {
  const normalized = source.clone().scale(0.4, 0.4, 0.4).translate(0, -0.56, 0);
  const clipped = clipNeck(normalized);
  const p = clipped.getAttribute('position'), n = clipped.getAttribute('normal');
  assert.ok(p.count < source.attributes.position.count, 'unused shoulder vertices removed');
  let capVertices = 0;
  for (let i = 0; i < p.count; i++) {
    assert.ok(p.getY(i) >= ANATOMY.neckY - 1e-6);
    if (Math.abs(p.getY(i) - ANATOMY.neckY) < 1e-6 && n.getY(i) < -0.99) capVertices++;
  }
  assert.ok(capVertices > 20, 'downward-facing closed neck cap');
  for (const geometry of [clipped, ...asset.meshes.map((m) => m.geometry)]) {
    for (const attribute of Object.values(geometry.attributes)) assert.ok(attribute.array.every(Number.isFinite));
    for (const targets of Object.values(geometry.morphAttributes)) for (const target of targets) assert.ok(target.array.every(Number.isFinite));
  }
});

test('fallback ears are closed, mirrored volumes with outward winding on BOTH sides', () => {
  const ears = buildFallbackEars();
  const l = ears.left.geometry.attributes.position, r = ears.right.geometry.attributes.position;
  assert.equal(l.count, r.count);
  for (let i = 0; i < l.count; i++) {
    assert.equal(l.getX(i), -r.getX(i));
    assert.equal(l.getY(i), r.getY(i));
    assert.equal(l.getZ(i), r.getZ(i));
  }
  for (const ear of Object.values(ears)) {
    const p = ear.geometry.attributes.position, indices = ear.geometry.index;
    const edges = new Map();
    let volume = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    for (let i = 0; i < indices.count; i += 3) {
      const ids = [indices.getX(i), indices.getX(i + 1), indices.getX(i + 2)];
      a.fromBufferAttribute(p, ids[0]); b.fromBufferAttribute(p, ids[1]); c.fromBufferAttribute(p, ids[2]);
      volume += a.dot(b.cross(c)) / 6;
      for (let j = 0; j < 3; j++) {
        const x = ids[j], y = ids[(j + 1) % 3];
        const key = x < y ? `${x}:${y}` : `${y}:${x}`;
        edges.set(key, (edges.get(key) ?? 0) + 1);
      }
    }
    assert.ok(volume > 0, 'outward-wound positive volume; no mirrored inside-out ear');
    assert.ok([...edges.values()].every((count) => count === 2), 'each edge has two faces, including the back/rim');
    assert.equal(ear.material.side, THREE.FrontSide);
    assert.equal(ear.material.depthTest, true);
    assert.equal(ear.material.depthWrite, true);
    assert.equal(ear.scale.x, 1, 'no negative mirror scale');
  }
});
