import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from '../../vendor/three.core.js';

export function readBundledGeometry() {
  const buffer = readFileSync(new URL('../../assets/anatomical-head/head.glb', import.meta.url));
  assert.equal(buffer.readUInt32LE(0), 0x46546c67, 'valid glTF binary');
  const jsonLength = buffer.readUInt32LE(12);
  const json = JSON.parse(buffer.subarray(20, 20 + jsonLength).toString());
  const start = 20 + jsonLength + 8;
  const primitive = json.meshes[0].primitives[0];
  const geometry = new THREE.BufferGeometry();
  for (const [key, name, size] of [['POSITION', 'position', 3], ['NORMAL', 'normal', 3], ['TEXCOORD_0', 'uv', 2]]) {
    const accessor = json.accessors[primitive.attributes[key]];
    const view = json.bufferViews[accessor.bufferView];
    const array = new Float32Array(buffer.buffer, buffer.byteOffset + start + (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0), accessor.count * size);
    geometry.setAttribute(name, new THREE.Float32BufferAttribute(array, size));
  }
  const accessor = json.accessors[primitive.indices];
  const view = json.bufferViews[accessor.bufferView];
  const array = new Uint16Array(buffer.buffer, buffer.byteOffset + start + view.byteOffset, accessor.count);
  geometry.setIndex(new THREE.Uint16BufferAttribute(array, 1));
  return geometry;
}
