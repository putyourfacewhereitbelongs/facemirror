/**
 * recorder.test.js — ring buffer, clip JSON, pure GLB writer + parse-back
 * (§9.2 "GLB export produces a parseable file, read back with GLTFLoader stub").
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  RingBuffer, Recorder, buildClipGlb, parseGlbJson, CONFIG,
} from '../recorder.js';
import { CHANNELS, Rig } from '../rig.js';

test('RingBuffer: FIFO order + wrap without reallocation', () => {
  const rb = new RingBuffer(10);
  for (let i = 0; i < 12; i++) {
    const f = new Float32Array(52).fill(i);
    rb.push(f, i / 30);
  }
  assert.equal(rb.count, 10);
  assert.equal(rb.wrapped, true);
  const out = new Float32Array(52);
  const t0 = rb.get(0, out);
  assert.equal(out[0], 2, 'oldest retained frame is #2');
  assert.equal(out[51], 2);
  void t0;
});

function makeClip(frameCount = 8) {
  const positions = new Float32Array(9 * 3);
  for (let i = 0; i < 9; i++) {
    positions[i * 3] = i * 0.1;
    positions[i * 3 + 1] = 0;
    positions[i * 3 + 2] = 0;
  }
  const morphDeltas = CHANNELS.map((_, k) => {
    const d = new Float32Array(9 * 3);
    d[k % 9] = 0.01 * (k + 1);
    return d;
  });
  const times = new Float32Array(frameCount);
  const weights = new Float32Array(frameCount * 52);
  for (let i = 0; i < frameCount; i++) {
    times[i] = i / 30;
    for (let c = 0; c < 52; c++) weights[i * 52 + c] = (c + i) % 52 / 52;
  }
  return { positions, morphDeltas, times, weights, frameCount };
}

test('buildClipGlb → parseGlbJson: valid glTF 2.0 with 52 morphs + weights animation', () => {
  const spec = makeClip(8);
  const glb = buildClipGlb({ ...spec, channels: CHANNELS });
  assert.ok(glb.byteLength > 1000, 'non-trivial file');
  const { json, binBytes } = parseGlbJson(glb);
  assert.equal(json.asset.version, '2.0');
  assert.equal(json.meshes[0].primitives[0].targets.length, 52, '52 morph targets');
  assert.equal(json.animations.length, 1);
  assert.equal(json.animations[0].channels[0].target.path, 'weights');
  assert.equal(json.animations[0].samplers[0].interpolation, 'LINEAR');
  // weights accessor: frameCount × 52 scalars
  const wAcc = json.accessors[json.animations[0].samplers[0].output];
  assert.equal(wAcc.count, 8 * 52);
  assert.equal(wAcc.componentType, 5126);
  // times accessor with min/max
  const tAcc = json.accessors[json.animations[0].samplers[0].input];
  assert.equal(tAcc.count, 8);
  assert.ok(Array.isArray(tAcc.min) && tAcc.min[0] === 0);
  assert.ok(binBytes > 0 && binBytes % 4 === 0, 'BIN chunk 4-byte aligned');
  // node has initial weights
  assert.equal(json.nodes[0].weights.length, 52);
});

test('buildClipGlb rejects nothing: GLB header/length consistent', () => {
  const glb = buildClipGlb({ ...makeClip(4), channels: CHANNELS });
  const dv = new DataView(glb);
  assert.equal(dv.getUint32(0, true), 0x46546C67, 'magic glTF');
  assert.equal(dv.getUint32(4, true), 2);
  assert.equal(dv.getUint32(8, true), glb.byteLength, 'header length matches file');
});

test('parseGlbJson throws on garbage', () => {
  assert.throws(() => parseGlbJson(new ArrayBuffer(16)));
});

test('Recorder: 30 fps capture ring + JSON export round-trips weights', () => {
  const rig = new Rig();
  const rec = new Recorder({ rig });
  rec.toggleRecord();
  // Simulate 2 s at 60 Hz substeps; recorder samples at 30 fps.
  for (let i = 0; i < 120; i++) {
    rig.weights[24] = i / 120; // jawOpen ramp
    rec.update(1 / 60, i / 60);
  }
  rec.capture = false;
  assert.ok(rec.ring.count >= 55 && rec.ring.count <= 65, `~60 frames recorded (${rec.ring.count})`);
  const json = JSON.parse(rec.exportJSON());
  assert.equal(json.fps, 30);
  assert.equal(json.channels.length, 52);
  assert.ok(Math.abs(json.frames[json.frames.length - 1][24] - 1) < 0.1, 'last frame carries the ramp');
});

test('Recorder playback feeds the record producer (§5.3)', () => {
  const rig = new Rig();
  const rec = new Recorder({ rig });
  rec.toggleRecord();
  for (let i = 0; i < 60; i++) {
    rig.weights[CH_IDX] = 0.7;
    rec.update(1 / 60, i / 60);
  }
  rec.capture = false;
  rec.togglePlay();
  for (let i = 0; i < 10; i++) rec.update(1 / 60, 2 + i / 60);
  assert.equal(rig.playbackActive, true);
  assert.ok(rig.weightsSource.record[CH_IDX] > 0.5);
  rec.togglePlay();
  assert.equal(rig.playbackActive, false);
});

const CH_IDX = CHANNELS.indexOf('mouthSmileLeft');

test('exportAnimGlb from a real recording parses (§9.2 smoke)', () => {
  const rig = new Rig();
  const rec = new Recorder({ rig });
  rec.toggleRecord();
  for (let i = 0; i < 90; i++) {
    rig.weights[0] = Math.sin(i / 10) * 0.5 + 0.5;
    rec.update(1 / 60, i / 60);
  }
  rec.capture = false;
  const spec = makeClip(3);
  const glb = rec.exportAnimGlb(spec.positions, spec.morphDeltas);
  const { json } = parseGlbJson(glb);
  assert.equal(json.animations[0].channels[0].target.path, 'weights');
  const wAcc = json.accessors[json.animations[0].samplers[0].output];
  assert.equal(wAcc.count, rec.ring.count * 52);
});

void CONFIG;
