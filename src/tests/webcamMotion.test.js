import test from 'node:test';
import assert from 'node:assert/strict';
import { Tracker } from '../tracker.js';
import { PoseTracker } from '../pose.js';
import { EyeSolver } from '../eyes.js';
import { Rig, CHANNELS, CH } from '../rig.js';
import { buildAnatomicalRig } from '../anatomy.js';
import { applyWeights } from '../headMotion.js';
import { readBundledGeometry } from './headFixture.js';
import * as THREE from '../../vendor/three.core.js';

function fakeStream() {
  const track = new EventTarget();
  track.readyState = 'live';
  track.stop = () => { track.readyState = 'ended'; };
  return { track, getTracks: () => [track], getVideoTracks: () => [track] };
}
function fixture(request) {
  const stream = fakeStream();
  const video = { srcObject: null, readyState: 2, currentTime: 1, play: async () => {} };
  const tracker = new Tracker({ requestStream: request ?? (async () => stream) });
  let closed = 0;
  tracker.initLandmarker = async () => { tracker.landmarker = { close: () => closed++, detectForVideo: () => ({ faceLandmarks: [] }) }; };
  return { tracker, video, stream, get closed() { return closed; } };
}

test('webcam startup requests permission immediately and coalesces rapid clicks', async () => {
  let calls = 0, grant;
  const stream = fakeStream();
  const f = fixture(() => { calls++; return new Promise((resolve) => { grant = resolve; }); });
  const a = f.tracker.start(f.video), b = f.tracker.start(f.video);
  assert.equal(calls, 1); assert.equal(a, b);
  assert.equal(f.tracker.running, false);
  grant(stream); await Promise.all([a, b]);
  assert.equal(f.tracker.running, true); assert.equal(f.video.srcObject, stream);
  f.tracker.stop();
});

test('stopping releases the physical camera, detector, stale expressions and video source', async () => {
  const f = fixture(); await f.tracker.start(f.video);
  f.tracker.weights[CH.mouthSmileLeft] = 0.9; f.tracker.hasFace = true;
  f.tracker.stop();
  assert.equal(f.stream.track.readyState, 'ended'); assert.equal(f.video.srcObject, null);
  assert.equal(f.tracker.running, false); assert.equal(f.tracker.hasFace, false);
  assert.equal(f.tracker.weights[CH.mouthSmileLeft], 0); assert.equal(f.closed, 1);
  assert.equal(f.tracker.poll(100, 1 / 60), false);
});

test('Stop during a pending camera permission cannot resurrect an obsolete stream', async () => {
  let grant;
  const stream = fakeStream(), f = fixture(() => new Promise((resolve) => { grant = resolve; }));
  const pending = f.tracker.start(f.video);
  f.tracker.stop(); grant(stream);
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(stream.track.readyState, 'ended'); assert.equal(f.tracker.stream, null);
  assert.equal(f.tracker.running, false);
});

test('Stop during model initialization closes the late detector as well as the camera', async () => {
  const f = fixture(); let finish, began;
  const started = new Promise((resolve) => { began = resolve; });
  let closed = 0;
  f.tracker.initLandmarker = async () => { began(); await new Promise((resolve) => { finish = resolve; }); f.tracker.landmarker = { close: () => closed++ }; };
  const pending = f.tracker.start(f.video); await started;
  f.tracker.stop(); finish();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(closed, 1); assert.equal(f.tracker.running, false); assert.equal(f.stream.track.readyState, 'ended');
});

test('camera permission denial is retryable', async () => {
  let deny = true;
  const stream = fakeStream(), f = fixture(async () => {
    if (deny) throw new DOMException('Denied', 'NotAllowedError');
    return stream;
  });
  await assert.rejects(f.tracker.start(f.video), { name: 'NotAllowedError' });
  assert.equal(f.tracker.running, false);
  deny = false; await f.tracker.start(f.video);
  assert.equal(f.tracker.running, true); f.tracker.stop();
});

test('camera disconnection notifies the app and clears the motion source', async () => {
  const f = fixture(); let ended = false;
  f.tracker.onEnded = () => { ended = true; };
  await f.tracker.start(f.video); f.stream.track.dispatchEvent(new Event('ended'));
  assert.equal(ended, true); assert.equal(f.tracker.running, false); assert.equal(f.video.srcObject, null);
});

test('a missing face or blendshape result cannot leave stale driver expressions behind', async () => {
  const f = fixture(); await f.tracker.start(f.video);
  f.tracker.weights[CH.jawOpen] = 0.8;
  assert.equal(f.tracker.poll(100, 1 / 60), true); assert.equal(f.tracker.hasFace, false);
  assert.equal(f.tracker.weights[CH.jawOpen], 0); assert.deepEqual(f.tracker.landmarks, []);
  f.tracker.weights[CH.jawOpen] = 0.9;
  f.tracker.landmarker.detectForVideo = () => ({ faceLandmarks: [Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }))] });
  f.video.currentTime++;
  f.tracker.poll(200, 1 / 60);
  assert.equal(f.tracker.hasFace, true); assert.equal(f.tracker.weights[CH.jawOpen], 0);
  f.tracker.stop();
});

test('stopping tracking eases the avatar back to neutral without replacing its head', () => {
  const pose = new PoseTracker(), matrix = new THREE.Matrix4().makeRotationY(0.5).elements;
  matrix[12] = 4; matrix[13] = 3; matrix[14] = -40;
  pose.setFromMatrix(matrix, 0);
  for (let i = 0; i < 90; i++) pose.update(1 / 60, i / 60);
  assert.ok(Math.abs(pose.quat[1]) > 0.1);
  pose.resetTracking();
  assert.equal(pose._lagQueue.length, 0);
  for (let i = 0; i < 120; i++) pose.update(1 / 60, 2 + i / 60);
  assert.ok(Math.abs(pose.quat[1]) < 0.001); assert.ok(Math.abs(pose.pos[0]) < 0.001);
});

test('live webcam eyelids hold closed and open on release, rather than playing a one-shot blink', () => {
  const eyes = new EyeSolver({ seed: 30 });
  eyes.toggles.saccades = eyes.toggles.jitter = eyes.toggles.vor = false;
  let result;
  for (let i = 0; i < 120; i++) result = eyes.update(1 / 60, { headYaw: 0, headPitch: 0, jawOpen: 0, cameraTracked: true, camBlinkL: 1, camBlinkR: 1 });
  assert.equal(result.lidL, 1); assert.equal(result.lidR, 1);
  result = eyes.update(1 / 60, { headYaw: 0, headPitch: 0, jawOpen: 0, cameraTracked: true, camBlinkL: 0, camBlinkR: 0 });
  assert.ok(result.lidL < 0.05 && result.lidR < 0.05);
});

test('live webcam supports a sustained wink without forcing the other image eye closed', () => {
  const eyes = new EyeSolver({ seed: 30 });
  eyes.toggles.saccades = eyes.toggles.jitter = eyes.toggles.vor = false;
  let result;
  for (let i = 0; i < 80; i++) result = eyes.update(1 / 60, { headYaw: 0, headPitch: 0, jawOpen: 0, cameraTracked: true, camBlinkL: 0.95, camBlinkR: 0 });
  assert.ok(result.lidL > 0.94); assert.ok(result.lidR < 0.1);
});

test('webcam expression/pose data drives an existing image rig without mutating its identity or texture', () => {
  const rig = new Rig(), pose = new PoseTracker(), weights = new Float32Array(CHANNELS.length);
  const material = new THREE.MeshPhysicalMaterial({ map: new THREE.DataTexture(new Uint8Array([160, 110, 80, 255]), 1, 1) });
  const source = readBundledGeometry(), image = buildAnatomicalRig(source, material, new THREE.Texture());
  image.likeness = { kind: 'image', label: 'Portrait' };
  const texture = material.map, before = image.meshes[0].geometry.attributes.position.array.slice();
  weights[CH.mouthSmileLeft] = weights[CH.mouthSmileRight] = 0.7;
  weights[CH.jawOpen] = 0.4; weights[CH.browInnerUp] = 0.5;
  const matrix = new THREE.Matrix4().makeRotationY(0.4).elements;
  rig.feedCamera(weights); pose.setFromMatrix(matrix, 0);
  for (let i = 0; i < 90; i++) { rig.update(1 / 60); pose.update(1 / 60, i / 60); }
  assert.ok(rig.weights[CH.mouthSmileLeft] > 0.5); assert.ok(rig.weights[CH.jawOpen] > 0.1);
  applyWeights(image, rig.weights); image.root.quaternion.fromArray(pose.quat);
  assert.ok(Math.abs(image.root.quaternion.y) > 0.1);
  assert.ok(image.meshes[0].morphTargetInfluences[CH.mouthSmileLeft] > 0.5);
  assert.ok(image.meshes[0].morphTargetInfluences[CH.jawOpen] > 0.1);
  assert.equal(material.map, texture); assert.equal(image.likeness.label, 'Portrait');
  assert.deepEqual(image.meshes[0].geometry.attributes.position.array, before);
  image.root.traverse((o) => { if (o.isMesh) o.geometry.dispose(); }); source.dispose();
});

test('a restart queued after Stop waits for the cancelled permission request to dispose', async () => {
  const first = fakeStream(), second = fakeStream();
  let grant, calls = 0;
  const f = fixture(() => ++calls === 1 ? new Promise((resolve) => { grant = resolve; }) : Promise.resolve(second));
  const obsolete = f.tracker.start(f.video);
  f.tracker.stop();
  const resumed = f.tracker.start(f.video);
  grant(first);
  await assert.rejects(obsolete, { name: 'AbortError' });
  await resumed;
  assert.equal(calls, 2); assert.equal(first.track.readyState, 'ended');
  assert.equal(second.track.readyState, 'live'); assert.equal(f.tracker.running, true);
  assert.equal(f.video.srcObject, second); f.tracker.stop();
});

test('a camera that disconnects during startup is not reported as a live driver', async () => {
  const f = fixture();
  f.video.play = async () => { f.stream.track.stop(); };
  await assert.rejects(f.tracker.start(f.video), /disconnected/);
  assert.equal(f.tracker.running, false); assert.equal(f.video.srcObject, null);
  assert.equal(f.closed, 1);
});

test('video playback failure cleans up the granted camera stream', async () => {
  const f = fixture();
  f.video.play = async () => { throw new DOMException('Playback failed', 'NotSupportedError'); };
  await assert.rejects(f.tracker.start(f.video), { name: 'NotSupportedError' });
  assert.equal(f.stream.track.readyState, 'ended'); assert.equal(f.video.srcObject, null);
  assert.equal(f.tracker.running, false);
});

test('idle releases stale camera expressions while preserving manual pins', () => {
  const rig = new Rig(), weights = new Float32Array(CHANNELS.length);
  weights[CH.jawOpen] = 0.7; rig.feedCamera(weights);
  rig.setManual('browInnerUp', 0.6); rig.setPin('browInnerUp', true);
  for (let i = 0; i < 60; i++) rig.update(1 / 60);
  assert.ok(rig.weights[CH.jawOpen] > 0.1);
  rig.goIdle();
  assert.equal(rig.weightsSource.camRaw[CH.jawOpen], 0);
  for (let i = 0; i < 180; i++) rig.update(1 / 60);
  assert.ok(rig.weights[CH.jawOpen] < 0.02);
  assert.ok(Math.abs(rig.weights[CH.browInnerUp] - 0.6) < 1e-5);
});
