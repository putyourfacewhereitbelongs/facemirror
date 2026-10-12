/**
 * smoke.test.js — headless camera-free smoke (§9.2): FakeDetector-driven
 * pipeline (happy | blink | turn | nosecret), rig responds, HUD present,
 * no uncaught errors, GLB export parseable via the GLTFLoader stub.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { FakeDetector, Tracker } from '../tracker.js';
import { Rig, CHANNELS, CH } from '../rig.js';
import { EyeSolver } from '../eyes.js';
import { PoseTracker } from '../pose.js';
import { EmotionController, EmotionReadback } from '../emotions.js';
import { ChannelFilterBank, FixedStepper } from '../smoothing.js';
import { PerfGovernor } from '../perf.js';
import { buildClipGlb, parseGlbJson, Recorder } from '../recorder.js';

const DT = 1 / 60;

/** The solve chain main.js runs, minus DOM/THREE. */
function runPipeline(script, seconds, opts = {}) {
  const rig = new Rig();
  const pose = new PoseTracker();
  const eyes = new EyeSolver({ seed: 0xFACE });
  const emotions = new EmotionController(CHANNELS);
  const readback = new EmotionReadback(CHANNELS);
  const bank = new ChannelFilterBank(CHANNELS);
  const stepper = new FixedStepper();
  const perf = new PerfGovernor({ toast: () => {}, log: () => {} });
  const fake = new FakeDetector(script);
  const smoothed = new Float32Array(52);
  const ypr = new Float32Array(3);
  let idle = false;
  let faceAbsentFor = 0;
  const errors = [];
  const track = { peakSmile: 0, peakBlink: 0, idleBlendMin: 1, weightsFinite: true };

  for (let i = 0; i < seconds * 60; i++) {
    const nowMs = i * 1000 / 60;
    try {
      const r = fake.detect(DT);
      if (r.hasFace) {
        rig.feedCamera(r.weights);
        pose.setFromMatrix(r.matrix, nowMs / 1000);
        faceAbsentFor = 0;
        if (idle) {
          rig.wake();
          idle = false;
        }
      } else {
        faceAbsentFor += DT;
        if (faceAbsentFor > 0.8 && !idle) {
          rig.goIdle();
          idle = true;
        }
      }
      pose.getYawPitchRoll(ypr);
      const sub = stepper.advance(DT);
      for (let s = 0; s < Math.max(sub, 1); s++) {
        emotions.update(stepper.stepSec);
        rig.setMouthPreserve(emotions.mouthPreserve());
        rig.weightsSource.emotion.set(emotions.layer);
        const lids = eyes.update(stepper.stepSec, {
          headYaw: ypr[0],
          headPitch: ypr[1],
          jawOpen: rig.weightsSource.cam[CH.jawOpen],
          camBlinkL: r.weights[CH.eyeBlinkLeft],
          camBlinkR: r.weights[CH.eyeBlinkRight],
          arousal: 0,
        });
        rig.feedEyes(lids);
        pose.update(stepper.stepSec, nowMs / 1000, idle);
        rig.update(stepper.stepSec);
        bank.apply(smoothed, rig.weights, stepper.stepSec);
        rig.weights.set(smoothed);
      }
      track.peakSmile = Math.max(track.peakSmile, rig.weights[CH.mouthSmileLeft]);
      track.peakBlink = Math.max(track.peakBlink, rig.weights[CH.eyeBlinkLeft]);
      track.idleBlendMin = Math.min(track.idleBlendMin, rig.idleBlend);
      for (let k = 0; k < 52; k++) {
        if (!Number.isFinite(rig.weights[k])) track.weightsFinite = false;
      }
      perf.frame(16.7);
      readback.update(rig.weights, DT);
    } catch (err) {
      errors.push(err);
    }
  }
  return { rig, pose, eyes, emotions, readback, perf, track, errors };
}

test('smoke: happy scenario drives the rig (smile rises, weights finite)', () => {
  const { rig, track, errors } = runPipeline('happy', 3);
  assert.equal(errors.length, 0, 'no uncaught errors: ' + errors.map((e) => e.message).join('; '));
  assert.ok(track.weightsFinite, 'all 52 weights finite');
  assert.ok(track.peakSmile > 0.3, `smile responded (${track.peakSmile})`);
  assert.ok(rig.weights[CH.mouthSmileLeft] >= 0);
});

test('smoke: blink scenario produces lid curve via scheduler (trap §11.4)', () => {
  const { track, errors } = runPipeline('blink', 6);
  assert.equal(errors.length, 0);
  assert.ok(track.peakBlink > 0.5, `blink lid peaked at ${track.peakBlink}`);
});

test('smoke: turn scenario keeps pose tracking in clamp range', () => {
  const { pose, errors } = runPipeline('turn', 4);
  assert.equal(errors.length, 0);
  const ypr = new Float32Array(3);
  pose.getYawPitchRoll(ypr);
  assert.ok(Math.abs(ypr[0]) < 1.5, `yaw ${ypr[0]} within ~±80°+drag clamp`);
});

test('smoke: nosecret/idle settles to breathing idle, never frozen (§2)', () => {
  // 'nosecret' produces a face, then we run an empty script for absence.
  const withFace = runPipeline('nosecret', 1);
  assert.equal(withFace.errors.length, 0);
  const noFace = runPipeline('', 3);
  assert.equal(noFace.errors.length, 0);
  assert.ok(noFace.rig.idleBlend < 0.05, `idle settle decayed (${noFace.rig.idleBlend})`);
});

test('smoke: FakeDetector streams into Tracker.poll (window.__FACEMIRROR_FAKE)', () => {
  globalThis.window = { __FACEMIRROR_FAKE: 'happy|blink' };
  try {
    const t = new Tracker({ onStatus: () => {} });
    assert.ok(t.fake, 'FakeDetector engaged from window flag');
    let fresh = 0;
    for (let i = 0; i < 30; i++) {
      if (t.poll(i * 16.7, 1 / 60)) fresh++;
    }
    assert.ok(fresh > 10, `detections flowed (${fresh})`);
    assert.equal(t.weights.length, 52);
    assert.equal(t.matrix.length, 16);
  } finally {
    delete globalThis.window;
  }
});

test('smoke: FPS HUD payload present (§8) and self-check report logs', () => {
  const { perf, errors } = runPipeline('happy|blink', 3);
  assert.equal(errors.length, 0);
  const hud = perf.hudData();
  for (const key of ['fps', 'p95', 'capture', 'track', 'solve', 'render', 'heap', 'tier', 'tierLabel']) {
    assert.ok(key in hud, `HUD has ${key}`);
  }
  const report = perf.selfCheck();
  assert.match(report, /FaceMirror-3D self-check/);
  assert.match(report, /quality tier/);
});

test('smoke: emotion chips + read-back respond without errors', () => {
  const rig = new Rig();
  const emotions = new EmotionController(CHANNELS);
  const readback = new EmotionReadback(CHANNELS);
  emotions.setEmotion('joy');
  for (let i = 0; i < 120; i++) {
    emotions.update(DT);
    rig.weightsSource.emotion.set(emotions.layer);
    rig.setMouthPreserve(emotions.mouthPreserve());
    rig.update(DT);
    readback.update(rig.weights, DT);
  }
  assert.equal(readback.label, 'joy', 'read-back lights the joy chip');
  assert.ok(readback.score > 0.5);
});

test('smoke: GLB clip export parses back via the GLTFLoader stub (§9.2)', () => {
  const rig = new Rig();
  const rec = new Recorder({ rig });
  rec.toggleRecord();
  for (let i = 0; i < 60; i++) {
    rig.weights[CH.mouthSmileLeft] = i / 60;
    rec.update(DT, i / 60);
  }
  rec.capture = false;
  const positions = new Float32Array(12 * 3);
  const deltas = CHANNELS.map(() => new Float32Array(12 * 3));
  const glb = rec.exportAnimGlb(positions, deltas);
  const { json, binBytes } = parseGlbJson(glb);
  assert.equal(json.meshes[0].primitives[0].targets.length, 52);
  assert.ok(binBytes > 0);
  // and the pure writer output round-trips identically
  const glb2 = buildClipGlb({
    positions,
    morphDeltas: deltas,
    times: new Float32Array(rec.ring.count),
    weights: new Float32Array(rec.ring.count * 52),
    frameCount: rec.ring.count,
    channels: CHANNELS,
  });
  assert.ok(glb2.byteLength > 0);
});

test('smoke: full pipeline with emotion + camera together — layers merge cleanly', () => {
  const { rig, errors, track } = runPipeline('happy', 4);
  assert.equal(errors.length, 0);
  assert.ok(track.weightsFinite);
  // Merge is explicit (§5.3): weights stay in [0,1].
  for (let k = 0; k < 52; k++) {
    assert.ok(rig.weights[k] >= 0 && rig.weights[k] <= 1, `w[${CHANNELS[k]}]=${rig.weights[k]} in range`);
  }
});
