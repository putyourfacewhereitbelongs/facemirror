/**
 * rig.test.js — §5.1 mapping, §5.3 merge/pinning, §5.6 mouth extras,
 * side-swap remap (trap §11.3).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  Rig, CHANNELS, CH, normalizeChannel, symmetrize, gainOf,
  applyMouthExtras, BILATERAL_PAIRS, makeSideSwapMap,
} from '../rig.js';

test('normalizeChannel implements (v−nf)/(hi−nf) with clamp (§5.1)', () => {
  assert.equal(normalizeChannel(0.2, 0.1, 0.6), 0.2, '(0.2−0.1)/(0.6−0.1)=0.2');
  assert.equal(normalizeChannel(0.05, 0.1, 0.6), 0, 'below floor clamps to 0');
  assert.equal(normalizeChannel(0.9, 0.1, 0.6), 1, 'above hi clamps to 1');
  assert.equal(normalizeChannel(0.1, 0.1, 0.1), 0, 'degenerate span safe');
});

test('symmetrize: default blend is avg·0.75 + own·0.25; asym=1 keeps raw', () => {
  const w = new Float32Array(52);
  const [li, ri] = BILATERAL_PAIRS[0];
  w[li] = 1;
  w[ri] = 0;
  symmetrize(w, 0);
  assert.ok(Math.abs(w[li] - 0.625) < 1e-6, `L ${w[li]} = 0.5*0.75+1*0.25`);
  assert.ok(Math.abs(w[ri] - 0.375) < 1e-6, `R ${w[ri]} = 0.5*0.75+0*0.25`);

  const w2 = new Float32Array(52);
  w2[li] = 1;
  w2[ri] = 0;
  symmetrize(w2, 1);
  assert.equal(w2[li], 1);
  assert.equal(w2[ri], 0);
});

test('gain table (§5.1 defaults)', () => {
  assert.equal(gainOf('jawOpen'), 1.35);
  assert.equal(gainOf('mouthSmileLeft'), 1.45);
  assert.equal(gainOf('eyeBlinkRight'), 1.15);
  assert.equal(gainOf('browInnerUp'), 1.5);
  assert.equal(gainOf('tongueOut'), 1.0);
});

test('rig.feedCamera runs the full §5.1 mapping', () => {
  const rig = new Rig();
  // Calibrate: nf=0, hi=1 → n = v.
  rig.applyCalibration({
    b0: new Float32Array(52),
    nf: new Float32Array(52),
    hi: new Float32Array(52).fill(1),
    blinkLRSwap: false,
    z0: 0,
  });
  const raw = new Float32Array(52);
  raw[CH.jawOpen] = 0.5;
  rig.feedCamera(raw);
  // symmetrize touches pairs only; jawOpen is unpaired → exact gain math.
  assert.ok(Math.abs(rig.weightsSource.cam[CH.jawOpen] - 0.5 * 1.35) < 1e-6
    || rig.weightsSource.cam[CH.jawOpen] === 1, 'clamped or gained');
  // eyeBlink must NOT be driven from camera (trap §11.4)
  raw[CH.eyeBlinkLeft] = 1;
  rig.feedCamera(raw);
  assert.equal(rig.weightsSource.cam[CH.eyeBlinkLeft], 0);
});

test('producer merge + pinning (§5.3)', () => {
  const rig = new Rig();
  rig.applyCalibration({
    b0: new Float32Array(52),
    nf: new Float32Array(52),
    hi: new Float32Array(52).fill(1),
    blinkLRSwap: false,
    z0: 0,
  });
  const raw = new Float32Array(52);
  raw[CH.mouthSmileLeft] = 0.4;
  raw[CH.mouthSmileRight] = 0.4;
  rig.feedCamera(raw);
  rig.weightsSource.emotion[CH.browInnerUp] = 0.5;
  rig.setMouthPreserve(0.5);
  rig.update(1 / 60);
  // cam path through merge: 0.4*1.45=0.58; symmetrize keeps pair equal.
  assert.ok(rig.weights[CH.mouthSmileLeft] > 0.4, `smile gained ${rig.weights[CH.mouthSmileLeft]}`);
  // emotion max-composite
  assert.ok(rig.weights[CH.browInnerUp] >= 0.5, 'emotion pushes brow up');
  // pin override (Float32Array rounds — compare with tolerance)
  rig.setManual('mouthSmileLeft', 0.11);
  rig.setPin('mouthSmileLeft', true);
  rig.update(1 / 60);
  assert.ok(Math.abs(rig.weights[CH.mouthSmileLeft] - 0.11) < 1e-6, 'pinned channel outputs manual');
});

test('dragAdd wins while dragging (§5.3)', () => {
  const rig = new Rig();
  rig.addDrag('mouthSmileLeft', 0.3);
  rig.update(1 / 60);
  assert.ok(rig.weights[CH.mouthSmileLeft] >= 0.3);
});

test('mouth extras: jawOpen ∩ mouthClose attenuation + no-teeth cap (§5.6/§11.8)', () => {
  const w = new Float32Array(52);
  w[CH.jawOpen] = 0.8;
  w[CH.mouthClose] = 0.5;
  applyMouthExtras(w, true);
  assert.ok(w[CH.jawOpen] < 0.8, 'open attenuated by close');
  assert.ok(w[CH.mouthClose] < 0.5, 'close attenuated by open');

  const w2 = new Float32Array(52);
  w2[CH.jawOpen] = 0.95;
  applyMouthExtras(w2, false);
  assert.ok(Math.abs(w2[CH.jawOpen] - 0.65) < 1e-6, 'jawOpen capped without teeth (trap §11.8)');
});

test('blinkLRSwap remaps every bilateral pair at load (trap §11.3)', () => {
  const map = makeSideSwapMap(true);
  assert.equal(map.mouthSmileLeft, 'mouthSmileRight');
  assert.equal(map.eyeBlinkLeft, 'eyeBlinkRight');
  const rig = new Rig();
  rig.applyCalibration({
    b0: new Float32Array(52),
    nf: new Float32Array(52),
    hi: new Float32Array(52).fill(1),
    blinkLRSwap: true,
    z0: 0,
  });
  const raw = new Float32Array(52);
  raw[CH.mouthSmileRight] = 1; // MediaPipe's Right is the user's Left
  rig.asym = 1; // keep sides independent for the assertion
  rig.feedCamera(raw);
  assert.ok(rig.weightsSource.cam[CH.mouthSmileLeft] > 0.5,
    'swapped naming: MediaPipe Right lands on rig Left');
});

test('mic jaw/viseme merges onto mouth/jaw only and respects pins', () => {
  const rig = new Rig();
  rig.resetAll();
  rig.feedMic(0.8, { a: 0.5, e: 0.6, i: 0, o: 0.4, u: 0 });
  rig.weightsSource.cam.fill(0);
  rig.weightsSource.emotion.fill(0);
  rig.weightsSource.eyes.fill(0);
  rig.update(1 / 60);
  const out = rig.weights;
  assert.ok(out[CH.jawOpen] > 0.5, `mic jaw merged (${out[CH.jawOpen]})`);
  assert.ok(out[CH.mouthFunnel] > 0.2, `viseme O merged (${out[CH.mouthFunnel]})`);
  assert.equal(out[CH.browDownLeft], 0, 'mic never drives brows');
  // Pin wins over mic (§5.3).
  rig.setManual('mouthFunnel', 0.1);
  rig.setPin('mouthFunnel', true);
  rig.update(1 / 60);
  assert.ok(Math.abs(rig.weights[CH.mouthFunnel] - 0.1) < 1e-6, 'pin overrides mic');
});

test('goIdle settles exponentially (§2), wake restores', () => {
  const rig = new Rig();
  rig.weights[CH.jawOpen] = 1;
  rig._settle[CH.jawOpen] = 1;
  rig.goIdle();
  for (let i = 0; i < 120; i++) rig.update(1 / 60);
  assert.ok(rig.idleBlend < 0.05, 'idle blend decays');
  rig.wake();
  assert.equal(rig.idleBlend, 0);
});

test('snapshot/restore undo path (§6)', () => {
  const rig = new Rig();
  rig.addDrag('jawOpen', 0.4);
  rig.update(1 / 60);
  const snap = rig.snapshot();
  rig.addDrag('jawOpen', 0.4);
  rig.update(1 / 60);
  assert.ok(rig.weights[CH.jawOpen] >= 0.4);
  rig.restore(snap);
  assert.ok(Math.abs(rig.weights[CH.jawOpen] - snap[CH.jawOpen]) < 1e-9);
});

test('CHANNELS is exactly the 52 ARKit names (§6 contract)', () => {
  assert.equal(CHANNELS.length, 52);
  assert.equal(new Set(CHANNELS).size, 52, 'unique');
  for (const n of [
    'browDownLeft', 'cheekPuff', 'eyeBlinkLeft', 'eyeLookUpRight', 'jawForward',
    'mouthSmileLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'tongueOut',
  ]) {
    assert.ok(CHANNELS.includes(n), `${n} present`);
  }
});
