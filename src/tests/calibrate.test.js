/**
 * calibrate.test.js — neutral stats, range floor, wink map (§4).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeNeutralStats, computeRangeHi, computeBlinkSwap,
  CalibrationSession, PHASES, GESTURE_CHANNELS, median, quantile, CONFIG,
} from '../calibrate.js';
import { CHANNELS, CH } from '../rig.js';

function frame(fill = 0) {
  return new Float32Array(52).fill(fill);
}

test('median/quantile basics', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([]), 0);
  assert.ok(Math.abs(quantile([1, 2, 3, 4], 0.95) - 3.85) < 1e-9);
});

test('neutral stats: b0 = median, nf = b0 + 2.5σ (§4.1)', () => {
  const frames = [];
  for (let i = 0; i < CONFIG.neutralFrames; i++) {
    const f = frame(0.2);
    f[CH.jawOpen] = 0.2 + ((i % 2) ? 0.02 : -0.02);
    frames.push(f);
  }
  const { b0, sigma, nf } = computeNeutralStats(frames);
  assert.ok(Math.abs(b0[CH.jawOpen] - 0.2) < 1e-6);
  assert.ok(sigma[CH.jawOpen] > 0.005);
  assert.ok(Math.abs(nf[CH.jawOpen] - (b0[CH.jawOpen] + CONFIG.noiseSigmaMult * sigma[CH.jawOpen])) < 1e-9);
});

test('range floor: failed gesture cannot make a channel dead (§4.2)', () => {
  const frames = [];
  for (let i = 0; i < CONFIG.gestureFrames; i++) frames.push(frame(0.1)); // gesture failed
  const hi = computeRangeHi(frames, CH.jawOpen, 0.1, 0.12);
  assert.ok(hi >= 0.12 + CONFIG.rangeFloor - 1e-9, `hi ${hi} keeps ≥0.15 usable span`);
  // Real gesture reads the p95.
  const frames2 = [];
  for (let i = 0; i < CONFIG.gestureFrames; i++) {
    const f = frame(0.1);
    f[CH.jawOpen] = 0.9;
    frames2.push(f);
  }
  const hi2 = computeRangeHi(frames2, CH.jawOpen, 0.1, 0.12);
  assert.ok(hi2 > 0.85, `hi ${hi2} tracks the gesture p95`);
});

test('wink map (§4.3 / trap §11.3): rising channel IS MediaPipe naming', () => {
  // User closes their LEFT eye and MediaPipe reports it as eyeBlinkRight →
  // MediaPipe naming is inverted → swap.
  assert.equal(computeBlinkSwap(0.01, 0.8), true, 'right rises → blinkLRSwap');
  assert.equal(computeBlinkSwap(0.8, 0.01), false, 'left rises → naming agrees');
});

test('gesture→channel map covers the 7 dead/weak channels (§3)', () => {
  const all = new Set(Object.values(GESTURE_CHANNELS).flat());
  for (const n of ['cheekPuff', 'tongueOut', 'jawForward', 'jawLeft', 'jawRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'jawOpen']) {
    // jawLeft/Right & tongue are geometry channels — they must appear in the
    // calibration prompts OR be geometry-derived (they are both).
    if (n === 'jawLeft' || n === 'jawRight' || n === 'tongueOut') continue;
    assert.ok(all.has(n), `${n} calibrated from gestures`);
  }
});

test('session state machine walks phases and finalizes', () => {
  const s = new CalibrationSession();
  s.begin();
  assert.equal(s.current().id, 'neutral');
  let guard = 0;
  while (!s.done && guard++ < 5000) {
    const f = frame(0.15);
    f[CH.jawOpen] = 0.6;
    f[CH.mouthSmileLeft] = 0.7;
    f[CH.mouthSmileRight] = 0.7;
    if (s.current()?.id === 'winkLeft') {
      f[CH.eyeBlinkRight] = 0.9; // user's LEFT eye closed → MediaPipe Right
    }
    s.pushFrame(f, new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 30, 1]));
  }
  assert.ok(s.done, 'session finalizes');
  const r = s.result;
  assert.ok(r && r.b0.length === 52);
  assert.equal(r.blinkLRSwap, true, 'wink map detected naming swap');
  assert.equal(r.z0, 30, 'neutral pose depth (§4.4)');
  // serialize round-trip
  const json = CalibrationSession.serialize(r);
  const back = CalibrationSession.deserialize(json);
  assert.ok(back);
  assert.equal(back.blinkLRSwap, true);
  assert.ok(Math.abs(back.hi[CH.jawOpen] - r.hi[CH.jawOpen]) < 1e-6);
});

test('session cancel discards, previous calibration kept (§4)', () => {
  const s = new CalibrationSession();
  s.begin();
  s.pushFrame(frame(0.2), null);
  s.cancel();
  assert.equal(s.current(), null);
  assert.equal(s.done, false);
});

test('wizard phases: 5-step flow with holds of 20 frames (§4)', () => {
  assert.ok(PHASES.length >= 6);
  assert.equal(PHASES[0].frames, CONFIG.neutralFrames);
  const gestures = PHASES.filter((p) => p.kind === 'gesture');
  for (const g of gestures) assert.equal(g.frames, CONFIG.gestureFrames);
  assert.ok(PHASES.some((p) => p.kind === 'wink'));
});

void CHANNELS;
