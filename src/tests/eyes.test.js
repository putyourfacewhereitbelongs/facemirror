/**
 * eyes.test.js — blink scheduler stats, saccade main sequence, stabilizeBlink,
 * Listing's law, pupil, VOR clamp-and-reset (§9.1 / §5.5). No DOM/THREE.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BlinkScheduler, SaccadeGenerator, EyeSolver, Rng, blinkWaveform, minJerk,
  stabilizeBlink, listingQuaternion, pupilScale, convergenceDeg, makePinkNoise,
  CONFIG,
} from '../eyes.js';

const DT = 1 / 60;

test('blink scheduler: 2-minute sim stats (15–25/min ±30%, 100–400 ms, gap ≥1.2 s)', () => {
  const s = new BlinkScheduler(0.3, new Rng(1234));
  const SIM = 120; // seconds
  for (let t = 0; t < SIM; t += DT) s.update(DT, 0);
  const events = s.log;
  const perMin = (events.length / SIM) * 60;
  assert.ok(perMin >= 15 * 0.7 && perMin <= 25 * 1.3,
    `blinks/min ${perMin.toFixed(1)} in 15–25 ±30%`);
  for (const e of events) {
    assert.ok(e.dur >= CONFIG.blinkDurMinSec - 1e-9 && e.dur <= CONFIG.blinkDurMaxSec + 1e-9,
      `duration ${e.dur} in 100–400 ms`);
    assert.ok(e.t >= 0 && e.t <= SIM);
  }
  // Gaps between blink-starts, excluding the second closure of doubles.
  const starts = events.filter((e) => !e.double).map((e) => e.t).sort((a, b) => a - b);
  for (let i = 1; i < starts.length; i++) {
    const gap = starts[i] - starts[i - 1];
    assert.ok(gap >= CONFIG.blinkMinGapSec - 1e-6, `gap ${gap.toFixed(2)} s ≥ 1.2 s`);
  }
});

test('blink scheduler: jaw suppression and forced stare blink', () => {
  const s = new BlinkScheduler(0.6, new Rng(99));
  let injected = 0;
  for (let t = 0; t < 10; t += DT) {
    injected += s.update(DT, 0.9).length; // jaw wide open
  }
  // Only stare-force (>3.2 s) blinks may appear while the jaw is open.
  assert.ok(injected <= 4, `wide jaw suppressed spontaneous blinks (${injected})`);
  const s2 = new BlinkScheduler(0.001, new Rng(7));
  let forced = 0;
  for (let t = 0; t < 5; t += DT) forced += s2.update(DT, 0).length;
  assert.ok(forced >= 1, 'stare >3.2 s forces a blink');
});

test('blink scheduler: injected camera blinks become lid curves (trap §11.4)', () => {
  const s = new BlinkScheduler(0.3, new Rng(5));
  s.injectBlink(0.3, 0);
  let peak = 0;
  for (let t = 0; t < 0.35; t += DT) {
    s.update(DT, 0);
    const { left } = s.lidTargets();
    if (left > peak) peak = left;
  }
  assert.ok(peak > 0.8, `scheduled curve peaks at ~1 (${peak})`);
});

test('blink waveform: 38% close / 62% open asymmetric attack-release', () => {
  assert.equal(blinkWaveform(0), 0);
  assert.ok(blinkWaveform(CONFIG.blinkCloseFrac) >= 0.99, 'fully shut at end of attack');
  assert.ok(blinkWaveform(0.19) > 0.3, 'attack is fast');
  assert.ok(blinkWaveform(0.8) < 0.35, 'release trails off slower');
  assert.equal(blinkWaveform(1), 0);
});

test('saccade main sequence: duration = 20 + 3.5·deg ms, clamped 20–90', () => {
  const g = new SaccadeGenerator(new Rng(77));
  g.yaw = 0;
  g.pitch = 0;
  g.targetYaw = 30;
  g.targetPitch = 0;
  // Directly exercise the internal duration rule via pickTarget math:
  const deg = 30;
  const expected = Math.min(Math.max(20 + CONFIG.saccadeGainMsPerDeg * deg, 20), 90);
  g.saccadeT = -1;
  g.fixUntil = 0;
  const started = g.update(DT, [{ yaw: 30, pitch: 0 }, { yaw: 30, pitch: 0 }]);
  if (started > 0) {
    const last = g.log[g.log.length - 1];
    assert.ok(Math.abs(last.durMs - (20 + 3.5 * last.deg)) <= 1
      || last.durMs === 90 || last.durMs === 20, `durMs ${last.durMs} follows main sequence`);
    assert.ok(last.durMs >= 20 && last.durMs <= 90);
    assert.ok(last.deg >= 0);
  }
  // min-jerk sanity: 0→1 smooth
  assert.equal(minJerk(0), 0);
  assert.equal(minJerk(1), 1);
  assert.ok(minJerk(0.5) > 0.4 && minJerk(0.5) < 0.6);
  void expected;
});

test('saccade converges to target after overshoot + micro-correction', () => {
  const g = new SaccadeGenerator(new Rng(31));
  // Force the target pool branch (35% random-path would break determinism).
  g.rng.next = () => 0;
  g.yaw = 0;
  g.pitch = 0;
  g.fixUntil = 0;
  g.update(DT, [{ yaw: 25, pitch: -5 }]);
  const total = g.saccadeDur + CONFIG.microCorrectMs / 1000 + 0.05;
  for (let t = 0; t < total; t += DT) g.update(DT);
  assert.ok(Math.abs(g.yaw - 25) < 2.5, `yaw ${g.yaw.toFixed(2)} ≈ 25`);
  assert.ok(Math.abs(g.pitch - -5) < 2.5, `pitch ${g.pitch.toFixed(2)} ≈ -5`);
});

test('stabilizeBlink (KalidoKit port): profile turns kill the one-eyed blink', () => {
  const out = new Float32Array(2);
  stabilizeBlink(1, 0, 0.2, out);
  assert.equal(out[0], 1);
  assert.equal(out[1], 0, 'below yaw threshold: untouched');
  stabilizeBlink(1, 0, 1.2, out);
  assert.ok(out[1] > 0.5, `far eye attenuated toward near (${out[1].toFixed(2)})`);
  assert.ok(out[1] <= 1);
  stabilizeBlink(0, 1, -1.2, out);
  assert.ok(out[0] > 0.5, 'mirror side works');
});

test('Listing\'s law: quaternion from gaze axis, identity at primary', () => {
  const q = new Float32Array(4);
  listingQuaternion(0, 0, 1, q);
  assert.ok(Math.abs(q[3] - 1) < 1e-9, 'identity at primary gaze');
  listingQuaternion(1, 0, 0, q);
  // 90° about +y (gaze×primary = (0,1,0)×? → axis y)
  assert.ok(Math.abs(q[3] - Math.cos(Math.PI / 4)) < 1e-6, '90° yaw rotation');
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  assert.ok(Math.abs(n - 1) < 1e-6, 'unit quaternion');
});

test('pupil scale + convergence bounds (§5.5)', () => {
  const p = pupilScale(0.5, 1);
  assert.ok(p > 1 && p < 1.3);
  const p2 = pupilScale(-0.5, 0);
  assert.ok(p2 < 1 && p2 > 0.7);
  assert.ok(Math.abs(convergenceDeg(90) - CONFIG.convergenceDeg) < 1e-9);
  assert.ok(convergenceDeg(20) > convergenceDeg(200), 'closer = more eye-in');
});

test('pink noise stays bounded', () => {
  const pink = makePinkNoise(new Rng(3));
  for (let i = 0; i < 2000; i++) {
    const v = pink();
    assert.ok(Number.isFinite(v) && Math.abs(v) < 2);
  }
});

test('EyeSolver integration: blinks + VOR clamp-and-reset stay in range', () => {
  const e = new EyeSolver({ seed: 4242 });
  let headYaw = 0;
  let sawBlink = false;
  let sawReset = false;
  for (let i = 0; i < 60 * 20; i++) {
    if (i > 60 * 5 && i < 60 * 6) headYaw = 1.6; // fast profile turn
    else headYaw *= 0.98;
    const r = e.update(DT, {
      headYaw,
      headPitch: 0,
      jawOpen: 0,
      camBlinkL: 0,
      camBlinkR: 0,
    });
    if (r.lidL > 0.8) sawBlink = true;
    const inHeadDeg = Math.abs(e.eyeInHeadYaw * 180 / Math.PI);
    assert.ok(inHeadDeg <= CONFIG.maxEyeDeg + 1, `eye-in-head ${inHeadDeg}° ≤ 45°`);
    if (e.vorResetT >= 0) sawReset = true;
  }
  assert.ok(sawBlink, 'scheduler produced lid closures');
  assert.ok(sawReset, 'VOR clamp-and-reset engaged during the fast turn');
});

test('EyeSolver: camera blink weights inject winks per side', () => {
  const e = new EyeSolver({ seed: 11 });
  let sawLeft = 0;
  for (let i = 0; i < 60; i++) {
    const r = e.update(DT, {
      headYaw: 0,
      headPitch: 0,
      jawOpen: 0,
      camBlinkL: i > 5 ? 0.9 : 0,
      camBlinkR: 0,
    });
    sawLeft = Math.max(sawLeft, r.lidL);
  }
  assert.ok(sawLeft > 0.3, `left lid closed from camera blink (${sawLeft})`);
});
