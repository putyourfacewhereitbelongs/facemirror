/**
 * smoothing.test.js — One-Euro, DampedSpring, FixedStepper, hysteresis (§9.1).
 * Zero DOM/THREE imports.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  OneEuro, DampedSpring, FixedStepper, ChannelFilterBank,
  hystereticEx, alphaFor, CONFIG,
} from '../smoothing.js';
import { CHANNELS, GROUP_OF } from '../rig.js';

const DT = 1 / 60;

/** Deterministic PRNG (mulberry32). */
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('OneEuro reduces jitter at rest', () => {
  const r = rng(42);
  const f = new OneEuro(1.0, 0.01);
  const raw = [];
  const out = [];
  for (let i = 0; i < 600; i++) {
    const x = 1 + (r() - 0.5) * 0.1; // 0.05 amplitude noise floor
    raw.push(x);
    out.push(f.filter(x, DT));
  }
  const std = (xs) => {
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  };
  const rawStd = std(raw.slice(100));
  const outStd = std(out.slice(100));
  assert.ok(outStd < rawStd / 2, `filtered std ${outStd} should be < half raw ${rawStd}`);
});

test('OneEuro step-response lag stays under 40 ms (jaw-class filter)', () => {
  // Jaw group runs fcMin=6 Hz (§5.4) — responsive motion channels must not
  // smear: 50% rise time on a unit step < 40 ms at 60 Hz.
  const f = new OneEuro(6.0, 0.01);
  f.filter(0, DT);
  let tMs = 0;
  let lag = Infinity;
  for (let i = 1; i < 20; i++) {
    tMs = i * DT * 1000;
    const y = f.filter(1, DT); // unit step at t=0
    if (y >= 0.5 && lag === Infinity) lag = tMs;
  }
  assert.ok(lag < 40, `50% rise time ${lag} ms must be < 40 ms`);
});

test('OneEuro never blows up on dt spikes', () => {
  const f = new OneEuro(3.0, 0.01);
  let y = 0;
  for (const dt of [DT, DT, 0.25, DT, DT, 0.2, DT]) {
    y = f.filter(1, dt);
    assert.ok(Number.isFinite(y));
    assert.ok(y >= 0 && y <= 1.5);
  }
});

test('DampedSpring: critically damped, stable, settles without ringing', () => {
  const s = new DampedSpring(18, 1);
  let y = 0;
  let over = 0;
  for (let i = 0; i < 120; i++) {
    y = s.step(1, DT);
    if (y > 1.001) over++;
    assert.ok(Number.isFinite(y));
  }
  assert.equal(over, 0, 'ζ=1 must never overshoot');
  assert.ok(Math.abs(y - 1) < 1e-3, `settled ${y} ≈ 1`);
});

test('DampedSpring: under-damped emotion transition overshoots ~8%', () => {
  // §5.7: ζ=0.7, ω=2.5 → ≈8% overshoot in ≈0.8 s.
  const s = new DampedSpring(2.5, 0.7);
  let peak = 0;
  for (let i = 0; i < 120; i++) {
    const y = s.step(1, DT);
    if (y > peak) peak = y;
  }
  assert.ok(peak > 1.03 && peak < 1.15, `overshoot peak ${peak} in 3–15% band`);
});

test('FixedStepper produces uniform substeps and drops stall time', () => {
  const st = new FixedStepper(60);
  let total = 0;
  for (let i = 0; i < 30; i++) total += st.advance(1 / 60);
  assert.ok(Math.abs(total - 30) <= 1, `substeps ${total} ≈ 30`);
  const burst = st.advance(1.0); // 1 s stall
  assert.ok(burst <= CONFIG.maxSubsteps, 'never fast-forwards after a stall');
});

test('alphaFor matches the One-Euro paper α(dt, fc)', () => {
  const a = alphaFor(1, 1 / 60);
  const r = 2 * Math.PI * 1 / 60;
  assert.ok(Math.abs(a - r / (r + 1)) < 1e-12);
});

test('hysteresis band prevents binary flicker', () => {
  let state = false;
  ({ state } = hystereticEx(0.51, state, 0.5));
  assert.equal(state, true, 'triggers at threshold');
  ({ state } = hystereticEx(0.45, state, 0.5));
  assert.equal(state, true, 'holds through the ±10% band');
  ({ state } = hystereticEx(0.35, state, 0.5));
  assert.equal(state, false, 'releases below threshold−band');
});

test('ChannelFilterBank skips eyeBlink channels (trap §11.4)', () => {
  const bank = new ChannelFilterBank(CHANNELS, GROUP_OF);
  const raw = new Float32Array(52);
  const out = new Float32Array(52);
  const blinkL = CHANNELS.indexOf('eyeBlinkLeft');
  raw[blinkL] = 1;
  raw.fill(0.5);
  raw[blinkL] = 1;
  bank.apply(out, raw, DT);
  assert.equal(out[blinkL], 1, 'blink passes through unsmoothed');
  assert.ok(Math.abs(out[0] - 0.5) < 0.5, 'other channels filtered');
});
