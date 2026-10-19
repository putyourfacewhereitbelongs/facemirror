/**
 * emotions.test.js — cosine-similarity read-back + presets (§9.1 / §5.7).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PRESETS, EMOTION_ORDER, EmotionReadback, EmotionController,
  presetVector, cosineSimilarity, EYE_MOUTH_PRESERVE,
} from '../emotions.js';
import { CHANNELS } from '../rig.js';

/** uniform noise in ±amp */
function noisy(v, amp, seed) {
  let s = seed >>> 0;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const n = ((s / 4294967296) * 2 - 1) * amp;
    out[i] = Math.max(0, v[i] + n);
  }
  return out;
}

test('presets: 7 FACS vectors, bilateral pairs equal except contempt', () => {
  assert.equal(EMOTION_ORDER.length, 7);
  for (const id of EMOTION_ORDER) {
    const p = PRESETS[id];
    assert.ok(Object.keys(p).length > 0, `${id} has weights`);
  }
  // joy symmetric
  assert.equal(PRESETS.joy.mouthSmileLeft, PRESETS.joy.mouthSmileRight);
  // contempt LEFT ONLY (§5.7) — its asymmetry is the signature
  assert.equal(PRESETS.contempt.mouthSmileLeft, 0.35);
  assert.equal(PRESETS.contempt.mouthSmileRight, undefined);
  assert.equal(PRESETS.contempt.mouthDimpleLeft, 0.3);
  assert.equal(EYE_MOUTH_PRESERVE.joy, 0.8);
  assert.equal(EYE_MOUTH_PRESERVE.fear, 0.35);
});

test('classifier: each preset at 0.6–1.0 intensity + noise 0.05 → ≥6/7 correct', () => {
  const rb = new EmotionReadback(CHANNELS);
  for (let trial = 0; trial < 10; trial++) {
    let correct = 0;
    for (const id of EMOTION_ORDER) {
      const intensity = 0.6 + ((trial * 7 + id.length) % 5) * 0.1; // 0.6–1.0
      const v = presetVector(id, CHANNELS);
      for (let i = 0; i < v.length; i++) v[i] *= intensity;
      const noisyV = noisy(v, 0.05, 1000 + trial * 13 + id.length);
      const got = rb.classifyOnce(noisyV);
      if (got === id) correct++;
    }
    assert.ok(correct >= 6, `trial ${trial}: ${correct}/7 correct (need ≥6)`);
  }
});

test('classifier: contempt ≠ joy (§9.1 explicit regression)', () => {
  const rb = new EmotionReadback(CHANNELS);
  const v = presetVector('contempt', CHANNELS);
  const got = rb.classifyOnce(noisy(v, 0.05, 77));
  assert.notEqual(got, 'joy');
  assert.equal(got, 'contempt', 'asymmetric smirk must read as contempt');
});

test('classifier: clean vectors classify 7/7', () => {
  const rb = new EmotionReadback(CHANNELS);
  for (const id of EMOTION_ORDER) {
    assert.equal(rb.classifyOnce(presetVector(id, CHANNELS)), id);
  }
});

test('read-back EMA + hysteresis holds labels briefly', () => {
  const rb = new EmotionReadback(CHANNELS);
  const joy = presetVector('joy', CHANNELS);
  let label = 'none';
  for (let i = 0; i < 30; i++) label = rb.update(joy, 1 / 30).label;
  assert.equal(label, 'joy');
  // Sudden swap: label should lag by ≥0.25 s (hysteresis).
  const anger = presetVector('anger', CHANNELS);
  let swapped = false;
  let t = 0;
  for (let i = 0; i < 20; i++) {
    const r = rb.update(anger, 1 / 30);
    t += 1 / 30;
    if (r.label === 'anger') swapped = true;
  }
  assert.ok(swapped, 'eventually switches');
});

test('EmotionController: spring layer rises then settles (8% overshoot band)', () => {
  const ec = new EmotionController(CHANNELS);
  ec.setEmotion('surprise');
  ec.setIntensity(1);
  let peak = 0;
  const idx = CHANNELS.indexOf('jawOpen');
  for (let i = 0; i < 120; i++) {
    ec.update(1 / 60);
    if (ec.layer[idx] > peak) peak = ec.layer[idx];
  }
  assert.ok(peak > 0.6 * 0.5, `layer reached ${peak}`);
  assert.ok(peak < 0.6 * 1.25, `overshoot bounded (${peak})`);
  // after 3 s the value is near target
  for (let i = 0; i < 180; i++) ec.update(1 / 60);
  assert.ok(Math.abs(ec.layer[idx] - 0.6) < 0.12, `settled ${ec.layer[idx]} ≈ 0.6`);
});

test('cosineSimilarity basics', () => {
  assert.ok(Math.abs(cosineSimilarity([1, 0], [1, 0]) - 1) < 1e-9);
  assert.ok(Math.abs(cosineSimilarity([1, 0], [0, 1])) < 1e-9);
  assert.equal(cosineSimilarity([0, 0], [1, 1]), 0);
});
