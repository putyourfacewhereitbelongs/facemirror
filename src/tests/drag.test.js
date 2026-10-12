/**
 * drag.test.js — Jacobian projection solver convergence (§9.1 / §6),
 * ridge fallback, undo stack, name normalization basics.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeJacobian, makeJacobian, solveJacobianStep, solveRidgeFull,
  UndoStack, assetHash, CONTROLS, CONFIG,
} from '../drag.js';
import { CHANNELS } from '../rig.js';

/**
 * Synthetic 3-channel "smile" system with near-orthogonal unit Jacobian
 * columns (the fast path assumes localized, near-orthogonal morphs — §6):
 * w0 pushes +x, w1 pushes +y, w2 pushes +z, with a little cross-coupling.
 */
function syntheticSample(weights, out) {
  out[0] = weights[0] + 0.15 * weights[1];
  out[1] = 0.15 * weights[0] + weights[1];
  out[2] = weights[2];
}

test('solveJacobianStep: known 3-channel smile offset converges to ±0.1 in <30 iters', () => {
  const J = makeJacobian(1, 52);
  computeJacobian(J, 52, syntheticSample);
  const wStar = new Float32Array(52);
  wStar[0] = 0.55;
  wStar[1] = 0.35;
  wStar[2] = 0.45;
  const target = new Float32Array(3);
  syntheticSample(wStar, target);

  const w = new Float32Array(52);
  let iters = 0;
  for (iters = 0; iters < 30; iters++) {
    const cur = new Float32Array(3);
    syntheticSample(w, cur);
    const dx = target[0] - cur[0];
    const dy = target[1] - cur[1];
    const dz = target[2] - cur[2];
    if (Math.abs(dx) < 0.02 && Math.abs(dy) < 0.02) break;
    solveJacobianStep(J, 0, dx, dy, dz, w, 52);
  }
  assert.ok(iters < 30, `converged in ${iters} iters`);
  assert.ok(Math.abs(w[0] - wStar[0]) < 0.1, `w0 ${w[0]} ≈ ${wStar[0]}`);
  assert.ok(Math.abs(w[1] - wStar[1]) < 0.1, `w1 ${w[1]} ≈ ${wStar[1]}`);
  assert.ok(Math.abs(w[2] - wStar[2]) < 0.1, `w2 ${w[2]} ≈ ${wStar[2]}`);
});

test('solveJacobianStep respects active-channel set and ±0.25 clamp', () => {
  const J = makeJacobian(1, 52);
  computeJacobian(J, 52, syntheticSample);
  const w = new Float32Array(52);
  solveJacobianStep(J, 0, 10, 10, 10, w, 52, CONFIG.step, CONFIG.clampPerIter, [2]);
  assert.equal(w[0], 0, 'inactive channel untouched');
  assert.equal(w[1], 0);
  assert.ok(w[2] > 0 && w[2] <= CONFIG.clampPerIter, `clamped to ${w[2]}`);
});

test('solveJacobianStep signed deltas (lo/hi) for drag layer', () => {
  const J = makeJacobian(1, 52);
  computeJacobian(J, 52, syntheticSample);
  const w = new Float32Array(52);
  solveJacobianStep(J, 0, -1, -1, -1, w, 52, CONFIG.step, CONFIG.clampPerIter, null, -1, 1);
  assert.ok(w[2] < 0, 'negative delta preserved');
});

test('solveRidgeFull: converges to the same solution (HUD toggle path)', () => {
  const J = makeJacobian(1, 52);
  computeJacobian(J, 52, syntheticSample);
  const wStar = new Float32Array(52);
  wStar[0] = 0.5;
  wStar[1] = 0.5;
  wStar[2] = 0.4;
  const target = new Float32Array(3);
  syntheticSample(wStar, target);
  const w = new Float32Array(52);
  for (let i = 0; i < 40; i++) {
    const cur = new Float32Array(3);
    syntheticSample(w, cur);
    solveRidgeFull(J, 0, target[0] - cur[0], target[1] - cur[1], target[2] - cur[2], w, 52, 0.001);
  }
  assert.ok(Math.abs(w[0] - 0.5) < 0.12, `w0 ${w[0]}`);
  assert.ok(Math.abs(w[2] - 0.4) < 0.12, `w2 ${w[2]}`);
});

test('computeJacobian finite differences match the analytic smile Jacobian', () => {
  const J = makeJacobian(1, 52);
  computeJacobian(J, 52, syntheticSample, 1.0);
  // Column 0: ∂p/∂w0 = (1, 0.15, 0)
  assert.ok(Math.abs(J[0] - 1) < 1e-6);
  assert.ok(Math.abs(J[1] - 0.15) < 1e-6);
  // Column 2: ∂p/∂w2 = (0, 0, 1)
  assert.ok(Math.abs(J[2 * 3 + 2] - 1) < 1e-6);
});

test('UndoStack depth 20 (§6)', () => {
  const u = new UndoStack(20);
  for (let i = 0; i < 25; i++) u.push(new Float32Array([i]));
  assert.equal(u.size, 20);
  const top = u.pop();
  assert.equal(top[0], 24);
});

test('assetHash is stable + controls list is the §6 set', () => {
  assert.equal(assetHash('abc'), assetHash('abc'));
  assert.notEqual(assetHash('abc'), assetHash('abd'));
  assert.ok(CONTROLS.length >= 14 && CONTROLS.length <= 16, `~16 controls (${CONTROLS.length})`);
  const names = CONTROLS.map((c) => c.name);
  for (const n of ['mouthCornerL', 'lipUpperMid', 'chinMid', 'noseTip', 'lidMidL']) {
    assert.ok(names.includes(n), `${n} control exists`);
  }
  void CHANNELS;
});
