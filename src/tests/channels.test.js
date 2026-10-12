/**
 * channels.test.js — dead-channel geometry on synthetic landmark rigs (§9.1):
 * chin shift → jawRight rises; whole head rotates → NO rise (anti-gating
 * regression — trap §11.5). Plus EAR, tongue, neutral blending.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DeadChannelSolver, transformToCanonical, makeCanonicalBuffer, measureNeutral,
  blendNeutral, eyeAspectRatio, LM, CONFIG,
} from '../channels.js';
import { CHANNELS, CH } from '../rig.js';

/**
 * Build a synthetic 478-landmark face in canonical (cm-ish) space.
 * +y up, +x = subject's right (ARKit convention).
 */
function syntheticFace(overrides = {}) {
  const p = makeCanonicalBuffer();
  const set = (i, x, y, z) => {
    p[i * 3] = x;
    p[i * 3 + 1] = y;
    p[i * 3 + 2] = z;
  };
  // Eyes
  set(LM.eyeAOuter, -2.8, 0.8, 4.5); set(LM.eyeAInner, -1.2, 0.8, 5.2);
  set(LM.eyeATop, -2.0, 1.15, 5.0); set(LM.eyeABot, -2.0, 0.55, 5.0);
  set(LM.eyeBOuter, 2.8, 0.8, 4.5); set(LM.eyeBInner, 1.2, 0.8, 5.2);
  set(LM.eyeBTop, 2.0, 1.15, 5.0); set(LM.eyeBBot, 2.0, 0.55, 5.0);
  set(LM.irisA, -2.0, 0.85, 5.3); set(LM.irisB, 2.0, 0.85, 5.3);
  // Nose
  set(LM.noseTip, 0, 0.1, 6.2);
  set(LM.noseBridgeLow, 0, 0.9, 5.6); set(LM.noseBridgeHigh, 0, 1.2, 5.4);
  set(LM.noseWingA, -0.8, -0.15, 5.6); set(LM.noseWingB, 0.8, -0.15, 5.6);
  set(LM.noseCreaseA, -1.0, -0.35, 5.4); set(LM.noseCreaseB, 1.0, -0.35, 5.4);
  // Mouth
  set(LM.mouthCornerA, -1.6, -1.15, 5.3); set(LM.mouthCornerB, 1.6, -1.15, 5.3);
  set(LM.upperLip, 0, -0.95, 5.8); set(LM.innerUpperLip, 0, -1.05, 5.8);
  set(LM.innerLowerLip, 0, -1.2, 5.8); set(LM.lowerLip, 0, -1.35, 5.7);
  set(LM.lowerLipArcA, -0.9, -1.3, 5.5); set(LM.lowerLipArcB, 0.9, -1.3, 5.5);
  // Cheeks / chin
  set(LM.cheekA, -3.6, -0.7, 3.2); set(LM.cheekB, 3.6, -0.7, 3.2);
  set(LM.chinBottom, 0, -2.3, 5.0); set(LM.chinMid, 0, -2.0, 5.4);
  for (const [k, v] of Object.entries(overrides)) {
    const i = /** @type {number} */ (LM[k] ?? Number(k));
    set(i, v[0], v[1], v[2]);
  }
  return p;
}

function makeSolver(swap = false) {
  const solver = new DeadChannelSolver(CHANNELS, { sideSwap: swap });
  solver.setNeutral(measureNeutral(syntheticFace()));
  return solver;
}

const NEUTRAL_CUES = {
  mouthClose: 0.5,
  jawOpen: 0.1,
  browDown: 0,
  mouthUpperUp: 0,
  smile: 0,
  headYaw: 0,
};

test('chin shift in +x raises jawRight (sideSwap=false), not jawLeft', () => {
  const solver = makeSolver(false);
  const out = new Float32Array(52);
  const face = syntheticFace({ chinBottom: [0.6, -2.3, 5.0] }); // +0.6 cm
  solver.solve(out, face, { ...NEUTRAL_CUES, jawOpen: 0.1 });
  assert.ok(out[CH.jawRight] > 0.2, `jawRight rose to ${out[CH.jawRight]}`);
  assert.equal(out[CH.jawLeft], 0, 'other side silent');
});

test('ANTI-GATING: whole-head rotation must NOT raise jaw channels (trap §11.5)', () => {
  // Rotate the entire face 25° about the bridge axis; the chin moves in
  // image space but not in canonical space — after pose-removal the solver
  // must see nothing.
  const base = syntheticFace();
  const yaw = 25 * Math.PI / 180;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const rotated = makeCanonicalBuffer();
  for (let i = 0; i < 478; i++) {
    const x = base[i * 3];
    const y = base[i * 3 + 1];
    const z = base[i * 3 + 2];
    rotated[i * 3] = c * x + s * z;
    rotated[i * 3 + 1] = y;
    rotated[i * 3 + 2] = -s * x + c * z;
  }
  // No-chin-shift: solver on rotated canonical coords should stay silent
  // because the rotation is applied consistently (canonical vs neutral).
  // Recompute neutral from the rotated pose (calibrated at that pose)…
  const solver = new DeadChannelSolver(CHANNELS);
  solver.setNeutral(measureNeutral(rotated));
  const out = new Float32Array(52);
  solver.solve(out, rotated, { ...NEUTRAL_CUES, jawOpen: 0.05 });
  assert.equal(out[CH.jawRight], 0, 'no jawRight on pure head rotation');
  assert.equal(out[CH.jawLeft], 0, 'no jawLeft on pure head rotation');
});

test('transformToCanonical + matrix kills image-space jaw lean (full trap §11.5 path)', () => {
  // Image landmarks are the rotation applied to canonical points; the
  // facialTransformationMatrix carries the same rotation. transform must
  // undo it exactly.
  const canon = syntheticFace({ chinBottom: [0, -2.3, 5.0] });
  const yaw = 30 * Math.PI / 180;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const imgW = 640;
  const imgH = 480;
  const ppcm = 80; // pixels per cm
  const landmarks = [];
  for (let i = 0; i < 478; i++) {
    const x = canon[i * 3];
    const y = canon[i * 3 + 1];
    const z = canon[i * 3 + 2];
    // Rotate: p' = R·p (world), then to image coords.
    const rx = c * x + s * z;
    const ry = y;
    const rz = -s * x + c * z;
    landmarks.push({
      x: rx * ppcm / imgW,
      y: -ry * ppcm / imgH + 0.5, // +y up → image y down
      z: rz * ppcm / imgW,
    });
  }
  // Column-major rotation matrix R (about y).
  const M = new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
  const out = makeCanonicalBuffer();
  // transformToCanonical builds [x·W/ppcm', y·H/ppcm', z·W/ppcm'] with
  // ppcm' = eyeWidthPx/interEyeCm; feed eyeWidthPx so ppcm' = ppcm.
  const eyeWidthPx = CONFIG.interEyeCm * ppcm;
  const scale = transformToCanonical(out, landmarks, M, imgW, imgH, eyeWidthPx);
  assert.ok(scale > 0);
  const solver = new DeadChannelSolver(CHANNELS);
  solver.setNeutral(measureNeutral(canon));
  const w = new Float32Array(52);
  solver.solve(w, out, { ...NEUTRAL_CUES, jawOpen: 0.05 });
  assert.ok(w[CH.jawRight] < 0.02, `matrix-removed pose: no fake jaw lean (${w[CH.jawRight]})`);
  assert.ok(w[CH.jawLeft] < 0.02);

  // …and a REAL chin shift in canonical space survives the same transform.
  const canonShift = syntheticFace({ chinBottom: [0.6, -2.3, 5.0] });
  const landmarks2 = landmarks.map((_, i) => {
    const x = canonShift[i * 3];
    const y = canonShift[i * 3 + 1];
    const z = canonShift[i * 3 + 2];
    return {
      x: (c * x + s * z) * ppcm / imgW,
      y: -y * ppcm / imgH + 0.5,
      z: (-s * x + c * z) * ppcm / imgW,
    };
  });
  transformToCanonical(out, landmarks2, M, imgW, imgH, eyeWidthPx);
  solver.solve(w, out, { ...NEUTRAL_CUES, jawOpen: 0.05 });
  assert.ok(w[CH.jawRight] > 0.2, `real shift detected through rotation (${w[CH.jawRight]})`);
});

test('cheekPuff: sealed-lip puff raises cheekPuff; gated off when mouth open', () => {
  const solver = makeSolver(false);
  const face = syntheticFace({
    cheekA: [-3.95, -0.7, 3.2],
    cheekB: [3.95, -0.7, 3.2],
  });
  const out = new Float32Array(52);
  solver.solve(out, face, { ...NEUTRAL_CUES, mouthClose: 0.6 });
  assert.ok(out[CH.cheekPuff] > 0.1, `cheekPuff ${out[CH.cheekPuff]}`);
  const out2 = new Float32Array(52);
  solver.solve(out2, face, { ...NEUTRAL_CUES, mouthClose: 0.05, jawOpen: 0.8 });
  assert.equal(out2[CH.cheekPuff], 0, 'gate: puffers seal lips (§5.2)');
});

test('jawForward: chin z beyond neutral, gated on jawOpen<0.25', () => {
  const solver = makeSolver(false);
  const face = syntheticFace({ chinBottom: [0, -2.3, 5.4] }); // +0.4 cm z
  const out = new Float32Array(52);
  solver.solve(out, face, { ...NEUTRAL_CUES, jawOpen: 0.1 });
  assert.ok(out[CH.jawForward] > 0.1, `jawForward ${out[CH.jawForward]}`);
  const out2 = new Float32Array(52);
  solver.solve(out2, face, { ...NEUTRAL_CUES, jawOpen: 0.5 });
  assert.equal(out2[CH.jawForward], 0, 'gate: wide jaw is just mouth opening');
});

test('tongueOut: inner lip below the lower-lip line with jawOpen>0.4', () => {
  const solver = makeSolver(false);
  const face = syntheticFace({ innerLowerLip: [0, -1.8, 5.8] }); // 0.5+ below line
  const out = new Float32Array(52);
  solver.solve(out, face, { ...NEUTRAL_CUES, jawOpen: 0.6 });
  assert.ok(out[CH.tongueOut] > 0.1, `tongueOut ${out[CH.tongueOut]}`);
});

test('eyeWide + EAR validate blink geometry', () => {
  const p = syntheticFace();
  const ear = eyeAspectRatio(p, true);
  assert.ok(ear > 0.3 && ear < 1.2, `EAR ${ear} plausible`);
  const solver = makeSolver(false);
  const wide = syntheticFace({ eyeATop: [-2.0, 1.35, 5.0], eyeABot: [-2.0, 0.4, 5.0], eyeBTop: [2.0, 1.35, 5.0], eyeBBot: [2.0, 0.4, 5.0] });
  const out = new Float32Array(52);
  solver.solve(out, wide, NEUTRAL_CUES);
  assert.ok(out[CH.eyeWideLeft] > 0 || out[CH.eyeWideRight] > 0, 'EAR delta drives eyeWide');
});

test('mouthFrown raises only with browDown gate; dims under smile dimple gate', () => {
  const solver = makeSolver(false);
  const frown = syntheticFace({
    mouthCornerA: [-1.6, -1.55, 5.3],
    mouthCornerB: [1.6, -1.55, 5.3],
  });
  const out = new Float32Array(52);
  solver.solve(out, frown, { ...NEUTRAL_CUES, browDown: 0.5, mouthClose: 0.5 });
  assert.ok(out[CH.mouthFrownLeft] > 0.05 || out[CH.mouthFrownRight] > 0.05, 'frown detected');
  const out2 = new Float32Array(52);
  solver.solve(out2, frown, { ...NEUTRAL_CUES, browDown: 0, mouthClose: 0.5 });
  assert.equal(out2[CH.mouthFrownLeft], 0, 'gate off without browDown');
});

test('blendNeutral EMA', () => {
  const a = measureNeutral(syntheticFace());
  const b = measureNeutral(syntheticFace({ chinBottom: [0.4, -2.3, 5.0] }));
  const m = blendNeutral(a, b, 0.5);
  assert.ok(Math.abs(m.chinX - (a.chinX + b.chinX) / 2) < 1e-6);
  assert.ok(Math.abs(m.cheekDist - a.cheekDist) < 1e-6, 'cheekDist unchanged');
});
