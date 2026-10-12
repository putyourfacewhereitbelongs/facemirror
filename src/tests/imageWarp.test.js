/**
 * imageWarp.test.js — unit & integration tests for 2D user-selected image
 * landmark extraction, 52-channel morph warping, edge pinning, drag Jacobian
 * convergence on 2D image control points, and Tracker.detectImage.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeImagePlaneSize,
  edgePinWeight,
  buildCanonicalImageLandmarks,
  extractImageFeatures,
  faceReliefZ,
  imageMorphDelta,
  evaluateControlAnchorMorphed,
} from '../imageWarp.js';
import { CHANNELS, CH } from '../rig.js';
import { CONTROLS, makeJacobian, computeJacobian, solveJacobianStep } from '../drag.js';
import { Tracker } from '../tracker.js';

test('computeImagePlaneSize preserves aspect ratio and fits bounds', () => {
  const portrait = computeImagePlaneSize(600, 800, 1.4, 1.0);
  assert.ok(Math.abs(portrait.aspect - 0.75) < 1e-6);
  assert.ok(portrait.planeW <= 1.4 + 1e-6 && portrait.planeH <= 1.0 + 1e-6);
  assert.ok(Math.abs(portrait.planeW / portrait.planeH - 0.75) < 1e-6);

  const landscape = computeImagePlaneSize(1600, 900, 1.4, 1.0);
  assert.ok(landscape.planeW <= 1.4 + 1e-6 && landscape.planeH <= 1.0 + 1e-6);
  assert.ok(Math.abs(landscape.planeW / landscape.planeH - 16 / 9) < 1e-6);
});

test('edgePinWeight pins outer image borders to 0 and interior to 1', () => {
  assert.equal(edgePinWeight(0, 0.5), 0);
  assert.equal(edgePinWeight(1, 0.5), 0);
  assert.equal(edgePinWeight(0.5, 0), 0);
  assert.equal(edgePinWeight(0.5, 1), 0);
  assert.equal(edgePinWeight(0.5, 0.5), 1);
});

test('extractImageFeatures maps 478 landmarks into 16 ordered drag anchors', () => {
  const lm = buildCanonicalImageLandmarks();
  assert.equal(lm.length, 478);
  const spec = computeImagePlaneSize(640, 800);
  const feat = extractImageFeatures(lm, spec);

  assert.equal(feat.controlAnchors.length, CONTROLS.length);
  assert.ok(feat.eyeL.outer[0] < feat.eyeR.outer[0], 'left eye is on -x side');
  assert.ok(feat.mouth.cornerL[0] < feat.mouth.cornerR[0], 'left mouth corner is on -x side');
  assert.ok(feat.eyeL.center[1] > feat.mouth.center[1], 'eyes are above mouth (+y)');
  assert.ok(feat.mouth.center[1] > feat.chinBottom[1], 'mouth is above chin (+y)');
});

test('imageMorphDelta warps smile, jawOpen, blink, and brows while keeping borders pinned', () => {
  const spec = computeImagePlaneSize(640, 800);
  const feat = extractImageFeatures(buildCanonicalImageLandmarks(), spec);
  const out = new Float32Array(3);

  // Left mouth corner during mouthSmileLeft: moves outward (-x) and upward (+y).
  const [mcx, mcy, mcu, mcv] = feat.mouth.cornerL;
  imageMorphDelta('mouthSmileLeft', mcx, mcy, mcu, mcv, feat, out);
  assert.ok(out[0] < -0.005, `smileLeft dx=${out[0]} moves outward (-x)`);
  assert.ok(out[1] > 0.01, `smileLeft dy=${out[1]} moves upward (+y)`);

  // Lower lip / chin during jawOpen: drops downward (-y), while brow stays still.
  const [llx, lly, llu, llv] = feat.mouth.lower;
  imageMorphDelta('jawOpen', llx, lly, llu, llv, feat, out);
  assert.ok(out[1] < -0.02, `jawOpen drops lower lip dy=${out[1]}`);

  const [bx, by, bu, bv] = feat.browL.inner;
  imageMorphDelta('jawOpen', bx, by, bu, bv, feat, out);
  assert.ok(Math.abs(out[1]) < 1e-4, `jawOpen does not move brow dy=${out[1]}`);

  // Left upper eyelid during eyeBlinkLeft vs eyeWideLeft.
  const [etx, ety, etu, etv] = feat.eyeL.top;
  imageMorphDelta('eyeBlinkLeft', etx, ety, etu, etv, feat, out);
  assert.ok(out[1] < -0.01, `eyeBlinkLeft closes upper lid dy=${out[1]}`);
  imageMorphDelta('eyeWideLeft', etx, ety, etu, etv, feat, out);
  assert.ok(out[1] > 0.005, `eyeWideLeft raises upper lid dy=${out[1]}`);

  // Outer border remains strictly zero for all 52 channels.
  for (const ch of CHANNELS) {
    imageMorphDelta(ch, -spec.planeW / 2, 0, 0, 0.5, feat, out);
    assert.equal(out[0], 0);
    assert.equal(out[1], 0);
    assert.equal(out[2], 0);
  }
});

test('all 52 ARKit channels produce finite deltas and relief across the image', () => {
  const spec = computeImagePlaneSize(800, 800);
  const feat = extractImageFeatures(null, spec);
  const out = new Float32Array(3);
  for (const ch of CHANNELS) {
    for (const [u, v] of [[0.35, 0.42], [0.5, 0.5], [0.42, 0.63], [0.62, 0.42]]) {
      const x = (u - 0.5) * spec.planeW;
      const y = (0.5 - v) * spec.planeH + spec.centerY;
      imageMorphDelta(ch, x, y, u, v, feat, out);
      assert.ok(Number.isFinite(out[0]) && Number.isFinite(out[1]) && Number.isFinite(out[2]));
      const rz = faceReliefZ(x, y, u, v, feat);
      assert.ok(Number.isFinite(rz) && rz >= spec.baseZ);
    }
  }
});

test('2D image drag Jacobian: dragging mouth corner up-left raises mouthSmileLeft', () => {
  const spec = computeImagePlaneSize(640, 800);
  const feat = extractImageFeatures(buildCanonicalImageLandmarks(), spec);
  const J = makeJacobian(CONTROLS.length, 52);
  const tmp3 = new Float32Array(3);

  computeJacobian(J, 52, (w, out) => {
    for (let c = 0; c < CONTROLS.length; c++) {
      evaluateControlAnchorMorphed(c, w, CHANNELS, feat, tmp3, 1);
      out[c * 3] = tmp3[0];
      out[c * 3 + 1] = tmp3[1];
      out[c * 3 + 2] = tmp3[2];
    }
  });

  const w = new Float32Array(52);
  const ctrlIdx = 0; // mouthCornerL
  const activeIdx = CONTROLS[ctrlIdx].channels.map((n) => CH[n]);
  for (let i = 0; i < 10; i++) {
    solveJacobianStep(J, ctrlIdx, -0.02, 0.04, 0, w, 52, 0.8, 0.25, activeIdx, 0, 1);
  }
  assert.ok(w[CH.mouthSmileLeft] > 0.2, `mouthSmileLeft rose to ${w[CH.mouthSmileLeft]}`);
});

test('Tracker.detectImage works with FakeDetector in headless mode', async () => {
  globalThis.window = { __FACEMIRROR_FAKE: 'happy' };
  try {
    const tracker = new Tracker({ onStatus: () => {} });
    const det = await tracker.detectImage({ width: 640, height: 480 });
    assert.equal(det.hasFace, true);
    assert.equal(det.landmarks.length, 478);
    assert.equal(det.weights.length, 52);
  } finally {
    delete globalThis.window;
  }
});
