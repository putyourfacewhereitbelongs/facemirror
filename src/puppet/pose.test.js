/**
 * Head-pose stage: rigid fit, expression decomposition, stitching, jaw hinge,
 * perspective projection and the stretch limiter.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { syntheticFace, fakeTesselation } from './synthetic.js';
import { buildRig } from './model.js';
import { createPoseState, buildDragField, driverPose, expressionField, driverJawAngle, stitchRegions, poseFrame, measureStretch, eyeVisibility, relaxStretch } from './pose.js';
import { RFromEuler, eulerFromR, mul3, fitSimilarity, centroid, hingeDisplacement } from './math3d.js';
import { RIG, MANDIBLE, OVAL, LIPSET } from './landmarks.js';

let rig, st, neutral, map;

/** Rotate a landmark cloud rigidly about its own centroid (a fake head turn). */
function rotateCloud(P, R, ids) {
  const c = centroid(P, ids || P.map((_, i) => i));
  return P.map(p => {
    const d = [p[0] - c[0], p[1] - c[1], (p[2] || 0) - c[2]];
    const r = mul3(R, d);
    return [r[0] + c[0], r[1] + c[1], r[2] + c[2]];
  });
}

beforeAll(() => {
  const f = syntheticFace({ W: 720, H: 900 });
  rig = buildRig(f.V, f.W, f.H, fakeTesselation(f.V));
  st = createPoseState(rig);
  st.dragField = buildDragField(rig, 1);
  neutral = f.V.map(p => [p[0], p[1], p[2]]);
  map = fitSimilarity(RIG.map(i => neutral[i]), RIG.map(i => [rig.Xc[i * 3], rig.Xc[i * 3 + 1], rig.Xc[i * 3 + 2]]), RIG.map(() => 1));
});

describe('rigid head pose fit', () => {
  it('recovers a known yaw rotation', () => {
    const R = RFromEuler(0.42, 0, 0);
    const cur = rotateCloud(neutral, R, RIG);
    const pose = driverPose(neutral, cur);
    const e = eulerFromR(pose.R);
    expect(e.yaw).toBeCloseTo(0.42, 1);
    expect(Math.abs(e.pitch)).toBeLessThan(0.05);
    expect(Math.abs(e.roll)).toBeLessThan(0.05);
    expect(pose.s).toBeCloseTo(1, 1);
  });

  it('recovers pitch and roll together', () => {
    const R = RFromEuler(-0.2, 0.3, 0.15);
    const pose = driverPose(neutral, rotateCloud(neutral, R, RIG));
    const e = eulerFromR(pose.R);
    expect(e.yaw).toBeCloseTo(-0.2, 1);
    expect(e.pitch).toBeCloseTo(0.3, 1);
    expect(e.roll).toBeCloseTo(0.15, 1);
  });

  it('reports an approach as scale, which becomes a dolly not a zoom', () => {
    const c = centroid(neutral, RIG);
    const cur = neutral.map(p => [c[0] + (p[0] - c[0]) * 1.25, c[1] + (p[1] - c[1]) * 1.25, p[2] * 1.25]);
    const pose = driverPose(neutral, cur);
    expect(pose.s).toBeGreaterThan(1.15);
    expect(pose.s).toBeLessThan(1.35);
    // the renderer turns that into depth: dz = F * (1 - 1/s) > 0 means toward camera
    const dz = rig.frame.F * (1 - 1 / pose.s);
    expect(dz).toBeGreaterThan(0);
  });
});

describe('expression decomposition (pose removed)', () => {
  it('is ~zero for a rigidly turned head with a frozen face', () => {
    const cur = rotateCloud(neutral, RFromEuler(0.5, 0.1, 0), RIG);
    const pose = driverPose(neutral, cur);
    const out = new Float32Array(478 * 3);
    expressionField(neutral, cur, pose, map, out, { lip: 1, eye: 1, nose: 1, jawW: rig.jawW });
    let worst = 0;
    for (let i = 0; i < 478; i++) for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(out[i * 3 + k]));
    expect(worst).toBeLessThan(3);            // px in canonical space: a pure turn leaks almost nothing
  });

  it('keeps a smile when the head is turned (pose invariance)', () => {
    const smile = neutral.map((p, i) => (LIPSET.has(i) && p[1] > rig.frame.mouthCY - 5 ? [p[0], p[1] - 9, p[2]] : p.slice()));
    const turned = rotateCloud(smile, RFromEuler(0.6, 0, 0), RIG);
    const pose = driverPose(neutral, turned);
    const out = new Float32Array(478 * 3);
    expressionField(neutral, turned, pose, map, out, { lip: 1, eye: 1, nose: 1, jawW: rig.jawW });
    const corner = out[61 * 3 + 1];
    expect(corner).toBeLessThan(-3);          // the lip corner still lifts in canonical space
    // and the skull does not carry the smile
    let skull = 0;
    for (const i of RIG) skull = Math.max(skull, Math.hypot(out[i * 3], out[i * 3 + 1]));
    expect(skull).toBeLessThan(Math.abs(corner));
  });

  it('scales the lip field by the retarget gain', () => {
    const smile = neutral.map((p, i) => (LIPSET.has(i) ? [p[0], p[1] - 8, p[2]] : p.slice()));
    const pose = driverPose(neutral, smile);
    const a = new Float32Array(478 * 3), b = new Float32Array(478 * 3);
    expressionField(neutral, smile, pose, map, a, { lip: 1, jawW: rig.jawW });
    expressionField(neutral, smile, pose, map, b, { lip: 2, jawW: rig.jawW });
    expect(Math.abs(b[61 * 3 + 1])).toBeGreaterThan(Math.abs(a[61 * 3 + 1]) * 1.6);
  });
});

describe('jaw hinge', () => {
  /** Rotate the mandible about the driver's own TMJ axis, like a real hinge. */
  const hingeMandible = (angle) => {
    const t = [(neutral[234][0] + neutral[454][0]) / 2, (neutral[234][1] + neutral[454][1]) / 2, 0];
    return neutral.map((p, i) => {
      if (!MANDIBLE.includes(i)) return p.slice();
      const v = [p[0] - t[0], p[1] - t[1], (p[2] || 0) - t[2]];
      const c = Math.cos(angle), s = Math.sin(angle);
      // rotation about +x (the intercondylar axis), y down / z toward viewer
      return [p[0], t[1] + v[1] * c - v[2] * s, t[2] + v[1] * s + v[2] * c];
    });
  };

  it('measures a hinged mandible as a negative (open) angle of the same size', () => {
    const open = hingeMandible(-0.32);
    const pose = driverPose(neutral, open);
    expect(driverJawAngle(neutral, open, pose)).toBeCloseTo(-0.32, 1);
  });

  it('is zero for a neutral face and positive when the jaw closes past neutral', () => {
    expect(Math.abs(driverJawAngle(neutral, neutral, driverPose(neutral, neutral)))).toBeLessThan(1e-6);
    const closed = hingeMandible(0.2);
    expect(driverJawAngle(neutral, closed, driverPose(neutral, closed))).toBeGreaterThan(0.1);
  });

  it('is invariant to head pose: the same jaw opening reads the same when turned', () => {
    const open = hingeMandible(-0.3);
    const R = RFromEuler(0.55, 0.12, 0);
    const c = centroid(neutral, RIG);
    const spin = P => P.map(p => { const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]], r = mul3(R, d); return [r[0] + c[0], r[1] + c[1], r[2] + c[2]]; });
    const a0 = driverJawAngle(neutral, open, driverPose(neutral, open));
    const a1 = driverJawAngle(spin(neutral), spin(open), driverPose(spin(neutral), spin(open)));
    expect(Math.abs(a0 - a1)).toBeLessThan(0.05);
  });

  it('rotating the mandible about the TMJ axis moves the chin down and back', () => {
    const chin = [rig.Xc[152 * 3], rig.Xc[152 * 3 + 1], rig.Xc[152 * 3 + 2]];
    const d = hingeDisplacement(chin, rig.frame.tmjOrigin, rig.frame.tmjAxis, -0.35);
    expect(d[1]).toBeGreaterThan(2);          // image y grows downward => chin dropped
    expect(d[2]).toBeLessThan(-2);            // and swung back, away from the camera
    // the hinge origin must live in canonical space, next to the chin's own frame
    expect(Math.abs(rig.frame.tmjOrigin[0])).toBeLessThan(rig.frame.FW * 0.2);
  });

  it('poseFrame applies the hinge only to the mandible', () => {
    const before = rig.V[10].slice();
    poseFrame(rig, st, { R: null, jawAngle: -0.35, stretch: 0 });
    const chinDrop = st.P[152 * 2 + 1] - rig.V[152][1];
    const forehead = Math.hypot(st.P[10 * 2] - before[0], st.P[10 * 2 + 1] - before[1]);
    expect(chinDrop).toBeGreaterThan(4);
    expect(forehead).toBeLessThan(0.5);
  });
});

describe('stitching (LivePortrait analogue)', () => {
  it('removes a rigid drift of the eye cloud', () => {
    const expr = new Float32Array(478 * 3);
    for (const i of rig.stitchGroups[0].ids) { expr[i * 3] = 6; expr[i * 3 + 1] = -4; }
    stitchRegions(rig, expr, 1, rig.stitchGroups);
    let dx = 0, dy = 0;
    for (const i of rig.stitchGroups[0].ids) { dx += expr[i * 3]; dy += expr[i * 3 + 1]; }
    dx /= rig.stitchGroups[0].ids.length; dy /= rig.stitchGroups[0].ids.length;
    expect(Math.abs(dx)).toBeLessThan(0.2);
    expect(Math.abs(dy)).toBeLessThan(0.2);
  });

  it('does nothing at zero amount and leaves far regions alone', () => {
    const expr = new Float32Array(478 * 3);
    for (const i of rig.stitchGroups[0].ids) expr[i * 3] = 5;
    stitchRegions(rig, expr, 0, rig.stitchGroups);
    expect(expr[rig.stitchGroups[0].ids[0] * 3]).toBe(5);
    const chin = 152;
    stitchRegions(rig, expr, 1, rig.stitchGroups);
    expect(Math.abs(expr[chin * 3])).toBeLessThan(0.6);
  });
});

describe('projection', () => {
  it('rest pose projects to the identity', () => {
    poseFrame(rig, st, { R: null, stretch: 0 });
    let worst = 0;
    for (let i = 0; i < rig.n; i++) worst = Math.max(worst, Math.hypot(st.P[i * 2] - rig.V[i][0], st.P[i * 2 + 1] - rig.V[i][1]));
    expect(worst).toBeLessThan(0.05);
  });

  it('a turn compresses the silhouette and moves the nose toward the viewer', () => {
    const width = () => { let a = Infinity, b = -Infinity; for (const i of OVAL) { a = Math.min(a, st.P[i * 2]); b = Math.max(b, st.P[i * 2]); } return b - a; };
    poseFrame(rig, st, { R: null, stretch: 1.6 });
    const rest = width();
    poseFrame(rig, st, { R: RFromEuler(0.7, 0, 0), stretch: 1.6 });
    expect(width()).toBeLessThan(rest * 0.92);
    // the near cheekbone comes forward, the far one goes back: real parallax
    expect(st.Pz[234]).toBeGreaterThan(st.Pz[454] + 5);
  });

  it('keeps vertical extent roughly constant in a pure yaw (no bending)', () => {
    const height = () => { let a = Infinity, b = -Infinity; for (const i of OVAL) { a = Math.min(a, st.P[i * 2 + 1]); b = Math.max(b, st.P[i * 2 + 1]); } return b - a; };
    poseFrame(rig, st, { R: null, stretch: 1.6 });
    const h0 = height();
    poseFrame(rig, st, { R: RFromEuler(0.55, 0, 0), stretch: 1.6 });
    expect(height()).toBeGreaterThan(h0 * 0.97);
    expect(height()).toBeLessThan(h0 * 1.03);
  });

  it('the stretch limiter caps skin elongation and reports it', () => {
    const R = RFromEuler(0.86, 0.22, 0.12);          // the app's soft yaw limit
    poseFrame(rig, st, { R, stretch: 0 });
    const unlimited = measureStretch(rig, st);
    poseFrame(rig, st, { R, stretch: 1.45, relaxIters: 4 });
    const limited = measureStretch(rig, st);
    expect(limited).toBeLessThan(unlimited);
    expect(limited).toBeLessThan(2.1);
    expect(st.lastStretch).toBeGreaterThan(1);        // the constraint did engage
  });

  it('relaxation is idempotent once the constraint is satisfied', () => {
    poseFrame(rig, st, { R: RFromEuler(0.4, 0.1, 0), stretch: 2.4, relaxIters: 4 });
    const snapshot = Float32Array.from(st.P);
    relaxStretch(rig, st, 2.4, 4);
    let worst = 0;
    for (let i = 0; i < snapshot.length; i++) worst = Math.max(worst, Math.abs(snapshot[i] - st.P[i]));
    expect(worst).toBeLessThan(2);
  });

  it('body and border vertices stay put when only the head rotates', () => {
    poseFrame(rig, st, { R: RFromEuler(0.6, 0.2, 0.1), stretch: 0 });
    let borderDrift = 0;
    for (let i = rig.borderStart; i < rig.bodyStart; i++) {
      borderDrift = Math.max(borderDrift, Math.hypot(st.P[i * 2] - rig.V[i][0], st.P[i * 2 + 1] - rig.V[i][1]));
    }
    expect(borderDrift).toBeLessThan(0.01);
  });
});

describe('drag field', () => {
  it('peaks at the dragged landmark and falls off with distance', () => {
    const df = st.dragField;
    const k = df.count - 1;                       // some draggable landmark
    const v = rig.shown.idx[k];
    const self = df.field[v * df.count + k];
    const far = df.field[10 * df.count + k];       // forehead, far from the mouth
    expect(self).toBeGreaterThan(0.9);
    expect(far).toBeLessThan(self);
  });

  it('moves the mesh when an edit is applied, and only near the edit', () => {
    poseFrame(rig, st, { R: null, stretch: 0 });
    const base = Float32Array.from(st.P);
    const drag = new Float32Array(rig.shown.count * 2);
    const k = rig.shown.idx.indexOf(61);
    drag[k * 2 + 1] = -14;                          // lift the left mouth corner
    poseFrame(rig, st, { R: null, drag, stretch: 0 });
    const moved = Math.hypot(st.P[61 * 2] - base[61 * 2], st.P[61 * 2 + 1] - base[61 * 2 + 1]);
    const far = Math.hypot(st.P[10 * 2] - base[10 * 2], st.P[10 * 2 + 1] - base[10 * 2 + 1]);
    expect(moved).toBeGreaterThan(6);
    expect(far).toBeLessThan(1.5);
  });
});

describe('eye visibility', () => {
  it('fades the far eye as yaw grows, and never the near one', () => {
    const v = eyeVisibility(0.9, 0);
    expect(v[0]).toBeGreaterThan(0.9);
    expect(v[1]).toBeLessThan(0.5);
    const v2 = eyeVisibility(-0.9, 0);
    expect(v2[1]).toBeGreaterThan(0.9);
    expect(v2[0]).toBeLessThan(0.5);
    const c = eyeVisibility(0, 0);
    expect(c[0]).toBe(1); expect(c[1]).toBe(1);
  });
});

describe('tracker confidence', () => {
  it('rotMix pulls the rotation back toward identity instead of snapping', () => {
    poseFrame(rig, st, { R: RFromEuler(0.8, 0, 0), rotMix: 0.5, stretch: 0 });
    const half = st.P[234 * 2];
    poseFrame(rig, st, { R: RFromEuler(0.8, 0, 0), rotMix: 1, stretch: 0 });
    const full = st.P[234 * 2];
    poseFrame(rig, st, { R: null, stretch: 0 });
    const rest = st.P[234 * 2];
    expect(Math.abs(half - rest)).toBeLessThan(Math.abs(full - rest));
    expect(Math.abs(half - rest)).toBeGreaterThan(0.5);
  });
});
