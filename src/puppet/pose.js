/**
 * Per-frame head posing: the "3DMM-style" stage of the pipeline.
 *
 * Motion is decomposed the way FaceVid2Vid / LivePortrait decompose it:
 *
 *   canonical shape  ->  expression displacement (head space)
 *                    ->  rigid head rotation + translation (head space)
 *                    ->  perspective projection to the image plane
 *
 * Expression is applied BEFORE rotation, so a smile stays a smile at any yaw
 * instead of shearing with the pose. Rotation is applied with linear-blend
 * skinning weights (the neck follows the skull only partially) and projected
 * through a pinhole whose rest projection is the identity - that is what stops
 * the turn from looking like a bent photograph.
 *
 * Two extra passes keep the result physical: back-face culling with depth
 * sorting (the far side of the skull goes behind), and an edge-length
 * relaxation that caps texture stretch so no triangle can be pulled into a
 * smear when the driver asks for pixels the photo does not contain.
 */

import { clamp, ramp, mul3, transpose, fitSimilarity, axisAngle, rotAxis, hingeDisplacement, eulerFromR } from './math3d.js';
import { LIPSET, NOSE_BRIDGE, NOSE_PTS, MANDIBLE, EYE, IRIS, RIG, RIG_W } from './landmarks.js';
import { LIGHT, R_FACE, R_HAIR, R_BODY, R_BORDER } from './model.js';

const EYE_SET = new Set([...EYE[0], ...EYE[1], ...IRIS, ...IRIS.map(i => i + 1), ...IRIS.map(i => i + 2), ...IRIS.map(i => i + 3), ...IRIS.map(i => i + 4)]);

export function createPoseState(rig) {
  const n = rig.n, t = rig.tri.count;
  const st = {
    n, triCount: t,
    P: new Float32Array(n * 2),      // projected positions
    X: new Float32Array(n * 3),      // rotated head-space positions
    Pz: new Float32Array(n),         // rotated depth per vertex
    triZ: new Float32Array(t),       // mean depth per triangle
    cull: new Uint8Array(t),         // 1 = back-facing, do not draw
    order: Int32Array.from(rig.tri.order),
    headOrder: [],
    mob: new Float32Array(n),        // mobility used by the stretch solver
    dragField: null,
    lamTri: new Float32Array(rig.shade.count),
    lastStretch: 1, preStretch: 1,
  };
  // Inverse-mass weights for the stretch solver. Hair shells are almost
  // massless, so an over-stretched hair band absorbs its own correction instead
  // of dragging the face outline (which is what the viewer actually checks).
  for (let i = 0; i < n; i++) {
    const r = rig.region[i];
    st.mob[i] = r === R_BORDER ? 0 : r === R_BODY ? 0.3 : r === R_HAIR ? 0.25 : 1;
  }
  for (let k = rig.tri.headStart; k < rig.tri.order.length; k++) st.headOrder.push(rig.tri.order[k]);
  return st;
}

/**
 * Gaussian spread field for landmark drags: how strongly each face vertex
 * follows each draggable landmark. Rebuilt when the image or the "drag area"
 * slider changes, never per frame.
 */
export function buildDragField(rig, areaScale) {
  const idx = rig.shown.idx, NL = idx.length;
  const sigma = 0.055 * (areaScale || 1) * rig.frame.FW;
  const inv = 1 / (2 * sigma * sigma);
  const field = new Float32Array(478 * NL);
  for (let v = 0; v < 478; v++) {
    const vx = rig.V[v][0], vy = rig.V[v][1];
    for (let k = 0; k < NL; k++) {
      const q = rig.V[idx[k]];
      const dx = vx - q[0], dy = vy - q[1];
      const w = Math.exp(-(dx * dx + dy * dy) * inv);
      if (w > 0.002) field[v * NL + k] = w;
    }
  }
  return { field, sigma, count: NL };
}

/** Rigid head pose of the driver: similarity fit from its neutral to its current frame. */
export function driverPose(neutral, current, ids = RIG, w = RIG_W) {
  const A = ids.map(i => neutral[i]), B = ids.map(i => current[i]);
  return fitSimilarity(A, B, w);
}

/** A driver landmark re-expressed in its own neutral (canonical) head frame. */
function toCanonical(neutral, current, i, pose, Rt, s) {
  const c = current[i], cb = pose.cb, ca = pose.ca;
  const v = mul3(Rt, [(c[0] - cb[0]) / s, (c[1] - cb[1]) / s, ((c[2] || 0) - (cb[2] || 0)) / s]);
  return [v[0] + ca[0], v[1] + ca[1], v[2] + (ca[2] || 0)];
}

/**
 * Expression field: the driver's per-landmark deformation with its head pose
 * removed, carried into the photo's canonical head space and retargeted by the
 * eye / lip / nose gains.
 */
export function expressionField(neutral, current, pose, map, out, gains) {
  const Rt = transpose(pose.R), s = pose.s || 1;
  const mR = map.R, ms = map.s || 1;
  const lipG = gains.lip ?? 1, noseG = gains.nose ?? 1, eyeG = gains.eye ?? 1;
  const jawW = gains.jawW, cancel = gains.jawCancel ?? 1;
  // mean rigid translation of the driver's mandible: the jaw hinge re-supplies
  // it as a real 3D rotation, so remove it here or the mouth opens twice
  let mx = 0, my = 0, mz = 0;
  for (const i of MANDIBLE) {
    const c = toCanonical(neutral, current, i, pose, Rt, s), n0 = neutral[i];
    mx += c[0] - n0[0]; my += c[1] - n0[1]; mz += c[2] - (n0[2] || 0);
  }
  const mn = MANDIBLE.length;
  mx /= mn; my /= mn; mz /= mn;
  for (let i = 0; i < 478; i++) {
    const c = toCanonical(neutral, current, i, pose, Rt, s), n0 = neutral[i];
    const wj = jawW ? jawW[i] : 0, k = wj * cancel;
    const g = LIPSET.has(i) ? lipG : (NOSE_BRIDGE.has(i) || NOSE_PTS.has(i)) ? noseG : EYE_SET.has(i) ? eyeG : 1;
    const e = mul3(mR, [(c[0] - n0[0] - mx * k) * ms, (c[1] - n0[1] - my * k) * ms, (c[2] - (n0[2] || 0) - mz * k) * ms]);
    out[i * 3] = e[0] * g; out[i * 3 + 1] = e[1] * g; out[i * 3 + 2] = e[2] * g;
  }
  return out;
}

/**
 * Jaw opening as a rotation angle about the temporomandibular axis, measured on
 * the driver in its own canonical frame (negative = open: the chin swings down
 * and back, which is what a real mandible does).
 */
export function driverJawAngle(neutral, current, pose) {
  const Rt = transpose(pose.R), s = pose.s || 1;
  const P = i => toCanonical(neutral, current, i, pose, Rt, s);
  const t0 = [(neutral[234][0] + neutral[454][0]) / 2, (neutral[234][1] + neutral[454][1]) / 2, ((neutral[234][2] || 0) + (neutral[454][2] || 0)) / 2];
  const t1a = P(234), t1b = P(454);
  const t1 = [(t1a[0] + t1b[0]) / 2, (t1a[1] + t1b[1]) / 2, (t1a[2] + t1b[2]) / 2];
  const v0 = [neutral[152][1] - t0[1], (neutral[152][2] || 0) - t0[2]];
  const v1 = [P(152)[1] - t1[1], P(152)[2] - t1[2]];
  const cross = v0[0] * v1[1] - v0[1] * v1[0];
  const dot = v0[0] * v1[0] + v0[1] * v1[1];
  const a = Math.atan2(cross, dot || 1e-6);
  return clamp(a, -0.9, 0.45);
}

/**
 * LivePortrait-style stitching: expression edits are shape changes, they must
 * not translate the eyes or the maxilla-anchored upper lip across the face. The
 * mean rigid drift of each anchor group is measured and removed with a smooth
 * falloff into the surrounding region.
 */
/**
 * Per-vertex stitch weights, computed once per rig (they depend only on the rest
 * geometry and the group sigmas, never on the frame).
 */
function stitchWeights(rig, groups) {
  const key = '__stitchW';
  if (rig[key] && rig[key].count === groups.length) return rig[key];
  const n = 478, m = groups.length;
  const w = new Float32Array(n * m);
  for (let g = 0; g < m; g++) {
    const ids = groups[g].ids, inv = 1 / (2 * groups[g].sigma * groups[g].sigma);
    for (let i = 0; i < n; i++) {
      let d2 = Infinity;
      for (let q = 0; q < ids.length; q++) {
        const j = ids[q];
        const a = rig.V[i][0] - rig.V[j][0], b = rig.V[i][1] - rig.V[j][1];
        const s = a * a + b * b;
        if (s < d2) d2 = s;
      }
      const ww = Math.exp(-d2 * inv);
      if (ww > 0.002) w[i * m + g] = ww;
    }
  }
  rig[key] = { w, count: m };
  return rig[key];
}

/**
 * LivePortrait-style stitching: expression edits are shape changes, they must
 * not translate the eyes or the maxilla-anchored upper lip across the face. The
 * mean rigid drift of each anchor group is measured and removed with a smooth
 * falloff into the surrounding region.
 *
 * All drifts are measured BEFORE any correction is applied. Applying them one
 * group at a time lets overlapping falloffs undo each other (the eye pass drags
 * the maxilla, whose pass then drags the eye back), which shows up as a slow
 * swim of the whole mid-face.
 */
export function stitchRegions(rig, expr, amount, groups) {
  if (amount <= 0.001 || !groups || !groups.length) return expr;
  const m = groups.length;
  const drift = [];
  let any = false;
  for (let g = 0; g < m; g++) {
    const ids = groups[g].ids;
    let dx = 0, dy = 0;
    for (let q = 0; q < ids.length; q++) { const i = ids[q]; dx += expr[i * 3]; dy += expr[i * 3 + 1]; }
    dx /= ids.length; dy /= ids.length;
    if (Math.abs(dx) > 1e-4 || Math.abs(dy) > 1e-4) { drift.push(dx, dy); any = true; } else drift.push(0, 0);
  }
  if (!any) return expr;
  const { w } = stitchWeights(rig, groups);
  for (let i = 0; i < 478; i++) {
    let ax = 0, ay = 0;
    for (let g = 0; g < m; g++) {
      const ww = w[i * m + g];
      if (ww === 0) continue;
      ax += drift[g * 2] * ww; ay += drift[g * 2 + 1] * ww;
    }
    if (ax !== 0 || ay !== 0) { expr[i * 3] -= ax * amount; expr[i * 3 + 1] -= ay * amount; }
  }
  return expr;
}

/**
 * Position-based-dynamics pass that caps edge elongation: the classic defence
 * against "long smear" artefacts when a warp asks for pixels that do not exist.
 */
export function relaxStretch(rig, st, maxStretch, iters = 3, stiffness = 0.7) {
  if (!maxStretch || maxStretch <= 0) return;
  const { a, b, len0, count } = rig.edges, P = st.P, mob = st.mob;
  const solve = rig.edges.solve;
  st.preStretch = measureStretch(rig, st);   // what the projection alone produced
  for (let k = 0; k < iters; k++) {
    for (let e = 0; e < count; e++) {
      if (solve && !solve[e]) continue;
      const i = a[e], j = b[e], mi = mob[i], mj = mob[j];
      if (mi + mj < 0.02) continue;
      const dx = P[j * 2] - P[i * 2], dy = P[j * 2 + 1] - P[i * 2 + 1];
      const len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
      const lim = len0[e] * maxStretch;
      if (len <= lim) continue;
      const corr = ((len - lim) / len) * stiffness;
      const wi = mi / (mi + mj), wj = mj / (mi + mj);
      P[i * 2] += dx * corr * wi; P[i * 2 + 1] += dy * corr * wi;
      P[j * 2] -= dx * corr * wj; P[j * 2 + 1] -= dy * corr * wj;
    }
  }
  // Report the post-relaxation value: that is the stretch the viewer actually
  // sees, and it is the same number measureStretch() would return a line later.
  st.lastStretch = measureStretch(rig, st);
}

/**
 * Maximum texture stretch currently in the drawn skin (HUD / diagnostics).
 * Only edges the solver constrains *and* whose both ends are face vertices are
 * counted: the hair shells swing as rigid bodies on purpose, and the skirt to
 * the frame border is allowed to smear because it is either covered by the
 * inpainted plate or not drawn at all.
 */
export function measureStretch(rig, st) {
  const { a, b, len0, count, solve } = rig.edges, P = st.P;
  let worst = 1;
  for (let e = 0; e < count; e++) {
    if (solve && !solve[e]) continue;
    const i = a[e], j = b[e];
    if (rig.region[i] !== R_FACE || rig.region[j] !== R_FACE) continue;
    const dx = P[j * 2] - P[i * 2], dy = P[j * 2 + 1] - P[i * 2 + 1];
    const r = Math.sqrt(dx * dx + dy * dy) / (len0[e] || 1);
    if (r > worst) worst = r;
  }
  return worst;
}

/**
 * The main pose pass. Fills st.P (projected positions), st.Pz, st.cull and
 * st.order (painter's order, far first).
 */
export function poseFrame(rig, st, ctl) {
  const n = rig.n, { Xc, V, wRot, wExpr, jawW, hairW, region, frame } = rig;
  const { cx, cy, F } = frame;
  const P = st.P, X = st.X, Pz = st.Pz;

  // rotation, optionally pulled back toward identity (tracker confidence fade)
  let R = ctl.R;
  const mix = ctl.rotMix === undefined ? 1 : ctl.rotMix;
  if (R && mix < 0.999) {
    const aa = axisAngle(R);
    R = aa.angle < 1e-5 ? null : rotAxis(aa.axis, aa.angle * mix);
  }
  const r0 = R ? R[0] : 1, r1 = R ? R[1] : 0, r2 = R ? R[2] : 0;
  const r3 = R ? R[3] : 0, r4 = R ? R[4] : 1, r5 = R ? R[5] : 0;
  const r6 = R ? R[6] : 0, r7 = R ? R[7] : 0, r8 = R ? R[8] : 1;
  const tx = ctl.tx || 0, ty = ctl.ty || 0, tz = ctl.tz || 0;

  const jawA = (ctl.jawAngle || 0) * (ctl.jawGain === undefined ? 1 : ctl.jawGain);
  const tO = frame.tmjOrigin, tA = frame.tmjAxis;
  const expr = ctl.expr, preset = ctl.preset, eG = ctl.exprGain === undefined ? 1 : ctl.exprGain;
  const drag = ctl.drag, dragField = st.dragField, dragN = dragField ? dragField.count : 0;
  const hairX = (ctl.hairX || 0) * (ctl.hairGain === undefined ? 1 : ctl.hairGain);
  const hairY = (ctl.hairY || 0) * (ctl.hairGain === undefined ? 1 : ctl.hairGain);
  const body = ctl.body, bInv = 1 / (2 * (ctl.bodySigma || 1) ** 2), bodyGain = ctl.bodyGain === undefined ? 1 : ctl.bodyGain;

  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    let px = Xc[i3], py = Xc[i3 + 1], pz = Xc[i3 + 2];

    // --- expression in canonical head space (pose invariant) ---
    const we = wExpr[i] * eG;
    if (we > 0) {
      if (expr) { px += expr[i3] * we; py += expr[i3 + 1] * we; pz += expr[i3 + 2] * we; }
      if (preset && i < 468) { px += preset[i * 2] * we; py += preset[i * 2 + 1] * we; }
      const wj = jawW[i];
      if (wj > 0 && jawA) {
        const d = hingeDisplacement([px, py, pz], tO, tA, jawA * wj);
        px += d[0]; py += d[1]; pz += d[2];
      }
    }
    // --- sculpted landmark edits, in canonical space so they turn with the head ---
    if (drag && dragN && i < 478) {
      let ox = 0, oy = 0;
      const row = i * dragN, f = dragField.field;
      for (let k = 0; k < dragN; k++) {
        const w = f[row + k];
        if (w !== 0) { ox += drag[k * 2] * w; oy += drag[k * 2 + 1] * w; }
      }
      px += ox; py += oy;
    }

    // --- rigid transform + linear blend skinning toward the static rest pose ---
    const rx = r0 * px + r1 * py + r2 * pz + tx;
    const ry = r3 * px + r4 * py + r5 * pz + ty;
    const rz = r6 * px + r7 * py + r8 * pz + tz;
    X[i3] = rx; X[i3 + 1] = ry; X[i3 + 2] = rz;
    Pz[i] = rz;
    const s = F / (F - rz);
    let x = cx + rx * s, y = cy + ry * s;
    const w = wRot[i];
    if (w < 1) { const vx = V[i][0], vy = V[i][1]; x = vx + (x - vx) * w; y = vy + (y - vy) * w; }
    const hw = hairW[i];
    if (hw > 0) { x += hairX * hw; y += hairY * hw; }
    if (body && region[i] === R_BODY) {
      let sx = 0, sy = 0, sw = 0;
      for (let j = 0; j < body.length; j++) {
        const b = body[j];
        if (!b) continue;
        const dx = V[i][0] - b.x, dy = V[i][1] - b.y;
        const ww = Math.exp(-(dx * dx + dy * dy) * bInv);
        sx += b.dx * ww; sy += b.dy * ww; sw += ww;
      }
      if (sw > 0) { const k = bodyGain / (sw + 0.3); x += sx * k; y += sy * k; }
    }
    P[i * 2] = x; P[i * 2 + 1] = y;
  }

  // --- per-triangle normals: back-face culling + depth sorting ---
  const tri = rig.tri, tv = tri.v, n0 = tri.n0, kind = tri.kind, triZ = st.triZ, cull = st.cull;
  const cullBias = ctl.cullBias === undefined ? 0.015 : ctl.cullBias;
  for (let t = 0; t < tri.count; t++) {
    const a = tv[t * 3], b = tv[t * 3 + 1], c = tv[t * 3 + 2];
    triZ[t] = (X[a * 3 + 2] + X[b * 3 + 2] + X[c * 3 + 2]) * 0.3333333;
    if (kind[t] >= 2) { cull[t] = 0; continue; }
    cull[t] = (r6 * n0[t * 3] + r7 * n0[t * 3 + 1] + r8 * n0[t * 3 + 2]) <= cullBias ? 1 : 0;
  }
  if (ctl.sort !== false) sortHead(st);
  if (ctl.stretch) relaxStretch(rig, st, ctl.stretch, ctl.relaxIters ?? 3);
  else { st.preStretch = st.lastStretch = measureStretch(rig, st); }
  return st;
}

/** Painter's order: static shell first, then head triangles far -> near. */
export function sortHead(st) {
  const z = st.triZ, head = st.headOrder, order = st.order, start = order.length - head.length;
  head.sort((a, b) => z[a] - z[b]);
  for (let i = 0; i < head.length; i++) order[start + i] = head[i];
}

/**
 * Differential relighting of the coarse shading mesh: how much brighter or
 * darker each patch is than at rest. Exactly zero at neutral, so the photo's own
 * lighting is untouched until the head actually turns.
 */
export function shadeField(rig, st, R, strength) {
  const d = rig.shade.data, cnt = rig.shade.count, out = st.lamTri;
  const r0 = R ? R[0] : 1, r1 = R ? R[1] : 0, r2 = R ? R[2] : 0;
  const r3 = R ? R[3] : 0, r4 = R ? R[4] : 1, r5 = R ? R[5] : 0;
  const r6 = R ? R[6] : 0, r7 = R ? R[7] : 0, r8 = R ? R[8] : 1;
  for (let t = 0; t < cnt; t++) {
    const o = t * 7;
    const nx = d[o + 3], ny = d[o + 4], nz = d[o + 5];
    const mx = r0 * nx + r1 * ny + r2 * nz, my = r3 * nx + r4 * ny + r5 * nz, mz = r6 * nx + r7 * ny + r8 * nz;
    const lam = Math.max(0, mx * LIGHT[0] + my * LIGHT[1] + mz * LIGHT[2]);
    out[t] = (lam - d[o + 6]) * strength;
  }
  return out;
}

/** Yaw / pitch / roll of the current head pose (HUD + FX gating). */
export function poseAngles(R) { return R ? eulerFromR(R) : { yaw: 0, pitch: 0, roll: 0 }; }

/**
 * Eye visibility at large yaw: once the nose bridge comes between the camera and
 * the far eye, its overlay (iris, lids, lashes) must fade out instead of being
 * painted on top of the nose.
 */
export function eyeVisibility(yaw, pitch) {
  const a = Math.abs(yaw), far = ramp(0.5, 0.95, a), p = ramp(0.7, 1.15, Math.abs(pitch)) * 0.4;
  const v = [1 - p, 1 - p];
  if (yaw > 0) v[1] = Math.max(0, v[1] - far); else v[0] = Math.max(0, v[0] - far);
  return v;
}

export { clamp, ramp };
