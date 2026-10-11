/**
 * Head model construction.
 *
 * This is the piece that turns a flat photo into something that can be rotated
 * like a head instead of bent like a card. Given the 478 landmarks detected on
 * the still image it builds:
 *
 *   1. a per-vertex DEPTH prior (ellipsoidal skull + anatomical bumps, refined
 *      with the detector's own relative depth, symmetrised and graph-smoothed),
 *   2. canonical head-space positions obtained by BACK-PROJECTING the photo
 *      through that depth, so re-projecting at rest is exactly the identity -
 *      the puppet matches the source pixel for pixel until it is driven,
 *   3. a single non-overlapping triangulation (official face topology inside the
 *      face, Delaunay for the hair/neck/body shell) with rest normals, rest
 *      areas and an edge list,
 *   4. per-vertex weight fields: head rotation blend (so the neck does not
 *      shear), jaw-hinge blend (the mandible is a bone, not a 2D slide), hair
 *      inertia and symmetry partners.
 */

import Delaunator from 'delaunator';
import { clamp, ramp, cross, norm3, dot3, mul3 } from './math3d.js';
import {
  OVAL, RIG, EYE, EYM, LIP, UP, LO, LIPSET, CHEEK, CHIN, FOREHEAD, BROW, NOSE_RIDGE, NOSE_TIP,
  NOSE_WING, IRIS, MANDIBLE, TEMPLE, MAXILLA, SHOWN,
} from './landmarks.js';

export const R_FACE = 0, R_HAIR = 1, R_BORDER = 2, R_BODY = 3;

/** Ring scales around the face oval: three shells that carry hair + silhouette. */
export const RINGS = [[1.25, 1, 1], [1.6, 0.7, 0.9], [2.1, 0.4, 0.6]];

/** Fixed world-space light used by the differential relighting pass. */
export const LIGHT = norm3([-0.38, -0.5, 0.78]);

/** Feature bumps for the depth prior; amplitudes/sigmas are fractions of face width. */
const FEATURES = [
  { pts: NOSE_RIDGE, amp: 0.058, sig: 0.062, line: true },
  { pts: NOSE_TIP, amp: 0.030, sig: 0.034 },
  { pts: NOSE_WING, amp: 0.016, sig: 0.034 },
  { pts: BROW, amp: 0.020, sig: 0.046 },
  { pts: EYE[0].concat(EYE[1]), amp: -0.026, sig: 0.032 },
  { pts: IRIS, amp: 0.034, sig: 0.024 },
  { pts: LIP, amp: 0.020, sig: 0.044 },
  { pts: CHIN, amp: 0.022, sig: 0.052 },
  { pts: CHEEK, amp: 0.012, sig: 0.085 },
  { pts: FOREHEAD, amp: 0.014, sig: 0.110 },
  { pts: TEMPLE, amp: -0.014, sig: 0.055 },
];

/** Build the mutual-adjacency triangle list from the official face tesselation. */
export function tessTriangles(edges, nVerts = 478) {
  const adj = Array.from({ length: nVerts }, () => new Set());
  for (const c of edges || []) {
    const a = c.start !== undefined ? c.start : c[0], b = c.end !== undefined ? c.end : c[1];
    if (a < nVerts && b < nVerts && a !== b) { adj[a].add(b); adj[b].add(a); }
  }
  const out = [];
  for (let a = 0; a < nVerts; a++) for (const b of adj[a]) if (b > a) for (const c of adj[b]) if (c > b && adj[a].has(c)) out.push(a, b, c);
  return { tris: new Int32Array(out), adj };
}

/** CSR adjacency over a triangle soup (used for Laplacian smoothing). */
export function adjacency(tris, n) {
  const sets = Array.from({ length: n }, () => []);
  for (let i = 0; i < tris.length; i += 3) {
    const a = tris[i], b = tris[i + 1], c = tris[i + 2];
    sets[a].push(b, c); sets[b].push(a, c); sets[c].push(a, b);
  }
  return sets.map(s => Array.from(new Set(s)));
}

function laplacian(z, adj, iters, lambda, freeze) {
  const out = Float32Array.from(z);
  const tmp = Float32Array.from(z);
  for (let k = 0; k < iters; k++) {
    for (let i = 0; i < z.length; i++) {
      if (freeze && freeze[i]) { tmp[i] = out[i]; continue; }
      const nb = adj[i];
      if (!nb || !nb.length) { tmp[i] = out[i]; continue; }
      let s = 0;
      for (let j = 0; j < nb.length; j++) s += out[nb[j]];
      tmp[i] = out[i] + lambda * (s / nb.length - out[i]);
    }
    out.set(tmp);
  }
  return out;
}

function distToSet(x, y, pts, V) {
  let best = Infinity;
  for (const i of pts) { const dx = x - V[i][0], dy = y - V[i][1]; const d = dx * dx + dy * dy; if (d < best) best = d; }
  return Math.sqrt(best);
}

/**
 * Depth prior for the 478 face landmarks. Returns z in pixels, positive toward
 * the viewer, with the head-centre plane at 0.
 */
export function depthPrior(V, frame, opts = {}) {
  const { cx, cy, ax, ay, az, FW } = frame;
  const flatPower = opts.flatPower ?? 0.55;
  const z = new Float32Array(478);
  // 1. super-ellipsoidal skull: flat across the front, falling away at the sides.
  for (let i = 0; i < 478; i++) {
    const dx = (V[i][0] - cx) / ax, dy = (V[i][1] - cy) / ay;
    const r2 = dx * dx + dy * dy;
    z[i] = az * Math.pow(Math.max(0, 1 - r2), flatPower);
  }
  // 2. anatomical bumps (nose, brow ridge, sockets, eyeballs, lips, chin, cheeks).
  for (const f of FEATURES) {
    const amp = f.amp * FW, sig = f.sig * FW, inv = 1 / (2 * sig * sig);
    for (let i = 0; i < 478; i++) {
      const d = distToSet(V[i][0], V[i][1], f.pts, V);
      z[i] += amp * Math.exp(-d * d * inv);
    }
  }
  // 3. the detector's own relative depth, sign-aligned, symmetrised and scaled.
  const blend = opts.detectorBlend ?? 0.34;
  if (blend > 0.001 && opts.sym) {
    const zd = new Float32Array(478);
    for (let i = 0; i < 478; i++) zd[i] = (V[i][2] || 0) * (opts.zSign || 1);
    let a = 0, b = 0, ma = 0, mb = 0;
    for (let i = 0; i < 478; i++) { a += zd[i]; b += z[i]; }
    ma = a / 478; mb = b / 478;
    let va = 0, vb = 0, cov = 0;
    for (let i = 0; i < 478; i++) { const da = zd[i] - ma, db = z[i] - mb; va += da * da; vb += db * db; cov += da * db; }
    // Auto-detect the API's depth sign by correlating with the analytic prior:
    // the nose always protrudes toward the camera, whatever the convention.
    const sign = cov < 0 ? -1 : 1;
    const scale = Math.sqrt(vb / (va || 1));
    for (let i = 0; i < 478; i++) zd[i] = (zd[i] - ma) * sign * scale;
    for (let i = 0; i < 478; i++) { // mirror symmetry: canonical shape, not pose
      const j = opts.sym[i];
      if (j >= 0 && j !== i) { const m = 0.5 * (zd[i] + zd[j]); zd[i] = m; }
    }
    const adj = opts.adj468;
    const smoothed = adj ? laplacian(zd, adj, 6, 0.55) : zd;
    // Re-standardise AFTER smoothing. Laplacian smoothing collapses the dynamic
    // range of the detector signal, and blending it as-is drags every landmark
    // toward the mean depth - which flattens the sides of the skull and ruins the
    // jaw hinge lever arm. Match the prior's standard deviation instead.
    let ms = 0;
    for (let i = 0; i < 478; i++) ms += smoothed[i];
    ms /= 478;
    let vs = 0;
    for (let i = 0; i < 478; i++) vs += (smoothed[i] - ms) ** 2;
    const ss = Math.sqrt(vs / 478) || 1;
    const target = Math.sqrt(vb / 478);
    for (let i = 0; i < 478; i++) z[i] = z[i] * (1 - blend) + (mb + (smoothed[i] - ms) * (target / ss)) * blend;
    opts.depthSign = sign;
  }
  // 4. graph smoothing keeps the surface C1 so shading has no facets.
  return opts.adj468 ? laplacian(z, opts.adj468, opts.smoothIters ?? 7, opts.smoothLambda ?? 0.42) : z;
}

/** Symmetry partner for every landmark (nearest point to its mirror image). */
export function symmetryPartners(V, cx, maxDist) {
  const n = V.length, out = new Int32Array(n).fill(-1);
  const lim = (maxDist || 1e9) ** 2;
  for (let k = 0; k < n; k++) {
    const mx = 2 * cx - V[k][0], my = V[k][1];
    let best = -1, bd = lim;
    for (let j = 0; j < n; j++) {
      if (j === k) continue;
      const dx = V[j][0] - mx, dy = V[j][1] - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = j; }
    }
    out[k] = best;
  }
  return out;
}

/**
 * Build the complete rig for one still image.
 * @param V array of 478 [x,y,zDetector] in canvas pixels
 * @param W,H canvas size
 * @param tess official tesselation edges (FaceLandmarker.FACE_LANDMARKS_TESSELATION)
 */
export function buildRig(V, W, H, tess, opts = {}) {
  if (V.length < 478) throw new Error('expected 478 face landmarks, got ' + V.length);

  // ---------- face frame ----------
  const topY = V[10][1], chinY = V[152][1], eyeY = (V[EYM[0][2]][1] + V[EYM[1][2]][1]) / 2;
  const FH = Math.hypot(V[152][0] - V[10][0], chinY - topY) || 1;
  const FW = Math.hypot(V[234][0] - V[454][0], V[234][1] - V[454][1]) || FH * 0.72;
  const MW = Math.hypot(V[61][0] - V[291][0], V[61][1] - V[291][1]) || FW * 0.42;
  const EW = [0, 1].map(n => Math.hypot(V[EYM[n][2]][0] - V[EYM[n][3]][0], V[EYM[n][2]][1] - V[EYM[n][3]][1]));
  const cx = (V[234][0] + V[454][0]) / 2;
  const cy = eyeY + FH * 0.03;
  const depthScale = opts.depthScale ?? 1;
  const frame = {
    cx, cy, topY, chinY, eyeY, FH, FW, MW, EW,
    // Anthropometric skull ellipsoid, expressed in face-width units:
    // bizygomatic width ~140 mm, biparietal ~155 mm, chin-to-vertex ~230 mm,
    // trichion-to-occiput ~190 mm. The eyes sit ~45 % down the skull.
    ax: FW * 0.55, ay: Math.max(FH * 0.62, (eyeY - topY) * 1.35),
    az: FW * 0.68 * depthScale,
    F: FW * (opts.focalScale ?? 4.2),
    lipY: V[13][1], mouthCX: (V[61][0] + V[291][0]) / 2, mouthCY: (V[13][1] + V[14][1]) / 2,
    G0: Math.hypot(V[13][0] - V[14][0], V[13][1] - V[14][1]),
  };
  // jaw hinge: the temporomandibular joints sit just in front of the ears.
  // The origin/axis are filled in below, in CANONICAL space - poseFrame applies
  // the hinge to back-projected points, so an image-space origin here would give
  // the mandible a lever arm hundreds of pixels long and swing it sideways.
  frame.tmjOffsetY = FH * 0.04;

  // ---------- vertices: face, hair shells, frame border, body grid ----------
  const V2 = V.slice(0, 478).map(p => [p[0], p[1], p[2] || 0]);
  const regionArr = new Array(478).fill(R_FACE);
  const hairArr = new Array(478).fill(0);
  const ringBase = [];
  const oc = [0, 0];
  OVAL.forEach(i => { oc[0] += V2[i][0] / OVAL.length; oc[1] += V2[i][1] / OVAL.length; });
  const ringWg = [];
  RINGS.forEach(([k, wg, hw], ri) => {
    OVAL.forEach(i => {
      const p = V2[i];
      const rx = oc[0] + (p[0] - oc[0]) * k, ry = oc[1] + (p[1] - oc[1]) * k;
      // A shell vertex that lands on the frame edge is background, not hair: it
      // is welded to the static border so the skirt between them cannot stretch.
      const clamped = rx <= 0.5 || rx >= W - 0.5 || ry <= 0.5 || ry >= H - 0.5;
      const x = clamp(rx, 0, W), y = clamp(ry, 0, H);
      // above the forehead the hair shell domes forward, at the sides it stays
      // on the silhouette plane - that is what makes the hair mass turn with
      // the skull instead of squashing flat.
      const phi = Math.atan2(p[0] - oc[0], -(p[1] - oc[1]));
      const dome = frame.az * [0.42, 0.3, 0.2][ri] * Math.pow(Math.max(0, Math.cos(phi)), 1.35);
      // deterministic sub-pixel jitter: keeps Delaunay away from degenerate input
      V2.push([x + ((ri * 0.37 + (i % 7) * 0.11) % 0.017), y + ((i * 0.29 + ri * 0.13) % 0.017), 0]);
      regionArr.push(clamped ? R_BORDER : R_HAIR);
      hairArr.push(clamped ? 0 : hw * clamp(1 - (p[1] - topY) / (FH * 0.9), 0, 1));
      ringWg.push(clamped ? 0 : wg);
      ringBase.push({ ring: ri, dome: clamped ? 0 : dome, x, y, wg: clamped ? 0 : wg });
    });
  });
  const BORDER = [[0, 0], [W / 2, 0], [W, 0], [W, H / 2], [W, H], [W / 2, H], [0, H], [0, H / 2],
    [W / 4, 0], [3 * W / 4, 0], [W / 4, H], [3 * W / 4, H], [0, H / 4], [0, 3 * H / 4], [W, H / 4], [W, 3 * H / 4]];
  const borderStart = V2.length;
  BORDER.forEach(([x, y]) => { V2.push([x, y, 0]); regionArr.push(R_BORDER); hairArr.push(0); ringWg.push(0); });
  const bodyStart = V2.length;
  {
    const y0 = chinY + FH * 0.12;
    if (H > y0 + 8) for (let r = 0; r < 5; r++) for (let q = 0; q < 9; q++) {
      V2.push([W * (q + 0.5) / 9, y0 + (H - y0) * (r + 0.5) / 5, 0]);
      regionArr.push(R_BODY); hairArr.push(0); ringWg.push(1);
    }
  }
  const n = V2.length;
  const region = Uint8Array.from(regionArr);
  const hairW = Float32Array.from(hairArr);
  // rotation weight taper of the hair shells: ring1 follows the skull, ring3
  // barely moves, so the photo background near the head does not shear.
  const wRing = new Float32Array(n).fill(1);
  for (let i = 0; i < ringWg.length; i++) wRing[478 + i] = ringWg[i];

  // ---------- depth prior + canonical back-projection ----------
  const { tris: faceTris, adj } = tessTriangles(tess, 478);
  const adj468 = adjacency(faceTris, 478);
  const sym = symmetryPartners(V2.slice(0, 478), cx, FH * 0.34);
  const Z = new Float32Array(n);
  const zf = depthPrior(V2, frame, { adj468, sym, zSign: opts.zSign, detectorBlend: opts.detectorBlend, depthSign: 0 });
  Z.set(zf.subarray(0, 478), 0);
  Z.set(ringBase.map(r => r.dome), 478); // hair shells
  // border/body stay on the z = 0 plane: they are the static world.

  const F = frame.F;
  const Xc = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const z = Z[i], k = (F - z) / F;
    Xc[i * 3] = (V2[i][0] - cx) * k;
    Xc[i * 3 + 1] = (V2[i][1] - cy) * k;
    Xc[i * 3 + 2] = z;
  }

  // Temporomandibular hinge, in the same canonical space the pose pass uses.
  // The origin is the CONDYLE, not the cheek landmark: it sits deep inside the
  // head, just inside the widest plane of the skull (z ~ 0.06*az). Using the soft
  // tissue depth of 234/454 would put the hinge almost straight above the chin,
  // and a rotation about that axis swings the chin backwards instead of dropping
  // it - the mouth would open by shrinking rather than by hinging.
  const oy = frame.tmjOffsetY * ((F - Z[234]) / F);
  frame.tmjOrigin = [(Xc[234 * 3] + Xc[454 * 3]) / 2, (Xc[234 * 3 + 1] + Xc[454 * 3 + 1]) / 2 + oy, frame.az * 0.06];
  frame.tmjAxis = norm3([Xc[454 * 3] - Xc[234 * 3], Xc[454 * 3 + 1] - Xc[234 * 3 + 1], Xc[454 * 3 + 2] - Xc[234 * 3 + 2]]);

  // ---------- weight fields ----------
  const wRot = new Float32Array(n), wExpr = new Float32Array(n), jawW = new Float32Array(n), dragW = new Float32Array(n);
  const neckLo = chinY - FH * 0.02, neckHi = chinY + FH * 0.38;
  const jawLo = frame.lipY - FH * 0.035, jawHi = frame.lipY + FH * 0.11;
  for (let i = 0; i < n; i++) {
    const y = V2[i][1], r = region[i];
    if (r === R_FACE || r === R_HAIR) {
      wRot[i] = (1 - 0.78 * ramp(neckLo, neckHi, y)) * wRing[i];
      wExpr[i] = i < 478 ? 1 : 0;
      jawW[i] = i < 478 ? ramp(jawLo, jawHi, y) : 0;
      dragW[i] = 1;
    } else { wRot[i] = 0; wExpr[i] = 0; jawW[i] = 0; dragW[i] = 0; }
  }
  // mandible points get the full hinge even if they sit above the ramp start
  for (const i of MANDIBLE) if (i < n) jawW[i] = Math.max(jawW[i], 0.85);

  // ---------- triangulation: official face topology + Delaunay shell ----------
  const del = Delaunator.from(V2, p => p[0], p => p[1]);
  const dt = del.triangles;
  const faceSet = new Set();
  const key = (a, b, c) => { const s = [a, b, c].sort((x, y) => x - y); return s[0] * 1000000 + s[1] * 1000 + s[2]; };
  for (let i = 0; i < faceTris.length; i += 3) faceSet.add(key(faceTris[i], faceTris[i + 1], faceTris[i + 2]));
  // Without the official topology the inner Delaunay triangles are the only
  // coverage the face has, so they must all be kept (otherwise the head would be
  // full of holes). With it, they would smear lips and lids together, so only the
  // iris triangles - which the official list does not contain - are added.
  const hasOfficial = faceTris.length > 300;
  const tv = [], kind = [];
  const push = (a, b, c, k) => { tv.push(a, b, c); kind.push(k); };
  for (let i = 0; i < faceTris.length; i += 3) push(faceTris[i], faceTris[i + 1], faceTris[i + 2], 0);
  for (let i = 0; i < dt.length; i += 3) {
    const a = dt[i], b = dt[i + 1], c = dt[i + 2];
    const inner = a < 478 && b < 478 && c < 478;
    if (inner) {
      // only keep inner Delaunay triangles that the official topology misses
      // (the iris points), otherwise the lips and lids would smear together.
      if (faceSet.has(key(a, b, c))) continue;
      if (hasOfficial && !(a >= 468 || b >= 468 || c >= 468)) continue;
      push(a, b, c, 0);
      continue;
    }
    const hasStatic = region[a] >= R_BORDER || region[b] >= R_BORDER || region[c] >= R_BORDER;
    const hasFace = region[a] === R_FACE || region[b] === R_FACE || region[c] === R_FACE;
    push(a, b, c, hasStatic ? (hasFace ? 2 : 3) : 1);
  }
  const count = kind.length;
  const triV = Int32Array.from(tv);
  const triKind = Uint8Array.from(kind);

  // rest normals, rest areas, rest lambert, rest depth
  const n0v = new Float32Array(count * 3), area0 = new Float32Array(count), lam0 = new Float32Array(count), zc0 = new Float32Array(count);
  for (let t = 0; t < count; t++) {
    const a = triV[t * 3], b = triV[t * 3 + 1], c = triV[t * 3 + 2];
    const ax = Xc[a * 3], ay = Xc[a * 3 + 1], az_ = Xc[a * 3 + 2];
    const u = [Xc[b * 3] - ax, Xc[b * 3 + 1] - ay, Xc[b * 3 + 2] - az_];
    const v = [Xc[c * 3] - ax, Xc[c * 3 + 1] - ay, Xc[c * 3 + 2] - az_];
    const nrm = cross(u, v);
    const s = nrm[2] < 0 ? -1 : 1; // orient so front-facing triangles have +z
    const nn = norm3([nrm[0] * s, nrm[1] * s, nrm[2] * s]);
    n0v[t * 3] = nn[0]; n0v[t * 3 + 1] = nn[1]; n0v[t * 3 + 2] = nn[2];
    lam0[t] = Math.max(0, dot3(nn, LIGHT));
    area0[t] = 0.5 * ((V2[b][0] - V2[a][0]) * (V2[c][1] - V2[a][1]) - (V2[b][1] - V2[a][1]) * (V2[c][0] - V2[a][0]));
    zc0[t] = (az_ + Xc[b * 3 + 2] + Xc[c * 3 + 2]) / 3;
  }

  // painter's order at rest: far first (small z), near last
  const order = Int32Array.from({ length: count }, (_, i) => i);
  const zOf = Array.from(zc0);
  const headIdx = [], statIdx = [];
  for (let t = 0; t < count; t++) (triKind[t] >= 2 ? statIdx : headIdx).push(t);
  headIdx.sort((p, q) => zOf[p] - zOf[q]);
  order.set(Int32Array.from(statIdx), 0);
  order.set(Int32Array.from(headIdx), statIdx.length);
  const headStart = statIdx.length;

  // ---------- edge list for the stretch limiter ----------
  const emap = new Map();
  for (let t = 0; t < count; t++) for (let e = 0; e < 3; e++) {
    const a = triV[t * 3 + e], b = triV[t * 3 + (e + 1) % 3];
    const k = a < b ? a * n + b : b * n + a;
    if (!emap.has(k)) emap.set(k, [a, b]);
  }
  const eArr = Array.from(emap.values());
  const edgeA = Int32Array.from(eArr, e => e[0]), edgeB = Int32Array.from(eArr, e => e[1]);
  const edgeLen = new Float32Array(eArr.length);
  eArr.forEach(([a, b], i) => { edgeLen[i] = Math.hypot(V2[a][0] - V2[b][0], V2[a][1] - V2[b][1]); });
  // Which edges the stretch solver is allowed to touch. Only edges inside one
  // region: a hair shell swings further than the face under pitch (it has real
  // depth), and letting that constraint pull on the face outline drags the whole
  // silhouette off its projection. Cross-region seams smear instead - they are
  // hairline / neck / background bands where a smear reads as motion blur.
  const minLen = frame.FH * 0.01;
  const edgeSolve = new Uint8Array(eArr.length);
  eArr.forEach(([a, b], i) => { edgeSolve[i] = (region[a] === region[b] && edgeLen[i] >= minLen) ? 1 : 0; });

  // ---------- coarse mesh for the shading pass ----------
  const shade = buildShadeMesh(V2, Xc, frame);

  frame.maxillaIds = Array.from(MAXILLA).filter(i => i < 478);
  // anchor groups used by the stitching pass (eyes + maxilla must not drift)
  // Falloff is deliberately tight: one eye width / 0.7 mouth widths. A wide
  // sigma bleeds an eye correction into the mouth (and back), which reads as the
  // whole mid-face slowly swimming instead of the anchor holding still.
  const stitchGroups = [
    { ids: EYE[0].slice(), sigma: frame.EW[0] * 1.05 },
    { ids: EYE[1].slice(), sigma: frame.EW[1] * 1.05 },
    { ids: frame.maxillaIds.slice(), sigma: MW * 0.7 },
  ];

  return {
    W, H, n, nFace: 478, frame, opts, shown: SHOWN, stitchGroups,
    rest: Float32Array.from(V2.flat()),
    V: V2, Z, Xc, region, wRot, wExpr, jawW, hairW, dragW, sym,
    tri: { count, v: triV, kind: triKind, n0: n0v, area0, lam0, zc0, headStart, order: Int32Array.from(order) },
    edges: { a: edgeA, b: edgeB, len0: edgeLen, solve: edgeSolve, count: edgeA.length },
    adj468, shade,
    borderStart, bodyStart, ringStart: 478,
    mouthCX: frame.mouthCX, mouthCY: frame.mouthCY,
  };
}

/** Stratified subset of head vertices + their own Delaunay, for cheap shading. */
export function buildShadeMesh(V2, Xc, frame, stride = 4) {
  const set = new Set([...RIG, ...OVAL, ...EYE[0], ...EYE[1], ...BROW, ...LIP, ...CHEEK, ...NOSE_RIDGE, ...FOREHEAD, ...CHIN]);
  for (let i = 0; i < 468; i += stride) set.add(i);
  const idx = Array.from(set).filter(i => i < V2.length);
  const del = Delaunator.from(idx, i => V2[i][0], i => V2[i][1]);
  const tris = del.triangles;
  const out = [];
  for (let t = 0; t < tris.length; t += 3) {
    const a = idx[tris[t]], b = idx[tris[t + 1]], c = idx[tris[t + 2]];
    if (a >= 478 || b >= 478 || c >= 478) continue; // shading only on the face
    const u = [Xc[b * 3] - Xc[a * 3], Xc[b * 3 + 1] - Xc[a * 3 + 1], Xc[b * 3 + 2] - Xc[a * 3 + 2]];
    const v = [Xc[c * 3] - Xc[a * 3], Xc[c * 3 + 1] - Xc[a * 3 + 1], Xc[c * 3 + 2] - Xc[a * 3 + 2]];
    const nrm = cross(u, v);
    const s = nrm[2] < 0 ? -1 : 1;
    const nn = norm3([nrm[0] * s, nrm[1] * s, nrm[2] * s]);
    out.push(a, b, c, nn[0], nn[1], nn[2], Math.max(0, dot3(nn, LIGHT)));
  }
  return { data: Float32Array.from(out), count: out.length / 7, idx };
}

/** Face-oval polygon in the projected point array (used to clip shading/FX). */
export function headSilhouette(P) {
  const pts = [];
  for (let k = 0; k < OVAL.length; k++) pts.push(P[OVAL[k] * 2], P[OVAL[k] * 2 + 1]);
  return pts;
}
