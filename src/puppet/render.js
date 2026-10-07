/**
 * Renderer: affine triangle warping with flip-free unfolding, depth-ordered
 * compositing, differential relighting, seam blending and ROI enhancement.
 *
 * Occlusion strategy (this is what stops the "bent card" look):
 *   triangles are painted far -> near using their rotated 3D depth, so the nose
 *   passes in front of the far cheek and the near side of the skull passes in
 *   front of the far side. Nothing is ever removed, so no holes can appear; the
 *   optional back-face cull is off by default and exists for inspection.
 *   Inverted triangles are "unfolded" (a vertex is reflected about the opposite
 *   edge) instead of being smeared across the frame, and a global edge-length
 *   relaxation caps how far any texel can be pulled.
 */

import { clamp } from './math3d.js';
import { polyPath } from './mouth.js';

export function signedArea(p, q, r) { return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x); }

function reflect(v, p, q) {
  const dx = q.x - p.x, dy = q.y - p.y, l2 = dx * dx + dy * dy || 1e-9;
  const t = ((v.x - p.x) * dx + (v.y - p.y) * dy) / l2;
  const fx = p.x + t * dx, fy = p.y + t * dy;
  return { x: 2 * fx - v.x, y: 2 * fy - v.y };
}

/**
 * Restore an orientation-preserving destination triangle. Returns the corrected
 * triple, or null if the triangle is degenerate and should be skipped.
 */
export function unfold(s0, s1, s2, d0, d1, d2) {
  const ss = signedArea(s0, s1, s2);
  if (Math.abs(ss) < 1e-6) return null;
  const want = ss > 0 ? 1 : -1;
  const D = [d0, d1, d2];
  for (let iter = 0; iter < 2; iter++) {
    const A = signedArea(D[0], D[1], D[2]);
    if (Math.abs(A) < 1e-6) return null;
    if ((A > 0 ? 1 : -1) === want) return D;
    let bi = 0, bh = Infinity;
    for (let i = 0; i < 3; i++) {
      const p = D[(i + 1) % 3], q = D[(i + 2) % 3];
      const len = Math.hypot(q.x - p.x, q.y - p.y) || 1e-6;
      const h = Math.abs(signedArea(p, q, D[i])) / len;
      if (h < bh) { bh = h; bi = i; }
    }
    D[bi] = reflect(D[bi], D[(bi + 1) % 3], D[(bi + 2) % 3]);
  }
  const A = signedArea(D[0], D[1], D[2]);
  return (A > 0 ? 1 : -1) === want ? D : null;
}

/**
 * Draw one source triangle into its destination with an affine texture map.
 * The clip polygon is expanded a fraction of a pixel outward so neighbouring
 * triangles overlap instead of leaving shimmering seams.
 */
export function warpTri(ctx, src, s0, s1, s2, d0, d1, d2, pad) {
  const u1 = s1.x - s0.x, v1 = s1.y - s0.y, u2 = s2.x - s0.x, v2 = s2.y - s0.y;
  const det = u1 * v2 - v1 * u2;
  if (Math.abs(det) < 1e-6) return false;
  const k = 1 / det;
  const p1x = d1.x - d0.x, p2x = d2.x - d0.x, p1y = d1.y - d0.y, p2y = d2.y - d0.y;
  const a = (p1x * v2 - v1 * p2x) * k, c = (u1 * p2x - p1x * u2) * k;
  const b = (p1y * v2 - v1 * p2y) * k, d = (u1 * p2y - p1y * u2) * k;
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || !Number.isFinite(d)) return false;
  const e = d0.x - a * s0.x - c * s0.y, f = d0.y - b * s0.x - d * s0.y;
  const gx = (d0.x + d1.x + d2.x) / 3, gy = (d0.y + d1.y + d2.y) / 3;
  ctx.save();
  ctx.beginPath();
  const pts = [d0, d1, d2];
  for (let i = 0; i < 3; i++) {
    const p = pts[i], vx = p.x - gx, vy = p.y - gy, l = Math.hypot(vx, vy) || 1;
    const X = p.x + (vx / l) * pad, Y = p.y + (vy / l) * pad;
    i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
  }
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, b, c, d, e, f);
  const x0 = Math.max(0, Math.floor(Math.min(s0.x, s1.x, s2.x)) - 2);
  const y0 = Math.max(0, Math.floor(Math.min(s0.y, s1.y, s2.y)) - 2);
  const x1 = Math.min(src.width, Math.ceil(Math.max(s0.x, s1.x, s2.x)) + 2);
  const y1 = Math.min(src.height, Math.ceil(Math.max(s0.y, s1.y, s2.y)) + 2);
  if (x1 > x0 && y1 > y0) ctx.drawImage(src, x0, y0, x1 - x0, y1 - y0, x0, y0, x1 - x0, y1 - y0);
  ctx.restore();
  return true;
}

/**
 * Paint the whole mesh: static shell first, then head triangles far -> near.
 * `st.P` holds the projected positions produced by poseFrame.
 */
export function drawMesh(ctx, rig, st, src, opts = {}) {
  const { pad = 0.85, cull = false, limit = Infinity, bridge = true } = opts;
  const P = st.P, V = rig.V, tv = rig.tri.v, order = st.order, cullFlags = st.cull;
  const kind = rig.tri.kind;
  const S = rig.S || rig.V.map(p => ({ x: p[0], y: p[1] }));
  rig.S = S;
  let drawn = 0;
  for (let k = 0; k < order.length; k++) {
    if (drawn >= limit) break;
    const t = order[k];
    if (cull && cullFlags[t]) continue;
    // bridge triangles tie the head to the static frame border. With an inpainted
    // plate underneath they only smear background, so they are skipped.
    if (!bridge && kind[t] >= 2) continue;
    const i0 = tv[t * 3], i1 = tv[t * 3 + 1], i2 = tv[t * 3 + 2];
    const d0 = { x: P[i0 * 2], y: P[i0 * 2 + 1] };
    const d1 = { x: P[i1 * 2], y: P[i1 * 2 + 1] };
    const d2 = { x: P[i2 * 2], y: P[i2 * 2 + 1] };
    if (!Number.isFinite(d0.x) || !Number.isFinite(d1.x) || !Number.isFinite(d2.x)) continue;
    const D = unfold(S[i0], S[i1], S[i2], d0, d1, d2);
    if (!D) continue;
    if (warpTri(ctx, src, S[i0], S[i1], S[i2], D[0], D[1], D[2], pad)) drawn++;
  }
  return drawn;
}

/**
 * Differential relighting overlay. Renders a coarse Gouraud-ish gray map into a
 * low-resolution buffer and composites it with `soft-light`, so mid-gray is a
 * no-op: the photo's own lighting is untouched at neutral and only the change
 * caused by the turn is added.
 */
export function drawShading(ctx, rig, st, o) {
  const { off, strength, scale = 0.34, blur = 5, silhouette } = o;
  if (strength <= 0.005) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const w = Math.max(2, Math.round(W * scale)), h = Math.max(2, Math.round(H * scale));
  off.width = w; off.height = h;
  const x = off.getContext('2d');
  x.setTransform(scale, 0, 0, scale, 0, 0);
  x.clearRect(0, 0, W, H);
  const d = rig.shade.data, lam = st.lamTri, P = st.P;
  for (let t = 0; t < rig.shade.count; t++) {
    const o7 = t * 7, a = d[o7], b = d[o7 + 1], c = d[o7 + 2];
    const dl = clamp(lam[t], -0.85, 0.85);
    if (Math.abs(dl) < 0.004) continue;
    const v = Math.round(clamp(128 + dl * 190, 0, 255));
    x.fillStyle = `rgb(${v},${v},${v})`;
    x.beginPath();
    x.moveTo(P[a * 2], P[a * 2 + 1]);
    x.lineTo(P[b * 2], P[b * 2 + 1]);
    x.lineTo(P[c * 2], P[c * 2 + 1]);
    x.closePath();
    x.fill();
  }
  ctx.save();
  if (silhouette) { polyPath(ctx, silhouette, 1.14); ctx.clip(); }
  ctx.globalCompositeOperation = 'soft-light';
  ctx.filter = `blur(${blur}px)`;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(off, 0, 0, w, h, 0, 0, W, H);
  ctx.filter = 'none';
  ctx.restore();
}

/**
 * Seamless-composite stage (gradient-domain blending, the cheap honest version):
 * the background plate is re-blurred right up to the head silhouette and a soft
 * contact shadow is laid along it, so the head sits IN the plate instead of
 * being cut out on top of it.
 */
export function drawSeam(ctx, plate, silhouette, o = {}) {
  if (!plate) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const ao = o.ao === undefined ? 0.5 : o.ao;
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  polyPath(ctx, silhouette, 1.0);
  ctx.clip('evenodd');
  ctx.filter = `blur(${o.blur || 2.5}px)`;
  ctx.globalAlpha = o.blend === undefined ? 0.55 : o.blend;
  ctx.drawImage(plate, 0, 0);
  ctx.filter = 'none';
  ctx.globalAlpha = 1;
  ctx.restore();
  if (ao > 0.01) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    polyPath(ctx, silhouette, 1.0);
    ctx.clip('evenodd');
    ctx.filter = 'blur(4px)';
    ctx.strokeStyle = `rgba(10,6,14,${0.5 * ao})`;
    ctx.lineWidth = Math.max(4, o.width || 7);
    polyPath(ctx, silhouette, 1.0);
    ctx.stroke();
    ctx.filter = 'none';
    ctx.restore();
  }
}

/**
 * Local restoration pass on the mouth and eye ROIs: high-pass (difference of a
 * blurred copy) re-applied with overlay blending. This is the browser analogue
 * of the enhancement stage in VideoReTalking / CodeFormer pipelines - it recovers
 * the detail the warp filter softens, on the regions the eye actually checks.
 */
export function enhanceROIs(ctx, rois, amount) {
  if (amount <= 0.01) return;
  const x = ctx;
  for (const r of rois) {
    const w = Math.max(4, Math.round(r.w)), h = Math.max(4, Math.round(r.h));
    if (r.off.width !== w || r.off.height !== h) { r.off.width = w; r.off.height = h; }
    const o = r.off.getContext('2d');
    o.setTransform(1, 0, 0, 1, 0, 0);
    o.clearRect(0, 0, w, h);
    o.drawImage(ctx.canvas, r.x, r.y, w, h, 0, 0, w, h);
    o.filter = `blur(${r.radius || 2}px)`;
    o.globalCompositeOperation = 'difference';
    o.drawImage(r.off, 0, 0);   // |A - blur(A)|: the high-pass residual
    o.filter = 'none';
    o.globalCompositeOperation = 'source-over';
    x.save();
    x.globalCompositeOperation = 'overlay';
    x.globalAlpha = clamp(amount, 0, 1);
    x.drawImage(r.off, 0, 0, w, h, r.x, r.y, w, h);
    x.restore();
  }
}

/** Landmark overlay for the draggable dots (radius animates instead of popping). */
export function drawLandmarks(ctx, P, shown, o) {
  const { idx, color } = shown;
  const r = o.radius || 2.8, halo = o.halo || 4.2;
  for (let k = 0; k < idx.length; k++) {
    const v = idx[k], x = P[v * 2], y = P[v * 2 + 1];
    const hot = k === o.drag || k === o.hover;
    const rr = hot ? r * 1.9 : r;
    ctx.beginPath(); ctx.arc(x, y, hot ? halo * 1.9 : halo, 0, 7);
    ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, rr, 0, 7);
    ctx.fillStyle = color[k]; ctx.fill();
  }
}

/** Mirrored driver overlay on the webcam canvas. */
export function drawDriverOverlay(ctx, camP, shown, scale = 1) {
  const { idx, color } = shown;
  for (let k = 0; k < idx.length; k++) {
    const v = idx[k], p = camP[v];
    if (!p) continue;
    ctx.beginPath(); ctx.arc(p[0], p[1], 3.4 * scale, 0, 7); ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fill();
    ctx.beginPath(); ctx.arc(p[0], p[1], 2.3 * scale, 0, 7); ctx.fillStyle = color[k]; ctx.fill();
  }
}
