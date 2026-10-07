/**
 * Mouth studio: oral cavity, rigid dental arches, tongue and lips.
 *
 * The single hardest part of animating a still photo is the mouth interior: the
 * source image contains one tooth configuration and no tongue geometry at all.
 * This module builds it the way a parametric (FLAME/3DMM) oral rig would, then
 * renders it procedurally:
 *
 *   - the dental arch is a RIGID row: its frame (centre, axis, width, arc
 *     length) and the lip spline it hangs from are temporally low-passed, so
 *     teeth do not shimmer or crawl between frames,
 *   - per-tooth crown geometry with real proportions (wide centrals, pointed
 *     canines, foreshortened premolars), enamel texture, interproximal shadows,
 *     corner ambient occlusion, translucent incisal edges and a specular sweep
 *     that moves with head yaw,
 *   - the arch is sampled from the projected 3D depth, so the back teeth
 *     foreshorten and darken as the head turns instead of staying flat,
 *   - the tongue is a viseme-driven parametric surface (tip raise for /l th t d/,
 *     back raise for /k g/, low+wide for /a/, narrowed for /u/), drawn BEHIND
 *     the teeth so occlusion is anatomically correct,
 *   - every visibility decision is a smooth gate: nothing pops on or off.
 *
 * Geometry builders are pure (they return point arrays) so they are unit tested;
 * the draw functions take any CanvasRenderingContext2D.
 */

import { clamp, lerp, ramp } from './math3d.js';
import { LIP, UP, LO, GROUPS } from './landmarks.js';

/** Tooth classes from the midline outwards: central, lateral, canine, pm1, pm2. */
export const TOOTH_W = [0.155, 0.108, 0.104, 0.078, 0.055];
export const TOOTH_TIP = [0.03, 0.02, 0.2, 0.05, 0.03];
export const UPPER_H = [1, 0.86, 0.98, 0.72, 0.58];
export const LOWER_H = [0.82, 0.86, 1.0, 0.78, 0.6];
export const ARCH_SPAN = 0.93; // fraction of the arch length covered by teeth

const dist = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);

/** Point at fraction t along a projected landmark loop (piecewise linear). */
export function splinePoint(ids, D, t) {
  t = clamp(t, 0, 1);
  const f = t * (ids.length - 1), i = Math.min(ids.length - 2, Math.floor(f)), u = f - i;
  const a = D[ids[i]], b = D[ids[i + 1]];
  return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, i, u };
}

/**
 * The arch frame + arc-length parameterised spline, temporally stabilised.
 * `prev` is the previous ArchFrame (or null); everything is low-passed in the
 * arch's own local frame so head motion does not fight the smoothing.
 */
export function fitArch(D, Pz, prev, dt, opts = {}) {
  const tau = opts.tau ?? 0.055;
  const k = 1 - Math.exp(-Math.max(dt, 0.0005) / tau);
  const a = D[61], b = D[291];
  const width = Math.max(1e-3, dist(a, b));
  const ang = Math.atan2(b.y - a.y, b.x - a.x);
  const ax = { x: Math.cos(ang), y: Math.sin(ang) };
  const dn = { x: -Math.sin(ang), y: Math.cos(ang) }; // +dn points into the mouth
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const toLocal = p => ({ u: (p.x - mid.x) * ax.x + (p.y - mid.y) * ax.y, v: (p.x - mid.x) * dn.x + (p.y - mid.y) * dn.y });

  const raw = { mid, ang, ax, dn, width, up: [], lo: [] };
  for (const ids of [UP, LO]) {
    const out = ids === UP ? raw.up : raw.lo;
    // arc-length parameterisation in the local frame
    const pts = ids.map(i => ({ l: toLocal(D[i]), z: Pz ? Pz[i] : 0, i }));
    let total = 0;
    for (let j = 1; j < pts.length; j++) total += Math.hypot(pts[j].l.u - pts[j - 1].l.u, pts[j].l.v - pts[j - 1].l.v);
    let acc = 0;
    pts.forEach((p, j) => {
      if (j > 0) acc += Math.hypot(p.l.u - pts[j - 1].l.u, p.l.v - pts[j - 1].l.v);
      p.s = total > 1e-6 ? acc / total : j / (pts.length - 1);
      out.push(p);
    });
  }
  if (prev && prev.ready) {
    // frame smoothing: unwrap the angle so a rotation cannot jump by 2pi
    let dAng = raw.ang - prev.ang;
    while (dAng > Math.PI) dAng -= Math.PI * 2;
    while (dAng < -Math.PI) dAng += Math.PI * 2;
    raw.ang = prev.ang + dAng * k;
    raw.mid = { x: prev.mid.x + (mid.x - prev.mid.x) * k, y: prev.mid.y + (mid.y - prev.mid.y) * k };
    raw.width = prev.width + (width - prev.width) * k;
    raw.ax = { x: Math.cos(raw.ang), y: Math.sin(raw.ang) };
    raw.dn = { x: -Math.sin(raw.ang), y: Math.cos(raw.ang) };
    const mix = (arr, prevArr) => arr.forEach((p, j) => {
      const q = prevArr[j];
      if (!q) return;
      p.l = { u: q.l.u + (p.l.u - q.l.u) * k, v: q.l.v + (p.l.v - q.l.v) * k };
      p.z = q.z + (p.z - q.z) * k;
    });
    mix(raw.up, prev.up); mix(raw.lo, prev.lo);
  }
  raw.ready = true;
  raw.toWorld = l => ({ x: raw.mid.x + l.u * raw.ax.x + l.v * raw.dn.x, y: raw.mid.y + l.u * raw.ax.y + l.v * raw.dn.y });
  raw.at = (side, s) => {
    const pts = side > 0 ? raw.up : raw.lo;
    s = clamp(s, 0, 1);
    for (let j = 1; j < pts.length; j++) {
      if (pts[j].s >= s || j === pts.length - 1) {
        const p = pts[j - 1], q = pts[j];
        const span = (q.s - p.s) || 1e-6, t = clamp((s - p.s) / span, 0, 1);
        return {
          l: { u: lerp(p.l.u, q.l.u, t), v: lerp(p.l.v, q.l.v, t) },
          z: lerp(p.z, q.z, t),
          tan: { u: q.l.u - p.l.u, v: q.l.v - p.l.v },
        };
      }
    }
    return { l: pts[0].l, z: 0, tan: { u: 1, v: 0 } };
  };
  /** depth (toward viewer, px) at normalised arc length s of one arch */
  raw.depth = (side, s) => raw.at(side, s).z;
  return raw;
}

/**
 * Crown geometry for one arch. Returns an array of teeth, each with its outline,
 * incisal edge, shading terms and the texture transform basis.
 */
export function archTeeth(arch, side, opts = {}) {
  const {
    thickness = 1, gap = 10, widths = TOOTH_W, heights = side > 0 ? UPPER_H : LOWER_H,
    tips = TOOTH_TIP, gum = 0.1, depthRef = 0, depthSpan = 1, yaw = 0, light = -0.35,
    span = ARCH_SPAN, jitter = 0.014,
  } = opts;
  const h = Math.max(0.5, thickness);
  const teeth = [];
  let cum = 0;
  for (let j = 0; j < widths.length; j++) {
    const w = widths[j];
    for (const sd of [-1, 1]) {
      // TOOTH_W sums to 0.5 per side, so `span` maps that onto the arch length
      const s0 = 0.5 + sd * cum * span, s1 = 0.5 + sd * (cum + w) * span;
      const A = arch.at(side, clamp(Math.min(s0, s1), 0, 1));
      const B = arch.at(side, clamp(Math.max(s0, s1), 0, 1));
      const Aw = arch.toWorld(A.l), Bw = arch.toWorld(B.l);
      // foreshortening + shading from the projected arch depth
      const zc = (A.z + B.z) * 0.5;
      const depth = clamp(0.5 + (zc - depthRef) / (2 * (depthSpan || 1)), 0.05, 1);
      const fs = 0.62 + 0.38 * depth;                       // vertical squash toward the corners
      const hh = h * heights[j] * fs * (1 + (((j * 13 + (sd > 0 ? 5 : 0)) % 7) - 3) * jitter);
      const nrm = { x: -(Bw.y - Aw.y), y: (Bw.x - Aw.x) };
      const nl = Math.hypot(nrm.x, nrm.y) || 1;
      nrm.x /= nl; nrm.y /= nl;
      const off = (a, o) => ({ x: a.x + nrm.x * o * side, y: a.y + nrm.y * o * side });
      const gm = h * gum;
      const A0 = off(Aw, gm), B0 = off(Bw, gm), A1 = off(Aw, gm + hh), B1 = off(Bw, gm + hh);
      const p1 = { x: lerp(A0.x, A1.x, 0.9), y: lerp(A0.y, A1.y, 0.9) };
      const p4 = { x: lerp(B0.x, B1.x, 0.9), y: lerp(B0.y, B1.y, 0.9) };
      const p2 = { x: lerp(A1.x, B1.x, 0.1), y: lerp(A1.y, B1.y, 0.1) };
      const p3 = { x: lerp(A1.x, B1.x, 0.9), y: lerp(A1.y, B1.y, 0.9) };
      const cm = { x: (A1.x + B1.x) / 2, y: (A1.y + B1.y) / 2 };
      const Cc = { x: cm.x + nrm.x * side * hh * (tips[j] + 0.02), y: cm.y + nrm.y * side * hh * (tips[j] + 0.02) };
      const centre = Math.abs((s0 + s1) / 2 - 0.5) * 2;     // 0 midline, 1 corner
      const lambert = clamp(0.55 + 0.45 * depth + light * (sd > 0 ? 1 : -1) * Math.sin(yaw) * 0.5, 0.15, 1.35);
      teeth.push({
        j, sd, s0, s1, A0, B0, A1, B1, p1, p2, p3, p4, Cc, hh,
        crown: [A0, p1, A1, p2, Cc, p3, B1, p4, B0],
        edge: [p1, A1, p2, Cc, p3, B1, p4],
        ao: clamp(Math.pow(centre / 0.85, 1.5) * 0.62 + 0.05 + (side < 0 ? 0.16 : 0), 0, 0.78),
        lambert, depth, centre,
        texBasis: { ox: A0.x, oy: A0.y, ex: (B0.x - A0.x) / 48, ey: (B0.y - A0.y) / 48, fx: (A1.x - A0.x) / 120, fy: (A1.y - A0.y) / 120 },
      });
      cum += w;
    }
  }
  return teeth;
}

/** Gum band geometry (visible when the lip lifts or the jaw drops). */
export function gumBand(arch, side, teeth, opts = {}) {
  const { thickness = 8, dn = arch.dn } = opts;
  if (!teeth.length) return null;
  const gum = teeth.map(t => t.A0);
  return {
    path: [...gum.map(p => ({ x: p.x - dn.x * thickness * side, y: p.y - dn.y * thickness * side })), ...gum.slice().reverse()],
    line: gum, dn,
  };
}

/**
 * Viseme-driven tongue surface. `ctl` comes from the tracker, the audio analysis
 * or a preset: {amount, tipUp, backUp, wide, protrude, yaw}.
 */
export function tongueGeom(D, arch, ctl = {}) {
  const amount = clamp(ctl.amount ?? 0, 0, 1.4);
  const tipUp = clamp(ctl.tipUp ?? 0, 0, 1), backUp = clamp(ctl.backUp ?? 0, 0, 1);
  const wide = clamp(ctl.wide ?? 0.5, 0, 1), protrude = clamp(ctl.protrude ?? 0, 0, 1);
  const gap = ctl.gap ?? 10, mw = ctl.mw ?? arch.width;
  const chin = D[152], brow = D[10], m = D[14];
  const cl = Math.hypot(chin.x - brow.x, chin.y - brow.y) || 1;
  let dn = { x: (chin.x - brow.x) / cl, y: (chin.y - brow.y) / cl };
  // /l th t d/: the tip rotates up toward the alveolar ridge
  const up = { x: -arch.dn.x, y: -arch.dn.y };
  dn = norm({ x: lerp(dn.x, up.x, tipUp * 0.85), y: lerp(dn.y, up.y, tipUp * 0.85) });
  const rt = { x: -dn.y, y: dn.x };
  const base = { x: m.x - dn.x * gap * 0.32, y: m.y - dn.y * gap * 0.32 };
  const L = mw * (0.34 + 0.34 * amount) * (1 + protrude * 0.55) + gap * 0.24;
  const hw = mw * (0.15 + 0.09 * wide) * (1 + amount * 0.22);
  const P = (r, d) => ({ x: base.x + rt.x * r + dn.x * d, y: base.y + rt.y * r + dn.y * d });
  const hump = backUp * L * 0.22 + tipUp * L * 0.05;
  return {
    base, dir: dn, right: rt, L, hw, amount, tipUp, backUp, protrude,
    outline: [P(-hw * 0.85, 0), P(-hw * 1.05, L * 0.45), P(-hw * (0.9 - hump * 0.1), L * 0.95), P(0, L),
      P(hw * (0.9 - hump * 0.1), L * 0.95), P(hw * 1.05, L * 0.45), P(hw * 0.85, 0)],
    tip: P(0, L), groove: [P(0, L * 0.1), P(0, L * 0.78)],
    hump: P(0, L * 0.42 - hump * 0.35),
    centre: P(0, L * 0.45),
  };
}
function norm(v) { const l = Math.hypot(v.x, v.y) || 1; return { x: v.x / l, y: v.y / l }; }

/** Interpolated control points -> a smooth closed bezier path on the context. */
export function smoothPath(ctx, pts, close = true, tension = 0.35) {
  const n = pts.length;
  if (n < 3) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
    const c1 = { x: p1.x + (p2.x - p0.x) * tension, y: p1.y + (p2.y - p0.y) * tension };
    const c2 = { x: p2.x - (p3.x - p1.x) * tension, y: p2.y - (p3.y - p1.y) * tension };
    ctx.bezierCurveTo(c1.x, c1.y, c2.x, c2.y, p2.x, p2.y);
    if (!close && i === n - 2) break;
  }
  if (close) ctx.closePath();
}

export function polyPath(ctx, pts, scale = 1, cx = 0, cy = 0) {
  ctx.beginPath();
  if (scale === 1 && !cx) { pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath(); return; }
  let mx = cx, my = cy;
  if (!cx && !cy) { pts.forEach(p => { mx += p.x; my += p.y; }); mx /= pts.length; my /= pts.length; }
  pts.forEach((p, i) => {
    const x = mx + (p.x - mx) * scale, y = my + (p.y - my) * scale;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.closePath();
}

/** The wet/dry border polygon of the mouth (what the cavity is clipped to). */
export const innerMouth = D => LIP.map(i => D[i]);
export const outerMouth = D => GROUPS.lips.map(i => D[i]);

/**
 * Full oral composite. `G` carries everything the frame produced:
 *   { D, arch, inner, upper, lower, gumUpper, tongue, amt, tex, W, H,
 *     lum, yaw, bright, warm }
 * `amt` holds the smooth gates: {open, gap, teeth, lower, gum, tongue, kiss, ao, wet, seam}.
 */
export function drawMouth(ctx, G) {
  const { arch, upper, lower, tongue, amt, tex, D, W, H, lum = 0.7, yaw = 0, bright = 1, warm = 0 } = G;
  const mw = arch.width, gap = amt.gap, toothH = G.thickness || 0;
  const a = D[61], b = D[291];
  const dn = arch.dn, ang = arch.ang;
  ctx.save();
  polyPath(ctx, G.inner, 1.03);
  ctx.clip();

  // ---- cavity ----
  let g = ctx.createLinearGradient(D[13].x, D[13].y, D[14].x, D[14].y);
  g.addColorStop(0, '#4a1519'); g.addColorStop(1, '#1b0608');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  const cm = { x: (D[13].x + D[14].x) / 2, y: (D[13].y + D[14].y) / 2 };
  g = ctx.createRadialGradient(cm.x, cm.y, 1, cm.x, cm.y, mw * 0.38);
  g.addColorStop(0, 'rgba(4,0,2,.88)'); g.addColorStop(1, 'rgba(4,0,2,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

  // ---- tongue (behind the teeth: anatomically correct occlusion) ----
  if (tongue && amt.tongue > 0.01) drawTongue(ctx, tongue, { amt: amt.tongue, mw, gap, W, H, yaw });

  // ---- lower arch, then upper arch (the upper overlaps the lower) ----
  if (lower && toothH > 0.4 && amt.lower > 0.01) drawTeeth(ctx, lower, { tex, h: toothH * 0.72, alpha: amt.lower, mw, gap, warm, lum, bright, yaw, lo: true });
  if (upper && toothH > 0.4) {
    if (amt.gum > 0.02 && G.gumUpper) drawGum(ctx, G.gumUpper, { amt: amt.gum, dn, h: toothH });
    drawTeeth(ctx, upper, { tex, h: toothH, alpha: 1, mw, gap, warm, lum, bright, yaw, lo: false });
  }
  // ---- corner occlusion + upper-lip shadow cast onto everything ----
  g = ctx.createLinearGradient(D[13].x, D[13].y, D[13].x + dn.x * toothH * 1.5, D[13].y + dn.y * toothH * 1.5);
  g.addColorStop(0, 'rgba(30,8,10,.62)'); g.addColorStop(1, 'rgba(30,8,10,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  [a, b].forEach(P => {
    const r = ctx.createRadialGradient(P.x, P.y, 1, P.x, P.y, mw * (0.15 + 0.06 * amt.ao));
    r.addColorStop(0, `rgba(8,0,2,${0.72 * (0.4 + 0.6 * amt.ao)})`); r.addColorStop(1, 'rgba(8,0,2,0)');
    ctx.fillStyle = r; ctx.fillRect(P.x - mw * 0.2, P.y - mw * 0.2, mw * 0.4, mw * 0.4);
  });
  // ---- saliva strand along the lower incisal edge ----
  if (amt.wet > 0.05) {
    ctx.filter = 'blur(1px)';
    ctx.strokeStyle = `rgba(255,235,230,${0.3 * amt.wet})`; ctx.lineWidth = Math.max(1, mw * 0.012);
    ctx.beginPath();
    LO.forEach((i, k) => (k ? ctx.lineTo(D[i].x - dn.x * 2.5, D[i].y - dn.y * 2.5) : ctx.moveTo(D[i].x - dn.x * 2.5, D[i].y - dn.y * 2.5)));
    ctx.stroke(); ctx.filter = 'none';
  }
  ctx.restore();

  // soft contact shadow around the whole mouth (drawn outside the clip)
  ctx.save();
  ctx.filter = 'blur(4px)';
  ctx.strokeStyle = `rgba(15,3,5,${0.55 * (0.35 + 0.65 * amt.open)})`;
  ctx.lineWidth = Math.max(3, mw * 0.06);
  polyPath(ctx, G.inner, 1); ctx.stroke();
  ctx.filter = 'none';
  ctx.strokeStyle = 'rgba(115,42,50,.5)'; ctx.lineWidth = Math.max(1.4, mw * 0.017);
  polyPath(ctx, G.inner, 1.02); ctx.stroke();
  ctx.restore();
  if (amt.kiss > 0.02) drawKiss(ctx, D, mw, amt.kiss);
}

export function drawTeeth(ctx, teeth, o) {
  const { tex, h, alpha = 1, mw, gap, warm = 0, lum = 0.7, bright = 1, yaw = 0, lo = false } = o;
  if (!teeth.length || h <= 0.4) return;
  ctx.save();
  if (alpha < 1) ctx.globalAlpha = clamp(alpha, 0, 1);
  const base = clamp(0.62 + lum * 0.5, 0.7, 1.05) * bright;
  for (const t of teeth) {
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(t.A0.x, t.A0.y);
      ctx.lineTo(t.p1.x, t.p1.y);
      ctx.quadraticCurveTo(t.A1.x, t.A1.y, t.p2.x, t.p2.y);
      ctx.quadraticCurveTo(t.Cc.x, t.Cc.y, t.p3.x, t.p3.y);
      ctx.quadraticCurveTo(t.B1.x, t.B1.y, t.p4.x, t.p4.y);
      ctx.lineTo(t.B0.x, t.B0.y);
      ctx.closePath();
    };
    // enamel texture, mapped onto the crown
    if (tex && tex.enamel) {
      ctx.save(); path(); ctx.clip();
      const b = t.texBasis;
      ctx.transform(b.ex, b.ey, b.fx, b.fy, b.ox, b.oy);
      ctx.globalAlpha = clamp(base * t.lambert, 0.25, 1.25);
      ctx.drawImage(tex.enamel, (t.j * 29 + (t.sd > 0 ? 13 : 0)) % 80, 0, 48, 120, 0, 0, 48, 120);
      ctx.restore();
    } else { ctx.save(); path(); ctx.fillStyle = `rgba(240,236,224,${clamp(base * t.lambert, 0.2, 1)})`; ctx.fill(); ctx.restore(); }
    // interproximal shading
    let g = ctx.createLinearGradient(t.A0.x, t.A0.y, t.B0.x, t.B0.y);
    g.addColorStop(0, 'rgba(110,85,60,.4)'); g.addColorStop(0.3, 'rgba(110,85,60,0)');
    g.addColorStop(0.7, 'rgba(110,85,60,0)'); g.addColorStop(1, 'rgba(110,85,60,.4)');
    path(); ctx.fillStyle = g; ctx.fill();
    // ambient occlusion toward the corners and behind the lower arch
    path(); ctx.fillStyle = `rgba(45,18,16,${t.ao})`; ctx.fill();
    // specular sweep: a soft band that travels with head yaw
    const sx = clamp(0.5 + yaw * 0.55, 0.05, 0.95);
    g = ctx.createLinearGradient(t.A0.x, t.A0.y, t.B0.x, t.B0.y);
    g.addColorStop(clamp(sx - 0.16, 0, 1), 'rgba(255,255,255,0)');
    g.addColorStop(sx, `rgba(255,255,255,${0.16 * t.depth})`);
    g.addColorStop(clamp(sx + 0.16, 0, 1), 'rgba(255,255,255,0)');
    path(); ctx.fillStyle = g; ctx.fill();
    if (warm > 0.01) { path(); ctx.fillStyle = `rgba(205,165,80,${warm * 0.2})`; ctx.fill(); }
  }
  // translucent incisal edges
  ctx.globalCompositeOperation = 'destination-out';
  ctx.strokeStyle = `rgba(0,0,0,${lo ? 0.34 : 0.45})`;
  ctx.lineWidth = Math.max(1.2, h * 0.15); ctx.lineCap = 'round';
  ctx.beginPath();
  teeth.forEach(t => {
    ctx.moveTo(t.p1.x, t.p1.y);
    ctx.quadraticCurveTo(t.A1.x, t.A1.y, t.p2.x, t.p2.y);
    ctx.quadraticCurveTo(t.Cc.x, t.Cc.y, t.p3.x, t.p3.y);
    ctx.quadraticCurveTo(t.B1.x, t.B1.y, t.p4.x, t.p4.y);
  });
  ctx.stroke();
  ctx.globalCompositeOperation = 'source-over';
  // interdental gaps
  ctx.strokeStyle = 'rgba(55,30,26,.34)'; ctx.lineWidth = 1.3;
  ctx.beginPath();
  teeth.forEach(t => { ctx.moveTo(t.A0.x, t.A0.y); ctx.lineTo(t.A1.x, t.A1.y); });
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.restore();
}

export function drawGum(ctx, band, o) {
  if (!band) return;
  ctx.save();
  ctx.globalAlpha = clamp(o.amt, 0, 1) * 0.62;
  polyPath(ctx, band.path, 1);
  const g = ctx.createLinearGradient(band.line[0].x, band.line[0].y, band.line[0].x + o.dn.x * o.h, band.line[0].y + o.dn.y * o.h);
  g.addColorStop(0, '#8f4450'); g.addColorStop(1, '#b0626d');
  ctx.fillStyle = g; ctx.fill();
  ctx.restore();
}

export function drawTongue(ctx, T, o) {
  const { amt, mw, gap, W, H, yaw = 0 } = o;
  const sh = () => smoothPath(ctx, T.outline, true, 0.32);
  const tip = T.tip, base = T.base;
  ctx.save();
  ctx.globalAlpha = clamp(amt, 0, 1);
  ctx.shadowColor = 'rgba(0,0,0,.42)'; ctx.shadowBlur = 9; ctx.shadowOffsetY = 3;
  const g = ctx.createLinearGradient(base.x, base.y, tip.x, tip.y);
  g.addColorStop(0, '#8f3a46'); g.addColorStop(0.5, '#c9636f'); g.addColorStop(1, '#d77e88');
  sh(); ctx.fillStyle = g; ctx.fill();
  ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
  sh(); ctx.clip();
  // papillae speckle (deterministic: identical every frame, so it never crawls)
  const rt = T.right, dn = T.dir;
  for (let i = 0; i < 90; i++) {
    const r = ((i * 0.6180339887) % 1 - 0.5) * T.hw * 1.8, d = ((i * 0.3819660113) % 1) * T.L;
    const x = base.x + rt.x * r + dn.x * d, y = base.y + rt.y * r + dn.y * d;
    ctx.beginPath(); ctx.arc(x, y, i % 3 ? 0.9 : 1.5, 0, 7);
    ctx.fillStyle = 'rgba(255,215,218,.16)'; ctx.fill();
  }
  // dorsum highlight + median groove + rim shading
  let sg = ctx.createRadialGradient(T.hump.x, T.hump.y, 1, T.hump.x, T.hump.y, T.hw * 1.25);
  sg.addColorStop(0, 'rgba(255,215,220,.32)'); sg.addColorStop(1, 'rgba(255,215,220,0)');
  ctx.fillStyle = sg; ctx.fillRect(0, 0, W, H);
  ctx.filter = 'blur(1px)';
  ctx.strokeStyle = `rgba(110,25,40,${0.35 + 0.2 * T.tipUp})`;
  ctx.lineWidth = Math.max(1.4, T.hw * 0.12);
  ctx.beginPath(); ctx.moveTo(T.groove[0].x, T.groove[0].y); ctx.lineTo(T.groove[1].x, T.groove[1].y); ctx.stroke();
  ctx.filter = 'blur(3px)'; ctx.strokeStyle = 'rgba(100,25,38,.6)'; ctx.lineWidth = T.hw * 0.32;
  sh(); ctx.stroke();
  ctx.filter = 'none';
  // wet highlight that tracks the head turn
  const hx = clamp(0.5 + yaw * 0.5, 0.1, 0.9);
  const wg = ctx.createLinearGradient(base.x - rt.x * T.hw, base.y - rt.y * T.hw, base.x + rt.x * T.hw, base.y + rt.y * T.hw);
  wg.addColorStop(clamp(hx - 0.12, 0, 1), 'rgba(255,255,255,0)');
  wg.addColorStop(hx, 'rgba(255,240,240,.2)');
  wg.addColorStop(clamp(hx + 0.12, 0, 1), 'rgba(255,255,255,0)');
  ctx.fillStyle = wg; ctx.fillRect(0, 0, W, H);
  ctx.restore();
}

/** Puckered-lip overlay. Kept as an overlay only: the mesh underneath stays
 *  intact so the mouth edge never flashes when a kiss ramps in or out. */
export function drawKiss(ctx, D, mw, pk) {
  const a = D[61], b = D[291];
  const c = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const ang = Math.atan2(b.y - a.y, b.x - a.x);
  const w = mw * (0.18 + 0.12 * pk), h = w * (0.42 + 0.15 * pk);
  ctx.save();
  ctx.globalAlpha = clamp(ramp(0.02, 0.35, pk), 0, 1);
  ctx.translate(c.x, c.y); ctx.rotate(ang);
  const g = ctx.createRadialGradient(0, -h * 0.15, 1, 0, 0, w);
  g.addColorStop(0, 'rgba(255,213,211,.3)'); g.addColorStop(0.42, 'rgba(208,75,94,.16)'); g.addColorStop(1, 'rgba(80,12,25,0)');
  ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(0, 0, w, h, 0, 0, 7); ctx.fill();
  ctx.strokeStyle = 'rgba(70,12,24,.35)'; ctx.lineCap = 'round'; ctx.lineWidth = Math.max(1, mw * 0.018);
  ctx.beginPath(); ctx.moveTo(-w * 0.55, 0); ctx.quadraticCurveTo(0, h * 0.12, w * 0.55, 0); ctx.stroke();
  ctx.restore();
}

/**
 * Lip shading: tint, vertical creases, moist highlights, contact seam and corner
 * occlusion. Always drawn (a closed mouth still needs a sealed seam), with the
 * seam strength gated by the measured opening.
 */
export function drawLips(ctx, D, o) {
  const { mw, gap, seamAmt, yaw = 0 } = o;
  const a = D[61], b = D[291];
  const ang = Math.atan2(b.y - a.y, b.x - a.x);
  const dn = { x: -Math.sin(ang), y: Math.cos(ang) };
  const seam = clamp(seamAmt, 0, 1);
  const O = GROUPS.lips.map(i => D[i]);
  const xs = O.map(p => p.x), ys = O.map(p => p.y);
  const x0 = Math.min(...xs) - 4, y0 = Math.min(...ys) - 4, x1 = Math.max(...xs) + 4, y1 = Math.max(...ys) + 4;
  const lc = { x: (D[17].x + D[14].x) / 2, y: (D[17].y + D[14].y) / 2 };
  const uc = { x: (D[0].x + D[13].x) / 2, y: (D[0].y + D[13].y) / 2 };
  ctx.save();
  ctx.beginPath();
  O.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  LIP.map(i => D[i]).forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.clip('evenodd');
  ctx.fillStyle = 'rgba(185,65,80,.14)'; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.lineWidth = 0.8; ctx.filter = 'blur(.4px)';
  for (let i = 0; i < 34; i++) {
    const t = (i + 0.5) / 34, bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t, h = mw * 0.2;
    ctx.strokeStyle = `rgba(85,28,36,${0.07 + 0.05 * (i % 3)})`;
    ctx.beginPath(); ctx.moveTo(bx - dn.x * h, by - dn.y * h); ctx.lineTo(bx + dn.x * h, by + dn.y * h); ctx.stroke();
  }
  ctx.filter = 'blur(1.5px)';
  const gl = (c, rw, rh, al) => {
    ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(ang); ctx.scale(1, rh / rw);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rw);
    g.addColorStop(0, `rgba(255,240,238,${al})`); g.addColorStop(1, 'rgba(255,240,238,0)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, rw, 0, 7); ctx.fill(); ctx.restore();
  };
  gl(lc, mw * 0.26, mw * 0.05, 0.4 * (1 - seam * 0.3));
  gl(uc, mw * 0.16, mw * 0.035, 0.16);
  ctx.filter = 'none';
  ctx.restore();
  if (seam > 0.02) {
    ctx.save();
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const sp = () => { ctx.beginPath(); UP.forEach((i, k) => (k ? ctx.lineTo(D[i].x, D[i].y) : ctx.moveTo(D[i].x, D[i].y))); };
    ctx.filter = 'blur(2px)'; ctx.strokeStyle = `rgba(55,15,22,${0.3 * seam})`; ctx.lineWidth = mw * 0.035; sp(); ctx.stroke();
    ctx.filter = 'blur(.7px)'; ctx.strokeStyle = `rgba(45,10,16,${0.6 * seam})`; ctx.lineWidth = Math.max(1, mw * 0.011); sp(); ctx.stroke();
    ctx.filter = 'none';
    [a, b].forEach(P => {
      const r = ctx.createRadialGradient(P.x, P.y, 0, P.x, P.y, mw * 0.07);
      r.addColorStop(0, `rgba(40,8,14,${0.4 * seam})`); r.addColorStop(1, 'rgba(40,8,14,0)');
      ctx.fillStyle = r; ctx.fillRect(P.x - mw * 0.1, P.y - mw * 0.1, mw * 0.2, mw * 0.2);
    });
    ctx.restore();
  }
  // the far corner of the mouth falls into shadow as the head turns
  const side = yaw > 0 ? 291 : 61;
  const al = clamp(Math.abs(yaw) * 0.5, 0, 0.3);
  if (al > 0.01) {
    const P = D[side];
    ctx.save();
    const r = ctx.createRadialGradient(P.x, P.y, 1, P.x, P.y, mw * 0.2);
    r.addColorStop(0, `rgba(20,6,10,${al})`); r.addColorStop(1, 'rgba(20,6,10,0)');
    ctx.fillStyle = r; ctx.fillRect(P.x - mw * 0.25, P.y - mw * 0.25, mw * 0.5, mw * 0.5);
    ctx.restore();
  }
}
