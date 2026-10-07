/**
 * Eyes: iris re-projection, eyelids and the wet-eye highlights.
 *
 * The eyeball is the one part of the face that must NOT be smeared by the mesh
 * warp - a stretched iris reads as fake instantly. So at load time the photo's
 * own iris texture is cut out into a sprite (with a synthetic limbus and sclera
 * colour sampled from the surrounding skin) and that sprite is re-painted every
 * frame, clipped to the current eye opening and slid by the measured gaze. The
 * lids are the photo's own skin stretched down over the eye with a lash line,
 * which keeps blink creases and skin tone identity-correct.
 *
 * Everything is gated by `vis` (eye visibility under head yaw) and by smooth
 * amounts, so nothing pops in or out.
 */

import { clamp, lerp, ramp } from './math3d.js';
import { EYE, EYM, IRIS } from './landmarks.js';
import { polyPath } from './mouth.js';

const EYE_RIM = [1, 2, 3, 4]; // iris ring offsets around 468 / 473

function meanColor(getImageData, x, y, size) {
  try {
    const d = getImageData(Math.round(x), Math.round(y), size, size).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
    const m = d.length / 4 || 1;
    return [r / m, g / m, b / m];
  } catch (e) { return [120, 92, 74]; }
}

/**
 * Cut the irises out of the source image into reusable sprites.
 * @param src   the source canvas (pixels of the still photo)
 * @param grab  (x,y,w,h)=>ImageData, clamped sampler bound to `src`
 * @param V     rest landmark array ([[x,y,z], ...])
 * @param mkCanvas (w,h)=>canvas
 */
export function prepIris(src, grab, V, mkCanvas) {
  const out = [null, null];
  const at = (x, y, s = 5) => meanColor(grab, x - (s >> 1), y - (s >> 1), s);
  for (let n = 0; n < 2; n++) {
    const c = V[IRIS[n]];
    const radii = EYE_RIM.map(k => Math.hypot(V[IRIS[n] + k][0] - c[0], V[IRIS[n] + k][1] - c[1]));
    const rs = Math.max(3, radii.reduce((a, b) => a + b, 0) / 4);
    const R = Math.ceil(rs * 1.35);
    const [iu, il, ic, io] = EYM[n];
    // sclera colour: whichever side of the eye is brighter is the white
    const s1 = at(c[0] + (V[ic][0] - c[0]) * 0.62, c[1] + (V[ic][1] - c[1]) * 0.62);
    const s2 = at(c[0] + (V[io][0] - c[0]) * 0.62, c[1] + (V[io][1] - c[1]) * 0.62);
    const sum = a => a[0] + a[1] + a[2];
    const scl = (sum(s1) > sum(s2) ? s1 : s2).map(v => Math.round(clamp(v, 60, 255))).join(',');
    // iris colour: sample left/right/below the pupil, never the pupil itself
    const q = [at(c[0] - rs * 0.62, c[1]), at(c[0] + rs * 0.62, c[1]), at(c[0], c[1] + rs * 0.62)];
    const ir = [0, 1, 2].map(k => (q[0][k] + q[1][k] + q[2][k]) / 3);
    const cv = mkCanvas(2 * R, 2 * R);
    const x = cv.getContext('2d');
    x.save();
    // synthetic limbus ring first: photos with a hidden iris still get an eye
    const g = x.createRadialGradient(R, R, 0, R, R, rs);
    g.addColorStop(0, 'rgb(14,10,8)'); g.addColorStop(0.3, 'rgb(14,10,8)');
    g.addColorStop(0.42, `rgb(${ir.map(v => Math.round(clamp(v, 0, 255))).join(',')})`);
    g.addColorStop(1, `rgb(${ir.map(v => Math.round(clamp(v * 0.6, 0, 255))).join(',')})`);
    x.fillStyle = g; x.beginPath(); x.arc(R, R, rs, 0, 7); x.fill();
    // then the photo's own iris texture, limited to the visible eye opening
    x.beginPath();
    EYE[n].forEach((i, k) => { const p = V[i]; k ? x.lineTo(p[0] - c[0] + R, p[1] - c[1] + R) : x.moveTo(p[0] - c[0] + R, p[1] - c[1] + R); });
    x.closePath(); x.clip();
    x.drawImage(src, c[0] - R, c[1] - R, 2 * R, 2 * R, 0, 0, 2 * R, 2 * R);
    x.restore();
    x.globalCompositeOperation = 'destination-in';
    const m = x.createRadialGradient(R, R, rs * 0.92, R, R, rs * 1.22);
    m.addColorStop(0, 'rgba(0,0,0,1)'); m.addColorStop(1, 'rgba(0,0,0,0)');
    x.fillStyle = m; x.fillRect(0, 0, 2 * R, 2 * R);
    out[n] = { cv, R, r: rs, sclera: scl };
  }
  return out;
}

/**
 * Raw gaze of one eye from a landmark set: the iris offset relative to the eye,
 * decomposed along/across the eye axis and normalised by eye width.
 */
export function gazeRaw(n, P) {
  const ir = P[IRIS[n]], [iu, il, ic, io] = EYM[n];
  const ex = P[io][0] - P[ic][0], ey = P[io][1] - P[ic][1];
  const ew = Math.hypot(ex, ey) || 1, ax = ex / ew, ay = ey / ew;
  const dx = ir[0] - (P[ic][0] + P[io][0]) / 2;
  const dy = ir[1] - (P[iu][1] + P[il][1]) / 2;
  return [(dx * ax + dy * ay) / ew, (-dx * ay + dy * ax) / ew];
}

/** Paint the iris sprite into the current eye opening. */
export function drawIris(ctx, n, D, o) {
  const I = o.sprite && o.sprite[n];
  if (!I) return;
  const vis = clamp(o.vis === undefined ? 1 : o.vis, 0, 1);
  if (vis < 0.02) return;
  const [iu, il, ic, io] = EYM[n], ci = IRIS[n];
  const c = { x: D[ci].x, y: D[ci].y };
  const ex = D[io].x - D[ic].x, ey = D[io].y - D[ic].y;
  const ew = Math.hypot(ex, ey) || 1, ax = ex / ew, ay = ey / ew;
  const gg = o.gazeGain ?? 1.9;
  const gz = o.gaze[n] || [0, 0];
  const ox = (gz[0] * ax - gz[1] * ay) * ew * gg;
  const oy = (gz[0] * ay + gz[1] * ax) * ew * gg;
  const r = EYE_RIM.reduce((t, k) => t + Math.hypot(D[ci + k].x - c.x, D[ci + k].y - c.y), 0) / 4 || I.r;
  const k = r / I.r, R = I.R * k;
  const X = c.x + ox, Y = c.y + oy;
  const reveal = clamp(Math.hypot(ox, oy) / (ew * 0.012), 0, 1);
  const ys = EYE[n].map(i => D[i].y);
  const top = Math.min(...ys), bot = Math.max(...ys);
  ctx.save();
  ctx.globalAlpha = vis;
  polyPath(ctx, EYE[n].map(i => D[i]), 0.97);
  ctx.clip();
  if (reveal > 0.01) { // sclera shows on the side the iris travelled away from
    const g = ctx.createRadialGradient(c.x, c.y, r * 0.2, c.x, c.y, r * 1.4);
    g.addColorStop(0, `rgba(${I.sclera},${reveal})`);
    g.addColorStop(0.8, `rgba(${I.sclera},${reveal})`);
    g.addColorStop(1, `rgba(${I.sclera},0)`);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(c.x, c.y, r * 1.45, 0, 7); ctx.fill();
  }
  ctx.drawImage(I.cv, X - R, Y - R, 2 * R, 2 * R);
  // limbus shadow: the eyeball is a sphere, so the iris edge falls off
  let lg = ctx.createRadialGradient(X, Y, r * 0.78, X, Y, r * 1.05);
  lg.addColorStop(0, 'rgba(15,10,8,0)'); lg.addColorStop(1, 'rgba(15,10,8,.36)');
  ctx.fillStyle = lg; ctx.beginPath(); ctx.arc(X, Y, r * 1.08, 0, 7); ctx.fill();
  // upper lid cast shadow
  const sg = ctx.createLinearGradient(0, top, 0, top + (bot - top) * 0.55);
  sg.addColorStop(0, `rgba(30,15,10,${0.36 * (0.45 + 0.55 * vis)})`); sg.addColorStop(1, 'rgba(30,15,10,0)');
  ctx.fillStyle = sg; ctx.fillRect(c.x - ew, top, ew * 2, bot - top);
  // corneal specular: fixed to the LIGHT, not to the eye, which is what makes
  // an eye read as wet rather than painted
  const hx = X - ax * ew * 0.1 + (o.lightX || 0) * ew * 0.16;
  const hy = Y - r * 0.34 + (o.lightY || 0) * r * 0.2;
  ctx.filter = 'blur(1px)';
  ctx.fillStyle = `rgba(255,252,248,${0.5 * (0.35 + 0.65 * vis)})`;
  ctx.beginPath(); ctx.ellipse(hx, hy, Math.max(0.8, r * 0.16), Math.max(0.6, r * 0.12), -0.4, 0, 7); ctx.fill();
  ctx.filter = 'none';
  ctx.restore();
}

/**
 * Eyelid: the photo's own skin from just above the eye, stretched down over the
 * opening, feathered, with a blurred crease and a lash line.
 */
export function drawEyelid(ctx, n, D, o) {
  const c = clamp(o.amount, 0, 1) * clamp(o.vis === undefined ? 1 : o.vis, 0, 1);
  if (c < 0.02) return;
  const [iu, il, ic, io] = EYM[n];
  const p = D[ic], q = D[io], u = D[iu], l = D[il];
  const ew = Math.hypot(p.x - q.x, p.y - q.y) || 1;
  const ys = EYE[n].map(i => D[i].y);
  const top = Math.min(...ys), bot = Math.max(...ys);
  const eh = Math.max(bot - top, ew * 0.14);
  const m = { x: u.x + (l.x - u.x) * c, y: u.y + (l.y - u.y) * c };       // lid margin
  const k = { x: m.x * 2 - (p.x + q.x) / 2, y: m.y * 2 - (p.y + q.y) / 2 }; // crease control
  const ty = top - eh * 0.5, m0 = ew * 0.06;
  const x0 = Math.floor(Math.min(p.x, q.x) - m0 - 6), y0 = Math.floor(ty - 6);
  const w = Math.ceil(ew + 2 * m0 + 12), h = Math.ceil(bot - ty + 12);
  const sy = top - eh * 0.95;
  if (w < 4 || h < 4) return;
  const fc = o.offscreen, fx = fc.getContext('2d');
  fc.width = w; fc.height = h;                    // resizing resets the transform
  fx.setTransform(1, 0, 0, 1, 0, 0);
  fx.translate(-x0, -y0);
  fx.fillStyle = '#000';
  fx.filter = 'blur(1.5px)';
  fx.beginPath();
  fx.moveTo(p.x, p.y);
  fx.quadraticCurveTo(k.x, k.y, q.x, q.y);
  fx.lineTo(q.x + (q.x > p.x ? m0 : -m0), ty);
  fx.lineTo(p.x + (q.x > p.x ? -m0 : m0), ty);
  fx.closePath(); fx.fill();
  fx.filter = 'none';
  fx.globalCompositeOperation = 'source-in';
  const srcY = Math.max(0, sy), srcH = Math.max(2, eh * 0.8);
  fx.drawImage(o.stage, x0, srcY, w, srcH, x0, ty, w, bot - ty + 6);
  ctx.save();
  ctx.globalAlpha = clamp(c * 3, 0, 1);
  ctx.drawImage(fc, x0, y0);
  const edge = () => { ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.quadraticCurveTo(k.x, k.y, q.x, q.y); };
  ctx.lineCap = 'round';
  ctx.filter = 'blur(2px)';
  ctx.strokeStyle = `rgba(60,30,25,${0.3 * clamp(o.vis ?? 1, 0, 1)})`;
  ctx.lineWidth = ew * 0.09; edge(); ctx.stroke();
  ctx.filter = 'none';
  ctx.strokeStyle = '#2b1a16';
  ctx.lineWidth = Math.max(1.3, ew * 0.035); edge(); ctx.stroke();
  ctx.restore();
}

/** Blink/droop/squint/wink combination, all continuous (no strobing lids). */
export function lidAmount(state, n, dt) {
  const s = state;
  const target = clamp(Math.max(s.blink[n], s.droop * 0.5 + s.squint * 0.22, n ? s.browR : s.browL) + Math.max(0, s.gaze[n][1]) * 1.5, 0, 1);
  const k = 1 - Math.exp(-dt / (target > s.lid[n] ? 0.045 : 0.075));
  s.lid[n] += (target - s.lid[n]) * k;
  return s.lid[n];
}

export { lerp, ramp };
