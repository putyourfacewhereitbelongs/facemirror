/**
 * Frame assembly: everything that happens after the mesh has been posed.
 *
 * This module is deliberately free of DOM glue - it takes a 2D context, the
 * rig, the posed state, the pose controls and a plain options bag, so the whole
 * compositing order can be unit-tested with a mock context and driven from any
 * front end.
 *
 * Compositing order (fixed, and the order matters):
 *   1. opaque base      - inpainted background plate, or the source photo
 *   2. warped head      - affine triangles, far -> near, flip-free
 *   3. seam composite   - plate re-blurred to the silhouette + contact AO
 *   4. relighting       - differential Lambert term, soft-light, mid-gray = no-op
 *   5. eyes             - iris sprites clipped to the opening, then lids
 *   6. mouth            - cavity, tongue, teeth, gums, lips, kiss
 *   7. protruding tongue- over the lips, clipped below the upper lip line
 *   8. ROI enhancement  - local high-pass on the mouth and eyes
 *   9. overlays         - landmark dots, iBUG-68 wiring, capture flash
 */

import { clamp, lerp } from './math3d.js';
import { OVAL, EYE, LIP, IBUG68 } from './landmarks.js';
import { fitArch, archTeeth, gumBand, tongueGeom, drawMouth, drawLips, drawTongue, innerMouth, outerMouth, polyPath } from './mouth.js';
import { drawIris, drawEyelid } from './eyes.js';
import { drawMesh, drawShading, drawSeam, enhanceROIs, drawLandmarks } from './render.js';
import { shadeField, eyeVisibility } from './pose.js';

/** Scratch buffers reused across frames (never reallocated per frame). */
export function createScratch(mkCanvas) {
  return { shade: mkCanvas(), lid: mkCanvas(), rois: [mkCanvas(), mkCanvas(), mkCanvas()].map(off => ({ off, x: 0, y: 0, w: 8, h: 8, radius: 2 })) };
}

/** Zero-copy view of the flat projected array as {x,y} objects. */
export function pointView(P, cache) {
  const n = P.length >> 1;
  const D = cache && cache.length === n ? cache : new Array(n);
  for (let i = 0; i < n; i++) {
    const p = D[i] || (D[i] = { x: 0, y: 0 });
    p.x = P[i * 2]; p.y = P[i * 2 + 1];
  }
  return D;
}

export function boxAround(D, ids, pad, radius, W, H) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of ids) {
    const p = D[i];
    if (!p) continue;
    if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y;
    if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y;
  }
  if (!Number.isFinite(x0)) return { x: 0, y: 0, w: 8, h: 8, radius: radius || 2 };
  x0 = clamp(Math.floor(x0 - pad), 0, Math.max(0, W - 8));
  y0 = clamp(Math.floor(y0 - pad), 0, Math.max(0, H - 8));
  x1 = clamp(Math.ceil(x1 + pad), x0 + 8, W);
  y1 = clamp(Math.ceil(y1 + pad), y0 + 8, H);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, radius: Math.max(1.2, radius || 2) };
}

/**
 * Draw one complete frame. `o` is the settings bag:
 *
 *  src            source photo canvas (texture for the warp)
 *  plate          inpainted background canvas, or null
 *  scratch        { shade, lid, rois } from createScratch()
 *  pose           { yaw, pitch, roll }
 *  amt            smooth gate values: {open,gap,teeth,lower,gum,tongue,tongueOut,kiss,ao,wet,seam,throat}
 *  mw, gap        measured mouth width / opening in projected pixels
 *  gaze, lid      per-eye gaze offsets and lid amounts
 *  mouth          {tongueAmt, tipUp, backUp, wide, protrude, toothAmt, toothScale, gumVis, archTau, bright, warm}
 *  iris, tex, lum iris sprites, procedural textures, photo luminance
 *  flags          {showOrig, cull, plateOn, seam, enhance, showPts, ibug, quality}
 *  shown, drag, hover   landmark overlay state
 *  dt, flash      frame delta and capture-flash alpha
 *
 * Returns a diagnostics object (triangle count, arch, tooth thickness, ...) that
 * the HUD and the tests read.
 */
export function renderFrame(ctx, rig, st, ctl, o) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const P = st.P;
  const D = o.D || pointView(P, o.Dcache);
  const flags = o.flags || {};
  const hi = (flags.quality || 1) > 1.5;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingQuality = hi ? 'high' : 'medium';
  const out = { drawn: 0, teeth: 0, tongue: 0, lid: [0, 0], vis: [1, 1], arch: null, thickness: 0, stretch: st.lastStretch };

  if (flags.showOrig) {
    ctx.drawImage(o.src, 0, 0);
    if (flags.showPts) drawLandmarks(ctx, P, o.shown, { drag: o.drag, hover: o.hover, radius: 2.8, halo: 4.2 });
    return out;
  }

  // 1 - opaque base
  const usePlate = o.plate && flags.plateOn !== false;
  if (usePlate) ctx.drawImage(o.plate, 0, 0);
  else ctx.drawImage(o.src, 0, 0);

  // 2 - warped head
  out.drawn = drawMesh(ctx, rig, st, o.src, { pad: hi ? 0.95 : 0.7, cull: !!flags.cull, bridge: !usePlate });
  out.bridge = !usePlate;
  const silhouette = OVAL.map(i => ({ x: P[i * 2], y: P[i * 2 + 1] }));

  // 3 - seamless composite
  if (usePlate && flags.seam !== false) {
    drawSeam(ctx, o.plate, silhouette, { blur: 2.5, blend: o.blend ?? 0.55, ao: o.seamAO ?? 0.5, width: rig.frame.FW * 0.02 + 4 });
  }

  // 4 - differential relighting
  const shadeStrength = o.shade ?? 0.75;
  if (shadeStrength > 0.005 && o.scratch) {
    shadeField(rig, st, ctl.R, shadeStrength);
    drawShading(ctx, rig, st, { off: o.scratch.shade, strength: shadeStrength, scale: hi ? 0.42 : 0.3, blur: 6, silhouette });
  }

  // 5..7 - face FX
  try { Object.assign(out, drawFX(ctx, rig, st, D, o, silhouette)); }
  catch (e) { console.error('FX stage failed', e); out.fxError = e.message || String(e); }

  // 8 - ROI enhancement
  const enh = o.enhance ?? 0;
  if (enh > 0.02 && o.scratch && !flags.showOrig) {
    const f = rig.frame, rois = o.scratch.rois;
    const boxes = [
      boxAround(D, LIP, f.MW * 0.35, f.MW * 0.035, W, H),
      boxAround(D, EYE[0], f.EW[0] * 0.5, f.EW[0] * 0.05, W, H),
      boxAround(D, EYE[1], f.EW[1] * 0.5, f.EW[1] * 0.05, W, H),
    ];
    for (let i = 0; i < 3; i++) Object.assign(rois[i], boxes[i]);
    try { enhanceROIs(ctx, rois, enh); } catch (e) { console.warn('enhance skipped', e); }
  }

  // 9 - overlays
  if (flags.showPts) {
    drawLandmarks(ctx, P, o.shown, { drag: o.drag, hover: o.hover, radius: 2.8, halo: 4.2 });
    if (flags.ibug) drawIbug(ctx, P);
  }
  if (o.flash > 0) {
    ctx.fillStyle = `rgba(255,255,255,${clamp(o.flash, 0, 1)})`;
    ctx.fillRect(0, 0, W, H);
  }
  return out;
}

/** Eyes + mouth + protruding tongue. Returns diagnostics for the HUD/tests. */
export function drawFX(ctx, rig, st, D, o, silhouette) {
  const f = rig.frame;
  const amt = o.amt || {};
  const mw = o.mw || f.MW, gap = o.gap || 0;
  const m = o.mouth || {};
  const pose = o.pose || { yaw: 0, pitch: 0, roll: 0 };
  const vis = eyeVisibility(pose.yaw, pose.pitch);
  const gaze = o.gaze || [[0, 0], [0, 0]];
  const lid = o.lid || [0, 0];
  const out = { teeth: 0, tongue: 0, lid: [0, 0], vis, arch: null, thickness: 0, tongueOut: 0, mouthDrawn: false };

  // ---- eyes ----
  for (let n = 0; n < 2; n++) {
    drawIris(ctx, n, D, {
      sprite: o.iris, gaze, vis: vis[n], gazeGain: o.gazeGain ?? 1.9,
      lightX: clamp(pose.yaw * 0.9, -1, 1), lightY: clamp(pose.pitch * 0.9, -1, 1),
    });
  }
  const lidOff = o.scratch ? o.scratch.lid : null;
  for (let n = 0; n < 2; n++) {
    out.lid[n] = lid[n];
    if (lidOff) drawEyelid(ctx, n, D, { amount: lid[n], vis: vis[n], offscreen: lidOff, stage: ctx.canvas });
  }

  // ---- mouth ----
  const arch = fitArch(D, st.Pz, o.archPrev, o.dt || 1 / 60, { tau: lerp(0.02, 0.11, clamp(m.archTau ?? 0.55, 0, 1)) });
  out.arch = arch;
  const depthRef = st.Pz[13] || 0, depthSpan = Math.max(1, f.FW * 0.35);
  const toothAmt = clamp(m.toothAmt ?? 0.4, 0, 1) * (m.toothScale ?? 1);
  const thickness = Math.min(gap * 0.62, mw * 0.2) * toothAmt * (1 - clamp(amt.kiss ?? 0, 0, 1) * 0.85);
  out.thickness = thickness;
  const archOpt = { depthRef, depthSpan, yaw: pose.yaw, light: -0.35 };
  const upper = archTeeth(arch, 1, Object.assign({ thickness, gum: 0.1 }, archOpt));
  const lower = archTeeth(arch, -1, Object.assign({ thickness: thickness * 0.72, gum: 0.06 }, archOpt));
  const tongue = tongueGeom(D, arch, {
    amount: m.tongueAmt ?? 0, tipUp: m.tipUp ?? 0, backUp: m.backUp ?? 0, wide: m.wide ?? 0.5,
    protrude: m.protrude ?? 0, gap, mw,
  });
  out.teeth = upper.length + lower.length;
  out.tongue = tongue;
  const G = {
    D, arch, inner: innerMouth(D), outer: outerMouth(D), upper, lower, tongue,
    gumUpper: gumBand(arch, 1, upper, { thickness: thickness * 0.42, dn: arch.dn }),
    thickness,
    amt: {
      open: amt.open ?? 0, gap, teeth: (amt.teeth ?? 0) * toothAmt * 2.2,
      lower: amt.lower ?? 0, gum: (amt.gum ?? 0) * (m.gumVis ?? 0.6),
      tongue: (amt.tongue ?? 0) * (1 - (amt.tongueOut ?? 0)), kiss: amt.kiss ?? 0,
      ao: amt.ao ?? 0, wet: amt.wet ?? 0, seam: amt.seam ?? 0, throat: amt.throat ?? 0,
    },
    tex: o.tex, W: ctx.canvas.width, H: ctx.canvas.height, lum: o.lum ?? 0.7, yaw: pose.yaw,
    bright: m.bright ?? 1.15, warm: m.warm ?? 0,
  };
  out.mouthDrawn = (amt.open ?? 0) > 0.012 || (amt.kiss ?? 0) > 0.02;
  if (out.mouthDrawn) drawMouth(ctx, G);
  drawLips(ctx, D, { mw, gap, seamAmt: amt.seam ?? 0, yaw: pose.yaw });

  // a protruding tongue is drawn over the lips, clipped below the upper lip line
  const outAmt = clamp(amt.tongueOut ?? 0, 0, 1) * clamp(m.tongueAmt ?? 0, 0, 1);
  out.tongueOut = outAmt;
  if (outAmt > 0.02) {
    ctx.save();
    ctx.beginPath();
    polyPath(ctx, G.outer, 1.5);
    ctx.rect(0, D[0].y + (D[14].y - D[0].y) * 0.55, ctx.canvas.width, ctx.canvas.height);
    ctx.clip();
    drawTongue(ctx, tongue, { amt: outAmt, mw, gap, W: ctx.canvas.width, H: ctx.canvas.height, yaw: pose.yaw });
    ctx.restore();
  }
  return out;
}

/** iBUG-68 wiring overlay, for comparing against the academic convention. */
export function drawIbug(ctx, P) {
  ctx.save();
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.85;
  const seg = (a, b) => {
    ctx.beginPath();
    for (let i = a; i <= b; i++) {
      const x = P[IBUG68[i] * 2], y = P[IBUG68[i] * 2 + 1];
      if (i === a) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  seg(0, 16); seg(17, 21); seg(22, 26); seg(27, 30); seg(31, 35); seg(36, 41); seg(42, 47); seg(48, 59); seg(60, 67);
  ctx.fillStyle = '#7dd3fc';
  for (const i of IBUG68) { ctx.beginPath(); ctx.arc(P[i * 2], P[i * 2 + 1], 1.6, 0, 7); ctx.fill(); }
  ctx.restore();
}
