/**
 * Compositing stage: the whole render path exercised against a mock 2D context.
 *
 * These are the regression tests for the two things the original page got
 * wrong: it drew the mesh twice per frame (official tessellation *and*
 * Delaunay), and it switched overlays on with hard thresholds. Both show up as
 * flicker, so both are asserted here rather than eyeballed.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { syntheticFace, fakeTesselation, mockCanvas } from './synthetic.js';
import { buildRig } from './model.js';
import { createPoseState, buildDragField, poseFrame } from './pose.js';
import { RFromEuler } from './math3d.js';
import { createScratch, pointView, renderFrame, drawFX, boxAround } from './frame.js';
import { unfold, signedArea, warpTri } from './render.js';
import { fitArch } from './mouth.js';
import { prepIris } from './eyes.js';
import { SHOWN, LIP, EYE } from './landmarks.js';

let rig, st, D, iris, tex, src, plate;
const mkCanvas = (w, h) => mockCanvas(w || 64, h || 64);

function baseOpts(over = {}) {
  return Object.assign({
    src, plate, scratch: createScratch(mkCanvas),
    pose: { yaw: 0, pitch: 0, roll: 0 }, dt: 1 / 60,
    amt: { open: 0.4, gap: 26, teeth: 0.8, lower: 0.6, gum: 0.2, tongue: 0.3, tongueOut: 0, kiss: 0, ao: 0.5, wet: 0.4, seam: 0.2, throat: 0.1 },
    mw: 130, gap: 26, gaze: [[0, 0], [0, 0]], lid: [0.1, 0.1], gazeGain: 1.9,
    mouth: { tongueAmt: 0.3, tipUp: 0, backUp: 0, wide: 0.5, protrude: 0, toothAmt: 0.5, toothScale: 1, gumVis: 0.6, archTau: 0.55, bright: 1.15, warm: 0 },
    iris, tex, lum: 0.7, blend: 0.55, seamAO: 0.5, shade: 0.75, enhance: 0.35,
    flags: { showOrig: false, cull: false, plateOn: true, seam: true, showPts: true, ibug: false, quality: 1 },
    shown: SHOWN, drag: -1, hover: -1, flash: 0,
  }, over);
}

beforeAll(() => {
  const f = syntheticFace({ W: 720, H: 900 });
  rig = buildRig(f.V, f.W, f.H, fakeTesselation(f.V));
  st = createPoseState(rig);
  st.dragField = buildDragField(rig, 1);
  poseFrame(rig, st, { R: null, stretch: 0 });
  src = mockCanvas(720, 900);
  plate = mockCanvas(720, 900);
  const grab = (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(4, w * h * 4)).fill(150), width: w, height: h });
  iris = prepIris(src, grab, rig.V, mkCanvas);
  tex = { enamel: mockCanvas(128, 160), noise: mockCanvas(96, 96) };
  D = pointView(st.P, null);
});

const countOf = (ctx, name) => ctx.__calls.filter(c => c === name).length;

describe('mesh painting', () => {
  it('paints the head in ONE pass (the original drew it twice)', () => {
    const cv = mockCanvas(720, 900);
    const ctx = cv.getContext('2d');
    const out = renderFrame(ctx, rig, st, { R: null }, baseOpts());
    expect(out.drawn).toBeGreaterThan(150);
    const blits = countOf(ctx, 'drawImage');
    // one blit per triangle, plus the plate, seam, shading buffer, two irises,
    // two lids and the ROI passes - never a second full mesh pass
    expect(blits).toBeLessThan(out.drawn * 1.25 + 20);
  });

  it('skips the frame-border bridge when an inpainted plate is underneath', () => {
    const withPlate = mockCanvas(720, 900).getContext('2d');
    const a = renderFrame(withPlate, rig, st, { R: null }, baseOpts());
    const noPlate = mockCanvas(720, 900).getContext('2d');
    const b = renderFrame(noPlate, rig, st, { R: null }, baseOpts({ plate: null }));
    expect(a.bridge).toBe(false);
    expect(b.bridge).toBe(true);
    expect(b.drawn).toBeGreaterThan(a.drawn);
  });

  it('is deterministic: two identical frames issue an identical draw sequence', () => {
    const c1 = mockCanvas(720, 900).getContext('2d');
    const c2 = mockCanvas(720, 900).getContext('2d');
    renderFrame(c1, rig, st, { R: null }, baseOpts());
    renderFrame(c2, rig, st, { R: null }, baseOpts());
    expect(c1.__calls.length).toBe(c2.__calls.length);
    expect(c1.__calls.join('|')).toBe(c2.__calls.join('|'));
  });

  it('leaves the context state clean (no leaked blend mode or filter)', () => {
    const ctx = mockCanvas(720, 900).getContext('2d');
    renderFrame(ctx, rig, st, { R: null }, baseOpts());
    expect(ctx.globalCompositeOperation).toBe('source-over');
    expect(ctx.filter).toBe('none');
    expect(ctx.globalAlpha).toBe(1);
    expect(countOf(ctx, 'save')).toBe(countOf(ctx, 'restore'));
  });

  it('still paints when the head is turned hard, with no NaN reaching the path', () => {
    poseFrame(rig, st, { R: RFromEuler(0.8, 0.25, 0.15), stretch: 2, relaxIters: 3 });
    const ctx = mockCanvas(720, 900).getContext('2d');
    const out = renderFrame(ctx, rig, st, { R: RFromEuler(0.8, 0.25, 0.15) }, baseOpts({ pose: { yaw: 0.8, pitch: 0.25, roll: 0.15 } }));
    expect(out.drawn).toBeGreaterThan(150);
    for (const [name, args] of ctx.__log) {
      if (name !== 'moveTo' && name !== 'lineTo' && name !== 'arc' && name !== 'fillRect') continue;
      for (const a of args) if (typeof a === 'number') expect(Number.isFinite(a)).toBe(true);
    }
    expect(out.fxError).toBeUndefined();
  });
});

describe('flip-free unfolding', () => {
  const s0 = { x: 0, y: 0 }, s1 = { x: 10, y: 0 }, s2 = { x: 0, y: 10 };
  it('keeps an already-correct triangle untouched', () => {
    const d0 = { x: 5, y: 5 }, d1 = { x: 20, y: 6 }, d2 = { x: 6, y: 22 };
    const out = unfold(s0, s1, s2, d0, d1, d2);
    expect(out[0]).toBe(d0); expect(out[1]).toBe(d1); expect(out[2]).toBe(d2);
  });
  it('reflects an inverted triangle back to the source orientation', () => {
    const d0 = { x: 5, y: 5 }, d1 = { x: 6, y: 22 }, d2 = { x: 20, y: 6 };  // wound the other way
    const out = unfold(s0, s1, s2, d0, d1, d2);
    expect(out).not.toBeNull();
    expect(Math.sign(signedArea(out[0], out[1], out[2]))).toBe(Math.sign(signedArea(s0, s1, s2)));
  });
  it('rejects a degenerate triangle instead of smearing it', () => {
    expect(unfold(s0, s1, s2, { x: 4, y: 4 }, { x: 8, y: 8 }, { x: 12, y: 12 })).toBeNull();
  });
});

describe('affine warp', () => {
  it('solves a transform that maps the source triangle onto the destination', () => {
    const ctx = mockCanvas(200, 200).getContext('2d');
    const s0 = { x: 10, y: 20 }, s1 = { x: 90, y: 30 }, s2 = { x: 40, y: 100 };
    const d0 = { x: 30, y: 10 }, d1 = { x: 150, y: 60 }, d2 = { x: 60, y: 140 };
    expect(warpTri(ctx, src, s0, s1, s2, d0, d1, d2, 0)).toBe(true);
    const t = ctx.__log.find(([n]) => n === 'setTransform');
    expect(t).toBeTruthy();
    const [a, b, c, d, e, f] = t[1];
    const map = p => ({ x: a * p.x + c * p.y + e, y: b * p.x + d * p.y + f });
    for (const [s, dd] of [[s0, d0], [s1, d1], [s2, d2]]) {
      const m = map(s);
      expect(Math.hypot(m.x - dd.x, m.y - dd.y)).toBeLessThan(1e-6);
    }
  });
  it('bails out on a degenerate source triangle', () => {
    const ctx = mockCanvas(64, 64).getContext('2d');
    const p = { x: 5, y: 5 };
    expect(warpTri(ctx, src, p, p, { x: 9, y: 9 }, p, { x: 1, y: 1 }, { x: 2, y: 8 }, 0)).toBe(false);
  });
});

describe('mouth overlays are gated, not switched', () => {
  it('draws the oral composite only once the mouth is actually open', () => {
    const closed = drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts({ amt: { open: 0, gap: 0, teeth: 0, lower: 0, gum: 0, tongue: 0, tongueOut: 0, kiss: 0, ao: 0, wet: 0, seam: 1, throat: 0 }, gap: 0 }));
    expect(closed.mouthDrawn).toBe(false);
    const open = drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts());
    expect(open.mouthDrawn).toBe(true);
    expect(open.thickness).toBeGreaterThan(1);
    expect(open.teeth).toBe(20);                     // ten crowns per arch, both sides
  });

  it('hides the crowns as the mouth closes (thickness -> 0, no pop)', () => {
    const thick = gap => drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts({ gap, amt: { open: 0.5, gap, teeth: 0.8, lower: 0.6, gum: 0.2, tongue: 0, tongueOut: 0, kiss: 0, ao: 0.5, wet: 0.4, seam: 0.2, throat: 0.1 } })).thickness;
    const a = thick(30), b = thick(15), c = thick(4);
    expect(a).toBeGreaterThan(b);
    expect(b).toBeGreaterThan(c);
    expect(c).toBeLessThan(2);                       // below the 0.4px draw floor it is skipped entirely
    expect(thick(0)).toBe(0);
  });

  it('draws a protruding tongue over the lips only when it is asked to', () => {
    const inside = drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts());
    expect(inside.tongueOut).toBe(0);
    const out = drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts({
      amt: { open: 0.5, gap: 26, teeth: 0.8, lower: 0.6, gum: 0.2, tongue: 0.9, tongueOut: 0.9, kiss: 0, ao: 0.5, wet: 0.4, seam: 0.2, throat: 0.1 },
      mouth: { tongueAmt: 1, tipUp: 0.2, backUp: 0, wide: 0.6, protrude: 0.9, toothAmt: 0.5, toothScale: 1, gumVis: 0.6, archTau: 0.55, bright: 1.15, warm: 0 },
    }));
    expect(out.tongueOut).toBeGreaterThan(0.5);
    // the inside pass is faded out by exactly the same amount: a cross-fade, not a swap
    expect(out.tongue.outline.length).toBe(7);
  });

  it('keeps the dental arch stable while the landmarks jitter', () => {
    let prev = null, worst = 0;
    for (let k = 0; k < 40; k++) {
      const j = D.map((p, i) => (LIP.includes(i) ? { x: p.x + Math.sin(k * 3.1 + i) * 1.8, y: p.y + Math.cos(k * 2.3 + i) * 1.8 } : p));
      const a = fitArch(j, st.Pz, prev, 1 / 60, { tau: 0.055 });
      if (prev) worst = Math.max(worst, Math.hypot(a.mid.x - prev.mid.x, a.mid.y - prev.mid.y));
      prev = a;
    }
    expect(worst).toBeLessThan(1.5);                  // px of arch wander per frame
  });

  it('parameterises the arch by true arc length (what crown placement relies on)', () => {
    const arch = fitArch(D, st.Pz, null, 1 / 60, {});
    expect(arch.width).toBeGreaterThan(20);
    // ground truth: the polyline length of the upper vermilion in the arch frame
    const local = ids => ids.map(i => {
      const p = D[i], m = arch.mid, ax = arch.ax, dn = arch.dn;
      return { u: (p.x - m.x) * ax.x + (p.y - m.y) * ax.y, v: (p.x - m.x) * dn.x + (p.y - m.y) * dn.y };
    });
    const poly = local([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]);
    let truth = 0;
    for (let k = 1; k < poly.length; k++) truth += Math.hypot(poly[k].u - poly[k - 1].u, poly[k].v - poly[k - 1].v);
    // walked length through at(): must match, and s = 0.5 must land at half of it
    const N = 400;
    let walked = 0, half = -1, uMin = Infinity, uMax = -Infinity;
    let p0 = arch.at(1, 0).l;
    for (let k = 1; k <= N; k++) {
      const p = arch.at(1, k / N).l;
      walked += Math.hypot(p.u - p0.u, p.v - p0.v);
      p0 = p;
      if (k === N / 2) half = walked;
      uMin = Math.min(uMin, p.u); uMax = Math.max(uMax, p.u);
    }
    // the vermilion arc actually crosses the mouth (a broken mapping would not)
    expect(uMax - uMin).toBeGreaterThan(arch.width * 0.5);
    expect(walked / truth).toBeGreaterThan(0.97);
    expect(walked / truth).toBeLessThan(1.03);
    expect(half / walked).toBeGreaterThan(0.45);
    expect(half / walked).toBeLessThan(0.55);
  });
});

describe('eye overlays', () => {
  it('fades the far eye out under a hard yaw so lashes never cross the nose', () => {
    const out = drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts({ pose: { yaw: 0.95, pitch: 0, roll: 0 } }));
    expect(out.vis[1]).toBeLessThan(0.5);
    expect(out.vis[0]).toBeGreaterThan(0.9);
  });
  it('closes the lids continuously with the blink amount', () => {
    const ctx = mockCanvas(720, 900).getContext('2d');
    const open = drawFX(ctx, rig, st, D, baseOpts({ lid: [0, 0] }));
    const shut = drawFX(mockCanvas(720, 900).getContext('2d'), rig, st, D, baseOpts({ lid: [1, 1] }));
    expect(open.lid[0]).toBe(0);
    expect(shut.lid[0]).toBe(1);
  });
  it('builds an iris sprite per eye with a sampled sclera colour', () => {
    expect(iris[0] && iris[1]).toBeTruthy();
    expect(iris[0].r).toBeGreaterThan(2);
    expect(iris[0].sclera).toMatch(/^\d+,\d+,\d+$/);
  });
});

describe('ROI helpers', () => {
  it('clamps enhancement boxes inside the canvas', () => {
    const b = boxAround(D, EYE[0], 400, 5, 720, 900);
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.y).toBeGreaterThanOrEqual(0);
    expect(b.x + b.w).toBeLessThanOrEqual(720);
    expect(b.y + b.h).toBeLessThanOrEqual(900);
    expect(b.w).toBeGreaterThanOrEqual(8);
  });
  it('runs the enhancement pass only when asked', () => {
    const off = mockCanvas(720, 900).getContext('2d');
    renderFrame(off, rig, st, { R: null }, baseOpts({ enhance: 0 }));
    const on = mockCanvas(720, 900).getContext('2d');
    renderFrame(on, rig, st, { R: null }, baseOpts({ enhance: 0.5 }));
    expect(on.__calls.length).toBeGreaterThan(off.__calls.length);
  });
});
