/**
 * Temporal layer — the anti-flicker regression suite.
 *
 * Every claim the technique-map panel makes about temporal stability ("One-Euro
 * smoothing", "asymmetric gates instead of binary thresholds", "tracker dropout
 * handling", "peak hold for eye openness") is asserted here against numbers, so
 * a future tweak that reintroduces chatter fails a test instead of shipping.
 */
import { describe, it, expect } from 'vitest';
import { smoothK, Scalar, Gate, OneEuroField, Confidence, PeakTracker } from './temporal.js';
import { fitArch } from './mouth.js';

const DT = 1 / 60;
const run = (fn, n, dt = DT) => { let v; for (let i = 0; i < n; i++) v = fn(dt, i); return v; };
/** deterministic pseudo-noise (no Math.random: the tests must be reproducible) */
const noise = i => Math.sin(i * 12.9898) * 43758.5453 % 1;

describe('smoothK', () => {
  it('is a frame-rate independent exponential step', () => {
    expect(smoothK(0.1, 0.1)).toBeCloseTo(1 - Math.exp(-1), 6);
    // two half-steps equal one full step to first order of the exponential
    const a = smoothK(0.1, 0.05), b = 1 - (1 - a) * (1 - a);
    expect(b).toBeCloseTo(smoothK(0.1, 0.1), 6);
    expect(smoothK(0, 0.016)).toBeLessThan(1.0001);   // no division by zero
  });
});

describe('Scalar', () => {
  it('converges monotonically to the target without overshoot', () => {
    const s = new Scalar(0, 0.05);
    let prev = 0;
    for (let i = 0; i < 240; i++) { const v = s.update(1, DT); expect(v).toBeGreaterThanOrEqual(prev - 1e-9); prev = v; }
    expect(s.v).toBeGreaterThan(0.999);
    expect(s.v).toBeLessThanOrEqual(1);
    expect(s.value).toBe(s.v);
  });
  it('is dt-aware: 30 fps and 120 fps agree after the same elapsed time', () => {
    const a = new Scalar(0, 0.08), b = new Scalar(0, 0.08);
    run(dt => a.update(1, dt), 60, 1 / 60);
    run(dt => b.update(1, dt), 30, 1 / 30);
    expect(Math.abs(a.v - b.v)).toBeLessThan(0.002);
  });
  it('limits the rate of change (no teleporting head)', () => {
    const s = new Scalar(0, 0.02, 2.0);            // <= 2 units per second
    s.update(1, DT);
    expect(s.v).toBeLessThanOrEqual(2 * DT + 1e-9);
    run(() => s.update(1, DT), 180);
    expect(s.v).toBeGreaterThan(0.99);
  });
  it('ignores non-finite targets and snaps on demand', () => {
    const s = new Scalar(0.3, 0.1);
    expect(s.update(NaN, DT)).toBe(0.3);
    expect(s.update(Infinity, DT)).toBe(0.3);
    expect(s.snap(5)).toBe(5);
    expect(s.set(0.2).tau).toBe(0.2);
  });
});

describe('Gate — hysteresis instead of binary thresholds', () => {
  /** the reference implementation the original single-file app used */
  const binary = (x, thr) => (x > thr ? 1 : 0);

  it('does not chatter when a noisy signal straddles the threshold', () => {
    const g = new Gate(0.035, 0.13, 0.05, 0.13);   // the real mouth-open gate
    let prev = 0, gateMove = 0, binaryFlips = 0, lastBin = 0;
    for (let i = 0; i < 300; i++) {
      const raw = 0.0825 + 0.009 * Math.sin(i * 0.31) + 0.004 * noise(i);  // hovers mid-band
      const v = g.update(raw, DT);
      gateMove += Math.abs(v - prev); prev = v;
      const b = binary(raw, 0.082);
      if (b !== lastBin) binaryFlips++;
      lastBin = b;
    }
    const binaryMove = binaryFlips;                // a hard gate moves a full 0->1 each flip
    expect(binaryFlips).toBeGreaterThan(20);       // the old code flipped ~30 times in 5 s
    expect(gateMove).toBeLessThan(binaryMove * 0.35);   // the ramp + tau absorbs most of it
    expect(g.v).toBeGreaterThan(0.2); expect(g.v).toBeLessThan(0.85);
  });
  it('still fully opens and fully closes', () => {
    const g = new Gate(0.035, 0.13, 0.05, 0.13);
    run(() => g.update(1, DT), 240); expect(g.v).toBeGreaterThan(0.999);
    run(() => g.update(0, DT), 240); expect(g.v).toBeLessThan(0.001);
  });
  it('is monotone in the raw signal', () => {
    const g = new Gate(0.05, 0.2, 0.05, 0.05);
    let last = -1;
    for (let x = 0; x <= 0.3; x += 0.02) { run(() => g.update(x, DT), 120); expect(g.v).toBeGreaterThanOrEqual(last - 1e-9); last = g.v; }
  });
  it('honours asymmetric attack and release', () => {
    const g = new Gate(0.2, 0.3, 0.01, 0.25);      // snaps open, closes slowly
    const up = run(() => g.update(1, DT), 30);
    expect(up).toBeGreaterThan(0.85);              // open within half a second
    const down = run(() => g.update(0, DT), 30);   // half a second later
    expect(down).toBeLessThan(up * 0.3);           // it is closing
    expect(down).toBeGreaterThan(up * 0.1);        // ...but still visibly open
    run(() => g.update(0, DT), 600);
    expect(g.v).toBeLessThan(0.01);
    expect(g.snap(0.4)).toBe(0.4);
    expect(g.snap(9)).toBe(1);
  });
  it('ignores NaN', () => {
    const g = new Gate(0.2, 0.3, 0.05, 0.1, 0.6);
    expect(g.update(NaN, DT)).toBe(0.6);
  });
});

describe('OneEuroField', () => {
  const flatOf = (xs) => { const f = new Float32Array(xs.length * 3); xs.forEach((p, i) => { f[i * 3] = p[0]; f[i * 3 + 1] = p[1]; f[i * 3 + 2] = p[2]; }); return f; };
  const variance = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length; };

  it('snaps on the first frame (no slow fade-in from the origin)', () => {
    const f = new OneEuroField(2, { minCutoff: 1.2, beta: 22 });
    const out = f.update([[100, 200, 0], [300, 400, 0]], 0);
    expect(out[0]).toBe(100); expect(out[1]).toBe(200); expect(out[4]).toBe(400);
  });
  it('kills landmark jitter while tracking real motion', () => {
    // "Flicker" is frame-to-frame roughness, so score the second difference: a
    // pure lag would not show up here, and neither would a smooth signal.
    const rough = xs => { let s = 0; for (let i = 2; i < xs.length; i++) s += (xs[i] - 2 * xs[i - 1] + xs[i - 2]) ** 2; return s / (xs.length - 2); };
    const f = new OneEuroField(1, { minCutoff: 0.55, beta: 26, dCutoff: 1.1, scale: 640 });
    const raw = [], out = [], clean = [];
    for (let i = 0; i < 300; i++) {
      const t = i / 60;
      const c = 100 + 40 * Math.sin(t * 2.2);                       // the real head motion
      const x = c + 1.6 * noise(i);                                 // + detector noise
      raw.push(x); clean.push(c);
      out.push(f.update([[x, 0, 0]], i * (1000 / 60))[0]);
    }
    const tail = a => a.slice(120);
    expect(rough(tail(out))).toBeLessThan(rough(tail(raw)) * 0.3);
    // ...without losing the motion: correlation with the clean signal stays ~1
    const o = tail(out), c = tail(clean);
    const mo = o.reduce((x, y) => x + y, 0) / o.length, mc = c.reduce((x, y) => x + y, 0) / c.length;
    let num = 0, d1 = 0, d2 = 0;
    for (let i = 0; i < o.length; i++) { num += (o[i] - mo) * (c[i] - mc); d1 += (o[i] - mo) ** 2; d2 += (c[i] - mc) ** 2; }
    expect(num / Math.sqrt(d1 * d2)).toBeGreaterThan(0.995);
    // the price of smoothing is a little lag: it must stay a small fraction of
    // the excursion (RMS < 3 px on a +/-40 px move), not a frozen face
    expect(variance(o.map((v, i) => v - c[i]))).toBeLessThan(9);
    expect(variance(o)).toBeGreaterThan(variance(c) * 0.9);   // amplitude preserved
  });
  it('raises the cutoff for fast motion (the One-Euro part)', () => {
    const mk = beta => new OneEuroField(1, { minCutoff: 0.55, beta, dCutoff: 1.1, scale: 640 });
    const slow = mk(0), fast = mk(26);
    slow.update([[0, 0, 0]], 0); fast.update([[0, 0, 0]], 0);
    // a 300 px jump: the adaptive filter must catch up sooner than the fixed one
    let ts = 0, tf = 0;
    for (let i = 1; i <= 90; i++) {
      const t = i * (1000 / 60);
      const vs = slow.update([[300, 0, 0]], t)[0], vf = fast.update([[300, 0, 0]], t)[0];
      if (!ts && vs > 270) ts = i;
      if (!tf && vf > 270) tf = i;
    }
    expect(tf).toBeGreaterThan(0);
    expect(tf).toBeLessThanOrEqual(ts);
  });
  it('boosts selected landmarks (the lips get a higher cutoff)', () => {
    const f = new OneEuroField(2, { minCutoff: 0.55, beta: 26, dCutoff: 1.1, scale: 640 });
    f.boost = new Float32Array(2).fill(1); f.boost[1] = 2.5;
    f.update([[0, 0, 0], [0, 0, 0]], 0);
    const o1 = f.update([[100, 0, 0], [100, 0, 0]], 1000 / 60);
    expect(o1[3]).toBeGreaterThan(o1[0] + 5);      // mid-flight the boosted vertex is less laggy
    let last = null;
    for (let i = 2; i <= 40; i++) last = f.update([[100, 0, 0], [100, 0, 0]], i * (1000 / 60));
    expect(last[0]).toBeCloseTo(100, 1);           // both settle on the same value
    expect(last[3]).toBeCloseTo(100, 1);
  });
  it('accepts a weight, an array source and a reset', () => {
    const f = new OneEuroField(1, { minCutoff: 0.55, beta: 26, scale: 640 });
    f.update(flatOf([[0, 0, 0]]), 0);
    let light = 0;
    for (let i = 1; i <= 30; i++) light = f.update(flatOf([[100, 0, 0]]), i * 16.7, 0.25)[0];
    f.reset();
    expect(f.update(flatOf([[100, 0, 0]]), 40 * 16.7)[0]).toBe(100);   // re-init snaps
    expect(light).toBeLessThan(100);
  });
  it('is finite for duplicated timestamps and huge gaps', () => {
    const f = new OneEuroField(1, { minCutoff: 0.55, beta: 26, scale: 640 });
    f.update([[10, 0, 0]], 1000);
    for (const t of [1000, 1000, 1001, 999999]) {
      const o = f.update([[20, 0, 0]], t);
      expect(Number.isFinite(o[0])).toBe(true);
    }
  });
});

describe('Confidence (tracker dropout)', () => {
  it('rises fast when the face is found and falls slowly when it is lost', () => {
    const c = new Confidence();                    // fallTau 0.45, riseTau 0.18
    run(() => c.update(true, DT), 120);
    expect(c.value).toBeGreaterThan(0.99);
    const after10 = run(() => c.update(false, DT), 10);
    expect(after10).toBeGreaterThan(0.6);          // a 1/6 s dropout must not kill the render
    expect(after10).toBeLessThan(0.9);             // but it is audibly decaying
    run(() => c.update(false, DT), 240);
    expect(c.value).toBeLessThan(0.02);
    run(() => c.update(true, DT), 120);
    expect(c.value).toBeGreaterThan(0.99);
  });
  it('falls slower than it rises with the default taus', () => {
    const c = new Confidence();
    run(() => c.update(true, DT), 200);
    const start = c.value;
    c.update(false, DT);
    const fallStep = start - c.value;
    const c2 = new Confidence();
    c2.update(false, DT);
    c2.update(true, DT);
    expect(fallStep).toBeLessThan(c2.value);       // |d| per frame is smaller when falling
  });
  it('reports hit timing for the HUD', () => {
    const c = new Confidence();
    c.hit(1234);
    expect(c.lastHit).toBe(1234);
    expect(c.holdMs).toBeGreaterThan(0);
  });
});

describe('PeakTracker (eye-openness reference)', () => {
  it('holds the open peak and drops immediately on a blink', () => {
    const pt = new PeakTracker(0.25);              // initial .25, riseTau 6s, fallTau .35s, floor .08
    run(() => pt.update(1, DT), 600);
    expect(pt.v).toBeGreaterThan(0.8);             // 10 s towards the fully-open reference
    run(() => pt.update(1, DT), 1800);
    expect(pt.v).toBeGreaterThan(0.98);            // converged after ~40 s of holding
    const open = pt.v;
    const closing = run(() => pt.update(0.05, DT), 40);
    expect(closing).toBeLessThan(open * 0.3);      // a blink is caught within two thirds of a second
    expect(pt.v).toBeGreaterThanOrEqual(0.08);     // never below the floor
    for (let i = 0; i < 120; i++) pt.update(0.02, DT);
    expect(pt.v).toBeCloseTo(0.08, 6);
  });
  it('rises much more slowly than it falls', () => {
    const pt = new PeakTracker(0.5);
    pt.update(1, DT); const rise = pt.v - 0.5;
    const pt2 = new PeakTracker(0.5);
    pt2.update(0, DT); const fall = 0.5 - pt2.v;
    expect(fall).toBeGreaterThan(rise * 5);
  });
  it('ignores non-finite input', () => {
    const pt = new PeakTracker(0.4);
    expect(pt.update(NaN, DT)).toBe(0.4);
    expect(pt.update(undefined, DT)).toBe(0.4);
  });
});

describe('fitArch frame smoothing (dental arch)', () => {
  /** every landmark on an ellipse whose long axis is `ang`, corners exactly on it */
  const lips = (cx, cy, w, ang) => {
    const D = [];
    for (let i = 0; i < 478; i++) {
      const t = (i / 478) * Math.PI * 2, u = Math.cos(t) * w, v = Math.sin(t) * w * 0.3;
      D.push({ x: cx + u * Math.cos(ang) - v * Math.sin(ang), y: cy + u * Math.sin(ang) + v * Math.cos(ang) });
    }
    D[61] = { x: cx - w * Math.cos(ang), y: cy - w * Math.sin(ang) };
    D[291] = { x: cx + w * Math.cos(ang), y: cy + w * Math.sin(ang) };
    return D;
  };

  it('unwraps the arch angle across the -pi/pi branch', () => {
    const prev = fitArch(lips(0, 0, 40, 3.1), null, null, DT);
    expect(prev.ang).toBeCloseTo(3.1, 6);
    expect(prev.ready).toBe(true);
    // the identical orientation written as -3.18 (i.e. +2pi away) must not spin the teeth
    const cur = fitArch(lips(0, 0, 40, -3.18), null, prev, DT, { tau: 0.05 });
    expect(cur.ang).toBeGreaterThan(2.9);
    expect(cur.ang).toBeLessThan(3.3);
  });
  it('low-passes arch centre, width and the per-tooth local coordinates', () => {
    let prev = null, last = null;
    for (let i = 0; i < 200; i++) {
      const jx = (i % 2 ? 2.5 : -2.5), jw = (i % 2 ? 2 : -2);
      last = fitArch(lips(jx, 0, 40 + jw, 0.05), null, prev, DT, { tau: 0.05 });
      prev = last;
    }
    expect(Math.abs(last.mid.x)).toBeLessThan(1.2);
    expect(Math.abs(last.width - 80)).toBeLessThan(1.5);
    expect(Number.isFinite(last.up[3].l.u)).toBe(true);
    expect(Number.isFinite(last.toWorld(last.up[3].l).x)).toBe(true);
  });
  it('reproduces the same arch for the same input (no hidden state drift)', () => {
    const a = fitArch(lips(0, 0, 40, 0.2), null, null, DT, { tau: 0.05 });
    const b = fitArch(lips(0, 0, 40, 0.2), null, null, DT, { tau: 0.05 });
    expect(a.mid.x).toBe(b.mid.x);
    expect(a.width).toBe(b.width);
    expect(a.up.length).toBe(b.up.length);
  });
});
