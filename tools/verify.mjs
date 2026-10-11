#!/usr/bin/env node
/* Pure-math unit tests for src/math.js — the geometry the whole puppet
   rests on.  No DOM, no canvas: if this fails, nothing else can work.
   usage: node tools/verify.mjs                                   */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
(0, eval)(await readFile(join(rootDir, 'src/math.js'), 'utf8'));
const FM = globalThis.FM;

let fails = 0, passes = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { passes++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fails++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const { m3, v3, kabsch, svd3, quatFromMat, matFromQuat, quatFromYPR, eulerFromQuat, oneEuro, oneEuroRun,
        quatSlew, quatAngle, quatFollow, quatMul, quatConj, polyOK, bounds, polyArea, splineClosed, slew, schmitt, softLimit,
        alphaFor, envelope } = FM;
let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

console.log('\n── quaternion / euler ----------------------------------------------------');
let worst = 0, worstAt = null;
for (let y = -1.45; y <= 1.45; y += 0.13)
  for (let p = -1.2; p <= 1.2; p += 0.17)
    for (let r = -1.3; r <= 1.3; r += 0.19) {
      const q = quatFromYPR(y, p, r);
      const e = eulerFromQuat(q);
      const R = matFromQuat(q);
      const R2 = m3.mul(m3.mul([Math.cos(y), 0, Math.sin(y), 0, 1, 0, -Math.sin(y), 0, Math.cos(y)],
                               [1, 0, 0, 0, Math.cos(p), -Math.sin(p), 0, Math.sin(p), Math.cos(p)]),
                        [Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r), 0, 0, 0, 1]);
      let d = 0;
      for (let i = 0; i < 9; i++) d = Math.max(d, Math.abs(R[i] - R2[i]));
      d = Math.max(d, Math.abs(e.yaw - y), Math.abs(e.pitch - p), Math.abs(e.roll - r));
      if (d > worst) { worst = d; worstAt = [y, p, r]; }
    }
ok('quatFromYPR == Ry·Rx·Rz and inverts exactly', worst < 1e-9,
   `worst ${worst.toExponential(2)} at ${worstAt.map(v => v.toFixed(2)).join(',')}`);
ok('yaw>0 swings the nose to +x', m3.mv(matFromQuat(quatFromYPR(0.5, 0, 0)), [0, 0, 1])[0] > 0);
ok('pitch>0 looks up', m3.mv(matFromQuat(quatFromYPR(0, 0.5, 0)), [0, 0, 1])[1] < 0);
ok('roll>0 dips the +x side (image-right goes down)', m3.mv(matFromQuat(quatFromYPR(0, 0, 0.5)), [1, 0, 0])[1] > 0);

console.log('\n── linear algebra --------------------------------------------------------');
const R0 = matFromQuat(quatFromYPR(0.6, -0.3, 0.2));
const { U, V, s } = svd3(m3.mul(R0, [0.9, 0, 0, 0, 1.3, 0, 0, 0, 0.5]));
ok('svd singular values', Math.abs(s[0] - 1.3) < 1e-9 && Math.abs(s[1] - 0.9) < 1e-9 && Math.abs(s[2] - 0.5) < 1e-9,
   s.map(v => v.toFixed(4)).join(', '));
let rerr = 0;
const Rr = m3.mul(U, m3.t(V));
for (let i = 0; i < 9; i++) rerr = Math.max(rerr, Math.abs(Rr[i] - R0[i]));
ok('svd recovers the rotation', rerr < 1e-9, `err ${rerr.toExponential(2)}`);
ok('svd factors stay proper rotations', Math.abs(m3.det(U) - 1) < 1e-9 && Math.abs(m3.det(V) - 1) < 1e-9,
   `detU ${m3.det(U).toFixed(9)} detV ${m3.det(V).toFixed(9)}`);
for (const scale of [1, 37, 0.2, 1000]) {
  const n = 60, src = [], dst = [];
  for (let i = 0; i < n; i++) {
    const p = [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1];
    src.push(p);
    dst.push(v3.add(v3.scale(m3.mv(R0, p), scale), [4, -3, 2]));
  }
  const f = kabsch(src, dst, src.map(() => 0.4 + rnd()));
  let rr = 0, tt = 0;
  for (let i = 0; i < 9; i++) rr = Math.max(rr, Math.abs(f.R[i] - R0[i]));
  for (let i = 0; i < 3; i++) tt = Math.max(tt, Math.abs(f.t[i] - [4, -3, 2][i]));
  ok(`kabsch recovers pose + scale (s=${scale})`, rr < 1e-6 && tt < 1e-6 && Math.abs(f.s - scale) < 1e-9,
     `rms ${f.rms.toExponential(1)}, ds ${Math.abs(f.s - scale).toExponential(1)}`);
}
{
  const refl = [1, 0, 0, 0, -1, 0, 0, 0, 1];
  const src = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]];
  const f = kabsch(src, src.map(p => m3.mv(refl, p)));
  ok('kabsch never returns a reflection', Math.abs(m3.det(f.R) - 1) < 1e-6, `det ${m3.det(f.R).toFixed(6)}`);
  ok('kabsch handles collinear/degenerate input', (() => {
    const g = kabsch([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]);
    return !g || (g.R.every(Number.isFinite) && Math.abs(m3.det(g.R) - 1) < 1e-6);
  })());
  ok('kabsch rejects NaN input safely', (() => {
    const g = kabsch([[NaN, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
    return g === null || g.R.every(Number.isFinite);
  })());
  let adversarialBad = 0;
  for (let it = 0; it < 400; it++) {
    const n = 3 + Math.floor(rnd() * 24), a = [], b = [];
    for (let i = 0; i < n; i++) {
      const p = [rnd() * 100 - 50, rnd() * 100 - 50, rnd() * 10 - 5];
      const q = [rnd() * 100 - 50, rnd() * 100 - 50, rnd() * 10 - 5];
      if (rnd() < 0.06) p[Math.floor(rnd() * 3)] = NaN;
      if (rnd() < 0.06) q[Math.floor(rnd() * 3)] = Infinity;
      a.push(p); b.push(q);
    }
    const g = kabsch(a, b);
    if (g && !(g.R.every(Number.isFinite) && g.t.every(Number.isFinite) && g.s > 0 && Number.isFinite(g.s))) adversarialBad++;
  }
  ok('kabsch survives 400 adversarial point sets', adversarialBad === 0, `bad ${adversarialBad}`);
}

console.log('\n── filters, limites, hysteresis ------------------------------------------');
{
  const f = oneEuro(2, 0.02, 1);
  let x = 0;
  for (let i = 0; i < 600; i++) x = oneEuroRun(f, i < 300 ? 0 : 1, i * 16.7);
  ok('one-euro converges', Math.abs(x - 1) < 0.02, `x ${x.toFixed(4)}`);
  const g = oneEuro(1.6, 0.0, 1);
  const vals = [];
  for (let i = 0; i < 400; i++) vals.push(oneEuroRun(g, 5 + (i % 2 ? 0.05 : -0.05), i * 16.7));
  const tail = vals.slice(-60);
  ok('one-euro kills 50 µm jitter', Math.max(...tail) - Math.min(...tail) < 0.02,
     `spread ${(Math.max(...tail) - Math.min(...tail)).toFixed(4)}`);
  const h = oneEuro(2, 0.3, 1);
  let y = 0;
  for (let i = 0; i < 12; i++) y = oneEuroRun(h, 1, i * 16.7);
  ok('one-euro tracks fast motion (low latency)', y > 0.25, `after 12 frames: ${y.toFixed(3)}`);
  ok('one-euro ignores NaN', Number.isFinite(oneEuroRun(h, NaN, 500)));
}
ok('slew respects the rate in both directions', Math.abs(slew(0, 1, 0.1, 5, 5) - 0.5) < 1e-12 && Math.abs(slew(1, 0, 0.1, 5, 5) - 0.5) < 1e-12);
ok('slew with dt=0 holds', slew(0.3, 1, 0, 5, 5) === 0.3);
let st2 = false, flips = 0, prevSt = st2;
for (let i = 0; i < 400; i++) {
  const v = 0.5 + 0.06 * Math.sin(i * 0.7);
  st2 = schmitt(st2, v, 0.62, 0.38);
  if (st2 !== prevSt) { flips++; prevSt = st2; }
}
ok('schmitt cannot strobe around a threshold', flips <= 1, `flips ${flips}`);
let mx = 0;
for (let i = 0; i < 2000; i++) mx = Math.max(mx, Math.abs(softLimit(i * 0.05, 0.85)));
ok('softLimit is a hard bound', mx <= 0.85 + 1e-12, `max ${mx.toFixed(6)}`);
/* ... but it must NOT squash ordinary angles: a 46° cap has to leave a
   35° turn (76% of the cap) essentially untouched, or every head turn
   comes out visibly wrong. */
let worstLin = 0;
for (let d = -35; d <= 35; d += 0.25) {
  const x = d * Math.PI / 180;
  worstLin = Math.max(worstLin, Math.abs(softLimit(x, 46 * Math.PI / 180) - x) / (Math.PI / 180));
}
ok('softLimit stays linear in the working range', worstLin < 0.5,
   `worst deviation ${worstLin.toFixed(3)}° at ±35°`);
const softDeg = d => softLimit(d * Math.PI / 180, 46 * Math.PI / 180) * 180 / Math.PI;
ok('softLimit only bends near the cap', Math.abs(softDeg(40) - 38.16) < 0.5 && softDeg(80) < 45.9 && softDeg(80) > 44,
   `40° -> ${softDeg(40).toFixed(2)}°, 80° -> ${softDeg(80).toFixed(2)}°`);
ok('softLimit is odd and monotone', (() => {
  let prev = -1e9, mon = true;
  for (let i = -2000; i <= 2000; i++) {
    const v = softLimit(i * 0.05, 0.85);
    if (v < prev - 1e-12) mon = false;
    prev = v;
    if (i < 0 && Math.abs(v + softLimit(-i * 0.05, 0.85)) > 1e-12) return false;
  }
  return mon;
})(), 'monotone, odd');

console.log('\n── rotation slew + follow ----------------------------------------------');
{
  const qa = quatFromYPR(0, 0, 0), qb = quatFromYPR(1.4, 0, 0);
  const step = quatSlew(qa, qb, 1 / 60, 3.0);
  ok('quatSlew caps angular speed', quatAngle(step) <= 3.0 / 60 + 1e-9, `${quatAngle(step).toFixed(4)} rad/step`);
  let q = qa, n = 0;
  while (quatAngle(quatMul(quatConj(q), qb)) > 1e-4 && n < 2000) { q = quatSlew(q, qb, 1 / 60, 4); n++; }
  ok('quatSlew reaches the target', n < 2000, `${n} steps (${(n / 60).toFixed(2)} s)`);
  let q2 = qa;
  for (let i = 0; i < 400; i++) q2 = quatFollow(q2, qb, 1 / 60, 12);
  ok('quatFollow converges', quatAngle(quatMul(quatConj(q2), qb)) < 1e-3);
  ok('rotations never produce NaN', [qa, qb].every(v => v.every(Number.isFinite)) && quatSlew(qa, qb, 1 / 60, 5).every(Number.isFinite));
}

/* envelope follower: attack/release, no NaN holes */
{
  let e = 0;
  for (let i = 0; i < 30; i++) e = envelope(e, 1, 1 / 60, 9, 2.2);
  ok('envelope rises toward its target', e > 0.9 && e <= 1, `after 0.5 s: ${e.toFixed(4)}`);
  let d = e;
  for (let i = 0; i < 60; i++) d = envelope(d, 0, 1 / 60, 9, 2.2);
  ok('envelope releases without NaN', Number.isFinite(d) && d < e * 0.25, `after 1 s: ${d.toFixed(4)}`);
  let f = 0;
  for (let i = 0; i < 120; i++) f = envelope(f, 1, 1 / 60, 9);          // 4-arg form
  let g = f;
  for (let i = 0; i < 120; i++) g = envelope(g, 0, 1 / 60, 9);
  ok('4-arg envelope falls back to the attack rate', Number.isFinite(g) && g < 0.2,
     `1 -> ${g.toFixed(5)} (no NaN)`);
  let h = 0.5, mon = true;
  for (let i = 0; i < 400; i++) { const n = envelope(h, Math.sin(i / 7), 1 / 60, 12, 5); if (!Number.isFinite(n)) mon = false; h = n; }
  ok('envelope tracks a moving target without blowing up', mon && Math.abs(h) <= 1, `final ${h.toFixed(4)}`);
}

console.log('\n── geometry guards ------------------------------------------------------');
ok('polyOK rejects empty / degenerate / non-finite clips',
   !polyOK([{ x: 0, y: 0 }]) && !polyOK([[0, 0], [0, 0], [0, 0]]) && !polyOK([[0, 0], [1, 1], [NaN, 2]]) && !polyOK([[0, 0], [1, 0]]));
ok('polyOK accepts a real triangle', polyOK([[0, 0], [10, 0], [0, 10]]));
const facePoly = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 0, y: 10 }];
ok('polygon helpers accept renderer {x,y} points', polyOK(facePoly) && Math.abs(polyArea(facePoly) - 50) < 1e-9
  && bounds(facePoly).w === 10 && bounds(facePoly).h === 10);
const sp = splineClosed([[0, 0], [10, 0], [10, 10], [0, 10]], 48);
ok('closed spline resamples without NaN', sp.length === 48 && sp.every(p => Number.isFinite(p[0]) && Number.isFinite(p[1])));

console.log(`\n${fails ? '✗' : '✓'} verify: ${passes} passed, ${fails} failed\n`);
process.exit(fails ? 1 : 0);
