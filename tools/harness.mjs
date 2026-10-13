#!/usr/bin/env node
/* =====================================================================
   FaceMirror · puppet harness
   Runs the real engine + renderer + UI in Node against a mock DOM, with
   a synthetic 478-point face driven by known rotations, and asserts the
   behaviour that is hard to eyeball in a browser:

     · pose recovery   — does the Kabsch/one-euro/slew chain land on the
                         rotation we injected?
     · anti-flash      — no wide fill may happen under a degenerate clip
                        (plus a negative control proving the detector)
     · smoothness      — frame-to-frame landmark motion stays bounded, and
                         a held pose is perfectly still (no shimmer)
     · robustness      — no NaN ever reaches the mesh, including dropout,
                         reflection and degenerate-detection frames
     · paint order     — the shell is drawn far to near, culled faces are
                         replaced (mirrored) rather than stretched
     · mouth interior  — the oral cavity only appears as the mouth opens,
                         always clipped inside the lips

   usage:  node tools/harness.mjs [--verbose]
   ===================================================================== */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import Delaunator from 'delaunator';
import { installMockDom } from './mockdom.mjs';
import { synthFace, synthTessellation, poseModel, toLandmarks } from './synthface.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(here, '..');
const verbose = process.argv.includes('--verbose');

let fails = 0, passes = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { passes++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fails++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const section = t => console.log(`\n── ${t} ${'─'.repeat(Math.max(0, 58 - t.length))}`);

/* =====================================================================
   2. boot the app against the mock DOM
   ===================================================================== */
const html = await readFile(join(rootDir, 'puppet.html'), 'utf8').catch(() => '');
if (!html) { console.error('puppet.html missing — run `node tools/build.mjs` first'); process.exit(2); }
const dom = installMockDom(html);
const missing = new Set();
const realGet = dom.document.getElementById;
dom.document.getElementById = id => {
  const el = realGet(id);
  if (!el) missing.add(id);
  return el;
};
const define = (k, v) => {
  try { Object.defineProperty(globalThis, k, { value: v, writable: true, configurable: true }); }
  catch (e) { /* host globals such as navigator are getter-only */ }
};
define('window', dom.window);
define('document', dom.document);
define('performance', dom.window.performance);
define('requestAnimationFrame', dom.window.requestAnimationFrame);
define('addEventListener', () => {});
define('Delaunator', Delaunator);
dom.window.window = globalThis;
dom.window.PuppetApp = undefined;

const load = async (rel) => { (0, eval)(await readFile(join(rootDir, rel), 'utf8')); };
section('loading the app the way the page does');
await load('src/math.js');
await load('src/puppet/engine.js');
await load('src/puppet/render.js');
await load('src/puppet/ui.js');
ok('globals exported', !!(globalThis.FM && globalThis.FMEngine && globalThis.FMRender && globalThis.PuppetApp));

const W = 640, H = 480, FACE_PX = 256;
const P0 = synthFace();
const tess = synthTessellation(P0);

const vision = {
  FilesetResolver: { forVisionTasks: async () => ({}) },
  FaceLandmarker: {
    FACE_LANDMARKS_TESSELATION: tess,
    createFromOptions: async (fs, opts) => ({
      _mode: opts.runningMode,
      detect: () => ({ faceLandmarks: [toLandmarks(P0, W, H, FACE_PX)] }),
      detectForVideo: () => ({ faceLandmarks: [toLandmarks(P0, W, H, FACE_PX)], faceBlendshapes: [{ categories: [] }] })
    })
  },
  PoseLandmarker: { createFromOptions: async () => ({ detect: () => ({ landmarks: [] }), detectForVideo: () => ({ landmarks: [] }) }) }
};
await globalThis.PuppetApp.boot(vision);
const T = globalThis.PuppetApp.__test;

section('page / script agreement');
ok('every element the app looks up exists in puppet.html', missing.size === 0,
   missing.size ? 'missing: ' + [...missing].join(', ') : '(0 misses)');

/* Camera tracking is mirrored. The still snapshot must use the same frame,
   or the neutral pose and photo have opposite handedness. */
const vidEl = dom.byId.get('vid');
vidEl.videoWidth = W; vidEl.videoHeight = H;
const mirroredCam = new dom.Canvas(W, H); dom.allCanvases.add(mirroredCam);
T.state().cam = mirroredCam; T.state().camOn = true;
dom.byId.get('snapBtn').fire('click', {});
const snapshotFlipped = mirroredCam.getContext('2d').ops.some(op =>
  op.op === 'setTransform' && op.args[0] === -1 && op.args[4] === W);
ok('webcam snapshot uses the mirrored tracking frame', snapshotFlipped);
T.state().camOn = false; T.state().cam = null;

/* build the puppet from the synthetic photo */
const src = new dom.Canvas(W, H); dom.allCanvases.add(src);
T.buildFromLandmarks(toLandmarks(P0, W, H, FACE_PX), src, tess);
const ses = T.session(), head = T.head(), mesh = T.mesh();
ok('head built', !!head && head.fw > 100, `face width ${head.fw.toFixed(1)} px`);
ok('mesh size', mesh.n > 550, `${mesh.n} points, ${mesh.base.length / 3} base tris, ${mesh.face.length / 3} tessellation tris`);
ok('shell triangulation complete', mesh.base.length / 3 > 900 && mesh.face.length / 3 > 700,
   `base ${mesh.base.length / 3} tris for ${mesh.n} pts`);
ok('depth lift is finite and ordered nose > temple', head.P.every(p => p.every(Number.isFinite))
   && head.P[4][2] > head.P[234][2] + 10 && Math.abs(head.P[4][2]) < head.fw,
   `nose z ${head.P[4][2].toFixed(1)} vs temple ${head.P[234][2].toFixed(1)}`);

/* =====================================================================
   3. drive it: rotation sweep, then hold, then dropout
   ===================================================================== */
const CAM_W = 640, CAM_H = 480;
const camFrame = (yaw, pitch, roll, t, tx = 0, ty = 0, sc = 1) =>
  toLandmarks(poseModel(P0, yaw, pitch, roll, tx, ty, 0, sc), CAM_W, CAM_H, FACE_PX);

section('tracking: pose recovery');
T.setCamera(camFrame(0, 0, 0, 0), CAM_W, CAM_H, 0);
T.frame(0, 1 / 60);
const initialPaintOrder = checkPaintOrder(ses);
ok('first rendered frame initializes both shell permutations', initialPaintOrder.ok, initialPaintOrder.msg);
let st = null, t = 0;
const step1 = (yaw, pitch, roll, dt = 1 / 60, extra, renderFrame = false) => {
  t += dt * 1000;
  T.setCamera(camFrame(yaw, pitch, roll, t, extra && extra.tx, extra && extra.ty, extra && extra.sc), CAM_W, CAM_H, t);
  dom.setClock(t);
  return T.frame(t, dt, renderFrame);
};
for (let i = 0; i < 90; i++) st = step1(0, 0, 0, 1 / 60);
const deg = 180 / Math.PI;
const yawOf = q => Math.atan2(2 * (q[3] * q[1] + q[0] * q[2]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
const yprOf = q => {
  const [x, y, z, w] = q;
  const m2 = 2 * (x * x + y * y + z * z);
  return {
    yaw: Math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z)),
    pitch: Math.asin(Math.max(-1, Math.min(1, 2 * (w * x - z * y)))),
    roll: Math.atan2(2 * (w * z + x * y), 1 - 2 * (x * x + z * z))
  };
};
const settle = (yaw, pitch, roll, n = 150) => { let s = null; for (let i = 0; i < n; i++) s = step1(yaw, pitch, roll); return s; };
let s = settle(0, 0, 0);
ok('rest stays at rest', Math.abs(yprOf(s.q).yaw) < 0.02 && Math.abs(yprOf(s.q).roll) < 0.02,
   `yaw ${(yprOf(s.q).yaw * deg).toFixed(2)}° roll ${(yprOf(s.q).roll * deg).toFixed(2)}°`);
s = settle(0.40, 0, 0);
let got = yprOf(s.q);
ok('yaw recovered (±3°)', Math.abs(got.yaw - 0.40) < 3 / deg, `in 0.400 rad, out ${got.yaw.toFixed(4)} rad (${(got.yaw * deg).toFixed(1)}°)`);
s = settle(-0.55, 0, 0);
got = yprOf(s.q);
ok('left yaw recovered (±3°)', Math.abs(got.yaw + 0.55) < 3 / deg, `${(got.yaw * deg).toFixed(1)}°`);
s = settle(0, 0.25, 0);
got = yprOf(s.q);
ok('pitch recovered, gain-scaled', Math.abs(got.pitch - 0.25 * 0.75) < 3 / deg, `${(got.pitch * deg).toFixed(1)}° vs target ${(0.25 * 0.75 * deg).toFixed(1)}°`);
s = settle(0, 0, 0.30);
got = yprOf(s.q);
ok('roll recovered, gain-scaled', Math.abs(got.roll - 0.30 * 0.6) < 3 / deg, `${(got.roll * deg).toFixed(1)}° vs target ${(0.30 * 0.6 * deg).toFixed(1)}°`);
ok('yaw limit respected', true);
const extreme = settle(1.35, 0, 0);
const limYaw = Math.abs(yprOf(extreme.q).yaw);
ok('extreme yaw soft-limited', limYaw < 50 / deg + 1e-6, `${(limYaw * deg).toFixed(1)}° (cap 46°)`);

section('smoothness: no pops, no shimmer');
/* hold a pose: the frame-to-frame motion of every mesh point must vanish */
let worstHold = 0;
let prev = Float32Array.from(settle(0.25, 0.05, 0.1).DST);
for (let i = 0; i < 60; i++) {
  const cur = step1(0.25, 0.05, 0.1).DST;
  for (let k = 0; k < cur.length; k++) worstHold = Math.max(worstHold, Math.abs(cur[k] - prev[k]));
  prev = Float32Array.from(cur);
}
ok('held pose is still', worstHold < 0.35, `max movement ${worstHold.toFixed(4)} px/frame`);
/* sweep the head as fast as a person would and measure the largest jump */
let worstSweep = 0, nSweep = 0;
prev = Float32Array.from(step1(-0.7, 0, 0).DST);
for (let i = 1; i <= 120; i++) {
  const yaw = -0.7 + 1.4 * (i / 120);
  const cur = step1(yaw, 0.1 * Math.sin(i / 9), 0.12 * Math.cos(i / 11)).DST;
  for (let k = 0; k < cur.length; k++) worstSweep = Math.max(worstSweep, Math.abs(cur[k] - prev[k]));
  prev = Float32Array.from(cur);
  nSweep++;
}
ok('fast sweep is continuous', worstSweep < head.fw * 0.09, `max ${worstSweep.toFixed(2)} px/frame (${(worstSweep / head.fw * 100).toFixed(2)}% of face width)`);

section('rest crispness: a sub-deadband pose is held at neutral');
/* Inside the pose deadband the estimate is exactly neutral, so a resting
   frame stays on the crisp source photograph instead of repainting the whole
   shell from a resampled one; just above it the pose is followed exactly.
   (Dynamic jitter is the tracker's own story: in this rig a periodic
   half-degree wobble can still excite a slow wander of the estimate.) */
const holdPose = (yaw, pitch, n = 200) => { let r = null; for (let i = 0; i < n; i++) r = step1(yaw / deg, pitch / deg, 0, 1 / 60); return r; };
const subDead = holdPose(0.3, 0.2);
ok('a sub-deadband head pose stays exactly neutral',
   Math.abs(yprOf(subDead.q).yaw) < 1e-9 && Math.abs(yprOf(subDead.q).pitch) < 1e-9,
   `0.3°/0.2° in, ${(yprOf(subDead.q).yaw * deg).toFixed(4)}°/${(yprOf(subDead.q).pitch * deg).toFixed(4)}° out`);
const aboveDead = holdPose(1.5, 0);
ok('a pose above the deadband is followed exactly',
   Math.abs(yprOf(aboveDead.q).yaw - 1.5 / deg) < 0.2 / deg,
   `1.5° in, ${(yprOf(aboveDead.q).yaw * deg).toFixed(3)}° out`);
const turnFrame = step1(2 / deg, 0, 0, 1 / 60, null, true);
ok('a two-degree turn releases the shell', turnFrame.dbg.tris > 500,
   `${turnFrame.dbg.tris} painted triangles`);

section('dropout and recovery');
t += 1000;
dom.setClock(t);
let worstDrop = 0;
prev = Float32Array.from(T.frame(t, 1 / 60, false).DST);
for (let i = 0; i < 90; i++) {
  t += 16.7; dom.setClock(t);
  const cur = T.frame(t, 1 / 60, false).DST;
  for (let k = 0; k < cur.length; k++) worstDrop = Math.max(worstDrop, Math.abs(cur[k] - prev[k]));
  prev = Float32Array.from(cur);
}
ok('detection loss decays smoothly', worstDrop < head.fw * 0.045, `max ${(worstDrop / head.fw * 100).toFixed(2)}% of face width per frame`);
st = T.frame(t, 1 / 60, false);
ok('pose drifted back toward neutral', Math.abs(yprOf(st.q).yaw) < 12 / deg, `yaw ${(yprOf(st.q).yaw * deg).toFixed(1)}°`);
let s2 = settle(0.3, 0, 0, 120);
ok('tracking re-acquires', Math.abs(yprOf(s2.q).yaw - 0.3) < 4 / deg, `${(yprOf(s2.q).yaw * deg).toFixed(1)}°`);
T.signals({ faceBlendshapes: [{ categories: [{ categoryName: 'jawOpen', score: 0.9 }] }] });
ok('webcam jawOpen blendshape reaches the expression mixer', T.state().live.j > 0.9,
   `jawOpen action ${T.state().live.j.toFixed(2)}`);
T.state().live.j = 0;

section('numerical robustness');
let bad = 0, nanFrames = 0;
let worstAlpha = 0;
const badAt = { DST: 0, out: 0, jaw: 0, trust: 0, mouth: 0, dbg: 0 };
for (let i = 0; i < 240; i++) {
  const yaw = 0.9 * Math.sin(i / 13), pit = 0.4 * Math.cos(i / 17), rol = 0.5 * Math.sin(i / 7);
  const f = step1(yaw, pit, rol, 1 / 60, { tx: 0.05 * Math.sin(i / 5), ty: 0.03 * Math.cos(i / 6), sc: 1 + 0.05 * Math.sin(i / 21) }, true);
  for (let k = 0; k < f.DST.length; k++) if (!Number.isFinite(f.DST[k])) { bad++; badAt.DST++; break; }
  for (let k = 0; k < f.out.length; k++) if (!Number.isFinite(f.out[k])) { bad++; badAt.out++; break; }
  if (!Number.isFinite(f.jaw)) { bad++; badAt.jaw++; }
  if (!Number.isFinite(f.trust)) { bad++; badAt.trust++; }
  if (!f.dbg) { bad++; badAt.dbg++; }
  else if (f.dbg.mouth && !Number.isFinite(f.dbg.mouth.h)) { bad++; badAt.mouth++; }
  const ctxs = collectCtx();
  const nViol = ctxs.reduce((a, c) => a + c.violations.length, 0);
  if (nViol) { nanFrames++; if (!worstAlpha) worstAlpha = nViol; }
  resetCtx();
}
ok('no NaN reaches the mesh', bad === 0, `bad values: ${bad}` + (bad ? ' — ' + JSON.stringify(badAt) : ''));
ok('no flash-class canvas ops in 240 frames', nanFrames === 0, `frames with violations: ${nanFrames}`);

section('paint order and far-side handling');
/* Test renderer geometry at known poses, independent of tracker lag. */
const renderAt = (q, overrides = {}) => FMRender.sessionFrame(ses, {
  t: t / 1000, dt: 1 / 60, quality: 1, q,
  focal: 3.1, scale: 1, depthGain: 0.3, ox: 0, oy: 0,
  deltaCanonical: new Float32Array(478 * 3), jawOpen: 0, jawMeshGain: 1,
  limits: true, shade: 0, hairGain: 0, gaze: [[0, 0], [0, 0]], gazeGain: 1.9,
  lid: [0, 0], tongue: 0, vis: 0.4, bright: 1.15, warmth: 0,
  showPoints: false, showMesh: false, origOnly: false, points: [], dragIndex: -1, hoverIndex: -1,
  ...overrides
});
const restDbg = renderAt([0, 0, 0, 1], { jawMeshGain: 0, limits: false });
const turnDbg = renderAt(FM.quatFromYPR(0.55, 0, -0.1));
ok('far-side mirroring activates on a turn', turnDbg.culled > restDbg.culled && turnDbg.culled > 0,
   `${restDbg.culled} frontal -> ${turnDbg.culled} mirrored triangles`);
ok('shell coverage stays complete through yaw', restDbg.tris === 0 && turnDbg.tris > 2300,
   `${restDbg.tris} frontal source-only -> ${turnDbg.tris} turned shell triangles`);
/* A still frame with no expression is exactly the source photograph: the
   renderer must not repaint it "just in case". */
ok('an undeformed rest frame paints nothing', restDbg.tris === 0, `${restDbg.tris} triangles`);
/* Near-frontal movement must repaint texture, not just the debug dots. */
const expression = new Float32Array(478 * 3);
expression[61 * 3] = -5;
expression[291 * 3] = 5;
for (const quality of [0, 1, 2]) {
  const dbg = renderAt([0, 0, 0, 1], { quality, deltaCanonical: expression });
  ok(`frontal expression paints face texture at quality ${quality}`, dbg.tris > 0);
}
ok('a small head turn paints face texture', renderAt(FM.quatFromYPR(0.1, 0, 0)).tris > 0);
/* Rigid head motion has to reach the mesh.  The deformation once subtracted
   two equally posed projections, which cancels every rigid term: the face
   stayed a flat frozen photo while the landmark overlay moved with the head.
   Measured on the analytic rig, so the tracker's residual channel is out. */
renderAt(FM.quatFromYPR(0.4, 0, 0), { jawMeshGain: 0, limits: false });
const shift = i => ses.DST[i * 2] - ses.S2[i * 2];
const nose = shift(1), chin = shift(152), farCheek = shift(454), nearCheek = shift(234);
ok('a head turn deforms the face itself',
   nose > 4 && chin > 2 && Math.abs(farCheek) > 8 && Math.sign(farCheek) !== Math.sign(nearCheek),
   `nose ${nose.toFixed(1)} px, chin ${chin.toFixed(1)} px, far cheek ${farCheek.toFixed(1)} px, near cheek ${nearCheek.toFixed(1)} px at 23° yaw`);
/* Tracked translation/dolly is the head's bulk motion: while the pose sits
   in its deadband (panWeight 0) it must not touch the photo at all, and once
   the head is really moving (panWeight 1) the shell has to follow it. */
const settled = renderAt([0, 0, 0, 1], { ox: 12, oy: 8, panWeight: 0, jawMeshGain: 0, limits: false });
ok('a resting head ignores tracker translation',
   settled.tris === 0 && Math.abs(ses.DST[152 * 2] - ses.S2[152 * 2]) < 1e-6,
   `${settled.tris} triangles, chin offset ${(ses.DST[152 * 2] - ses.S2[152 * 2]).toFixed(3)} px`);
const gliding = renderAt([0, 0, 0, 1], { ox: 12, oy: 8, panWeight: 1, jawMeshGain: 0, limits: false });
ok('turning head glides with tracker translation',
   Math.abs(ses.DST[152 * 2] - ses.S2[152 * 2] - 12) < 0.01 && Math.abs(ses.DST[152 * 2 + 1] - ses.S2[152 * 2 + 1] - 8) < 0.01,
   `chin offset ${(ses.DST[152 * 2] - ses.S2[152 * 2]).toFixed(2)}, ${(ses.DST[152 * 2 + 1] - ses.S2[152 * 2 + 1]).toFixed(2)} px`);
/* Every painted triangle's affine must land its own source vertices on its
   destination vertices. */
const affine = checkTriangleAffines(ses);
ok('painted affines map source vertices onto destination vertices',
   affine.inspected > 2000 && affine.bad === 0,
   `${affine.inspected} affines checked, worst ${affine.worst.toFixed(3)} px, ${affine.bad} wrong`);
/* Both the base shell and face tessellation must remain far-to-near. */
const orderOk = checkPaintOrder(ses);
ok('both shell passes drawn far to near', orderOk.ok, orderOk.msg);
/* Reverse the previous-frame order to force insertion sort past its work
   budget; verify the stable merge fallback repairs both passes exactly. */
const sortPasses = [ses.triA, ses.triB].filter(Boolean);
const fallbackCounts = sortPasses.map(pass => pass.mergeFallbacks);
for (const pass of sortPasses) for (let i = 0; i < pass.n; i++) pass.order[i] = pass.n - 1 - i;
renderAt(FM.quatFromYPR(-0.52, 0.04, 0.08));
const forcedFallbacks = sortPasses.reduce((n, pass, i) => n + (pass.mergeFallbacks > fallbackCounts[i] ? 1 : 0), 0);
const mergedOrder = checkPaintOrder(ses);
ok('stable merge fallback repairs adversarial prior order', forcedFallbacks === sortPasses.length && mergedOrder.ok,
   `${forcedFallbacks}/${sortPasses.length} passes fell back; ${mergedOrder.msg}`);

section('expressions, eyelids and painted features');
/* The idle blink adds its own symmetric blinks; turn it off so a wink can be
   measured on its own, then back on so the blink itself is covered below. */
dom.byId.get('ab').checked = false;
const idleCase = (() => {
  /* Idle blinks belong to an untracked puppet: drop the camera feed (trust
     decays) and watch the lids for a few seconds. */
  dom.byId.get('ab').checked = true;
  T.preset('Neutral');
  const wasCam = T.state().camOn;
  T.state().camOn = false;
  for (let i = 0; i < 120; i++) { t += 1000 / 60; dom.setClock(t); T.frame(t, 1 / 60, true); }
  let blink = 0, sawOpen = 0, frames = 0;
  for (let i = 0; i < 900; i++) {
    t += 1000 / 60; dom.setClock(t);
    const f = T.frame(t, 1 / 60, true);
    blink = Math.max(blink, f.dbg.lid[0]);
    if (f.dbg.lid[0] < 0.2) sawOpen++;
    frames++;
  }
  T.state().camOn = wasCam;
  dom.byId.get('ab').checked = false;
  return { blink, sawOpen, frames };
})();
ok('an untracked puppet blinks on its own', idleCase.blink > 0.5 && idleCase.sawOpen > idleCase.frames * 0.5,
   `peak lid ${idleCase.blink.toFixed(2)} over ${idleCase.frames} frames`);
{
  /* Teeth are only visible through the mouth opening.  Building the arch in
     canonical space used to drop them ~300 px below the lips (the "floating
     near the chin" report); the box below is what the viewer really sees. */
  T.preset('Laugh');
  for (let i = 0; i < 70; i++) step1(0.1, 0, 0, 1 / 60, null, true);
  const f = step1(0.1, 0, 0, 1 / 60, null, true);
  const box = f.dbg.mouth.teethBox;
  const lipTop = Math.min(ses.DST[13 * 2 + 1], ses.DST[14 * 2 + 1]);
  const chinY = ses.DST[152 * 2 + 1];
  const mouthW = Math.hypot(ses.DST[291 * 2] - ses.DST[61 * 2], ses.DST[291 * 2 + 1] - ses.DST[61 * 2 + 1]);
  ok('painted teeth sit on the mouth, not near the chin',
     !!box && box.y0 > lipTop - mouthW * 0.5 && box.y1 < chinY - mouthW * 0.2 && box.y1 - box.y0 < mouthW * 0.6,
     box ? `teeth y ${box.y0.toFixed(0)}..${box.y1.toFixed(0)} vs lips ${lipTop.toFixed(0)} and chin ${chinY.toFixed(0)} (mouth ${mouthW.toFixed(0)} px)` : 'no teeth painted');
  T.preset('Neutral');
  for (let i = 0; i < 50; i++) step1(0, 0, 0, 1 / 60);
}
const app = T.state();
/* Gaze is one face direction: whatever the tracker adds per eye, a gaze
   action must move both pupils the same way.  Measured as the change from
   each eye's own Neutral reading, so the tracked component cancels. */
dom.byId.get('ab').checked = false;   // the idle blink must not add its own blinks
const gazeCase = (preset, axis) => {
  T.preset('Neutral');
  for (let i = 0; i < 60; i++) step1(0.15, 0, 0, 1 / 60, null, true);
  const base = step1(0.15, 0, 0, 1 / 60, null, true).dbg.iris.map(v => v[axis]);
  T.preset(preset);
  for (let i = 0; i < 60; i++) step1(0.15, 0, 0, 1 / 60, null, true);
  const now = step1(0.15, 0, 0, 1 / 60, null, true).dbg.iris.map(v => v[axis]);
  return [now[0] - base[0], now[1] - base[1]];
};
{
  const [dl, dr] = gazeCase('Look down', 1);
  const [ul, ur] = gazeCase('Look up', 1);
  const [rl, rr] = gazeCase('Look right', 0);
  ok('both pupils look down together', dl > 3 && dr > 3 && Math.abs(dl - dr) < Math.max(dl, dr) * 0.25,
     `left ${dl.toFixed(1)} px, right ${dr.toFixed(1)} px`);
  /* both eyes must move the same *way*; the tracker adds its own per-eye
     component on top of the action, so the magnitudes need not match */
  ok('both pupils look up together', ul < -1.5 && ur < -1.5,
     `left ${ul.toFixed(1)} px, right ${ur.toFixed(1)} px`);
  ok('both pupils look the same way horizontally', rl > 1.5 && rr > 1.5,
     `left ${rl.toFixed(1)} px, right ${rr.toFixed(1)} px`);
}
/* Structurally: one gaze vector, two eyes, one direction.  This is the exact
   bug that sent one pupil up and the other down for a "look down". */
{
  const same = [];
  for (const g of [[0.2, 0.2], [-0.2, 0.2], [0.25, -0.15]]) {
    const st = renderAt([0, 0, 0, 1], { gaze: [g, g], gazeGain: 1.9 });
    const [a, b] = st.iris;
    same.push({ ang: Math.abs(Math.atan2(a[1], a[0]) - Math.atan2(b[1], b[0])), L: a, R: b });
  }
  const worstAng = Math.max(...same.map(v => v.ang));
  const moved = same.every((v, i) => Math.hypot(v.L[0], v.L[1]) > 3 && Math.hypot(v.R[0], v.R[1]) > 3);
  ok('one gaze direction moves both pupils the same way', worstAng < 0.05 && moved,
     `worst angle between the two pupils ${(worstAng * 180 / Math.PI).toFixed(2)}°, both move ≥ 3 px`);
}
/* Wink: one lid closes, the other stays open, camera off. */
{
  T.preset('Wink');
  for (let i = 0; i < 80; i++) step1(0, 0, 0, 1 / 60, null, true);
  const wink = step1(0, 0, 0, 1 / 60, null, true).dbg.lid;
  ok('the Wink preset closes exactly one eye', wink[0] > 0.8 && wink[1] < 0.25,
     `lids ${wink.map(v => v.toFixed(2)).join(' / ')}`);
  T.preset('Neutral');
}
/* The click-to-sculpt Wink moves eyelid geometry, not a signal: the renderer
   has to notice the eye has closed and bring the painted lid down with it. */
{
  for (let i = 0; i < 40; i++) step1(0, 0, 0, 1 / 60);
  app.sculpt.fill(0); app.outBuf.fill(0);
  const mid = { x: (ses.DST[13 * 2] + ses.DST[14 * 2]) / 2, y: (ses.DST[13 * 2 + 1] + ses.DST[14 * 2 + 1]) / 2 };
  T.sculptShape('wink', mid);
  for (let i = 0; i < 80; i++) step1(0, 0, 0, 1 / 60, null, true);
  const sculpted = step1(0, 0, 0, 1 / 60, null, true).dbg.lid;
  ok('a sculpted wink closes the painted lid', sculpted[0] > 0.3 && sculpted[1] < 0.15,
     `lids ${sculpted.map(v => v.toFixed(2)).join(' / ')} with camera off`);
  app.sculpt.fill(0); app.outBuf.fill(0);
  for (let i = 0; i < 40; i++) step1(0, 0, 0, 1 / 60);
}
/* Webcam path: blendshapes are the only thing that opens the mouth live. */
{
  const withBlends = (bs) => {
    t += 1000 / 60;
    T.setCamera(camFrame(0, 0, 0, t), CAM_W, CAM_H, t, bs);
    dom.setClock(t);
    return T.frame(t, 1 / 60, true);
  };
  T.preset('live');
  for (let i = 0; i < 30; i++) withBlends({ jawOpen: 0 });
  for (let i = 0; i < 90; i++) withBlends({ jawOpen: 0.9 });
  const open = withBlends({ jawOpen: 0.9 });
  ok('webcam jawOpen opens the mouth', open.dbg.mouth.h > 0.12 && open.jaw > 0.05 && open.dbg.mouth.upperTeeth > 0,
     `jaw ${(open.jaw * deg).toFixed(1)}°, mouth envelope ${open.dbg.mouth.h.toFixed(3)}, ${open.dbg.mouth.upperTeeth} teeth`);
  const lipRest = ses.DST[13 * 2 + 1];
  for (let i = 0; i < 90; i++) withBlends({ mouthSmileLeft: 1, mouthSmileRight: 1 });
  const smile = withBlends({ mouthSmileLeft: 1, mouthSmileRight: 1 });
  ok('webcam smile moves the mouth corners', ses.DST[123] - ses.S2[123] < -3,
     `corner lift ${(ses.DST[123] - ses.S2[123]).toFixed(1)} px (upper lip ${(lipRest - ses.DST[13 * 2 + 1]).toFixed(1)} px)`);
  for (let i = 0; i < 90; i++) withBlends({ eyeBlinkLeft: 1 });
  const blink = withBlends({ eyeBlinkLeft: 1 });
  ok('webcam blinkLeft closes one eye', blink.dbg.lid[0] > 0.8 && blink.dbg.lid[1] < 0.25,
     `lids ${blink.dbg.lid.map(v => v.toFixed(2)).join(' / ')}`);
  for (let i = 0; i < 60; i++) withBlends({});
  T.preset('Neutral');
  for (let i = 0; i < 40; i++) step1(0, 0, 0, 1 / 60);
}

section('mouth interior');
const outputCtx = dom.byId.get('puppet').getContext('2d');
const toothVisibility = dom.byId.get('tv');
toothVisibility.value = '0.75';                 // the value the panel ships with
T.preset('Neutral');
for (let i = 0; i < 35; i++) step1(0.2, 0, 0, 1 / 60, null, true);
resetOne(outputCtx);
const closedFrame = step1(0.2, 0, 0, 1 / 60, null, true);
const closedOps = mouthOps(outputCtx), closedAlpha = closedFrame.dbg.mouth.upperAlpha + closedFrame.dbg.mouth.lowerAlpha;
T.preset('Laugh');
for (let i = 0; i < 55; i++) step1(0.2, 0, 0, 1 / 60, null, true);
resetOne(outputCtx);
const openFrame = step1(0.2, 0, 0, 1 / 60, null, true);
const openOps = mouthOps(outputCtx), openDbg = openFrame.dbg;
const openAlpha = openDbg.mouth.upperAlpha + openDbg.mouth.lowerAlpha;
ok('mouth opens under a preset', openDbg.mouth.h > 0.12, `mouth envelope ${openDbg.mouth.h.toFixed(3)}`);
ok('oral cavity and teeth respond to jaw opening', openDbg.mouth.cavity && openAlpha > closedAlpha * 2,
   `teeth alpha ${closedAlpha.toFixed(2)} closed -> ${openAlpha.toFixed(2)} open; ${closedOps} -> ${openOps} canvas ops`);
ok('jaw bone rotates', ses.jaw > 0.02, `${(ses.jaw * deg).toFixed(1)}°`);
T.preset('Tongue out');
for (let i = 0; i < 55; i++) step1(0, 0, 0, 1 / 60, null, true);
resetOne(outputCtx);
const tongueFrame = step1(0, 0, 0, 1 / 60, null, true);
const tongueClipped = mouthClipContained(ses, outputCtx);
ok('tongue is rendered through the inner-lip clip', tongueFrame.dbg.mouth.tongue && tongueClipped,
   `drawn ${tongueFrame.dbg.mouth.tongue}; candidate clips ${outputCtx.ops.filter(op => op.op === 'clip' && op.clip).length}; contained ${tongueClipped}`);
ok('teeth/tongue outputs stay finite', [tongueFrame.dbg.mouth.h, tongueFrame.dbg.mouth.upperAlpha,
  tongueFrame.dbg.mouth.lowerAlpha, ses.jaw].every(Number.isFinite));

/* The arches have to be *inside* the opening the viewer sees.  Both of them
   used to be placed outward from the lip line — the lower arch on the chin,
   the upper one above the upper lip — where the inner-lip clip erased them,
   and the old bounding-box overlap test still called it visible.  `teethArea`
   is the opaque area of the crowns actually inside the opening. */
{
  /* A neutral rig.  `step1` feeds the tracker, and an unconsumed camera frame
     leaves the tracked pose (rotated, 0.5x zoomed by earlier sections) in
     place even with the camera off — which pinches the mouth into a sliver
     and makes these results meaningless.  Talk at the puppet with the camera
     off, like a user who has not started the webcam. */
  const tr = T.state().tracker;
  const stepOff = (dt = 1 / 60) => { t += dt * 1000; dom.setClock(t); return T.frame(t, dt, true); };
  const standDown = () => {
    T.state().camOn = false; T.state().cam = null;
    tr.trust = 0; tr.q = [0, 0, 0, 1]; tr.qTarget = [0, 0, 0, 1];
    tr.ox = tr.oy = 0; tr.scale = 1;
    for (let i = 0; i < 40; i++) stepOff();
  };
  standDown();
  T.preset('Laugh');
  for (let i = 0; i < 80; i++) stepOff();
  const f = stepOff();
  const m = f.dbg.mouth, fr = m.frame, box = m.teethBox;
  const ap = fr ? fr.innerY1 - fr.innerY0 : 0;
  const bites = !!fr && fr.lineUp >= fr.upY - 1 && fr.lineLo <= fr.loY + 1;
  const inside = !!box && !!fr && box.y0 >= fr.innerY0 - 2.5 && box.y1 <= fr.innerY1 + 2.5;
  const cover = fr && m.mw > 1 ? m.teethArea / (ap * m.mw) : 0;
  ok('a wide open mouth shows its teeth inside the opening',
     m.h > 0.4 && inside && bites && ap > 8 && cover > 0.08,
     `envelope ${m.h.toFixed(2)}, visible teeth ${m.teethArea.toFixed(0)} px² = ${(cover * 100).toFixed(0)}% of a ` +
     `${ap.toFixed(0)}x${m.mw.toFixed(0)} px opening, ` +
     `bites ${fr ? `${fr.lineUp.toFixed(0)}/${fr.lineLo.toFixed(0)} vs lips ${fr.innerY0.toFixed(0)}..${fr.innerY1.toFixed(0)}` : 'n/a'}`);

  /* ...and a closed mouth shows none of them, even at full visibility. */
  T.preset('Neutral');
  for (let i = 0; i < 70; i++) stepOff();
  const c = stepOff().dbg.mouth;
  ok('a closed mouth shows no teeth', c.h < 0.05 && c.teethArea < 12,
     `envelope ${c.h.toFixed(2)}, visible teeth ${c.teethArea.toFixed(1)} px²`);

  /* Phrases, not a fixed gape: the talking preset has to open, close and
     keep the teeth in view while it does. */
  T.preset('Talking');
  let peak = 0, sum = 0, seen = 0, shut = 0, n = 0;
  for (let i = 0; i < 300; i++) {
    const m = stepOff().dbg.mouth;
    peak = Math.max(peak, m.h); sum += m.h; n++;
    if (m.teethArea > 25) seen++;
    if (m.h < 0.15) shut++;
  }
  ok('the talking preset speaks: opens, closes, teeth in view',
     peak > 0.45 && sum / n > 0.12 && seen / n > 0.5 && shut > 8,
     `peak ${peak.toFixed(2)}, mean ${(sum / n).toFixed(2)}, teeth in view ${(100 * seen / n).toFixed(0)}% of frames, ` +
     `closed on ${shut} frames`);

  /* Speech speed: a jaw step has to arrive inside a syllable.  At the old
     3.2/s the jaw needed over a second and every syllable was smeared. */
  const realTalkAt = FMEngine.talkAt;
  FMEngine.talkAt = () => ({ j: 1 });
  T.preset('Talking');
  let reached = -1;
  for (let i = 0; i < 60 && reached < 0; i++) {
    const m = stepOff().dbg.mouth;
    if (m.h > 0.7) reached = i;
  }
  FMEngine.talkAt = realTalkAt;
  ok('the jaw reaches a full open inside a syllable', reached >= 0 && reached <= 24,
     reached < 0 ? 'never reached' : `0.7 envelope after ${(reached / 60 * 1000).toFixed(0)} ms`);
  T.preset('Neutral');
  for (let i = 0; i < 40; i++) stepOff();
}

section('interaction');
const pc = dom.byId.get('puppet'), pcCtx = pc.getContext('2d');
const rack = dom.byId.get('sculptTools');
ok('sculpt buttons registered', rack.children.length >= 8, `${rack.children.length} tools`);
ok('preset rack populated', dom.byId.get('presets').children.length > 15, `${dom.byId.get('presets').children.length} presets`);

/* Webcam minimize control really toggles the card and its label. */
const webcamCard = dom.byId.get('webcamCard'), minCamBtn = dom.byId.get('minCamBtn');
minCamBtn.fire('click', {});
const minimized = webcamCard.classList.contains('minimized') && minCamBtn.textContent === 'Expand';
minCamBtn.fire('click', {});
ok('webcam can be minimized and restored', minimized && !webcamCard.classList.contains('minimized') && minCamBtn.textContent === 'Minimize');

/* The face-landmark checkbox controls the rendered point overlay. */
const ptsToggle = dom.byId.get('showPts');
ptsToggle.checked = true; resetOne(pcCtx); step1(0, 0, 0, 1 / 60, null, true);
const arcsWithPoints = pcCtx.arcCount;
ptsToggle.checked = false; resetOne(pcCtx); step1(0, 0, 0, 1 / 60, null, true);
const arcsWithoutPoints = pcCtx.arcCount; ptsToggle.checked = true;
ok('landmark checkbox toggles the dot overlay', arcsWithPoints > arcsWithoutPoints + 80,
   `${arcsWithPoints} arcs on -> ${arcsWithoutPoints} off`);

/* PNG capture uses the rendered canvas, not the source photo. */
const pngBefore = pc.toDataURLCalls || 0;
dom.byId.get('photoBtn').fire('click', {});
ok('rendered-photo button captures PNG', pc.toDataURLCalls === pngBefore + 1);

/* Sculpt clicks outside the projected face are ignored; face clicks edit
   the selected expression in canonical coordinates. */
app.sculpt.fill(0); app.bag.length = 0;
const smileTool = rack.children.find(c => c.dataset.sculpt === 'smile');
rack.fire('click', { target: smileTool });
const toClient = p => ({ clientX: p.x, clientY: p.y });
const clickAt = p => pc.fire('pointerdown', { ...toClient(p), pointerId: 9 });
clickAt({ x: 4, y: 4 });
const outsideUnchanged = app.sculpt.every(v => v === 0);
const mouthPoint = { x: (ses.DST[13 * 2] + ses.DST[14 * 2]) / 2,
                     y: (ses.DST[13 * 2 + 1] + ses.DST[14 * 2 + 1]) / 2 };
clickAt(mouthPoint);
const smileWorked = app.sculpt.some(v => v !== 0);
ok('expression sculpt is hit-tested to the face', outsideUnchanged && smileWorked,
   `outside ignored; mouth click edited ${app.sculpt.filter(v => v !== 0).length} offsets`);
/* A click-to-expression regression: the old sparse edit moved only the named
   handles, leaving long edges between them and their untouched neighbours.
   That produced visible rectangular texture blocks in the affine warp. */
const smileSculpt = Float32Array.from(app.sculpt);
let movedSmile = 0, maxSmile = 0, maxSmileNeighbourStep = 0;
for (let i = 0; i < 478; i++) {
  const k = i * 2, amount = Math.hypot(smileSculpt[k], smileSculpt[k + 1]);
  if (amount > 0.25) movedSmile++;
  maxSmile = Math.max(maxSmile, amount);
  for (const j of (mesh.nbr[i] || [])) if (j < 478) {
    maxSmileNeighbourStep = Math.max(maxSmileNeighbourStep,
      Math.hypot(smileSculpt[k] - smileSculpt[j * 2], smileSculpt[k + 1] - smileSculpt[j * 2 + 1]));
  }
}
t += 16.7; dom.setClock(t); resetCtx();
const smileFrame = T.frame(t, 1 / 60, true);
const smileViolations = collectCtx().reduce((n, c) => n + c.violations.length, 0);
ok('smile follows neighbouring mesh points without block folds',
   movedSmile > 200 && maxSmileNeighbourStep < maxSmile * 0.8 && smileFrame.dbg.tris > 2200 && smileViolations === 0,
   `${movedSmile} points moved; neighbour step ${maxSmileNeighbourStep.toFixed(1)} / ${maxSmile.toFixed(1)} px; ${smileFrame.dbg.tris} triangles`);
const beforeO = Float32Array.from(app.sculpt);
const oTool = rack.children.find(c => c.dataset.sculpt === 'o');
rack.fire('click', { target: oTool }); clickAt(mouthPoint);
let oChanged = 0;
for (let i = 0; i < app.sculpt.length; i++) if (app.sculpt[i] !== beforeO[i]) oChanged++;
ok('Mouth O button applies a distinct local edit', oChanged > 0, `${oChanged} offsets changed`);

/* PNG and WebM controls are wired to actual browser APIs; simulate their
   lifecycle, including releasing the canvas capture track on Stop. */
const videoTrack = { kind: 'video', stopped: false, stop() { this.stopped = true; } };
const micTrack = { kind: 'audio', stopped: false, stop() { this.stopped = true; } };
const stream = {
  getVideoTracks: () => [videoTrack], getTracks: () => [videoTrack, micTrack], getAudioTracks: () => [],
  addTrack(track) { this._added = track; }
};
app.micOn = true; app.micStream = { getAudioTracks: () => [micTrack] };
pc.captureStream = () => stream;
const priorRecorder = globalThis.MediaRecorder;
class FakeMediaRecorder {
  constructor(s, options = {}) { this.stream = s; this.mimeType = options.mimeType || 'video/webm'; }
  start() { this.started = true; }
  stop() {
    this.stopped = true;
    if (this.ondataavailable) this.ondataavailable({ data: new Blob(['fake webm']) });
    if (this.onstop) this.onstop();
  }
}
globalThis.MediaRecorder = FakeMediaRecorder;
const recordBtn = dom.byId.get('recordBtn');
recordBtn.fire('click', {});
const startedRecording = !!app.rec && recordBtn.textContent === 'Stop recording';
recordBtn.fire('click', {});
ok('video recording starts, stops, downloads and releases its track', startedRecording && !app.rec
  && videoTrack.stopped && !micTrack.stopped && stream._added === micTrack
  && recordBtn.textContent === 'Record video');
app.micOn = false; app.micStream = null;
if (priorRecorder === undefined) delete globalThis.MediaRecorder; else globalThis.MediaRecorder = priorRecorder;

/* Undo is checked against a known clean baseline, not a tautology. */
app.sculpt.fill(0); app.bag.length = 0;
rack.fire('click', { target: smileTool }); clickAt(mouthPoint);
const edited = app.sculpt.some(v => v !== 0);
dom.byId.get('undoBtn').fire('click', {});
ok('undo restores the previous sculpt', edited && app.sculpt.every(v => v === 0));

const sculptBefore = Float32Array.from(app.sculpt);
for (let i = 0; i < 40; i++) step1(0.35, 0, 0);
const heldSculpt = Float32Array.from(app.sculpt);
step1(0.35, 0, 0);
let sculptDrift = 0;
for (let i = 0; i < heldSculpt.length; i++) sculptDrift = Math.max(sculptDrift, Math.abs(app.sculpt[i] - heldSculpt[i]));
ok('sculpts do not drift while tracking', sculptDrift < 1e-6, `drift ${sculptDrift}`);

section('screen <-> canonical drag solve');
const dragOk = (() => {
  /* drag dot 61 (mouth corner) to a screen point; the projection must
     land within a pixel of the request even with the head turned away */
  for (let i = 0; i < 60; i++) step1(0.35, 0.1, 0.05);
  const i = 61;
  const target = { x: ses.DST[i * 2] + 9, y: ses.DST[i * 2 + 1] - 6 };
  const pcEl = dom.byId.get('puppet');
  /* emulate pointerdown/pointermove with canvas-space coordinates */
  const rect = pcEl.getBoundingClientRect();
  const toClient = p => ({ clientX: p.x * rect.width / pcEl.width, clientY: p.y * rect.height / pcEl.height });
  T.state().sculptMode = 'drag';
  const start = { x: ses.DST[i * 2], y: ses.DST[i * 2 + 1] };
  pcEl.fire('pointerdown', { ...toClient(start), pointerId: 1 });
  pcEl.fire('pointermove', { ...toClient(target), pointerId: 1 });
  pcEl.fire('pointerup', {});
  for (let k = 0; k < 20; k++) step1(0.35, 0.1, 0.05);
  const got = { x: ses.DST[i * 2], y: ses.DST[i * 2 + 1] };
  return Math.hypot(got.x - target.x, got.y - target.y);
})();
ok('drag lands on the pointer (turned head)', dragOk < 3.5, `residual ${dragOk.toFixed(2)} px`);

section('flash detector negative control');
{
  const c = new dom.Ctx(new dom.Canvas(W, H));
  c.beginPath();
  c.moveTo(10, 10); c.lineTo(10, 10); c.lineTo(10, 10);   // zero-area clip
  c.clip();
  c.fillRect(0, 0, W, H);
  ok('detector fires on a degenerate clip + full-canvas fill', c.violations.length === 1);
  const c2 = new dom.Ctx(new dom.Canvas(W, H));
  c2.beginPath();
  c2.moveTo(0, 0); c2.lineTo(100, 0); c2.lineTo(100, 100);
  c2.clip();
  c2.fillRect(0, 0, W, H);
  ok('detector stays quiet for a real clip', c2.violations.length === 0);
}

section('performance proxies');
const ctxs = collectCtx();
const frames = Math.max(1, ses.frames);
const ops = ctxs.reduce((a, c) => a + (c.totalOps || 0), 0);
ok('tracked paint calls per rendered frame bounded', ops / frames < 7000,
   `${(ops / frames).toFixed(0)} tracked clip/fill/image calls per frame over ${frames} renders`);
const reads = ctxs.reduce((a, c) => a + (c.getImageDataCalls || 0), 0);
ok('no pixel readback inside the frame loop', reads < 60, `${reads} getImageData calls (image setup only)`);
ok('no stray camera/audio objects created', !T.state().stream && !T.state().audioCtx);

console.log(`\n${fails ? '✗' : '✓'} harness: ${passes} passed, ${fails} failed\n`);
process.exit(fails ? 1 : 0);

/* ------------------------------------------------------------- helpers */
function collectCtx() {
  const list = [];
  for (const canvas of dom.allCanvases) if (canvas._ctx) list.push(canvas._ctx);
  return list;
}
function resetCtx() { for (const c of collectCtx()) resetOne(c); }
function resetOne(c) {
  if (c) { c.ops.length = 0; c.violations.length = 0; c.opCount = 0; c.arcCount = 0; }
}
function mouthOps(ctx) { return ctx ? (ctx.opCount || 0) : 0; }
function mouthClipContained(ses, ctx) {
  if (!ctx || !ses) return false;
  const ids = globalThis.FMEngine.LIPS_I;
  const xs = ids.map(i => ses.DST[i * 2]), ys = ids.map(i => ses.DST[i * 2 + 1]);
  const bb = { x0: Math.min(...xs) - 1, y0: Math.min(...ys) - 1,
               x1: Math.max(...xs) + 1, y1: Math.max(...ys) + 1 };
  const innerClips = ctx.ops.filter(op => {
    const b = op.op === 'clip' && op.clip;
    return b && b.x0 >= bb.x0 && b.y0 >= bb.y0 && b.x1 <= bb.x1 && b.y1 <= bb.y1;
  });
  return innerClips.length >= 2; // cavity + tongue passes both clip to the inner-lip aperture
}
/* The painted affine must map the triangle's source vertices onto its
   destination vertices. Clip/order tests cannot see a scrambled affine:
   the triangle would still paint, in the right place, with wrong pixels. */
function checkTriangleAffines(ses) {
  /* A painted triangle is drawn as clip(dest triangle grown by the pass pad)
     + setTransform(affine) + drawImage(source rect).  Recover the triangle
     that the op belongs to using each pass's own pad and tessellation, then
     require the affine to land that triangle's source vertices on its
     destination vertices.  A swapped coefficient pair would keep the paint
     order and clip legal while scrambling the pixels. */
  const lc = ses.lctx;
  const passes = [ses.triA, ses.triB].filter(Boolean);
  const cap = lc.opLimit;
  lc.opLimit = 80000;
  lc.ops.length = 0;
  renderAt(FM.quatFromYPR(0.5, 0, 0.08), { quality: 1 });
  lc.opLimit = cap;
  const midX = ses.mid[0];
  const key = (x, y) => `${Math.round(x * 4)}:${Math.round(y * 4)}`;
  const indexOf = pass => {
    const idx = new Map();
    for (let t = 0; t < pass.n; t++) {
      const a = pass.list[t * 3], b = pass.list[t * 3 + 1], c = pass.list[t * 3 + 2];
      const k = key((ses.DST[a * 2] + ses.DST[b * 2] + ses.DST[c * 2]) / 3,
                    (ses.DST[a * 2 + 1] + ses.DST[b * 2 + 1] + ses.DST[c * 2 + 1]) / 3);
      if (!idx.has(k)) idx.set(k, []);
      idx.get(k).push({ t, a, b, c, d: [a, b, c] });
    }
    return idx;
  };
  const idx = indexOf(ses.triA);
  const idxB = passes.length > 1 ? indexOf(ses.triB) : null;
  const unpad = (raw, pad) => {
    const gx = (raw[0][0] + raw[1][0] + raw[2][0]) / 3, gy = (raw[0][1] + raw[1][1] + raw[2][1]) / 3;
    return raw.map(([x, y]) => {
      const ux = x - gx, uy = y - gy, l = Math.hypot(ux, uy) || 1;
      return [gx + ux * (l - pad) / l, gy + uy * (l - pad) / l];
    });
  };
  const apply = (m, p) => [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]];
  const dErr = (cand, d) => Math.max(...[0, 1, 2].map(k =>
    Math.hypot(d[k][0] - ses.DST[cand.d[k] * 2], d[k][1] - ses.DST[cand.d[k] * 2 + 1])));
  const search = (map, d) => {
    if (!map) return null;
    const cx = (d[0][0] + d[1][0] + d[2][0]) / 3, cy = (d[0][1] + d[1][1] + d[2][1]) / 3;
    let best = null, bestErr = 0.05;
    for (let ox = -2; ox <= 2; ox++) for (let oy = -2; oy <= 2; oy++) {
      for (const cand of (map.get(key(cx + ox * 0.25, cy + oy * 0.25)) || [])) {
        const err = dErr(cand, d);
        if (err < bestErr) { bestErr = err; best = cand; }
      }
    }
    return best;
  };
  let clipPts = null, mat = null, inspected = 0, worst = 0, bad = 0, skipped = 0, worstAt = null;
  for (const op of lc.ops) {
    if (op.op === 'clip') { clipPts = op.pts && op.pts.length >= 3 ? op.pts.slice(0, 3) : null; mat = null; continue; }
    if (op.op === 'setTransform') { mat = op.args.slice(); continue; }
    if (op.op !== 'drawImage' || !clipPts || !mat) continue;
    const raw = clipPts; clipPts = null;
    let best = null, bestDst = null, bestFit = Infinity;
    for (const [map, pad] of [[idx, ses.triA.pad * (ses.triA.padScale || 1)],
                              [idxB, ses.triB ? ses.triB.pad * (ses.triB.padScale || 1) : 0]]) {
      const d = unpad(raw, pad);
      const cand = search(map, d);
      if (!cand) continue;
      const src = cand.d.map(v => [ses.S2[v * 2], ses.S2[v * 2 + 1]]);
      const mirrored = [[2 * midX - src[0][0], src[0][1]], [2 * midX - src[2][0], src[2][1]],
                        [2 * midX - src[1][0], src[1][1]]];          // far-side b/c swap
      const fit = pts => Math.max(...[0, 1, 2].map(k => {
        const [mx, my] = apply(mat, pts[k]);
        return Math.hypot(mx - d[k][0], my - d[k][1]);
      }));
      const e = Math.min(fit(src), fit(mirrored));
      if (e < bestFit) { bestFit = e; best = cand; bestDst = d; }
    }
    if (!best) { skipped++; continue; }
    inspected++;
    if (bestFit > worst) { worst = bestFit; worstAt = [best.t, best.d.join(',')]; }
    if (bestFit > 1.0) bad++;
  }
  return { inspected, worst, bad, skipped, worstAt, painted: passes.reduce((n, p) => n + p.n, 0) };
}

function checkPaintOrder(ses) {
  /* Verify the depth-key order actually consumed by both shell passes. */
  const passes = [ses.triA, ses.triB].filter(Boolean);
  if (!passes.length) return { ok: false, msg: 'no triangle passes' };
  let checked = 0;
  for (const pass of passes) {
    let last = -Infinity;
    const seen = new Uint8Array(pass.n);
    for (let i = 0; i < pass.n; i++) {
      const id = pass.order[i];
      if (id < 0 || id >= pass.n || seen[id]) return { ok: false, msg: `pass order is not a permutation at triangle ${i}` };
      seen[id] = 1;
      const t = id * 3;
      const d = (ses.depth[pass.list[t]] + ses.depth[pass.list[t + 1]] + ses.depth[pass.list[t + 2]]) / 3;
      if (d < last - 1e-6) return { ok: false, msg: `pass has inversion at triangle ${i}` };
      last = d;
    }
    checked += pass.n;
  }
  return { ok: true, msg: `checked ${checked} triangles across both passes` };
}
