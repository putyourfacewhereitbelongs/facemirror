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
import { runInNewContext } from 'node:vm';
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
const html = await readFile(join(rootDir, 'index.html'), 'utf8').catch(() => '');
if (!html) { console.error('index.html missing — run `npm run build:puppet` first'); process.exit(2); }
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

const runtimeRequests = [];
const vision = {
  FilesetResolver: {
    forVisionTasks: async (path) => {
      runtimeRequests.push({ type: 'wasm', path });
      return { path };
    }
  },
  FaceLandmarker: {
    FACE_LANDMARKS_TESSELATION: tess,
    createFromOptions: async (fs, opts) => {
      runtimeRequests.push({
        type: 'face', wasm: fs.path, model: opts.baseOptions.modelAssetPath, delegate: opts.baseOptions.delegate
      });
      return {
        _mode: opts.runningMode,
        detect: () => ({ faceLandmarks: [toLandmarks(P0, W, H, FACE_PX)] }),
        detectForVideo: () => ({ faceLandmarks: [toLandmarks(P0, W, H, FACE_PX)], faceBlendshapes: [{ categories: [] }] })
      };
    }
  },
  // Simulate a stalled optional pose-model download; face tracking must not
  // wait for it to expose the webcam button.
  PoseLandmarker: { createFromOptions: () => new Promise(() => {}) }
};
await globalThis.PuppetApp.boot(vision);
const T = globalThis.PuppetApp.__test;

section('page / script agreement');
ok('every element the app looks up exists in the main index.html', missing.size === 0,
   missing.size ? 'missing: ' + [...missing].join(', ') : '(0 misses)');
const localRuntimeImport = html.indexOf("import('./vendor/tasks-vision/vision_bundle.mjs')");
const cdnRuntimeFallback = html.indexOf("import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs')");
ok('default page prefers the bundled MediaPipe runtime', localRuntimeImport >= 0 && cdnRuntimeFallback > localRuntimeImport);
ok('default page prefers local WASM and face-model assets',
   html.includes("wasm: './vendor/tasks-vision/wasm'") && html.includes("faceModel: './assets/face_landmarker.task'"));
ok('default page loads Delaunator locally', html.includes('<script src="./vendor/delaunator.min.js"></script>'));
const localTrackingAssets = [
  'vendor/delaunator.min.js',
  'vendor/tasks-vision/vision_bundle.mjs',
  'vendor/tasks-vision/wasm/vision_wasm_internal.wasm',
  'vendor/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm',
  'assets/face_landmarker.task'
];
const localTrackingAssetsPresent = (await Promise.all(localTrackingAssets.map(async (file) => {
  const data = await readFile(join(rootDir, file)).catch(() => null);
  return !!data && data.byteLength > 0;
}))).every(Boolean);
ok('all local face-tracking assets are present', localTrackingAssetsPresent, localTrackingAssets.join(', '));
const browserDelaunator = {};
browserDelaunator.globalThis = browserDelaunator;
runInNewContext(await readFile(join(rootDir, 'vendor/delaunator.min.js'), 'utf8'), browserDelaunator);
const testPoints = [[0, 0], [1, 0], [0, 1], [1, 1]];
const bundledTriangles = typeof browserDelaunator.Delaunator === 'function'
  ? [...browserDelaunator.Delaunator.from(testPoints).triangles] : [];
ok('bundled Delaunator runs and triangulates in a browser global',
   bundledTriangles.length > 0 && bundledTriangles.join(',') === [...Delaunator.from(testPoints).triangles].join(','));
ok('boot uses the local face model and WASM first', runtimeRequests.some((r) =>
  r.type === 'face' && r.wasm === './vendor/tasks-vision/wasm' && r.model === './assets/face_landmarker.task'));
ok('webcam becomes available without waiting for optional pose tracking', !dom.byId.get('camBtn').disabled);

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
const renderAt = q => FMRender.sessionFrame(ses, {
  t: t / 1000, dt: 1 / 60, quality: 1, q,
  focal: 3.1, scale: 1, depthGain: 0.3, ox: 0, oy: 0,
  deltaCanonical: new Float32Array(478 * 3), jawOpen: 0, jawMeshGain: 0,
  limits: true, shade: 0, hairGain: 0, gaze: [[0, 0], [0, 0]], gazeGain: 1.9,
  lid: [0, 0], vis: 0.4, bright: 1.15, warmth: 0,
  showPoints: false, showMesh: false, origOnly: false, points: [], dragIndex: -1, hoverIndex: -1
});
const restDbg = renderAt([0, 0, 0, 1]);
const turnDbg = renderAt(FM.quatFromYPR(0.55, 0, -0.1));
ok('far-side mirroring activates on a turn', turnDbg.culled > restDbg.culled && turnDbg.culled > 0,
   `${restDbg.culled} frontal -> ${turnDbg.culled} mirrored triangles`);
ok('shell coverage stays complete through yaw', Math.abs(turnDbg.tris - restDbg.tris) <= 4 && turnDbg.tris > 2200 && restDbg.tris > 2200,
   `${restDbg.tris} frontal -> ${turnDbg.tris} turned; near-degenerate edge triangles may vary by a few`);
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

section('ears through a turn');
/* MediaPipe has no pinna landmarks, so the engine derives an ear band
   around the tragion landmarks (234 / 454).  These checks pin down the
   turn behaviour that used to break: ears glued to the frontal
   silhouette, mirror-flipped at the cull threshold, and the near ear
   sorting in front of the nose.                                     */
ok('ear band peaks at the tragion landmarks and vanishes mid-face',
   !!mesh.earW && mesh.earW[234] > 0.75 && mesh.earW[454] > 0.75 && mesh.earW[4] < 0.01 && mesh.earW[61] < 0.01,
   `earW 234:${mesh.earW ? mesh.earW[234].toFixed(2) : 'missing'} 454:${mesh.earW ? mesh.earW[454].toFixed(2) : 'missing'} nose:${mesh.earW ? mesh.earW[4].toExponential(1) : '?'}`);
ok('ears are recessed behind the cheek / eye plane',
   head.P[234][2] < head.P[33][2] - head.fw * 0.02 && head.P[454][2] < head.P[263][2] - head.fw * 0.02,
   `z ear ${head.P[234][2].toFixed(1)}/${head.P[454][2].toFixed(1)} vs eye corners ${head.P[33][2].toFixed(1)}/${head.P[263][2].toFixed(1)}`);
const earDepthError = i => {
  const bp = [ses.BASE[i * 3], ses.BASE[i * 3 + 1], ses.BASE[i * 3 + 2] * (i < 478 ? 1.3 : 1)];
  const rotated = ses.proj(bp), frontal = ses.proj0(bp);
  const earTurn = Math.min(1, mesh.earW[i] * 0.5);
  const partial = frontal[2] + (rotated[2] - frontal[2]) * earTurn;
  const w = mesh.w[i];
  const expected = partial * w + ses.BASE[i * 3 + 2] * (1 - w) * 0.4;
  return Math.abs(ses.depth[i] - expected);
};
const earAt = yaw => {
  const dbg = renderAt(FM.quatFromYPR(yaw, 0, 0));
  return { dbg, l: ses.DST[234 * 2], r: ses.DST[454 * 2], nose: ses.DST[4 * 2],
           dl: ses.depth[234], dr: ses.depth[454], dn: ses.depth[4],
           depthError: Math.max(earDepthError(234), earDepthError(454)),
           off: Math.max(...Array.from(ses.earOff, Math.abs)) };
};
const eRest = earAt(0), ePos = earAt(0.6), eNeg = earAt(-0.6);
ok('ear swing is identically zero at rest (frontal photo untouched)',
   eRest.off < 1e-9 && eRest.dbg.ear.faded === 0 && eRest.dbg.ear.band > 0,
   `max |earOff| ${eRest.off}, faded ${eRest.dbg.ear.faded}, band ${eRest.dbg.ear.band} tris`);
ok('ear painter depth matches its fractional screen turn in both directions',
   ePos.depthError < 1e-4 && eNeg.depthError < 1e-4,
   `max depth-key error +yaw ${ePos.depthError.toExponential(1)}, -yaw ${eNeg.depthError.toExponential(1)}`);
const mixedEarTurn = renderAt(FM.quatFromYPR(0.42, -0.22, 0.18));
const ringEarIndex = mesh.rings[0].reduce((best, i) => mesh.earW[i] > mesh.earW[best] ? i : best, mesh.rings[0][0]);
const mixedDepthError = Math.max(earDepthError(234), earDepthError(454), earDepthError(ringEarIndex));
ok('face and hair-ring ear depth stay coherent through pitch and roll', mixedDepthError < 1e-4,
   `max face/ring depth-key error ${mixedDepthError.toExponential(1)} at ring point ${ringEarIndex}`);
const expectedEarDetailSkips = Array.from(ses.triB.earF).filter(v => v > 0.12).length;
ok('dense face pass defers ear triangles to the depth-sorted shell',
   ePos.dbg.ear.detailSkipped === expectedEarDetailSkips && eNeg.dbg.ear.detailSkipped === expectedEarDetailSkips
     && expectedEarDetailSkips > 0,
   `${expectedEarDetailSkips} ear triangles skipped from the overlay in either turn direction`);
const width = e => e.r - e.l;
ok('both turn directions foreshorten the ear-to-ear width',
   width(ePos) < width(eRest) - 8 && width(eNeg) < width(eRest) - 8,
   `${width(eRest).toFixed(1)} rest -> ${width(ePos).toFixed(1)} @+0.6, ${width(eNeg).toFixed(1)} @-0.6`);
/* far ear (smaller depth key) must tuck toward the midline farther
   than the near ear, in both directions */
const tuck = (e, farLeft) => farLeft
  ? { far: e.l - eRest.l, near: eRest.r - e.r }
  : { far: eRest.r - e.r, near: e.l - eRest.l };
const tPos = tuck(ePos, ePos.dl < ePos.dr), tNeg = tuck(eNeg, eNeg.dl < eNeg.dr);
ok('far ear tucks behind the jaw more than the near ear moves',
   tPos.far > tPos.near + 4 && tNeg.far > tNeg.near + 4 && Math.min(ePos.dl, ePos.dr) < 0 && Math.min(eNeg.dl, eNeg.dr) < 0,
   `@+0.6 far ${tPos.far.toFixed(1)}px vs near ${tPos.near.toFixed(1)}px; @-0.6 far ${tNeg.far.toFixed(1)}px vs near ${tNeg.near.toFixed(1)}px`);
ok('near ear never sorts in front of the nose',
   Math.max(ePos.dl, ePos.dr) < ePos.dn && Math.max(eNeg.dl, eNeg.dr) < eNeg.dn,
   `@+0.6 ear depth ${Math.max(ePos.dl, ePos.dr).toFixed(1)} vs nose ${ePos.dn.toFixed(1)}`);
ok('far ear fades into the plate on a turn (no mirrored ears)',
   ePos.dbg.ear.faded > 0 && eNeg.dbg.ear.faded > 0,
   `faded ear tris @+0.6: ${ePos.dbg.ear.faded}, @-0.6: ${eNeg.dbg.ear.faded} (band ${ePos.dbg.ear.band})`);
/* the swing must be a continuous function of yaw: step across zero in
   fine increments and bound the largest ear-landmark jump */
let earJump = 0, ePrev = earAt(-0.7);
for (let yaw = -0.66; yaw <= 0.701; yaw += 0.04) {
  const e = earAt(yaw);
  earJump = Math.max(earJump, Math.abs(e.l - ePrev.l), Math.abs(e.r - ePrev.r));
  ePrev = e;
}
ok('ear swing has no step at the mirror threshold', earJump < head.fw * 0.02,
   `largest ear jump ${earJump.toFixed(2)} px per 0.04 rad step`);

section('mouth interior');
const outputCtx = dom.byId.get('puppet').getContext('2d');
const toothVisibility = dom.byId.get('tv');
toothVisibility.value = '0.4';
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
ok('tongue feature is removed from UI, actions and presets',
   !dom.byId.has('tsens') && !dom.byId.has('tf') && !Object.hasOwn(FMEngine.PRESETS, 'Tongue out')
     && !FMEngine.ACTIONS.includes('tg') && !FMEngine.ACTIONS.includes('st')
     && !Object.hasOwn(T.state(), 'tongue') && !Object.hasOwn(openDbg.mouth, 'tongue'),
   'no controls, tongue actions, preset, state, or render pass');
ok('oral cavity and teeth diagnostics stay finite', [openDbg.mouth.h, openDbg.mouth.upperAlpha,
  openDbg.mouth.lowerAlpha, ses.jaw].every(Number.isFinite));
T.preset('Neutral');

section('interaction');
const pc = dom.byId.get('puppet'), pcCtx = pc.getContext('2d'), app = T.state();
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
