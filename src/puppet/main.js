/**
 * Trill Face Puppet - runtime orchestration.
 *
 * Pipeline (one pass per animation frame):
 *
 *   webcam ----> MediaPipe FaceLandmarker (478 3D) + PoseLandmarker
 *      |                        |
 *      |                 One-Euro field filter (jitter out, lag out)
 *      v                        v
 *   head pose fit ----------> Euler retarget (per-axis gains, soft limits,
 *      |                      rate limiting, dt-normalised smoothing)
 *      v
 *   expression field (pose removed, stitched, retargeted to the photo's
 *      |                eye / mouth / nose scale)
 *      v
 *   jaw hinge (mandible as a bone) + preset / sculpt / audio parameters
 *      v
 *   canonical head-space displacement -> rigid rotation -> pinhole projection
 *      |            (model.js built canonical space so rest == identity)
 *      v
 *   stretch relaxation (PBD edge caps) + far-to-near depth sort
 *      v
 *   render: inpainted background plate -> affine triangle warp (flip-free)
 *           -> seam blend + contact AO -> differential relighting
 *           -> oral cavity / tongue / rigid dental arches / lips / kiss
 *           -> iris sprites + eyelids -> ROI enhancement -> landmark overlay
 *
 * Every scalar that reaches the renderer passes through a dt-normalised filter
 * or a smooth gate; nothing in the frame path makes a binary decision. That is
 * the anti-flicker contract of this file.
 */

import './puppet.css';
import { loadVision, CDN, MODELS } from './vendor.js';
import { buildRig } from './model.js';
import { createPoseState, buildDragField, driverPose, expressionField, driverJawAngle, stitchRegions, poseFrame, measureStretch } from './pose.js';
import { clamp, lerp, ID3, eulerFromR, RFromEuler, fitSimilarity, softLimit } from './math3d.js';
import { SHOWN, GROUPS, OVAL, EYE, EYM, LIP, LIPSET, NOSE_BRIDGE, NOSE_PTS, IRIS, AU_MAP, RIG } from './landmarks.js';
import { Scalar, Gate, OneEuroField, Confidence, PeakTracker, smoothK } from './temporal.js';
import { rasterizePolygon, dilateMask, inpaintPlate } from './plate.js';
import { fitArch } from './mouth.js';
import { prepIris, gazeRaw } from './eyes.js';
import { drawDriverOverlay } from './render.js';
import { createScratch, pointView, renderFrame } from './frame.js';
import { AudioEngine, bandsToViseme, BANDS } from './audio.js';
import { Recorder, saveCanvasPNG, download } from './capture.js';
import { syntheticFace } from './synthetic.js';

const $ = id => document.getElementById(id);
const num = (id, dflt = 0) => { const el = $(id); return el ? +el.value : dflt; };
const chk = (id, dflt = false) => { const el = $(id); return el ? !!el.checked : dflt; };
const status = t => { const el = $('status'); if (el) el.textContent = t; };

/* ------------------------------------------------------------------ */
/* canvases                                                            */
/* ------------------------------------------------------------------ */
const vid = $('vid');
const camC = $('cam'), cx = camC ? camC.getContext('2d') : null;
const pc = $('puppet'), px = pc.getContext('2d', { alpha: false });
const sc = document.createElement('canvas');          // source photo pixels
const sx = sc.getContext('2d', { willReadFrequently: true });
const plateC = document.createElement('canvas');      // inpainted background plate
const mkOff = () => document.createElement('canvas');
const scratch = createScratch(mkOff);                 // relighting + eyelid + ROI buffers

/* ------------------------------------------------------------------ */
/* runtime state                                                       */
/* ------------------------------------------------------------------ */
const ML = { FaceLandmarker: null, PoseLandmarker: null, FilesetResolver: null };
let modelsReady = false;
let rig = null, st = null, plateImg = null, irisSprites = null, lastSource = null;
let camOn = false, camP = null, camRaw = null, camPts = null, neutral = null, map = null;
let poseNeutral = null, poseP = null, bodyJoints = null;
let bs = {}, lastFaceAt = 0, lastVideoTime = -1, lastMpTs = 0, lastPoseTs = 0, lastPoseFrame = 0;
let seenFace = false, zSign = -1, oneEuro = null;
let dragIdx = -1, hoverIdx = -1, dragOff = { x: 0, y: 0 }, sculptMode = 'drag';
let E = null, Ec = null, UND = [], partners = null;
let preset = 'live';
let lastT = performance.now(), frames = 0, fpsAt = 0, fpsShown = 0;
let recorder = null, photoFlash = 0, minimized = false;
let audio = new AudioEngine(), viseme = null, audioOn = false;
let hair = { x: 0, y: 0, vx: 0, vy: 0 };
let LUM = 0.7, posedOnce = false;
let archCache = null, Dview = null, auRows = null, fxOut = null;
const gaze0 = [[0, 0], [0, 0]];
const mouthMeas = { mw: 1, gap: 0 };
const textures = { enamel: null, noise: null };

/** Animated scalars: each is dt-normalised, so frame drops cannot pop. */
const SM = {
  yaw: new Scalar(0, 0.075, 5.2), pitch: new Scalar(0, 0.09, 4.2), roll: new Scalar(0, 0.11, 3.4),
  tx: new Scalar(0, 0.09), ty: new Scalar(0, 0.09), tz: new Scalar(0, 0.14),
  jaw: new Scalar(0, 0.05), kiss: new Scalar(0, 0.07), tongue: new Scalar(0, 0.09),
  lid: [new Scalar(0, 0.045), new Scalar(0, 0.045)],
  gaze: [[new Scalar(0, 0.05), new Scalar(0, 0.05)], [new Scalar(0, 0.05), new Scalar(0, 0.05)]],
  tTip: new Scalar(0, 0.07), tBack: new Scalar(0, 0.09), tWide: new Scalar(0.5, 0.1), tOut: new Scalar(0, 0.09),
  eyeOpen: [new PeakTracker(0.25), new PeakTracker(0.25)],
  conf: new Confidence(),
};
/** Smooth gates: nothing the renderer switches on ever flips in one frame. */
const GATE = {
  open: new Gate(0.035, 0.13, 0.05, 0.13),
  teeth: new Gate(0.055, 0.2, 0.07, 0.2),
  lower: new Gate(0.075, 0.26, 0.08, 0.22),
  gum: new Gate(0.02, 0.12, 0.09, 0.22),
  tongue: new Gate(0.03, 0.2, 0.07, 0.2),
  tongueOut: new Gate(0.12, 0.5, 0.09, 0.2),
  kiss: new Gate(0.04, 0.32, 0.06, 0.16),
  wet: new Gate(0.02, 0.1, 0.1, 0.25),
  seam: new Gate(0.0, 0.05, 0.12, 0.3),
  ao: new Gate(0.02, 0.14, 0.08, 0.2),
  throat: new Gate(0.09, 0.3, 0.08, 0.2),
};
const PARAMS = ['sL', 'sR', 'j', 'bu', 'buR', 'bi', 'bd', 'eo', 'sq', 'dr', 'pk', 'fr', 'nz', 'ur', 'st', 'bL', 'bR', 'tg', 'tt', 'tb', 'cl', 'gx', 'gy'];
const PP = {}, PT = {};
for (const k of PARAMS) { PP[k] = 0; PT[k] = 0; }

/* ------------------------------------------------------------------ */
/* presets                                                             */
/* ------------------------------------------------------------------ */
export const PRESETS = {
  Neutral: {},
  Smile: { sL: 0.55, sR: 0.55, sq: 0.12 },
  'Big smile': { sL: 0.9, sR: 0.9, j: 0.18, sq: 0.3, bu: 0.15 },
  Laugh: { sL: 1, sR: 1, j: 0.5, sq: 0.55, bu: 0.2, nz: 0.2, tt: 0.15 },
  Smirk: { sR: 0.75, bu: 0.1 },
  Sad: { fr: 0.7, bi: 0.9, dr: 0.3, j: 0.04, gy: 0.8 },
  Angry: { bd: 0.9, fr: 0.45, sq: 0.3, nz: 0.35, ur: 0.15 },
  Surprised: { bu: 0.9, eo: 0.85, j: 0.55 },
  Fear: { bu: 0.55, bi: 0.7, eo: 0.75, j: 0.3, st: 0.7 },
  Disgust: { nz: 0.9, ur: 0.7, bd: 0.35, fr: 0.35, sq: 0.25 },
  Skeptical: { buR: 0.8, bd: 0.25, sR: 0.25, dr: 0.15 },
  Sleepy: { dr: 0.65, j: 0.1 },
  Wink: { bL: 1, sL: 0.45, sR: 0.45, sq: 0.1 },
  Kiss: { pk: 1, bu: 0.1 },
  'Tongue out': { tg: 1, j: 0.35, tt: 0.25 },
  'Say /l/': { tt: 1, j: 0.18 },
  'Say /k/': { tb: 1, j: 0.22 },
  'Say /a/': { j: 0.85, tt: 0.1, st: 0.1 },
  'Say /ee/': { sL: 0.6, sR: 0.6, st: 0.7, j: 0.18 },
  'Say /oo/': { pk: 0.8, j: 0.22 },
  'Look left': { gx: -1 }, 'Look right': { gx: 1 }, 'Look up': { gy: -1 }, 'Look down': { gy: 1 },
  Talking: null,
};
const h1 = i => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return Math.abs(x - Math.floor(x)); };
/** Synthetic syllable generator, used when no audio source is attached. */
function talkParams(t) {
  const env = clamp(Math.sin(t * 0.7) * 3 + 1.2, 0, 1);
  const k = Math.floor(t * 4.3), f = t * 4.3 - k;
  const a = h1(k), b = h1(k * 3.7 + 1);
  const o = Math.pow(Math.sin(Math.PI * f), 0.7) * env;
  return {
    j: o * (0.12 + 0.45 * a),
    pk: b > 0.78 ? o * 0.85 : 0,
    st: b < 0.22 ? o * 0.6 : 0,
    tt: b > 0.42 && b < 0.6 ? o * 0.8 : 0,
    tb: b > 0.6 && b < 0.78 ? o * 0.7 : 0,
    tg: b < 0.08 ? o * 0.5 : 0,
    cl: b > 0.9 ? o * 0.7 : 0,
    sL: 0.12, sR: 0.12, bu: env * 0.12 * b,
  };
}

/* ------------------------------------------------------------------ */
/* model loading                                                       */
/* ------------------------------------------------------------------ */
export async function initModels(onProgress) {
  const say = m => { onProgress && onProgress(m); status(m); };
  try {
    say('Loading the MediaPipe vision runtime…');
    const vision = await loadVision();
    const { FaceLandmarker, PoseLandmarker, FilesetResolver } = vision;
    ML.FaceLandmarker = FaceLandmarker; ML.PoseLandmarker = PoseLandmarker; ML.FilesetResolver = FilesetResolver;
    const fs = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    say('Downloading the 478-point face landmarker weights…');
    const makeFace = async mode => {
      for (const delegate of ['GPU', 'CPU']) {
        try {
          return await FaceLandmarker.createFromOptions(fs, {
            baseOptions: { modelAssetPath: MODELS.face, delegate },
            runningMode: mode, numFaces: 1, outputFaceBlendshapes: mode === 'VIDEO',
          });
        } catch (e) { if (delegate === 'CPU') throw e; }
      }
    };
    ML.imgLM = await makeFace('IMAGE');
    ML.vidLM = await makeFace('VIDEO');
    try {
      const makePose = async m => {
        for (const delegate of ['GPU', 'CPU']) {
          try { return await PoseLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: MODELS.pose, delegate }, runningMode: m, numPoses: 1 }); }
          catch (e) { if (delegate === 'CPU') throw e; }
        }
      };
      say('Downloading the pose landmarker weights (shoulders and arms)…');
      ML.poseImg = await makePose('IMAGE');
      ML.poseVid = await makePose('VIDEO');
    } catch (e) { console.warn('Pose model unavailable - shoulders and arms stay still.', e); }
    modelsReady = true;
    $('camBtn').disabled = false;
    say('Ready. Start the webcam, then choose an image (or snapshot yourself from the camera).');
    return true;
  } catch (e) {
    say('Could not load the face model: ' + (e && e.message ? e.message : e) + ' — this page needs reachability to cdn.jsdelivr.net and storage.googleapis.com.');
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* image -> rig                                                        */
/* ------------------------------------------------------------------ */
/**
 * Learn the sign convention of the API's z at runtime. The nose always
 * protrudes toward the camera, so correlate the reported z against a quick
 * analytic front-back estimate on the same landmarks.
 */
export function detectDepthSign(V) {
  const cx = (V[234][0] + V[454][0]) / 2, cy = (V[33][1] + V[263][1]) / 2;
  const fw = Math.hypot(V[234][0] - V[454][0], V[234][1] - V[454][1]) || 1;
  const ax = fw * 0.55, ay = fw * 0.78;
  let mz = 0;
  for (let i = 0; i < 478; i++) mz += V[i][2] || 0;
  mz /= 478;
  const prior = new Float64Array(478);
  let mp = 0;
  for (let i = 0; i < 478; i++) {
    const dx = (V[i][0] - cx) / ax, dy = (V[i][1] - cy) / ay;
    prior[i] = Math.pow(Math.max(0, 1 - dx * dx - dy * dy), 0.55);
    mp += prior[i];
  }
  mp /= 478;
  let cov = 0;
  for (let i = 0; i < 478; i++) cov += ((V[i][2] || 0) - mz) * (prior[i] - mp);
  return cov < 0 ? -1 : 1;
}

export function setImageFrom(source, w, h, opts = {}) {
  if (!ML.imgLM) { status('The face model is not loaded yet.'); return false; }
  lastSource = { el: source, w, h };
  const s = Math.min((opts.maxW || 960) / w, (opts.maxH || 780) / h, 1);
  const W = Math.max(2, Math.round(w * s)), H = Math.max(2, Math.round(h * s));
  sc.width = W; sc.height = H;
  pc.width = W; pc.height = H;
  plateImg = null; irisSprites = null; rig = null; st = null; posedOnce = false;
  archCache = null; Dview = null; partners = null;
  sx.setTransform(1, 0, 0, 1, 0, 0);
  sx.drawImage(source, 0, 0, W, H);
  let r;
  try { r = ML.imgLM.detect(sc); }
  catch (e) { status('Face detection failed: ' + (e.message || e)); return false; }
  if (!r.faceLandmarks || !r.faceLandmarks.length) {
    px.drawImage(sc, 0, 0);
    status('No face detected in that image. Try a clearer, front-facing photo with even lighting.');
    return false;
  }
  const raw = r.faceLandmarks[0].map(p => [p.x * W, p.y * H, p.z * W]);
  zSign = detectDepthSign(raw);
  const V = raw.map(p => [p[0], p[1], p[2] * zSign]);
  try {
    rig = buildRig(V, W, H, tessellation(V), {
      depthScale: num('depth', 1), focalScale: focalFromSlider(), detectorBlend: num('detBlend', 0.34), zSign: 1,
    });
  } catch (e) { console.error(e); status('Could not build the head model: ' + e.message); return false; }
  st = createPoseState(rig);
  st.dragField = buildDragField(rig, num('da', 1));
  const keepE = E && E.length === SHOWN.count * 2 ? Float32Array.from(E) : null;
  E = new Float32Array(SHOWN.count * 2);
  Ec = new Float32Array(SHOWN.count * 2);
  UND = [];
  if (keepE) { E.set(keepE); Ec.set(keepE); }
  buildPlate(W, H);
  buildIrisSprites();
  detectBodyOnPhoto();
  sampleLuminance();
  neutral = null; map = null;                     // a new photo invalidates the mapping
  hair = { x: 0, y: 0, vx: 0, vy: 0 };
  SM.yaw.snap(0); SM.pitch.snap(0); SM.roll.snap(0); SM.tx.snap(0); SM.ty.snap(0); SM.tz.snap(0);
  for (const g of Object.values(GATE)) g.snap(0);
  $('photoBtn').disabled = false;
  $('recordBtn').disabled = false;
  status(`Head model built: 478-point 3D mesh, ${rig.tri.count} triangles, ${SHOWN.count} draggable landmarks, `
    + `${rig.bodyStart < rig.n ? 'upper-body grid' : 'head only'}, background plate ${plateImg ? 'inpainted' : 'off'}, `
    + `depth sign ${zSign > 0 ? '+z toward camera' : '−z toward camera'}. Drag landmarks, click-to-sculpt, or start the webcam.`);
  return true;
}
function tessellation(V) {
  const t = ML.FaceLandmarker && ML.FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  return t || knnTess(V, 6);
}
function focalFromSlider() { return lerp(9.5, 2.3, clamp(num('focal', 0.45), 0, 1)); }

/** Depth-model sliders need a rebuild because canonical space itself changes. */
export function rebuildRig() {
  if (!lastSource) return false;
  return setImageFrom(lastSource.el, lastSource.w, lastSource.h, { quiet: true });
}

/** Inpaint the background plate so the head can turn without ghosting. */
function buildPlate(W, H) {
  if (!chk('plate', true)) { plateImg = null; return; }
  const f = rig.frame;
  const cutY = f.chinY + f.FH * 0.30;
  const pts = [];
  for (let k = 0; k < OVAL.length; k++) {
    const i = rig.ringStart + k;                   // ring at scale 1.25
    pts.push(rig.V[i][0], Math.min(rig.V[i][1], cutY));
  }
  pts.push(f.cx + f.FW * 0.62, cutY, f.cx - f.FW * 0.62, cutY);
  let mask = rasterizePolygon(W, H, pts);
  mask = dilateMask(mask, W, H, Math.max(2, f.FW * 0.045));
  const img = sx.getImageData(0, 0, W, H);
  const t0 = performance.now();
  const res = inpaintPlate(img, mask, { levels: 4, iters: 14, blur: Math.max(1, f.FW * 0.01) });
  plateC.width = W; plateC.height = H;
  plateC.getContext('2d').putImageData(new ImageData(res.data, W, H), 0, 0);
  plateImg = plateC;
  console.info(`background plate inpainted in ${(performance.now() - t0).toFixed(1)} ms (coverage ${(res.coverage * 100).toFixed(0)}%)`);
}

function buildIrisSprites() {
  const W = sc.width, H = sc.height;
  const grab = (x, y, w, h) => {
    const xx = clamp(Math.round(x), 0, Math.max(0, W - 1)), yy = clamp(Math.round(y), 0, Math.max(0, H - 1));
    const ww = clamp(Math.round(w), 1, W - xx), hh = clamp(Math.round(h), 1, H - yy);
    return sx.getImageData(xx, yy, ww, hh);
  };
  try { irisSprites = prepIris(sc, grab, rig.V, (w, h) => { const c = mkOff(); c.width = w; c.height = h; return c; }); }
  catch (e) { console.warn('iris sprite prep failed', e); irisSprites = null; }
}

function sampleLuminance() {
  try {
    const W = sc.width, H = sc.height;
    const at = i => {
      const x = clamp(Math.round(rig.V[i][0]) - 4, 0, Math.max(0, W - 9)), y = clamp(Math.round(rig.V[i][1]) - 4, 0, Math.max(0, H - 9));
      const d = sx.getImageData(x, y, 9, 9).data;
      let l = 0;
      for (let k = 0; k < d.length; k += 4) l += 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2];
      return l / (d.length / 4) / 255;
    };
    LUM = (at(205) + at(425)) / 2;
  } catch (e) { LUM = 0.7; }
}

/** Shoulders / elbows / wrists of the photo, detected once per image. */
function detectBodyOnPhoto() {
  bodyJoints = null;
  if (!ML.poseImg || !rig) return;
  try {
    const pr = ML.poseImg.detect(sc).landmarks[0];
    if (pr && pr[11].visibility > 0.4 && pr[12].visibility > 0.4) {
      const W = sc.width, H = sc.height;
      const q = pr.map(p => [p.x * W, p.y * H, p.visibility]);
      if (Math.hypot(q[11][0] - q[12][0], q[11][1] - q[12][1]) > rig.frame.FH * 0.35) bodyJoints = q;
    }
  } catch (e) { console.warn(e); }
}

/* ------------------------------------------------------------------ */
/* webcam                                                              */
/* ------------------------------------------------------------------ */
export async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
    });
    vid.srcObject = stream;
    await vid.play();
    camC.width = vid.videoWidth || 640;
    camC.height = vid.videoHeight || 480;
    camOn = true;
    oneEuro = new OneEuroField(478, { minCutoff: 0.55, beta: 26, dCutoff: 1.1, scale: camC.width });
    const boost = new Float32Array(478).fill(1);
    for (const i of LIPSET) boost[i] = 2.6;                       // lips need a faster cutoff for crisp speech
    for (const i of EYE[0]) boost[i] = 1.6;
    for (const i of EYE[1]) boost[i] = 1.6;
    for (const i of IRIS) boost[i] = 1.4;
    oneEuro.boost = boost;
    camRaw = null; camPts = null; neutral = null;
    $('camBtn').disabled = true;
    $('snapBtn').disabled = false;
    $('calBtn').disabled = false;
    status('Lift off. Sit back so your head and shoulders are visible, then relax your face - that becomes the neutral pose ("Set neutral pose" redoes it any time).');
    return true;
  } catch (e) {
    status('Webcam unavailable: ' + (e && e.message ? e.message : e) + ' (camera access needs https:// or localhost).');
    return false;
  }
}

function setNeutral() {
  if (!camP) return;
  neutral = camP.map(p => [p[0], p[1], p[2]]);
  for (let n = 0; n < 2; n++) {
    gaze0[n] = gazeRaw(n, neutral);
    SM.gaze[n][0].snap(0); SM.gaze[n][1].snap(0);
    // the vertical/horizontal eye ratio measured right now is the "wide open" reference
    const d = (p, q) => Math.hypot(neutral[p][0] - neutral[q][0], neutral[p][1] - neutral[q][1]);
    SM.eyeOpen[n].v = clamp(d(EYM[n][0], EYM[n][1]) / (d(EYM[n][2], EYM[n][3]) || 1), 0.12, 0.5);
  }
  if (rig) {
    const A = RIG.map(i => neutral[i]);
    const B = RIG.map(i => [rig.Xc[i * 3], rig.Xc[i * 3 + 1], rig.Xc[i * 3 + 2]]);
    map = fitSimilarity(A, B, RIG.map(() => 1));
  }
  if (poseP) poseNeutral = poseP.map(p => p.slice());
}

function trackDriver(now, dt) {
  if (!camOn || !ML.vidLM || vid.readyState < 2) { seenFace = false; return; }
  if (vid.currentTime === lastVideoTime) return;      // wait for a genuinely new frame
  lastVideoTime = vid.currentTime;
  const w = camC.width, h = camC.height;
  const ts = Math.max(now, lastMpTs + 1);              // MediaPipe needs monotonically increasing stamps
  lastMpTs = ts;
  let res;
  try { res = ML.vidLM.detectForVideo(vid, ts); } catch (e) { console.warn(e); return; }
  bs = {};
  if (res.faceBlendshapes && res.faceBlendshapes[0]) for (const c of res.faceBlendshapes[0].categories) bs[c.categoryName] = c.score;
  const L = res.faceLandmarks && res.faceLandmarks[0];
  if (L) {
    lastFaceAt = now;
    if (!camRaw || camRaw.length !== L.length) camRaw = new Array(L.length);
    // mirrored: the puppet behaves like the reflection the user sees
    for (let i = 0; i < L.length; i++) camRaw[i] = [w - L[i].x * w, L[i].y * h, L[i].z * w * zSign];
    const sm = clamp(num('sm', 0.8), 0, 1);
    oneEuro.minCutoff = lerp(0.3, 1.5, sm);
    oneEuro.beta = lerp(34, 10, sm);
    const flat = oneEuro.update(camRaw, now);
    if (!camPts || camPts.length !== 478) camPts = new Array(478);
    for (let i = 0; i < 478; i++) {
      const p = camPts[i] || (camPts[i] = [0, 0, 0]);
      p[0] = flat[i * 3]; p[1] = flat[i * 3 + 1]; p[2] = flat[i * 3 + 2];
    }
    camP = camPts;
    if (!neutral) setNeutral();
  }
  seenFace = !!L && (now - lastFaceAt < 420);
  if (!minimized && cx) drawCamPreview(w, h);
  if (ML.poseVid && now - lastPoseFrame > 66) { lastPoseFrame = now; trackPose(now, w, h); }
}

function drawCamPreview(w, h) {
  cx.save();
  cx.setTransform(-1, 0, 0, 1, w, 0);
  cx.drawImage(vid, 0, 0, w, h);
  cx.restore();
  if (camP && chk('showPts', true)) drawDriverOverlay(cx, camP, SHOWN);
  drawPoseOverlay();
}

function trackPose(now, w, h) {
  try {
    const ts = Math.max(now, lastPoseTs + 1);
    lastPoseTs = ts;
    const pr = ML.poseVid.detectForVideo(vid, ts).landmarks[0];
    if (!pr) { poseP = null; return; }
    const q = pr.map(p => [w - p.x * w, p.y * h, p.visibility || 0]);
    poseP = poseP ? q.map((p, i) => [poseP[i][0] + (p[0] - poseP[i][0]) * 0.5, poseP[i][1] + (p[1] - poseP[i][1]) * 0.5, p[2]]) : q;
    if (!poseNeutral && q[11][2] > 0.5 && q[12][2] > 0.5) poseNeutral = poseP.map(p => p.slice());
  } catch (e) { console.warn(e); }
}

function drawPoseOverlay() {
  if (!poseP || !chk('showPts', true)) return;
  cx.strokeStyle = '#fbbf24'; cx.lineWidth = 2; cx.lineCap = 'round';
  for (const [a, b] of [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]) {
    if (poseP[a][2] > 0.4 && poseP[b][2] > 0.4) { cx.beginPath(); cx.moveTo(poseP[a][0], poseP[a][1]); cx.lineTo(poseP[b][0], poseP[b][1]); cx.stroke(); }
  }
  for (const i of [11, 12, 13, 14, 15, 16]) {
    if (poseP[i][2] > 0.4) { cx.beginPath(); cx.arc(poseP[i][0], poseP[i][1], 5, 0, 7); cx.fillStyle = '#fbbf24'; cx.fill(); }
  }
}

/* ------------------------------------------------------------------ */
/* expression: presets, audio, live blendshapes                        */
/* ------------------------------------------------------------------ */
let tkey = false, nextBlink = 0, blinkAt = -1e9, sacT = 0, sacV = [0, 0];

function stepPreset(t, dt) {
  const amt = num('pi', 1);
  let target = {};
  if (preset === 'Talking') target = talkParams(t);
  else if (preset !== 'live') target = PRESETS[preset] || {};
  if (audioOn && viseme) {
    const g = num('audGain', 1), tg = g * num('audTongue', 1);
    target = Object.assign({}, target);
    target.j = Math.max(target.j || 0, viseme.jaw * g);
    target.sL = Math.max(target.sL || 0, viseme.spread * 0.55 * g);
    target.sR = Math.max(target.sR || 0, viseme.spread * 0.55 * g);
    target.pk = Math.max(target.pk || 0, viseme.pucker * g);
    target.st = Math.max(target.st || 0, viseme.stretch * 0.8 * g);
    target.tt = Math.max(target.tt || 0, viseme.tongueTip * tg);
    target.tb = Math.max(target.tb || 0, viseme.tongueBack * tg);
    target.tg = Math.max(target.tg || 0, viseme.tongueOut * tg);
    target.cl = Math.max(target.cl || 0, viseme.closure * g);
    target.bu = Math.max(target.bu || 0, viseme.brow * 0.4 * g);
    target.fr = Math.max(target.fr || 0, viseme.stretch * 0.2 * g);
  }
  // presets glide, speech stays snappy
  const rt = 1 - Math.exp(-dt / (preset === 'Talking' || audioOn ? 0.035 : 0.075));
  for (const k of PARAMS) {
    const v = (target[k] || 0) * amt;
    PT[k] = v;
    PP[k] += (v - PP[k]) * rt;
    if (Math.abs(PP[k]) < 1e-4) PP[k] = 0;
  }
}

function updateExpression(now, dt) {
  const t = now / 1000;
  stepPreset(t, dt);
  const cf = SM.conf.value;
  const live = camP && neutral && rig && cf > 0.02;
  const useLive = live && (preset === 'live' || chk('pmix', false));
  let liveJawAngle = 0, tv = 0;

  if (useLive) {
    // ---- eyes: geometric openness cross-checked with the blink blendshapes ----
    const geo = EYM.map((e, n) => {
      const d = (p, q) => Math.hypot(camP[p][0] - camP[q][0], camP[p][1] - camP[q][1]);
      const ratio = d(e[0], e[1]) / (d(e[2], e[3]) || 1);
      const peak = SM.eyeOpen[n].update(ratio, dt);
      return clamp((peak * 0.85 - ratio) / (peak * 0.6), 0, 1);
    });
    let bl = geo.slice();
    if (Object.keys(bs).length > 0) {
      bl = [clamp(((bs.eyeBlinkRight || 0) - 0.15) / 0.6, 0, 1), clamp(((bs.eyeBlinkLeft || 0) - 0.15) / 0.6, 0, 1)];
      // the API's left/right can disagree with image left/right: keep the cheaper pairing
      const same = Math.abs(bl[0] - geo[0]) + Math.abs(bl[1] - geo[1]);
      const swap = Math.abs(bl[0] - geo[1]) + Math.abs(bl[1] - geo[0]);
      if (swap < same) bl = [bl[1], bl[0]];
    }
    const BL = num('bl', 1);
    for (let n = 0; n < 2; n++) {
      const mix = (geo[n] + bl[n]) / 2;
      // a wink keeps the other eye open: suppress the isolated blink
      const other = (geo[1 - n] + bl[1 - n]) / 2;
      const m = mix > 0.4 && other < 0.12 ? mix * 0.35 : mix;
      SM.lid[n].update(clamp((m * BL - 0.1) / 0.75, 0, 1) * cf, dt);
    }
    // ---- gaze from the irises ----
    for (let n = 0; n < 2; n++) {
      const r = gazeRaw(n, camP);
      SM.gaze[n][0].update(clamp((r[0] - gaze0[n][0]) * cf, -0.16, 0.16), dt);
      SM.gaze[n][1].update(clamp((r[1] - gaze0[n][1]) * cf, -0.1, 0.1), dt);
    }
    // ---- mouth ----
    tv = clamp(((bs.tongueOut || 0) * num('tsens', 1.4) - 0.08) / 0.45, 0, 1);
    const pkLive = clamp(Math.max(bs.mouthPucker || 0, (bs.mouthFunnel || 0) * 0.7) * 1.4 - 0.15 - (bs.mouthSmileLeft || 0) * 0.3, 0, 1);
    SM.kiss.update(pkLive * cf, dt);
    try {
      const pose = driverPose(neutral, camP);
      liveJawAngle = driverJawAngle(neutral, camP, pose) * cf;
    } catch (e) { liveJawAngle = 0; }
  } else {
    // idle: natural blinking + microsaccades so a still photo never looks dead
    if (now > nextBlink) { nextBlink = now + 2400 + Math.random() * 3600; blinkAt = now; }
    const k = (now - blinkAt) / 210;
    const idle = chk('ab', false) && k >= 0 && k < 1 ? Math.sin(Math.PI * k) : 0;
    for (let n = 0; n < 2; n++) SM.lid[n].update(idle, dt);
    if (now > sacT) { sacT = now + 700 + Math.random() * 2200; sacV = [(Math.random() - 0.5) * 0.07, (Math.random() - 0.5) * 0.045]; }
    for (let n = 0; n < 2; n++) { SM.gaze[n][0].update(sacV[0], dt); SM.gaze[n][1].update(sacV[1], dt); }
    SM.kiss.update(0, dt);
  }
  // preset / audio contributions ride on top of the live signal
  SM.kiss.update(Math.max(SM.kiss.value, PP.pk), dt);
  const forced = (chk('tf', false) || tkey) ? 1 : 0;
  SM.tongue.update(clamp(Math.max(tv * cf, PP.tg, forced), 0, 1), dt);
  SM.tTip.update(clamp(PP.tt, 0, 1), dt);
  SM.tBack.update(clamp(PP.tb, 0, 1), dt);
  SM.tWide.update(clamp(0.5 + PP.st * 0.5 - PP.pk * 0.35, 0, 1), dt);
  SM.tOut.update(clamp(PP.tg * 0.9 + forced * 0.9, 0, 1), dt);
  const openMag = Math.max(0, -liveJawAngle) * num('jawGain', 1);
  SM.jaw.update(clamp(openMag + PP.j * 0.44 * num('jawGain', 1), 0, 0.62), dt);
  return { useLive };
}

/** Mouth measurements + gates, computed after the mesh has been posed. */
function updateMouthGates(dt) {
  if (!rig || !st || !posedOnce) return;
  const P = st.P;
  const g = (a, b) => Math.hypot(P[a * 2] - P[b * 2], P[a * 2 + 1] - P[b * 2 + 1]);
  const mw = g(61, 291) || 1, gap = g(13, 14);
  const bsu = ((bs.mouthUpperUpLeft || 0) + (bs.mouthUpperUpRight || 0)) / 2;
  GATE.open.update(gap / mw, dt);
  GATE.teeth.update(gap / mw, dt);
  GATE.lower.update(gap / mw, dt);
  GATE.gum.update(clamp(bsu + PP.ur + gap / mw * 0.5, 0, 1.4), dt);
  GATE.tongue.update(SM.tongue.value, dt);
  GATE.tongueOut.update(SM.tOut.value, dt);
  GATE.kiss.update(SM.kiss.value, dt);
  GATE.wet.update(clamp(gap / mw + SM.kiss.value * 0.3, 0, 1), dt);
  GATE.seam.update(clamp(1 - gap / (mw * 0.07), 0, 1), dt);
  GATE.ao.update(gap / mw, dt);
  GATE.throat.update(gap / mw, dt);
  mouthMeas.mw = mw; mouthMeas.gap = gap;
}

/* ------------------------------------------------------------------ */
/* head pose                                                           */
/* ------------------------------------------------------------------ */
const ctl = { R: null, expr: null, exprGain: 1, preset: null, drag: null, body: null, rotMix: 0 };
const exprBuf = new Float32Array(478 * 3);
const presetBuf = new Float32Array(468 * 2);
let poseNow = { yaw: 0, pitch: 0, roll: 0, R: ID3() };

function updatePose(dt, now, useLive) {
  const t = now / 1000;
  let R = null, tx = 0, ty = 0, tz = 0, jawAngle = 0, expr = null;
  const cf = SM.conf.value;
  const sm = clamp(num('sm', 0.8), 0, 1);
  const tau = lerp(0.02, 0.22, sm);
  SM.yaw.tau = tau; SM.pitch.tau = tau * 1.15; SM.roll.tau = tau * 1.3;

  if (useLive && camP && neutral && map && rig) {
    let pose = null;
    try { pose = driverPose(neutral, camP); } catch (e) { pose = null; }
    if (pose) {
      const e = eulerFromR(pose.R);
      const turn = num('turn', 1);
      const yaw = SM.yaw.update(softLimit(e.yaw, num('yawG', 1) * turn, num('yawMax', 0.86)), dt);
      const pitch = SM.pitch.update(softLimit(e.pitch, num('pitchG', 1) * turn, num('pitchMax', 0.5)), dt);
      const roll = SM.roll.update(softLimit(e.roll, num('rollG', 1) * turn, num('rollMax', 0.5)), dt);
      R = RFromEuler(yaw, pitch, roll);
      poseNow = { yaw, pitch, roll, R };
      // pan + dolly: a scale change becomes a real camera approach, not a 2D zoom
      const f = rig.frame;
      const sc2 = clamp(Math.pow(pose.s, 0.72), 0.6, 1.6);
      const panGain = num('pan', 0.8);
      tx = SM.tx.update((pose.cb[0] - pose.ca[0]) * map.s * panGain, dt);
      ty = SM.ty.update((pose.cb[1] - pose.ca[1]) * map.s * panGain, dt);
      tz = SM.tz.update(f.F * (1 - 1 / sc2) * num('dolly', 0.7), dt);
      expressionField(neutral, camP, pose, map, exprBuf, {
        lip: num('lx', 1.25) * num('retMouth', 1),
        eye: num('retEye', 1),
        nose: num('nf', 1),
        jawW: rig.jawW,
        jawCancel: num('jawCancel', 1),
      });
      stitchRegions(rig, exprBuf, num('stitch', 0.85) * cf, rig.stitchGroups);
      expr = exprBuf;
    }
  } else if (rig) {
    // no driver: presets still move the face, the head relaxes toward rest
    const yaw = SM.yaw.update(0, dt), pitch = SM.pitch.update(0, dt), roll = SM.roll.update(0, dt);
    R = RFromEuler(yaw, pitch, roll);
    poseNow = { yaw, pitch, roll, R };
  }
  jawAngle = -SM.jaw.value;
  // audio prosody: gentle head motion from the pitch contour and accent envelope
  if (R && audioOn && viseme && num('audHead', 0.6) > 0) {
    const g = num('audHead', 0.6) * 0.1;
    const e = eulerFromR(R);
    const yaw2 = e.yaw + Math.sin(viseme.accent * 3.1) * g * 0.45;
    const pitch2 = e.pitch + clamp(viseme.f0Delta, -1, 1) * g;
    R = RFromEuler(yaw2, pitch2, e.roll + viseme.accent * g * 0.12);
    poseNow = { yaw: yaw2, pitch: pitch2, roll: e.roll, R };
  }

  const wantPreset = preset !== 'live' || chk('pmix', false) || audioOn;
  ctl.preset = (wantPreset && rig) ? presetOffsets(rig, PP, presetBuf, audioOn && viseme ? viseme : null) : null;
  ctl.R = R;
  ctl.rotMix = clamp(cf, 0, 1);
  ctl.tx = tx; ctl.ty = ty; ctl.tz = tz;
  ctl.expr = expr;
  ctl.exprGain = num('exs', 0.8) * clamp(cf, 0, 1);
  ctl.jawAngle = jawAngle;
  ctl.jawGain = num('jawGain', 1);
  ctl.cullBias = chk('cull', false) ? 0.02 : -2;
  ctl.stretch = num('stretch', 2);
  ctl.relaxIters = 3;
  ctl.hairGain = num('hs', 0.6);
  ctl.bodyGain = num('bm', 1);
  ctl.drag = Ec;

  if (rig) {
    const f = rig.frame;
    // hair: a damped spring chasing the skull (continuous time, frame-rate independent)
    const Tx = tx - f.FH * 0.55 * (R ? R[1] : 0) + f.FH * 0.25 * (R ? R[2] : 0);
    const Ty = ty - f.FH * 0.55 * ((R ? R[4] : 1) - 1) + f.FH * 0.10 * (R ? R[5] : 0) + Math.sin(t * 1.3) * f.FH * 0.004;
    const k = 92, c = 8.6;
    hair.vx += (k * (Tx - hair.x) - c * hair.vx) * dt;
    hair.vy += (k * (Ty - hair.y) - c * hair.vy) * dt;
    hair.x += hair.vx * dt; hair.y += hair.vy * dt;
    let hlx = (hair.x - Tx) * 2.4, hly = (hair.y - Ty) * 2.4;
    const hm = Math.hypot(hlx, hly), hc = f.FH * 0.14;
    if (hm > hc) { hlx *= hc / hm; hly *= hc / hm; }
    ctl.hairX = hlx; ctl.hairY = hly;
    ctl.body = bodyDisplacements();
    ctl.bodySigma = f.FH * 0.55;
  }
}

/** Shoulder / elbow / wrist motion of the photo, from the live pose tracker. */
function bodyDisplacements() {
  if (!bodyJoints || !poseP || !poseNeutral) return null;
  const B = num('bm', 1);
  if (B <= 0.001) return null;
  const iw = Math.hypot(bodyJoints[11][0] - bodyJoints[12][0], bodyJoints[11][1] - bodyJoints[12][1]);
  const cw = Math.hypot(poseNeutral[11][0] - poseNeutral[12][0], poseNeutral[11][1] - poseNeutral[12][1]) || 1;
  const sg = iw / cw;
  const swapped = (bodyJoints[11][0] < bodyJoints[12][0]) !== (poseNeutral[11][0] < poseNeutral[12][0]);
  const breath = Math.sin(performance.now() / 1000 * 1.6) * rig.frame.FH * 0.006 * B;
  const out = [];
  for (let j = 11; j <= 16; j++) {
    const c = swapped ? (j % 2 ? j + 1 : j - 1) : j;
    const q = bodyJoints[j];
    const ok = poseP[c] && poseP[c][2] > 0.4 && poseNeutral[c][2] > 0.4 && q[2] > 0.4;
    if (!ok) { out.push(j < 13 ? { x: q[0], y: q[1], dx: 0, dy: breath } : null); continue; }
    out.push({
      x: q[0], y: q[1],
      dx: (poseP[c][0] - poseNeutral[c][0]) * sg * B,
      dy: (poseP[c][1] - poseNeutral[c][1]) * sg * B + (j < 13 ? breath : 0),
    });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* preset offsets (image space, applied in canonical head space)       */
/* ------------------------------------------------------------------ */
const BROWS = new Set(GROUPS.brow);
const INB = new Set([66, 107, 336, 296, 55, 285]);
const UL = new Set([159, 158, 160, 157, 161, 246, 386, 385, 387, 384, 388, 466]);
const LL = new Set([145, 153, 144, 154, 163, 7, 374, 380, 373, 381, 390, 249]);
const NO_GEOM = new Set(['gx', 'gy', 'tg', 'tt', 'tb']);

export function presetOffsets(rig, A, out, vis) {
  const V = rig.V, { MW, FH, cx } = rig.frame;
  const cxm = (V[61][0] + V[291][0]) / 2, cym = (V[13][1] + V[14][1]) / 2;
  const y13 = V[13][1];
  const EW = (rig.frame.EW[0] + rig.frame.EW[1]) / 2;
  const cl = vis ? vis.closure : 0;
  let any = Math.abs(cl) > 0.02;
  for (const k of PARAMS) if (!NO_GEOM.has(k) && Math.abs(A[k]) > 0.002) { any = true; break; }
  if (!any) { out.fill(0); return null; }
  const jawShare = num('jawSlide', 0.3);
  for (let i = 0; i < 468; i++) {
    const x = V[i][0], y = V[i][1];
    const u = clamp((x - cxm) / (MW * 0.5), -1.6, 1.6), au = Math.min(1, Math.abs(u));
    const sd = x < cx ? 0 : 1, sm = sd ? A.sR : A.sL;
    let dx = 0, dy = 0;
    if (LIPSET.has(i)) {
      dy += -sm * MW * 0.15 * au * au + A.fr * MW * 0.10 * au * au
        - A.ur * MW * 0.10 * (y < cym ? 1 - au * 0.5 : 0)
        - (y > cym ? A.fr * MW * 0.04 * (1 - au) : 0);
      dx += u * (sm * 0.06 + A.st * 0.12) * MW * au;
      // bilabial closure from audio: the lips press together
      dy += (y < cym ? cl * MW * 0.055 : -cl * MW * 0.075);
    } else {
      const wx = (x - cxm) - Math.sign(x - cxm || 1) * MW * 0.7, wy = y - cym + MW * 0.25;
      const wc = Math.exp(-(wx * wx + wy * wy) / (MW * MW * 0.25));
      dy -= sm * MW * 0.06 * wc;
      dx += Math.sign(x - cxm || 1) * sm * MW * 0.02 * wc;
    }
    // the hinge does most of the jaw work; this 2D slide only softens the chin skin
    dy += A.j * MW * 0.55 * jawShare * clamp((y - y13) / (FH * 0.05), 0, 1);
    if (BROWS.has(i)) {
      const inn = INB.has(i);
      dy += -(A.bu + (sd ? A.buR : 0)) * FH * 0.055
        - (inn ? A.bi * FH * 0.06 : -A.bi * FH * 0.015)
        + A.bd * FH * (inn ? 0.05 : 0.03);
      if (inn) dx += -Math.sign(x - cx || 1) * A.bd * FH * 0.02;
    }
    if (NOSE_PTS.has(i)) dy -= A.nz * FH * 0.018;
    else if (NOSE_BRIDGE.has(i) && i !== 1) dy -= A.nz * FH * 0.009;
    if (UL.has(i)) dy += -A.eo * EW * 0.12 + A.sq * EW * 0.05;
    else if (LL.has(i)) dy += A.eo * EW * 0.04 - A.sq * EW * 0.12;
    out[i * 2] = dx; out[i * 2 + 1] = dy;
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* render                                                              */
/* ------------------------------------------------------------------ */
/** Assemble the settings bag and hand the frame to frame.js. */
function render(dt) {
  px.setTransform(1, 0, 0, 1, 0, 0);
  if (!rig || !st) {
    px.fillStyle = '#0a0c10'; px.fillRect(0, 0, pc.width, pc.height);
    px.fillStyle = '#8b93a7'; px.font = '15px system-ui, sans-serif'; px.textAlign = 'center';
    px.fillText('Load an image, take a webcam snapshot, or use the demo portrait to build the puppet', pc.width / 2, pc.height / 2);
    return;
  }
  poseFrame(rig, st, ctl);
  posedOnce = true;
  updateMouthGates(dt);
  Dview = pointView(st.P, Dview);
  const gaze = [[SM.gaze[0][0].value + PP.gx * 0.28, SM.gaze[0][1].value + PP.gy * 0.24],
                [SM.gaze[1][0].value + PP.gx * 0.28, SM.gaze[1][1].value + PP.gy * 0.24]];
  const lid = [0, 1].map(n => clamp(SM.lid[n].value + PP.dr * 0.5 + PP.sq * 0.22 + Math.max(0, gaze[n][1]) * 1.4 + (n ? PP.bR : PP.bL), 0, 1));
  fxOut = renderFrame(px, rig, st, ctl, {
    D: Dview, Dcache: Dview, src: sc, plate: plateImg, scratch,
    pose: poseNow, dt,
    amt: {
      open: GATE.open.value, gap: mouthMeas.gap, teeth: GATE.teeth.value, lower: GATE.lower.value,
      gum: GATE.gum.value, tongue: GATE.tongue.value, tongueOut: GATE.tongueOut.value,
      kiss: GATE.kiss.value, ao: GATE.ao.value, wet: GATE.wet.value, seam: GATE.seam.value, throat: GATE.throat.value,
    },
    mw: mouthMeas.mw, gap: mouthMeas.gap,
    gaze, lid, gazeGain: num('gg', 1.9),
    mouth: {
      tongueAmt: SM.tongue.value, tipUp: SM.tTip.value, backUp: SM.tBack.value,
      wide: SM.tWide.value, protrude: SM.tOut.value,
      toothAmt: num('tv', 0.4), toothScale: num('toothScale', 1), gumVis: num('gumV', 0.6),
      archTau: num('archTau', 0.55), bright: num('tb', 1.15), warm: num('tw', 0),
    },
    archPrev: archCache, iris: irisSprites, tex: textures, lum: LUM,
    blend: num('blend', 0.55), seamAO: num('seamAO', 0.5), shade: num('shade', 0.75), enhance: num('enh', 0.35),
    flags: {
      showOrig: chk('showOrig', false), cull: chk('cull', false), plateOn: chk('plate', true),
      seam: chk('seam', true), showPts: chk('showPts', true), ibug: chk('ibug', false), quality: num('quality', 1),
    },
    shown: SHOWN, drag: dragIdx, hover: hoverIdx, flash: photoFlash,
  });
  archCache = fxOut.arch || archCache;
  if (photoFlash > 0) photoFlash = Math.max(0, photoFlash - dt * 3.2);
  if (recorder && recorder.active) recorder.frame();
}

/* ------------------------------------------------------------------ */
/* procedural textures (enamel + grain), built once                    */
/* ------------------------------------------------------------------ */
function buildTextures() {
  const en = mkOff(); en.width = 128; en.height = 160;
  const c = en.getContext('2d');
  const g = c.createLinearGradient(0, 0, 0, 160);
  [[0, '#d6c7a4'], [0.22, '#ece3cc'], [0.6, '#f6f0e2'], [0.88, '#efefe9'], [1, '#d5dcdf']].forEach(q => g.addColorStop(q[0], q[1]));
  c.fillStyle = g; c.fillRect(0, 0, 128, 160);
  const R = (() => { let z = 7; return () => (z = (z * 16807) % 2147483647) / 2147483647; })();
  for (let i = 0; i < 46; i++) {                      // mottled enamel: warm and cool blotches
    const x = R() * 128, y = R() * 160, r = 18 + R() * 34;
    const rg = c.createRadialGradient(x, y, 0, x, y, r), warm = R() < 0.5;
    rg.addColorStop(0, warm ? 'rgba(255,252,240,.18)' : 'rgba(190,160,100,.14)');
    rg.addColorStop(1, 'rgba(255,255,255,0)');
    c.fillStyle = rg; c.fillRect(0, 0, 128, 160);
  }
  for (let i = 0; i < 90; i++) {                      // perikymata: faint vertical growth lines
    const x = R() * 128;
    c.fillStyle = R() < 0.5 ? 'rgba(255,255,255,.07)' : 'rgba(120,95,60,.06)';
    c.fillRect(x, R() * 60, 1 + R() * 2, 60 + R() * 100);
  }
  textures.enamel = en;
  const nz = mkOff(); nz.width = nz.height = 96;
  const nc = nz.getContext('2d'), d = nc.createImageData(96, 96);
  for (let i = 0; i < d.data.length; i += 4) { d.data[i] = d.data[i + 1] = d.data[i + 2] = Math.random() * 255; d.data[i + 3] = 255; }
  nc.putImageData(d, 0, 0);
  textures.noise = nz;
}

/* ------------------------------------------------------------------ */
/* main loop                                                           */
/* ------------------------------------------------------------------ */
let raf = 0;
function loop() { raf = requestAnimationFrame(loop); tick(); }

function tick() {
  const now = performance.now();
  const dt = clamp((now - lastT) / 1000, 0.001, 0.1);
  lastT = now;
  frames++;
  if (now - fpsAt > 500) { fpsShown = Math.round(frames * 1000 / (now - fpsAt)); frames = 0; fpsAt = now; }
  try {
    if (audioOn) viseme = audio.update(dt, { gain: num('audGain', 1), forceTongue: chk('tf', false) || tkey });
    trackDriver(now, dt);
    SM.conf.update(seenFace, dt);
    const { useLive } = updateExpression(now, dt);
    updatePose(dt, now, useLive);
    smoothEdits(dt);
    render(dt);
    hud();
  } catch (e) {
    console.error(e);
    if (now - (tick.lastErr || 0) > 4000) { tick.lastErr = now; status('Render hiccup: ' + (e && e.message ? e.message : e)); }
  }
}

/** Sculpt edits glide to their target instead of snapping (another flicker source). */
function smoothEdits(dt) {
  if (!Ec || !E) return false;
  const k = smoothK(num('sculptTau', 0.06), dt);
  let moving = false;
  for (let i = 0; i < Ec.length; i++) {
    const d = E[i] - Ec[i];
    if (Math.abs(d) > 1e-4) { Ec[i] += d * k; moving = true; } else Ec[i] = E[i];
  }
  return moving;
}

/* ------------------------------------------------------------------ */
/* HUD                                                                 */
/* ------------------------------------------------------------------ */
function hud() {
  const m = $('meters');
  if (m) {
    if (!rig) m.textContent = 'No image loaded yet';
    else {
      const p = v => Math.round(clamp(v, 0, 1) * 100) + '%';
      const deg = r => Math.round(r * 180 / Math.PI) + '°';
      m.innerHTML =
        `<span>jaw ${p(SM.jaw.value / 0.5)}</span><span>kiss ${p(SM.kiss.value)}</span>`
        + `<span>lid L ${p(SM.lid[0].value)}</span><span>lid R ${p(SM.lid[1].value)}</span>`
        + `<span>tongue ${p(SM.tongue.value)}</span>`
        + `<span>yaw ${deg(poseNow.yaw)}</span><span>pitch ${deg(poseNow.pitch)}</span><span>roll ${deg(poseNow.roll)}</span>`
        + `<span>stretch ×${(st && st.lastStretch ? st.lastStretch : 1).toFixed(2)}`
          + `${st && st.preStretch > st.lastStretch + 0.005 ? ` (was ×${st.preStretch.toFixed(2)})` : ''}</span>`
        + `<span>tracker ${p(SM.conf.value)}</span><span>body ${poseP ? 'tracked' : '—'}</span>`
        + `<span>${fpsShown} fps</span>`
        + (audioOn && viseme ? `<span class="hot">♪ ${viseme.name}</span>` : '');
    }
  }
  const au = $('aus');
  if (au && camOn) {
    if (!auRows) {
      au.innerHTML = '';
      for (const [label] of AU_MAP) {
        const row = document.createElement('div');
        row.className = 'au';
        row.innerHTML = `<span>${label}</span><i><em></em></i><b>0%</b>`;
        au.appendChild(row);
      }
      auRows = Array.from(au.children);
    }
    AU_MAP.forEach(([label, names], i) => {
      const v = names.reduce((a, n) => Math.max(a, bs[n] || 0), 0);
      const row = auRows[i];
      if (row) { row.querySelector('em').style.width = (v * 100).toFixed(0) + '%'; row.querySelector('b').textContent = Math.round(v * 100) + '%'; }
    });
  }
  const bands = $('bands');
  if (bands && audioOn && viseme) {
    if (!bands.children.length) {
      for (const b of BANDS) {
        const d = document.createElement('div');
        d.className = 'au';
        d.innerHTML = `<span>${b.name}</span><i><em></em></i><b>0%</b>`;
        bands.appendChild(d);
      }
    }
    Array.from(bands.children).forEach((row, i) => {
      const v = clamp((viseme.bands[i] || 0) * 2.2, 0, 1);
      row.querySelector('em').style.width = (v * 100).toFixed(0) + '%';
      row.querySelector('b').textContent = Math.round(v * 100) + '%';
    });
  }
  const rec = $('recBadge');
  if (rec) {
    if (recorder && recorder.active) {
      rec.hidden = false;
      const s = recorder.elapsed;
      rec.textContent = `● REC ${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')} · ${recorder.frames} frames`;
    } else rec.hidden = true;
  }
}

/* ------------------------------------------------------------------ */
/* interaction: drag, sculpt, keyboard                                 */
/* ------------------------------------------------------------------ */
const toCanvas = e => {
  const r = pc.getBoundingClientRect();
  return { x: (e.clientX - r.left) * pc.width / r.width, y: (e.clientY - r.top) * pc.height / r.height };
};
function nearest(m, radius = 36) {
  if (!st) return -1;
  const P = st.P;
  let bi = -1, bd = radius * radius;
  for (let k = 0; k < SHOWN.count; k++) {
    const v = SHOWN.idx[k];
    const dx = P[v * 2] - m.x, dy = P[v * 2 + 1] - m.y, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; bi = k; }
  }
  return bi;
}
function pushUndo() { if (E) { UND.push(Float32Array.from(E)); if (UND.length > 60) UND.shift(); } }
function undo() {
  if (!E) return;
  if (UND.length) { E = UND.pop(); status('Undo.' + (UND.length ? ' ' + UND.length + ' steps left.' : ' Nothing left to undo.')); }
  else status('Nothing to undo.');
}
function partnerOf(k) {
  if (!partners) {
    partners = new Int32Array(SHOWN.count).fill(-1);
    for (let k = 0; k < SHOWN.count; k++) {
      const v = SHOWN.idx[k], p = rig.sym[v];
      if (p < 0 || p === v) continue;
      const j = SHOWN.idx.indexOf(p);
      if (j >= 0 && Math.abs(rig.V[v][1] - rig.V[p][1]) < rig.frame.FH * 0.09) partners[k] = j;
    }
  }
  return partners[k];
}

/** Set one landmark's edit, honouring the per-region drag cap and symmetry. */
function setEdit(k, target) {
  if (!rig || !st) return;
  const v = SHOWN.idx[k], P = st.P;
  // remove what the neighbours already contribute at this vertex
  let ox = 0, oy = 0;
  const df = st.dragField;
  if (df) for (let j = 0; j < df.count; j++) {
    if (j === k) continue;
    const w = df.field[v * df.count + j];
    if (w > 0) { ox += Ec[j * 2] * w; oy += Ec[j * 2 + 1] * w; }
  }
  let ex = target.x - P[v * 2] - ox, ey = target.y - P[v * 2 + 1] - oy;
  const L = Math.hypot(ex, ey), cap = SHOWN.cap[k] * rig.frame.FH * num('dr', 1);
  if (L > cap) { ex *= cap / L; ey *= cap / L; }
  E[k * 2] = ex; E[k * 2 + 1] = ey;
  if (chk('sym', false)) {
    const j = partnerOf(k);
    if (j >= 0) { E[j * 2] = -ex; E[j * 2 + 1] = ey; }
  }
}

const SCULPTS = {
  smile: rig => {
    const u = rig.frame.MW * 0.13;
    return [[61, -u * 0.18, -u * 0.78], [291, u * 0.18, -u * 0.78], [13, 0, -u * 0.12], [14, 0, u * 0.24], [17, 0, u * 0.10], [84, -u * 0.10, -u * 0.30], [314, u * 0.10, -u * 0.30]];
  },
  frown: rig => {
    const u = rig.frame.MW * 0.12;
    return [[61, -u * 0.10, u * 0.72], [291, u * 0.10, u * 0.72], [13, 0, u * 0.10], [14, 0, -u * 0.14], [17, 0, -u * 0.06]];
  },
  o: rig => {
    const u = rig.frame.MW * 0.13;
    return [[61, u * 0.85, -u * 0.10], [291, -u * 0.85, -u * 0.10], [13, 0, -u * 0.60], [14, 0, u * 0.95], [82, 0, -u * 0.30], [312, 0, -u * 0.30], [87, 0, u * 0.40], [317, 0, u * 0.40]];
  },
  kiss: rig => {
    const u = rig.frame.MW * 0.12;
    return [[61, u * 1.10, 0], [291, -u * 1.10, 0], [13, 0, -u * 0.35], [14, 0, u * 0.35], [0, 0, -u * 0.25], [17, 0, u * 0.25]];
  },
  sneer: rig => {
    const u = rig.frame.MW * 0.11;
    return [[61, -u * 0.10, -u * 0.55], [49, 0, -u * 0.50], [107, 0, -u * 0.30], [66, 0, -u * 0.25], [98, 0, -u * 0.20]];
  },
  tongue: rig => {
    const u = rig.frame.MW * 0.12;
    return [[14, 0, u * 0.80], [17, 0, u * 0.55], [87, 0, u * 0.35], [317, 0, u * 0.35]];
  },
};

function sculpt(m) {
  if (!rig || !st || !posedOnce) return false;
  const f = rig.frame;
  const cxp = (rig.V[234][0] + rig.V[454][0]) / 2, cyp = (rig.V[10][1] + rig.V[152][1]) / 2;
  if (((m.x - cxp) / (f.FH * 0.62)) ** 2 + ((m.y - cyp) / (f.FH * 0.72)) ** 2 > 1.6) {
    status('That click was outside the face. Aim near the mouth.');
    return false;
  }
  const defs = SCULPTS[sculptMode];
  if (!defs) return false;
  pushUndo();
  const P = st.P, list = defs(rig);
  for (const [id, dx, dy] of list) {
    const k = SHOWN.idx.indexOf(id);
    if (k >= 0) { setEdit(k, { x: P[id * 2] + dx, y: P[id * 2 + 1] + dy }); continue; }
    // not a draggable landmark: nudge the nearest draggable one instead
    let bk = -1, bd = Infinity;
    for (let j = 0; j < SHOWN.count; j++) {
      const v = SHOWN.idx[j];
      const d = Math.hypot(rig.V[v][0] - rig.V[id][0], rig.V[v][1] - rig.V[id][1]);
      if (d < bd) { bd = d; bk = j; }
    }
    if (bk >= 0 && bd < f.FH * 0.12) setEdit(bk, { x: P[SHOWN.idx[bk] * 2] + dx * 0.7, y: P[SHOWN.idx[bk] * 2 + 1] + dy * 0.7 });
  }
  const names = { smile: 'Smile', frown: 'Frown', o: 'Mouth "O"', kiss: 'Kiss', sneer: 'Sneer', tongue: 'Tongue drop' };
  status(`${names[sculptMode] || sculptMode} sculpt applied to ${list.length} landmarks (mirroring ${chk('sym', false) ? 'on' : 'off'}). Drag dots to refine, Ctrl+Z to undo.`);
  return true;
}

const GROUPS_NAME = { oval: 'Face outline', leye: 'Eye', reye: 'Eye', brow: 'Eyebrow', lips: 'Lips', inner: 'Inner mouth', nose: 'Nose' };

function wireCanvas() {
  pc.style.touchAction = 'none';
  pc.addEventListener('pointerdown', e => {
    if (!rig || chk('showOrig', false)) return;
    const m = toCanvas(e);
    if (sculptMode !== 'drag') { if (sculpt(m)) e.preventDefault(); return; }
    const k = nearest(m);
    if (k < 0) return;
    pushUndo();
    dragIdx = k;
    const v = SHOWN.idx[k];
    dragOff = { x: st.P[v * 2] - m.x, y: st.P[v * 2 + 1] - m.y };
    try { pc.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    pc.style.cursor = 'grabbing';
    e.preventDefault();
  });
  pc.addEventListener('pointermove', e => {
    const m = toCanvas(e);
    if (dragIdx >= 0) {
      setEdit(dragIdx, { x: m.x + dragOff.x, y: m.y + dragOff.y });
      Ec[dragIdx * 2] = E[dragIdx * 2]; Ec[dragIdx * 2 + 1] = E[dragIdx * 2 + 1]; // direct while dragging
      status(`${GROUPS_NAME[SHOWN.group[dragIdx]]}: dragging · release to keep · double-click to reset · Ctrl+Z to undo`);
      return;
    }
    hoverIdx = rig ? nearest(m) : -1;
    pc.style.cursor = hoverIdx >= 0 ? 'grab' : (sculptMode === 'drag' ? 'crosshair' : 'pointer');
    if (hoverIdx >= 0) status(`${GROUPS_NAME[SHOWN.group[hoverIdx]]}: drag to move · double-click to reset · Ctrl+Z to undo`);
  });
  pc.addEventListener('pointerleave', () => { hoverIdx = -1; });
  const end = () => { dragIdx = -1; pc.style.cursor = sculptMode === 'drag' ? 'crosshair' : 'pointer'; };
  pc.addEventListener('pointerup', end);
  pc.addEventListener('pointercancel', end);
  pc.addEventListener('dblclick', e => {
    const k = nearest(toCanvas(e));
    if (k >= 0) { pushUndo(); E[k * 2] = 0; E[k * 2 + 1] = 0; status('Landmark reset.'); }
  });
}

/* ------------------------------------------------------------------ */
/* UI wiring                                                           */
/* ------------------------------------------------------------------ */
const PREF_KEY = 'trill-puppet-prefs-v3';
function savePrefs() {
  const o = {};
  document.querySelectorAll('input[id],select[id]').forEach(el => {
    if (el.type === 'file') return;
    o[el.id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  try { localStorage.setItem(PREF_KEY, JSON.stringify(o)); } catch (e) { /* private mode */ }
}
function loadPrefs() {
  try {
    const o = JSON.parse(localStorage.getItem(PREF_KEY) || '{}');
    for (const k in o) {
      const el = $(k);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = !!o[k]; else el.value = o[k];
    }
  } catch (e) { /* ignore */ }
}

function onSettingChanged(id, live) {
  if (id === 'da' && rig && st) { st.dragField = buildDragField(rig, num('da', 1)); return; }
  if ((id === 'plate' || id === 'seamAO') && rig) { buildPlate(pc.width, pc.height); return; }
  if ((id === 'depth' || id === 'focal' || id === 'detBlend') && !live && rig) { rebuildRig(); status('Head depth model rebuilt.'); return; }
}

function setMicUI(on, label) {
  const b = $('micBtn');
  if (!b) return;
  b.classList.toggle('on', on);
  b.textContent = on ? `Stop ${label || 'audio'}` : 'Drive lips from microphone';
  const p = $('audioPanel'); if (p) p.classList.toggle('live', on);
}

async function toggleMic() {
  if (audioOn && audio.mode === 'mic') { audio.stop(); audioOn = false; viseme = null; setMicUI(false); status('Audio-driven speech off.'); return; }
  try {
    await audio.useMicrophone();
    audioOn = true; setMicUI(true, 'microphone');
    status('Microphone is driving the mouth: jaw from F1, spread from F2, rounding from a low F2, tongue from fricatives and bursts, brows from accents. Speak normally.');
  } catch (e) { status('Microphone unavailable: ' + (e.message || e)); }
}

function toggleRecord() {
  if (!rig) { status('Load an image first.'); return; }
  if (!recorder) {
    recorder = new Recorder(pc);
    recorder.onStop = (blob, ext, nFrames, secs) => {
      download(blob, `trill-face-puppet-${Date.now()}.${ext}`);
      $('recordBtn').classList.remove('recording');
      $('recordBtn').textContent = 'Record video';
      status(`Video saved: ${ext.toUpperCase()}, ${nFrames} frames, ${secs.toFixed(1)} s at ${pc.width}×${pc.height}.`);
    };
    recorder.onError = msg => {
      status('Recording failed: ' + msg);
      $('recordBtn').classList.remove('recording');
      $('recordBtn').textContent = 'Record video';
    };
  }
  if (recorder.active) { recorder.stop(); return; }
  const track = chk('recAudio', true) ? audio.audioTrack() : null;
  const ok = recorder.start({ audioTrack: track, fps: 30, bitrate: num('recBitrate', 8) * 1e6 });
  if (ok) {
    $('recordBtn').classList.add('recording');
    $('recordBtn').textContent = 'Stop recording';
    status(track ? 'Recording the rendered puppet with microphone audio… press Stop (or V) to save a WebM.' : 'Recording the rendered puppet… press Stop (or V) to save a WebM.');
  }
}

async function loadDemoPortrait() {
  try {
    const im = new Image();
    im.crossOrigin = 'anonymous';
    await new Promise((res, rej) => { im.onload = res; im.onerror = () => rej(new Error('not found')); im.src = 'demo-portrait.jpg'; });
    setImageFrom(im, im.naturalWidth, im.naturalHeight);
  } catch (e) { status('The bundled demo portrait could not be loaded (' + (e.message || e) + ').'); }
}

function wireUI() {
  document.querySelectorAll('input[id],select[id]').forEach(el => {
    if (el.type === 'file') return;
    el.addEventListener('change', () => { savePrefs(); onSettingChanged(el.id, false); });
    if (el.type === 'range') el.addEventListener('input', () => onSettingChanged(el.id, true));
  });
  $('camBtn').onclick = () => startCamera();
  $('file').onchange = e => {
    const f = e.target.files[0];
    if (!f) return;
    if (!modelsReady) { status('The face model is still loading - try again in a second.'); return; }
    const im = new Image();
    im.onload = () => { setImageFrom(im, im.naturalWidth, im.naturalHeight); URL.revokeObjectURL(im.src); };
    im.onerror = () => status('That file could not be decoded as an image.');
    im.src = URL.createObjectURL(f);
  };
  $('snapBtn').onclick = () => {
    if (vid.videoWidth) setImageFrom(vid, vid.videoWidth, vid.videoHeight);
    else status('Start the webcam first.');
  };
  $('demoBtn').onclick = () => loadDemoPortrait();
  $('photoBtn').onclick = async () => {
    if (!rig) { status('Load an image first.'); return; }
    photoFlash = 0.85;
    const ok = await saveCanvasPNG(pc, `trill-face-puppet-${Date.now()}.png`);
    status(ok ? `Photo saved at ${pc.width}×${pc.height}.` : 'The browser refused to export the canvas.');
  };
  $('recordBtn').onclick = () => toggleRecord();
  $('micBtn').onclick = () => toggleMic();
  $('afile').onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;
    const el = $('audioEl');
    el.src = URL.createObjectURL(f);
    el.loop = true;
    try {
      await audio.useElement(el);
      audioOn = true; setMicUI(true, 'audio file');
      status('Audio file is driving the mouth: visemes come from the measured formant bands.');
    } catch (err) { status('Could not decode that audio file: ' + (err.message || err)); }
  };
  $('calBtn').onclick = () => { setNeutral(); status('Neutral pose captured. Relax your face and press it again any time to redo it.'); };
  $('resetBtn').onclick = () => { pushUndo(); if (E) E.fill(0); status('All drags reset.'); };
  $('undoBtn').onclick = undo;
  $('minCamBtn').onclick = () => {
    minimized = $('webcamCard').classList.toggle('minimized');
    $('minCamBtn').textContent = minimized ? 'Expand' : 'Minimize';
    status(minimized ? 'Webcam minimized - tracking keeps running at full speed, and the preview redraw stops.' : 'Webcam preview expanded.');
  };
  const tools = $('sculptTools');
  if (tools) tools.onclick = e => {
    const b = e.target.closest('button[data-sculpt]');
    if (!b) return;
    sculptMode = b.dataset.sculpt;
    Array.from(tools.querySelectorAll('button')).forEach(x => x.classList.toggle('on', x === b));
    pc.style.cursor = sculptMode === 'drag' ? 'crosshair' : 'pointer';
    status(sculptMode === 'drag' ? 'Drag mode: pull any dot to sculpt the photo.' : `${b.textContent} mode: click the face near the mouth to apply it.`);
  };
  const pbox = $('presets');
  if (pbox) {
    pbox.innerHTML = '';
    ['Live webcam', ...Object.keys(PRESETS)].forEach(nm => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = nm;
      if (nm === 'Live webcam') b.classList.add('on');
      b.onclick = () => {
        preset = nm === 'Live webcam' ? 'live' : nm;
        Array.from(pbox.children).forEach(c => c.classList.toggle('on', c === b));
        status(preset === 'live' ? 'Live webcam expression.' : `Preset "${nm}" - ${chk('pmix', false) ? 'mixed with' : 'replacing'} the webcam face.`);
      };
      pbox.appendChild(b);
    });
  }
  $('selfTest').onclick = () => runSelfTest();
  addEventListener('keydown', e => {
    if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.key === 't') tkey = true;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
    if (e.ctrlKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if (k === 'r') { pushUndo(); if (E) E.fill(0); status('All drags reset.'); }
    else if (k === 'p') $('photoBtn').click();
    else if (k === 'v') toggleRecord();
    else if (k === 'm') $('minCamBtn').click();
    else if (k === 'l') { const c = $('showPts'); c.checked = !c.checked; savePrefs(); }
    else if (k === 'c') $('camBtn').disabled ? status('Webcam already running.') : startCamera();
  });
  addEventListener('keyup', e => { if (e.key === 't') tkey = false; });
  addEventListener('beforeunload', () => { if (recorder && recorder.active) recorder.stop(); });
}

/* ------------------------------------------------------------------ */
/* self test (no camera, no network needed)                            */
/* ------------------------------------------------------------------ */
export function runSelfTest() {
  const out = { ok: true, steps: [] };
  const step = (name, fn) => {
    const t0 = performance.now();
    try {
      const value = fn();
      out.steps.push({ name, ms: +(performance.now() - t0).toFixed(2), value });
    } catch (e) {
      out.ok = false;
      out.steps.push({ name, ms: +(performance.now() - t0).toFixed(2), error: e.message || String(e) });
    }
  };
  const { V, W, H } = syntheticFace({ W: 720, H: 860 });
  let lr = null, ls = null;
  step('synthetic face', () => V.length);
  step('build rig', () => {
    lr = buildRig(V, W, H, (ML.FaceLandmarker && ML.FaceLandmarker.FACE_LANDMARKS_TESSELATION) || knnTess(V));
    ls = createPoseState(lr);
    ls.dragField = buildDragField(lr, 1);
    return { vertices: lr.n, triangles: lr.tri.count, edges: lr.edges.count };
  });
  step('rest projection is identity', () => {
    poseFrame(lr, ls, { R: null, expr: null, preset: null, drag: null, body: null, stretch: 0 });
    let worst = 0;
    for (let i = 0; i < lr.n; i++) worst = Math.max(worst, Math.hypot(ls.P[i * 2] - lr.V[i][0], ls.P[i * 2 + 1] - lr.V[i][1]));
    if (worst > 0.05) throw new Error('rest drift ' + worst.toFixed(4) + ' px');
    return +worst.toFixed(6);
  });
  step('yaw compresses the silhouette (perspective, not shear)', () => {
    const yaw = 40 * Math.PI / 180;
    const width = () => { let a = Infinity, b = -Infinity; for (const i of OVAL) { a = Math.min(a, ls.P[i * 2]); b = Math.max(b, ls.P[i * 2]); } return b - a; };
    poseFrame(lr, ls, { R: null, stretch: 1.6 });
    const rest = width();
    poseFrame(lr, ls, { R: RFromEuler(yaw, 0, 0), stretch: 1.6 });
    const turned = width();
    if (turned >= rest) throw new Error('the head got wider while turning: ' + turned.toFixed(1) + ' >= ' + rest.toFixed(1));
    return { restWidth: +rest.toFixed(1), turnedWidth: +turned.toFixed(1), ratio: +(turned / rest).toFixed(3), orthographic: +Math.cos(yaw).toFixed(3) };
  });
  step('stretch limiter cuts elongation', () => {
    // Perspective alone magnifies the near side of a turned head by 1.5-1.6x, so
    // the limiter cannot pull the mesh back to 1.0; its job is to remove the
    // extreme smears (hair shells used to reach 11x). Compare, do not expect.
    poseFrame(lr, ls, { R: RFromEuler(0.9, 0.3, 0.2), stretch: 0, relaxIters: 0 });
    const unlimited = measureStretch(lr, ls);
    poseFrame(lr, ls, { R: RFromEuler(0.9, 0.3, 0.2), stretch: 1.45, relaxIters: 4 });
    const limited = measureStretch(lr, ls);
    if (limited >= unlimited) throw new Error('the limiter made it worse: ' + limited.toFixed(2) + ' >= ' + unlimited.toFixed(2));
    if (limited > 2.2) throw new Error('stretch still ' + limited.toFixed(2));
    if (Math.abs(limited - ls.lastStretch) > 1e-6) throw new Error('HUD stretch disagrees with the measurement');
    return { unlimited: +unlimited.toFixed(3), limited: +limited.toFixed(3), saved: Math.round((1 - limited / unlimited) * 100) + '%' };
  });
  step('every projected point finite', () => {
    for (let i = 0; i < ls.P.length; i++) if (!Number.isFinite(ls.P[i])) throw new Error('non-finite at ' + i);
    return ls.P.length / 2;
  });
  step('jaw hinge opens downward', () => {
    poseFrame(lr, ls, { R: null, jawAngle: -0.3, stretch: 0 });
    const down = ls.P[152 * 2 + 1] - lr.V[152][1];
    if (down <= 0) throw new Error('the chin moved up by ' + down.toFixed(2));
    return +down.toFixed(2);
  });
  step('dental arch is stable under landmark jitter', () => {
    const D = lr.V.map(p => ({ x: p[0], y: p[1] }));
    let prev = null, worst = 0;
    for (let k = 0; k < 40; k++) {
      const j = D.map((p, i) => (LIP.includes(i) ? { x: p.x + Math.sin(k * 3 + i) * 1.8, y: p.y + Math.cos(k * 2.3 + i) * 1.8 } : p));
      const a = fitArch(j, lr.Z, prev, 1 / 60, { tau: 0.055 });
      if (prev) worst = Math.max(worst, Math.hypot(a.mid.x - prev.mid.x, a.mid.y - prev.mid.y));
      prev = a;
    }
    if (worst > 1.8) throw new Error('arch jitter ' + worst.toFixed(2) + ' px');
    return +worst.toFixed(3);
  });
  step('audio visemes are discriminative', () => {
    // Six band RMS values in the order BANDS declares them (F0, F1, F2, F2-high,
    // F3, air). Each phoneme gets its own smoother state and is run to steady
    // state, exactly like src/puppet/audio.test.js does.
    const mk = bands => {
      const s2 = {};
      let v = null;
      for (let i = 0; i < 8; i++) v = bandsToViseme({ bands, total: bands.reduce((a, b) => a + b, 0) }, s2, 1 / 60, {});
      return v;
    };
    const floor = new Array(BANDS.length).fill(Math.pow(10, -95 / 20));  // analyser noise floor
    const a = mk([0.10, 0.55, 0.12, 0.05, 0.02, 0.01]);   // open vowel  /a/
    const e = mk([0.15, 0.12, 0.50, 0.30, 0.05, 0.02]);   // front vowel /i e/
    const s = mk([0.03, 0.08, 0.15, 0.12, 0.50, 0.31]);   // sibilant    /s/
    const p = mk([0.35, 0.30, 0.05, 0.02, 0.02, 0.01]);   // rounded     /o u/
    const q = mk(floor);                                  // silence
    if (a.jaw < 0.5) throw new Error('open vowel did not open the jaw: ' + a.jaw.toFixed(3));
    if (e.spread < a.spread + 0.2) throw new Error('front vowel did not spread: ' + e.spread.toFixed(3));
    if (e.jaw > a.jaw * 0.5) throw new Error('front vowel opened the jaw like /a/: ' + e.jaw.toFixed(3));
    if (s.fricative < 0.5) throw new Error('fricative not detected: ' + s.fricative.toFixed(3));
    if (p.pucker < 0.2) throw new Error('rounded vowel did not pucker: ' + p.pucker.toFixed(3));
    if (q.jaw > 0.02 || q.name !== 'silence') throw new Error('noise floor drove the mouth: ' + q.name + ' ' + q.jaw.toFixed(3));
    const names = new Set([a.name, e.name, s.name, p.name]);
    if (names.size !== 4) throw new Error('visemes collapsed onto each other: ' + [...names].join(', '));
    return { a: a.name, e: e.name, s: s.name, p: p.name, silence: q.name };
  });
  step('plate inpainting converges', () => {
    const w = 64, h = 64;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) { data[i * 4] = 90; data[i * 4 + 1] = 120; data[i * 4 + 2] = 160; data[i * 4 + 3] = 255; }
    const mask = new Uint8Array(w * h);
    for (let y = 20; y < 44; y++) for (let x = 20; x < 44; x++) mask[y * w + x] = 1;
    const res = inpaintPlate({ data, width: w, height: h }, mask, { levels: 3, iters: 12 });
    const i = (32 * w + 32) * 4;
    const err = Math.abs(res.data[i] - 90) + Math.abs(res.data[i + 1] - 120) + Math.abs(res.data[i + 2] - 160);
    if (err > 24) throw new Error('inpaint drift ' + err);
    return { centreError: err };
  });
  step('depth sign detection', () => {
    const Vz = V.map(p => [p[0], p[1], -Math.hypot(p[0] - V[1][0], p[1] - V[1][1]) * 0.001]);
    const s = detectDepthSign(Vz);
    return s;
  });
  const el = $('selfTestOut');
  if (el) {
    el.textContent = out.steps.map(s => s.error ? `✗ ${s.name}: ${s.error}` : `✓ ${s.name} — ${s.ms} ms ${s.value !== undefined ? JSON.stringify(s.value) : ''}`).join('\n');
    el.hidden = false;
  }
  const total = out.steps.reduce((a, b) => a + (b.ms || 0), 0);
  status(out.ok ? `Pipeline self-test passed: ${out.steps.length} stages in ${total.toFixed(1)} ms.` : 'Self-test found problems - see the debug panel output.');
  return out;
}
function knnTess(V, k = 6) {
  const edges = [];
  for (let i = 0; i < 468; i++) {
    const d = [];
    for (let j = 0; j < 468; j++) { if (j !== i) d.push([Math.hypot(V[j][0] - V[i][0], V[j][1] - V[i][1]), j]); }
    d.sort((a, b) => a[0] - b[0]);
    for (let q = 0; q < k; q++) edges.push({ start: i, end: d[q][1] });
  }
  return edges;
}

/* ------------------------------------------------------------------ */
/* boot                                                                */
/* ------------------------------------------------------------------ */
export async function boot() {
  buildTextures();
  loadPrefs();
  wireUI();
  wireCanvas();
  sculptMode = 'drag';
  const ok = await initModels();
  loop();
  if (ok && chk('autoDemo', true)) loadDemoPortrait();
  return ok;
}

if (typeof window !== 'undefined') {
  window.__puppet = {
    get rig() { return rig; }, get st() { return st; }, get modelsReady() { return modelsReady; },
    get state() { return { camP, neutral, preset, viseme, poseNow, gates: GATE, sm: SM }; },
    selfTest: runSelfTest, setImageFrom, rebuildRig, startCamera, initModels, PRESETS,
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}
