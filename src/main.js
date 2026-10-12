/**
 * main.js — app bootstrap, state, event wiring (spec §1/§12).
 *
 * The rAF loop is a fixed 60 Hz substep solver (§5.4) over five producers:
 * camera / drag / emotion / record-playback / idle — all merged explicitly
 * by rig.js (§5.3). Stage timings feed the perf governor (§8).
 */

import { Rig, CHANNELS, CH, GROUP_OF } from './rig.js';
import { Tracker, MicJaw } from './tracker.js';
import { CalibrationSession } from './calibrate.js';
import { ChannelFilterBank, FixedStepper } from './smoothing.js';
import {
  DeadChannelSolver, transformToCanonical, makeCanonicalBuffer,
  measureNeutral, blendNeutral, LM,
} from './channels.js';
import { EyeSolver } from './eyes.js';
import { PoseTracker } from './pose.js';
import { EmotionController, EmotionReadback, EMOTION_LABELS } from './emotions.js';
import { DragController, CONTROLS, computeJacobian, makeJacobian, assetHash } from './drag.js';
import {
  loadGlbHead, buildProceduralHead, buildImageHead, makeSamplePortraitCanvas,
  MirrorSkin, buildMirrorPatch, proceduralMorphDelta, CONFIG as HEAD_CONFIG,
} from './headAsset.js';
import { evaluateControlAnchorMorphed } from './imageWarp.js';
import { Stage } from './renderer.js';
import { Recorder, buildClipGlb } from './recorder.js';
import { PerfGovernor } from './perf.js';
import { UI } from './ui.js';
import * as THREE from 'three';

/** @type {Record<string, string>} */
const SETTINGS_KEY = {
  ui: 'facemirror.ui.v1',
};

/** ui settings persisted in localStorage (§0.4 offline + prefs). */
function loadUiSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY.ui) ?? '{}');
  } catch {
    return {};
  }
}

function saveUiSettings(s) {
  try {
    localStorage.setItem(SETTINGS_KEY.ui, JSON.stringify(s));
  } catch {
    /* private mode — settings just don't persist */
  }
}

/**
 * Browser download helper.
 * @param {string} name
 * @param {Blob|string|ArrayBuffer} data
 */
function download(name, data) {
  const blob = data instanceof Blob ? data : new Blob([data]);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

const settings = loadUiSettings();

const rig = new Rig();
const pose = new PoseTracker();
const eyes = new EyeSolver({ seed: 0xFACE });
const emotions = new EmotionController(CHANNELS);
const readback = new EmotionReadback(CHANNELS);
const filterBank = new ChannelFilterBank(CHANNELS, GROUP_OF);
const stepper = new FixedStepper();
const deadSolver = new DeadChannelSolver(CHANNELS);
const canon = makeCanonicalBuffer();
const smoothed = new Float32Array(CHANNELS.length);
const ypr = new Float32Array(3);
let deadNeutral = null;

const perf = new PerfGovernor({
  toast: (m) => ui.toast(m),
  log: (m) => console.log(m),
});

const ui = new UI({
  root: document.getElementById('app') ?? document.body,
  cb: {},
});

// --- Canvas / stage --------------------------------------------------------

/** @type {Stage} */
let stage = null;
/** @type {import('./headAsset.js').HeadAsset|null} */
let headAsset = null;
/** @type {import('./headAsset.js').HeadAsset|null} */
let saved3dAsset = null;
/** @type {import('./headAsset.js').HeadAsset|null} */
let imageAsset = null;
/** @type {{name: string, badge: string, thumbUrl: string, detectedWeights: Float32Array|null, hasFace: boolean}|null} */
let imageMeta = null;
let warpScale = 1.0;
let imageTilt = 0.4;
let showImagePoints = true;
let mirrorSkin = null;
let mirrorPatch = null;
let mirrorMode = true;

// --- Drag wiring ------------------------------------------------------------

const drag = new DragController({
  rig,
  pose,
  raycast: (x, y) => raycastControl(x, y),
  ghostTarget: (control, visible) => {
    if (stage && stage.renderer) {
      // Ghost target: small emissive marker over the control point.
      ghost.visible = visible;
      if (visible && control >= 0 && headAsset) {
        const p = controlWorld(control);
        ghost.position.copy(p);
      }
    }
  },
  onEdit: () => {
    const activeCtrl = CONTROLS[drag.active];
    const names = activeCtrl ? activeCtrl.channels : CONTROLS.flatMap((c) => c.channels);
    for (const name of names) {
      const idx = CH[name];
      if (idx === undefined) continue;
      const val = Math.min(Math.max(rig.weightsSource.drag[idx], 0), 1);
      rig.weightsSource.manual[idx] = val;
      ui.setSlider(name, val);
      ui.setPin(name, true);
    }
    if (ghost && ghost.visible && drag.active >= 0 && headAsset) {
      ghost.position.copy(controlWorld(drag.active));
    }
  },
});

// --- Raycast helpers (three injected via stage) -----------------------------

/** @type {import('three').Object3D|null} */
let ghost = null;

/**
 * Screen → nearest control-vertex pick (§6). Controls live in the head's
 * local space at the fixed anchor positions; we project them to screen and
 * pick the nearest within CONFIG.grabRadiusPx.
 * @param {number} x @param {number} y client px
 */
function raycastControl(x, y) {
  if (!stage || !stage.renderer || !headAsset) return null;
  const rect = stage.canvas.getBoundingClientRect();
  const px = x - rect.left;
  const py = y - rect.top;
  let best = -1;
  let bestD = Infinity;
  for (let c = 0; c < CONTROLS.length; c++) {
    const p = controlWorld(c);
    const v = p.clone().project(stage.camera);
    const sx = (v.x * 0.5 + 0.5) * rect.width;
    const sy = (-v.y * 0.5 + 0.5) * rect.height;
    const d = Math.hypot(sx - px, sy - py);
    if (d < bestD) {
      bestD = d;
      best = c;
    }
  }
  const depthCm = 90;
  if (bestD <= 28) {
    return { hit: true, control: best, depthCm, worldX: px, worldY: py, worldZ: 0 };
  }
  // Whole-face rotation drag (§6): grab anywhere not a control point.
  return { hit: true, control: -1, depthCm, worldX: px, worldY: py, worldZ: 0 };
}

/** Control anchors in head-local space (mirrors headAsset mouth/eye layout). */
const CONTROL_ANCHORS = [
  [-0.30, -0.30, 0.66], [0.30, -0.30, 0.66],
  [0, -0.20, 0.74], [0, -0.40, 0.74],
  [-0.14, 0.30, 0.62], [0.14, 0.30, 0.62],
  [-0.42, 0.28, 0.46], [0.42, 0.28, 0.46],
  [0, -0.62, 0.52], [0, 0.02, 0.82],
  [-0.30, 0.10, 0.70], [0.30, 0.10, 0.70],
  [-0.52, -0.22, 0.36], [0.52, -0.22, 0.36],
  [-0.52, -0.05, 0.42], [0.52, -0.05, 0.42],
];

const _ctrlTmp3 = new Float32Array(3);

/**
 * @param {number} c control index
 * @returns {import('three').Vector3}
 */
function controlWorld(c) {
  if (headAsset && headAsset.isImage && headAsset.features) {
    evaluateControlAnchorMorphed(
      c,
      rig.weights,
      CHANNELS,
      headAsset.features,
      _ctrlTmp3,
      headAsset.warpScale ?? 1
    );
    const v = new THREE.Vector3(_ctrlTmp3[0], _ctrlTmp3[1], _ctrlTmp3[2]);
    stage.headNode.localToWorld(v);
    return v;
  }
  const anchors = headAsset?.controlAnchors ?? CONTROL_ANCHORS;
  const a = anchors[c] ?? [0, 0, 0];
  const v = new THREE.Vector3(a[0], a[1], a[2]);
  stage.headNode.localToWorld(v);
  return v;
}

// --- Tracker ----------------------------------------------------------------

const tracker = new Tracker({
  onStatus: (msg, kind) => ui.setStatus(msg, kind ?? 'ok'),
});

/** Mic jaw/viseme producer (§5.6) — default OFF, opt-in from the panel. */
const mic = new MicJaw();

// --- Recorder ---------------------------------------------------------------

const recorder = new Recorder({
  rig,
  onState: (s) => ui.setStatus(s, 'ok'),
});

// --- Calibration ------------------------------------------------------------

/** @type {CalibrationSession|null} */
let wizard = null;
let wizardHolding = false;

function startCalibration() {
  wizard = new CalibrationSession();
  wizard.begin();
  ui.openWizard();
  ui.setStatus('Calibration: follow the prompts', 'ok');
}

function cancelCalibration() {
  wizard?.cancel();
  wizard = null;
  ui.closeWizard();
}

// --- Producer glue ----------------------------------------------------------

/**
 * Fresh detection → producers (cam layer, pose, dead channels, eyes gaze).
 * @param {number} nowMs
 */
function onDetection(nowMs) {
  rig.feedCamera(tracker.weights);
  pose.setFromMatrix(tracker.matrix, nowMs / 1000);
  pose.getYawPitchRoll(ypr);

  if (tracker.landmarks.length >= 478) {
    // Pose-removal FIRST (trap §11.5): canonical space, then geometry.
    const eyeW = Math.hypot(
      tracker.landmarks[LM.eyeAOuter].x - tracker.landmarks[LM.eyeBOuter].x,
      tracker.landmarks[LM.eyeAOuter].y - tracker.landmarks[LM.eyeBOuter].y
    ) * 640;
    transformToCanonical(canon, tracker.landmarks, tracker.matrix, 640, 480, eyeW);

    // Neutral geometry profile: EMA while the face is near-neutral.
    const nearNeutral = tracker.weights[CH.jawOpen] < 0.08
      && tracker.weights[CH.mouthSmileLeft] < 0.1
      && tracker.weights[CH.mouthSmileRight] < 0.1;
    if (nearNeutral) {
      const m = measureNeutral(canon);
      deadNeutral = deadNeutral ? blendNeutral(deadNeutral, m, 0.02) : m;
      deadSolver.setNeutral(deadNeutral);
    }

    // Dead-channel revival (§5.2) into the cam layer.
    if (deadSolver.neutral) {
      const raw = tracker.weights;
      deadSolver.solve(rig.weightsSource.cam, canon, {
        mouthClose: raw[CH.mouthClose] ?? 0,
        jawOpen: raw[CH.jawOpen] ?? 0,
        browDown: ((raw[CH.browDownLeft] ?? 0) + (raw[CH.browDownRight] ?? 0)) / 2,
        mouthUpperUp: ((raw[CH.mouthUpperUpLeft] ?? 0) + (raw[CH.mouthUpperUpRight] ?? 0)) / 2,
        smile: ((raw[CH.mouthSmileLeft] ?? 0) + (raw[CH.mouthSmileRight] ?? 0)) / 2,
        headYaw: ypr[0],
      });
      deadSolver.validateBlink(rig.weightsSource.eyes, ypr[0]);
    }

    // Iris-derived gaze (primary in camera mode — §5.5).
    const irisA = tracker.landmarks[LM.irisA];
    const irisB = tracker.landmarks[LM.irisB];
    const eyeA = tracker.landmarks[LM.eyeAOuter];
    const eyeB = tracker.landmarks[LM.eyeBOuter];
    const innerA = tracker.landmarks[LM.eyeAInner];
    const innerB = tracker.landmarks[LM.eyeBInner];
    const gazeAX = (irisA.x - (eyeA.x + innerA.x) / 2) / Math.abs(eyeA.x - innerA.x);
    const gazeAY = (irisA.y - (eyeA.y + innerA.y) / 2) / Math.abs(eyeA.y - innerA.y || 0.01);
    const gazeBX = (irisB.x - (eyeB.x + innerB.x) / 2) / Math.abs(eyeB.x - innerB.x);
    const gazeBY = (irisB.y - (eyeB.y + innerB.y) / 2) / Math.abs(eyeB.y - innerB.y || 0.01);
    camGazeYaw = ((gazeAX + gazeBX) / 2) * 28;   // ±°-ish scale
    camGazePitch = -((gazeAY + gazeBY) / 2) * 18;
    haveCamGaze = true;
  }

  // Wizard sampling (hold SPACE / hold button to record frames — §4;
  // doubles as the live debug view).
  if (wizard && (wizardHolding || typeof window !== 'undefined' && window.__FACEMIRROR_FAKE)) {
    const r = wizard.pushFrame(tracker.weights, tracker.matrix);
    const phase = wizard.current();
    if (phase) {
      ui.updateWizard(phase.prompt, r.progress, rig.weights,
        `${wizard.phaseIdx + 1}/${wizard.phases.length}`);
    }
    if (r.finished && wizard.done && wizard.result) {
      rig.applyCalibration(wizard.result);
      pose.setNeutralDepth(wizard.result.z0);
      CalibrationSession.save(wizard.result);
      ui.closeWizard();
      ui.toast('Calibration saved');
      wizard = null;
    }
  }
}

let camGazeYaw = 0;
let camGazePitch = 0;
let haveCamGaze = false;

// --- UI callbacks -----------------------------------------------------------

let pipVisible = true;
let skeletonOn = false;
let emotionSolo = null;

ui.cb = {
  onMirrorToggle: () => startCamera(),
  onImageFile: (file) => loadImageFromFile(file),
  onImageSample: () => {
    const sample = makeSamplePortraitCanvas(640, 800);
    installUserImage(sample, 'Sample Portrait');
  },
  onImageToggleMode: () => toggleImageMode(),
  onImageMatchExpression: () => matchImageExpression(),
  onWarpScale: (v) => {
    warpScale = v;
    if (imageAsset && imageAsset.setWarpScale) imageAsset.setWarpScale(v);
    persistUi();
  },
  onImageTilt: (v) => {
    imageTilt = v;
    if (stage) stage.imageTiltScale = v;
    persistUi();
  },
  onImagePointsToggle: (on) => {
    showImagePoints = on;
    if (imageAsset && imageAsset.setControlPointsVisible) {
      imageAsset.setControlPointsVisible(on);
    }
    persistUi();
  },
  onMirrorSkin: () => {
    mirrorMode = !mirrorMode;
    if (mirrorPatch) mirrorPatch.visible = mirrorMode && !!tracker.stream && !headAsset?.isImage;
    if (mirrorSkin) mirrorSkin.enabled = mirrorMode;
    ui.setMirrorSkin(mirrorMode);
    ui.setStatus(mirrorMode ? 'Mirror Skin on' : 'Mirror Skin off', 'ok');
  },
  onFullscreen: () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  },
  onHud: () => {
    perf.hudVisible = !perf.hudVisible;
    ui.setHudVisible(perf.hudVisible);
    if (perf.hudVisible) ui.updateHud(perf.hudData());
  },
  onEmotion: (id) => {
    if (emotionSolo && id !== emotionSolo) return; // solo/lock (§10)
    emotions.setEmotion(id === 'none' ? 'none' : id);
    rig.emotionWeight = emotions.intensity;
    ui.setActiveEmotion(emotions.emotion);
  },
  onEmotionSolo: (id) => {
    emotionSolo = emotionSolo === id ? null : id;
    if (emotionSolo) emotions.setEmotion(id);
    ui.setActiveEmotion(emotions.emotion);
    ui.setStatus(emotionSolo ? `Emotion locked: ${id}` : 'Emotion unlocked', 'ok');
  },
  isSolo: () => emotionSolo !== null,
  onIntensity: (v) => {
    emotions.setIntensity(v);
    rig.emotionWeight = v;
  },
  onSlider: (name, v) => {
    rig.setManual(name, v);
    rig.setPin(name, true);
    ui.setPin(name, true);
  },
  onPin: (name, on) => rig.setPin(name, on),
  onGain: (name, v) => {
    rig.gainOverride[CH[name]] = v;
  },
  onMaster: (key, v) => {
    if (key === 'asym') rig.asym = v;
    else if (key === 'camWeight') rig.camWeight = v;
    persistUi();
  },
  onMotion: (key, v) => {
    if (key === 'blinkRate') eyes.scheduler.lambda = v;
    else if (key === 'globalGain') rig.globalGain = v;
    else if (key === 'responsiveness') filterBank.setBetaScale(v);
    else if (key === 'headLag') pose.neckLagSec = v;
    persistUi();
  },
  onEyeToggle: (key, on) => {
    eyes.toggles[key] = on;
    persistUi();
  },
  onMic: async (on) => {
    if (on) {
      try {
        await mic.start();
        ui.setStatus('Mic jaw on (local only)', 'ok');
      } catch (err) {
        ui.setStatus('Mic unavailable: ' + err, 'warn');
      }
    } else {
      mic.stop();
      ui.setStatus('Mic jaw off', 'ok');
    }
  },
  onRecord: () => recorder.toggleRecord(),
  onPlay: () => recorder.togglePlay(),
  onExport: (kind) => exportClip(kind),
  onCalibrate: () => startCalibration(),
  onWizardHold: (down) => { wizardHolding = down; },
  onWizardCancel: () => cancelCalibration(),
  onReset: () => {
    rig.resetAll();
    for (const n of CHANNELS) {
      ui.setSlider(n, 0);
      ui.setPin(n, false);
    }
    drag.undo.clear();
    ui.setStatus('All edits reset', 'ok');
  },
  onUndo: () => {
    if (drag.undoLast()) ui.setStatus('Undo', 'ok');
  },
  onPip: () => {
    // V: show the PiP; when visible, toggles the landmark skeleton (§10).
    if (!pipVisible) {
      pipVisible = true;
      ui.setPipVisible(true);
    } else {
      skeletonOn = !skeletonOn;
      ui.setStatus('Skeleton ' + (skeletonOn ? 'on' : 'off'), 'debug');
    }
  },
  onEscape: () => {
    drag.end();
    if (wizard) cancelCalibration();
  },
};

function persistUi() {
  saveUiSettings({
    blinkRate: eyes.scheduler.lambda,
    globalGain: rig.globalGain,
    asym: rig.asym,
    camWeight: rig.camWeight,
    headLag: pose.neckLagSec,
    toggles: eyes.toggles,
    mirrorMode,
    warpScale,
    imageTilt,
    showImagePoints,
  });
}

// --- Export -----------------------------------------------------------------

/** Proxy morph deltas for .anim.glb (computed once). */
const proxyPositions = new Float32Array((HEAD_CONFIG.glbProxyW + 1) * (HEAD_CONFIG.glbProxyH + 1) * 3);
const proxyDeltas = CHANNELS.map(() => new Float32Array(proxyPositions.length));
let proxyReady = false;

function buildProxy() {
  if (proxyReady) return;
  const W = HEAD_CONFIG.glbProxyW;
  const H = HEAD_CONFIG.glbProxyH;
  const d = new Float32Array(3);
  let i = 0;
  for (let j = 0; j <= H; j++) {
    for (let k = 0; k <= W; k++) {
      const u = (k / W - 0.5) * 2 * Math.PI;
      const v = (j / H - 0.5) * Math.PI;
      const x = Math.cos(v) * Math.sin(u) * 0.78;
      const y = Math.sin(v);
      const z = Math.cos(v) * Math.cos(u) * 0.86;
      proxyPositions[i * 3] = x;
      proxyPositions[i * 3 + 1] = y;
      proxyPositions[i * 3 + 2] = z;
      for (let c = 0; c < CHANNELS.length; c++) {
        proceduralMorphDelta(CHANNELS[c], x, y, z, d);
        proxyDeltas[c][i * 3] = d[0];
        proxyDeltas[c][i * 3 + 1] = d[1];
        proxyDeltas[c][i * 3 + 2] = d[2];
      }
      i++;
    }
  }
  proxyReady = true;
}

/**
 * @param {'json'|'anim'|'pose'} kind
 */
async function exportClip(kind) {
  try {
    if (kind === 'json') {
      download('facemirror-clip.json', recorder.exportJSON());
      ui.setStatus('Exported facemirror-clip.json', 'ok');
    } else if (kind === 'anim') {
      buildProxy();
      const glb = recorder.exportAnimGlb(proxyPositions, proxyDeltas);
      download('facemirror-clip.anim.glb', glb);
      ui.setStatus('Exported facemirror-clip.anim.glb', 'ok');
    } else if (kind === 'pose') {
      const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
      const exporter = new GLTFExporter();
      if (!headAsset) return;
      const buf = await recorder.exportPoseGlb(exporter, headAsset.root);
      download('facemirror-head.glb', buf);
      ui.setStatus('Exported facemirror-head.glb (morphs baked)', 'ok');
    }
  } catch (err) {
    ui.setStatus('Export failed: ' + err, 'error');
  }
}

// --- Camera / GLB drop ------------------------------------------------------

async function startCamera() {
  const video = ui.videoEl();
  try {
    await tracker.start(video);
    ui.setPipVisible(pipVisible);
    if (mirrorPatch) mirrorPatch.visible = mirrorMode;
    if (mirrorSkin) mirrorSkin.enabled = mirrorMode;
    mirrorPatchPending = true; // attach Mirror Skin on first landmarks (§6)
  } catch {
    // Graceful: full manual mode (§0.5) — sliders/drag/emotions still work.
    ui.setStatus('No camera — manual mode (drag + emotions + recording)', 'warn');
    rig.goIdle();
  }
}

let mirrorPatchPending = false;

/**
 * Drop-in GLB head (§6). Also auto-probes assets/head.glb at boot.
 * @param {ArrayBuffer} buffer
 */
async function installHead(buffer) {
  try {
    const asset = await loadGlbHead(buffer);
    saved3dAsset = asset;
    stage.setHead(asset);
    headAsset = asset;
    rig.hasTeeth = asset.hasTeeth || asset.hasTongue;
    for (const w of asset.warnings) console.warn('[head]', w);
    ui.setStatus(`GLB head loaded (${asset.missingChannels.length} channels missing)`, 'ok');
    if (imageMeta) {
      ui.setImageState({
        active: false,
        hasImage: true,
        name: imageMeta.name,
        badge: imageMeta.badge,
        thumbUrl: imageMeta.thumbUrl,
        canMatch: imageMeta.hasFace,
      });
    }
    installMirrorPatch();
    installDragJacobian(asset);
  } catch (err) {
    ui.setStatus('GLB load failed: ' + err, 'error');
  }
}

/**
 * Load a user-selected image file (from file input, drag-and-drop, or paste),
 * scale to a safe working resolution, detect face landmarks, and build the
 * 2D warpable image mesh.
 *
 * @param {File|Blob} file
 */
async function loadImageFromFile(file) {
  const label = (/** @type {File} */ (file).name) || 'Pasted Image';
  ui.setStatus(`Loading image: ${label}…`, 'ok');
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Could not decode image file'));
      el.src = url;
    });
    const maxDim = 1024;
    const nw = img.naturalWidth || img.width || 640;
    const nh = img.naturalHeight || img.height || 800;
    const scale = Math.min(1, maxDim / Math.max(nw, nh, 1));
    const cw = Math.max(1, Math.round(nw * scale));
    const ch = Math.max(1, Math.round(nh * scale));
    const canvas = document.createElement('canvas');
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.drawImage(img, 0, 0, cw, ch);
    await installUserImage(canvas, label);
  } catch (err) {
    ui.setStatus('Image load failed: ' + (err?.message ?? err), 'error');
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Detect face landmarks on a canvas and install it as the active 2D warped
 * image puppet on the stage.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {string} name
 */
async function installUserImage(canvas, name) {
  ui.setStatus(`Analyzing face in "${name}"…`, 'ok');
  const det = await tracker.detectImage(canvas);
  const id = `${name}:${canvas.width}x${canvas.height}:${Date.now()}`;
  const asset = buildImageHead(canvas, det.landmarks, {
    warpScale,
    showControls: showImagePoints,
    id,
  });
  imageAsset = asset;
  if (stage) {
    stage.imageTiltScale = imageTilt;
    stage.setHead(asset);
  }
  headAsset = asset;
  rig.hasTeeth = true;
  if (mirrorPatch) mirrorPatch.visible = false;
  installDragJacobian(asset);

  let thumbUrl = '';
  try {
    thumbUrl = canvas.toDataURL('image/jpeg', 0.82);
  } catch {
    thumbUrl = '';
  }
  const badge = det.hasFace
    ? '✓ 478 face landmarks detected'
    : 'Centered face warp map';
  imageMeta = {
    name: `${name} (${canvas.width}×${canvas.height})`,
    badge,
    thumbUrl,
    detectedWeights: det.hasFace ? det.weights : null,
    hasFace: det.hasFace,
  };
  ui.setImageState({
    active: true,
    hasImage: true,
    name: imageMeta.name,
    badge: imageMeta.badge,
    thumbUrl: imageMeta.thumbUrl,
    canMatch: imageMeta.hasFace,
  });
  ui.setStatus(
    det.hasFace
      ? `Image ready: ${name} — drag face parts, pick emotions, or use webcam`
      : `Image ready: ${name} (canonical landmarks) — drag or animate`,
    'ok'
  );
}

/**
 * Switch between the user-selected 2D image and the 3D head.
 */
function toggleImageMode() {
  if (!imageAsset || !saved3dAsset || !stage) return;
  const toImage = headAsset !== imageAsset;
  const next = toImage ? imageAsset : saved3dAsset;
  stage.setHead(next);
  headAsset = next;
  rig.hasTeeth = next.hasTeeth || next.hasTongue;
  if (mirrorPatch) {
    mirrorPatch.visible = !toImage && mirrorMode && !!tracker.stream;
  }
  installDragJacobian(next);
  if (imageMeta) {
    ui.setImageState({
      active: toImage,
      hasImage: true,
      name: imageMeta.name,
      badge: imageMeta.badge,
      thumbUrl: imageMeta.thumbUrl,
      canMatch: imageMeta.hasFace,
    });
  }
  ui.setStatus(toImage ? `Switched to 2D image: ${imageMeta?.name ?? ''}` : 'Switched to 3D head', 'ok');
}

/**
 * Copy the blendshape expression detected on the user-selected photo onto
 * the rig's manual sliders.
 */
function matchImageExpression() {
  if (!imageMeta || !imageMeta.detectedWeights) {
    ui.setStatus('No detected photo expression to match', 'warn');
    return;
  }
  const raw = imageMeta.detectedWeights;
  rig.feedCamera(raw);
  for (let i = 0; i < CHANNELS.length; i++) {
    const name = CHANNELS[i];
    if (name === 'eyeBlinkLeft' || name === 'eyeBlinkRight') continue;
    const v = Math.min(Math.max(rig.weightsSource.cam[i], 0), 1);
    rig.setManual(name, v);
    const pin = v > 0.08;
    rig.setPin(name, pin);
    ui.setSlider(name, v);
    ui.setPin(name, pin);
  }
  ui.setStatus('Matched expression from selected photo (press 0 to reset)', 'ok');
}

function installMirrorPatch() {
  if (!headAsset || headAsset.isImage || !tracker.landmarks.length) return;
  if (mirrorPatch) {
    stage.headNode.remove(mirrorPatch);
    mirrorPatch = null;
  }
  if (!mirrorSkin && tracker.video) mirrorSkin = new MirrorSkin(tracker.video);
  const geo = buildMirrorPatch(tracker.landmarks);
  const mat = new THREE.MeshPhysicalMaterial({
    map: mirrorSkin ? mirrorSkin.texture : null,
    roughness: 0.42,
    metalness: 0,
    envMapIntensity: 0.65,
  });
  mirrorPatch = new THREE.Mesh(geo, mat);
  mirrorPatch.visible = mirrorMode;
  headAsset.root.add(mirrorPatch);
  mirrorPatchPending = false;
}

/**
 * Jacobian at load (§6): 52×2 evaluations, cached per asset hash.
 * Procedural heads sample the analytic morph fields; GLB and 2D Image heads
 * evaluate the real morph-target deltas at the mesh vertex nearest each
 * control anchor.
 * @param {import('./headAsset.js').HeadAsset} asset
 */
function installDragJacobian(asset) {
  const hash = assetHash(asset.hash + ':' + CONTROLS.length);
  const cached = localStorage.getItem('facemirror.jacobian.' + hash);
  if (cached) {
    try {
      const arr = Float32Array.from(JSON.parse(cached));
      if (arr.length === CONTROLS.length * 52 * 3) {
        drag.setJacobian(arr);
        return;
      }
    } catch {
      /* recompute below */
    }
  }
  const J = makeJacobian(CONTROLS.length, 52);
  const mesh = asset.meshes[0];
  const posAttr = mesh?.geometry?.attributes?.position;
  const morphAttrs = mesh?.geometry?.morphAttributes?.position ?? [];
  const anchors = asset.controlAnchors ?? CONTROL_ANCHORS;

  // Precompute the mesh vertex nearest each control anchor (GLB / 2D Image path).
  const nearest = anchors.map((a) => {
    if (!posAttr) return -1;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < posAttr.count; i++) {
      const d = (posAttr.getX(i) - a[0]) ** 2 + (posAttr.getY(i) - a[1]) ** 2 + (posAttr.getZ(i) - a[2]) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  });

  computeJacobian(J, 52, (w, out) => {
    for (let c = 0; c < CONTROLS.length; c++) {
      const a = anchors[c];
      if (asset.isProcedural || !posAttr) {
        const d = new Float32Array(3);
        out[c * 3] = a[0];
        out[c * 3 + 1] = a[1];
        out[c * 3 + 2] = a[2];
        for (let k = 0; k < 52; k++) {
          if (w[k] === 0) continue;
          proceduralMorphDelta(CHANNELS[k], a[0], a[1], a[2], d);
          out[c * 3] += d[0] * w[k];
          out[c * 3 + 1] += d[1] * w[k];
          out[c * 3 + 2] += d[2] * w[k];
        }
      } else {
        const vi = nearest[c];
        let x = posAttr.getX(vi);
        let y = posAttr.getY(vi);
        let z = posAttr.getZ(vi);
        for (let k = 0; k < 52; k++) {
          const targets = asset.morphMap[CHANNELS[k]];
          if (!targets || w[k] === 0) continue;
          for (const t of targets) {
            const attr = morphAttrs[t.index] ?? morphAttrs[0];
            if (!attr || attr.getX(vi) === undefined) continue;
            const rel = mesh.geometry.morphTargetsRelative !== false;
            if (rel) {
              x += attr.getX(vi) * w[k];
              y += attr.getY(vi) * w[k];
              z += attr.getZ(vi) * w[k];
            } else {
              x += (attr.getX(vi) - posAttr.getX(vi)) * w[k];
              y += (attr.getY(vi) - posAttr.getY(vi)) * w[k];
              z += (attr.getZ(vi) - posAttr.getZ(vi)) * w[k];
            }
          }
        }
        out[c * 3] = x;
        out[c * 3 + 1] = y;
        out[c * 3 + 2] = z;
      }
    }
  });
  drag.setJacobian(J);
  try {
    localStorage.setItem('facemirror.jacobian.' + hash, JSON.stringify(Array.from(J)));
  } catch {
    /* cache optional */
  }
}

// --- Main loop --------------------------------------------------------------

let lastNow = 0;
let idleActive = false;

/**
 * @param {number} now
 */
function frame(now) {
  requestAnimationFrame(frame);
  const dtMs = lastNow ? now - lastNow : 16.7;
  lastNow = now;
  const t0 = performance.now();

  // Perf governor tier application (§8).
  const tier = perf.settings();
  tracker.detectEveryNth = tier.detectEveryNth;
  if (stage && stage.renderScale !== tier.renderScale) stage.setRenderScale(tier.renderScale);
  if (tier.capFps === 30 && Math.floor(now / 33.3) === Math.floor((now - dtMs) / 33.3)) {
    return; // half-vsync cap at the lowest tier
  }

  // --- capture + track ---
  const tCap = performance.now();
  const fresh = tracker.poll(now, dtMs / 1000);
  perf.stage.capture = performance.now() - tCap;

  const tTrack = performance.now();
  if (fresh && tracker.hasFace) {
    onDetection(now);
    if (idleActive) {
      rig.wake();
      idleActive = false;
      eyes.saccades.pickTarget();
    }
    if (mirrorPatchPending && tracker.landmarks.length >= 478) installMirrorPatch();
  } else if (tracker.faceAbsent(now) && !idleActive) {
    rig.goIdle();
    idleActive = true;
    haveCamGaze = false;
    ui.setStatus('No face — breathing idle (§2)', 'debug');
  }
  perf.stage.track = performance.now() - tTrack;

  // --- solve (fixed 60 Hz substeps §5.4) ---
  const tSolve = performance.now();
  const substeps = stepper.advance(dtMs / 1000);
  const dt = stepper.stepSec;
  if (mic.enabled) mic.feed(rig);
  /** @type {{pupil: number}|null} */
  let lastEyes = null;
  for (let s = 0; s < substeps; s++) {
    emotions.update(dt);
    rig.setMouthPreserve(emotions.mouthPreserve());
    rig.weightsSource.emotion.set(emotions.layer);

    const lidState = eyes.update(dt, {
      headYaw: ypr[0],
      headPitch: ypr[1],
      jawOpen: rig.weightsSource.cam[CH.jawOpen],
      camBlinkL: rig.weightsSource.camRaw[CH.eyeBlinkLeft],
      camBlinkR: rig.weightsSource.camRaw[CH.eyeBlinkRight],
      camGazeYaw: haveCamGaze && !idleActive ? camGazeYaw : null,
      camGazePitch: haveCamGaze && !idleActive ? camGazePitch : null,
      arousal: emotions.intensity * (emotions.emotion !== 'none' ? 1 : 0),
      squint: (rig.weightsSource.cam[CH.eyeSquintLeft] + rig.weightsSource.cam[CH.eyeSquintRight]) / 2,
      convergeDistCm: 90,
    });
    rig.feedEyes(lidState);
    lastEyes = lidState;

    pose.update(dt, now / 1000, idleActive);
    rig.update(dt);
    filterBank.apply(smoothed, rig.weights, dt);
    rig.weights.set(smoothed);
  }
  recorder.update(stepper.stepSec * Math.max(substeps, 1), now / 1000);
  perf.stage.solve = performance.now() - tSolve;

  // --- render ---
  const tRender = performance.now();
  if (stage) {
    pose.getQuaternion(ypr4);
    stage.applyPose(ypr4, pose.pos, dtMs / 1000, idleActive);
    stage.render(rig.weights, rig.hasTeeth ? 1 : 0.65);
    if (headAsset && headAsset.setPupil && lastEyes) headAsset.setPupil(lastEyes.pupil);
    if (headAsset && headAsset.setGaze && lastEyes) {
      headAsset.setGaze(eyes.eyeInHeadYaw, eyes.eyeInHeadPitch, rig.weights);
    }
    if (mirrorSkin && mirrorMode) mirrorSkin.update(now);
  }
  perf.stage.render = performance.now() - tRender;
  perf.frame(dtMs);

  // --- HUD / read-back / PiP ---
  if (perf.hudVisible && Math.floor(now / 250) !== Math.floor((now - dtMs) / 250)) {
    ui.updateHud(perf.hudData());
  }
  if (Math.floor(now / 200) !== Math.floor((now - dtMs) / 200)) {
    const rb = readback.update(rig.weights, 0.2);
    const label = rb.label === 'none' ? 'none' : (EMOTION_LABELS[rb.label] ?? rb.label).toLowerCase();
    ui.setReadback(label.toLowerCase(), rb.score);
  }
  if (pipVisible && Math.floor(now / 100) !== Math.floor((now - dtMs) / 100)) {
    ui.drawSkeleton(tracker.landmarks, skeletonOn && tracker.landmarks.length > 0);
  }
}

const ypr4 = new Float32Array(4);

// --- Boot sequence ----------------------------------------------------------

async function boot() {
  ui.mount();
  stage = new Stage(/** @type {HTMLCanvasElement} */ (ui.root.querySelector('#fm-canvas')));
  const resize = () => {
    const c = stage.canvas;
    c.width = window.innerWidth;
    c.height = window.innerHeight;
    stage.resize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener('resize', resize);
  resize();

  // Ghost drag target marker.
  ghost = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 12, 10),
    new THREE.MeshBasicMaterial({ color: 0x66e0b0, transparent: true, opacity: 0.55, depthTest: false })
  );
  ghost.visible = false;
  stage.scene.add(ghost);

  // Procedural head first — the app must work with zero assets (§6).
  const proc = buildProceduralHead();
  saved3dAsset = proc;
  stage.setHead(proc);
  headAsset = proc;
  rig.hasTeeth = true;
  installDragJacobian(proc);

  // Restore calibration + settings.
  const cal = CalibrationSession.load();
  if (cal) {
    rig.applyCalibration(cal);
    pose.setNeutralDepth(cal.z0);
  }
  const st = loadUiSettings();
  if (st.blinkRate) eyes.scheduler.lambda = st.blinkRate;
  if (st.globalGain) rig.globalGain = st.globalGain;
  if (st.asym !== undefined) rig.asym = st.asym;
  if (st.camWeight !== undefined) rig.camWeight = st.camWeight;
  if (st.headLag !== undefined) pose.neckLagSec = st.headLag;
  if (st.toggles) Object.assign(eyes.toggles, st.toggles);
  if (st.mirrorMode !== undefined) mirrorMode = st.mirrorMode;
  if (st.warpScale !== undefined) {
    warpScale = st.warpScale;
    const wsEl = /** @type {HTMLInputElement|null} */ (ui.root.querySelector('#fm-warpscale'));
    if (wsEl) wsEl.value = String(warpScale);
  }
  if (st.imageTilt !== undefined) {
    imageTilt = st.imageTilt;
    stage.imageTiltScale = imageTilt;
    const itEl = /** @type {HTMLInputElement|null} */ (ui.root.querySelector('#fm-imgtilt'));
    if (itEl) itEl.value = String(imageTilt);
  }
  if (st.showImagePoints !== undefined) {
    showImagePoints = st.showImagePoints;
    const ipEl = /** @type {HTMLInputElement|null} */ (ui.root.querySelector('#fm-imgpts'));
    if (ipEl) ipEl.checked = showImagePoints;
  }

  // Drag + wheel/pinch dolly + Image / GLB drop onto the canvas.
  const canvas = stage.canvas;
  /** @type {Map<number, {x: number, y: number}>} */
  const pointers = new Map();
  let pinchStart = 0;
  const applyDolly = () => {
    stage.camera.position.z = 2.35 * drag.dolly;
  };
  canvas.addEventListener('pointerdown', (e) => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    canvas.setPointerCapture(e.pointerId);
    if (pointers.size === 1) {
      drag.begin(e.clientX, e.clientY);
    } else if (pointers.size === 2) {
      drag.end(); // two fingers = pinch, not drag
      const [a, b] = [...pointers.values()];
      pinchStart = Math.hypot(a.x - b.x, a.y - b.y);
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchStart > 0) {
        drag.dollyBy(-((d - pinchStart) / 120)); // spread = zoom in (§6)
        pinchStart = d;
        applyDolly();
      }
      return;
    }
    drag.move(e.clientX, e.clientY, 0);
  });
  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinchStart = 0;
    if (pointers.size === 0) drag.end();
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    drag.dollyBy(Math.sign(e.deltaY));
    applyDolly();
  }, { passive: false });
  canvas.addEventListener('dragover', (e) => e.preventDefault());
  canvas.addEventListener('drop', async (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const isImg = (file.type && file.type.startsWith('image/'))
      || /\.(png|jpe?g|webp|gif|bmp|svg|avif)$/i.test(file.name ?? '');
    if (isImg) {
      await loadImageFromFile(file);
    } else {
      await installHead(await file.arrayBuffer());
    }
  });
  window.addEventListener('paste', async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type && item.type.startsWith('image/')) {
        const blob = item.getAsFile();
        if (blob) {
          await loadImageFromFile(blob);
          break;
        }
      }
    }
  });

  // Auto-load assets/head.glb when present (optional drop-in — §1/assets).
  try {
    const res = await fetch('./assets/head.glb', { method: 'GET' });
    const ct = res.headers.get('content-type') ?? '';
    if (res.ok && !ct.includes('text/html')) {
      installHead(await res.arrayBuffer());
      ui.setStatus('assets/head.glb loaded', 'ok');
    }
  } catch {
    /* no drop-in head — procedural stays */
  }

  // Mirror skin needs the video element; start camera lazily on user click
  // (permission must be a gesture), but also try once at boot for kiosk use.
  mirrorSkin = new MirrorSkin(ui.videoEl());

  ui.setStatus('Ready — press Select Image 🖼, Mirror (webcam), or drop a photo/GLB on canvas', 'ok');
  perf.selfCheck();
  requestAnimationFrame(frame);
}

boot();
