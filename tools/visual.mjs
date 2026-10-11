#!/usr/bin/env node
/* =====================================================================
   FaceMirror · visual harness  (optional)

   The headless harness in tools/harness.mjs asserts numbers; this one
   *looks* at the result.  It paints a synthetic portrait, runs the real
   puppet page in Node against a real 2D canvas, drives it through the
   real webcam-inference path and writes a PNG contact sheet — so the
   mouth, the teeth, the turn and the shading can be reviewed without a
   browser.

   usage:  node tools/visual.mjs [out.png] [scenario]
           scenarios: portrait | mouth | turn | all
   flags:  MOUTH_OFF=1 | SHELL_OFF=1 | TONGUE_OFF=1 | ARCH_OFF=1
           (each disables one pass, to isolate an artifact)

   Needs @napi-rs/canvas (`npm i -D @napi-rs/canvas`); the module is
   optional and is not required by `npm test`.
   ===================================================================== */
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(here, '..');
const args = process.argv.slice(2).filter(a => !a.startsWith('-'));
const outFile = args[0] || join(rootDir, '.visual', 'sheet.png');
const scenario = args[1] || 'mouth';

let canvasMod, Delaunator, mockdom, synth;
try {
  canvasMod = await import('@napi-rs/canvas');
} catch (e) {
  console.error('visual harness needs @napi-rs/canvas:  npm i -D @napi-rs/canvas');
  process.exit(2);
}
Delaunator = (await import('delaunator')).default;
const { parseHTMLIds } = await import('./htmlids.mjs');
synth = await import('./synthface.mjs');
const { createCanvas } = canvasMod;

const W = 640, H = 480, FACE_PX = 256, CX = W / 2, CY = 0.48 * H;
const u = v => v * FACE_PX;

/* =====================================================================
   1. a synthetic portrait to drive the puppet with
   ===================================================================== */
function drawPortrait(g) {
  const grd = g.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, '#3a3540'); grd.addColorStop(1, '#20242c');
  g.fillStyle = grd; g.fillRect(0, 0, W, H);
  g.fillStyle = '#2b3340';                                      // shoulders
  g.beginPath(); g.ellipse(CX, H * 1.35, W * 0.62, H * 0.62, 0, 0, 7); g.fill();
  g.fillStyle = '#2a1c18';                                      // hair
  g.beginPath(); g.ellipse(CX, CY - u(0.10), u(0.56), u(0.66), 0, 0, 7); g.fill();
  const skin = g.createRadialGradient(CX - u(0.10), CY - u(0.12), u(0.06), CX, CY + u(0.05), u(0.70));
  skin.addColorStop(0, '#f2cdb0'); skin.addColorStop(.62, '#e6b795'); skin.addColorStop(1, '#c89877');
  g.fillStyle = skin;                                           // face
  g.beginPath(); g.ellipse(CX, CY + u(0.02), u(0.50), u(0.635), 0, 0, 7); g.fill();
  for (const s of [-1, 1]) {                                    // cheeks
    const rg = g.createRadialGradient(CX + s * u(0.30), CY + u(0.16), u(0.01), CX + s * u(0.30), CY + u(0.16), u(0.16));
    rg.addColorStop(0, 'rgba(220,130,110,.30)'); rg.addColorStop(1, 'rgba(220,130,110,0)');
    g.fillStyle = rg; g.beginPath(); g.arc(CX + s * u(0.30), CY + u(0.16), u(0.16), 0, 7); g.fill();
  }
  for (const s of [-1, 1]) {                                    // eyes
    const ex = CX + s * u(0.205), ey = CY - u(0.145);
    g.fillStyle = '#efe4dd';
    g.beginPath(); g.ellipse(ex, ey, u(0.088), u(0.033), 0, 0, 7); g.fill();
    const ig = g.createRadialGradient(ex, ey - u(0.004), u(0.004), ex, ey, u(0.032));
    ig.addColorStop(0, '#6b4a2c'); ig.addColorStop(.7, '#4a3220'); ig.addColorStop(1, '#241a12');
    g.fillStyle = ig; g.beginPath(); g.arc(ex, ey, u(0.031), 0, 7); g.fill();
    g.fillStyle = '#120c08'; g.beginPath(); g.arc(ex, ey, u(0.013), 0, 7); g.fill();
    g.fillStyle = 'rgba(255,255,255,.75)'; g.beginPath(); g.arc(ex - u(0.010), ey - u(0.010), u(0.006), 0, 7); g.fill();
    g.strokeStyle = 'rgba(50,30,22,.65)'; g.lineWidth = u(0.008); g.lineCap = 'round';
    g.beginPath(); g.moveTo(ex - u(0.088), ey - u(0.004));
    g.quadraticCurveTo(ex, ey - u(0.050), ex + u(0.088), ey - u(0.004)); g.stroke();
    g.beginPath(); g.moveTo(ex - u(0.086), ey + u(0.018));
    g.quadraticCurveTo(ex, ey + u(0.046), ex + u(0.086), ey + u(0.020)); g.stroke();
  }
  g.strokeStyle = '#4a3327'; g.lineWidth = u(0.022); g.lineCap = 'round';   // brows
  for (const s of [-1, 1]) {
    g.beginPath();
    g.moveTo(CX + s * u(0.07), CY - u(0.245));
    g.quadraticCurveTo(CX + s * u(0.21), CY - u(0.285), CX + s * u(0.36), CY - u(0.243));
    g.stroke();
  }
  g.strokeStyle = 'rgba(170,110,85,.55)'; g.lineWidth = u(0.012);           // nose
  for (const s of [-1, 1]) {
    g.beginPath(); g.moveTo(CX + s * u(0.030), CY - u(0.05));
    g.quadraticCurveTo(CX + s * u(0.055), CY + u(0.045), CX + s * u(0.040), CY + u(0.062)); g.stroke();
  }
  g.fillStyle = 'rgba(90,50,40,.55)';
  for (const s of [-1, 1]) { g.beginPath(); g.ellipse(CX + s * u(0.040), CY + u(0.068), u(0.011), u(0.007), 0, 0, 7); g.fill(); }
  const mouthY = CY + u(0.285);                                              // lips
  /* NOLIP=1 paints the portrait *without* lips: then everything inside the
     mouth — rim, mucosa, teeth, tongue, throat — is the puppet's own work,
     which is the only way to judge the mouth pass on its own merits      */
  const lipPath = (rx, ry) => {
    g.beginPath();
    g.moveTo(CX - u(rx), mouthY);
    g.quadraticCurveTo(CX - u(rx * 0.55), mouthY - u(ry), CX, mouthY - u(ry * 0.62));
    g.quadraticCurveTo(CX + u(rx * 0.55), mouthY - u(ry), CX + u(rx), mouthY);
    g.quadraticCurveTo(CX + u(rx * 0.55), mouthY + u(ry), CX, mouthY + u(ry * 0.55));
    g.quadraticCurveTo(CX - u(rx * 0.55), mouthY + u(ry), CX - u(rx), mouthY);
    g.closePath();
  };
  if (!process.env.NOLIP) {
    const lg = g.createLinearGradient(0, mouthY - u(0.05), 0, mouthY + u(0.06));
    lg.addColorStop(0, '#b4574f'); lg.addColorStop(.45, '#c9655c'); lg.addColorStop(1, '#a94b46');
    g.fillStyle = lg; lipPath(0.155, 0.058); g.fill();
    g.strokeStyle = 'rgba(70,28,26,.85)'; g.lineWidth = u(0.006);
    g.beginPath(); g.moveTo(CX - u(0.152), mouthY);
    g.quadraticCurveTo(CX, mouthY - u(0.010), CX + u(0.152), mouthY); g.stroke();
    g.fillStyle = 'rgba(255,225,215,.20)';
    g.beginPath(); g.ellipse(CX, mouthY - u(0.032), u(0.055), u(0.011), 0, 0, 7); g.fill();
    g.fillStyle = 'rgba(255,235,230,.30)';
    g.beginPath(); g.ellipse(CX, mouthY + u(0.030), u(0.045), u(0.011), 0, 0, 7); g.fill();
  }
}

/* =====================================================================
   2. the driver: a synthetic face with a movable mouth
   ===================================================================== */
export function driveFace(P, o = {}) {
  const Q = P.map(p => [p[0], p[1], p[2]]);
  const open = o.open || 0, smile = o.smile || 0, pucker = o.pucker || 0;
  const mouthY = 0.285;
  for (let i = 0; i < 478; i++) {
    const [x, y, z] = Q[i];
    const dy = y - mouthY;
    if (dy > -0.02) {
      const w = Math.min(1, (dy + 0.02) / 0.25);
      Q[i][1] = y + open * 0.085 * w;
      Q[i][2] = z - open * 0.02 * w;
    }
    if (y > mouthY - 0.005 && y < mouthY + 0.09) Q[i][1] = y + open * 0.095;
    if (y < mouthY - 0.005 && y > mouthY - 0.08) Q[i][1] = y - open * 0.028;
    if (Math.abs(x) > 0.06) {
      const s = Math.max(0, 1 - Math.abs(dy) / 0.22);
      if (s > 0) {
        Q[i][0] = x + Math.sign(x) * smile * 0.050 * s;
        Q[i][1] = y - smile * 0.034 * s * Math.abs(x) / 0.36;
      }
    }
    if (Math.abs(y - mouthY) < 0.10) Q[i][0] = x * (1 - pucker * 0.30);
  }
  return Q;
}
const blendshapes = o => []
  .concat(o.jaw ? [{ categoryName: 'jawOpen', score: o.jaw }] : [])
  .concat(o.smile ? [{ categoryName: 'mouthSmileLeft', score: o.smile }, { categoryName: 'mouthSmileRight', score: o.smile }] : [])
  .concat(o.pucker ? [{ categoryName: 'mouthPucker', score: o.pucker }] : []);

/* =====================================================================
   3. mock DOM with a real canvas
   ===================================================================== */
function installRealDom(html) {
  const ids = parseHTMLIds(html);
  const classList = () => ({
    _s: new Set(),
    add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
    contains(c) { return this._s.has(c); },
    toggle(c) { const h = this._s.has(c); h ? this._s.delete(c) : this._s.add(c); return !h; }
  });
  const byId = new Map();
  for (const [id, info] of ids) {
    const tag = info.tag.toLowerCase();
    if (tag === 'canvas') {
      const c = createCanvas(parseInt(info.attrs.width, 10) || 640, parseInt(info.attrs.height, 10) || 480);
      Object.assign(c, { id, classList: classList(), style: {}, listeners: {}, parentElement: { clientWidth: 700 } });
      c.addEventListener = () => {}; c.setPointerCapture = () => {};
      byId.set(id, c);
      continue;
    }
    const el = {
      tagName: tag.toUpperCase(), id, attrs: info.attrs, dataset: {},
      value: info.attrs.value !== undefined ? info.attrs.value : '',
      checked: 'checked' in info.attrs, disabled: 'disabled' in info.attrs,
      textContent: '', style: {}, children: [], listeners: {}, classList: classList(), clientWidth: 700,
      parentElement: { clientWidth: 700 },
      addEventListener(k, f) { (this.listeners[k] = this.listeners[k] || []).push(f); },
      appendChild(c) { this.children.push(c); return c; },
      querySelectorAll(sel) {
        const m = /\[([\w-]+)(?:=([^\]]+))?\]/.exec(sel);
        return this.children.filter(c => (m ? (m[2] === undefined ? c.dataset[m[1]] !== undefined : c.dataset[m[1]] === m[2].replace(/["']/g, '')) : true));
      },
      closest() { return null; },
      getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: 480 }; }
    };
    byId.set(id, el);
  }
  const video = createCanvas(W, H);                    // stands in for <video>
  Object.assign(video, { videoWidth: W, videoHeight: H, readyState: 4, currentTime: 0,
    play: async () => {}, addEventListener() {}, listeners: {} });
  byId.set('vid', video);
  let clock = 0;
  const document = {
    readyState: 'complete',
    getElementById: id => byId.get(id) || null,
    createElement: tag => tag === 'canvas'
      ? Object.assign(createCanvas(300, 150), { style: {}, listeners: {}, dataset: {}, addEventListener() {}, classList: classList(), parentElement: { clientWidth: 700 } })
      : {
        tagName: tag.toUpperCase(), style: {}, dataset: {}, children: [], listeners: {}, classList: classList(), textContent: '',
        addEventListener(k, f) { (this.listeners[k] = this.listeners[k] || []).push(f); },
        appendChild(c) { this.children.push(c); return c; },
        querySelectorAll(sel) {
          const m = /\[([\w-]+)(?:=([^\]]+))?\]/.exec(sel);
          return this.children.filter(c => (m ? (m[2] === undefined ? c.dataset[m[1]] !== undefined : c.dataset[m[1]] === m[2].replace(/["']/g, '')) : true));
        },
        closest() { return null; }
      },
    querySelectorAll: () => [],
    addEventListener() {},
    body: {}
  };
  const win = {
    document, console, innerWidth: 1280,
    performance: { now: () => clock },
    requestAnimationFrame: () => 0,
    addEventListener() {}, setTimeout: () => 0, clearTimeout: () => {},
    navigator: { mediaDevices: { getUserMedia: async () => { throw new Error('no camera in the visual harness'); } } }
  };
  const setClock = v => { clock = v; };
  return { document, win, byId, setClock, video };
}

/* =====================================================================
   4. run it
   ===================================================================== */
const html = await readFile(join(rootDir, 'puppet.html'), 'utf8');
const dom = installRealDom(html);
const define = (k, v) => Object.defineProperty(globalThis, k, { value: v, writable: true, configurable: true });
define('window', dom.win); define('document', dom.document);
define('performance', dom.win.performance);
define('requestAnimationFrame', () => 0);
define('addEventListener', () => {});
define('setTimeout', () => 0);
define('Delaunator', Delaunator);
dom.win.window = globalThis;

const photo = createCanvas(W, H);
drawPortrait(photo.getContext('2d'));
const P0 = synth.synthFace();
const tess = synth.synthTessellation(P0);
let DRIVER = { face: P0, categories: [] };
const vision = {
  FilesetResolver: { forVisionTasks: async () => ({}) },
  FaceLandmarker: {
    FACE_LANDMARKS_TESSELATION: tess,
    createFromOptions: async () => ({
      detect: () => ({ faceLandmarks: [synth.toLandmarks(DRIVER.face, W, H, FACE_PX)] }),
      detectForVideo: () => ({ faceLandmarks: [synth.toLandmarks(DRIVER.face, W, H, FACE_PX)], faceBlendshapes: [{ categories: DRIVER.categories }] })
    })
  },
  PoseLandmarker: { createFromOptions: async () => ({ detect: () => ({ landmarks: [] }), detectForVideo: () => ({ landmarks: [] }) }) }
};
/* the app modules are inlined IIFEs: load them with the pass-isolating
   flags honoured, exactly as puppet.html would */
/* `__off` swaps a pass out for a stub when its env flag is set, so one pass
   can be removed from the picture to see what it was contributing */
define('__off', (flag, fn, stub) => (process.env[flag] ? (() => stub) : fn));
const flags = [
  ['SHELL_OFF', 'drawShell(S, lc, st, S.triA,', "__off('SHELL_OFF', drawShell, 0)(S, lc, st, S.triA,"],
  ['TONGUE_OFF', 'drawTongueSurface(S, ctx, C, runs,', "__off('TONGUE_OFF', drawTongueSurface, 0)(S, ctx, C, runs,"],
  ['ARCH_OFF', 'drawArchNew(S, ctx, proj3,', "__off('ARCH_OFF', drawArchNew, { alphaSum: 0, drawn: 0 })(S, ctx, proj3,"],
  ['LIP_OFF', 'drawLipShell(S, ctx, C, D, runs, h, mouthM);', "__off('LIP_OFF', drawLipShell, 0)(S, ctx, C, D, runs, h, mouthM);"],
  ['MOUTH_OFF', 'const mm = drawMouth(S, ctx, st, C, D, mouthM);',
    "const mm = __off('MOUTH_OFF', drawMouth, { h: 0, drawn: 0, upper: 0, lower: 0, seen: 0, tongueOut: 0 })(S, ctx, st, C, D, mouthM);"],
  ['SHADE_OFF', 'lipShading(S, ctx, C, D, runs, h, mouthM);', "__off('SHADE_OFF', lipShading, 0)(S, ctx, C, D, runs, h, mouthM);"],
  ['PLATE_OFF', 'lc.drawImage(S.cache.plate || S.src, 0, 0);',
    "lc.drawImage(process.env.PLATE_OFF ? S.src : (S.cache.plate || S.src), 0, 0);"]
];
for (const rel of ['src/math.js', 'src/puppet/engine.js', 'src/puppet/render.js', 'src/puppet/ui.js']) {
  let code = await readFile(join(rootDir, rel), 'utf8');
  for (const [flag, anchor, repl] of flags) {
    if (process.env[flag] && code.includes(anchor)) code = code.replace(anchor, repl);
  }
  (0, eval)(code);
}
await globalThis.PuppetApp.boot(vision);
const T = globalThis.PuppetApp.__test;
const A = T.state();
for (const id of ['showPts', 'showMesh', 'showOrig']) { const el = dom.byId.get(id); if (el) el.checked = false; }
const camCanvas = createCanvas(W, H);
A.cam = camCanvas; A.camOn = true;
T.buildFromLandmarks(synth.toLandmarks(P0, W, H, FACE_PX), photo, tess);
A.camOn = true;

let t = 0;
const step = (o = {}, render = true) => {
  DRIVER = { face: o.face || P0, categories: o.cats || [] };
  dom.video.currentTime += 1 / 30;
  t += 1000 / 60;
  dom.setClock(t);
  return T.frame(t, 1 / 60, render);
};
const settle = (o, n = 75, render = false) => { let s = null; for (let i = 0; i < n; i++) s = step(o, render); return s; };
const poseIt = (yaw, pitch = 0) => {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return P0.map(([x, y, z]) => [cy * x + sy * z, cp * y - sp * (x * 0), -sy * x + cy * z]);
};

const scenarios = {
  mouth: [['closed', {}], ['open 0.35', { open: 0.35, jaw: 0.35 }], ['open 0.6', { open: 0.6, jaw: 0.6 }],
    ['open 1.0', { open: 1, jaw: 1 }], ['smile', { smile: 1, jaw: 0.1 }], ['smile+open', { smile: 0.8, open: 0.7, jaw: 0.7 }],
    ['pucker O', { pucker: 1, jaw: 0.25 }], ['grin + teeth', { smile: 1, open: 0.45, jaw: 0.45, tv: 1 }]],
  turn: [['rest', {}], ['yaw +20°', { yaw: 0.35 }], ['yaw +40°', { yaw: 0.70 }], ['yaw -40°', { yaw: -0.70 }]],
  portrait: [['neutral', {}], ['open', { open: 0.7, jaw: 0.7 }], ['smile', { smile: 1 }], ['yaw', { yaw: 0.5 }]],
  zoom: [['closed', {}], ['open 0.35', { open: 0.35, jaw: 0.35 }], ['open 0.7', { open: 0.7, jaw: 0.7 }],
    ['open 1.0', { open: 1, jaw: 1 }], ['grin', { smile: 1, open: 0.5, jaw: 0.5, tv: 1 }]],
  closeup: [['open 0.7', { open: 0.7, jaw: 0.7 }], ['open 1.0', { open: 1, jaw: 1 }],
    ['grin', { smile: 0.8, open: 0.6, jaw: 0.6, tv: 1 }]],
  bend: [['neutral', {}], ['jaw 1.0', { open: 1, jaw: 1 }], ['smile 1.0', { smile: 1, jaw: 0.2 }],
    ['jaw + yaw', { open: 1, jaw: 1, yaw: 0.5 }]]
};
const shots = [];
for (const [label, o] of scenarios[scenario] || scenarios.mouth) {
  if (o.tv !== undefined) { const el = dom.byId.get('tv'); if (el) el.value = String(o.tv); }
  const opt = { ...o, face: driveFace(o.yaw ? poseIt(o.yaw) : P0, o), cats: blendshapes(o) };
  settle(opt, 80, false);
  step(opt, true);
  const c = createCanvas(W, H);
  c.getContext('2d').drawImage(dom.byId.get('puppet'), 0, 0);
  shots.push({ label, c });
}

const zoomSheet = scenario === 'zoom', closeup = scenario === 'closeup', bendSheet = scenario === 'bend';
if (bendSheet) {
  /* How much photo detail survives the warp?  Laplacian energy over a
     cheek patch, against the source photo the puppet was built from:
     anything well under 100% is the warp paying for it, not the picture. */
  const box = [240, 300, 420, 430];
  const lap = (c, bx) => {
    const d = c.getContext('2d').getImageData(bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]).data;
    const w = bx[2] - bx[0], h = bx[3] - bx[1];
    let sum = 0, n = 0;
    const lum = i => 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const v = 4 * lum(i) - lum(i - 4) - lum(i + 4) - lum(i - w * 4) - lum(i + w * 4);
      sum += v * v; n++;
    }
    return sum / n;
  };
  const srcC = createCanvas(W, H);
  srcC.getContext('2d').drawImage(photo, 0, 0);
  const base = lap(srcC, box);
  console.log(`source photo detail: ${base.toFixed(1)}`);
  console.log('skin detail through the bend:');
  for (const sh of shots) {
    const v = lap(sh.c, box);
    console.log(`  ${sh.label.padEnd(11)} ${v.toFixed(1).padStart(7)}   ${(v / base * 100).toFixed(1)}% of the photo`);
  }
}
const cols = zoomSheet ? 5 : (closeup ? 3 : 4), pad = 6, rows = Math.ceil(shots.length / cols);
const sheet = createCanvas(cols * (W + pad) + pad, rows * (H + pad + 18) + pad);
const sg = sheet.getContext('2d');
sg.fillStyle = '#0b0d12'; sg.fillRect(0, 0, sheet.width, sheet.height);
shots.forEach((s, i) => {
  const x = pad + (i % cols) * (W + pad), y = pad + Math.floor(i / cols) * (H + pad + 18);
  sg.drawImage(s.c, x, y);
  sg.fillStyle = '#5eead4'; sg.font = '14px sans-serif';
  sg.fillText(s.label, x + 4, y + H + 14);
  if (zoomSheet || closeup || bendSheet) {                      // one big crop per frame
    const box = closeup ? [278, 286, 362, 370] : bendSheet ? [230, 200, 450, 470] : [252, 232, 400, 380];
    const zn = closeup ? 5.9 : (bendSheet ? 2.5 : 4);
    sg.imageSmoothingEnabled = true;
    sg.drawImage(s.c, box[0], box[1], box[2] - box[0], box[3] - box[1],
                 x, y, (box[2] - box[0]) * zn, (box[3] - box[1]) * zn);
    return;
  }
  const zx = 280, zy = 195, zw = 220, zh = 170, scale = 1.35;             // mouth, magnified
  const zx2 = x + 330, zy2 = y + 300;
  sg.save();
  sg.beginPath(); sg.rect(zx2, zy2, zw * scale, zh * scale); sg.clip();
  sg.drawImage(s.c, zx, zy, zw, zh, zx2, zy2, zw * scale, zh * scale);
  sg.restore();
  sg.strokeStyle = 'rgba(94,234,212,.5)'; sg.lineWidth = 1; sg.strokeRect(zx2, zy2, zw * scale, zh * scale);
  sg.fillStyle = '#f472b6'; sg.font = '13px sans-serif'; sg.fillText('mouth ×1.35', zx2 + 4, zy2 + 16);
});
await writeFile(outFile, sheet.toBuffer('image/png')).catch(async e => {
  const { mkdir } = await import('node:fs/promises');
  await mkdir(dirname(outFile), { recursive: true });
  await writeFile(outFile, sheet.toBuffer('image/png'));
});
const dbg = A.ses && A.ses.lastDebug;
console.log(`wrote ${outFile} — ${shots.length} frames, ${sheet.width}x${sheet.height}`);
if (dbg) console.log('mouth:', JSON.stringify(dbg.mouth));
