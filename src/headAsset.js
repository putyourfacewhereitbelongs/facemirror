/**
 * headAsset.js — GLB loader (ARKit-52 morph wiring), local anatomical head,
 * procedural fallback, and webcam "Mirror Skin" mode (spec §6).
 *
 * Trap §11.8: cap jawOpen when the asset has no teeth.
 * Trap §11.11: GLB morph names must be exact-case ARKit names — we auto-
 * correct common variants (_l/_L, snake_case) at load with a warning.
 * Trap §11.3: eye-look channels map to eyeBONE rotations when the asset has
 * eye nodes; the L/R assignment is resolved by calibration, not assumption.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CHANNELS, CH } from './rig.js';
import { assetHash } from './drag.js';
import { MORPH_CONFIG, proceduralMorphDelta } from './headShape.js';
import { buildAnatomicalRig } from './anatomy.js';
import { buildFallbackEars } from './ears.js';
export { proceduralMorphDelta } from './headShape.js';

/** @type {Record<string, any>} */
export const CONFIG = {
  irisTextureSize: 256,
  skullSegmentsW: 56,
  skullSegmentsH: 40,
  skullScale: [0.78, 1.0, 0.86],
  eyeRadius: 0.115,
  eyeOffset: [0.30, 0.10, 0.62],
  lidThickness: 0.012,
  ...MORPH_CONFIG,
  mirrorGridU: 30,
  mirrorGridV: 26,
  mirrorTexSize: 512,
  mirrorFps: 20,          // webcam texture refresh 15–30 fps (§6)
  autoExposeEma: 0.06,
  autoExposeTarget: 0.42,
  cacheKey: 'facemirror.morphcache.v1',
  // eye node name probes (§6): case-insensitive substring matches
  eyeNodeNames: ['lefteye', 'righteye', 'l_eye', 'r_eye', 'l.eyes', 'r.eyes', 'eye_l', 'eye_r', 'eyeleft', 'eyeright'],
  teethNodeNames: ['teeth', 'tooth', 'molar', 'gums'],
  tongueNodeNames: ['tongue'],
  // jawOpen cap when no teeth/tongue geometry exists (trap §11.8)
  jawOpenNoTeeth: 0.65,
};

/**
 * Normalize a GLB morph-target name to exact ARKit casing (trap §11.11).
 * Handles snake_case, kebab, and _l/_L/_r/_R suffixes.
 * @param {string} name
 * @returns {{name: string, corrected: boolean}}
 */
export function normalizeMorphName(name) {
  const exact = new Set(CHANNELS);
  if (exact.has(name)) return { name, corrected: false };
  // snake/kebab → camel
  let camel = name.replace(/[-_]+([a-zA-Z0-9])/g, (_, c) => c.toUpperCase());
  // _l / _R style side suffixes already covered above; handle bare suffix.
  camel = camel.replace(/([a-zA-Z])([lLrR])$/, (m, a, b) => {
    if (m.length === 2 && (b === 'l' || b === 'L' || b === 'r' || b === 'R')) {
      return a + (b === 'l' || b === 'L' ? 'Left' : 'Right');
    }
    return m;
  });
  // Fix lowercase first letter of the shape part if needed
  if (exact.has(camel)) return { name: camel, corrected: true };
  // Case-insensitive last resort
  const lower = camel.toLowerCase();
  for (const c of CHANNELS) {
    if (c.toLowerCase() === lower) return { name: c, corrected: true };
  }
  return { name, corrected: false };
}

/**
 * @typedef {Object} HeadAsset
 * @property {THREE.Object3D} root
 * @property {Array<THREE.Mesh>} meshes meshes with morph targets
 * @property {Record<string, Array<{mesh: THREE.Mesh, index: number}>>} morphMap name → targets
 * @property {Array<string>} warnings missing names / corrections
 * @property {Array<string>} missingChannels ARKit names with no morph
 * @property {{left: THREE.Object3D|null, right: THREE.Object3D|null}} eyeNodes
 * @property {boolean} hasTeeth
 * @property {boolean} hasTongue
 * @property {boolean} isProcedural
 * @property {boolean} [isAnatomical]
 * @property {(weights: Float32Array) => void} [updateLids]
 * @property {string} hash identity for Jacobian cache
 */

/**
 * Load and wire a GLB head (spec §6 asset contract).
 * @param {ArrayBuffer} buffer
 * @returns {Promise<HeadAsset>}
 */
export async function loadGlbHead(buffer) {
  const loader = new GLTFLoader();
  const gltf = await new Promise((resolve, reject) => loader.parse(buffer, '', resolve, reject));
  const root = gltf.scene;
  /** @type {Array<THREE.Mesh>} */
  const meshes = [];
  /** @type {Record<string, Array<{mesh: THREE.Mesh, index: number}>>} */
  const morphMap = {};
  /** @type {Array<string>} */
  const warnings = [];
  const eyeNodes = { left: null, right: null };
  let hasTeeth = false;
  let hasTongue = false;

  root.traverse((obj) => {
    const nameL = obj.name.toLowerCase().replace(/[\s._-]/g, '');
    for (const probe of CONFIG.eyeNodeNames) {
      const p = probe.replace(/[\s._-]/g, '');
      if (nameL.includes(p)) {
        if (/l(eft)?/.test(obj.name) && !eyeNodes.left) eyeNodes.left = obj;
        else if (/r(ight)?/.test(obj.name) && !eyeNodes.right) eyeNodes.right = obj;
      }
    }
    for (const probe of CONFIG.teethNodeNames) if (nameL.includes(probe)) hasTeeth = true;
    for (const probe of CONFIG.tongueNodeNames) if (nameL.includes(probe)) hasTongue = true;
    const mesh = /** @type {THREE.Mesh} */ (obj);
    if (!mesh.isMesh || !mesh.morphTargetDictionary) return;
    meshes.push(mesh);
    for (const [rawName, idx] of Object.entries(mesh.morphTargetDictionary)) {
      const { name, corrected } = normalizeMorphName(rawName);
      if (corrected) warnings.push(`morph "${rawName}" → "${name}" (auto-corrected, trap §11.11)`);
      if (CH[name] === undefined) {
        warnings.push(`morph "${rawName}" is not an ARKit-52 name — ignored`);
        continue;
      }
      (morphMap[name] ??= []).push({ mesh, index: idx });
    }
  });

  const present = new Set(Object.keys(morphMap));
  const missing = CHANNELS.filter((c) => !present.has(c));
  if (missing.length) warnings.push(`${missing.length} channels missing on asset (disabled, no crash): ${missing.join(', ')}`);

  return {
    root,
    meshes,
    morphMap,
    warnings,
    missingChannels: missing,
    eyeNodes,
    hasTeeth,
    hasTongue,
    isProcedural: false,
    hash: assetHash('glb:' + meshes.length + ':' + missing.length),
  };
}

/**
 * A small, fully local anatomical default (CC BY 3.0; attribution in assets/).
 * Expressions are generated at load, keeping the vendored mesh under 400 KB.
 * Custom assets/head.glb and dropped-in ARKit heads still take precedence.
 * @returns {Promise<HeadAsset>}
 */
export async function loadAnatomicalHead() {
  const base = new URL('../assets/anatomical-head/', import.meta.url);
  const textures = new THREE.TextureLoader();
  const [gltf, albedo, normal] = await Promise.all([
    new GLTFLoader().loadAsync(new URL('head.glb', base).href),
    textures.loadAsync(new URL('albedo.jpg', base).href),
    textures.loadAsync(new URL('normal.jpg', base).href),
  ]);
  albedo.colorSpace = THREE.SRGBColorSpace;
  albedo.anisotropy = 2;
  const material = new THREE.MeshPhysicalMaterial({
    map: albedo,
    normalMap: normal,
    normalScale: new THREE.Vector2(0.40, 0.40),
    roughness: 0.56,
    metalness: 0,
    envMapIntensity: 0.65,
  });
  material.userData.skin = true;
  const iris = new THREE.CanvasTexture(makeIrisTexture());
  iris.colorSpace = THREE.SRGBColorSpace;
  let source = null;
  gltf.scene.traverse((obj) => { if (obj.isMesh && !source) source = obj; });
  if (!source) throw new Error('The anatomical head contains no mesh');
  const asset = buildAnatomicalRig(source.geometry, material, iris);
  source.geometry.dispose();
  source.material.dispose();
  return asset;
}

// Pure motion application is shared with headless webcam-transfer tests.
export { applyWeights } from './headMotion.js';

// ---------------------------------------------------------------------------
// Procedural fallback head (§6): "decent, not janky-toy" — UV-sphere skull
// with load-time vertex morphs, real eyeballs with generated iris texture,
// lid shells driven by the same blink math, and Mirror Skin mode.
// ---------------------------------------------------------------------------

/**
 * Generate a 256 px iris + veined sclera canvas texture (§6).
 * @returns {HTMLCanvasElement}
 */
export function makeIrisTexture() {
  const S = CONFIG.irisTextureSize;
  const canvas = document.createElement('canvas');
  canvas.width = S;
  canvas.height = S;
  const g = canvas.getContext('2d');
  // Sclera
  const grad = g.createRadialGradient(S / 2, S / 2, S * 0.1, S / 2, S / 2, S * 0.7);
  grad.addColorStop(0, '#f2ece6');
  grad.addColorStop(1, '#c9b8ae');
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  // Veins
  g.strokeStyle = 'rgba(190, 60, 60, 0.35)';
  for (let i = 0; i < 26; i++) {
    const a = Math.random() * Math.PI * 2;
    const r0 = S * 0.22 + Math.random() * S * 0.22;
    g.beginPath();
    g.moveTo(S / 2 + Math.cos(a) * r0, S / 2 + Math.sin(a) * r0);
    for (let k = 0; k < 3; k++) {
      const aa = a + (Math.random() - 0.5) * 1.2;
      g.lineTo(S / 2 + Math.cos(aa) * (r0 + 12 * (k + 1)), S / 2 + Math.sin(aa) * (r0 + 12 * (k + 1)));
    }
    g.stroke();
  }
  // Iris
  const irisR = S * 0.20;
  const ig = g.createRadialGradient(S / 2, S / 2, irisR * 0.15, S / 2, S / 2, irisR);
  ig.addColorStop(0, '#3b2a18');
  ig.addColorStop(0.55, '#6b4a25');
  ig.addColorStop(1, '#2e2011');
  g.fillStyle = ig;
  g.beginPath();
  g.arc(S / 2, S / 2, irisR, 0, Math.PI * 2);
  g.fill();
  // Iris fibers
  g.strokeStyle = 'rgba(255, 220, 160, 0.22)';
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2 + Math.random() * 0.1;
    g.beginPath();
    g.moveTo(S / 2 + Math.cos(a) * irisR * 0.25, S / 2 + Math.sin(a) * irisR * 0.25);
    g.lineTo(S / 2 + Math.cos(a) * irisR * 0.95, S / 2 + Math.sin(a) * irisR * 0.95);
    g.stroke();
  }
  // Limbal ring (§7: dark limbal ring, eyes never share the skin material)
  g.strokeStyle = 'rgba(20, 14, 8, 0.85)';
  g.lineWidth = S * 0.022;
  g.beginPath();
  g.arc(S / 2, S / 2, irisR * 0.97, 0, Math.PI * 2);
  g.stroke();
  // Pupil (driven by UV scale at runtime)
  g.fillStyle = '#070503';
  g.beginPath();
  g.arc(S / 2, S / 2, irisR * 0.42, 0, Math.PI * 2);
  g.fill();
  // Cornea highlight
  g.fillStyle = 'rgba(255,255,255,0.5)';
  g.beginPath();
  g.ellipse(S * 0.42, S * 0.40, S * 0.05, S * 0.03, -0.6, 0, Math.PI * 2);
  g.fill();
  return canvas;
}

/**
 * Build the procedural fallback head (§6): skull + 52 morphs + eyeballs +
 * lid shells. Returns a full HeadAsset plus eye/lid handles for eyes.js.
 * @returns {HeadAsset & {eyes: {left: THREE.Mesh, right: THREE.Mesh}, lids: {left: THREE.Mesh, right: THREE.Mesh}, eyeTexture: THREE.CanvasTexture, setBlink: (l: number, r: number) => void, setPupil: (s: number) => void}}
 */
export function buildProceduralHead() {
  const geo = new THREE.SphereGeometry(1, CONFIG.skullSegmentsW, CONFIG.skullSegmentsH);
  geo.applyMatrix4(new THREE.Matrix4().makeScale(...CONFIG.skullScale));
  const pos = geo.attributes.position;
  const count = pos.count;
  const morphs = [];
  /** @type {Record<string, Array<{mesh: THREE.Mesh, index: number}>>} */
  const morphMap = {};
  const delta = new Float32Array(3);
  const mesh = new THREE.Mesh(geo, null); // material assigned by renderer.js

  for (const name of CHANNELS) {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      proceduralMorphDelta(name, pos.getX(i), pos.getY(i), pos.getZ(i), delta);
      arr[i * 3] = delta[0];
      arr[i * 3 + 1] = delta[1];
      arr[i * 3 + 2] = delta[2];
    }
    morphs.push(new THREE.Float32BufferAttribute(arr, 3));
    morphMap[name] = [{ mesh, index: morphs.length - 1 }];
  }
  geo.morphAttributes.position = morphs;
  geo.morphTargetsRelative = true;
  mesh.morphTargetInfluences = new Array(morphs.length).fill(0);
  mesh.morphTargetDictionary = Object.fromEntries(CHANNELS.map((n, i) => [n, i]));

  // Eyeballs with generated iris texture (glossy — never the skin material).
  const irisCanvas = makeIrisTexture();
  const irisTex = new THREE.CanvasTexture(irisCanvas);
  irisTex.colorSpace = THREE.SRGBColorSpace;
  const eyeGeo = new THREE.SphereGeometry(CONFIG.eyeRadius, 24, 18);
  const eyeMat = new THREE.MeshPhysicalMaterial({
    map: irisTex,
    roughness: 0.18,
    metalness: 0,
    clearcoat: 1,
    clearcoatRoughness: 0.08,
    envMapIntensity: 1.15, // §7
  });
  const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
  const eyeR = new THREE.Mesh(eyeGeo.clone(), eyeMat);
  eyeL.position.set(-CONFIG.eyeOffset[0], CONFIG.eyeOffset[1], CONFIG.eyeOffset[2]);
  eyeR.position.set(CONFIG.eyeOffset[0], CONFIG.eyeOffset[1], CONFIG.eyeOffset[2]);
  eyeL.rotation.y = -0.12; // slight divergent set for realism
  eyeR.rotation.y = 0.12;

  // Lid shells: partial spheres that rotate with the blink curve (§6).
  const lidGeo = new THREE.SphereGeometry(CONFIG.eyeRadius * 1.13, 20, 10, 0, Math.PI * 2, 0, Math.PI * 0.62);
  const lidMat = new THREE.MeshStandardMaterial({ color: 0xc99e86, roughness: 0.55, metalness: 0 });
  const lidL = new THREE.Mesh(lidGeo, lidMat);
  const lidR = new THREE.Mesh(lidGeo.clone(), lidMat);
  lidL.position.copy(eyeL.position);
  lidR.position.copy(eyeR.position);
  lidL.rotation.x = -0.42;
  lidR.rotation.x = -0.42;

  const root = new THREE.Group();
  const ears = buildFallbackEars();
  root.add(mesh, eyeL, eyeR, lidL, lidR, ears.left, ears.right);
  lidL.userData.skin = lidR.userData.skin = true;

  /** @param {number} l @param {number} r blink 0..1 */
  const setBlink = (l, r) => {
    lidL.rotation.x = -0.42 + l * 1.15;
    lidR.rotation.x = -0.42 + r * 1.15;
  };
  /** @param {number} s pupil scale (iris UV scale about center) */
  const setPupil = (s) => {
    const k = 1 / Math.max(s, 0.2);
    irisTex.repeat.set(k, k);
    irisTex.offset.set((1 - k) / 2, (1 - k) / 2);
    irisTex.needsUpdate = true;
  };

  return {
    root,
    meshes: [mesh],
    morphMap,
    warnings: ['procedural head: all 52 channels synthesized from morph fields'],
    missingChannels: [],
    eyeNodes: { left: eyeL, right: eyeR },
    hasTeeth: true,   // procedural jaw pivots cleanly; no teeth gap to show
    hasTongue: true,
    isProcedural: true,
    hash: assetHash('procedural:v2:' + CHANNELS.length),
    ears,
    eyes: { left: eyeL, right: eyeR },
    lids: { left: lidL, right: lidR },
    eyeTexture: irisTex,
    setBlink,
    setPupil,
  };
}

// ---------------------------------------------------------------------------
// Mirror Skin (§6): project the webcam frame onto the face using the
// 478-landmark UV mapping — per-vertex UVs from landmark interpolation
// (computed once), texture refreshed from `video` at 15–30 fps with an
// auto-expose scalar from mean luma (EMA) so bright rooms don't blow out.
// ---------------------------------------------------------------------------

/**
 * k-nearest-landmark barycentric UV assignment for the face region of the
 * procedural skull. Landmark (x, y) normalized coords ARE the video UVs.
 *
 * @param {Array<{x: number, y: number, z: number}>} landmarks478
 * @returns {THREE.BufferGeometry} face patch with UVs (a new overlay mesh)
 */
export function buildMirrorPatch(landmarks478) {
  const W = CONFIG.mirrorGridU;
  const H = CONFIG.mirrorGridV;
  const verts = [];
  const uvs = [];
  const idx = [];
  // Face region on the skull: u ∈ [−0.62, 0.62], v ∈ [−0.62, 0.42].
  for (let j = 0; j <= H; j++) {
    for (let i = 0; i <= W; i++) {
      const u = (i / W - 0.5) * 1.24;
      const v = 0.42 - (j / H) * 1.04;
      // Project onto the scaled unit sphere front.
      const r2 = Math.max(1 - u * u - v * v * 0.85, 0.02);
      const z = Math.sqrt(r2) * CONFIG.skullScale[2];
      const x = u * CONFIG.skullScale[0];
      const y = v * CONFIG.skullScale[1];
      verts.push(x, y, z * 1.012); // just outside the skull
      // kNN (k=3) inverse-distance UV from the 478 landmarks.
      let uAcc = 0;
      let vAcc = 0;
      let wAcc = 0;
      // Coarse landmark subset keeps this O(grid·subset): every 2nd point.
      for (let li = 0; li < 478; li += 2) {
        const L = landmarks478[li];
        const lx = (L.x - 0.5) * 1.6;
        const ly = (0.5 - L.y) * 1.4;
        const lz = (L.z ?? 0) * 2;
        const d2 = (x - lx) * (x - lx) + (y - ly) * (y - ly) + (z - lz) * (z - lz);
        const w = 1 / (d2 * 40 + 0.012);
        uAcc += L.x * w;
        vAcc += L.y * w;
        wAcc += w;
      }
      uvs.push(uAcc / wAcc, vAcc / wAcc);
    }
  }
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const a = j * (W + 1) + i;
      const b = a + 1;
      const c = a + (W + 1);
      const d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Mirror-skin video texture: manual canvas upload at CONFIG.mirrorFps with
 * auto-exposure from mean luma (§6).
 */
export class MirrorSkin {
  /**
   * @param {HTMLVideoElement} video
   */
  constructor(video) {
    this.video = video;
    this.canvas = document.createElement('canvas');
    this.canvas.width = CONFIG.mirrorTexSize;
    this.canvas.height = CONFIG.mirrorTexSize * 0.75;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.exposure = 1;
    this._ema = CONFIG.autoExposeTarget;
    this._lastUpload = 0;
    this.enabled = true;
  }

  /**
   * Refresh from the video at 15–30 fps (§6), auto-exposing.
   * @param {number} nowMs
   * @returns {boolean} true when a new upload happened
   */
  update(nowMs) {
    if (!this.enabled || !this.video || this.video.readyState < 2) return false;
    const interval = 1000 / CONFIG.mirrorFps;
    if (nowMs - this._lastUpload < interval) return false;
    this._lastUpload = nowMs;
    const g = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    // Mirror the *video*, not the rig (trap §11.9): selfie view.
    g.save();
    g.translate(W, 0);
    g.scale(-1, 1);
    g.filter = `brightness(${this.exposure.toFixed(3)})`;
    g.drawImage(this.video, 0, 0, W, H);
    g.restore();
    // Mean luma EMA → exposure (bright rooms don't blow out).
    try {
      const sample = g.getImageData(0, 0, 16, 16).data;
      let sum = 0;
      for (let i = 0; i < sample.length; i += 4) {
        sum += 0.2126 * sample[i] + 0.7152 * sample[i + 1] + 0.0722 * sample[i + 2];
      }
      const luma = sum / (sample.length / 4) / 255;
      this._ema += (luma - this._ema) * CONFIG.autoExposeEma;
      this.exposure = Math.min(Math.max(CONFIG.autoExposeTarget / Math.max(this._ema, 0.03), 0.55), 1.75);
    } catch {
      // getImageData can throw with cross-origin video — ignore (selfie cam is fine).
    }
    this.texture.needsUpdate = true;
    return true;
  }
}
