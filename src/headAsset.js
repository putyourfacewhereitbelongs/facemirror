/**
 * headAsset.js — GLB loader (ARKit-52 morph wiring) + procedural fallback
 * head + webcam "Mirror Skin" mode (spec §6).
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
import { CONTROLS, computeJacobian, makeJacobian, assetHash } from './drag.js';
import {
  computeImagePlaneSize,
  buildCanonicalImageLandmarks,
  extractImageFeatures,
  faceReliefZ,
  imageMorphDelta,
  evaluateControlAnchorMorphed,
  CONFIG as IMAGE_WARP_CONFIG,
} from './imageWarp.js';

/** @type {Record<string, any>} */
export const CONFIG = {
  irisTextureSize: 256,
  skullSegmentsW: 56,
  skullSegmentsH: 40,
  skullScale: [0.78, 1.0, 0.86],
  eyeRadius: 0.115,
  eyeOffset: [0.30, 0.10, 0.62],
  lidThickness: 0.012,
  jawPivot: [0, -0.18, 0.18],
  jawOpenRad: 0.5,
  morphRadius: 0.34,
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
 * Apply 52 weights to a loaded head (GLB or procedural).
 * Eye-look channels rotate eye bones when present (§6), else morphs apply.
 * @param {HeadAsset} asset
 * @param {Float32Array} weights52
 * @param {number} [jawCap] optional jawOpen cap (no-teeth assets)
 */
export function applyWeights(asset, weights52, jawCap = 1) {
  const warpScale = asset.warpScale ?? 1;
  for (const name of CHANNELS) {
    const targets = asset.morphMap[name];
    if (!targets) continue;
    let w = weights52[CH[name]];
    if (name === 'jawOpen' && w > jawCap) w = jawCap;
    w *= warpScale;
    for (const t of targets) {
      const inf = t.mesh.morphTargetInfluences;
      if (inf) inf[t.index] = w;
    }
  }
  if (asset.updateControlPoints) {
    asset.updateControlPoints(weights52);
  }
  // Eye bones (trap §11.3 naming resolved upstream; look weights are read
  // from the eyeLook* channels).
  const lookYaw = (weights52[CH.eyeLookOutLeft] - weights52[CH.eyeLookInLeft])
    - (weights52[CH.eyeLookOutRight] - weights52[CH.eyeLookInRight]);
  const lookPitch = weights52[CH.eyeLookUpLeft] - weights52[CH.eyeLookDownLeft];
  if (asset.eyeNodes.left) {
    asset.eyeNodes.left.rotation.y = lookYaw * 0.45;
    asset.eyeNodes.left.rotation.x = -lookPitch * 0.45;
  }
  if (asset.eyeNodes.right) {
    asset.eyeNodes.right.rotation.y = lookYaw * 0.45;
    asset.eyeNodes.right.rotation.x = -lookPitch * 0.45;
  }
}

// ---------------------------------------------------------------------------
// Procedural fallback head (§6): "decent, not janky-toy" — UV-sphere skull
// with load-time vertex morphs, real eyeballs with generated iris texture,
// lid shells driven by the same blink math, and Mirror Skin mode.
// ---------------------------------------------------------------------------

/**
 * Gaussian falloff for morph displacement fields.
 * @param {number} dx @param {number} dy @param {number} dz
 * @param {number} radius
 * @returns {number}
 */
function gauss(dx, dy, dz, radius) {
  const d2 = dx * dx + dy * dy + dz * dz;
  const r2 = radius * radius;
  return Math.exp(-d2 / (2 * r2 * 0.55));
}

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
 * Displacement field for one procedural morph, evaluated per vertex.
 * Returns delta [dx,dy,dz] into `out3`.
 * @param {string} name ARKit channel name
 * @param {number} x @param {number} y @param {number} z vertex (head-local)
 * @param {Float32Array} out3
 */
export function proceduralMorphDelta(name, x, y, z, out3) {
  let dx = 0;
  let dy = 0;
  let dz = 0;
  const R = CONFIG.morphRadius;
  const g = (ax, ay, az, r = R) => gauss(x - ax, y - ay, z - az, r);
  const mouthY = -0.30;
  const mouthZ = 0.72;
  const jawGate = Math.max(0, Math.min(1, (-0.12 - y) / 0.55)) * Math.max(0, Math.min(1, (z + 0.25) / 0.6));

  switch (name) {
    case 'jawOpen': {
      // Jaw rotation about the TMJ-like pivot (§6 jaw rotation via morphs).
      const [px, py, pz] = CONFIG.jawPivot;
      const w = jawGate;
      const ang = CONFIG.jawOpenRad * w;
      const ry = y - py;
      const rz = z - pz;
      dy += (ry * (Math.cos(ang) - 1) - rz * Math.sin(ang)) * w;
      dz += (ry * Math.sin(ang) + rz * (Math.cos(ang) - 1)) * w;
      break;
    }
    case 'jawForward': dz += jawGate * 0.09; break;
    case 'jawLeft': dx -= jawGate * 0.10; break;
    case 'jawRight': dx += jawGate * 0.10; break;
    case 'mouthClose': {
      const w = g(0, mouthY - 0.10, mouthZ, R * 0.8);
      dy += w * 0.12;
      break;
    }
    case 'mouthSmileLeft': {
      const w = g(-0.30, mouthY, mouthZ * 0.92, R * 0.85);
      dy += w * 0.115; dx -= w * 0.055; dz -= w * 0.02;
      break;
    }
    case 'mouthSmileRight': {
      const w = g(0.30, mouthY, mouthZ * 0.92, R * 0.85);
      dy += w * 0.115; dx += w * 0.055; dz -= w * 0.02;
      break;
    }
    case 'mouthDimpleLeft': { const w = g(-0.33, mouthY + 0.03, mouthZ * 0.85); dx += w * 0.035; dz -= w * 0.03; break; }
    case 'mouthDimpleRight': { const w = g(0.33, mouthY + 0.03, mouthZ * 0.85); dx -= w * 0.035; dz -= w * 0.03; break; }
    case 'mouthFrownLeft': { const w = g(-0.28, mouthY, mouthZ * 0.9); dy -= w * 0.10; break; }
    case 'mouthFrownRight': { const w = g(0.28, mouthY, mouthZ * 0.9); dy -= w * 0.10; break; }
    case 'mouthLeft': { const w = g(0, mouthY, mouthZ); dx -= w * 0.12; break; }
    case 'mouthRight': { const w = g(0, mouthY, mouthZ); dx += w * 0.12; break; }
    case 'mouthPucker': { const w = g(0, mouthY, mouthZ, R * 0.75); dz += w * 0.10; dx *= 1; break; }
    case 'mouthFunnel': { const w = g(0, mouthY, mouthZ, R * 0.8); dz += w * 0.13; dy -= w * 0.015; break; }
    case 'mouthUpperUpLeft': { const w = g(-0.12, mouthY + 0.10, mouthZ * 0.95); dy += w * 0.075; break; }
    case 'mouthUpperUpRight': { const w = g(0.12, mouthY + 0.10, mouthZ * 0.95); dy += w * 0.075; break; }
    case 'mouthLowerDownLeft': { const w = g(-0.12, mouthY - 0.11, mouthZ * 0.95); dy -= w * 0.075; break; }
    case 'mouthLowerDownRight': { const w = g(0.12, mouthY - 0.11, mouthZ * 0.95); dy -= w * 0.075; break; }
    case 'mouthPressLeft': { const w = g(-0.16, mouthY, mouthZ * 0.98, R * 0.7); dz -= w * 0.035; dy *= 1; break; }
    case 'mouthPressRight': { const w = g(0.16, mouthY, mouthZ * 0.98, R * 0.7); dz -= w * 0.035; break; }
    case 'mouthRollLower': { const w = g(0, mouthY - 0.10, mouthZ * 0.97, R * 0.7); dy += w * 0.05; dz -= w * 0.02; break; }
    case 'mouthRollUpper': { const w = g(0, mouthY + 0.10, mouthZ * 0.97, R * 0.7); dy -= w * 0.05; dz -= w * 0.02; break; }
    case 'mouthShrugLower': { const w = g(0, mouthY - 0.12, mouthZ * 0.95); dy += w * 0.05; break; }
    case 'mouthShrugUpper': { const w = g(0, mouthY + 0.12, mouthZ * 0.95); dy += w * 0.05; break; }
    case 'mouthStretchLeft': { const w = g(-0.34, mouthY, mouthZ * 0.85); dx -= w * 0.10; break; }
    case 'mouthStretchRight': { const w = g(0.34, mouthY, mouthZ * 0.85); dx += w * 0.10; break; }
    case 'browInnerUp': {
      const wl = g(-0.14, 0.30, mouthZ * 0.78);
      const wr = g(0.14, 0.30, mouthZ * 0.78);
      dy += (wl + wr) * 0.10;
      break;
    }
    case 'browDownLeft': { const w = g(-0.20, 0.27, mouthZ * 0.75); dy -= w * 0.075; break; }
    case 'browDownRight': { const w = g(0.20, 0.27, mouthZ * 0.75); dy -= w * 0.075; break; }
    case 'browOuterUpLeft': { const w = g(-0.42, 0.28, mouthZ * 0.55); dy += w * 0.11; break; }
    case 'browOuterUpRight': { const w = g(0.42, 0.28, mouthZ * 0.55); dy += w * 0.11; break; }
    case 'cheekPuff': {
      const wl = g(-0.52, -0.05, 0.42);
      const wr = g(0.52, -0.05, 0.42);
      dx -= wl * 0.10; dx += wr * 0.10;
      dz += (wl + wr) * 0.04;
      break;
    }
    case 'cheekSquintLeft': { const w = g(-0.34, 0.02, 0.62); dy += w * 0.05; break; }
    case 'cheekSquintRight': { const w = g(0.34, 0.02, 0.62); dy += w * 0.05; break; }
    case 'noseSneerLeft': { const w = g(-0.13, 0.02, 0.82, R * 0.55); dy += w * 0.055; break; }
    case 'noseSneerRight': { const w = g(0.13, 0.02, 0.82, R * 0.55); dy += w * 0.055; break; }
    case 'eyeWideLeft': { const w = g(-0.30, 0.12, 0.68, R * 0.62); dy += w * 0.05; break; }
    case 'eyeWideRight': { const w = g(0.30, 0.12, 0.68, R * 0.62); dy += w * 0.05; break; }
    case 'eyeSquintLeft': { const w = g(-0.30, 0.05, 0.68, R * 0.58); dy += w * 0.035; break; }
    case 'eyeSquintRight': { const w = g(0.30, 0.05, 0.68, R * 0.58); dy += w * 0.035; break; }
    case 'eyeBlinkLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.55); dy -= w * 0.045; break; }
    case 'eyeBlinkRight': { const w = g(0.30, 0.10, 0.70, R * 0.55); dy -= w * 0.045; break; }
    case 'eyeLookUpLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dy += w * 0.025; break; }
    case 'eyeLookUpRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dy += w * 0.025; break; }
    case 'eyeLookDownLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dy -= w * 0.025; break; }
    case 'eyeLookDownRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dy -= w * 0.025; break; }
    case 'eyeLookInLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dx += w * 0.02; break; }
    case 'eyeLookInRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dx -= w * 0.02; break; }
    case 'eyeLookOutLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dx -= w * 0.02; break; }
    case 'eyeLookOutRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dx += w * 0.02; break; }
    case 'tongueOut': {
      const w = g(0, mouthY - 0.05, mouthZ * 0.95, R * 0.6);
      dz += w * 0.16;
      break;
    }
    default: break; // jawForward etc. handled above; unknown → zero
  }
  out3[0] = dx;
  out3[1] = dy;
  out3[2] = dz;
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
  root.add(mesh, eyeL, eyeR, lidL, lidR);

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
    hash: assetHash('procedural:v1:' + CHANNELS.length),
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

// ---------------------------------------------------------------------------
// 2D User-Selected Image Mesh + Landmark-Anchored 52-Channel Warping
// ---------------------------------------------------------------------------

/**
 * Build a 2D/2.5D warpable image head asset from a user-selected image
 * (HTMLCanvasElement, HTMLImageElement, or ImageBitmap) and 478 normalized
 * landmarks (detected via MediaPipe FaceLandmarker or canonical fallback).
 *
 * All 52 ARKit blendshapes are precomputed into morph targets anchored at the
 * photo's facial landmarks, so webcam mirroring, emotion presets, 52 sliders,
 * automatic blinks/saccades, and 16-point click-drag sculpting all animate the
 * photo directly.
 *
 * @param {HTMLCanvasElement|HTMLImageElement|ImageBitmap|{width:number,height:number}} imageSource
 * @param {Array<{x: number, y: number, z?: number}>|null} [landmarks478]
 * @param {{warpScale?: number, showControls?: boolean, id?: string}} [opts]
 */
export function buildImageHead(imageSource, landmarks478 = null, opts = {}) {
  const imgW = imageSource?.width || imageSource?.naturalWidth || 640;
  const imgH = imageSource?.height || imageSource?.naturalHeight || 800;
  const planeSpec = computeImagePlaneSize(imgW, imgH);
  const features = extractImageFeatures(landmarks478, planeSpec);

  const W = IMAGE_WARP_CONFIG.gridSegmentsW;
  const H = IMAGE_WARP_CONFIG.gridSegmentsH;
  const vertCount = (W + 1) * (H + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const uGrid = new Float32Array(vertCount);
  const vGrid = new Float32Array(vertCount);
  const indices = [];

  let vi = 0;
  for (let j = 0; j <= H; j++) {
    const v = j / H; // 0 = top of image, 1 = bottom of image
    const y = (0.5 - v) * planeSpec.planeH + planeSpec.centerY;
    for (let i = 0; i <= W; i++) {
      const u = i / W; // 0 = left of image, 1 = right of image
      const x = (u - 0.5) * planeSpec.planeW;
      const z = faceReliefZ(x, y, u, v, features);
      positions[vi * 3] = x;
      positions[vi * 3 + 1] = y;
      positions[vi * 3 + 2] = z;
      uvs[vi * 2] = u;
      uvs[vi * 2 + 1] = 1 - v;
      uGrid[vi] = u;
      vGrid[vi] = v;
      vi++;
    }
  }

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const a = j * (W + 1) + i;
      const b = a + 1;
      const c = a + (W + 1);
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  // Texture from the user-selected image (unlit MeshBasicMaterial preserves
  // true photo colors without ACES tone-mapping washout).
  const tex = new THREE.Texture(/** @type {any} */ (imageSource));
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    toneMapped: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);

  // Build all 52 ARKit morph targets anchored to the image's landmarks.
  const morphs = [];
  /** @type {Record<string, Array<{mesh: THREE.Mesh, index: number}>>} */
  const morphMap = {};
  const delta = new Float32Array(3);
  for (const name of CHANNELS) {
    const arr = new Float32Array(vertCount * 3);
    for (let k = 0; k < vertCount; k++) {
      const x = positions[k * 3];
      const y = positions[k * 3 + 1];
      imageMorphDelta(name, x, y, uGrid[k], vGrid[k], features, delta);
      arr[k * 3] = delta[0];
      arr[k * 3 + 1] = delta[1];
      arr[k * 3 + 2] = delta[2];
    }
    morphs.push(new THREE.Float32BufferAttribute(arr, 3));
    morphMap[name] = [{ mesh, index: morphs.length - 1 }];
  }
  geo.morphAttributes.position = morphs;
  geo.morphTargetsRelative = true;
  mesh.morphTargetInfluences = new Array(morphs.length).fill(0);
  mesh.morphTargetDictionary = Object.fromEntries(CHANNELS.map((n, idx) => [n, idx]));

  // Subtle studio frame backing behind the image edge.
  const frameGeo = new THREE.PlaneGeometry(planeSpec.planeW + 0.03, planeSpec.planeH + 0.03);
  const frameMat = new THREE.MeshBasicMaterial({
    color: 0x1c2433,
    toneMapped: false,
  });
  const frameMesh = new THREE.Mesh(frameGeo, frameMat);
  frameMesh.position.set(0, planeSpec.centerY, planeSpec.baseZ - 0.012);

  // Interactive 16-control-point overlay markers so the user can see and
  // grab the sculpt handles right on their selected photo.
  const controlGroup = new THREE.Group();
  controlGroup.visible = opts.showControls !== false;
  const ringGeo = new THREE.RingGeometry(0.011, 0.018, 18);
  const dotGeo = new THREE.CircleGeometry(0.006, 14);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x66e0b0,
    transparent: true,
    opacity: 0.72,
    depthTest: false,
    toneMapped: false,
  });
  const dotMat = new THREE.MeshBasicMaterial({
    color: 0xf7b267,
    transparent: true,
    opacity: 0.85,
    depthTest: false,
    toneMapped: false,
  });
  /** @type {THREE.Object3D[]} */
  const controlMarkers = [];
  for (let c = 0; c < CONTROLS.length; c++) {
    const a = features.controlAnchors[c];
    const holder = new THREE.Group();
    const ring = new THREE.Mesh(ringGeo, ringMat);
    const dot = new THREE.Mesh(dotGeo, dotMat);
    ring.renderOrder = 10;
    dot.renderOrder = 11;
    holder.add(ring, dot);
    holder.position.set(a[0], a[1], a[2] + 0.01);
    controlGroup.add(holder);
    controlMarkers.push(holder);
  }

  const root = new THREE.Group();
  root.add(frameMesh, mesh, controlGroup);

  const morphedPt = new Float32Array(3);
  const asset = {
    root,
    meshes: [mesh],
    morphMap,
    warnings: [],
    missingChannels: [],
    eyeNodes: { left: null, right: null },
    hasTeeth: true,
    hasTongue: true,
    isProcedural: false,
    isImage: true,
    imageSource,
    imageTexture: tex,
    features,
    controlAnchors: features.controlAnchors,
    warpScale: opts.warpScale ?? 1,
    hash: assetHash(
      'image:v1:' + (opts.id ?? `${imgW}x${imgH}:${features.mouth.cornerL[0].toFixed(3)}`)
    ),
    /**
     * Update the 16 control-point visual markers to follow the warped mesh.
     * @param {Float32Array} weights52
     */
    updateControlPoints(weights52) {
      if (!controlGroup.visible) return;
      for (let c = 0; c < controlMarkers.length; c++) {
        evaluateControlAnchorMorphed(c, weights52, CHANNELS, features, morphedPt, asset.warpScale);
        controlMarkers[c].position.set(morphedPt[0], morphedPt[1], morphedPt[2] + 0.01);
      }
    },
    /**
     * Toggle visibility of the 16 sculpt markers on the photo.
     * @param {boolean} visible
     */
    setControlPointsVisible(visible) {
      controlGroup.visible = !!visible;
    },
    /**
     * Set overall warp multiplier (0.2..2.5).
     * @param {number} scale
     */
    setWarpScale(scale) {
      asset.warpScale = Math.min(Math.max(scale, 0.1), 3.0);
    },
    /**
     * Drive the 2D image's eye-look morphs from EyeSolver gaze + sliders.
     * @param {number} inYawRad
     * @param {number} inPitchRad
     * @param {Float32Array} weights52
     */
    setGaze(inYawRad, inPitchRad, weights52) {
      const inf = mesh.morphTargetInfluences;
      if (!inf) return;
      const ws = asset.warpScale ?? 1;
      const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
      const gRight = clamp01(inYawRad / 0.36);
      const gLeft = clamp01(-inYawRad / 0.36);
      const gUp = clamp01(inPitchRad / 0.28);
      const gDown = clamp01(-inPitchRad / 0.28);

      inf[CH.eyeLookOutLeft] = Math.max(weights52[CH.eyeLookOutLeft] ?? 0, gLeft) * ws;
      inf[CH.eyeLookInLeft] = Math.max(weights52[CH.eyeLookInLeft] ?? 0, gRight) * ws;
      inf[CH.eyeLookOutRight] = Math.max(weights52[CH.eyeLookOutRight] ?? 0, gRight) * ws;
      inf[CH.eyeLookInRight] = Math.max(weights52[CH.eyeLookInRight] ?? 0, gLeft) * ws;
      inf[CH.eyeLookUpLeft] = Math.max(weights52[CH.eyeLookUpLeft] ?? 0, gUp) * ws;
      inf[CH.eyeLookUpRight] = Math.max(weights52[CH.eyeLookUpRight] ?? 0, gUp) * ws;
      inf[CH.eyeLookDownLeft] = Math.max(weights52[CH.eyeLookDownLeft] ?? 0, gDown) * ws;
      inf[CH.eyeLookDownRight] = Math.max(weights52[CH.eyeLookDownRight] ?? 0, gDown) * ws;
    },
  };

  return asset;
}

/**
 * Generate a built-in illustrated portrait canvas aligned with the canonical
 * 478 landmarks so users can test 2D image warping with one click even before
 * picking a photo from disk.
 *
 * @param {number} [W]
 * @param {number} [H]
 * @returns {HTMLCanvasElement}
 */
export function makeSamplePortraitCanvas(W = 640, H = 800) {
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const g = canvas.getContext('2d');
  if (!g) return canvas;

  // Studio portrait backdrop
  const bg = g.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#1e293b');
  bg.addColorStop(0.55, '#172030');
  bg.addColorStop(1, '#0f141e');
  g.fillStyle = bg;
  g.fillRect(0, 0, W, H);

  // Soft ambient halo
  const halo = g.createRadialGradient(W * 0.5, H * 0.45, W * 0.06, W * 0.5, H * 0.45, W * 0.52);
  halo.addColorStop(0, 'rgba(102, 224, 176, 0.14)');
  halo.addColorStop(1, 'rgba(102, 224, 176, 0)');
  g.fillStyle = halo;
  g.fillRect(0, 0, W, H);

  // Back hair / shoulders
  g.fillStyle = '#1b2434';
  g.beginPath();
  g.ellipse(W * 0.5, H * 0.96, W * 0.40, H * 0.20, 0, Math.PI, 0, false);
  g.fill();

  // Neck
  const neckGrad = g.createLinearGradient(W * 0.4, H * 0.68, W * 0.6, H * 0.86);
  neckGrad.addColorStop(0, '#be8c73');
  neckGrad.addColorStop(1, '#9d6e58');
  g.fillStyle = neckGrad;
  g.beginPath();
  g.roundRect(W * 0.41, H * 0.67, W * 0.18, H * 0.20, 24);
  g.fill();

  // Ears
  g.fillStyle = '#c8967c';
  g.beginPath();
  g.ellipse(W * 0.255, H * 0.46, W * 0.032, H * 0.058, -0.1, 0, Math.PI * 2);
  g.ellipse(W * 0.745, H * 0.46, W * 0.032, H * 0.058, 0.1, 0, Math.PI * 2);
  g.fill();

  // Face oval (aligned with canonical landmarks)
  const skin = g.createRadialGradient(W * 0.48, H * 0.43, W * 0.04, W * 0.5, H * 0.48, W * 0.30);
  skin.addColorStop(0, '#eabfa6');
  skin.addColorStop(0.65, '#d6a389');
  skin.addColorStop(1, '#b68168');
  g.fillStyle = skin;
  g.beginPath();
  g.moveTo(W * 0.50, H * 0.20);
  g.bezierCurveTo(W * 0.72, H * 0.20, W * 0.75, H * 0.40, W * 0.71, H * 0.57);
  g.bezierCurveTo(W * 0.68, H * 0.67, W * 0.59, H * 0.75, W * 0.50, H * 0.755);
  g.bezierCurveTo(W * 0.41, H * 0.75, W * 0.32, H * 0.67, W * 0.29, H * 0.57);
  g.bezierCurveTo(W * 0.25, H * 0.40, W * 0.28, H * 0.20, W * 0.50, H * 0.20);
  g.closePath();
  g.fill();

  // Cheek blush
  for (const cx of [0.34, 0.66]) {
    const blush = g.createRadialGradient(W * cx, H * 0.52, 4, W * cx, H * 0.52, W * 0.085);
    blush.addColorStop(0, 'rgba(218, 112, 102, 0.28)');
    blush.addColorStop(1, 'rgba(218, 112, 102, 0)');
    g.fillStyle = blush;
    g.beginPath();
    g.arc(W * cx, H * 0.52, W * 0.085, 0, Math.PI * 2);
    g.fill();
  }

  // Hair crown
  const hair = g.createLinearGradient(0, H * 0.14, 0, H * 0.40);
  hair.addColorStop(0, '#2a211d');
  hair.addColorStop(1, '#1c1512');
  g.fillStyle = hair;
  g.beginPath();
  g.moveTo(W * 0.26, H * 0.43);
  g.bezierCurveTo(W * 0.23, H * 0.22, W * 0.35, H * 0.13, W * 0.50, H * 0.14);
  g.bezierCurveTo(W * 0.66, H * 0.13, W * 0.77, H * 0.22, W * 0.74, H * 0.43);
  g.bezierCurveTo(W * 0.71, H * 0.30, W * 0.61, H * 0.24, W * 0.48, H * 0.25);
  g.bezierCurveTo(W * 0.36, H * 0.24, W * 0.29, H * 0.30, W * 0.26, H * 0.43);
  g.closePath();
  g.fill();

  // Eyebrows (left ~ 0.32..0.435, y ~ 0.332..0.355; right ~ 0.565..0.68)
  g.strokeStyle = '#2d201a';
  g.lineCap = 'round';
  g.lineWidth = W * 0.019;
  g.beginPath();
  g.moveTo(W * 0.320, H * 0.355);
  g.quadraticCurveTo(W * 0.382, H * 0.324, W * 0.435, H * 0.348);
  g.stroke();
  g.beginPath();
  g.moveTo(W * 0.565, H * 0.348);
  g.quadraticCurveTo(W * 0.618, H * 0.324, W * 0.680, H * 0.355);
  g.stroke();

  // Eyes (left center 0.390, 0.415; right center 0.610, 0.415)
  const drawEye = (cx, cy, outerX, innerX) => {
    const rx = Math.abs(innerX - outerX) * 0.52;
    const ry = H * 0.022;
    g.save();
    g.beginPath();
    g.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    g.clip();
    // Sclera
    g.fillStyle = '#f5f0ec';
    g.fillRect(cx - rx, cy - ry, rx * 2, ry * 2);
    // Iris
    const ir = ry * 0.92;
    const ig = g.createRadialGradient(cx, cy, ir * 0.15, cx, cy, ir);
    ig.addColorStop(0, '#2c4c5e');
    ig.addColorStop(0.65, '#3f6e82');
    ig.addColorStop(1, '#1b2e38');
    g.fillStyle = ig;
    g.beginPath();
    g.arc(cx, cy, ir, 0, Math.PI * 2);
    g.fill();
    // Pupil
    g.fillStyle = '#0b0d12';
    g.beginPath();
    g.arc(cx, cy, ir * 0.42, 0, Math.PI * 2);
    g.fill();
    // Specular catchlight
    g.fillStyle = 'rgba(255,255,255,0.85)';
    g.beginPath();
    g.arc(cx - ir * 0.28, cy - ir * 0.28, ir * 0.18, 0, Math.PI * 2);
    g.fill();
    g.restore();

    // Upper lash line
    g.strokeStyle = '#241914';
    g.lineWidth = W * 0.008;
    g.beginPath();
    g.ellipse(cx, cy, rx, ry, 0, Math.PI, Math.PI * 2);
    g.stroke();
    // Lower lid line
    g.strokeStyle = 'rgba(110, 74, 60, 0.55)';
    g.lineWidth = W * 0.004;
    g.beginPath();
    g.ellipse(cx, cy, rx * 0.96, ry * 0.98, 0, 0, Math.PI);
    g.stroke();
  };
  drawEye(W * 0.390, H * 0.415, W * 0.345, W * 0.435);
  drawEye(W * 0.610, H * 0.415, W * 0.655, W * 0.565);

  // Nose (bridge 0.50, 0.385 -> tip 0.50, 0.510; wings 0.452..0.548, 0.525)
  g.strokeStyle = 'rgba(145, 96, 76, 0.55)';
  g.lineWidth = W * 0.006;
  g.beginPath();
  g.moveTo(W * 0.485, H * 0.395);
  g.quadraticCurveTo(W * 0.478, H * 0.465, W * 0.465, H * 0.512);
  g.stroke();
  g.fillStyle = 'rgba(255, 236, 222, 0.32)';
  g.beginPath();
  g.ellipse(W * 0.500, H * 0.506, W * 0.018, H * 0.015, 0, 0, Math.PI * 2);
  g.fill();
  g.strokeStyle = 'rgba(130, 82, 65, 0.72)';
  g.lineWidth = W * 0.007;
  g.beginPath();
  g.moveTo(W * 0.452, H * 0.522);
  g.quadraticCurveTo(W * 0.500, H * 0.538, W * 0.548, H * 0.522);
  g.stroke();

  // Lips (corners 0.405..0.595, 0.625; upper 0.598; lower 0.654)
  // Upper lip
  g.fillStyle = '#b96b64';
  g.beginPath();
  g.moveTo(W * 0.405, H * 0.625);
  g.quadraticCurveTo(W * 0.455, H * 0.592, W * 0.485, H * 0.602);
  g.quadraticCurveTo(W * 0.500, H * 0.608, W * 0.515, H * 0.602);
  g.quadraticCurveTo(W * 0.545, H * 0.592, W * 0.595, H * 0.625);
  g.quadraticCurveTo(W * 0.500, H * 0.628, W * 0.405, H * 0.625);
  g.closePath();
  g.fill();

  // Lower lip
  const lipGrad = g.createLinearGradient(0, H * 0.624, 0, H * 0.658);
  lipGrad.addColorStop(0, '#c97b73');
  lipGrad.addColorStop(0.55, '#d48980');
  lipGrad.addColorStop(1, '#a85d56');
  g.fillStyle = lipGrad;
  g.beginPath();
  g.moveTo(W * 0.405, H * 0.625);
  g.quadraticCurveTo(W * 0.500, H * 0.628, W * 0.595, H * 0.625);
  g.quadraticCurveTo(W * 0.555, H * 0.658, W * 0.500, H * 0.656);
  g.quadraticCurveTo(W * 0.445, H * 0.658, W * 0.405, H * 0.625);
  g.closePath();
  g.fill();

  // Lip parting line
  g.strokeStyle = '#6e3430';
  g.lineWidth = W * 0.005;
  g.beginPath();
  g.moveTo(W * 0.405, H * 0.625);
  g.quadraticCurveTo(W * 0.500, H * 0.628, W * 0.595, H * 0.625);
  g.stroke();

  return canvas;
}

