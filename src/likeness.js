/**
 * likeness.js — bounded, image-derived identity fitting; no DOM or network.
 *
 * Correspondences come from detecting the *neutral rendered template*, not
 * guessed scan UVs. Photo pose/scale/roll are removed before fitting. A smooth
 * displacement field is applied to the rest surface AND every posed target,
 * including the eye/lid geometry. Ears/back/neck stay pinned by the same mask
 * as the anatomical expression rig. This is single-view fitting, not identity
 * recognition or recovery of unseen anatomy/hair.
 */
import * as THREE from '../vendor/three.core.js';
import { faceDeformationMask } from './anatomy.js';

export const FACE_OVAL = Object.freeze([
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]);
export const FIT_LANDMARKS = Object.freeze([...new Set([
  ...FACE_OVAL,
  33, 133, 159, 145, 160, 144, 158, 153, 263, 362, 386, 374, 387, 373, 385, 380,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 129, 358,
  50, 101, 205, 280, 330, 425,
  61, 291, 0, 13, 14, 17, 37, 267, 40, 270, 84, 314, 181, 405, 199,
])]);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/** Remove image dimensions, translation, distance, roll and estimated pose. */
export function normalizeFaceLandmarks(landmarks, width, height, rotation = null) {
  if (![width, height].every((x) => Number.isFinite(x) && x > 0) || landmarks.length < 468) {
    throw new Error('A complete face image is required');
  }
  const ids = [33, 133, 263, 362];
  if (ids.some((i) => !landmarks[i])) throw new Error('A complete face image is required');
  const center = ids.reduce((a, i) => [a[0] + landmarks[i].x * width / 4,
    a[1] + landmarks[i].y * height / 4, a[2] + landmarks[i].z * width / 4], [0, 0, 0]);
  const points = landmarks.map((p) => {
    if (!p || ![p.x, p.y, p.z].every(Number.isFinite)) throw new Error('Invalid face landmarks');
    const x = p.x * width - center[0], y = center[1] - p.y * height, z = center[2] - p.z * width;
    // Inverse rotation only — MediaPipe's metric translation/scale do not
    // share normalized-image units. R is column-major, so multiply by R^T.
    return rotation ? [rotation[0] * x + rotation[1] * y + rotation[2] * z,
      rotation[4] * x + rotation[5] * y + rotation[6] * z,
      rotation[8] * x + rotation[9] * y + rotation[10] * z] : [x, y, z];
  });
  const a = points[33].map((v, k) => (v + points[133][k]) / 2);
  const b = points[263].map((v, k) => (v + points[362][k]) / 2);
  const distance = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (!Number.isFinite(distance) || distance < 1) throw new Error('The face is too small to fit');
  const ux = (b[0] - a[0]) / distance, uy = (b[1] - a[1]) / distance;
  const scale = 0.56 / distance;
  return points.map(([x, y, z]) => [(x * ux + y * uy) * scale, 0.105 + (-x * uy + y * ux) * scale, z * scale]);
}

/** Never silently use the wrong person, an empty image, or a clipped profile. */
export function validatePortrait(result, width, height) {
  const faces = result.faceLandmarks ?? [];
  if (faces.length !== 1) {
    throw new Error(faces.length ? 'Use an image with just one face' : 'No face found. Use a clear, front-facing portrait');
  }
  const lm = faces[0];
  normalizeFaceLandmarks(lm, width, height); // includes finite/complete validation
  if (FACE_OVAL.some((i) => lm[i].x < 0.005 || lm[i].x > 0.995 || lm[i].y < 0.005 || lm[i].y > 0.995)) {
    throw new Error('Include the whole face, forehead and chin in the image');
  }
  const eyeDistance = Math.hypot((lm[263].x - lm[33].x) * width, (lm[263].y - lm[33].y) * height);
  if (eyeDistance < 32) throw new Error('The face is too small. Use a closer or higher-resolution portrait');
  const matrix = result.facialTransformationMatrixes?.[0]?.data;
  if (matrix) {
    const yaw = Math.atan2(matrix[8], matrix[10]);
    const pitch = Math.atan2(-matrix[9], Math.hypot(matrix[8], matrix[10]));
    if (Math.abs(yaw) > 25 * Math.PI / 180 || Math.abs(pitch) > 22 * Math.PI / 180) {
      throw new Error('Use a front-facing image (less than 25° turn), with both eyes visible');
    }
  }
  return lm;
}

/** Dense linear solve, multiple RHS; small (~100-point) one-time TPS system. */
function solve(matrix, rhs) {
  const n = matrix.length;
  const rows = matrix.map((row, i) => Float64Array.from([...row, ...rhs[i]]));
  const channels = rhs[0].length;
  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(rows[i][k]) > Math.abs(rows[pivot][k])) pivot = i;
    if (Math.abs(rows[pivot][k]) < 1e-12) throw new Error('Could not fit this face. Try a clearer frontal image');
    [rows[k], rows[pivot]] = [rows[pivot], rows[k]];
    const d = rows[k][k];
    for (let j = k; j < n + channels; j++) rows[k][j] /= d;
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = rows[i][k];
      for (let j = k; j < n + channels; j++) rows[i][j] -= f * rows[k][j];
    }
  }
  return rows.map((row) => row.slice(n));
}
const kernel = (r2) => r2 < 1e-14 ? 0 : r2 * Math.log(r2);

/**
 * Sample the smooth fit once onto a small grid: all 52 morphs then use O(1)
 * interpolation, not an RBF sum per vertex on every animation frame.
 * controls = [{from:[x,y,z], delta:[dx,dy,dz]}]. Returns an allocation-free warp.
 */
export function makeFaceWarp(controls, cols = 65, rows = 85) {
  if (controls.length < 4 || controls.some((c) => [...c.from, ...c.delta].some((v) => !Number.isFinite(v)))) {
    throw new Error('Not enough valid face correspondences');
  }
  const n = controls.length;
  const matrix = Array.from({ length: n + 3 }, () => new Float64Array(n + 3));
  const rhs = Array.from({ length: n + 3 }, () => new Float64Array(3));
  for (let i = 0; i < n; i++) {
    const [x, y] = controls[i].from;
    for (let j = 0; j < n; j++) {
      const [u, v] = controls[j].from;
      matrix[i][j] = kernel((x - u) ** 2 + (y - v) ** 2) + (i === j ? 0.00015 : 0);
    }
    matrix[i][n] = matrix[n][i] = 1;
    matrix[i][n + 1] = matrix[n + 1][i] = x;
    matrix[i][n + 2] = matrix[n + 2][i] = y;
    rhs[i].set(controls[i].delta);
  }
  const coefficients = solve(matrix, rhs);
  const grid = new Float32Array(cols * rows * 3);
  const minX = -0.9, maxX = 0.9, minY = -1.15, maxY = 1.15;
  for (let j = 0; j < rows; j++) {
    const y = minY + j / (rows - 1) * (maxY - minY);
    for (let i = 0; i < cols; i++) {
      const x = minX + i / (cols - 1) * (maxX - minX);
      const delta = Array.from(coefficients[n], (c, k) => c + x * coefficients[n + 1][k] + y * coefficients[n + 2][k]);
      for (let c = 0; c < n; c++) {
        const r = kernel((x - controls[c].from[0]) ** 2 + (y - controls[c].from[1]) ** 2);
        for (let k = 0; k < 3; k++) delta[k] += coefficients[c][k] * r;
      }
      for (let k = 0; k < 3; k++) grid[(j * cols + i) * 3 + k] = clamp(delta[k], -[0.16, 0.22, 0.12][k], [0.16, 0.22, 0.12][k]);
    }
  }
  const foreheadTop = Math.max(...controls.map((c) => c.from[1]));
  return (x, y, z, out) => {
    // There are no identity correspondences above the visible forehead.
    // Unconstrained TPS extrapolation there can create a ridge on the scalp
    // in profile. Fade smoothly to the original round cranium instead.
    const cap = clamp((y - foreheadTop - 0.025) / 0.25, 0, 1);
    const mask = faceDeformationMask(x, y, z) * (1 - cap * cap * (3 - 2 * cap));
    if (!mask) { out[0] = x; out[1] = y; out[2] = z; return out; }
    const u = clamp((x - minX) / (maxX - minX) * (cols - 1), 0, cols - 1.00001);
    const v = clamp((y - minY) / (maxY - minY) * (rows - 1), 0, rows - 1.00001);
    const i = Math.floor(u), j = Math.floor(v), tx = u - i, ty = v - j;
    const a = (j * cols + i) * 3, b = a + cols * 3;
    out[0] = x; out[1] = y; out[2] = z;
    for (let k = 0; k < 3; k++) {
      out[k] += mask * ((grid[a + k] * (1 - tx) + grid[a + 3 + k] * tx) * (1 - ty)
        + (grid[b + k] * (1 - tx) + grid[b + 3 + k] * tx) * ty);
    }
    return out;
  };
}

/** Fit from the immutable template, never from the previously fitted person. */
export function makeIdentityControls(reference, portrait, width, height, rotation = null) {
  const base = normalizeFaceLandmarks(reference.landmarks, reference.width, reference.height, reference.rotation);
  const target = normalizeFaceLandmarks(portrait, width, height, rotation);
  return reference.anchors.map(({ index, point }) => ({
    from: point,
    delta: target[index].map((v, k) => clamp((v - base[index][k]) * (k === 2 ? 0.5 : 1),
      -[0.16, 0.22, 0.12][k], [0.16, 0.22, 0.12][k])),
  }));
}

/**
 * Warp normals by the inverse-transpose Jacobian. This preserves scan/lid
 * smoothing and UV seams; recomputing normals would reopen the lid collars.
 */
function makeNormalWarp(warp) {
  const p = new Float64Array(3), a = new Float64Array(3), b = new Float64Array(3), c = new Float64Array(3);
  return (x, y, z, nx, ny, nz, out) => {
    if (!faceDeformationMask(x, y, z)) { out[0] = nx; out[1] = ny; out[2] = nz; return; }
    const e = 0.001;
    warp(x, y, z, p);
    warp(x + e, y, z, a); warp(x, y + e, z, b); warp(x, y, z + e, c);
    for (let i = 0; i < 3; i++) { a[i] = (a[i] - p[i]) / e; b[i] = (b[i] - p[i]) / e; c[i] = (c[i] - p[i]) / e; }
    out[0] = nx * (b[1] * c[2] - b[2] * c[1]) + ny * (c[1] * a[2] - c[2] * a[1]) + nz * (a[1] * b[2] - a[2] * b[1]);
    out[1] = nx * (b[2] * c[0] - b[0] * c[2]) + ny * (c[2] * a[0] - c[0] * a[2]) + nz * (a[2] * b[0] - a[0] * b[2]);
    out[2] = nx * (b[0] * c[1] - b[1] * c[0]) + ny * (c[0] * a[1] - c[1] * a[0]) + nz * (a[0] * b[1] - a[1] * b[0]);
    const length = Math.hypot(...out) || 1;
    for (let i = 0; i < 3; i++) out[i] /= length;
  };
}

/**
 * Clone/rebase rest geometry, relative position morphs AND normal morphs.
 * offset/nextOffset keep eyeballs rotating around their fitted local centers.
 * No mutation of the pristine geometry: photo replacement cannot accumulate.
 */
export function deformIdentityGeometry(source, warp, offset = [0, 0, 0], nextOffset = [0, 0, 0]) {
  const geo = source.clone();
  const position = source.attributes.position;
  const normal = source.attributes.normal;
  const fitted = geo.attributes.position.array, fittedNormals = geo.attributes.normal.array;
  const normalWarp = makeNormalWarp(warp);
  const point = new Float64Array(3), n = new Float64Array(3);
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i) + offset[0], y = position.getY(i) + offset[1], z = position.getZ(i) + offset[2];
    warp(x, y, z, point);
    normalWarp(x, y, z, normal.getX(i), normal.getY(i), normal.getZ(i), n);
    for (let k = 0; k < 3; k++) { fitted[i * 3 + k] = point[k] - nextOffset[k]; fittedNormals[i * 3 + k] = n[k]; }
  }
  const morphs = source.morphAttributes.position ?? [];
  const normalMorphs = source.morphAttributes.normal ?? [];
  for (let m = 0; m < morphs.length; m++) {
    const delta = morphs[m], dn = normalMorphs[m];
    const out = geo.morphAttributes.position[m].array;
    const outN = geo.morphAttributes.normal?.[m]?.array;
    for (let i = 0; i < position.count; i++) {
      const j = i * 3;
      const dx = delta.getX(i), dy = delta.getY(i), dz = delta.getZ(i);
      const x = position.getX(i) + offset[0] + dx, y = position.getY(i) + offset[1] + dy, z = position.getZ(i) + offset[2] + dz;
      if (dx || dy || dz) {
        warp(x, y, z, point);
        for (let k = 0; k < 3; k++) out[j + k] = point[k] - nextOffset[k] - fitted[j + k];
      } else out.fill(0, j, j + 3);
      if (outN && dn) {
        if (dx || dy || dz || dn.getX(i) || dn.getY(i) || dn.getZ(i)) {
          normalWarp(x, y, z, normal.getX(i) + dn.getX(i), normal.getY(i) + dn.getY(i), normal.getZ(i) + dn.getZ(i), n);
          for (let k = 0; k < 3; k++) outN[j + k] = n[k] - fittedNormals[j + k];
        } else outN.fill(0, j, j + 3);
      }
    }
  }
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return geo;
}

/**
 * Barycentric lookup on the canonical face topology. The grid is only an
 * accelerator; actual photo UVs always interpolate the matching landmark
 * triangle, never inverse-distance averages across lips/eyes.
 */
export function createTriangleSampler(points, indices) {
  const size = 32;
  const bins = Array.from({ length: size * size }, () => []);
  const tris = [];
  for (let i = 0; i < indices.length; i += 3) {
    const ids = indices.slice(i, i + 3);
    const [a, b, c] = ids.map((j) => points[j]);
    const den = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if (Math.abs(den) < 1e-10) continue;
    const t = { ids, a, b, c, den };
    const ti = tris.push(t) - 1;
    const x0 = clamp(Math.floor(Math.min(a.x, b.x, c.x) * size), 0, size - 1);
    const x1 = clamp(Math.floor(Math.max(a.x, b.x, c.x) * size), 0, size - 1);
    const y0 = clamp(Math.floor(Math.min(a.y, b.y, c.y) * size), 0, size - 1);
    const y1 = clamp(Math.floor(Math.max(a.y, b.y, c.y) * size), 0, size - 1);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) bins[y * size + x].push(ti);
  }
  return (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > 1 || y > 1) return null;
    const bin = bins[Math.min(size - 1, Math.floor(y * size)) * size + Math.min(size - 1, Math.floor(x * size))];
    for (const ti of bin) {
      const { ids, a, b, c, den } = tris[ti];
      const u = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / den;
      const v = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / den;
      const w = 1 - u - v;
      if (u >= -1e-5 && v >= -1e-5 && w >= -1e-5) return { ids, weights: [u, v, w] };
    }
    return null;
  };
}

/** Latest-source-wins serial jobs, including reset/GLB/camera cancellation. */
export class LatestFaceSource {
  constructor() { this.version = 0; this.tail = Promise.resolve(); }
  cancel() { this.version++; }
  run(job) {
    const version = ++this.version;
    const current = () => version === this.version;
    const promise = this.tail.catch(() => {}).then(() => current() ? job(current) : null);
    this.tail = promise;
    return promise;
  }
}
