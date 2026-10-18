/** Portrait UV baking. Pure pixels/math so it is testable without WebGL/DOM. */
import { FACE_OVAL, createTriangleSampler } from './likeness.js';
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };

function nearestOutline(x, y, landmarks) {
  let nearest = { distance: Infinity };
  for (let i = 0; i < FACE_OVAL.length; i++) {
    const ia = FACE_OVAL[i], ib = FACE_OVAL[(i + 1) % FACE_OVAL.length];
    const a = landmarks[ia], b = landmarks[ib];
    const dx = b.x - a.x, dy = b.y - a.y;
    const t = clamp(((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy || 1));
    const distance = Math.hypot(x - a.x - dx * t, y - a.y - dy * t);
    if (distance < nearest.distance) nearest = { distance, ia, ib, t };
  }
  return nearest;
}

function insideOutline(x, y, landmarks) {
  let inside = false;
  for (let i = 0, j = FACE_OVAL.length - 1; i < FACE_OVAL.length; j = i++) {
    const a = landmarks[FACE_OVAL[i]], b = landmarks[FACE_OVAL[j]];
    if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** Similarity projection for the visible forehead above the face mesh. */
function foreheadProjection(reference, portrait, refWidth, refHeight, width, height) {
  const eye = (lm, w, h) => {
    const a = { x: (lm[33].x + lm[133].x) * w / 2, y: (lm[33].y + lm[133].y) * h / 2 };
    const b = { x: (lm[263].x + lm[362].x) * w / 2, y: (lm[263].y + lm[362].y) * h / 2 };
    const distance = Math.hypot(b.x - a.x, b.y - a.y);
    return { distance, angle: Math.atan2(b.y - a.y, b.x - a.x) };
  };
  const a = eye(reference, refWidth, refHeight), b = eye(portrait, width, height);
  const scale = b.distance / a.distance, angle = b.angle - a.angle;
  const c = Math.cos(angle), s = Math.sin(angle);
  return (x, y) => {
    const dx = (x - reference[10].x) * refWidth, dy = (y - reference[10].y) * refHeight;
    return [portrait[10].x + (dx * c - dy * s) * scale / width,
      portrait[10].y + (dx * s + dy * c) * scale / height];
  };
}

/**
 * Actual landmark triangle → source pixel UV. Eyes/lips never borrow color
 * from distant features. Ears and the rear skull get no photo projection.
 */
export function mapPortraitSurface(surface, reference, portrait, topology, width, height) {
  const sample = createTriangleSampler(reference.landmarks, topology);
  const upper = foreheadProjection(reference.landmarks, portrait, reference.width, reference.height, width, height);
  const n = surface.projected.length / 2;
  const uv = new Float32Array(n * 2), alpha = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = surface.projected[i * 2], y = surface.projected[i * 2 + 1];
    const z = surface.positions[i * 3 + 2], nz = surface.normals[i * 3 + 2];
    if (z < 0.32 || nz < -0.25) continue;
    const hit = sample(x, y);
    let u, v, weight = 0;
    if (hit) {
      u = hit.ids.reduce((sum, id, k) => sum + portrait[id].x * hit.weights[k], 0);
      v = hit.ids.reduce((sum, id, k) => sum + portrait[id].y * hit.weights[k], 0);
      weight = 1;
    } else if (y < reference.landmarks[10].y + 0.007 && surface.positions[i * 3 + 1] > 0.4) {
      [u, v] = upper(x, y);
      weight = 1;
    } else if (!insideOutline(x, y, reference.landmarks)) {
      // Extend edge skin over the round chin/temples, not the photograph's
      // background or shirt. A broad feather prevents a flat polygon patch.
      const edge = nearestOutline(x, y, reference.landmarks);
      if (edge.distance > 0.085) continue;
      const a = portrait[edge.ia], b = portrait[edge.ib];
      u = a.x * (1 - edge.t) + b.x * edge.t;
      v = a.y * (1 - edge.t) + b.y * edge.t;
      weight = 1 - smooth(0.012, 0.085, edge.distance);
    } else continue; // eye/mouth holes are not stretched across the aperture
    if (u < 0 || v < 0 || u > 1 || v > 1) continue;
    uv[i * 2] = u; uv[i * 2 + 1] = v;
    alpha[i] = weight * smooth(0.32, 0.52, z) * smooth(-0.25, 0.10, nz);
  }
  return { ...surface, photoUV: uv, alpha };
}

/** Bilinear sRGB sampling; caller supplies the destination to avoid pixel GC. */
export function samplePortraitPixel(pixels, width, height, u, v, out) {
  const x = clamp(u) * (width - 1), y = clamp(v) * (height - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
  const tx = x - x0, ty = y - y0;
  for (let k = 0; k < 3; k++) {
    out[k] = (pixels[(y0 * width + x0) * 4 + k] * (1 - tx) + pixels[(y0 * width + x1) * 4 + k] * tx) * (1 - ty)
      + (pixels[(y1 * width + x0) * 4 + k] * (1 - tx) + pixels[(y1 * width + x1) * 4 + k] * tx) * ty;
  }
  return out;
}

/** Robust local cheek tone, not a fixed ethnicity/skin tint or a stock image. */
export function sampleSkinTone(pixels, width, height, landmarks) {
  const colors = [];
  for (const id of [50, 101, 118, 119, 280, 330, 347, 348]) {
    const p = landmarks[id];
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const rgb = samplePortraitPixel(pixels, width, height, p.x + dx / width, p.y + dy / height, [0, 0, 0]);
      colors.push(rgb);
    }
  }
  return [0, 1, 2].map((k) => colors.map((c) => c[k]).sort((a, b) => a - b)[Math.floor(colors.length / 2)]);
}

/** Color extension for the unseen scalp only; never claim unseen hair geometry. */
export function sampleScalpTone(pixels, width, height, landmarks, reference, skin) {
  const project = foreheadProjection(reference.landmarks, landmarks, reference.width, reference.height, width, height);
  const colors = [];
  for (const dx of [-0.025, 0, 0.025]) {
    const [u, v] = project(reference.landmarks[10].x + dx, reference.landmarks[10].y - 0.085);
    if (u < 0 || u > 1 || v < 0 || v > 1) continue;
    colors.push(samplePortraitPixel(pixels, width, height, u, v, [0, 0, 0]));
  }
  if (!colors.length) return skin;
  const color = [0, 1, 2].map((k) => colors.map((c) => c[k]).sort((a, b) => a - b)[Math.floor(colors.length / 2)]);
  // Similar tones are simply skin, avoiding a hard cap on a bald portrait.
  const distance = Math.hypot(...color.map((c, k) => c - skin[k]));
  const grayHair = Math.max(...color) - Math.min(...color) < (Math.max(...skin) - Math.min(...skin)) * 0.75;
  return distance > 42 || distance > 25 && grayHair ? color : skin;
}

/**
 * Bake onto the head's existing UV atlas, so exported GLBs contain the current
 * person's texture (no video/shader-only overlay, no source-image dependency).
 * Pixel rasterization avoids thousands of full-image Canvas drawImage calls.
 */
export function bakePortraitAtlas(surfaces, pixels, width, height, skin, scalp = skin, size = 1024) {
  const data = new Uint8ClampedArray(size * size * 4), coverage = new Uint8Array(size * size);
  for (let i = 0; i < size * size; i++) { data.set(skin, i * 4); data[i * 4 + 3] = 255; }
  const rgb = new Float64Array(3);
  let photoPixels = 0;
  for (const surface of surfaces) {
    const { indices, atlasUV, photoUV, alpha, positions } = surface;
    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i], b = indices[i + 1], c = indices[i + 2];
      const ax = atlasUV[a * 2] * size, ay = (1 - atlasUV[a * 2 + 1]) * size;
      const bx = atlasUV[b * 2] * size, by = (1 - atlasUV[b * 2 + 1]) * size;
      const cx = atlasUV[c * 2] * size, cy = (1 - atlasUV[c * 2 + 1]) * size;
      const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (Math.abs(den) < 1e-8) continue;
      const x0 = clamp(Math.floor(Math.min(ax, bx, cx)), 0, size - 1), x1 = clamp(Math.ceil(Math.max(ax, bx, cx)), 0, size - 1);
      const y0 = clamp(Math.floor(Math.min(ay, by, cy)), 0, size - 1), y1 = clamp(Math.ceil(Math.max(ay, by, cy)), 0, size - 1);
      const ha = smooth(0.48, 0.82, positions[a * 3 + 1]), hb = smooth(0.48, 0.82, positions[b * 3 + 1]), hc = smooth(0.48, 0.82, positions[c * 3 + 1]);
      // The neck cap has deliberately arbitrary UVs spanning the atlas. It
      // must never paint over the face. Plain skin is already the background.
      if (alpha[a] + alpha[b] + alpha[c] < 0.001 && ha + hb + hc < 0.001) continue;
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const u = ((by - cy) * (x + 0.5 - cx) + (cx - bx) * (y + 0.5 - cy)) / den;
        const v = ((cy - ay) * (x + 0.5 - cx) + (ax - cx) * (y + 0.5 - cy)) / den;
        const w = 1 - u - v;
        if (u < -0.0001 || v < -0.0001 || w < -0.0001) continue;
        const index = y * size + x, p = index * 4;
        const blend = clamp(alpha[a] * u + alpha[b] * v + alpha[c] * w);
        const hair = clamp(ha * u + hb * v + hc * w);
        if (blend > 0.001) {
          const pa = alpha[a] * u, pb = alpha[b] * v, pc = alpha[c] * w;
          samplePortraitPixel(pixels, width, height,
            (photoUV[a * 2] * pa + photoUV[b * 2] * pb + photoUV[c * 2] * pc) / blend,
            (photoUV[a * 2 + 1] * pa + photoUV[b * 2 + 1] * pb + photoUV[c * 2 + 1] * pc) / blend, rgb);
          photoPixels++;
        }
        for (let k = 0; k < 3; k++) data[p + k] = (skin[k] * (1 - hair) + scalp[k] * hair) * (1 - blend) + (blend > 0.001 ? rgb[k] * blend : 0);
        coverage[index] = 1;
      }
    }
  }
  // Chart gutters: a two-pixel dilation avoids skin-colored UV seam pinholes
  // with bilinear filtering/mips, without mixing unrelated atlas islands.
  for (let pass = 0; pass < 2; pass++) {
    const next = coverage.slice();
    for (let y = 1; y < size - 1; y++) for (let x = 1; x < size - 1; x++) {
      const p = y * size + x;
      if (coverage[p]) continue;
      const neighbor = [p - 1, p + 1, p - size, p + size].find((q) => coverage[q]);
      if (neighbor !== undefined) { data.set(data.subarray(neighbor * 4, neighbor * 4 + 4), p * 4); next[p] = 1; }
    }
    coverage.set(next);
  }
  return { data, width: size, height: size, photoPixels };
}
