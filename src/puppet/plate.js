/**
 * Background plate: occlusion-aware inpainting of the region the head occupies.
 *
 * A photograph only contains the head where the head was. If the puppet turns
 * and nothing is behind it, the original silhouette stays burned into the frame
 * as a ghost - one of the main reasons 2D face puppets "look fake" on a turn.
 * So at load time the head interior is removed and filled with a multi-scale
 * harmonic (Poisson, zero-gradient) solve seeded from the surrounding
 * background, then softened. The head is composited on top of that plate every
 * frame; where it rotates away, believable background shows through instead of
 * a stretched copy of itself.
 */

/** Even-odd scanline rasterisation of a polygon into a byte mask. */
export function rasterizePolygon(W, H, pts) {
  const mask = new Uint8Array(W * H);
  const n = pts.length >> 1;
  if (n < 3) return mask;
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) { xs[i] = pts[i * 2]; ys[i] = pts[i * 2 + 1]; }
  const nodes = new Float64Array(n);
  for (let y = 0; y < H; y++) {
    const py = y + 0.5;
    let k = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      if ((ys[i] < py && ys[j] >= py) || (ys[j] < py && ys[i] >= py)) {
        nodes[k++] = xs[i] + ((py - ys[i]) / (ys[j] - ys[i])) * (xs[j] - xs[i]);
      }
    }
    const slice = Array.prototype.slice.call(nodes, 0, k).sort((a, b) => a - b);
    for (let i = 0; i + 1 < slice.length; i += 2) {
      const x0 = Math.max(0, Math.ceil(slice[i] - 0.5)), x1 = Math.min(W - 1, Math.floor(slice[i + 1] - 0.5));
      for (let x = x0; x <= x1; x++) mask[y * W + x] = 1;
    }
  }
  return mask;
}

/** Binary dilation by `r` pixels, via two separable box passes (fast, good enough). */
export function dilateMask(mask, W, H, r) {
  if (r <= 0) return mask;
  let a = Uint8Array.from(mask), b = new Uint8Array(W * H);
  const rad = Math.max(1, Math.round(r));
  for (let pass = 0; pass < 2; pass++) {
    // horizontal
    for (let y = 0; y < H; y++) {
      const row = y * W;
      let sum = 0;
      for (let x = -rad; x <= rad; x++) sum += a[row + Math.min(W - 1, Math.max(0, x))];
      for (let x = 0; x < W; x++) {
        b[row + x] = sum > 0 ? 1 : 0;
        const add = a[row + Math.min(W - 1, x + rad + 1)], sub = a[row + Math.max(0, x - rad)];
        sum += add - sub;
      }
    }
    [a, b] = [b, a];
    // vertical
    for (let x = 0; x < W; x++) {
      let sum = 0;
      for (let y = -rad; y <= rad; y++) sum += a[Math.min(H - 1, Math.max(0, y)) * W + x];
      for (let y = 0; y < H; y++) {
        b[y * W + x] = sum > 0 ? 1 : 0;
        const add = a[Math.min(H - 1, y + rad + 1) * W + x], sub = a[Math.max(0, y - rad) * W + x];
        sum += add - sub;
      }
    }
    [a, b] = [b, a];
  }
  return a;
}

/** Feather a binary mask into 0..1 coverage with `r` box-blur passes. */
export function featherMask(mask, W, H, r, passes = 2) {
  let a = new Float32Array(W * H);
  for (let i = 0; i < mask.length; i++) a[i] = mask[i];
  const b = new Float32Array(W * H);
  const rad = Math.max(1, Math.round(r)), inv = 1 / (2 * rad + 1);
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < H; y++) {                       // horizontal: a -> b
      const row = y * W;
      let sum = 0;
      for (let x = -rad; x <= rad; x++) sum += a[row + Math.min(W - 1, Math.max(0, x))];
      for (let x = 0; x < W; x++) {
        b[row + x] = sum * inv;
        sum += a[row + Math.min(W - 1, x + rad + 1)] - a[row + Math.max(0, x - rad)];
      }
    }
    for (let x = 0; x < W; x++) {                        // vertical: b -> a
      let sum = 0;
      for (let y = -rad; y <= rad; y++) sum += b[Math.min(H - 1, Math.max(0, y)) * W + x];
      for (let y = 0; y < H; y++) {
        a[y * W + x] = sum * inv;
        sum += b[Math.min(H - 1, y + rad + 1) * W + x] - b[Math.max(0, y - rad) * W + x];
      }
    }
  }
  return a;
}

function toFloat(rgba, w, h) {
  const c = new Float32Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) { c[i * 3] = rgba[j]; c[i * 3 + 1] = rgba[j + 1]; c[i * 3 + 2] = rgba[j + 2]; }
  return c;
}

function downLevel(c, m, w, h) {
  const w2 = Math.max(1, w >> 1), h2 = Math.max(1, h >> 1);
  const c2 = new Float32Array(w2 * h2 * 3), m2 = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    let r = 0, g = 0, b = 0, wt = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const sx = Math.min(w - 1, x * 2 + dx), sy = Math.min(h - 1, y * 2 + dy), i = sy * w + sx;
      const k = 1 - m[i];
      if (k > 0) { r += c[i * 3] * k; g += c[i * 3 + 1] * k; b += c[i * 3 + 2] * k; wt += k; }
    }
    const j = y * w2 + x;
    if (wt > 0.001) { c2[j * 3] = r / wt; c2[j * 3 + 1] = g / wt; c2[j * 3 + 2] = b / wt; m2[j] = 0; }
    else m2[j] = 1;
  }
  return { c: c2, m: m2, w: w2, h: h2 };
}

function upSample(c, m, w, h, W, H) {
  // bilinear resample of the coarse solution, used to seed the finer level
  const out = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    const fy = Math.min(h - 1.001, (y + 0.5) * h / H - 0.5), y0 = Math.max(0, Math.floor(fy)), y1 = Math.min(h - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(w - 1.001, (x + 0.5) * w / W - 0.5), x0 = Math.max(0, Math.floor(fx)), x1 = Math.min(w - 1, x0 + 1), wx = fx - x0;
      const i00 = y0 * w + x0, i01 = y0 * w + x1, i10 = y1 * w + x0, i11 = y1 * w + x1;
      const o = (y * W + x) * 3;
      for (let k = 0; k < 3; k++) {
        const a = c[i00 * 3 + k] * (1 - wx) + c[i01 * 3 + k] * wx;
        const b = c[i10 * 3 + k] * (1 - wx) + c[i11 * 3 + k] * wx;
        out[o + k] = a * (1 - wy) + b * wy;
      }
    }
  }
  return out;
}

/** Gauss-Seidel diffusion over the hole cells (harmonic inpainting). */
function diffuse(c, m, w, h, iters) {
  for (let it = 0; it < iters; it++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (m[i] < 0.5) continue;
        let r = 0, g = 0, b = 0, n = 0;
        if (x > 0) { const j = (i - 1) * 3; r += c[j]; g += c[j + 1]; b += c[j + 2]; n++; }
        if (x < w - 1) { const j = (i + 1) * 3; r += c[j]; g += c[j + 1]; b += c[j + 2]; n++; }
        if (y > 0) { const j = (i - w) * 3; r += c[j]; g += c[j + 1]; b += c[j + 2]; n++; }
        if (y < h - 1) { const j = (i + w) * 3; r += c[j]; g += c[j + 1]; b += c[j + 2]; n++; }
        if (!n) continue;
        c[i * 3] = r / n; c[i * 3 + 1] = g / n; c[i * 3 + 2] = b / n;
      }
    }
  }
}

/**
 * @param src  {data:Uint8ClampedArray,width,height} RGBA source
 * @param mask Uint8Array w*h, 1 = remove (head interior)
 * @returns Uint8ClampedArray RGBA plate (same size), plus the float mask used
 */
export function inpaintPlate(src, mask, opts = {}) {
  const W = src.width, H = src.height;
  const levels = Math.max(1, Math.min(5, opts.levels ?? 4));
  const iters = opts.iters ?? 16;
  const out = Uint8ClampedArray.from(src.data);
  const pyramid = [{ c: toFloat(src.data, W, H), m: Float32Array.from(mask), w: W, h: H }];
  for (let l = 1; l < levels; l++) {
    const p = pyramid[l - 1];
    if (p.w < 8 || p.h < 8) break;
    pyramid.push(downLevel(p.c, p.m, p.w, p.h));
  }
  // seed the coarsest level with the mean background colour, then refine down
  const top = pyramid[pyramid.length - 1];
  let holes = 0, mean = [0, 0, 0], known = 0;
  for (let i = 0; i < top.w * top.h; i++) if (top.m[i] > 0.5) holes++; else { mean[0] += top.c[i * 3]; mean[1] += top.c[i * 3 + 1]; mean[2] += top.c[i * 3 + 2]; known++; }
  if (known) for (let k = 0; k < 3; k++) mean[k] /= known;
  if (holes === top.w * top.h) return { data: out, mask, coverage: 1 }; // whole frame is head: nothing to inpaint
  for (let i = 0; i < top.w * top.h; i++) if (top.m[i] > 0.5) { top.c[i * 3] = mean[0]; top.c[i * 3 + 1] = mean[1]; top.c[i * 3 + 2] = mean[2]; }
  diffuse(top.c, top.m, top.w, top.h, iters + 8);
  for (let l = pyramid.length - 2; l >= 0; l--) {
    const cur = pyramid[l], finer = pyramid[l + 1];
    const up = upSample(finer.c, finer.m, finer.w, finer.h, cur.w, cur.h);
    for (let i = 0; i < cur.w * cur.h; i++) if (cur.m[i] > 0.5) { cur.c[i * 3] = up[i * 3]; cur.c[i * 3 + 1] = up[i * 3 + 1]; cur.c[i * 3 + 2] = up[i * 3 + 2]; }
    diffuse(cur.c, cur.m, cur.w, cur.h, iters);
  }
  const base = pyramid[0];
  const blur = opts.blur ?? 1;
  for (let i = 0, j = 0; i < W * H; i++, j += 4) {
    if (mask[i] < 0.5) continue;
    out[j] = base.c[i * 3]; out[j + 1] = base.c[i * 3 + 1]; out[j + 2] = base.c[i * 3 + 2]; out[j + 3] = 255;
  }
  if (blur > 0) softBlurRegion(out, mask, W, H, blur);
  return { data: out, mask, coverage: holes / (top.w * top.h) };
}

/** Blur only inside the filled region (fake depth of field hides solve artefacts). */
export function softBlurRegion(rgba, mask, W, H, radius) {
  const r = Math.max(1, Math.round(radius)), norm = 1 / (2 * r + 1);
  const tmp = Uint8ClampedArray.from(rgba);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {   // horizontal: rgba -> tmp
    const i = y * W + x;
    if (mask[i] < 0.5) continue;
    let R = 0, G = 0, B = 0;
    for (let k = -r; k <= r; k++) {
      const j = (y * W + Math.min(W - 1, Math.max(0, x + k))) * 4;
      R += rgba[j]; G += rgba[j + 1]; B += rgba[j + 2];
    }
    const o = i * 4;
    tmp[o] = R * norm; tmp[o + 1] = G * norm; tmp[o + 2] = B * norm; tmp[o + 3] = 255;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {   // vertical: tmp -> rgba
    const i = y * W + x;
    if (mask[i] < 0.5) continue;
    let R = 0, G = 0, B = 0;
    for (let k = -r; k <= r; k++) {
      const j = (Math.min(H - 1, Math.max(0, y + k)) * W + x) * 4;
      R += tmp[j]; G += tmp[j + 1]; B += tmp[j + 2];
    }
    const o = i * 4;
    rgba[o] = R * norm; rgba[o + 1] = G * norm; rgba[o + 2] = B * norm;
  }
}
