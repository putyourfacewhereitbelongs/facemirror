/**
 * Background plate: mask rasterisation, dilation, feathering and the multi-scale
 * harmonic inpaint that lets the head turn without ghosting itself.
 */
import { describe, it, expect } from 'vitest';
import { rasterizePolygon, dilateMask, featherMask, inpaintPlate, softBlurRegion } from './plate.js';

const W = 64, H = 64;
const square = (x0, y0, x1, y1) => [x0, y0, x1, y0, x1, y1, x0, y1];
const count = m => m.reduce((a, b) => a + (b > 0.5 ? 1 : 0), 0);

function flatImage(w, h, r, g, b) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255; }
  return { data, width: w, height: h };
}

describe('rasterizePolygon', () => {
  it('fills the interior and leaves the exterior empty', () => {
    const m = rasterizePolygon(W, H, square(16, 16, 48, 48));
    expect(m[32 * W + 32]).toBe(1);
    expect(m[2 * W + 2]).toBe(0);
    expect(m.length).toBe(W * H);
    // within a couple of rows of the exact area (scanline sampling)
    expect(count(m)).toBeGreaterThan(30 * 30);
    expect(count(m)).toBeLessThan(34 * 34);
  });
  it('handles a concave polygon with the even-odd rule', () => {
    const star = [32, 4, 40, 28, 60, 32, 40, 40, 32, 60, 24, 40, 4, 32, 24, 28];
    const m = rasterizePolygon(W, H, star);
    expect(m[32 * W + 32]).toBe(1);
    expect(m[1 * W + 1]).toBe(0);
    expect(count(m)).toBeLessThan(W * H * 0.5);
  });
  it('clips to the canvas instead of throwing', () => {
    const m = rasterizePolygon(W, H, square(-40, -40, 200, 200));
    expect(count(m)).toBe(W * H);
    expect(() => rasterizePolygon(W, H, [1, 1, 2, 2])).not.toThrow();
    expect(count(rasterizePolygon(W, H, [1, 1, 2, 2]))).toBe(0);
  });
});

describe('dilateMask', () => {
  it('grows the mask and never erodes it', () => {
    const m = rasterizePolygon(W, H, square(24, 24, 40, 40));
    const before = count(m);
    const d = dilateMask(m, W, H, 4);
    expect(count(d)).toBeGreaterThan(before + 40);
    for (let i = 0; i < m.length; i++) if (m[i]) expect(d[i]).toBe(1);
  });
  it('does not mutate the caller\'s mask', () => {
    const m = rasterizePolygon(W, H, square(24, 24, 40, 40));
    const before = count(m);
    dilateMask(m, W, H, 5);
    expect(count(m)).toBe(before);
  });
  it('is a no-op for a zero radius', () => {
    const m = rasterizePolygon(W, H, square(24, 24, 40, 40));
    expect(dilateMask(m, W, H, 0)).toBe(m);
  });
});

describe('featherMask', () => {
  it('produces a soft 0..1 ramp across the boundary', () => {
    const m = rasterizePolygon(W, H, square(16, 16, 48, 48));
    const f = featherMask(m, W, H, 3, 2);
    expect(f[32 * W + 32]).toBeGreaterThan(0.9);      // deep inside
    expect(f[2 * W + 2]).toBeLessThan(0.05);          // far outside
    let mid = 0;
    for (const v of f) if (v > 0.15 && v < 0.85) mid++;
    expect(mid).toBeGreaterThan(50);                  // a real gradient, not a step
    for (const v of f) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1.0001); }
  });
});

describe('inpaintPlate', () => {
  it('fills a hole with the surrounding colour and keeps the outside untouched', () => {
    const src = flatImage(W, H, 90, 120, 160);
    const mask = rasterizePolygon(W, H, square(20, 20, 44, 44));
    // paint garbage into the hole first: the solve must not read it back
    for (let y = 20; y < 44; y++) for (let x = 20; x < 44; x++) {
      const i = (y * W + x) * 4;
      src.data[i] = 255; src.data[i + 1] = 0; src.data[i + 2] = 255;
    }
    const res = inpaintPlate(src, mask, { levels: 4, iters: 16 });
    const i = (32 * W + 32) * 4;
    expect(Math.abs(res.data[i] - 90)).toBeLessThan(10);
    expect(Math.abs(res.data[i + 1] - 120)).toBeLessThan(10);
    expect(Math.abs(res.data[i + 2] - 160)).toBeLessThan(10);
    expect(res.data[i + 3]).toBe(255);
    // outside the mask the photo is byte-identical
    const o = (2 * W + 2) * 4;
    expect(res.data[o]).toBe(90); expect(res.data[o + 1]).toBe(120); expect(res.data[o + 2]).toBe(160);
    expect(res.coverage).toBeGreaterThan(0.05);
    expect(res.coverage).toBeLessThan(0.5);
  });

  it('propagates a gradient instead of a flat plug', () => {
    const src = flatImage(W, H, 0, 0, 0);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const v = x < W / 2 ? 40 : 200;                  // left dark, right bright
      src.data[i] = src.data[i + 1] = src.data[i + 2] = v; src.data[i + 3] = 255;
    }
    const mask = rasterizePolygon(W, H, square(24, 8, 40, 56));
    const res = inpaintPlate(src, mask, { levels: 4, iters: 24, blur: 0 });
    const at = x => res.data[(32 * W + x) * 4];
    expect(at(26)).toBeLessThan(at(38));               // the ramp survives across the hole
    expect(at(38) - at(26)).toBeGreaterThan(20);
  });

  it('survives a fully masked frame', () => {
    const src = flatImage(32, 32, 10, 20, 30);
    const mask = new Uint8Array(32 * 32).fill(1);
    const res = inpaintPlate(src, mask, { levels: 3, iters: 6 });
    expect(res.data.length).toBe(32 * 32 * 4);
    expect(res.coverage).toBe(1);
  });

  it('softBlurRegion only touches the filled region', () => {
    const src = flatImage(W, H, 90, 120, 160);
    const mask = rasterizePolygon(W, H, square(20, 20, 44, 44));
    for (let y = 20; y < 44; y++) for (let x = 20; x < 44; x++) {
      const i = (y * W + x) * 4;
      src.data[i] = (x * 7) % 255; src.data[i + 1] = (y * 11) % 255; src.data[i + 2] = 128;
    }
    const copy = Uint8ClampedArray.from(src.data);
    softBlurRegion(src.data, mask, W, H, 2);
    // a pixel well outside the mask is unchanged
    const o = (4 * W + 4) * 4;
    expect(src.data[o]).toBe(copy[o]);
    expect(src.data[o + 1]).toBe(copy[o + 1]);
    // the noisy interior got smoother
    let varBefore = 0, varAfter = 0;
    for (let y = 24; y < 40; y++) for (let x = 24; x < 40; x++) {
      const i = (y * W + x) * 4;
      varBefore += Math.abs(copy[i] - copy[i + 4] || 0);
      varAfter += Math.abs(src.data[i] - src.data[i + 4] || 0);
    }
    expect(varAfter).toBeLessThan(varBefore);
  });
});
