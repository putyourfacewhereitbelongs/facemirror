/**
 * Audio-driven visemes: band analysis, pitch tracking and the mapping onto jaw /
 * lip / tongue parameters. Everything here is deterministic DSP, so it can be
 * tested with synthesised spectra instead of a microphone.
 */
import { describe, it, expect } from 'vitest';
import { BANDS, bandEnergies, spectralCentroid, estimateF0, bandsToViseme, pickViseme, VISEMES, AudioEngine } from './audio.js';

const SR = 48000, FFT = 2048;
const bins = FFT / 2;
const binHz = SR / FFT;

/** A dB spectrum with `lo`..`hi` Hz energised and everything else at the floor. */
function spectrum(ranges, levelDb = -30, floorDb = -120) {
  const out = new Float32Array(bins).fill(floorDb);
  for (const [lo, hi] of ranges) {
    for (let i = Math.floor(lo / binHz); i <= Math.min(bins - 1, Math.ceil(hi / binHz)); i++) out[i] = levelDb;
  }
  return out;
}
const settle = (bands, frames = 12, dt = 1 / 60, opts = {}) => {
  const st = {};
  const E = { bands, total: bands.reduce((a, b) => a + b, 0) };
  let v = null;
  for (let k = 0; k < frames; k++) v = bandsToViseme(E, st, dt, opts);
  return v;
};

describe('band analysis', () => {
  it('assigns energy to the right formant band', () => {
    const E = bandEnergies(spectrum([[300, 850]]), SR, FFT);
    expect(E.bands.length).toBe(BANDS.length);
    const top = E.bands.indexOf(Math.max(...E.bands));
    expect(top).toBe(1);                              // F1 openness
    expect(E.bands[1]).toBeGreaterThan(E.bands[3] * 20);
    expect(E.total).toBeGreaterThan(0);
  });

  it('separates a sibilant from a vowel', () => {
    const s = bandEnergies(spectrum([[3500, 6000]]), SR, FFT);
    const v = bandEnergies(spectrum([[300, 850]]), SR, FFT);
    expect(s.bands.indexOf(Math.max(...s.bands))).toBe(4);   // F3 fricative
    expect(v.bands.indexOf(Math.max(...v.bands))).toBe(1);
  });

  it('reports silence as (near) zero energy', () => {
    const E = bandEnergies(new Float32Array(bins).fill(-120), SR, FFT);
    expect(E.total).toBeLessThan(1e-4);
  });

  it('measures a brighter spectrum with a higher centroid', () => {
    const dark = spectralCentroid(spectrum([[100, 400]]), SR, FFT);
    const bright = spectralCentroid(spectrum([[4000, 7000]]), SR, FFT);
    expect(dark).toBeGreaterThan(100);
    expect(dark).toBeLessThan(600);
    expect(bright).toBeGreaterThan(4000);
  });
});

describe('pitch tracking', () => {
  const sine = (f0, sr, n, amp = 0.6) => {
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = amp * Math.sin(2 * Math.PI * f0 * i / sr);
    return x;
  };
  it('finds a 150 Hz voice', () => {
    expect(estimateF0(sine(150, SR, 2048), SR)).toBeCloseTo(150, -1);
  });
  it('finds a 240 Hz voice', () => {
    const f = estimateF0(sine(240, SR, 2048), SR);
    expect(Math.abs(f - 240)).toBeLessThan(20);
  });
  it('returns 0 for silence and for out-of-range noise', () => {
    expect(estimateF0(new Float32Array(2048), SR)).toBe(0);
    expect(estimateF0(sine(150, SR, 64), SR)).toBe(0);      // too short to judge
  });
});

describe('viseme mapping', () => {
  it('opens the jaw on an open vowel', () => {
    const v = settle([0.10, 0.55, 0.12, 0.05, 0.02, 0.01]);
    expect(v.jaw).toBeGreaterThan(0.4);
    expect(v.name).toBe(VISEMES[6]);
  });

  it('spreads the lips on a front vowel more than on an open one', () => {
    const a = settle([0.10, 0.55, 0.12, 0.05, 0.02, 0.01]);
    const e = settle([0.15, 0.12, 0.50, 0.30, 0.05, 0.02]);
    expect(e.spread).toBeGreaterThan(a.spread + 0.2);
    expect(e.name).toBe(VISEMES[7]);
  });

  it('rounds the lips when F2 is low and voicing is strong', () => {
    const p = settle([0.40, 0.30, 0.05, 0.02, 0.02, 0.01]);
    expect(p.pucker).toBeGreaterThan(0.2);
    expect(p.name).toBe(VISEMES[8]);
  });

  it('flags fricatives from the high bands', () => {
    const s = settle([0.01, 0.02, 0.05, 0.20, 0.50, 0.35]);
    expect(s.fricative).toBeGreaterThan(0.4);
    expect(s.teeth).toBeGreaterThan(0.15);           // narrow opening, teeth near
    expect([VISEMES[3], VISEMES[2]]).toContain(s.name);
  });

  it('stays silent on silence', () => {
    const v = settle([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]);
    expect(v.level).toBeLessThan(0.05);
    expect(v.jaw).toBeLessThan(0.05);
    expect(v.name).toBe(VISEMES[0]);
  });

  it('detects a bilabial closure after voicing stops', () => {
    const st = {};
    const loud = { bands: [0.4, 0.4, 0.1, 0.05, 0.02, 0.01], total: 1 };
    const quiet = { bands: [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002], total: 0.001 };
    let v = null;
    for (let k = 0; k < 10; k++) v = bandsToViseme(loud, st, 1 / 60, {});
    expect(v.closure).toBeLessThan(0.3);
    for (let k = 0; k < 6; k++) v = bandsToViseme(quiet, st, 1 / 60, {});
    expect(v.closure).toBeGreaterThan(0.3);          // lips pressed for /p b m/
  });

  it('responds to the gain and to a forced tongue', () => {
    const bands = [0.10, 0.55, 0.12, 0.05, 0.02, 0.01];
    const lo = settle(bands, 12, 1 / 60, { gain: 0.4 });
    const hi = settle(bands, 12, 1 / 60, { gain: 1.6 });
    expect(hi.jaw).toBeGreaterThan(lo.jaw);
    expect(settle(bands, 12, 1 / 60, { forceTongue: true }).tongueOut).toBeGreaterThan(0.5);
    expect(settle(bands, 12, 1 / 60, {}).tongueOut).toBeLessThan(0.2);
  });

  it('never jumps to full amplitude in one frame (the strobe this replaces)', () => {
    const st = {};
    const quiet = { bands: [0, 0, 0, 0, 0, 0], total: 0 };
    const loud = { bands: [0.10, 0.55, 0.12, 0.05, 0.02, 0.01], total: 0.85 };
    bandsToViseme(quiet, st, 1 / 60, {});
    const first = bandsToViseme(loud, st, 1 / 60, {});
    const steady = settle([0.10, 0.55, 0.12, 0.05, 0.02, 0.01], 30);
    expect(first.jaw).toBeLessThan(steady.jaw * 0.75);
    expect(first.jaw).toBeGreaterThan(0);
  });

  it('pickViseme labels the parameter block', () => {
    expect(pickViseme({ level: 0, burst: 0, closure: 0, fricative: 0, voiced: 0, jaw: 0, spread: 0, pucker: 0, tip: 0, back: 0 })).toBe(VISEMES[0]);
    expect(pickViseme({ level: 0.6, burst: 0, closure: 0.9, fricative: 0, voiced: 1, jaw: 0.1, spread: 0, pucker: 0, tip: 0, back: 0 })).toBe(VISEMES[1]);
    expect(pickViseme({ level: 0.6, burst: 0, closure: 0, fricative: 0.9, voiced: 0, jaw: 0.1, spread: 0, pucker: 0, tip: 0, back: 0 })).toBe(VISEMES[3]);
    expect(pickViseme({ level: 0.6, burst: 0, closure: 0, fricative: 0, voiced: 1, jaw: 0.9, spread: 0, pucker: 0, tip: 0, back: 0 })).toBe(VISEMES[6]);
  });

  it('keeps every output in range over random spectra', () => {
    const st = {};
    for (let k = 0; k < 200; k++) {
      const bands = Array.from({ length: 6 }, (_, i) => Math.abs(Math.sin(k * 0.7 + i * 2.1)) * (i === 0 ? 0.5 : 1));
      const v = bandsToViseme({ bands, total: bands.reduce((a, b) => a + b, 0) }, st, 1 / 60, {});
      for (const key of ['jaw', 'spread', 'pucker', 'stretch', 'tongueTip', 'tongueBack', 'tongueOut', 'closure', 'teeth']) {
        expect(v[key]).toBeGreaterThanOrEqual(0);
        expect(v[key]).toBeLessThanOrEqual(1.3);
        expect(Number.isFinite(v[key])).toBe(true);
      }
      expect(VISEMES).toContain(v.name);
    }
  });
});

describe('audio plumbing', () => {
  it('is inert without a Web Audio context', () => {
    const a = new AudioEngine();
    expect(a.running).toBe(false);
    expect(a.update(1 / 60, {})).toBeNull();
    expect(a.audioTrack()).toBeNull();
    a.stop();                                        // must not throw
    expect(a.mode).toBe('off');
  });
  it('exposes six named bands in speech order', () => {
    expect(BANDS.length).toBe(6);
    for (let i = 1; i < BANDS.length; i++) expect(BANDS[i].lo).toBe(BANDS[i - 1].hi);
    expect(BANDS[0].lo).toBeLessThan(300);
    expect(BANDS[5].hi).toBeGreaterThan(8000);
  });
});
