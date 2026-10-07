/**
 * Integration test for the in-page self-test.
 *
 * `src/puppet/main.js` is the only module that touches the DOM, so nothing else
 * in this suite can prove that its import graph resolves, that its module-level
 * initialisers run, or that `runSelfTest()` - the diagnostic the user can click
 * in the browser - actually passes. This file stubs just enough DOM to import it
 * and then runs the real thing.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mockCanvas, mockCtx } from './synthetic.js';

const html = readFileSync(fileURLToPath(new URL('../../puppet.html', import.meta.url)), 'utf8');
const mainSrc = readFileSync(fileURLToPath(new URL('./main.js', import.meta.url)), 'utf8');

const CTX_W = 900, CTX_H = 1100;

function fakeElement(id) {
  const cv = mockCanvas(CTX_W, CTX_H);
  return Object.assign(cv, {
    id,
    value: '0', checked: false, textContent: '', innerHTML: '', hidden: false,
    style: {}, dataset: {}, width: CTX_W, height: CTX_H,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener() {}, removeEventListener() {}, appendChild() {}, setAttribute() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: CTX_W, height: CTX_H, right: CTX_W, bottom: CTX_H }),
    captureStream: () => ({ getTracks: () => [], getVideoTracks: () => [], getAudioTracks: () => [] }),
    play: async () => {}, pause() {},
  });
}

function installDom() {
  const cache = new Map();
  const handlers = [];
  const document = {
    getElementById: id => { if (!cache.has(id)) cache.set(id, fakeElement(id)); return cache.get(id); },
    createElement: tag => (tag === 'canvas' ? fakeElement('created') : fakeElement(tag)),
    querySelector: () => null, querySelectorAll: () => [],
    // "loading" keeps main.js from calling boot() (which would try to fetch the
    // MediaPipe WASM from the CDN); the DOMContentLoaded registration is still
    // exercised and asserted below.
    readyState: 'loading',
    addEventListener: (type, fn) => handlers.push([type, fn]), removeEventListener() {},
    body: fakeElement('body'), documentElement: fakeElement('html'),
  };
  globalThis.document = document;
  globalThis.window = globalThis;
  // Node >= 21 exposes `navigator` as a getter-only global; only patch what is missing.
  const nav = { mediaDevices: { getUserMedia: async () => { throw new Error('no camera in tests'); } }, userAgent: 'vitest' };
  try {
    if (!globalThis.navigator) Object.defineProperty(globalThis, 'navigator', { value: nav, configurable: true });
    else if (!globalThis.navigator.mediaDevices) Object.defineProperty(globalThis.navigator, 'mediaDevices', { value: nav.mediaDevices, configurable: true });
  } catch (e) { /* the app only touches navigator inside startCamera() */ }
  globalThis.HTMLCanvasElement = class {};
  globalThis.requestAnimationFrame = fn => setTimeout(() => fn(performance.now()), 16);
  globalThis.cancelAnimationFrame = id => clearTimeout(id);
  globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
  return { cache, handlers };
}

let main = null, handlers = [];

beforeAll(async () => {
  handlers = installDom().handlers;
  main = await import('./main.js');
});

describe('main.js module graph', () => {
  it('imports and exposes the debug surface', () => {
    expect(main).toBeTruthy();
    expect(typeof main.runSelfTest).toBe('function');
    expect(typeof main.boot).toBe('function');
    expect(typeof main.detectDepthSign).toBe('function');
    expect(window.__puppet).toBeTruthy();
    expect(main.PRESETS).toBeTruthy();
  });

  it('registers boot() on DOMContentLoaded instead of running it at import time', () => {
    expect(handlers.some(([type, fn]) => type === 'DOMContentLoaded' && fn === main.boot)).toBe(true);
  });
});

describe('runSelfTest()', () => {
  let out = null;

  it('runs every stage without throwing', () => {
    out = main.runSelfTest();
    expect(out.steps.length).toBeGreaterThanOrEqual(9);
    const failed = out.steps.filter(s => s.error).map(s => `${s.name}: ${s.error}`);
    expect(failed).toEqual([]);
    expect(out.ok).toBe(true);
  });

  it('proves the rest pose is an identity projection', () => {
    const s = out.steps.find(x => x.name.startsWith('rest projection'));
    expect(s.value).toBeLessThan(0.05);
  });

  it('proves the head turn is perspective, not a shear', () => {
    const s = out.steps.find(x => x.name.startsWith('yaw compresses'));
    // an orthographic shear would give exactly cos(yaw); perspective gives less
    expect(s.value.ratio).toBeLessThan(s.value.orthographic);
    expect(s.value.ratio).toBeGreaterThan(s.value.orthographic * 0.7);
  });

  it('proves the stretch limiter helps', () => {
    const s = out.steps.find(x => x.name.startsWith('stretch limiter'));
    expect(s.value.limited).toBeLessThan(s.value.unlimited);
    expect(s.value.limited).toBeLessThan(2.2);
  });

  it('proves the jaw hinge drops the chin', () => {
    const s = out.steps.find(x => x.name.startsWith('jaw hinge'));
    expect(s.value).toBeGreaterThan(1);
  });

  it('proves the dental arch is temporally stable under jitter', () => {
    const s = out.steps.find(x => x.name.startsWith('dental arch'));
    expect(s.value).toBeLessThan(1.8);
  });

  it('separates the visemes and stays quiet on the noise floor', () => {
    const s = out.steps.find(x => x.name.startsWith('audio visemes'));
    expect(s.value.silence).toBe('silence');
    expect(new Set([s.value.a, s.value.e, s.value.s, s.value.p]).size).toBe(4);
  });

  it('rebuilds a rig and inpaints a plate', () => {
    const rigStep = out.steps.find(x => x.name === 'build rig');
    expect(rigStep.value.vertices).toBeGreaterThan(600);
    expect(rigStep.value.triangles).toBeGreaterThan(400);
    const plate = out.steps.find(x => x.name.startsWith('plate inpainting'));
    expect(plate.value.centreError).toBeLessThan(24);
  });

  it('reports a plausible timing for each stage', () => {
    for (const s of out.steps) {
      expect(typeof s.ms).toBe('number');
      expect(s.ms).toBeGreaterThanOrEqual(0);
      expect(s.ms).toBeLessThan(20000);
    }
    expect(out.steps.reduce((a, b) => a + b.ms, 0)).toBeGreaterThan(0);   // timing is real, not 0
  });

  it('detects the detector depth sign', () => {
    const s = out.steps.find(x => x.name.startsWith('depth sign'));
    expect([1, -1]).toContain(s.value);
  });
});

describe('control surface wiring', () => {
  const idsIn = src => new Set([...src.matchAll(/\bid=["']([^"']+)["']/g)].map(m => m[1]));
  const refsIn = src => {
    const refs = new Map();
    const patterns = [/\$\(\s*'([^']+)'/g, /num\(\s*'([^']+)'/g, /chk\(\s*'([^']+)'/g, /getElementById\(\s*'([^']+)'/g];
    for (const re of patterns) for (const m of src.matchAll(re)) refs.set(m[1], (refs.get(m[1]) || 0) + 1);
    return refs;
  };

  it('every id main.js looks up exists in puppet.html', () => {
    const ids = idsIn(html), refs = refsIn(mainSrc);
    expect(refs.size).toBeGreaterThan(60);                 // the panel is not empty
    const missing = [...refs.keys()].filter(r => !ids.has(r));
    expect(missing).toEqual([]);
  });

  it('every id in puppet.html is used by main.js (no dead controls)', () => {
    const ids = idsIn(html), refs = refsIn(mainSrc);
    expect([...ids].filter(i => !refs.has(i))).toEqual([]);
  });

  it('exposes both pages as build inputs', () => {
    const cfg = readFileSync(fileURLToPath(new URL('../../vite.config.ts', import.meta.url)), 'utf8');
    expect(cfg).toMatch(/puppet/);
    expect(html).toMatch(/src\/puppet\/main\.js/);
  });

  it('ships the features the brief asks for', () => {
    for (const label of ['record', 'photo', 'showPts', 'minimize', 'sculpt', 'selfTest']) {
      expect(html.toLowerCase()).toContain(label.toLowerCase());
    }
  });
});
