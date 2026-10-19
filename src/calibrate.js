/**
 * calibrate.js — 60-frame neutral + range + wink wizard (spec §4).
 * Without calibration the app "feels broken" (§4 preamble): MediaPipe
 * under-reads channels, carries a noise floor, and can invert L/R naming
 * (trap §11.3). Pure math + session state machine: no DOM/THREE (§9.1).
 */

import { CHANNELS, CH } from './rig.js';

/** @type {Record<string, number>} */
export const CONFIG = {
  neutralFrames: 60,
  gestureFrames: 20,
  noiseSigmaMult: 2.5,   // nf = b0 + 2.5σ (§4.1)
  rangeFloor: 0.15,      // failed gesture can't make a channel dead (§4.2)
  p95: 0.95,
  storageKey: 'facemirror.calibration.v1',
};

/**
 * Wizard phases. UI presents these as 5 steps (neutral / mouth range /
 * eye+brow range / cheeks+wink / pose+save); each range step may hold
 * several sequential gestures.
 * @type {ReadonlyArray<{id: string, prompt: string, frames: number, kind: 'neutral'|'gesture'|'wink', gesture?: string}>}
 */
export const PHASES = Object.freeze([
  { id: 'neutral', prompt: 'Relax your face — neutral, lips together, eyes open', frames: CONFIG.neutralFrames, kind: 'neutral' },
  { id: 'openMouth', prompt: 'Open your mouth WIDE (jaw down) — hold', frames: CONFIG.gestureFrames, kind: 'gesture', gesture: 'openMouth' },
  { id: 'bigSmile', prompt: 'Big smile — hold', frames: CONFIG.gestureFrames, kind: 'gesture', gesture: 'bigSmile' },
  { id: 'squeezeEyes', prompt: 'Squeeze your eyes shut — hold', frames: CONFIG.gestureFrames, kind: 'gesture', gesture: 'squeezeEyes' },
  { id: 'raiseBrows', prompt: 'Raise your brows high — hold', frames: CONFIG.gestureFrames, kind: 'gesture', gesture: 'raiseBrows' },
  { id: 'puffCheeks', prompt: 'Puff your cheeks (lips sealed) — hold', frames: CONFIG.gestureFrames, kind: 'gesture', gesture: 'puffCheeks' },
  { id: 'winkLeft', prompt: 'Close your LEFT eye only (keep right open)', frames: CONFIG.gestureFrames, kind: 'wink' },
  { id: 'save', prompt: 'Done — saving profile', frames: 1, kind: 'neutral' },
]);

/** Which channels each gesture is allowed to calibrate (§4.2). */
export const GESTURE_CHANNELS = {
  openMouth: ['jawOpen', 'mouthFunnel', 'mouthStretchLeft', 'mouthStretchRight', 'jawForward', 'mouthClose'],
  bigSmile: ['mouthSmileLeft', 'mouthSmileRight', 'cheekSquintLeft', 'cheekSquintRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'eyeSquintLeft', 'eyeSquintRight'],
  squeezeEyes: ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight'],
  raiseBrows: ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'eyeWideLeft', 'eyeWideRight'],
  puffCheeks: ['cheekPuff'],
};

/**
 * Median of an unsorted numeric array.
 * @param {number[]} xs
 * @returns {number}
 */
export function median(xs) {
  if (xs.length === 0) return 0;
  const s = Float64Array.from(xs).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * p-quantile of an unsorted numeric array (linear interpolation).
 * @param {number[]} xs
 * @param {number} p 0..1
 * @returns {number}
 */
export function quantile(xs, p) {
  if (xs.length === 0) return 0;
  const s = Float64Array.from(xs).sort();
  const idx = (s.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

/**
 * Neutral statistics (§4.1): per-channel median b0 and noise floor nf=b0+2.5σ.
 * @param {Array<Float32Array|number[]>} frames collected neutral frames (52 each)
 * @returns {{b0: Float32Array, sigma: Float32Array, nf: Float32Array}}
 */
export function computeNeutralStats(frames) {
  const n = CHANNELS.length;
  const b0 = new Float32Array(n);
  const sigma = new Float32Array(n);
  const nf = new Float32Array(n);
  const col = new Array(frames.length);
  for (let c = 0; c < n; c++) {
    for (let f = 0; f < frames.length; f++) col[f] = frames[f][c];
    const med = median(col);
    let acc = 0;
    for (const v of col) acc += (v - med) * (v - med);
    const std = Math.sqrt(acc / Math.max(col.length - 1, 1));
    b0[c] = med;
    sigma[c] = std;
    nf[c] = med + CONFIG.noiseSigmaMult * std;
  }
  return { b0, sigma, nf };
}

/**
 * Range top for one gesture (§4.2): hi[c] = max(p95, b0+0.15) — the floor
 * keeps a failed gesture from zeroing the channel's usable span.
 * @param {Array<Float32Array|number[]>} frames gesture frames
 * @param {number} channelIdx
 * @param {number} b0c neutral baseline for the channel
 * @param {number} nfc noise floor for the channel
 * @returns {number} hi value
 */
export function computeRangeHi(frames, channelIdx, b0c, nfc) {
  const col = new Array(frames.length);
  for (let f = 0; f < frames.length; f++) col[f] = frames[f][channelIdx];
  const p = quantile(col, CONFIG.p95);
  // §4.2 formula read as: keep at least CONFIG.rangeFloor of usable span
  // above the noise floor even when the gesture failed.
  return Math.max(p, nfc + CONFIG.rangeFloor, b0c + CONFIG.rangeFloor);
}

/**
 * Wink map (§4.3): user closes their LEFT eye; whichever eyeBlink* channel
 * rises is what MediaPipe *calls* the left. If MediaPipe's naming disagrees
 * with the user's, blinkLRSwap = true and rig.js remaps every bilateral pair
 * at load time (trap §11.3 — never flip sliders later).
 *
 * @param {number} deltaLeft rise of eyeBlinkLeft during the wink
 * @param {number} deltaRight rise of eyeBlinkRight during the wink
 * @returns {boolean} blinkLRSwap
 */
export function computeBlinkSwap(deltaLeft, deltaRight) {
  return deltaRight > deltaLeft;
}

/**
 * Full calibration result.
 * @typedef {Object} CalibrationResult
 * @property {Float32Array} b0 neutral baseline
 * @property {Float32Array} nf noise floor
 * @property {Float32Array} hi range tops
 * @property {boolean} blinkLRSwap
 * @property {number} z0 neutral matrix translation z (§4.4)
 */

/**
 * Session state machine. The UI calls `begin()`/`pushFrame()`/`cancel()`;
 * `pushFrame` returns live progress so the wizard doubles as a debug view.
 */
export class CalibrationSession {
  /**
   * @param {object} [opts]
   * @param {string[]} [opts.phaseIds] subset/rerun of phases
   */
  constructor(opts = {}) {
    const wanted = opts.phaseIds ? new Set(opts.phaseIds) : null;
    this.phases = wanted ? PHASES.filter((p) => wanted.has(p.id)) : PHASES.slice();
    this.phaseIdx = -1;
    /** @type {Array<Float32Array|number[]>} */
    this.frames = [];
    /** @type {Array<Float32Array|number[]>} */
    this.neutralFrames = [];
    /** @type {Record<string, Array<Float32Array|number[]>>} */
    this.gestureFrames = {};
    this.blinkLeftBase = 0;
    this.blinkRightBase = 0;
    this.zSamples = [];
    this.done = false;
    this.result = /** @type {CalibrationResult|null} */ (null);
    this.phaseFrameCounts = {};
  }

  /** @returns {{id: string, prompt: string, frames: number, kind: string}|null} */
  current() {
    return this.phaseIdx >= 0 ? this.phases[this.phaseIdx] : null;
  }

  /** Start (or restart) the wizard. */
  begin() {
    this.phaseIdx = 0;
    this.done = false;
    this.result = null;
    this.frames = [];
    this.neutralFrames = [];
    this.gestureFrames = {};
    this.zSamples = [];
    this.phaseFrameCounts = {};
    return this.current();
  }

  /** Cancel (Esc / Cancel button) — state discarded, previous calibration kept. */
  cancel() {
    this.phaseIdx = -1;
    this.done = false;
    this.frames = [];
  }

  /**
   * Feed one new-frame detection.
   * @param {Float32Array|number[]} raw52 raw scores
   * @param {Float32Array|number[]|null} matrixData 16-float matrix or null
   * @returns {{finished: boolean, phase: object|null, progress: number, phaseDone: boolean}}
   */
  pushFrame(raw52, matrixData) {
    const phase = this.current();
    if (!phase || this.done) return { finished: true, phase: null, progress: 1, phaseDone: true };

    if (phase.kind === 'neutral' && phase.id === 'neutral') {
      this.neutralFrames.push(Float32Array.from(raw52));
      if (matrixData) this.zSamples.push(matrixData[14]);
    } else if (phase.kind === 'gesture') {
      const g = /** @type {string} */ (phase.gesture);
      (this.gestureFrames[g] ??= []).push(Float32Array.from(raw52));
    } else if (phase.kind === 'wink') {
      this.frames.push(Float32Array.from(raw52));
    } else if (phase.id === 'save') {
      // finalize below
    }

    const count = phase.kind === 'neutral' && phase.id === 'neutral'
      ? this.neutralFrames.length
      : phase.kind === 'gesture'
        ? (this.gestureFrames[/** @type {string} */ (phase.gesture)]?.length ?? 0)
        : phase.kind === 'wink' ? this.frames.length : phase.frames;
    this.phaseFrameCounts[phase.id] = count;

    const phaseDone = count >= phase.frames;
    if (phaseDone) {
      if (phase.kind === 'wink') this._finishWink();
      this.phaseIdx++;
      const next = this.current();
      if (!next) {
        this._finalize();
        return { finished: true, phase: null, progress: 1, phaseDone: true };
      }
      return { finished: false, phase: next, progress: 0, phaseDone: true };
    }
    return { finished: false, phase, progress: count / phase.frames, phaseDone: false };
  }

  _finishWink() {
    // Baselines from neutral medians; deltas during the wink (§4.3).
    const { b0 } = this.neutralFrames.length
      ? computeNeutralStats(this.neutralFrames)
      : { b0: new Float32Array(CHANNELS.length) };
    let lSum = 0;
    let rSum = 0;
    for (const f of this.frames) {
      lSum += f[CH.eyeBlinkLeft];
      rSum += f[CH.eyeBlinkRight];
    }
    const n = Math.max(this.frames.length, 1);
    this._blinkSwap = computeBlinkSwap(lSum / n - b0[CH.eyeBlinkLeft], rSum / n - b0[CH.eyeBlinkRight]);
    this.frames = [];
  }

  _finalize() {
    const n = CHANNELS.length;
    const { b0, nf } = computeNeutralStats(this.neutralFrames);
    const hi = new Float32Array(n).fill(0);
    for (let c = 0; c < n; c++) hi[c] = b0[c] + CONFIG.rangeFloor;
    for (const [g, frames] of Object.entries(this.gestureFrames)) {
      const names = GESTURE_CHANNELS[g] ?? [];
      for (const name of names) {
        const c = CH[name];
        if (c !== undefined) hi[c] = computeRangeHi(frames, c, b0[c], nf[c]);
      }
    }
    let z0 = 0;
    if (this.zSamples.length) z0 = median(this.zSamples);
    this.result = {
      b0,
      nf,
      hi,
      blinkLRSwap: !!this._blinkSwap,
      z0,
    };
    this.done = true;
    this.phaseIdx = -1;
  }

  /**
   * Serialize for localStorage.
   * @param {CalibrationResult} r
   * @returns {string} JSON
   */
  static serialize(r) {
    return JSON.stringify({
      v: 1,
      b0: Array.from(r.b0),
      nf: Array.from(r.nf),
      hi: Array.from(r.hi),
      blinkLRSwap: r.blinkLRSwap,
      z0: r.z0,
    });
  }

  /**
   * Deserialize from localStorage.
   * @param {string} json
   * @returns {CalibrationResult|null}
   */
  static deserialize(json) {
    try {
      const d = JSON.parse(json);
      if (!d || d.v !== 1) return null;
      return {
        b0: Float32Array.from(d.b0),
        nf: Float32Array.from(d.nf),
        hi: Float32Array.from(d.hi),
        blinkLRSwap: !!d.blinkLRSwap,
        z0: d.z0 ?? 0,
      };
    } catch {
      return null;
    }
  }

  /**
   * Save to localStorage (browser only; no-op in node tests).
   * @param {CalibrationResult} r
   */
  static save(r) {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(CONFIG.storageKey, CalibrationSession.serialize(r));
    }
  }

  /** @returns {CalibrationResult|null} */
  static load() {
    if (typeof localStorage === 'undefined') return null;
    return CalibrationSession.deserialize(localStorage.getItem(CONFIG.storageKey) ?? '');
  }
}
