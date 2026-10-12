/**
 * emotions.js — 7 FACS-derived emotion presets over the 52 ARKit channels
 * (spec §5.7) + cosine-similarity read-back. Pure math: no DOM/THREE (§9.1).
 *
 * No FER network ships here — read-back is local linear algebra on the
 * calibrated 52-vector ("Emotion read-back (local)" in the UI).
 */

import { DampedSpring } from './smoothing.js';
import { Rng, makePinkNoise } from './eyes.js';

/** @type {Record<string, number>} */
export const CONFIG = {
  springOmega: 2.5,      // ≈0.8 s transition with 8% overshoot (§5.7)
  springZeta: 0.7,
  readbackEma: 0.12,     // EMA α on similarity scores
  readbackHystSec: 0.25, // label hysteresis
  microBreathAmp: 0.04,  // idle emotionWeight self-noise ±0.04 (§5.7)
  microBreathSec: 15,    // 15 s pink cycle
  intensityMin: 0.6,     // read-back only reports ≥60% matches (test contract)
};

/**
 * Per-emotion channel weights (§5.7). Bilateral pairs identical except
 * contempt (LEFT ONLY — that asymmetry is its signature).
 * Values are the preset at full intensity.
 * @type {Record<string, Record<string, number>>}
 */
export const PRESETS = {
  joy: {
    mouthSmileLeft: 0.85, mouthSmileRight: 0.85,
    cheekSquintLeft: 0.55, cheekSquintRight: 0.55,
    mouthDimpleLeft: 0.3, mouthDimpleRight: 0.3,
    eyeSquintLeft: 0.35, eyeSquintRight: 0.35,
    browInnerUp: 0.05,
  },
  sadness: {
    browInnerUp: 0.55,
    browDownLeft: 0.15, browDownRight: 0.15,
    mouthFrownLeft: 0.5, mouthFrownRight: 0.5,
    mouthLowerDownLeft: 0.15, mouthLowerDownRight: 0.15,
    eyeSquintLeft: 0.2, eyeSquintRight: 0.2,
    mouthClose: 0.1,
  },
  anger: {
    browDownLeft: 0.8, browDownRight: 0.8,
    eyeSquintLeft: 0.45, eyeSquintRight: 0.45,
    mouthPressLeft: 0.5, mouthPressRight: 0.5,
    mouthFunnel: 0.25,
    jawOpen: 0.12,
    noseSneerLeft: 0.2, noseSneerRight: 0.2,
  },
  fear: {
    browInnerUp: 0.7,
    browOuterUpLeft: 0.65, browOuterUpRight: 0.65,
    eyeWideLeft: 0.7, eyeWideRight: 0.7,
    mouthStretchLeft: 0.45, mouthStretchRight: 0.45,
    jawOpen: 0.25,
  },
  disgust: {
    noseSneerLeft: 0.7, noseSneerRight: 0.7,
    mouthUpperUpLeft: 0.6, mouthUpperUpRight: 0.6,
    browDownLeft: 0.45, browDownRight: 0.45,
    tongueOut: 0.15,
    cheekPuff: 0.1,
  },
  surprise: {
    browInnerUp: 0.75,
    browOuterUpLeft: 0.7, browOuterUpRight: 0.7,
    eyeWideLeft: 0.65, eyeWideRight: 0.65,
    jawOpen: 0.6,
    mouthStretchLeft: 0.2, mouthStretchRight: 0.2,
  },
  contempt: {
    // LEFT ONLY by design (§5.7) — do not mirror this.
    mouthSmileLeft: 0.35,
    mouthDimpleLeft: 0.3,
  },
};

/** How much camera mouth motion leaks through each preset (§5.7). */
export const EYE_MOUTH_PRESERVE = {
  joy: 0.8,
  surprise: 0.5,
  fear: 0.35,
  default: 0.15,
};

/** Display order for chips / keys 1..7 (§10). */
export const EMOTION_ORDER = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'contempt'];

/** UI chip labels. */
export const EMOTION_LABELS = {
  joy: 'Joy', sadness: 'Sad', anger: 'Anger', fear: 'Fear',
  disgust: 'Disgust', surprise: 'Surprise', contempt: 'Contempt',
};

/**
 * Build a 52-vector for a preset.
 * @param {string} emotion preset id
 * @param {string[]} channels ordered ARKit names
 * @returns {Float32Array}
 */
export function presetVector(emotion, channels) {
  const p = PRESETS[emotion] ?? {};
  const v = new Float32Array(channels.length);
  for (let i = 0; i < channels.length; i++) v[i] = p[channels[i]] ?? 0;
  return v;
}

/**
 * Cosine similarity of two vectors.
 * @param {Float32Array|number[]} a
 * @param {Float32Array|number[]} b
 * @returns {number} in [-1, 1] (0 for zero vectors)
 */
export function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na < 1e-12 || nb < 1e-12) return 0;
  return dot / Math.sqrt(na * nb);
}

/**
 * Emotion read-back state: EMA'd similarities + label hysteresis (§5.7).
 */
export class EmotionReadback {
  /**
   * @param {string[]} channels ordered ARKit names
   */
  constructor(channels) {
    this.channels = channels;
    this.templates = EMOTION_ORDER.map((e) => ({ id: e, v: presetVector(e, channels) }));
    this.ema = new Float32Array(this.templates.length);
    this.label = 'none';
    this.score = 0;
    this._holdT = 0;
  }

  /**
   * Feed the calibrated 52-vector.
   * @param {Float32Array} weights
   * @param {number} dtSec
   * @returns {{label: string, score: number}} smoothed read-back
   */
  update(weights, dtSec) {
    const a = CONFIG.readbackEma;
    let best = 0;
    let bestI = -1;
    for (let i = 0; i < this.templates.length; i++) {
      const sim = cosineSimilarity(weights, this.templates[i].v);
      this.ema[i] += (sim - this.ema[i]) * a;
      if (this.ema[i] > best) {
        best = this.ema[i];
        bestI = i;
      }
    }
    const cand = bestI >= 0 && best > 0.35 ? this.templates[bestI].id : 'none';
    // 0.25 s hysteresis before switching labels (§5.7).
    if (cand !== this.label) {
      this._holdT += dtSec;
      if (this._holdT >= CONFIG.readbackHystSec) {
        this.label = cand;
        this._holdT = 0;
      }
    } else {
      this._holdT = 0;
    }
    this.score = this.label === 'none' ? 0 : best;
    return { label: this.label, score: this.score };
  }

  /**
   * One-shot classification without state (tests).
   * @param {Float32Array} weights
   * @returns {string} label
   */
  classifyOnce(weights) {
    let best = -Infinity;
    let bestI = -1;
    for (let i = 0; i < this.templates.length; i++) {
      const sim = cosineSimilarity(weights, this.templates[i].v);
      if (sim > best) {
        best = sim;
        bestI = i;
      }
    }
    return bestI >= 0 && best > 0.35 ? this.templates[bestI].id : 'none';
  }
}

/**
 * Emotion driver: selected preset + intensity, under-damped spring
 * transitions (ζ=0.7, ω=2.5 → 8% overshoot), idle micro-breathing.
 * Outputs the `emotion` producer layer for rig.js (§5.3).
 */
export class EmotionController {
  /**
   * @param {string[]} channels ordered ARKit names
   */
  constructor(channels) {
    this.channels = channels;
    this.n = channels.length;
    /** @type {Float32Array} current sprung layer (producer output) */
    this.layer = new Float32Array(this.n);
    this.springs = channels.map(() => new DampedSpring(CONFIG.springOmega, CONFIG.springZeta));
    this.emotion = 'none';
    this.intensity = 1;
    this.solo = false;
    this._target = new Float32Array(this.n);
    this._pink = makePinkNoise(new Rng(0xE0E0));
    this._pinkT = 0;
    this._micro = 0;
    this._vec = new Float32Array(this.n);
  }

  /**
   * Select an emotion ('none' releases toward zero).
   * @param {string} emotion
   */
  setEmotion(emotion) {
    this.emotion = PRESETS[emotion] ? emotion : 'none';
    this._rebuildTarget();
  }

  /**
   * @param {number} intensity 0..1 (the "Intensity" slider / emotionWeight)
   */
  setIntensity(intensity) {
    this.intensity = Math.min(Math.max(intensity, 0), 1);
    this._rebuildTarget();
  }

  _rebuildTarget() {
    const v = this.emotion === 'none' ? null : presetVector(this.emotion, this.channels);
    for (let i = 0; i < this.n; i++) {
      this._target[i] = v ? v[i] * this.intensity : 0;
    }
  }

  /**
   * Advance one substep; springs `layer` toward the preset target.
   * @param {number} dtSec
   * @returns {Float32Array} this.layer (do not copy — reused, §11.12)
   */
  update(dtSec) {
    // Idle micro-emotion breathing: ±0.04 on a 15 s pink cycle (§5.7).
    this._pinkT += dtSec;
    if (this._pinkT >= 1 / CONFIG.springOmega) {
      this._pinkT = 0;
      this._micro = this._pink() * CONFIG.microBreathAmp * 2;
    }
    const microScale = 1 + this._micro;
    for (let i = 0; i < this.n; i++) {
      const sp = this.springs[i];
      this.layer[i] = sp.step(this._target[i] * microScale, dtSec);
      if (this.layer[i] < 0) this.layer[i] = 0;
    }
    return this.layer;
  }

  /**
   * How much camera mouth motion leaks through the current preset (§5.7).
   * @returns {number} 0..1
   */
  mouthPreserve() {
    return EYE_MOUTH_PRESERVE[this.emotion] ?? EYE_MOUTH_PRESERVE.default;
  }
}
