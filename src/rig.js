/**
 * rig.js — the ONE source of truth (spec §1, §5).
 * `rig.weights` = Float32Array(52), `rig.pose` = {quat, pos} owned by pose.js.
 *
 * Every driver (camera, drag, emotion, record-playback, manual sliders, idle)
 * is a *producer* that writes a `weightsSource[producer]` layer. Layers are
 * merged explicitly every frame (§5.3):
 *
 *   w_final = pin ? manual : clamp01(max(cam·camW, emotion) + dragAdd)
 *
 * Keep it this explicit — implicit blending is the #1 source of bugs in face
 * apps (§1). Pure math: no DOM/THREE (§9.1).
 */

/** The 52 ARKit names, order fixed forever (index = channel id). */
export const CHANNELS = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
  'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
  'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
  'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
  'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
  'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper',
  'mouthShrugLower', 'mouthShrugUpper',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight',
  'noseSneerLeft', 'noseSneerRight', 'tongueOut',
];

/** name → index */
export const CH = Object.fromEntries(CHANNELS.map((n, i) => [n, i]));

/** Channel groups for sliders + per-group filters (§10). */
export const GROUPS = {
  brows: ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'],
  eyes: [
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
  ],
  cheeks: ['cheekPuff', 'cheekSquintLeft', 'cheekSquintRight'],
  jaw: ['jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose'],
  mouth: [
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
  ],
  nose: ['noseSneerLeft', 'noseSneerRight'],
  tongue: ['tongueOut'],
};

/** name → group key (for filters and UI). */
export const GROUP_OF = (() => {
  /** @type {Record<string, string>} */
  const m = {};
  for (const [g, names] of Object.entries(GROUPS)) for (const n of names) m[n] = g;
  return m;
})();

/** Bilateral pairs (Left/Right naming — remapped by blinkLRSwap, trap §11.3). */
export const BILATERAL_PAIRS = (() => {
  /** @type {Array<[number, number]>} */
  const pairs = [];
  for (const name of CHANNELS) {
    if (name.endsWith('Left')) {
      const right = name.slice(0, -4) + 'Right';
      if (CH[right] !== undefined) pairs.push([CH[name], CH[right]]);
    }
  }
  return pairs;
})();

/**
 * Per-group gain table (§5.1). Default 1.0.
 * @type {Record<string, number>}
 */
export const GAIN_TABLE = {
  jawOpen: 1.35,
  mouthSmileLeft: 1.45, mouthSmileRight: 1.45,
  mouthPucker: 1.3,
  eyeBlinkLeft: 1.15, eyeBlinkRight: 1.15,
  browDownLeft: 1.5, browDownRight: 1.5,
  browInnerUp: 1.5,
  eyeSquintLeft: 1.2, eyeSquintRight: 1.2,
  mouthStretchLeft: 1.4, mouthStretchRight: 1.4,
  mouthFunnel: 1.4,
};

/** Channels whose names swap when calibration proves L/R inverted. */
export function makeSideSwapMap(swap) {
  /** @type {Record<string, string>} */
  const map = {};
  if (!swap) return map;
  for (const [l, r] of BILATERAL_PAIRS) {
    map[CHANNELS[l]] = CHANNELS[r];
    map[CHANNELS[r]] = CHANNELS[l];
  }
  return map;
}

/** Default per-channel gain (§5.1). */
export function gainOf(name) {
  return GAIN_TABLE[name] ?? 1.0;
}

/**
 * §5.1 normalize: v raw ∈[0,1] → n = clamp((v − nf)/(hi − nf), 0, 1).
 * @param {number} v raw score
 * @param {number} nf noise floor
 * @param {number} hi calibrated range top
 * @returns {number}
 */
export function normalizeChannel(v, nf, hi) {
  const span = Math.max(hi - nf, 1e-3);
  const n = (v - nf) / span;
  return n < 0 ? 0 : n > 1 ? 1 : n;
}

/**
 * §5.1 bilateral symmetrize: avg=(L+R)/2; L'=avg*0.75+L*0.25 (the default
 * "Asymmetry" 0.25 feel); slider asymmetry 0..1 mixes 1..0 between the
 * symmetric blend and fully independent sides.
 *
 * @param {Float32Array} w weights in/out (both pair slots)
 * @param {number} asym 0 = fully symmetric, 1 = raw asymmetric
 */
export function symmetrize(w, asym) {
  const mixSym = 1 - Math.min(Math.max(asym, 0), 1); // 1 at asym=0
  for (const [li, ri] of BILATERAL_PAIRS) {
    const L = w[li];
    const R = w[ri];
    const avg = (L + R) / 2;
    const targetL = avg * 0.75 + L * 0.25;
    const targetR = avg * 0.75 + R * 0.25;
    w[li] = L + (targetL - L) * mixSym;
    w[ri] = R + (targetR - R) * mixSym;
  }
}

/**
 * §5.6 mouth extras: mouthClose ∩ jawOpen mutual attenuation so they never
 * fight. `min(open−0.15, 0)`-style overlap removal, then jawOpen cap when the
 * asset has no teeth (trap §11.8).
 *
 * @param {Float32Array} w weights in/out
 * @param {boolean} hasTeeth asset has teeth/tongue that follow jawOpen
 */
export function applyMouthExtras(w, hasTeeth) {
  const open = w[CH.jawOpen];
  const close = w[CH.mouthClose];
  if (open > 0.05 && close > 0.05) {
    // Overlap rule: the more one is active, the more the other is pushed down.
    w[CH.jawOpen] = Math.max(0, open - close * 0.6);
    w[CH.mouthClose] = Math.max(0, close - open * 0.6);
  }
  if (!hasTeeth && w[CH.jawOpen] > 0.65) w[CH.jawOpen] = 0.65;
}

/**
 * The rig: 52-channel state + producer layers + merge (§5.3).
 */
export class Rig {
  constructor() {
    this.n = CHANNELS.length;
    /** @type {Float32Array} final output weights (THE state). */
    this.weights = new Float32Array(this.n);
    /**
     * Producer layers. Each is a Float32Array(52) of already-normalized
     * 0..1 channel weights unless noted:
     * - cam: post §5.1 mapping (normalize→symmetrize→gain) of camera scores
     * - emotion: sprung preset layer (EmotionController.layer)
     * - eyes: lid/look outputs from EyeSolver (blink channels etc.)
     * - manual: slider values
     * - drag: dragAdd increments while dragging (§5.3)
     * - record: playback override of cam (30 fps ring buffer)
     * - idle: settle-to-neutral + micro-motion target
     * @type {Record<string, Float32Array>}
     */
    this.weightsSource = {
      cam: new Float32Array(this.n),
      camRaw: new Float32Array(this.n),
      emotion: new Float32Array(this.n),
      eyes: new Float32Array(this.n),
      manual: new Float32Array(this.n),
      drag: new Float32Array(this.n),
      record: new Float32Array(this.n),
      idle: new Float32Array(this.n),
      mic: new Float32Array(this.n),
    };
    /** Per-channel pins: pinned channels output `manual` (🔒 in UI). */
    this.pins = new Uint8Array(this.n);
    this.asym = 0.25;      // "Asymmetry" slider default
    this.camWeight = 1;    // "Camera strength" slider default
    this.emotionWeight = 1;// "Intensity" slider
    this.globalGain = 1;   // "Gain global" (§10)
    /** Per-channel gain multipliers (§10 "gain drag"), 1 = table default. */
    this.gainOverride = new Float32Array(this.n).fill(1);
    this.hasTeeth = true;  // procedural head fakes it; asset may not
    this.blinkLRSwap = false;
    this.sideSwapMap = makeSideSwapMap(false);
    // Calibration (§4) — defaults are neutral-ish until calibrated.
    this.baseline = new Float32Array(this.n);
    this.noiseFloor = new Float32Array(this.n);
    this.rangeHi = new Float32Array(this.n).fill(1);
    for (let i = 0; i < this.n; i++) this.noiseFloor[i] = 0.02;
    this.idleBlend = 0;    // 0 = live, 1 = full idle settle (§2 goIdle)
    this.playbackActive = false;
    this._settle = new Float32Array(this.n);
  }

  /**
   * Install calibration (§4.3): wink map remaps bilateral L/R *naming* once,
   * never by flipping sliders later (trap §11.3).
   * @param {{b0: Float32Array|number[], nf: Float32Array|number[], hi: Float32Array|number[], blinkLRSwap: boolean, z0: number|null}} cal
   */
  applyCalibration(cal) {
    this.baseline.set(cal.b0);
    this.noiseFloor.set(cal.nf);
    this.rangeHi.set(cal.hi);
    if (cal.blinkLRSwap !== undefined) {
      this.blinkLRSwap = cal.blinkLRSwap;
      this.sideSwapMap = makeSideSwapMap(cal.blinkLRSwap);
    }
  }

  /**
   * Feed one camera detection (new frames only) into the cam producer.
   * Runs the §5.1 mapping: normalize → symmetrize → gain.
   * eyeBlink channels are deliberately NOT written here — camera blinks
   * inject into the blink scheduler instead (trap §11.4).
   *
   * @param {Float32Array|number[]} raw 52 raw MediaPipe scores (ARKit order)
   */
  feedCamera(raw) {
    const cam = this.weightsSource.cam;
    const camRaw = this.weightsSource.camRaw;
    for (let i = 0; i < this.n; i++) {
      camRaw[i] = raw[i] ?? 0;
      const name = CHANNELS[i];
      if (name === 'eyeBlinkLeft' || name === 'eyeBlinkRight') {
        cam[i] = 0;
        continue;
      }
      // Trap §11.3: channel identity may be swapped wholesale at load time.
      const srcIdx = this.sideSwapMap[name] ? CH[this.sideSwapMap[name]] : i;
      const v = raw[srcIdx] ?? 0;
      const n = normalizeChannel(v, this.noiseFloor[i], this.rangeHi[i]);
      cam[i] = n * gainOf(name) * this.gainOverride[i];
      if (cam[i] > 1) cam[i] = 1;
    }
    symmetrize(cam, this.asym);
  }

  /**
   * Feed the mic producer (§5.6): jaw from energy + A/E/I/O/U viseme bias.
   * Mic is a camera-family source — it max-composites onto mouth/jaw only.
   * @param {number} jawOpen 0..1 (LP 12 Hz of audio energy·2.2)
   * @param {{a: number, e: number, i: number, o: number, u: number}} viseme
   */
  feedMic(jawOpen, viseme) {
    const m = this.weightsSource.mic;
    m.fill(0);
    const cl = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
    m[CH.jawOpen] = cl(jawOpen * 0.75 + (viseme?.a ?? 0) * 0.35);
    const e = (viseme?.e ?? 0) * 0.55;
    const i = (viseme?.i ?? 0) * 0.25;
    const o = (viseme?.o ?? 0) * 0.55;
    const u = (viseme?.u ?? 0) * 0.55;
    m[CH.mouthStretchLeft] = e;
    m[CH.mouthStretchRight] = e;
    m[CH.mouthSmileLeft] = i;
    m[CH.mouthSmileRight] = i;
    m[CH.mouthFunnel] = o;
    m[CH.mouthPucker] = u;
  }

  /**
   * Feed EyeSolver outputs (lid curves, pupil-adjacent look morphs).
   * @param {{lidL: number, lidR: number, lookYawL?: number, lookPitchL?: number}} lids
   */
  feedEyes(lids) {
    const e = this.weightsSource.eyes;
    e.fill(0);
    // Trap §11.3: blink L/R naming resolved by calibration, not assumption.
    const lIsLeft = !this.blinkLRSwap;
    e[CH.eyeBlinkLeft] = lIsLeft ? lids.lidL : lids.lidR;
    e[CH.eyeBlinkRight] = lIsLeft ? lids.lidR : lids.lidL;
  }

  /**
   * Record-playback override of the cam producer (§1 recorder).
   * @param {Float32Array|null} frame52 null when playback stops
   */
  feedPlayback(frame52) {
    this.playbackActive = frame52 !== null;
    if (frame52) this.weightsSource.record.set(frame52);
    else this.weightsSource.record.fill(0);
  }

  /** Enter idle (no face for >0.8 s — §2). */
  goIdle() {
    this.idleBlend = 1;
    for (let i = 0; i < this.n; i++) this._settle[i] = this.weights[i];
  }

  /** Back to live tracking. */
  wake() {
    this.idleBlend = 0;
    this.weightsSource.cam.fill(0);
    this.weightsSource.camRaw.fill(0);
    this.weightsSource.record.fill(0);
    this.playbackActive = false;
  }

  /**
   * Merge all producer layers into `weights` (§5.3). Call once per substep.
   * @param {number} dtSec substep seconds
   */
  update(dtSec) {
    const s = this.weightsSource;
    // Idle: exponential settle toward neutral (never a snap, §2).
    if (this.idleBlend > 0) {
      const k = Math.exp(-2.2 * dtSec);
      this.idleBlend *= k;
      if (this.idleBlend < 0.003) this.idleBlend = 0;
      for (let i = 0; i < this.n; i++) {
        s.idle[i] = this._settle[i] * this.idleBlend + s.emotion[i] * (1 - this.idleBlend) * 0;
      }
    }

    for (let i = 0; i < this.n; i++) {
      const name = CHANNELS[i];
      const isBlink = name === 'eyeBlinkLeft' || name === 'eyeBlinkRight';
      if (this.pins[i]) {
        this.weights[i] = s.manual[i];
        continue;
      }
      const camLayer = this.playbackActive ? s.record[i] : s.cam[i];
      let w = camLayer * this.camWeight;

      // Emotion is max-composited (§5.3): preset pushes channels up but the
      // camera can exceed it. eyeMouthPreserve (§5.7) says how much camera
      // mouth signal leaks through the preset (joy .8 … fear .35).
      const emo = s.emotion[i] * this.emotionWeight;
      if (emo > 0) {
        const isMouth = GROUP_OF[name] === 'mouth' || GROUP_OF[name] === 'jaw';
        const preserve = this._mouthPreserve ?? 0.15;
        const camEff = camLayer * (isMouth
          ? preserve + (1 - preserve) * this.camWeight
          : this.camWeight);
        w = Math.max(camEff, emo);
      }

      // Eye layer (scheduler lids / looks) always wins over raw cam for
      // blink channels; dragAdd wins while dragging.
      if (isBlink) w = Math.max(s.eyes[i], emo * 0.25);

      // Mic jaw/viseme (§5.6, default off) max-composites onto mouth/jaw.
      if (s.mic[i] > w && (GROUP_OF[name] === 'jaw' || GROUP_OF[name] === 'mouth')) {
        w = s.mic[i];
      }

      if (this.idleBlend > 0) w = w * (1 - this.idleBlend) + s.idle[i] * this.idleBlend;

      w = (w + s.drag[i]) * this.globalGain;
      this.weights[i] = w < 0 ? 0 : w > 1 ? 1 : w;
    }

    applyMouthExtras(this.weights, this.hasTeeth);
  }

  /**
   * Set the emotion mouth-preserve scalar (from EmotionController).
   * @param {number} v 0..1
   */
  setMouthPreserve(v) {
    this._mouthPreserve = v;
  }

  /**
   * Set a manual slider value.
   * @param {string} name channel name
   * @param {number} v 0..1
   */
  setManual(name, v) {
    const i = CH[name];
    if (i !== undefined) this.weightsSource.manual[i] = v;
  }

  /**
   * Pin/unpin a channel (🔒 = output manual value, §5.3).
   * @param {string} name
   * @param {boolean} on
   */
  setPin(name, on) {
    const i = CH[name];
    if (i !== undefined) this.pins[i] = on ? 1 : 0;
  }

  /**
   * Drag add: accumulated solver increments, cleared on release (§6).
   * @param {string} name
   * @param {number} delta
   */
  addDrag(name, delta) {
    const i = CH[name];
    if (i !== undefined) this.addDragIndex(i, delta);
  }

  /**
   * Drag add by channel index (hot path from drag.js — no string lookup).
   * @param {number} i channel index
   * @param {number} delta
   */
  addDragIndex(i, delta) {
    const v = this.weightsSource.drag[i] + delta;
    this.weightsSource.drag[i] = v < -1 ? -1 : v > 1 ? 1 : v;
  }

  /** Snapshot for the undo stack (§6: 20 weight snapshots). */
  snapshot() {
    const snap = new Float32Array(this.n + this.n);
    snap.set(this.weights, 0);
    snap.set(this.weightsSource.drag, this.n);
    return snap;
  }

  /**
   * Restore a snapshot (U = undo).
   * @param {Float32Array} snap
   */
  restore(snap) {
    for (let i = 0; i < this.n; i++) {
      this.weights[i] = snap[i];
      this.weightsSource.drag[i] = snap[this.n + i];
      this.weightsSource.manual[i] = snap[i];
      if (this.weightsSource.drag[i] !== 0) this.pins[i] = 1; // keep edited channels pinned
    }
  }

  /** Reset all edits (key `0`). */
  resetAll() {
    this.weightsSource.drag.fill(0);
    this.weightsSource.manual.fill(0);
    this.pins.fill(0);
    this.gainOverride.fill(1);
    this.asym = 0.25;
    this.camWeight = 1;
    this.emotionWeight = 1;
    this.globalGain = 1;
  }
}
