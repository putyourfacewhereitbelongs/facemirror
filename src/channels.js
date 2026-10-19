/**
 * channels.js — dead-channel revival from 478 face-mesh landmarks (spec §5.2).
 * Pure math over Float32Arrays: no DOM, no THREE (§9.1).
 *
 * Known MediaPipe facts (§3 preamble): 7 channels are dead/flat
 * (cheekPuff, tongueOut, jawForward, jawLeft, jawRight, mouthDimpleL/R) and
 * several are weak. No slider can fix a zero-signal channel (trap §11.6) —
 * they must be re-derived from landmark geometry, AFTER pose removal
 * (trap §11.5: never measure jaw/lean in image space).
 */

/** @type {Record<string, number>} */
export const CONFIG = {
  interEyeCm: 6.3,        // average inter-pupillary distance in cm (metric prior)
  cheekPuffGain: 3.0,     // §5.2 "×3"
  cheekPuffMouthGate: 0.2,
  jawShiftCm: 0.8,        // ±0.8 cm chin x-offset → 0..1
  jawForwardCm: 0.5,      // chin z beyond neutral → 0..1
  jawOpenGate: 0.25,      // jawForward disabled while mouth open wider
  dimpleSmileGate: 0.3,
  frownBrowGate: 0.2,
  sneerGateBrow: 0.2,
  sneerGateUpperUp: 0.3,
  tongueCm: 0.6,
  tongueJawGate: 0.4,
  eyeWideGain: 2.2,       // EAR delta amplification
  earProfileYawRad: 0.61, // |yaw| > 35° → distrust model blink channels (§5.2)
};

/**
 * MediaPipe 478-mesh landmark indices used by the geometry solver.
 * Eyes are labelled A/B geometrically; Left/Right naming is resolved at
 * runtime by x-position in canonical space + wink calibration (trap §11.3).
 * @type {Record<string, number>}
 */
export const LM = {
  noseTip: 1,
  noseBridgeLow: 168,
  noseBridgeHigh: 6,
  noseWingA: 98,
  noseWingB: 327,
  noseCreaseA: 129,
  noseCreaseB: 358,
  eyeAOuter: 33,
  eyeAInner: 133,
  eyeATop: 159,
  eyeABot: 145,
  eyeBInner: 362,
  eyeBOuter: 263,
  eyeBTop: 386,
  eyeBBot: 374,
  irisA: 468,
  irisB: 473,
  mouthCornerA: 61,
  mouthCornerB: 291,
  upperLip: 0,
  innerUpperLip: 13,
  innerLowerLip: 14,
  lowerLip: 17,
  lowerLipArcA: 181,
  lowerLipArcB: 405,
  cheekA: 50,
  cheekB: 280,
  chinBottom: 152,
  chinMid: 199,
  browAInner: 107,
  browAOuter: 70,
  browBInner: 336,
  browBOuter: 300,
};

/** 478×3 preallocated workspace factory (trap §11.12: zero per-frame alloc). */
export function makeCanonicalBuffer() {
  return new Float32Array(478 * 3);
}

/**
 * Squared distance between two landmark points in canonical buffer.
 * @param {Float32Array} p canonical buffer
 * @param {number} i landmark index
 * @param {number} j landmark index
 * @returns {number}
 */
function dist2(p, i, j) {
  const dx = p[i * 3] - p[j * 3];
  const dy = p[i * 3 + 1] - p[j * 3 + 1];
  const dz = p[i * 3 + 2] - p[j * 3 + 2];
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Euclidean distance between two landmarks in canonical buffer.
 * @param {Float32Array} p
 * @param {number} i
 * @param {number} j
 * @returns {number}
 */
export function dist(p, i, j) {
  return Math.sqrt(dist2(p, i, j));
}

/**
 * Invert a 4×4 column-major matrix (MediaPipe's facialTransformationMatrixes
 * is column-major — data[12..14] is translation).
 * @param {Float32Array|number[]} m input 16 floats (column-major)
 * @param {Float32Array} out 16-float preallocated output
 * @returns {boolean} false if singular
 */
export function invertMatrix4(m, out) {
  // Column-major layout: m[col*4+row].
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return false;
  det = 1 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return true;
}

/**
 * Transform raw normalized landmarks into canonical metric space:
 * `p_can = M⁻¹ · [x·W/ppcm, y·H/ppcm, z·faceScale]` (spec §5.2).
 * Units are centimetres so thresholds like "±0.8 cm" mean what they say.
 * Run this BEFORE any jaw/lean measurement (trap §11.5).
 *
 * @param {Float32Array} out preallocated 478×3 canonical output
 * @param {{x:number,y:number,z:number}[]} landmarks 478 normalized landmarks
 * @param {Float32Array|number[]|null} matrixData 16-float column-major
 *   facialTransformationMatrixes[0].data (null → identity, tests only)
 * @param {number} imgW capture width (px)
 * @param {number} imgH capture height (px)
 * @param {number} [eyeWidthPx] eye-corner pixel distance used for metric scale
 * @returns {number} pixels-per-cm scale used (0 if matrix singular)
 */
export function transformToCanonical(out, landmarks, matrixData, imgW, imgH, eyeWidthPx) {
  const ppcm = eyeWidthPx > 1 ? eyeWidthPx / CONFIG.interEyeCm : imgW / 32;
  const inv = new Float32Array(16);
  if (matrixData) {
    if (!invertMatrix4(matrixData, inv)) return 0;
  } else {
    inv.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }
  for (let i = 0; i < 478; i++) {
    const lx = landmarks[i].x * imgW / ppcm;
    const ly = landmarks[i].y * imgH / ppcm;
    const lz = landmarks[i].z * imgW / ppcm;
    out[i * 3] = inv[0] * lx + inv[4] * ly + inv[8] * lz + inv[12];
    out[i * 3 + 1] = inv[1] * lx + inv[5] * ly + inv[9] * lz + inv[13];
    out[i * 3 + 2] = inv[2] * lx + inv[6] * ly + inv[10] * lz + inv[14];
  }
  return ppcm;
}

/**
 * Eye Aspect Ratio (EAR): vertical lid gap / eye width.
 * @param {Float32Array} p canonical buffer
 * @param {boolean} sideA true for landmark-set A (33/133 eye)
 * @returns {number}
 */
export function eyeAspectRatio(p, sideA) {
  const outer = sideA ? LM.eyeAOuter : LM.eyeBOuter;
  const inner = sideA ? LM.eyeAInner : LM.eyeBInner;
  const top = sideA ? LM.eyeATop : LM.eyeBTop;
  const bot = sideA ? LM.eyeABot : LM.eyeBBot;
  const w = dist(p, outer, inner);
  if (w < 1e-6) return 0;
  return dist(p, top, bot) / w;
}

/**
 * Per-user neutral geometry profile, captured during calibration (§4).
 * @typedef {Object} NeutralProfile
 * @property {number} cheekDist  corner→pre-ear distance (mean of both sides)
 * @property {number} chinX      chin-mid x offset from nose-bridge line
 * @property {number} chinZ      chin z
 * @property {number} dimpleA    cornerA↔nose-creaseA distance
 * @property {number} dimpleB    cornerB↔nose-creaseB distance
 * @property {number} sneerA     upper-lip↔nose-wingA vertical gap
 * @property {number} sneerB     upper-lip↔nose-wingB vertical gap
 * @property {number} earA       eye aspect ratio A
 * @property {number} earB       eye aspect ratio B
 */

/**
 * Measure the neutral geometry profile from a canonical buffer.
 * @param {Float32Array} p canonical landmarks (neutral pose)
 * @returns {NeutralProfile}
 */
export function measureNeutral(p) {
  return {
    cheekDist: (dist(p, LM.mouthCornerA, LM.cheekA) + dist(p, LM.mouthCornerB, LM.cheekB)) / 2,
    chinX: p[LM.chinBottom * 3] - (p[LM.noseBridgeLow * 3] + p[LM.noseBridgeHigh * 3]) / 2,
    chinZ: p[LM.chinBottom * 3 + 2],
    dimpleA: dist(p, LM.mouthCornerA, LM.noseCreaseA),
    dimpleB: dist(p, LM.mouthCornerB, LM.noseCreaseB),
    sneerA: Math.abs(p[LM.upperLip * 3 + 1] - p[LM.noseWingA * 3 + 1]),
    sneerB: Math.abs(p[LM.upperLip * 3 + 1] - p[LM.noseWingB * 3 + 1]),
    earA: eyeAspectRatio(p, true),
    earB: eyeAspectRatio(p, false),
  };
}

/**
 * Merge two neutral profiles with an EMA (used to refine neutral after
 * calibration while the user sits still).
 * @param {NeutralProfile} prev
 * @param {NeutralProfile} next
 * @param {number} alpha
 * @returns {NeutralProfile}
 */
export function blendNeutral(prev, next, alpha) {
  /** @type {Record<string, number>} */
  const out = {};
  for (const k of Object.keys(prev)) {
    const a = /** @type {Record<string, number>} */ (prev)[k];
    const b = /** @type {Record<string, number>} */ (next)[k];
    out[k] = a + (b - a) * alpha;
  }
  return /** @type {NeutralProfile} */ (out);
}

/** @param {number} x @param {number} lo @param {number} hi @returns {number} */
export function clamp01(x, lo = 0, hi = 1) {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Dead-channel solver. Fills the geometry-derived channels of a 52-weight
 * array every frame from canonical landmarks + a few model-channel cues.
 * Channel indices are resolved through `channels` (the ARKit name list).
 */
export class DeadChannelSolver {
  /**
   * @param {string[]} channels ordered ARKit names (index = channel id)
   * @param {{sideSwap?: boolean}} [opts] sideSwap=true when calibration proved
   *   MediaPipe L/R naming inverted (trap §11.3)
   */
  constructor(channels, opts = {}) {
    this.idx = Object.fromEntries(channels.map((n, i) => [n, i]));
    this.sideSwap = !!opts.sideSwap;
    /** @type {NeutralProfile|null} */
    this.neutral = null;
    // Scratch scalars — reused, never allocated per frame (§11.12).
    this._earA = 0;
    this._earB = 0;
  }

  /**
   * Install the neutral profile (from calibrate.js or lazily).
   * @param {NeutralProfile} profile
   */
  setNeutral(profile) {
    this.neutral = profile;
  }

  /**
   * Solve geometry channels into `out` (Float32Array(52)).
   * Only writes the dead/weak channels; other channels are untouched.
   *
   * @param {Float32Array} out 52 weights, modified in place
   * @param {Float32Array} canon canonical landmark buffer
   * @param {Object} cues
   * @param {number} cues.mouthClose model mouthClose weight (gate)
   * @param {number} cues.jawOpen model jawOpen weight (gate)
   * @param {number} cues.browDown mean browDown weight (gate)
   * @param {number} cues.mouthUpperUp mean mouthUpperUp weight (gate)
   * @param {number} cues.smile mean mouthSmile weight (gate)
   * @param {number} cues.headYaw head yaw radians (profile-turn blink check)
   * @param {number} [scale] optional global scale for test rigs (default 1)
   */
  solve(out, canon, cues, scale = 1) {
    const n = this.neutral;
    if (!n) return; // nothing to compare against yet
    const ix = this.idx;

    // --- cheekPuff: mouth corners push out toward the pre-ear contour. ---
    if (cues.mouthClose > CONFIG.cheekPuffMouthGate) {
      const cheekDist = (dist(canon, LM.mouthCornerA, LM.cheekA) + dist(canon, LM.mouthCornerB, LM.cheekB)) / 2;
      const puff = clamp01((cheekDist - n.cheekDist) * CONFIG.cheekPuffGain * scale);
      if (puff > out[ix.cheekPuff]) out[ix.cheekPuff] = puff;
    }

    // --- jawLeft / jawRight: chin x-offset in canonical space (trap §11.5). ---
    const chinX = canon[LM.chinBottom * 3] - (canon[LM.noseBridgeLow * 3] + canon[LM.noseBridgeHigh * 3]) / 2;
    const shift = (chinX - n.chinX) / CONFIG.jawShiftCm; // +1 at +0.8 cm
    if (shift > 0) {
      const w = clamp01(shift * scale);
      const key = this.sideSwap ? 'jawLeft' : 'jawRight';
      if (w > out[ix[key]]) out[ix[key]] = w;
    } else if (shift < 0) {
      const w = clamp01(-shift * scale);
      const key = this.sideSwap ? 'jawRight' : 'jawLeft';
      if (w > out[ix[key]]) out[ix[key]] = w;
    }

    // --- jawForward: chin z beyond neutral, gated on mouth not being open. ---
    if (cues.jawOpen < CONFIG.jawOpenGate) {
      const dz = canon[LM.chinBottom * 3 + 2] - n.chinZ;
      const w = clamp01((dz / CONFIG.jawForwardCm) * scale);
      if (w > out[ix.jawForward]) out[ix.jawForward] = w;
    }

    // --- mouthDimpleL/R: corner drawn toward nose crease while smiling. ---
    if (cues.smile > CONFIG.dimpleSmileGate) {
      const dA = dist(canon, LM.mouthCornerA, LM.noseCreaseA);
      const dB = dist(canon, LM.mouthCornerB, LM.noseCreaseB);
      const wA = clamp01(((n.dimpleA - dA) / CONFIG.jawShiftCm) * scale);
      const wB = clamp01(((n.dimpleB - dB) / CONFIG.jawShiftCm) * scale);
      const keyA = this.sideSwap ? 'mouthDimpleRight' : 'mouthDimpleLeft';
      const keyB = this.sideSwap ? 'mouthDimpleLeft' : 'mouthDimpleRight';
      if (wA > out[ix[keyA]]) out[ix[keyA]] = wA;
      if (wB > out[ix[keyB]]) out[ix[keyB]] = wB;
    }

    // --- mouthFrownL/R: corner drops below the lower-lip arc. ---
    if (cues.browDown > CONFIG.frownBrowGate) {
      const fA = cornerBelowLipArc(canon, true);
      const fB = cornerBelowLipArc(canon, false);
      const wA = clamp01((fA / CONFIG.tongueCm) * scale);
      const wB = clamp01((fB / CONFIG.tongueCm) * scale);
      const keyA = this.sideSwap ? 'mouthFrownRight' : 'mouthFrownLeft';
      const keyB = this.sideSwap ? 'mouthFrownLeft' : 'mouthFrownRight';
      if (wA > out[ix[keyA]]) out[ix[keyA]] = wA;
      if (wB > out[ix[keyB]]) out[ix[keyB]] = wB;
    }

    // --- noseSneerL/R: nose-wing↔upper-lip gap shrinks. ---
    if (cues.browDown > CONFIG.sneerGateBrow || cues.mouthUpperUp > CONFIG.sneerGateUpperUp) {
      const sA = Math.abs(canon[LM.upperLip * 3 + 1] - canon[LM.noseWingA * 3 + 1]);
      const sB = Math.abs(canon[LM.upperLip * 3 + 1] - canon[LM.noseWingB * 3 + 1]);
      const wA = clamp01(((n.sneerA - sA) / CONFIG.tongueCm) * scale);
      const wB = clamp01(((n.sneerB - sB) / CONFIG.tongueCm) * scale);
      const keyA = this.sideSwap ? 'noseSneerRight' : 'noseSneerLeft';
      const keyB = this.sideSwap ? 'noseSneerLeft' : 'noseSneerRight';
      if (wA > out[ix[keyA]]) out[ix[keyA]] = wA;
      if (wB > out[ix[keyB]]) out[ix[keyB]] = wB;
    }

    // --- tongueOut: inner-lip mid dips below the outer lower-lip line. ---
    if (cues.jawOpen > CONFIG.tongueJawGate) {
      const below = tongueBelowLipLine(canon);
      const w = clamp01((below / CONFIG.tongueCm) * scale);
      if (w > out[ix.tongueOut]) out[ix.tongueOut] = w;
    }

    // --- eyeWide: override with EAR vs neutral (also blink sanity at yaw). ---
    const earA = eyeAspectRatio(canon, true);
    const earB = eyeAspectRatio(canon, false);
    this._earA = earA;
    this._earB = earB;
    const wideA = clamp01((earA - n.earA) * CONFIG.eyeWideGain * scale);
    const wideB = clamp01((earB - n.earB) * CONFIG.eyeWideGain * scale);
    const keyA = this.sideSwap ? 'eyeWideRight' : 'eyeWideLeft';
    const keyB = this.sideSwap ? 'eyeWideLeft' : 'eyeWideRight';
    if (wideA > out[ix[keyA]]) out[ix[keyA]] = wideA;
    if (wideB > out[ix[keyB]]) out[ix[keyB]] = wideB;
  }

  /**
   * Validate model blink channels with EAR when the head turns in profile
   * (§5.2: model blink glitches past |yaw| ≈ 35°).
   * @param {Float32Array} out weights in/out
   * @param {number} headYaw radians
   * @returns {boolean} true when profile fallback is active
   */
  validateBlink(out, headYaw) {
    if (Math.abs(headYaw) < CONFIG.earProfileYawRad) return false;
    const ix = this.idx;
    const n = this.neutral;
    if (!n) return false;
    // Near eye is trusted, far eye is attenuated toward it (stabilizeBlink
    // mirrors this policy in eyes.js).
    const closedA = clamp01(1 - this._earA / Math.max(n.earA, 1e-3));
    const closedB = clamp01(1 - this._earB / Math.max(n.earB, 1e-3));
    const keyA = this.sideSwap ? 'eyeBlinkRight' : 'eyeBlinkLeft';
    const keyB = this.sideSwap ? 'eyeBlinkLeft' : 'eyeBlinkRight';
    out[ix[keyA]] = Math.max(out[ix[keyA]], closedA);
    out[ix[keyB]] = Math.max(out[ix[keyB]], closedB);
    return true;
  }
}

/**
 * How far (cm) the mouth corner sits below the lower-lip arc interpolation.
 * Positive → frown geometry.
 * @param {Float32Array} p canonical buffer
 * @param {boolean} sideA true for corner A
 * @returns {number}
 */
export function cornerBelowLipArc(p, sideA) {
  const corner = sideA ? LM.mouthCornerA : LM.mouthCornerB;
  const arcMid = LM.lowerLip;
  const arcEnd = sideA ? LM.lowerLipArcA : LM.lowerLipArcB;
  // Interpolate lip-arc height at the corner's x (2-point arc: end & mid).
  const xC = p[corner * 3];
  const xE = p[arcEnd * 3];
  const xM = p[arcMid * 3];
  const span = xM - xE;
  const t = Math.abs(span) < 1e-6 ? 0 : clamp01((xC - xE) / span, -0.5, 1.5);
  const yArc = p[arcEnd * 3 + 1] + t * (p[arcMid * 3 + 1] - p[arcEnd * 3 + 1]);
  return yArc - p[corner * 3 + 1]; // y-down image convention: below arc → positive after pose-normal
}

/**
 * How far (cm) the inner-lip mid point sits below the outer lower-lip line
 * between the corners — the tongue proxy (the mesh has no tongue vertices).
 * @param {Float32Array} p canonical buffer
 * @returns {number}
 */
export function tongueBelowLipLine(p) {
  const xA = p[LM.mouthCornerA * 3];
  const xB = p[LM.mouthCornerB * 3];
  const yA = p[LM.mouthCornerA * 3 + 1];
  const yB = p[LM.mouthCornerB * 3 + 1];
  const xC = p[LM.innerLowerLip * 3];
  const span = xB - xA;
  const t = Math.abs(span) < 1e-6 ? 0.5 : clamp01((xC - xA) / span);
  const yLine = yA + t * (yB - yA);
  // Canonical +y is up: "below the line" means smaller y → measure yLine − y.
  return yLine - p[LM.innerLowerLip * 3 + 1];
}
