/**
 * eyes.js — the alive-vs-doll module (spec §5.5). Pure math: no DOM/THREE.
 *
 * Blink Poisson scheduler, saccades with minimum-jerk profiles, microsaccade
 * pink-noise drift, Listing's-law eye quaternions, VOR clamp-and-reset, and
 * the KalidoKit stabilizeBlink math (ported; its face solver is pre-
 * blendshape-era and would regress us — no KalidoKit dependency).
 *
 * Trap §11.4: camera blinks never drive lids directly — they inject into the
 * scheduler so the lid curve always has real attack/release.
 * Trap §11.7: eyes are quaternions under Listing's law, never yaw-then-pitch
 * Euler; VOR needs clamp-and-reset or eyes stick at max.
 */

/** @type {Record<string, number>} */
export const CONFIG = {
  blinkRateDefault: 0.3,      // Poisson λ (blinks/s); slider 0.1–0.6
  blinkMinGapSec: 1.2,        // never blink-start twice inside this
  blinkDurMinSec: 0.10,
  blinkDurMaxSec: 0.40,
  blinkCloseFrac: 0.38,       // closing is 38% of the duration, opening 62%
  blinkDoubleProb: 0.08,
  blinkDoubleGapSec: 0.32,    // gap before the second closure of a double
  jawSuppress: 0.65,          // spontaneous blinks suppressed above this jawOpen
  stareForceSec: 3.2,         // forced blink after staring this long
  saccadeBlinkProb: 0.35,     // chance a big saccade drags a blink along
  saccadeBlinkWindowSec: 0.12,
  saccadeBigDeg: 15,
  fixationMuSec: 0.65,        // log-normal fixation μ (in log-seconds)
  fixationSigmaSec: 0.45,
  saccadeGainMsPerDeg: 3.5,   // duration = 20 + 3.5·deg, clamp 20–90 ms
  saccadeMinMs: 20,
  saccadeMaxMs: 90,
  saccadeOvershoot: 0.17,     // 17% overshoot then a 25 ms micro-correction
  microCorrectMs: 25,
  microSaccadeRate: 1.3,      // events/s during fixation
  microSaccadeRayleigh: 0.2,  // Rayleigh amplitude σ in degrees
  driftDeg: 0.15,             // pink-noise fixation drift amplitude
  maxEyeDeg: 45,              // total eye-in-head range (±)
  vorResetFrac: 0.78,         // clamp&reset beyond 78% of range (§5.5)
  vorResetSec: 0.4,
  convergenceDeg: 6.5,        // eye-in per meter at fixation distance
  convergeDistCm: 90,
  pupilHippusHz: 0.5,
  pupilHippusAmp: 0.06,
  pupilArousalAmp: 0.10,
  lowerLidPer10Deg: 0.015,    // lower lid follows squint/up-gaze 1–2%/10°
  upperLagFrames: 3,          // upper lid lags downward gaze by 3 frames
  listingPrimaryX: 0, listingPrimaryY: 0, listingPrimaryZ: 1,
  stabilizeYawRad: 0.5,       // stabilizeBlink kicks in past ~29° yaw
  stabilizeBlend: 0.85,
  attackCurveExp: 1.35,       // lid close acceleration shape
  releaseCurveExp: 1.6,       // lid open deceleration shape
};

/** Minimal deterministic RNG (mulberry32) for reproducible sims/tests. */
export class Rng {
  /** @param {number} seed */
  constructor(seed = 1) {
    this.s = seed >>> 0;
  }
  /** @returns {number} uniform [0,1) */
  next() {
    this.s = (this.s + 0x6D2B79F5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** @param {number} mu @param {number} sigma @returns {number} log-normal sample */
  logNormal(mu, sigma) {
    const u = Math.max(this.next(), 1e-9);
    const v = this.next();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return Math.exp(mu + sigma * z);
  }
  /** @param {number} sigma Rayleigh scale @returns {number} */
  rayleigh(sigma) {
    return sigma * Math.sqrt(-2 * Math.log(Math.max(this.next(), 1e-9)));
  }
}

/**
 * Voss-McCartney 1/f pink noise (8 octaves of random walks + white).
 * @returns {function(): number} generator returning ~[-0.5, 0.5]
 */
export function makePinkNoise(rng = new Rng(7)) {
  const rows = new Float32Array(8);
  let counter = 0;
  let prev = 0;
  return function pink() {
    counter++;
    let sum = 0;
    for (let i = 0; i < 8; i++) {
      if ((counter & (1 << i)) === 0) rows[i] = rng.next() - 0.5;
      sum += rows[i];
    }
    const out = (sum / 4 + (rng.next() - 0.5)) * 0.5;
    prev = out;
    return prev;
  };
}

/**
 * Minimum-jerk profile (Flash & Hogan): smooth 0→1 over u∈[0,1].
 * @param {number} u progress
 * @returns {number}
 */
export function minJerk(u) {
  const t = u < 0 ? 0 : u > 1 ? 1 : u;
  return t * t * t * (10 + t * (-15 + 6 * t));
}

/**
 * Blink lid curve for one blink event (trap §11.4: generated curve, never
 * smoothed camera signal). Attack is faster than release (§5.5).
 * @param {number} progress 0..1 through the blink event
 * @returns {number} lid closure 0 (open) .. 1 (shut)
 */
export function blinkWaveform(progress) {
  const p = progress < 0 ? 0 : progress > 1 ? 1 : progress;
  const closeFrac = CONFIG.blinkCloseFrac;
  if (p < closeFrac) {
    const u = p / closeFrac;
    return Math.pow(u, CONFIG.attackCurveExp);
  }
  const u = (p - closeFrac) / (1 - closeFrac);
  return Math.pow(1 - u, CONFIG.releaseCurveExp);
}

/**
 * Poisson blink scheduler (spec §5.5).
 * Emits per-blink events; EyeSolver turns events into lid curves.
 */
export class BlinkScheduler {
  /**
   * @param {number} [lambda] blinks per second (0.1–0.6)
   * @param {Rng} [rng] seeded RNG for tests
   */
  constructor(lambda = CONFIG.blinkRateDefault, rng = new Rng(0xB11E)) {
    this.lambda = lambda;
    this.rng = rng;
    this.pink = makePinkNoise(rng);
    this.time = 0;
    this.nextAt = this._drawGap();
    this.sinceBlink = 99; // pretend we just blinked at t=0
    /** @type {Array<{t: number, dur: number, double: boolean, injected: boolean, side: number}>} */
    this.log = [];
    /** @type {Array<{t: number, dur: number, side: number}>} */
    this.active = [];
    this._pendingDouble = -1;
  }

  /** @returns {number} */
  _drawGap() {
    const gap = -Math.log(Math.max(this.rng.next(), 1e-9)) / Math.max(this.lambda, 1e-3);
    return Math.max(gap, CONFIG.blinkMinGapSec);
  }

  /**
   * Inject a blink now (camera blink events, saccade-linked blinks, or the
   * UI "blink" test). Mirroring the user still works through here (§5.5).
   * @param {number} [durSec] duration override
   * @param {number} [side] 0=both, 1=left only, 2=right only (winks)
   * @param {boolean} [isDoublePart] internal flag for double-blink pairs
   */
  injectBlink(durSec = 0, side = 0, isDoublePart = false) {
    const dur = durSec > 0
      ? Math.min(Math.max(durSec, CONFIG.blinkDurMinSec), CONFIG.blinkDurMaxSec)
      : CONFIG.blinkDurMinSec + this.rng.next() * (CONFIG.blinkDurMaxSec - CONFIG.blinkDurMinSec);
    this.active.push({ t: this.time, dur, side });
    this.log.push({ t: this.time, dur, double: isDoublePart, injected: true, side });
    this.sinceBlink = 0;
    return dur;
  }

  /**
   * Advance time.
   * @param {number} dtSec
   * @param {number} [jawOpen] spontaneous blinks suppressed above jawSuppress
   * @param {boolean} [forceNow] UI/forced blink request
   * @returns {Array<{t: number, dur: number, side: number}>} newly started blinks
   */
  update(dtSec, jawOpen = 0, forceNow = false) {
    this.time += dtSec;
    this.sinceBlink += dtSec;
    /** @type {Array<{t: number, dur: number, side: number}>} */
    const started = [];

    // Finish the second half of a double blink.
    if (this._pendingDouble >= 0 && this.time >= this._pendingDouble) {
      started.push({ t: this.time, dur: this.injectBlink(0, 0, true), side: 0 });
      this._pendingDouble = -1;
    }

    // Forced blink after staring (§5.5).
    if (this.sinceBlink > CONFIG.stareForceSec) {
      const dur = this.injectBlink();
      started.push({ t: this.time, dur, side: 0 });
      this.nextAt = this.time + this._drawGap();
    }

    // Poisson spontaneous blinks (suppressed while the jaw is wide open).
    if (this.time >= this.nextAt) {
      this.nextAt = this.time + this._drawGap();
      if (jawOpen <= CONFIG.jawSuppress || forceNow) {
        const dur = this.injectBlink();
        started.push({ t: this.time, dur, side: 0 });
        if (this.rng.next() < CONFIG.blinkDoubleProb) {
          this._pendingDouble = this.time + dur + CONFIG.blinkDoubleGapSec;
        }
      }
    }

    if (forceNow && started.length === 0) {
      const dur = this.injectBlink();
      started.push({ t: this.time, dur, side: 0 });
    }

    // Retire finished active blinks.
    for (let i = this.active.length - 1; i >= 0; i--) {
      if (this.time - this.active[i].t > this.active[i].dur) this.active.splice(i, 1);
    }
    return started;
  }

  /**
   * Current raw lid targets from scheduled blinks (pre-stabilize).
   * @param {number} [side] 0 both, 1 left, 2 right
   * @returns {{left: number, right: number}}
   */
  lidTargets(side = 0) {
    let l = 0;
    let r = 0;
    for (const b of this.active) {
      const u = (this.time - b.t) / b.dur;
      const w = blinkWaveform(u);
      const hitL = b.side === 0 || b.side === 1;
      const hitR = b.side === 0 || b.side === 2;
      if (hitL) l = Math.max(l, w);
      if (hitR) r = Math.max(r, w);
    }
    return { left: l, right: r };
  }
}

/**
 * Saccade / fixation generator (spec §5.5).
 * Fixation durations are log-normal; saccade duration follows the main
 * sequence (20 + 3.5·deg ms, clamped 20–90); trajectories are minimum-jerk
 * with a 17% overshoot corrected 25 ms later.
 */
export class SaccadeGenerator {
  /**
   * @param {Rng} [rng]
   */
  constructor(rng = new Rng(0x5ACC)) {
    this.rng = rng;
    this.pink = makePinkNoise(new Rng((rng.next() * 1e6) | 0));
    this.yaw = 0;
    this.pitch = 0;
    this.targetYaw = 0;
    this.targetPitch = 0;
    this.fixUntil = 0.5;
    this.saccadeT = -1;
    this.saccadeDur = 0.1;
    this.fromYaw = 0;
    this.fromPitch = 0;
    this.overYaw = 0;
    this.overPitch = 0;
    this.microT = 0;
    this.driftYaw = 0;
    this.driftPitch = 0;
    this.time = 0;
    /** @type {Array<{t: number, deg: number, durMs: number}>} */
    this.log = [];
  }

  /**
   * Pick the next target from a pool (cursor / other face / scene points).
   * @param {Array<{yaw: number, pitch: number}>} [targets] degrees
   */
  pickTarget(targets = []) {
    if (targets.length > 0 && this.rng.next() < 0.7) {
      const t = targets[(this.rng.next() * targets.length) | 0];
      this.targetYaw = t.yaw;
      this.targetPitch = t.pitch;
    } else {
      // Random points on the scene: ±35° yaw, ±20° pitch.
      this.targetYaw = (this.rng.next() - 0.5) * 70;
      this.targetPitch = (this.rng.next() - 0.5) * 40;
    }
    const dy = (this.targetYaw - this.yaw) * Math.PI / 180;
    const dp = (this.targetPitch - this.pitch) * Math.PI / 180;
    const deg = Math.sqrt(dy * dy + dp * dp) * 180 / Math.PI;
    this.saccadeDur = Math.min(Math.max(20 + CONFIG.saccadeGainMsPerDeg * deg, CONFIG.saccadeMinMs), CONFIG.saccadeMaxMs) / 1000;
    this.saccadeT = 0;
    this.fromYaw = this.yaw;
    this.fromPitch = this.pitch;
    // Overshoot target: 17% beyond, corrected after 25 ms (§5.5).
    this.overYaw = this.targetYaw + (this.targetYaw - this.yaw) * CONFIG.saccadeOvershoot;
    this.overPitch = this.targetPitch + (this.targetPitch - this.pitch) * CONFIG.saccadeOvershoot;
    this.log.push({ t: this.time, deg, durMs: this.saccadeDur * 1000 });
    return deg;
  }

  /**
   * Is this a "large" saccade (blink-linked)? Also used by EyeSolver to
   * schedule the 35% saccade-linked blinks.
   * @param {number} deg amplitude in degrees
   * @returns {boolean}
   */
  static isLarge(deg) {
    return deg >= CONFIG.saccadeBigDeg;
  }

  /**
   * Advance one substep.
   * @param {number} dtSec
   * @param {Array<{yaw: number, pitch: number}>} [targets]
   * @returns {number} amplitude in degrees if a new saccade just started, else 0
   */
  update(dtSec, targets = undefined) {
    this.time += dtSec;
    let startedDeg = 0;

    if (this.saccadeT >= 0) {
      // Mid-saccade: minimum-jerk ballistic phase to the overshoot point.
      this.saccadeT += dtSec;
      const u = this.saccadeT / this.saccadeDur;
      if (u < 1) {
        const j = minJerk(u);
        this.yaw = this.fromYaw + (this.overYaw - this.fromYaw) * j;
        this.pitch = this.fromPitch + (this.overPitch - this.fromPitch) * j;
      } else if (this.saccadeT < this.saccadeDur + CONFIG.microCorrectMs / 1000) {
        // 25 ms micro-correction back to the true target (§5.5).
        const j = minJerk((this.saccadeT - this.saccadeDur) / (CONFIG.microCorrectMs / 1000));
        this.yaw = this.overYaw + (this.targetYaw - this.overYaw) * j;
        this.pitch = this.overPitch + (this.targetPitch - this.overPitch) * j;
      } else {
        // Land exactly on target even if the correction window ended
        // mid-substep (otherwise the overshoot sticks).
        this.yaw = this.targetYaw;
        this.pitch = this.targetPitch;
        this.saccadeT = -1;
        this.fixUntil = this.time + Math.min(Math.max(this.rng.logNormal(CONFIG.fixationMuSec, CONFIG.fixationSigmaSec), 0.15), 2.5);
      }
    } else if (this.time >= this.fixUntil) {
      const deg = this.pickTarget(targets);
      startedDeg = deg;
    } else {
      // Fixation: pink-noise drift 0.15° + microsaccades at 1.3/s with a
      // deadband retrigger (Rayleigh amplitude ~0.2°) (§5.5).
      const dY = this.pink() * CONFIG.driftDeg;
      const dP = this.pink() * CONFIG.driftDeg;
      this.driftYaw += dY * dtSec * 60;
      this.driftPitch += dP * dtSec * 60;
      if (Math.abs(this.driftYaw) > 0.5 || Math.abs(this.driftPitch) > 0.5) {
        this.driftYaw *= 0.5;
        this.driftPitch *= 0.5;
      }
      this.microT += dtSec;
      const microGap = 1 / CONFIG.microSaccadeRate;
      if (this.microT >= microGap) {
        this.microT = 0;
        const amp = this.rng.rayleigh(CONFIG.microSaccadeRayleigh);
        const ang = this.rng.next() * 2 * Math.PI;
        this.yaw += amp * Math.cos(ang);
        this.pitch += amp * Math.sin(ang);
      }
    }
    return startedDeg;
  }

  /** @returns {{yaw: number, pitch: number}} current gaze incl. drift */
  gaze() {
    return {
      yaw: this.yaw + this.driftYaw,
      pitch: this.pitch + this.driftPitch,
    };
  }
}

// ---------------------------------------------------------------------------
// Quaternion helpers (pure, no THREE — keeps eyes.js testable in node).
// Quaternions are {x,y,z,w} plain objects; outputs write into Float32Array(4).
// ---------------------------------------------------------------------------

/**
 * Quaternion from axis-angle (Listing's law building block).
 * @param {number} ax @param {number} ay @param {number} az axis (unit)
 * @param {number} angleRad
 * @param {Float32Array} out 4 floats [x,y,z,w]
 */
export function quatAxisAngle(ax, ay, az, angleRad, out) {
  const s = Math.sin(angleRad / 2);
  out[0] = ax * s;
  out[1] = ay * s;
  out[2] = az * s;
  out[3] = Math.cos(angleRad / 2);
}

/**
 * Listing's law eye quaternion (§5.5): rotate from primary gaze to
 * (gazeDir) about normalize(gazeDir × primary). NEVER yaw-then-pitch Euler
 * (trap §11.7).
 * @param {number} gazeX @param {number} gazeY @param {number} gazeZ unit gaze
 * @param {Float32Array} out 4 floats
 */
export function listingQuaternion(gazeX, gazeY, gazeZ, out) {
  const px = CONFIG.listingPrimaryX;
  const py = CONFIG.listingPrimaryY;
  const pz = CONFIG.listingPrimaryZ;
  // axis = normalize(gaze × primary)
  let ax = gazeY * pz - gazeZ * py;
  let ay = gazeZ * px - gazeX * pz;
  let az = gazeX * py - gazeY * px;
  const len = Math.hypot(ax, ay, az);
  const dot = gazeX * px + gazeY * py + gazeZ * pz;
  if (len < 1e-6) {
    // Gaze parallel to primary: identity (or 180° flip — not reachable for eyes).
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
    return;
  }
  ax /= len; ay /= len; az /= len;
  const angle = Math.atan2(len, dot);
  quatAxisAngle(ax, ay, az, angle, out);
}

/**
 * KalidoKit stabilizeBlink port (§5.5 / trap §11.4 companion): when the head
 * yaws away, the far eye's blink is attenuated toward the near eye's so
 * profile views never show a one-eyed blink.
 * @param {number} blinkL left lid 0..1
 * @param {number} blinkR right lid 0..1
 * @param {number} headYawRad positive = head turned toward subject's left
 * @param {Float32Array} out2 preallocated 2 floats [left, right]
 * @param {number} [threshold] yaw threshold (rad)
 * @param {number} [blend] max blend toward the near eye (0..1)
 */
export function stabilizeBlink(blinkL, blinkR, headYawRad, out2, threshold = CONFIG.stabilizeYawRad, blend = CONFIG.stabilizeBlend) {
  const yaw = headYawRad;
  if (Math.abs(yaw) <= threshold) {
    out2[0] = blinkL;
    out2[1] = blinkR;
    return;
  }
  // k grows 0→blend as yaw moves past the threshold.
  const over = (Math.abs(yaw) - threshold) / (Math.PI / 2 - threshold);
  const k = Math.min(Math.max(over, 0), 1) * blend;
  if (yaw > 0) {
    // Head turned left: right eye is far → pull toward left (near).
    out2[0] = blinkL;
    out2[1] = blinkR + (blinkL - blinkR) * k;
  } else {
    out2[0] = blinkL + (blinkR - blinkL) * k;
    out2[1] = blinkR;
  }
}

/**
 * Convergence (§5.5): inward rotation from fixation distance.
 * @param {number} distCm fixation distance
 * @returns {number} degrees of eye-in per eye
 */
export function convergenceDeg(distCm) {
  const d = Math.max(distCm, 20);
  return Math.min(CONFIG.convergenceDeg * (CONFIG.convergeDistCm / d), CONFIG.convergenceDeg * 3);
}

/**
 * Pupil scale (§5.5): 1 + hippus + arousal.
 * @param {number} hippusUnit pink noise in ~[-0.5,0.5]
 * @param {number} arousal 0..1 from emotion intensity
 * @returns {number} scale factor
 */
export function pupilScale(hippusUnit, arousal) {
  return 1 + CONFIG.pupilHippusAmp * 2 * hippusUnit + CONFIG.pupilArousalAmp * Math.min(Math.max(arousal, 0), 1);
}

/**
 * Full eye solver: owns blink curve generation, saccades, Listing quats,
 * VOR clamp-and-reset, convergence, pupil.
 */
export class EyeSolver {
  /**
   * @param {object} [opts]
   * @param {number} [opts.blinkRate] λ blinks/s
   * @param {number} [opts.seed] RNG seed
   */
  constructor(opts = {}) {
    const rng = new Rng(opts.seed ?? 0xE7E5);
    this.scheduler = new BlinkScheduler(opts.blinkRate ?? CONFIG.blinkRateDefault, rng);
    this.saccades = new SaccadeGenerator(rng);
    this.pink = makePinkNoise(new Rng((rng.next() * 1e6) | 0));
    this.eyeWorldYaw = 0;
    this.eyeWorldPitch = 0;
    this.eyeInHeadYaw = 0;
    this.eyeInHeadPitch = 0;
    this.vorResetT = -1;
    this.vorResetFromY = 0;
    this.vorResetFromP = 0;
    this.upperLagY = 0;
    this._lagBuf = new Float32Array(CONFIG.upperLagFrames + 1);
    this._lagIdx = 0;
    this._stabilize = new Float32Array(2);
    this.outL = new Float32Array(4);
    this.outR = new Float32Array(4);
    this.headMoving = 0;
    // Feature toggles (each individually off-switchable — §10).
    this.toggles = {
      saccades: true,
      jitter: true,
      vor: true,
      pupil: true,
    };
  }

  /**
   * Per-substep update.
   * @param {number} dtSec
   * @param {Object} input
   * @param {number} input.headYaw rad
   * @param {number} input.headPitch rad
   * @param {number} input.jawOpen 0..1
   * @param {number} [input.camBlinkL] camera lid closure 0..1 (live, or inject when untracked)
   * @param {boolean} [input.cameraTracked] real webcam lids override idle blinks
   * @param {number} [input.camBlinkR]
   * @param {number|null} [input.camGazeYaw] iris-derived gaze (camera mode)
   * @param {number|null} [input.camGazePitch]
   * @param {number} [input.arousal] 0..1 emotion intensity for pupil
   * @param {number} [input.squint] 0..1 lower-lid follow
   * @param {Array<{yaw: number, pitch: number}>} [input.gazeTargets]
   * @param {number} [input.convergeDistCm]
   * @returns {{lidL: number, lidR: number, pupil: number, gazeYaw: number, gazePitch: number}}
   */
  update(dtSec, input) {
    const s = this.scheduler;
    // Camera blink weights → scheduler injection with hysteresis (§11.4).
    this._camBlinkStateL = this._camBlinkStateL ?? false;
    this._camBlinkStateR = this._camBlinkStateR ?? false;
    const thOn = 0.55;
    const thOff = 0.35;
    const cl = input.camBlinkL ?? 0;
    const cr = input.camBlinkR ?? 0;
    if (!this._camBlinkStateL && cl > thOn) {
      this._camBlinkStateL = true;
      s.injectBlink(0, 1);
    } else if (this._camBlinkStateL && cl < thOff) {
      this._camBlinkStateL = false;
    }
    if (!this._camBlinkStateR && cr > thOn) {
      this._camBlinkStateR = true;
      s.injectBlink(0, 2);
    } else if (this._camBlinkStateR && cr < thOff) {
      this._camBlinkStateR = false;
    }

    // Saccades (camera mode adds micro-noise only; iris gaze is primary).
    let startedDeg = 0;
    if (this.toggles.saccades || this.toggles.jitter) {
      startedDeg = this.saccades.update(dtSec, input.gazeTargets);
      if (SaccadeGenerator.isLarge(startedDeg) && this.saccades.rng.next() < CONFIG.saccadeBlinkProb) {
        s.injectBlink(); // 35% of big saccades drag a blink along (§5.5)
      }
    }
    s.update(dtSec, input.jawOpen ?? 0);

    // Gaze source: iris-derived when provided, else saccade generator.
    const sg = this.saccades.gaze();
    let wantYaw;
    let wantPitch;
    if (input.camGazeYaw != null && input.camGazePitch != null) {
      // Saccades/jitter contribute micro-noise on top of camera gaze (§5.5).
      const nx = this.toggles.jitter ? sg.yaw * 0.08 : 0;
      const ny = this.toggles.jitter ? sg.pitch * 0.08 : 0;
      wantYaw = input.camGazeYaw + nx;
      wantPitch = input.camGazePitch + ny;
      this.eyeWorldYaw = wantYaw;
      this.eyeWorldPitch = wantPitch;
    } else {
      wantYaw = sg.yaw;
      wantPitch = sg.pitch;
      this.eyeWorldYaw = wantYaw;
      this.eyeWorldPitch = wantPitch;
    }

    // VOR: eye-in-head = world gaze − head. Clamp-and-reset at 78% range
    // (trap §11.7: without reset the eyes stick at max — the VRChat bug).
    const maxDeg = CONFIG.maxEyeDeg;
    const maxRad = maxDeg * Math.PI / 180;
    const limit = maxRad * CONFIG.vorResetFrac;
    const headYaw = input.headYaw ?? 0;
    const headPitch = input.headPitch ?? 0;
    let inY = (this.eyeWorldYaw * Math.PI / 180) - headYaw;
    let inP = (this.eyeWorldPitch * Math.PI / 180) - headPitch;

    if (this.toggles.vor) {
      const prevY = this._prevHeadYaw ?? headYaw;
      const prevP = this._prevHeadPitch ?? headPitch;
      const headVel = Math.hypot(headYaw - prevY, headPitch - prevP) / Math.max(dtSec, 1e-4);
      this.headMoving = headVel > 0.25; // rad/s
      this._prevHeadYaw = headYaw;
      this._prevHeadPitch = headPitch;

      const over = Math.max(Math.abs(inY) - limit, Math.abs(inP) - limit);
      if (over > 0 && this.vorResetT < 0 && this.headMoving) {
        // Smoothly re-anchor toward neutral over 0.4 s in microsteps (§5.5).
        this.vorResetT = 0;
        this.vorResetFromY = inY;
        this.vorResetFromP = inP;
      }
      if (this.vorResetT >= 0) {
        this.vorResetT += dtSec;
        const u = Math.min(this.vorResetT / CONFIG.vorResetSec, 1);
        const step = minJerk(u);
        // Reset toward 22% inside the limit (multi-step feel comes from the
        // continuous curve; 2–4 visual microsteps land via jitter below).
        const targY = Math.sign(this.vorResetFromY) * limit * 0.78;
        const targP = Math.sign(this.vorResetFromP) * limit * 0.78;
        inY = this.vorResetFromY + (targY - this.vorResetFromY) * step;
        inP = this.vorResetFromP + (targP - this.vorResetFromP) * step;
        if (u >= 1) this.vorResetT = -1;
      }
    }
    inY = Math.min(Math.max(inY, -maxRad), maxRad);
    inP = Math.min(Math.max(inP, -maxRad), maxRad);
    this.eyeInHeadYaw = inY;
    this.eyeInHeadPitch = inP;

    // Listing's law quaternions per eye (convergence adds symmetric eye-in).
    const conv = convergenceDeg(input.convergeDistCm ?? CONFIG.convergeDistCm) * Math.PI / 180;
    const gz = 1;
    // Left eye (negative x offset): convergence rotates inward (+y world).
    const yawL = inY + conv;
    const yawR = inY - conv;
    const gxL = Math.sin(yawL) * Math.cos(inP);
    const gyL = Math.sin(inP);
    const gzL = Math.cos(yawL) * Math.cos(inP);
    const gxR = Math.sin(yawR) * Math.cos(inP);
    const gyR = Math.sin(inP);
    const gzR = Math.cos(yawR) * Math.cos(inP);
    const nL = Math.hypot(gxL, gyL, gzL) || 1;
    const nR = Math.hypot(gxR, gyR, gzR) || 1;
    listingQuaternion(gxL / nL, gyL / nL, gzL / nL, this.outL);
    listingQuaternion(gxR / nR, gyR / nR, gzR / nR, this.outR);

    // Lids from the scheduler, then stabilizeBlink for profile turns.
    const lids = s.lidTargets();
    // Live motion is authoritative, including a held wink/closed eyes. A
    // one-shot scheduler pulse alone reopens the photo's eyes too early.
    stabilizeBlink(input.cameraTracked ? Math.max(0, Math.min(1, cl)) : lids.left,
      input.cameraTracked ? Math.max(0, Math.min(1, cr)) : lids.right, headYaw, this._stabilize);
    let lidL = this._stabilize[0];
    let lidR = this._stabilize[1];

    // Lower lid follows squint and up-gaze (1–2% per 10°); upper lid lags
    // downward gaze by 3 frames (§5.5).
    const upGaze10 = (inP * 180 / Math.PI) / 10;
    const lowerFollow = Math.max(0, upGaze10) * CONFIG.lowerLidPer10Deg
      + (input.squint ?? 0) * 0.35;
    this._lagBuf[this._lagIdx] = inP;
    this._lagIdx = (this._lagIdx + 1) % this._lagBuf.length;
    const laggedP = this._lagBuf[(this._lagIdx + this._lagBuf.length - CONFIG.upperLagFrames) % this._lagBuf.length];
    const downLag = Math.max(0, -laggedP * 180 / Math.PI / 10) * CONFIG.lowerLidPer10Deg;
    lidL = Math.min(1, lidL + lowerFollow * 0.5 + downLag * 0.5);
    lidR = Math.min(1, lidR + lowerFollow * 0.5 + downLag * 0.5);

    // Pupil: hippus + arousal (§5.5).
    const pupil = this.toggles.pupil
      ? pupilScale(this.pink(), input.arousal ?? 0)
      : 1;

    return {
      lidL,
      lidR,
      pupil,
      gazeYaw: inY + headYaw,
      gazePitch: inP + headPitch,
    };
  }
}
