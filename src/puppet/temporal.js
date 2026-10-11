/**
 * Temporal stability primitives.
 *
 * Every animated quantity in the puppet goes through one of these. All of them
 * are frame-rate independent: the blend factor is derived from the real elapsed
 * time (`dt`), so a dropped frame cannot produce a jump. That is the single
 * biggest defence against the flicker a naive per-frame lerp produces.
 */

import { clamp, ramp } from './math3d.js';

/** Exponential blend factor for time constant `tau` (seconds) over `dt`. */
export const smoothK = (tau, dt) => 1 - Math.exp(-dt / Math.max(1e-4, tau));

/** Scalar low-pass with an optional slew-rate limit (used for pose angles). */
export class Scalar {
  constructor(value = 0, tau = 0.07, rate = Infinity) { this.v = value; this.tau = tau; this.rate = rate; }
  set(tau) { this.tau = tau; return this; }
  snap(v) { this.v = v; return this.v; }
  update(target, dt) {
    if (!Number.isFinite(target)) return this.v;
    let step = (target - this.v) * smoothK(this.tau, dt);
    if (Number.isFinite(this.rate)) step = clamp(step, -this.rate * dt, this.rate * dt);
    this.v += step;
    return this.v;
  }
  get value() { return this.v; }
}

/**
 * Continuous gate with two thresholds and asymmetric response times. Replaces
 * every `if (x > threshold)` in the renderer: the output ramps instead of
 * switching, and the release is slower than the attack so noise on the driver
 * signal cannot make teeth / tongue / kiss overlays strobe.
 */
export class Gate {
  constructor(lo, hi, tauUp = 0.05, tauDown = 0.14, v = 0) {
    this.lo = lo; this.hi = hi; this.tauUp = tauUp; this.tauDown = tauDown; this.v = clamp(v, 0, 1);
  }
  update(raw, dt) {
    if (!Number.isFinite(raw)) return this.v;
    const target = ramp(this.lo, this.hi, raw);
    const tau = target > this.v ? this.tauUp : this.tauDown;
    this.v += (target - this.v) * smoothK(tau, dt);
    return this.v;
  }
  snap(v) { this.v = clamp(v, 0, 1); return this.v; }
  get value() { return this.v; }
}

/**
 * One-Euro filter over a flat n*3 field: strong smoothing while the face is
 * still (kills detector noise), high cutoff while it moves (kills lag). Lips get
 * a faster cutoff so speech stays crisp.
 */
export class OneEuroField {
  constructor(n, { minCutoff = 1.2, beta = 22, dCutoff = 1.0, scale = 640 } = {}) {
    this.n = n; this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff; this.scale = scale;
    this.x = new Float32Array(n * 3); this.d = new Float32Array(n * 3); this.init = false; this.t = 0;
    this.boost = null; // per-vertex cutoff multiplier (e.g. 2.5x for lip landmarks)
  }
  reset() { this.init = false; }
  static alpha(cutoff, dt) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  /** `src` may be an array of [x,y,z] or a flat Float32Array of length n*3. */
  update(src, t, weight) {
    const n = this.n;
    if (!this.init) {
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) this.x[i * 3 + k] = flat(src, i, k);
      this.d.fill(0); this.init = true; this.t = t;
      return this.x;
    }
    const dt = clamp((t - this.t) / 1000, 0.002, 0.25);
    this.t = t;
    const aD = OneEuroField.alpha(this.dCutoff, dt);
    const boost = this.boost;
    for (let i = 0; i < n; i++) {
      const b = boost ? boost[i] : 1;
      for (let k = 0; k < 3; k++) {
        const j = i * 3 + k;
        const raw = flat(src, i, k);
        const dv = (raw - this.x[j]) / dt;
        this.d[j] += (dv - this.d[j]) * aD;
        const cutoff = (this.minCutoff + this.beta * (Math.abs(this.d[j]) / this.scale)) * b * (weight || 1);
        this.x[j] += (raw - this.x[j]) * OneEuroField.alpha(cutoff, dt);
      }
    }
    return this.x;
  }
}
function flat(src, i, k) { return Array.isArray(src) ? (src[i] ? src[i][k] || 0 : 0) : src[i * 3 + k] || 0; }

/**
 * Confidence envelope for the tracker. When the detector loses the face the
 * value decays over ~1 s instead of snapping to zero, and on re-acquisition it
 * ramps back in: the puppet relaxes toward its neutral pose and re-engages
 * without a visible pop.
 */
export class Confidence {
  constructor(fallTau = 0.45, riseTau = 0.18) { this.v = 0; this.fallTau = fallTau; this.riseTau = riseTau; this.lostAt = 0; this.holdMs = 120; }
  hit(now) { this.lastHit = now; }
  update(seen, dt) {
    const target = seen ? 1 : 0;
    this.v += (target - this.v) * smoothK(target > this.v ? this.riseTau : this.fallTau, dt);
    return this.v;
  }
  get value() { return this.v; }
}

/** Adaptive "fully open" reference for the eyes: slow to rise, quick to fall. */
export class PeakTracker {
  constructor(initial = 0.25, riseTau = 6, fallTau = 0.35, floor = 0.08) {
    this.v = initial; this.riseTau = riseTau; this.fallTau = fallTau; this.floor = floor;
  }
  update(raw, dt) {
    if (!Number.isFinite(raw)) return this.v;
    const tau = raw > this.v ? this.riseTau : this.fallTau;
    this.v = Math.max(this.floor, this.v + (raw - this.v) * smoothK(tau, dt));
    return this.v;
  }
}
