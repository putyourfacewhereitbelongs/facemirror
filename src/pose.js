/**
 * pose.js — head pose: quaternion from the facial transformation matrix,
 * clamps, inertia (neck lag), and rubber-band drag (spec §5.4 / §6).
 * Pure math: no DOM, no THREE (§9.1). Quats are Float32Array(4) [x,y,z,w].
 *
 * Trap §11.5: pose comes from the transformation matrix, never from
 * image-space landmark slopes.
 */

/** @type {Record<string, number>} */
export const CONFIG = {
  followOmega: 18,          // critically damped ζ=1 ω≈18 head follow (§5.4)
  neckLagSec: 0.05,         // +0.05 s extra rotation lag for the neck (§5.4)
  yawClampRad: 80 * Math.PI / 180,
  pitchClampRad: 50 * Math.PI / 180,
  rollClampRad: 40 * Math.PI / 180,
  rubberBand: 0.35,         // movement scale beyond the clamp (§6)
  dragSensitivity: 0.0055,  // rad per pixel of whole-face drag
  idleWanderYawDeg: 0.6,    // idle 0.1 Hz gaze/pose wander (§2)
  idleWanderPitchDeg: 0.35,
  idleWanderHz: 0.1,
};

// ---------------------------------------------------------------------------
// Quaternion helpers
// ---------------------------------------------------------------------------

/**
 * Quaternion from the rotation part of a 4×4 column-major matrix.
 * @param {Float32Array|number[]} m 16 floats column-major
 * @param {Float32Array} out 4 floats
 * @returns {Float32Array} out
 */
export function quatFromMatrix(m, out) {
  // Column-major: m[0..3] = col0 ...
  const m00 = m[0], m10 = m[1], m20 = m[2];
  const m01 = m[4], m11 = m[5], m21 = m[6];
  const m02 = m[8], m12 = m[9], m22 = m[10];
  const tr = m00 + m11 + m22;
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  out[0] = x; out[1] = y; out[2] = z; out[3] = w;
  return out;
}

/**
 * @param {Float32Array} a @param {Float32Array} b @param {Float32Array} out
 * @returns {Float32Array} a·b
 */
export function quatMul(a, b, out) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

/**
 * @param {Float32Array} a @param {Float32Array} out
 * @returns {Float32Array} conjugate (inverse for unit quats)
 */
export function quatConj(a, out) {
  out[0] = -a[0]; out[1] = -a[1]; out[2] = -a[2]; out[3] = a[3];
  return out;
}

/**
 * @param {Float32Array} a @param {Float32Array} out
 * @returns {Float32Array} normalized a
 */
export function quatNormalize(a, out) {
  const n = Math.hypot(a[0], a[1], a[2], a[3]) || 1;
  out[0] = a[0] / n; out[1] = a[1] / n; out[2] = a[2] / n; out[3] = a[3] / n;
  return out;
}

/**
 * Slerp.
 * @param {Float32Array} a @param {Float32Array} b @param {number} t 0..1
 * @param {Float32Array} out
 * @returns {Float32Array}
 */
export function quatSlerp(a, b, t, out) {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cos < 0) {
    cos = -cos;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  let s0, s1;
  if (cos > 0.9995) {
    s0 = 1 - t;
    s1 = t;
  } else {
    const theta = Math.acos(cos);
    const sin = Math.sin(theta);
    s0 = Math.sin((1 - t) * theta) / sin;
    s1 = Math.sin(t * theta) / sin;
  }
  out[0] = s0 * a[0] + s1 * bx;
  out[1] = s0 * a[1] + s1 * by;
  out[2] = s0 * a[2] + s1 * bz;
  out[3] = s0 * a[3] + s1 * bw;
  return quatNormalize(out, out);
}

/**
 * Rotation-vector (log map) of the relative rotation conj(a)·b, scaled:
 * used to spring quaternions component-wise.
 * @param {Float32Array} a @param {Float32Array} b
 * @param {Float32Array} out3 axis×angle floats
 */
export function quatLogRelative(a, b, out3) {
  const inv = quatConj(a, new Float32Array(4));
  const rel = quatMul(inv, b, new Float32Array(4));
  const w = Math.min(Math.max(rel[3], -1), 1);
  const angle = 2 * Math.acos(w);
  const s = Math.sqrt(1 - w * w);
  if (s < 1e-6 || angle < 1e-6) {
    out3[0] = 0; out3[1] = 0; out3[2] = 0;
    return out3;
  }
  out3[0] = rel[0] / s * angle;
  out3[1] = rel[1] / s * angle;
  out3[2] = rel[2] / s * angle;
  return out3;
}

/**
 * @param {Float32Array} rotvec 3 floats axis×angle
 * @param {Float32Array} out4 quaternion
 */
export function quatFromRotvec(rotvec, out4) {
  const angle = Math.hypot(rotvec[0], rotvec[1], rotvec[2]);
  if (angle < 1e-8) {
    out4[0] = 0; out4[1] = 0; out4[2] = 0; out4[3] = 1;
    return out4;
  }
  const s = Math.sin(angle / 2) / angle;
  out4[0] = rotvec[0] * s;
  out4[1] = rotvec[1] * s;
  out4[2] = rotvec[2] * s;
  out4[3] = Math.cos(angle / 2);
  return out4;
}

/**
 * Extract yaw/pitch/roll (Y-X-Z order, "Fick") from a quaternion.
 * @param {Float32Array} q
 * @param {Float32Array} out3 [yaw, pitch, roll] radians
 */
export function quatToYawPitchRoll(q, out3) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const sinp = 2 * (w * x - y * z);
  const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
  const siny = 2 * (w * y + z * x);
  const cosy = 1 - 2 * (x * x + y * y);
  const yaw = Math.atan2(siny, cosy);
  const sinr = 2 * (w * z + x * y);
  const cosr = 1 - 2 * (x * x + z * z);
  const roll = Math.atan2(sinr, cosr);
  out3[0] = yaw; out3[1] = pitch; out3[2] = roll;
  return out3;
}

/**
 * Rubber-band clamp: beyond `limit` extra movement is compressed.
 * @param {number} v value
 * @param {number} limit soft clamp
 * @param {number} band movement scale beyond the clamp (0..1)
 * @returns {number}
 */
export function rubberBand(v, limit, band = CONFIG.rubberBand) {
  if (Math.abs(v) <= limit) return v;
  const over = Math.abs(v) - limit;
  return Math.sign(v) * (limit + over * band);
}

/**
 * Head pose tracker: matrix → target quaternion, critically-damped follow
 * with neck lag, drag offsets with rubber-band clamps, z0-relative depth.
 */
export class PoseTracker {
  constructor() {
    this.quat = new Float32Array([0, 0, 0, 1]);   // smoothed head quat
    this.target = new Float32Array([0, 0, 0, 1]);
    this._m = new Float32Array(16);
    this._tmp4 = new Float32Array(4);
    this._log = new Float32Array(3);
    this._logVel = new Float32Array(3);
    this._ypr = new Float32Array(3);
    this._lagQueue = []; // neck-lag ring of target quats (0.05 s)
    this.pos = new Float32Array(3);
    this.posTarget = new Float32Array(3);
    this.z0 = null; // calibrated neutral depth (§4.4)
    this.neckLagSec = CONFIG.neckLagSec; // UI "Head lag" slider (§10)
    this.dragYaw = 0;
    this.dragPitch = 0;
    this.dragVelYaw = 0;
    this.dragVelPitch = 0;
    this.idleWanderT = 0;
  }

  /** @param {number|null} z0 neutral translation z from calibration */
  setNeutralDepth(z0) {
    this.z0 = z0;
  }

  /**
   * Feed one detection frame (only on new camera frames!).
   * @param {Float32Array|number[]|null} matrixData 16 floats or null
   * @param {number} timestampSec monotonic time (for neck-lag queue)
   */
  setFromMatrix(matrixData, timestampSec) {
    if (!matrixData) return;
    quatFromMatrix(matrixData, this.target);
    quatNormalize(this.target, this.target);
    this._lagQueue.push({ t: timestampSec, q: this.target.slice() });
    // Drop entries older than CONFIG.neckLagSec*3 (queue window).
    while (this._lagQueue.length > 12) this._lagQueue.shift();
    const tx = matrixData[12];
    const ty = matrixData[13];
    const tz = matrixData[14];
    if (this.z0 === null) this.z0 = tz;
    this.posTarget[0] = tx * 0.01; // cm → m-ish scene units
    this.posTarget[1] = ty * 0.01;
    this.posTarget[2] = (tz - this.z0) * 0.01;
  }

  /** Smoothly return to neutral on camera stop/loss; keep the image rig intact. */
  resetTracking() {
    this.target.set([0, 0, 0, 1]);
    this.posTarget.fill(0);
    this._lagQueue.length = 0;
    this.z0 = null;
    this.dragRelease();
  }

  /**
   * Whole-face rotation drag (§6: grab anywhere that is not a control point).
   * @param {number} dxPx pointer delta x
   * @param {number} dyPx pointer delta y
   */
  dragBy(dxPx, dyPx) {
    this._dragReleased = false;
    this.dragYaw += dxPx * CONFIG.dragSensitivity;
    this.dragPitch += dyPx * CONFIG.dragSensitivity;
    this.dragYaw = rubberBand(this.dragYaw, CONFIG.yawClampRad * 1.25);
    this.dragPitch = rubberBand(this.dragPitch, CONFIG.pitchClampRad * 1.25);
  }

  /** Release drag: spring back to camera-driven pose (rubber-band snap). */
  dragRelease() {
    this._dragReleased = true;
  }

  /**
   * Advance one fixed substep (§5.4: fixed-rate substeps).
   * @param {number} dtSec substep seconds
   * @param {number} timestampSec current monotonic seconds
   * @param {boolean} [idle] add idle wander when the face is absent
   */
  update(dtSec, timestampSec, idle = false) {
    // Neck lag: follow the delayed target (queue head of this.neckLagSec).
    let qTarget = this.target;
    const wantT = timestampSec - this.neckLagSec;
    for (const e of this._lagQueue) {
      if (e.t <= wantT) qTarget = e.q;
      else break;
    }

    // Idle wander (0.1 Hz) so the head is never a frozen stare (§2).
    if (idle) {
      this.idleWanderT += dtSec;
      const w = CONFIG.idleWanderHz * 2 * Math.PI * this.idleWanderT;
      const yaw = Math.sin(w) * CONFIG.idleWanderYawDeg * Math.PI / 180;
      const pitch = Math.sin(w * 0.63 + 1.7) * CONFIG.idleWanderPitchDeg * Math.PI / 180;
      const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
      const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
      this._tmp4[0] = sp * cy;
      this._tmp4[1] = sy * cp;
      this._tmp4[2] = -sy * sp;
      this._tmp4[3] = cy * cp;
      qTarget = this._tmp4;
    }

    // Critically-damped follow: exponential slerp toward the lagged target
    // (ζ=1 ⇒ no overshoot; inertia comes from the neck-lag queue above).
    quatLogRelative(this.quat, qTarget, this._log);
    const k = 1 - Math.exp(-CONFIG.followOmega * dtSec);
    this._log[0] *= k;
    this._log[1] *= k;
    this._log[2] *= k;
    quatFromRotvec(this._log, this._tmp4);
    quatMul(this.quat, this._tmp4, this.quat);
    quatNormalize(this.quat, this.quat);

    // Rubber-band spring-back after release (§6).
    if (this._dragReleased && (this.dragYaw !== 0 || this.dragPitch !== 0)) {
      const decay = Math.exp(-10 * dtSec);
      this.dragYaw *= decay;
      this.dragPitch *= decay;
      if (Math.abs(this.dragYaw) < 1e-4 && Math.abs(this.dragPitch) < 1e-4) {
        this.dragYaw = 0;
        this.dragPitch = 0;
        this._dragReleased = false;
      }
    }

    // Drag offset applied as extra yaw/pitch around the tracked pose.
    if (this.dragYaw !== 0 || this.dragPitch !== 0) {
      const cy = Math.cos(this.dragYaw / 2), sy = Math.sin(this.dragYaw / 2);
      const cp = Math.cos(this.dragPitch / 2), sp = Math.sin(this.dragPitch / 2);
      this._tmp4[0] = sp * cy;
      this._tmp4[1] = sy * cp;
      this._tmp4[2] = -sy * sp;
      this._tmp4[3] = cy * cp;
      quatMul(this._tmp4, this.quat, this._dragQuat ??= new Float32Array(4));
    }

    // Position follow (same spring, slower).
    for (let i = 0; i < 3; i++) {
      this.pos[i] += (this.posTarget[i] - this.pos[i]) * Math.min(1, CONFIG.followOmega * dtSec * 0.6);
    }
  }

  /**
   * Final head quaternion including drag offset.
   * @param {Float32Array} out4
   */
  getQuaternion(out4) {
    if (this.dragYaw !== 0 || this.dragPitch !== 0) {
      out4.set(this._dragQuat ?? this.quat);
    } else {
      out4.set(this.quat);
    }
    return out4;
  }

  /**
   * Current yaw/pitch/roll (for HUD, blink stabilization, debugging).
   * @param {Float32Array} out3
   */
  getYawPitchRoll(out3) {
    return quatToYawPitchRoll(this.quat, out3);
  }
}
