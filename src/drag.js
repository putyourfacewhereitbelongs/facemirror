/**
 * drag.js — click-and-drag facial editing via morph Jacobian solve (spec §6).
 *
 * The solver is pure math (node-testable); DragController takes injected
 * raycast/scene hooks so this module never imports THREE (§9.1).
 *
 * Fast artistic path (spec §6): per-control-vertex Jacobian `J = 3×52`,
 * finite differences at load (52×2 evaluations, cached), then per-frame
 * `wₖ += clamp(dot(δp, uₖ)·step, −0.25, 0.25)` with `uₖ = Jₖ/‖Jₖ‖`.
 * Full ridge-Cholesky is implemented but only used behind a HUD toggle.
 */

import { CH as CHANNEL_INDEX } from './rig.js';

/** @type {Record<string, number>} */
export const CONFIG = {
  step: 0.8,            // §6 update gain
  clampPerIter: 0.25,   // §6 clamp on per-channel change
  undoDepth: 20,        // U = undo last edit, stack of 20 (§6)
  grabRadiusPx: 28,     // nearest control-vertex grab radius (screen px)
  ridgeLambda: 0.08,    // λ for the full solver path
  dollyStep: 0.06,      // wheel/pinch dolly on the neck ring (§6)
  dollyMin: 0.45,
  dollyMax: 2.2,
  jacEps: 1.0,          // morph influence step for finite differences
  cacheKey: 'facemirror.jacobian.v1',
  screenToHeadScale: 0.01, // screen px → head-local units
};

/**
 * Canonical control vertices (§6): mouth corners, lip mid upper/lower,
 * brow inner/outer ×2, chin mid, nose tip, lid mid ×2, ear-side jaw points.
 * Positions are authored in head-local units by headAsset.js; this module
 * only cares about names + channel groups.
 * @type {ReadonlyArray<{name: string, channels: string[], tag: string}>}
 */
export const CONTROLS = Object.freeze([
  { name: 'mouthCornerL', channels: ['mouthSmileLeft', 'mouthDimpleLeft', 'mouthFrownLeft', 'mouthLeft', 'mouthStretchLeft'], tag: 'mouth' },
  { name: 'mouthCornerR', channels: ['mouthSmileRight', 'mouthDimpleRight', 'mouthFrownRight', 'mouthRight', 'mouthStretchRight'], tag: 'mouth' },
  { name: 'lipUpperMid', channels: ['mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthRollUpper', 'mouthShrugUpper', 'mouthPucker'], tag: 'mouth' },
  { name: 'lipLowerMid', channels: ['mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthRollLower', 'mouthShrugLower', 'mouthClose', 'jawOpen'], tag: 'mouth' },
  { name: 'browInnerL', channels: ['browInnerUp', 'browDownLeft'], tag: 'brows' },
  { name: 'browInnerR', channels: ['browInnerUp', 'browDownRight'], tag: 'brows' },
  { name: 'browOuterL', channels: ['browOuterUpLeft', 'browDownLeft'], tag: 'brows' },
  { name: 'browOuterR', channels: ['browOuterUpRight', 'browDownRight'], tag: 'brows' },
  { name: 'chinMid', channels: ['jawOpen', 'jawForward', 'mouthShrugLower'], tag: 'jaw' },
  { name: 'noseTip', channels: ['noseSneerLeft', 'noseSneerRight', 'mouthUpperUpLeft', 'mouthUpperUpRight'], tag: 'nose' },
  { name: 'lidMidL', channels: ['eyeBlinkLeft', 'eyeWideLeft', 'eyeSquintLeft'], tag: 'eyes' },
  { name: 'lidMidR', channels: ['eyeBlinkRight', 'eyeWideRight', 'eyeSquintRight'], tag: 'eyes' },
  { name: 'jawSideL', channels: ['jawLeft', 'mouthLeft'], tag: 'jaw' },
  { name: 'jawSideR', channels: ['jawRight', 'mouthRight'], tag: 'jaw' },
  { name: 'cheekL', channels: ['cheekPuff', 'cheekSquintLeft'], tag: 'cheeks' },
  { name: 'cheekR', channels: ['cheekPuff', 'cheekSquintRight'], tag: 'cheeks' },
]);

/**
 * Per-control Jacobian layout: J[(control*52 + k)*3 + 0..2] = ∂p/∂wₖ.
 * @param {number} nControls
 * @param {number} nChannels
 * @returns {Float32Array}
 */
export function makeJacobian(nControls = CONTROLS.length, nChannels = 52) {
  return new Float32Array(nControls * nChannels * 3);
}

/**
 * Finite-difference Jacobian at load (§6): one base evaluation + one per
 * channel (52×2 influence states overall). `sample(positions)` must fill
 * `out` with control-point positions for the current morph state.
 *
 * @param {Float32Array} jacobian output (makeJacobian)
 * @param {number} nChannels
 * @param {(weights: Float32Array, out: Float32Array) => void} sample
 * @param {number} [eps]
 */
export function computeJacobian(jacobian, nChannels, sample, eps = CONFIG.jacEps) {
  const nControls = jacobian.length / (nChannels * 3);
  const w = new Float32Array(nChannels);
  const base = new Float32Array(nControls * 3);
  const probe = new Float32Array(nControls * 3);
  sample(w, base);
  for (let k = 0; k < nChannels; k++) {
    w[k] = eps;
    sample(w, probe);
    w[k] = 0;
    for (let c = 0; c < nControls; c++) {
      const j = (c * nChannels + k) * 3;
      jacobian[j] = (probe[c * 3] - base[c * 3]) / eps;
      jacobian[j + 1] = (probe[c * 3 + 1] - base[c * 3 + 1]) / eps;
      jacobian[j + 2] = (probe[c * 3 + 2] - base[c * 3 + 2]) / eps;
    }
  }
}

/**
 * One fast solve step (§6): project the residual δp onto each normalized
 * Jacobian column and nudge that channel. Artistic, stable, no matrix
 * factorization — the columns are already near-orthogonal for morphs.
 *
 * @param {Float32Array} jacobian
 * @param {number} control control-vertex index being dragged
 * @param {number} dx @param {number} dy @param {number} dz residual δp
 * @param {Float32Array} weights52 weights in/out
 * @param {number} nChannels
 * @param {number} [step]
 * @param {number} [clampPerIter]
 * @param {number[]|null} [activeChannels] restrict solve to these channel
 *   indices (group freeze)
 * @param {number} [lo] output clamp low (0 for weights, −1 for drag deltas)
 * @param {number} [hi] output clamp high
 */
export function solveJacobianStep(jacobian, control, dx, dy, dz, weights52, nChannels, step = CONFIG.step, clampPerIter = CONFIG.clampPerIter, activeChannels = null, lo = 0, hi = 1) {
  const active = activeChannels ? new Set(activeChannels) : null;
  for (let k = 0; k < nChannels; k++) {
    if (active && !active.has(k)) continue;
    const j = (control * nChannels + k) * 3;
    const jx = jacobian[j];
    const jy = jacobian[j + 1];
    const jz = jacobian[j + 2];
    const norm = Math.hypot(jx, jy, jz);
    if (norm < 1e-8) continue;
    const ux = jx / norm;
    const uy = jy / norm;
    const uz = jz / norm;
    // Fast path (§6): project residual on normalized column uₖ = Jₖ/‖Jₖ‖.
    let delta = (dx * ux + dy * uy + dz * uz) * step;
    if (delta > clampPerIter) delta = clampPerIter;
    if (delta < -clampPerIter) delta = -clampPerIter;
    const v = weights52[k] + delta;
    weights52[k] = v < lo ? lo : v > hi ? hi : v;
  }
}

/**
 * Full ridge-regularized step (§6 fallback): `δw = (JᵀJ+λI)⁻¹Jᵀδp` via
 * Gaussian elimination on the 52×52 normal equations. Only enabled from the
 * perf HUD toggle when the projection path visibly wanders.
 *
 * @param {Float32Array} jacobian
 * @param {number} control
 * @param {number} dx @param {number} dy @param {number} dz
 * @param {Float32Array} weights52
 * @param {number} nChannels
 * @param {number} [lambda]
 * @param {number} [scale]
 * @param {number} [lo] output clamp low
 * @param {number} [hi] output clamp high
 */
export function solveRidgeFull(jacobian, control, dx, dy, dz, weights52, nChannels, lambda = CONFIG.ridgeLambda, scale = CONFIG.step, lo = 0, hi = 1) {
  const cols = new Float32Array(nChannels * 3);
  for (let k = 0; k < nChannels; k++) {
    const j = (control * nChannels + k) * 3;
    cols[k * 3] = jacobian[j];
    cols[k * 3 + 1] = jacobian[j + 1];
    cols[k * 3 + 2] = jacobian[j + 2];
  }
  // Normal equations A = JᵀJ + λI (nChannels × nChannels), b = Jᵀδp.
  const A = new Float32Array(nChannels * nChannels);
  const b = new Float32Array(nChannels);
  for (let k = 0; k < nChannels; k++) {
    let bk = 0;
    for (let t = 0; t < 3; t++) {
      const v = cols[k * 3 + t];
      bk += v * (t === 0 ? dx : t === 1 ? dy : dz);
    }
    b[k] = bk;
    for (let m = 0; m < nChannels; m++) {
      let s = 0;
      for (let t = 0; t < 3; t++) s += cols[k * 3 + t] * cols[m * 3 + t];
      A[k * nChannels + m] = s + (k === m ? lambda : 0);
    }
  }
  // Gaussian elimination with partial pivoting.
  for (let i = 0; i < nChannels; i++) {
    let piv = i;
    for (let r = i + 1; r < nChannels; r++) {
      if (Math.abs(A[r * nChannels + i]) > Math.abs(A[piv * nChannels + i])) piv = r;
    }
    if (piv !== i) {
      for (let c = 0; c < nChannels; c++) {
        const t = A[i * nChannels + c];
        A[i * nChannels + c] = A[piv * nChannels + c];
        A[piv * nChannels + c] = t;
      }
      const t = b[i]; b[i] = b[piv]; b[piv] = t;
    }
    const d = A[i * nChannels + i] || 1e-9;
    for (let r = i + 1; r < nChannels; r++) {
      const f = A[r * nChannels + i] / d;
      if (f === 0) continue;
      for (let c = i; c < nChannels; c++) A[r * nChannels + c] -= f * A[i * nChannels + c];
      b[r] -= f * b[i];
    }
  }
  const x = new Float32Array(nChannels);
  for (let i = nChannels - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < nChannels; c++) s -= A[i * nChannels + c] * x[c];
    x[i] = s / (A[i * nChannels + i] || 1e-9);
  }
  for (let k = 0; k < nChannels; k++) {
    let delta = x[k] * scale;
    if (delta > CONFIG.clampPerIter) delta = CONFIG.clampPerIter;
    if (delta < -CONFIG.clampPerIter) delta = -CONFIG.clampPerIter;
    const v = weights52[k] + delta;
    weights52[k] = v < lo ? lo : v > hi ? hi : v;
  }
}

/**
 * FNV-1a hash for Jacobian cache keys (asset identity).
 * @param {string} s
 * @returns {string}
 */
export function assetHash(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

/**
 * Undo stack (§6: 20 weight snapshots).
 */
export class UndoStack {
  /** @param {number} [depth] */
  constructor(depth = CONFIG.undoDepth) {
    this.depth = depth;
    /** @type {Float32Array[]} */
    this.items = [];
  }
  /** @param {Float32Array} snap */
  push(snap) {
    this.items.push(snap);
    if (this.items.length > this.depth) this.items.shift();
  }
  /** @returns {Float32Array|null} */
  pop() {
    return this.items.pop() ?? null;
  }
  /** @returns {number} */
  get size() {
    return this.items.length;
  }
  clear() {
    this.items.length = 0;
  }
}

/**
 * Drag controller. Dependencies are injected so drag.js stays free of THREE:
 * - `raycast(x, y)` → `{hit: boolean, control: number, depthCm: number}` or
 *   null (control = nearest control-vertex index or −1 for whole-face drag)
 * - `ghostTarget(control, x, y)` shows the ghost marker
 * - `rig` is the live Rig (weightsSource.drag producer writes here)
 * - `pose` is the PoseTracker (rubber-band whole-face rotation)
 */
export class DragController {
  /**
   * @param {object} deps
   * @param {import('./rig.js').Rig} deps.rig
   * @param {import('./pose.js').PoseTracker} deps.pose
   * @param {(x: number, y: number) => {hit: boolean, control: number, depthCm: number, worldX: number, worldY: number, worldZ: number}|null} deps.raycast
   * @param {(control: number, visible: boolean) => void} [deps.ghostTarget]
   * @param {() => void} [deps.onEdit]
   */
  constructor({ rig, pose, raycast, ghostTarget, onEdit }) {
    this.rig = rig;
    this.pose = pose;
    this.raycast = raycast;
    this.ghostTarget = ghostTarget ?? (() => {});
    this.onEdit = onEdit ?? (() => {});
    this.undo = new UndoStack();
    this.active = -1;
    this.dragging = false;
    this.wholeFace = false;
    this.lastX = 0;
    this.lastY = 0;
    this.dolly = 1;
    this.useFullSolver = false;
    /** @type {Float32Array|null} */
    this.jacobian = null;
    this._drag52 = new Float32Array(52);
  }

  /**
   * Install a precomputed Jacobian (from headAsset.js, possibly cached).
   * @param {Float32Array} jacobian
   */
  setJacobian(jacobian) {
    this.jacobian = jacobian;
  }

  /**
   * Pointer down.
   * @param {number} x @param {number} y
   * @returns {boolean} true when a drag started
   */
  begin(x, y) {
    const hit = this.raycast(x, y);
    if (!hit || !hit.hit) return false;
    this.lastX = x;
    this.lastY = y;
    this.undo.push(this.rig.snapshot());
    if (hit.control >= 0) {
      this.active = hit.control;
      this.dragging = true;
      this.wholeFace = false;
      // Freeze the channel group's producer via drag layer (§5.3).
      this._drag52.fill(0);
      this.ghostTarget(hit.control, true);
      return true;
    }
    // Whole-face rotation drag (§6).
    this.wholeFace = true;
    this.dragging = true;
    this.ghostTarget(-1, true);
    return true;
  }

  /**
   * Pointer move while dragging.
   * @param {number} x @param {number} y
   * @param {number} [dz] depth delta for 3D drags (default 0)
   */
  move(x, y, dz = 0) {
    if (!this.dragging) return;
    const dx = x - this.lastX;
    const dy = y - this.lastY;
    this.lastX = x;
    this.lastY = y;
    if (this.wholeFace) {
      this.pose.dragBy(dx, dy);
      return;
    }
    if (!this.jacobian || this.active < 0) return;
    // Screen delta → head-local residual (drag moves the feature with cursor).
    const scale = CONFIG.screenToHeadScale;
    const ctrl = CONTROLS[this.active];
    this._drag52.fill(0);
    const w = this._drag52;
    const residualX = dx * scale;
    const residualY = -dy * scale; // screen y is down; head-local y is up
    const activeIdx = ctrl.channels.map((n) => CHANNEL_INDEX[n]).filter((i) => i !== undefined);
    if (this.useFullSolver) {
      solveRidgeFull(this.jacobian, this.active, residualX, residualY, dz * scale, w, 52,
        CONFIG.ridgeLambda, CONFIG.step, -1, 1);
    } else {
      solveJacobianStep(this.jacobian, this.active, residualX, residualY, dz * scale, w, 52,
        CONFIG.step, CONFIG.clampPerIter, activeIdx, -1, 1);
    }
    for (let k = 0; k < 52; k++) {
      const delta = w[k];
      if (delta !== 0) this.rig.addDragIndex(k, delta);
    }
    this.onEdit();
  }

  /** Pointer up / Esc. */
  end() {
    if (!this.dragging) return;
    this.dragging = false;
    if (this.wholeFace) {
      this.pose.dragRelease();
      this.wholeFace = false;
      this.ghostTarget(-1, false);
      return;
    }
    // Pin the edited channels (🔒) so camera stops overriding them (§5.3).
    const ctrl = CONTROLS[this.active];
    if (ctrl) {
      for (const name of ctrl.channels) this.rig.setPin(name, true);
    }
    this.ghostTarget(this.active, false);
    this.active = -1;
  }

  /** U = undo last edit (20-deep). */
  undoLast() {
    const snap = this.undo.pop();
    if (snap) {
      this.rig.restore(snap);
      return true;
    }
    return false;
  }

  /**
   * Wheel / pinch on the neck ring → dolly (§6).
   * @param {number} delta positive = zoom out
   */
  dollyBy(delta) {
    this.dolly = Math.min(Math.max(this.dolly + delta * CONFIG.dollyStep, CONFIG.dollyMin), CONFIG.dollyMax);
    return this.dolly;
  }
}
