/**
 * recorder.js — 52-channel 30 fps ring buffer, play/pause, export .json +
 * .anim.glb + pose .glb (spec §5.3 producer "record-playback" / §6 export).
 *
 * `buildClipGlb` is a pure hand-rolled glTF-2.0/GLB writer (node-testable:
 * §9.2 reads it back with a GLTFLoader stub). The browser path uses
 * GLTFExporter only for "Export head" (pose .glb with baked influences).
 */

import { CHANNELS } from './rig.js';

/** @type {Record<string, number>} */
export const CONFIG = {
  fps: 30,
  maxSec: 300,          // hard cap 300 s (§6)
  uiSec: 60,            // UI ring display default 60 s
  jsonVersion: 1,
  glbProxyW: 14,        // head-proxy grid for the self-contained clip GLB
  glbProxyH: 10,
};

/**
 * Fixed-capacity ring buffer of 52-channel frames (trap §11.12: one big
 * preallocated Float32Array, never per-frame allocations).
 */
export class RingBuffer {
  /** @param {number} maxFrames */
  constructor(maxFrames) {
    this.maxFrames = maxFrames;
    this.data = new Float32Array(maxFrames * CHANNELS.length);
    this.times = new Float32Array(maxFrames);
    this.count = 0;
    this.head = 0;
    this.wrapped = false;
  }

  /**
   * @param {Float32Array} frame52
   * @param {number} tSec
   */
  push(frame52, tSec) {
    const n = CHANNELS.length;
    const i = this.head * n;
    this.data.set(frame52.subarray(0, n), i);
    this.times[this.head] = tSec;
    this.head = (this.head + 1) % this.maxFrames;
    if (this.count < this.maxFrames) this.count++;
    else this.wrapped = true;
  }

  /**
   * Oldest-first access.
   * @param {number} k 0 … count−1
   * @param {Float32Array} out52
   * @returns {number} timestamp
   */
  get(k, out52) {
    const n = CHANNELS.length;
    const idx = ((this.head - this.count + k) % this.maxFrames + this.maxFrames) % this.maxFrames;
    out52.set(this.data.subarray(idx * n, idx * n + n));
    return this.times[idx];
  }

  clear() {
    this.count = 0;
    this.head = 0;
    this.wrapped = false;
  }
}

/**
 * Recorder: capture ring + playback producer + exporters.
 */
export class Recorder {
  /**
   * @param {object} deps
   * @param {import('./rig.js').Rig} deps.rig
   * @param {(state: string) => void} [deps.onState]
   */
  constructor({ rig, onState }) {
    this.rig = rig;
    this.onState = onState ?? (() => {});
    this.ring = new RingBuffer(CONFIG.maxSec * CONFIG.fps);
    this.capture = false;
    this.playing = false;
    this.playIdx = 0;
    this._acc = 0;
    this._t0 = 0;
    this._frame = new Float32Array(CHANNELS.length);
    this._framesRecorded = 0;
  }

  /** @returns {boolean} recording state */
  toggleRecord() {
    if (this.capture) {
      this.capture = false;
      this.onState('recording stopped: ' + this._framesRecorded + ' frames');
    } else {
      this.ring.clear();
      this.capture = true;
      this.playing = false;
      this._framesRecorded = 0;
      this._t0 = 0;
      this.onState('recording');
    }
    return this.capture;
  }

  /** Play / pause the ring (record-playback producer §5.3). */
  togglePlay() {
    if (this.playing) {
      this.playing = false;
      this.rig.feedPlayback(null);
      this.onState('playback paused');
    } else {
      if (this.ring.count === 0) {
        this.onState('nothing recorded');
        return false;
      }
      this.playing = true;
      this.playIdx = 0;
      this.capture = false;
      this.onState('playback');
    }
    return this.playing;
  }

  /**
   * Advance at 30 fps sampling (call every substep).
   * @param {number} dtSec
   * @param {number} nowSec monotonic seconds
   */
  update(dtSec, nowSec) {
    if (this.capture) {
      if (this._t0 === 0) this._t0 = nowSec;
      this._acc += dtSec;
      const interval = 1 / CONFIG.fps;
      while (this._acc >= interval) {
        this._acc -= interval;
        this.ring.push(this.rig.weights, nowSec - this._t0);
        this._framesRecorded++;
      }
    }
    if (this.playing && this.ring.count > 0) {
      this._acc += dtSec;
      const interval = 1 / CONFIG.fps;
      while (this._acc >= interval) {
        this._acc -= interval;
        const t = this.ring.get(this.playIdx, this._frame);
        this.rig.feedPlayback(this._frame);
        this.playIdx++;
        if (this.playIdx >= this.ring.count) {
          this.playIdx = 0; // loop the ring
        }
      }
    }
  }

  /**
   * Export .json (§6).
   * @returns {string}
   */
  exportJSON() {
    const frames = [];
    const n = CHANNELS.length;
    const tmp = new Float32Array(n);
    for (let k = 0; k < this.ring.count; k++) {
      this.ring.get(k, tmp);
      frames.push(Array.from(tmp, (v) => Math.round(v * 1000) / 1000));
    }
    return JSON.stringify({
      app: 'facemirror-3d',
      version: CONFIG.jsonVersion,
      fps: CONFIG.fps,
      channels: CHANNELS,
      frames,
    });
  }

  /**
   * Export morph-animated .anim.glb via the pure writer.
   * @param {Float32Array} positions proxy mesh positions (3/vert)
   * @param {Float32Array[]} morphDeltas 52 arrays of 3/vert deltas
   * @returns {ArrayBuffer} GLB bytes
   */
  exportAnimGlb(positions, morphDeltas) {
    const n = CHANNELS.length;
    const times = new Float32Array(this.ring.count);
    const weights = new Float32Array(this.ring.count * n);
    const tmp = new Float32Array(n);
    let tPrev = 0;
    for (let k = 0; k < this.ring.count; k++) {
      const t = this.ring.get(k, tmp);
      times[k] = k === 0 ? 0 : Math.max(t - tPrev, 1 / CONFIG.fps);
      tPrev = t;
      weights.set(tmp, k * n);
    }
    return buildClipGlb({
      positions,
      morphDeltas,
      times,
      weights,
      frameCount: this.ring.count,
      channels: CHANNELS,
    });
  }

  /**
   * Export pose .glb — "a usable 3D image to keep" (§6): the current head
   * with morph influences baked into a base-pose copy.
   * @param {THREE.GLTFExporter} exporter from three/addons (injected)
   * @param {THREE.Object3D} root live head root
   * @returns {Promise<ArrayBuffer>}
   */
  exportPoseGlb(exporter, root) {
    const clone = root.clone(true);
    // Bake current influences into the clone's morph weights (they carry over
    // via clone) — base pose stays, influences persist in the file.
    return new Promise((resolve, reject) => {
      exporter.parse(
        clone,
        (result) => {
          if (result instanceof ArrayBuffer) resolve(result);
          else resolve(new TextEncoder().encode(JSON.stringify(result)).buffer);
        },
        (err) => reject(err),
        { binary: true, onlyVisible: false }
      );
    });
  }
}

// ---------------------------------------------------------------------------
// Pure glTF 2.0 / GLB writer for .anim.glb (§9.2 reads this back in node).
// ---------------------------------------------------------------------------

/**
 * Build a self-contained morph-animated GLB: a small head proxy mesh with 52
 * morph targets and a weights animation carrying the clip.
 *
 * @param {Object} spec
 * @param {Float32Array} spec.positions vertex positions (3/vert)
 * @param {Float32Array[]} spec.morphDeltas 52 × (3/vert) morph POSITION deltas
 * @param {Float32Array} spec.times keyframe times (seconds)
 * @param {Float32Array} spec.weights frameCount × 52 weights
 * @param {number} spec.frameCount
 * @param {string[]} spec.channels
 * @returns {ArrayBuffer} GLB bytes
 */
export function buildClipGlb({ positions, morphDeltas, times, weights, frameCount, channels }) {
  const vertCount = positions.length / 3;
  const morphCount = channels.length;

  // ---- BIN chunk assembly (4-byte aligned pieces) ----
  const binParts = [];
  let binLen = 0;
  const align4 = (n) => (n + 3) & ~3;
  /** @param {ArrayBufferView} view @returns {number} byteOffset */
  const addPart = (view) => {
    const off = align4(binLen);
    if (off > binLen) binParts.push(new Uint8Array(off - binLen));
    binParts.push(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
    binLen = off + view.byteLength;
    return off;
  };

  const minMax = (arr, comps) => {
    const min = new Array(comps).fill(Infinity);
    const max = new Array(comps).fill(-Infinity);
    for (let i = 0; i < arr.length; i += comps) {
      for (let c = 0; c < comps; c++) {
        const v = arr[i + c];
        if (v < min[c]) min[c] = v;
        if (v > max[c]) max[c] = v;
      }
    }
    return { min, max };
  };

  const posOff = addPart(positions);
  const posMM = minMax(positions, 3);

  /** @type {Array<{byteOffset: number, min?: number[], max?: number[]}>} */
  const morphOffs = morphDeltas.map((d) => {
    const off = addPart(d);
    const mm = minMax(d, 3);
    return { byteOffset: off, min: mm.min, max: mm.max };
  });

  const timesOff = addPart(times);
  const tMM = minMax(times, 1);
  const weightsOff = addPart(weights);
  // Pad BIN to 4 bytes
  const binPadded = align4(binLen);
  if (binPadded > binLen) binParts.push(new Uint8Array(binPadded - binLen));

  // ---- JSON ----
  /** @type {any[]} */
  const accessors = [];
  /** @type {any[]} */
  const bufferViews = [];
  const pushView = (byteOffset, byteLength, target) => {
    bufferViews.push({ buffer: 0, byteOffset, byteLength, ...(target ? { target } : {}) });
    return bufferViews.length - 1;
  };
  accessors.push({
    bufferView: pushView(posOff, positions.byteLength, 34962),
    componentType: 5126,
    count: vertCount,
    type: 'VEC3',
    min: posMM.min,
    max: posMM.max,
  });
  for (const m of morphOffs) {
    accessors.push({
      bufferView: pushView(m.byteOffset, morphDeltas[0].byteLength, 34962),
      componentType: 5126,
      count: vertCount,
      type: 'VEC3',
      min: m.min,
      max: m.max,
    });
  }
  const timesAccessor = accessors.length;
  accessors.push({
    bufferView: pushView(timesOff, times.byteLength, 34962),
    componentType: 5126,
    count: times.length,
    type: 'SCALAR',
    min: tMM.min,
    max: tMM.max,
  });
  const weightsAccessor = accessors.length;
  accessors.push({
    bufferView: pushView(weightsOff, weights.byteLength, 34962),
    componentType: 5126,
    count: frameCount * morphCount,
    type: 'SCALAR',
  });

  const gltf = {
    asset: { version: '2.0', generator: 'facemirror-3d recorder' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'FaceMirrorClip', weights: new Array(morphCount).fill(0) }],
    meshes: [{
      name: 'clipHead',
      primitives: [{ attributes: { POSITION: 0 }, targets: morphDeltas.map((_, i) => ({ POSITION: 1 + i })) }],
      weights: new Array(morphCount).fill(0),
    }],
    animations: [{
      name: 'facemirror-52',
      samplers: [{ input: timesAccessor, output: weightsAccessor, interpolation: 'LINEAR' }],
      channels: [{ sampler: 0, target: { node: 0, path: 'weights' } }],
    }],
    buffer: [{ byteLength: binPadded }],
    bufferViews,
    accessors,
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonChunkLen = align4(jsonBytes.length);
  const total = 12 + 8 + jsonChunkLen + 8 + binPadded;
  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);
  // Header
  dv.setUint32(0, 0x46546C67, true); // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  // JSON chunk
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, 0x4E4F534A, true); // 'JSON'
  u8.set(jsonBytes, 20);
  for (let i = jsonBytes.length; i < jsonChunkLen; i++) u8[20 + i] = 0x20; // space pad
  // BIN chunk
  const binHeader = 20 + jsonChunkLen;
  dv.setUint32(binHeader, binPadded, true);
  dv.setUint32(binHeader + 4, 0x004E4942, true); // 'BIN\0'
  let p = binHeader + 8;
  for (const part of binParts) {
    u8.set(part, p);
    p += part.length;
  }
  return out;
}

/**
 * Minimal GLB JSON extractor — the "GLTFLoader stub" of §9.2 (used by tests
 * and the export sanity check in the browser console self-check).
 * @param {ArrayBuffer} glb
 * @returns {any} parsed glTF JSON (+ binary byte length)
 */
export function parseGlbJson(glb) {
  const dv = new DataView(glb);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a GLB (bad magic)');
  const version = dv.getUint32(4, true);
  const total = dv.getUint32(8, true);
  if (version !== 2) throw new Error('GLB version must be 2');
  if (total > glb.byteLength) throw new Error('GLB length mismatch');
  const jsonLen = dv.getUint32(12, true);
  const jsonType = dv.getUint32(16, true);
  if (jsonType !== 0x4E4F534A) throw new Error('first chunk must be JSON');
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(glb, 20, jsonLen)));
  let binBytes = 0;
  if (glb.byteLength > 20 + jsonLen + 8) {
    binBytes = dv.getUint32(20 + jsonLen, true);
    const binType = dv.getUint32(20 + jsonLen + 4, true);
    if (binType !== 0x004E4942) throw new Error('second chunk must be BIN');
  }
  return { json, binBytes };
}
