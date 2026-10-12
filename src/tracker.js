/**
 * tracker.js — getUserMedia + MediaPipe FaceLandmarker loop (spec §2).
 *
 * Trap §11.1: numFaces:1 or temporal smoothing silently off.
 * Trap §11.2: detectForVideo timestamps must be strictly increasing.
 * Trap §11.12: every heavy array preallocated — no per-frame `new` in the
 * loop; GC pauses look like face stutter on a 4 GB machine.
 */

import { CHANNELS, CH } from './rig.js';

/** @type {Record<string, any>} */
export const CONFIG = {
  // Local-first for offline (§0.4); CDN only as fallback (§0.3 pins).
  visionModule: './vendor/tasks-vision/vision_bundle.mjs',
  visionModuleCdn: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs',
  wasmLocal: './vendor/tasks-vision/wasm',
  modelLocal: './assets/face_landmarker.task',
  modelCdn: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  constraints: {
    audio: false,
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 },
      facingMode: 'user',
    },
  },
  idleAfterSec: 0.8,        // no face for >0.8 s → rig.goIdle() (§2)
  detectMinIntervalMs: 15,  // never detect more than ~66 Hz
};

/**
 * Map MediaPipe blendshape category names → our channel indices (trap
 * §11.11: exact-case ARKit names; missing names just stay unused).
 * @param {Array<{categoryName: string}>} categories
 * @returns {number[]} index map parallel to categories
 */
function buildCategoryIndex(categories) {
  return categories.map((c) => CH[c.categoryName] ?? -1);
}

/**
 * Synthetic face generator for headless smoke tests and camera-free demos.
 * Activate with `window.__FACEMIRROR_FAKE = 'happy|blink|turn|nosecret'`.
 */
export class FakeDetector {
  /** @param {string} script pipe-separated scenario words */
  constructor(script) {
    this.scenarios = script.split('|').map((s) => s.trim()).filter(Boolean);
    this.t = 0;
    this.frame = 0;
    this.weights = new Float32Array(CHANNELS.length);
    this.matrix = new Float32Array(16);
    this.landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    this.matrix.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    this.catIndex = CHANNELS.map((_, i) => i);
    this.hasFace = true;
    this.buildLandmarks();
  }

  /** Cheap static landmark cloud (enough for geometry solvers in tests). */
  buildLandmarks() {
    const L = this.landmarks;
    // Eyes roughly at (0.38, 0.42) and (0.62, 0.42); mouth at (0.5, 0.62).
    const set = (i, x, y, z) => { L[i].x = x; L[i].y = y; L[i].z = z; };
    set(33, 0.36, 0.42, -0.02); set(133, 0.40, 0.42, -0.02);
    set(159, 0.38, 0.405, -0.02); set(145, 0.38, 0.435, -0.02);
    set(263, 0.64, 0.42, -0.02); set(362, 0.60, 0.42, -0.02);
    set(386, 0.62, 0.405, -0.02); set(374, 0.62, 0.435, -0.02);
    set(468, 0.38, 0.42, -0.02); set(473, 0.62, 0.42, -0.02);
    set(61, 0.42, 0.62, -0.02); set(291, 0.58, 0.62, -0.02);
    set(0, 0.50, 0.585, -0.02); set(13, 0.50, 0.60, -0.02);
    set(14, 0.50, 0.625, -0.02); set(17, 0.50, 0.65, -0.02);
    set(181, 0.455, 0.645, -0.02); set(405, 0.545, 0.645, -0.02);
    set(1, 0.50, 0.515, -0.03); set(168, 0.50, 0.47, -0.02); set(6, 0.50, 0.46, -0.02);
    set(98, 0.465, 0.535, -0.02); set(327, 0.535, 0.535, -0.02);
    set(129, 0.475, 0.555, -0.02); set(358, 0.525, 0.555, -0.02);
    set(50, 0.33, 0.56, -0.01); set(280, 0.67, 0.56, -0.01);
    set(152, 0.50, 0.72, -0.02); set(199, 0.50, 0.70, -0.02);
  }

  /**
   * @param {number} dtSec
   * @returns {{hasFace: boolean, weights: Float32Array, matrix: Float32Array, landmarks: Array<{x:number,y:number,z:number}>}}
   */
  detect(dtSec) {
    this.t += dtSec;
    this.frame++;
    const w = this.weights;
    w.fill(0);
    let yaw = 0;
    for (const s of this.scenarios) {
      if (s === 'happy') {
        // Slow smile + brow raise + blink flutter.
        const p = Math.min(this.t / 1.5, 1);
        w[CH.mouthSmileLeft] = 0.75 * p;
        w[CH.mouthSmileRight] = 0.75 * p;
        w[CH.cheekSquintLeft] = 0.5 * p;
        w[CH.cheekSquintRight] = 0.5 * p;
        w[CH.browInnerUp] = 0.2 * p;
      } else if (s === 'blink') {
        const phase = (this.t % 3.2);
        const b = phase < 0.18 ? Math.sin((phase / 0.18) * Math.PI) : 0;
        w[CH.eyeBlinkLeft] = b;
        w[CH.eyeBlinkRight] = b;
      } else if (s === 'turn') {
        yaw = Math.sin(this.t * 0.5) * 0.7;
      } else if (s === 'nosecret') {
        // Deliberately flat — asserts dead-channel revival has signal to find.
        w[CH.jawOpen] = 0.1 * Math.sin(this.t * 2);
      }
    }
    // Rotation matrix about Y by yaw (column-major).
    const c = Math.cos(yaw);
    const sN = Math.sin(yaw);
    this.matrix.set([c, 0, -sN, 0, 0, 1, 0, 0, sN, 0, c, 0, 0, 0, 30, 1]);
    this.hasFace = this.scenarios.length > 0;
    return {
      hasFace: this.hasFace,
      weights: w,
      matrix: this.matrix,
      landmarks: this.landmarks,
    };
  }
}

/**
 * Mic-driven jaw + viseme bias (§5.6): AudioWorklet with a 64-bin FFT → band
 * energies → jawOpen = LP(12 Hz)(energy·2.2) + formant-ratio A/E/I/O/U bias.
 * DEFAULT OFF, clearly labeled in the UI — privacy: audio never leaves the
 * page, like the video.
 */
export class MicJaw {
  constructor() {
    this.ctx = null;
    this.node = null;
    this.stream = null;
    this.jawOpen = 0;
    this.viseme = { a: 0, e: 0, i: 0, o: 0, u: 0 };
    this.enabled = false;
    this._lp = 0; // 12 Hz low-pass state
    this._lastT = 0;
  }

  /** @returns {Promise<void>} */
  async start() {
    if (this.enabled) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const workletCode = `
      class FMFFT extends AudioWorkletProcessor {
        constructor() {
          super();
          this.buf = new Float32Array(128);
          this.n = 0;
          // 128-point radix-2 FFT tables
          this.re = new Float32Array(128);
          this.im = new Float32Array(128);
        }
        fft() {
          const re = this.re, im = this.im, N = 128;
          for (let i = 1, j = 0; i < N; i++) {
            let bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
          }
          for (let len = 2; len <= N; len <<= 1) {
            const ang = -2 * Math.PI / len;
            const wr = Math.cos(ang), wi = Math.sin(ang);
            for (let i = 0; i < N; i += len) {
              let cr = 1, ci = 0;
              for (let k = 0; k < len / 2; k++) {
                const ur = re[i + k], ui = im[i + k];
                const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
              }
            }
          }
        }
        process(inputs) {
          const ch = inputs[0] && inputs[0][0];
          if (!ch) return true;
          for (let i = 0; i < ch.length; i++) {
            this.buf[this.n++] = ch[i];
            if (this.n === 128) {
              this.n = 0;
              let energy = 0;
              for (let k = 0; k < 128; k++) {
                energy += this.buf[k] * this.buf[k];
                this.re[k] = this.buf[k];
                this.im[k] = 0;
              }
              this.fft();
              // 64 bins of magnitude
              const bins = new Float32Array(64);
              for (let k = 0; k < 64; k++) {
                bins[k] = Math.hypot(this.re[k], this.im[k]) / 64;
              }
              this.port.postMessage({ rms: Math.sqrt(energy / 128), bins });
            }
          }
          return true;
        }
      }
      registerProcessor('fm-fft', FMFFT);
    `;
    const url = URL.createObjectURL(new Blob([workletCode], { type: 'application/javascript' }));
    await this.ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'fm-fft');
    this.node.port.onmessage = (e) => this._onLevels(e.data);
    src.connect(this.node);
    this.enabled = true;
    this._lastT = performance.now();
  }

  stop() {
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    if (this.ctx) this.ctx.close();
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.enabled = false;
    this.jawOpen = 0;
  }

  /**
   * Band energies → jaw + viseme bias (§5.6).
   * @param {{rms: number, bins: Float32Array}} data
   */
  _onLevels(data) {
    const now = performance.now();
    const dt = Math.min(Math.max((now - this._lastT) / 1000, 0.001), 0.25);
    this._lastT = now;
    const { rms, bins } = data;
    // jawOpen = LP(12 Hz)(energy · 2.2), clamped.
    const target = Math.min(Math.max(rms * 2.2 * 12, 0), 1); // speech RMS scale
    const a = 1 - Math.exp(-2 * Math.PI * 12 * dt);
    this._lp += (target - this._lp) * a;
    this.jawOpen = this._lp;

    // 3 band energies from the 64 bins (≈0–12 kHz at 48 kHz/128).
    let b1 = 0;
    let b2 = 0;
    let b3 = 0;
    for (let i = 1; i < 8; i++) b1 += bins[i];
    for (let i = 8; i < 24; i++) b2 += bins[i];
    for (let i = 24; i < 64; i++) b3 += bins[i];
    const total = b1 + b2 + b3 + 1e-6;
    const rMid = b2 / total;
    const rHigh = b3 / total;
    const rLow = b1 / total;
    // Formant-ratio → A/E/I/O/U bias (heuristic: jaw vowel space).
    const v = this.viseme;
    v.a = Math.min(1, rLow * 1.6);
    v.e = Math.min(1, rMid * 1.8);
    v.i = Math.min(1, rHigh * 2.2);
    v.o = Math.min(1, (rLow + rMid) * 0.9);
    v.u = Math.min(1, rLow * 1.1 * (1 - rHigh));
  }

  /**
   * Write into the rig's mic producer layer (camera-family: §5.3 merge).
   * @param {import('./rig.js').Rig} rig
   */
  feed(rig) {
    rig.feedMic(this.jawOpen, this.viseme);
  }
}

/**
 * Webcam + FaceLandmarker tracker. Exact MediaPipe API usage from §2.
 */
export class Tracker {
  /**
   * @param {object} deps
   * @param {(status: string, kind?: string) => void} deps.onStatus UI feedback
   */
  constructor({ onStatus }) {
    this.onStatus = onStatus ?? (() => {});
    this.video = null;
    this.stream = null;
    this.landmarker = null;
    this.lastVideoTime = -1;
    this.lastTs = -1;          // trap §11.2: strictly increasing
    this.faceAbsentSince = -1;
    this.detectEveryNth = 1;
    this._frameCounter = 0;
    this._lastDetectMs = 0;
    /** Preallocated outputs (trap §11.12). */
    this.weights = new Float32Array(CHANNELS.length);
    this.matrix = new Float32Array(16);
    this.landmarks = [];
    this.hasFace = false;
    this.fpsDetect = 0;
    this._fpsAcc = 0;
    this._fpsN = 0;
    /** @type {FakeDetector|null} */
    this.fake = null;
    if (typeof window !== 'undefined' && window.__FACEMIRROR_FAKE) {
      this.fake = new FakeDetector(String(window.__FACEMIRROR_FAKE));
      this.onStatus('FakeDetector: ' + window.__FACEMIRROR_FAKE, 'debug');
    }
    this.categoryIndex = null;
    /** Detection-time ring for the §2 capture downgrade (p95 > 30 ms). */
    this._detectRing = new Float32Array(40);
    this._detectRingN = 0;
    this._detectRingHead = 0;
    this._captureDowngraded = false;
    this._tDowngrade = 0;
  }

  /**
   * Load MediaPipe (vendored wasm; CDN module with local-first model — §0.4).
   * Falls back to CPU delegate automatically.
   * @returns {Promise<void>}
   */
  async initLandmarker() {
    if (this.landmarker || this.fake) return;
    let visionMod;
    try {
      visionMod = await import(CONFIG.visionModule);
    } catch {
      this.onStatus('Local tasks-vision missing — CDN fallback', 'warn');
      visionMod = await import(CONFIG.visionModuleCdn);
    }
    const { FilesetResolver, FaceLandmarker } = visionMod;
    const fileset = await FilesetResolver.forVisionTasks(CONFIG.wasmLocal);
    const make = (delegate, modelPath) => FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: modelPath ?? CONFIG.modelLocal,
        delegate,
      },
      runningMode: 'VIDEO',
      numFaces: 1, // REQUIRED: trap §11.1 — temporal smoothing needs numFaces=1
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      this.landmarker = await make('GPU');
    } catch {
      this.onStatus('GPU delegate unavailable — using CPU', 'warn');
      try {
        this.landmarker = await make('CPU');
      } catch (err) {
        this.onStatus('Local model load failed — trying CDN fallback', 'warn');
        try {
          this.landmarker = await make('CPU', CONFIG.modelCdn);
        } catch {
          throw err;
        }
      }
    }
  }

  /**
   * Start the camera.
   * @param {HTMLVideoElement} video
   */
  async start(video) {
    this.video = video;
    if (!this.fake) {
      await this.initLandmarker();
      try {
        this.stream = await navigator.mediaDevices.getUserMedia(CONFIG.constraints);
      } catch (err) {
        this.onStatus('Camera unavailable: ' + (err && err.name ? err.name : err) + ' — manual mode', 'error');
        throw err;
      }
      video.srcObject = this.stream;
      await video.play();
      this.onStatus('Camera live', 'ok');
    } else {
      this.onStatus('FakeDetector active — camera bypassed', 'debug');
    }
  }

  /** Stop camera + detector. */
  stop() {
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
    if (this.landmarker && this.landmarker.close) this.landmarker.close();
    this.landmarker = null;
    this.lastVideoTime = -1;
    this.lastTs = -1;
  }

  /**
   * Per-rAF poll. Detects only on NEW video frames (§2) and only when the
   * perf governor's every-Nth gate allows.
   *
   * @param {number} nowMs performance.now()
   * @param {number} dtSec frame delta
   * @returns {boolean} true when a detection result is fresh in .weights/.matrix
   */
  poll(nowMs, dtSec) {
    if (this.fake) {
      this._frameCounter++;
      if (this._frameCounter % this.detectEveryNth !== 0) return false;
      const r = this.fake.detect(dtSec);
      this.weights.set(r.weights);
      this.matrix.set(r.matrix);
      this.landmarks = r.landmarks;
      this.hasFace = r.hasFace;
      this._trackFps(nowMs);
      return true;
    }

    const video = this.video;
    if (!video || video.readyState < 2) return false;
    if (video.currentTime === this.lastVideoTime) return false; // no new frame
    this.lastVideoTime = video.currentTime;

    this._frameCounter++;
    if (this._frameCounter % this.detectEveryNth !== 0) return false;
    if (nowMs - this._lastDetectMs < CONFIG.detectMinIntervalMs) return false;

    // Trap §11.2: strictly increasing timestamps; skip stale repeats.
    let ts = nowMs;
    if (ts <= this.lastTs) ts = this.lastTs + 0.01;
    this.lastTs = ts;

    let res = null;
    const tDetect = typeof performance !== 'undefined' ? performance.now() : 0;
    try {
      res = this.landmarker.detectForVideo(video, ts);
    } catch {
      return false; // never spam the console (§2)
    }
    this._lastDetectMs = nowMs;
    if (tDetect) {
      this._detectRing[this._detectRingHead] = performance.now() - tDetect;
      this._detectRingHead = (this._detectRingHead + 1) % this._detectRing.length;
      if (this._detectRingN < this._detectRing.length) this._detectRingN++;
      this._maybeDowngradeCapture(nowMs);
    }

    const lm = res.faceLandmarks && res.faceLandmarks[0];
    this.hasFace = !!lm;
    if (!lm) return true; // absence is information too (idle trigger)

    this.landmarks = lm;
    const bs = res.faceBlendshapes && res.faceBlendshapes[0];
    if (bs) {
      if (!this.categoryIndex) this.categoryIndex = buildCategoryIndex(bs.categories);
      const cats = bs.categories;
      const idx = this.categoryIndex;
      for (let i = 0; i < cats.length; i++) {
        const ci = idx[i];
        if (ci >= 0) this.weights[ci] = cats[i].score;
      }
    }
    const mats = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    if (mats && mats.data) {
      this.matrix.set(mats.data);
    }
    this._trackFps(nowMs);
    return true;
  }

  /**
   * §2: if detection frameTime p95 > 30 ms, auto drop capture to 480×360
   * (the every-2nd-frame step is the perf governor's tier 3).
   * @param {number} nowMs
   */
  _maybeDowngradeCapture(nowMs) {
    if (this._captureDowngraded || this._detectRingN < 20) return;
    const sorted = Array.prototype.slice.call(this._detectRing, 0, this._detectRingN).sort((a, b) => a - b);
    const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
    if (p95 > 30) {
      this._captureDowngraded = true;
      this._tDowngrade = nowMs;
      const track = this.stream?.getVideoTracks?.()[0];
      if (track?.applyConstraints) {
        track.applyConstraints({ width: 480, height: 360 }).catch(() => {});
      }
      this.onStatus('Detect p95 > 30 ms — capture 480×360 (§2)', 'warn');
    }
  }

  /** @param {number} nowMs */
  _trackFps(nowMs) {
    this._fpsAcc += 1;
    if (!this._fpsT0) this._fpsT0 = nowMs;
    if (nowMs - this._fpsT0 > 500) {
      this.fpsDetect = (this._fpsAcc * 1000) / (nowMs - this._fpsT0);
      this._fpsAcc = 0;
      this._fpsT0 = nowMs;
    }
  }

  /**
   * Idle bookkeeping (§2): no face for >0.8 s → caller runs rig.goIdle().
   * @param {number} nowMs
   * @returns {boolean} true when idle should engage
   */
  faceAbsent(nowMs) {
    if (this.hasFace) {
      this.faceAbsentSince = -1;
      return false;
    }
    if (this.faceAbsentSince < 0) this.faceAbsentSince = nowMs;
    return nowMs - this.faceAbsentSince > CONFIG.idleAfterSec * 1000;
  }
}
