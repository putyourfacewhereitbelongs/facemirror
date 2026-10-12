/**
 * perf.js — adaptive quality governor + frame-time stats HUD (spec §8).
 * Ring of last 120 frame times; p95>22 ms tiers down, p95<13 ms for 5 s
 * tiers up. Budget guard: dev warn if heap > 700 MB (§9.4).
 */

/** @type {Record<string, number>} */
export const CONFIG = {
  ringSize: 120,
  tierDownP95Ms: 22,
  tierUpP95Ms: 13,
  tierUpHoldSec: 5,
  heapWarnMb: 700,        // §9.4 budget guard
  heapWarnIntervalSec: 10,
  detectFpsTarget: 30,
  renderFpsTarget: 60,
};

/**
 * Quality tiers, one step each (§8):
 * 0: renderScale 1.25, full detail, every-frame detection
 * 1: renderScale 1.0
 * 2: renderScale 0.85
 * 3: + detection every 2nd frame
 * 4: + disable detail normal map
 * 5: + cap 30 fps (vsync half)
 * @type {ReadonlyArray<{renderScale: number, detectEveryNth: number, detailNormal: boolean, capFps: number}>}
 */
export const TIERS = Object.freeze([
  { renderScale: 1.25, detectEveryNth: 1, detailNormal: true, capFps: 60 },
  { renderScale: 1.0, detectEveryNth: 1, detailNormal: true, capFps: 60 },
  { renderScale: 0.85, detectEveryNth: 1, detailNormal: true, capFps: 60 },
  { renderScale: 0.85, detectEveryNth: 2, detailNormal: true, capFps: 60 },
  { renderScale: 0.85, detectEveryNth: 2, detailNormal: false, capFps: 60 },
  { renderScale: 0.85, detectEveryNth: 2, detailNormal: false, capFps: 30 },
]);

/**
 * Percentile of the ring contents (copies — run at 1 Hz, not per frame).
 * @param {Float32Array} ring
 * @param {number} count
 * @param {number} p 0..1
 * @returns {number}
 */
export function ringPercentile(ring, count, p) {
  if (count <= 0) return 0;
  const tmp = Array.prototype.slice.call(ring, 0, count);
  tmp.sort((a, b) => a - b);
  const idx = Math.min(count - 1, Math.max(0, Math.round((count - 1) * p)));
  return tmp[idx];
}

/**
 * Adaptive governor. Stages: capture / track / solve / render ms (HUD §8).
 */
export class PerfGovernor {
  /**
   * @param {object} [deps]
   * @param {(msg: string) => void} [deps.toast] tier-change toast (§8)
   * @param {(report: string) => void} [deps.log] startup self-check sink
   */
  constructor({ toast, log } = {}) {
    this.toast = toast ?? (() => {});
    this.log = log ?? (() => {});
    this.ring = new Float32Array(CONFIG.ringSize);
    this.ringCount = 0;
    this.ringHead = 0;
    this.tier = 0;
    this.goodFor = 0;       // seconds under the tier-up threshold
    this.lastP95 = 0;
    this.fps = 0;
    this._fpsAcc = 0;
    this._fpsT0 = 0;
    /** Stage timings (ms) for the HUD (§8). */
    this.stage = { capture: 0, track: 0, solve: 0, render: 0 };
    this._heapWarnT = 0;
    this.hudVisible = false;
  }

  /** Startup self-check report (§8) — logged once from main.js. */
  selfCheck() {
    const t = TIERS[this.tier];
    const heap = this.heapMb();
    const lines = [
      'FaceMirror-3D self-check',
      `  quality tier: ${this.tier} (renderScale ${t.renderScale}, detect 1/${t.detectEveryNth}, cap ${t.capFps} fps)`,
      `  heap: ${heap !== null ? heap.toFixed(0) + ' MB' : 'n/a'}`,
      `  frames recorded: ${this.ringCount}`,
      `  p95 frame: ${this.lastP95.toFixed(2)} ms`,
    ];
    const report = lines.join('\n');
    this.log(report);
    return report;
  }

  /** @returns {number|null} JS heap MB when the browser exposes it */
  heapMb() {
    if (typeof performance !== 'undefined' && performance.memory) {
      return performance.memory.usedJSHeapSize / (1024 * 1024);
    }
    return null;
  }

  /**
   * Feed one frame time.
   * @param {number} dtMs total frame time in ms
   */
  frame(dtMs) {
    this.ring[this.ringHead] = dtMs;
    this.ringHead = (this.ringHead + 1) % CONFIG.ringSize;
    if (this.ringCount < CONFIG.ringSize) this.ringCount++;
    this._fpsAcc++;
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!this._fpsT0) this._fpsT0 = now;
    if (now - this._fpsT0 > 500) {
      this.fps = (this._fpsAcc * 1000) / (now - this._fpsT0);
      this._fpsAcc = 0;
      this._fpsT0 = now;
      this._evaluate(now);
    }
  }

  /** @param {number} now */
  _evaluate(now) {
    this.lastP95 = ringPercentile(this.ring, this.ringCount, 0.95);
    const dtSec = 0.5;
    if (this.lastP95 > CONFIG.tierDownP95Ms && this.tier < TIERS.length - 1) {
      this.tier++;
      this.goodFor = 0;
      this.toast(`Quality tier ${this.tier}: ${this._tierLabel()}`);
    } else if (this.lastP95 < CONFIG.tierUpP95Ms) {
      this.goodFor += dtSec;
      if (this.goodFor >= CONFIG.tierUpHoldSec && this.tier > 0) {
        this.tier--;
        this.goodFor = 0;
        this.toast(`Quality tier ${this.tier}: ${this._tierLabel()}`);
      }
    } else {
      this.goodFor = 0;
    }
    // Budget guard (§9.4): warn when the tab heap passes 700 MB.
    const heap = this.heapMb();
    if (heap !== null && heap > CONFIG.heapWarnMb && now - this._heapWarnT > CONFIG.heapWarnIntervalSec * 1000) {
      this._heapWarnT = now;
      this.log(`⚠ heap ${heap.toFixed(0)} MB > ${CONFIG.heapWarnMb} MB (§9.4 budget guard)`);
    }
  }

  /** @returns {string} */
  _tierLabel() {
    const t = TIERS[this.tier];
    return `scale ${t.renderScale}, detect 1/${t.detectEveryNth}${t.detailNormal ? '' : ', no detail normal'}, cap ${t.capFps}`;
  }

  /** @returns {{renderScale: number, detectEveryNth: number, detailNormal: boolean, capFps: number}} */
  settings() {
    return TIERS[this.tier];
  }

  /**
   * HUD payload (§8: fps, ms/stage, RAM, tier).
   * @returns {Record<string, string|number>}
   */
  hudData() {
    const heap = this.heapMb();
    return {
      fps: this.fps.toFixed(1),
      p95: this.lastP95.toFixed(1),
      capture: this.stage.capture.toFixed(1),
      track: this.stage.track.toFixed(1),
      solve: this.stage.solve.toFixed(1),
      render: this.stage.render.toFixed(1),
      heap: heap !== null ? heap.toFixed(0) : 'n/a',
      tier: this.tier,
      tierLabel: this._tierLabel(),
    };
  }
}
