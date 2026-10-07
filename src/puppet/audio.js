/**
 * Audio-driven speech: acoustic viseme estimation.
 *
 * This is the browser-side analogue of the audio stage in Wav2Lip / SadTalker /
 * VideoReTalking. Those systems regress mouth shape with a trained network; here
 * the same *pipeline shape* is driven by signal processing instead, which is
 * honest about what runs locally and still gives phoneme-aware mouth motion:
 *
 *   audio -> short-time spectrum -> formant band energies (F0/F1/F2/F3)
 *         -> voicing + fricative + burst/closure transients
 *         -> viseme parameters (jaw, spread, pucker, tongue tip/back/out,
 *            bilabial closure, teeth visibility)
 *         -> prosody (F0 contour + energy accents) -> subtle head motion
 *
 * Vowel acoustics used for the mapping:
 *   /a/  high F1 (~700-800 Hz)              -> jaw open, tongue low and wide
 *   /i/  low F1, high F2 (~2300 Hz)         -> lips spread, teeth visible
 *   /u/  low F1, low F2 (~800 Hz)           -> lips rounded/pursed
 *   /l th t d/  voiced or fricative with a low-mid F2 -> tongue tip raised
 *   /k g/       velar burst after a closure -> tongue back raised
 *   /p b m/     closure (energy collapse)   -> lips pressed together
 *   /f v s z/   high-frequency frication    -> narrow opening, teeth near
 *
 * The pure functions below take plain arrays so they are unit tested; only the
 * WebAudio plumbing touches the DOM.
 */

import { clamp } from './math3d.js';
import { Scalar, Gate, smoothK } from './temporal.js';

export const BANDS = [
  { name: 'F0 voicing', lo: 60, hi: 280 },
  { name: 'F1 openness', lo: 280, hi: 900 },
  { name: 'F2 front', lo: 900, hi: 1800 },
  { name: 'F2 high / spread', lo: 1800, hi: 3000 },
  { name: 'F3 fricative', lo: 3000, hi: 6500 },
  { name: 'air', lo: 6500, hi: 11000 },
];

/** Mean linear power per band from a float frequency spectrum in dB. */
export function bandEnergies(freqDb, sampleRate, fftSize, bands = BANDS) {
  const n = freqDb.length, binHz = sampleRate / fftSize;
  const out = new Array(bands.length).fill(0);
  let total = 0;
  for (let b = 0; b < bands.length; b++) {
    const i0 = Math.max(0, Math.floor(bands[b].lo / binHz)), i1 = Math.min(n - 1, Math.ceil(bands[b].hi / binHz));
    let s = 0, k = 0;
    for (let i = i0; i <= i1; i++) { const p = Math.pow(10, (freqDb[i] || -120) / 20); s += p * p; k++; }
    out[b] = k ? Math.sqrt(s / k) : 0;
    total += out[b];
  }
  return { bands: out, total };
}

/** Spectral centroid (Hz) - a cheap brightness measure. */
export function spectralCentroid(freqDb, sampleRate, fftSize) {
  const n = freqDb.length, binHz = sampleRate / fftSize;
  let num = 0, den = 0;
  for (let i = 1; i < n; i++) {
    const p = Math.pow(10, (freqDb[i] || -120) / 20);
    num += i * binHz * p; den += p;
  }
  return den > 1e-9 ? num / den : 0;
}

/**
 * Normalised-autocorrelation F0 estimate on a decimated frame. Only searches the
 * plausible speech range (70-400 Hz) so it stays cheap enough for 60 fps.
 */
export function estimateF0(time, sampleRate, { decimate = 4, minHz = 70, maxHz = 400 } = {}) {
  const n = Math.floor(time.length / decimate);
  if (n < 64) return 0;
  const x = new Float32Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) { x[i] = time[i * decimate]; mean += x[i]; }
  mean /= n;
  for (let i = 0; i < n; i++) x[i] -= mean;
  let e0 = 0;
  for (let i = 0; i < n; i++) e0 += x[i] * x[i];
  if (e0 < 1e-4) return 0;
  const sr = sampleRate / decimate;
  const lagMin = Math.max(2, Math.floor(sr / maxHz)), lagMax = Math.min(n - 2, Math.ceil(sr / minHz));
  let best = 0, bestVal = 0;
  for (let lag = lagMin; lag <= lagMax; lag++) {
    let s = 0, e1 = 0, e2 = 0;
    for (let i = 0; i + lag < n; i++) { s += x[i] * x[i + lag]; e1 += x[i] * x[i]; e2 += x[i + lag] * x[i + lag]; }
    const v = s / (Math.sqrt(e1 * e2) || 1);
    if (v > bestVal) { bestVal = v; best = lag; }
  }
  return bestVal > 0.45 && best ? sr / best : 0;
}

export const VISEMES = ['silence', 'closure /p b m/', 'labiodental /f v/', 'sibilant /s z/', 'alveolar /t d l/',
  'velar /k g/', 'open /a/', 'spread /i e/', 'rounded /o u/', 'glide /r w/'];

/**
 * Map band energies + transients onto viseme parameters. `st` is the persistent
 * state (smoothers); everything is low-passed so the mouth never strobes.
 */
export function bandsToViseme(E, st, dt, opts = {}) {
  const { bands, total } = E;
  const gain = opts.gain ?? 1;
  // Saturating loudness gate. Ratios (f1/f2h/fricative) carry the phonetic
  // information, so level is deliberately clamped at 1: above that it only says
  // "there is voice", and letting it run to 1.6 would open the jaw on every
  // vowel including the closed ones.
  const level = clamp(total * (opts.levelScale ?? 26), 0, 1);
  const sum = bands.reduce((a, b) => a + b, 0) || 1e-9;
  const r = bands.map(b => b / sum);
  const voiced = clamp((r[0] * 3.4 - 0.16) * (level > 0.06 ? 1 : 0), 0, 1);
  const fricative = clamp(r[4] * 3.1 + r[5] * 2.2 - 0.22, 0, 1);
  const f1 = clamp(r[1] * 2.5, 0, 1), f2h = clamp(r[3] * 3.6, 0, 1), f2l = clamp(r[2] * 2.8, 0, 1);

  // --- transients: closures and bursts, the consonant cues ---
  const prev = st.level === undefined ? level : st.level;
  const fall = clamp((prev - level) / Math.max(dt, 0.004) / 9, 0, 1);   // fast energy collapse
  const rise = clamp((level - prev) / Math.max(dt, 0.004) / 9, 0, 1);   // burst onset
  st.level = level;
  st.recentVoiced = Math.max((st.recentVoiced || 0) - dt * 5, voiced > 0.35 || level > 0.22 ? 1 : (st.recentVoiced || 0));
  st.burst = Math.max((st.burst || 0) * Math.exp(-dt / 0.09), rise * (st.recentVoiced > 0.2 ? 0 : 1) * clamp(fricative + 0.25, 0, 1));
  const closure = clamp((1 - level * 5.5) * st.recentVoiced, 0, 1);

  // --- vowel shaping ---
  // Jaw follows F1 (the openness formant) quadratically: /a/ ~1.0, /i/ ~0.15.
  const jawT = clamp(level * (0.06 + 1.15 * f1 * f1) * gain, 0, 1.25);
  const spreadT = clamp(level * (1.5 * f2h - 0.35 * f1) * gain, 0, 1);
  const puckerT = clamp(level * voiced * (1.25 - 1.5 * f2h - 0.6 * f1) * gain, 0, 1);
  const stretchT = clamp(level * (fricative * 0.7 + f2h * 0.35) * gain, 0, 1);
  const tipT = clamp(voiced * (0.85 - 0.75 * f2h) * f2l * 1.5 + fricative * 0.45, 0, 1);
  const backT = clamp(st.burst * 1.5 + voiced * clamp(0.5 - f2h * 1.2, 0, 1) * 0.35, 0, 1);
  // Protrusion is rare: it needs a wide open jaw AND a dark (low F2) timbre,
  // which is the /th/ - tongue-out family. Loud vowels alone must not stick it out.
  const outT = clamp(opts.forceTongue ? 1 : (level * (0.55 + 0.35 * f1) * (1 - f2h) - 0.62) * gain, 0, 1);

  st.jaw = st.jaw || new Scalar(0, 0.035); st.spread = st.spread || new Scalar(0, 0.05);
  st.pucker = st.pucker || new Scalar(0, 0.06); st.stretch = st.stretch || new Scalar(0, 0.06);
  st.tip = st.tip || new Scalar(0, 0.055); st.back = st.back || new Scalar(0, 0.07);
  st.out = st.out || new Scalar(0, 0.09); st.close = st.close || new Scalar(0, 0.03);
  st.teeth = st.teeth || new Scalar(0, 0.07); st.brow = st.brow || new Scalar(0, 0.14);

  const jaw = st.jaw.update(jawT * (1 - closure * 0.85), dt);
  const spread = st.spread.update(spreadT * (1 - closure), dt);
  const pucker = st.pucker.update(puckerT * (1 - closure * 0.8), dt);
  const stretch = st.stretch.update(stretchT, dt);
  const tip = st.tip.update(tipT * (1 - closure * 0.5), dt);
  const back = st.back.update(backT, dt);
  const out = st.out.update(outT, dt);
  const close = st.close.update(closure, dt);
  const teeth = st.teeth.update(clamp(jaw * 0.75 + spread * 0.4 - close * 0.9, 0, 1), dt);
  const brow = st.brow.update(clamp(level * 0.35 + f2h * 0.15, 0, 0.6), dt);

  // --- prosody: pitch contour + accents drive subtle head motion ---
  const f0 = st.f0 || 0;
  if (f0 > 0) {
    st.f0Base = st.f0Base ? st.f0Base + (f0 - st.f0Base) * smoothK(1.4, dt) : f0;
    st.f0Delta = (f0 - st.f0Base) / (st.f0Base || 1);
  } else { st.f0Delta = (st.f0Delta || 0) * Math.exp(-dt / 0.3); }
  st.accent = Math.max((st.accent || 0) * Math.exp(-dt / 0.22), clamp(level * 1.3 - 0.3, 0, 1));

  const name = pickViseme({ level, closure, fricative, voiced, jaw, spread, pucker, tip, back, burst: st.burst });
  return {
    level, voiced, fricative, jaw, spread, pucker, stretch, tongueTip: tip, tongueBack: back,
    tongueOut: out, closure: close, teeth, brow, f0, f0Delta: st.f0Delta || 0, accent: st.accent || 0, name, bands: r,
  };
}

export function pickViseme(s) {
  if (s.level < 0.045 && s.burst < 0.05) return VISEMES[0];
  if (s.closure > 0.55) return VISEMES[1];
  if (s.burst > 0.3 && s.fricative < 0.4) return VISEMES[5];
  if (s.fricative > 0.55) return s.level > 0.3 ? VISEMES[3] : VISEMES[2];
  if (s.tip > 0.5) return VISEMES[4];
  if (s.spread > 0.34 && s.jaw < 0.5) return VISEMES[7];
  if (s.pucker > 0.3) return VISEMES[8];
  if (s.jaw > 0.42) return VISEMES[6];
  if (s.jaw > 0.14 || s.level > 0.1) return VISEMES[9];
  return VISEMES[0];
}

/** WebAudio plumbing: microphone or an audio file -> analyser -> visemes. */
export class AudioEngine {
  constructor() {
    this.ctx = null; this.analyser = null; this.source = null; this.stream = null;
    this.freq = null; this.time = null; this.state = {}; this.running = false; this.mode = 'off';
    this.sampleRate = 48000; this.fftSize = 2048;
    this.gate = { teeth: new Gate(0.12, 0.5, 0.06, 0.18) };
  }
  async _ensureCtx() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error('Web Audio is not available in this browser');
      this.ctx = new AC();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = this.fftSize;
      this.analyser.smoothingTimeConstant = 0.42;
      this.analyser.minDecibels = -95; this.analyser.maxDecibels = -22;
      this.freq = new Float32Array(this.analyser.frequencyBinCount);
      this.time = new Float32Array(this.analyser.fftSize);
      this.sampleRate = this.ctx.sampleRate;
    }
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    return this.ctx;
  }
  async useMicrophone() {
    await this._ensureCtx();
    this.stopSource();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.source.connect(this.analyser);
    this.running = true; this.mode = 'mic';
    return this.stream;
  }
  async useElement(el) {
    await this._ensureCtx();
    this.stopSource();
    this.source = this.ctx.createMediaElementSource(el);
    this.source.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this.running = true; this.mode = 'file';
    await el.play();
  }
  stopSource() {
    try { this.source && this.source.disconnect(); } catch (e) { /* already gone */ }
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = null; this.source = null;
  }
  stop() { this.stopSource(); this.running = false; this.mode = 'off'; this.state = {}; }
  /** One analysis step; returns the viseme parameter block. */
  update(dt, opts) {
    if (!this.running || !this.analyser) return null;
    this.analyser.getFloatFrequencyData(this.freq);
    this.analyser.getFloatTimeDomainData(this.time);
    const E = bandEnergies(this.freq, this.sampleRate, this.fftSize);
    E.centroid = spectralCentroid(this.freq, this.sampleRate, this.fftSize);
    this.state.f0 = estimateF0(this.time, this.sampleRate);
    return bandsToViseme(E, this.state, dt, opts);
  }
  /** The raw audio track, so recordings can include the driving speech. */
  audioTrack() { return this.stream ? this.stream.getAudioTracks()[0] : null; }
}
