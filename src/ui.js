/**
 * ui.js — one right-side collapsible panel + canvas full-bleed (spec §10).
 * Emotion chips, 52 sliders with 🔒 pins + gain drags, motion toggles,
 * recording, help cheat card, calibration wizard UI, webcam PiP.
 */

import { CHANNELS, GROUPS, GROUP_OF, gainOf } from './rig.js';
import { EMOTION_ORDER, EMOTION_LABELS } from './emotions.js';

/** @type {Record<string, string>} */
export const CONFIG = {
  panelId: 'fm-panel',
  canvasId: 'fm-canvas',
  pipId: 'fm-pip',
  hudId: 'fm-hud',
  readbackId: 'fm-readback',
};

/** Cheat card (§10: 6 lines). */
const CHEAT = [
  'C — calibrate · 1..7 — emotions · 0 — reset all',
  'U — undo drag edit · R — record · M — Mirror Skin',
  'V — webcam PiP / skeleton · Shift+D — perf HUD',
  'Drag face parts to sculpt; 🔒 pins keep your edit',
  'Wheel on the head — dolly · Esc — release drag',
  'Emotion chips long-press = solo/lock',
];

/**
 * UI root. All state flows through the callbacks (ui.js holds no face state).
 */
export class UI {
  /**
   * @param {object} deps
   * @param {HTMLElement} deps.root
   * @param {Record<string, Function>} deps.cb callbacks (see constructor uses)
   */
  constructor({ root, cb }) {
    this.root = root;
    this.cb = cb;
    this.sliders = new Map();
    this.pinButtons = new Map();
    this.gainLabels = new Map();
    this.chips = new Map();
    this.calibration = null;
    this._longPressT = 0;
  }

  /** Build the shell + panel. */
  mount() {
    this.root.innerHTML = `
      <canvas id="${CONFIG.canvasId}"></canvas>
      <div id="fm-privacy" title="Video frames never leave this page">100% local</div>
      <div id="fm-topbar">
        <button id="fm-mirror" class="fm-big" aria-pressed="false">Start webcam motion</button>
        <button id="fm-face-image" class="fm-chip" disabled title="Use the current face image for likeness">Face image…</button>
        <input id="fm-face-file" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" hidden aria-label="Choose the face image to render">
        <button id="fm-skin" class="fm-chip" title="Project your webcam texture onto the head (M)">Mirror Skin</button>
        <button id="fm-full" class="fm-chip" title="Fullscreen">⛶</button>
        <button id="fm-hudbtn" class="fm-chip" title="Shift+D">HUD</button>
        <button id="fm-controls" class="fm-chip" aria-controls="fm-panel" aria-expanded="true">Controls</button>
        <span id="fm-status"></span>
      </div>
      <aside id="${CONFIG.panelId}">
        <div class="fm-section" id="fm-face-source" data-kind="generic" aria-busy="false">
          <h3>Face likeness</h3>
          <div class="fm-face-reference">
            <canvas id="fm-face-preview" width="96" height="112" hidden aria-label="Current face image reference"></canvas>
            <div><div id="fm-face-label">No face image selected</div>
              <div id="fm-face-mode" class="fm-rec-hint">Generic preview — not your likeness</div></div>
          </div>
          <button id="fm-face-animate" class="fm-big fm-face-animate" disabled aria-pressed="false">Animate image with webcam</button>
          <div id="fm-face-driver" class="fm-rec-hint" role="status" aria-live="polite">Image = appearance · Webcam = movement</div>
          <div class="fm-row fm-btns">
            <button id="fm-face-choose" class="fm-chip" disabled>Choose image…</button>
            <button id="fm-face-camera" class="fm-chip" disabled title="Replace the image identity with a webcam snapshot, rather than animate the image">Replace with webcam face</button>
            <button id="fm-face-clear" class="fm-chip" disabled>Clear likeness</button>
          </div>
          <div id="fm-face-message" class="fm-rec-hint" role="status" aria-live="polite">Drop a JPEG, PNG or WebP onto the head. Use one clear, front-facing face.</div>
          <div class="fm-rec-hint">Fits this image's proportions, skin and eyes locally. Unseen profile, ears and full hair are approximated. No uploads or saved photos.</div>
        </div>
        <div class="fm-section">
          <h3>Emotions</h3>
          <div id="fm-chips">
            <button class="fm-emo" data-e="none">None</button>
            ${EMOTION_ORDER.map((e) => `<button class="fm-emo" data-e="${e}">${EMOTION_LABELS[e]}</button>`).join('')}
          </div>
          <label class="fm-row"><span>Intensity</span>
            <input type="range" id="fm-intensity" min="0" max="1" step="0.01" value="1"></label>
          <div id="${CONFIG.readbackId}">You: —</div>
        </div>
        <div class="fm-section">
          <h3>Face controls</h3>
          <label class="fm-row"><span>Asymmetry</span>
            <input type="range" id="fm-asym" min="0" max="1" step="0.01" value="0.25"></label>
          <label class="fm-row"><span>Camera strength</span>
            <input type="range" id="fm-camw" min="0" max="1" step="0.01" value="1"></label>
          <div id="fm-groups"></div>
        </div>
        <div class="fm-section">
          <h3>Motion</h3>
          <label class="fm-row"><span>Blink rate</span>
            <input type="range" id="fm-blink" min="0.1" max="0.6" step="0.01" value="0.3"></label>
          <label class="fm-row"><span>Gain global</span>
            <input type="range" id="fm-gain" min="0.2" max="1.8" step="0.01" value="1"></label>
          <label class="fm-row"><span>Responsiveness</span>
            <input type="range" id="fm-resp" min="0.2" max="5" step="0.01" value="1"></label>
          <label class="fm-row"><span>Head lag</span>
            <input type="range" id="fm-lag" min="0" max="0.15" step="0.005" value="0.05"></label>
          <div class="fm-toggles">
            ${['saccades', 'jitter', 'vor', 'pupil'].map((k) => `
            <label class="fm-toggle"><input type="checkbox" id="fm-eye-${k}" checked>
              <span>${k === 'vor' ? 'VOR' : k[0].toUpperCase() + k.slice(1)}</span></label>`).join('')}
            <label class="fm-toggle" title="Drive jawOpen + visemes from microphone energy (default off; audio never leaves the page)">
              <input type="checkbox" id="fm-mic"><span>Mic jaw (β)</span></label>
          </div>
          <div class="fm-rec-hint">Mic is opt-in and 100% local, like the camera.</div>
        </div>
        <div class="fm-section">
          <h3>Recording</h3>
          <div class="fm-row fm-btns">
            <button id="fm-rec" class="fm-chip">⏺ rec</button>
            <button id="fm-play" class="fm-chip">▶ play</button>
            <button id="fm-exjson" class="fm-chip">.json</button>
            <button id="fm-exanim" class="fm-chip">.anim.glb</button>
            <button id="fm-expose" class="fm-chip">head .glb</button>
          </div>
          <div class="fm-rec-hint">ring 60 s visible · up to 300 s captured</div>
        </div>
        <div class="fm-section">
          <h3>Help</h3>
          <div class="fm-cheat">${CHEAT.map((l) => `<div>${l}</div>`).join('')}</div>
          <button id="fm-cal" class="fm-chip">Calibrate / replay wizard (C)</button>
          <div class="fm-rec-hint">Head scan: <a href="https://www.triplegangers.com/" target="_blank" rel="noopener">Lee Perry-Smith / Infinite Realities</a> · <a href="https://creativecommons.org/licenses/by/3.0/" target="_blank" rel="noopener">CC BY 3.0</a> · adapted for FaceMirror</div>
        </div>
      </aside>
      <div id="fm-wizard" class="fm-hidden">
        <div class="fm-wizard-card">
          <h2 id="fm-wiz-title">Calibration</h2>
          <div id="fm-wiz-prompt"></div>
          <div class="fm-wiz-bar"><div id="fm-wiz-fill"></div></div>
          <canvas id="fm-wiz-bars" width="520" height="70"></canvas>
          <div class="fm-row fm-btns">
            <button id="fm-wiz-hold" class="fm-big">Hold SPACE</button>
            <button id="fm-wiz-cancel" class="fm-chip">Cancel</button>
          </div>
          <div id="fm-wiz-note">5 quick steps · doubles as the live debug view</div>
        </div>
      </div>
      <div id="${CONFIG.pipId}" class="fm-hidden">
        <video id="fm-video" playsinline muted></video>
        <canvas id="fm-skel"></canvas>
      </div>
      <div id="${CONFIG.hudId}" class="fm-hidden"></div>
    `;
    if (window.innerWidth < 960) {
      this.root.querySelector('#fm-panel').classList.add('fm-collapsed');
      this.root.querySelector('#fm-controls').setAttribute('aria-expanded', 'false');
    }
    this._wire();
    this._buildSliders();
    this.installKeyboard();
  }

  _buildSliders() {
    const host = this.root.querySelector('#fm-groups');
    const uiGroups = [
      ['Mouth', GROUPS.mouth],
      ['Eyes', GROUPS.eyes],
      ['Brows', GROUPS.brows],
      ['Cheeks', [...GROUPS.cheeks, ...GROUPS.nose]],
      ['Jaw', GROUPS.jaw],
      ['Tongue', GROUPS.tongue],
    ];
    for (const [label, names] of uiGroups) {
      const details = document.createElement('details');
      details.className = 'fm-group';
      details.open = label === 'Mouth';
      const summary = document.createElement('summary');
      summary.textContent = label + ' · dbl-click resets group gains';
      summary.addEventListener('dblclick', () => {
        for (const n of names) {
          this.cb.onGain?.(n, 1);
          const gl = this.gainLabels.get(n);
          if (gl) gl.textContent = '×1.00';
        }
      });
      details.appendChild(summary);
      for (const name of names) {
        const row = document.createElement('div');
        row.className = 'fm-slider-row';
        row.innerHTML = `
          <span class="fm-sname" title="${name}">${name.replace(/([A-Z])/g, ' $1').trim()}</span>
          <input type="range" min="0" max="1" step="0.01" value="0">
          <button class="fm-pin" title="Pin (keep manual value)">🔓</button>
          <span class="fm-gain" title="drag to set channel gain">×${gainOf(name).toFixed(2)}</span>
        `;
        const slider = /** @type {HTMLInputElement} */ (row.querySelector('input'));
        const pin = /** @type {HTMLButtonElement} */ (row.querySelector('.fm-pin'));
        const gain = /** @type {HTMLElement} */ (row.querySelector('.fm-gain'));
        slider.addEventListener('input', () => this.cb.onSlider?.(name, parseFloat(slider.value)));
        pin.addEventListener('click', () => {
          const on = pin.textContent !== '🔒';
          pin.textContent = on ? '🔒' : '🔓';
          pin.classList.toggle('fm-pinned', on);
          this.cb.onPin?.(name, on);
        });
        // §10 "gain drag": drag the ×label vertically.
        let gy = 0;
        let gStart = 0;
        gain.addEventListener('pointerdown', (e) => {
          gy = e.clientY;
          gStart = parseFloat(gain.textContent.replace('×', ''));
          gain.setPointerCapture(e.pointerId);
        });
        gain.addEventListener('pointermove', (e) => {
          if (gy === 0) return;
          const v = Math.min(Math.max(gStart + (gy - e.clientY) * 0.01, 0), 3);
          gain.textContent = '×' + v.toFixed(2);
          this.cb.onGain?.(name, v);
        });
        gain.addEventListener('pointerup', () => { gy = 0; });
        this.sliders.set(name, slider);
        this.pinButtons.set(name, pin);
        this.gainLabels.set(name, gain);
        details.appendChild(row);
      }
      host.appendChild(details);
    }
  }

  _wire() {
    const $ = (id) => this.root.querySelector(id);
    $('#fm-mirror').addEventListener('click', () => this.cb.onMirrorToggle?.());
    for (const id of ['#fm-face-image', '#fm-face-choose']) {
      $(id).addEventListener('click', () => $('#fm-face-file').click());
    }
    $('#fm-face-file').addEventListener('change', (event) => {
      const file = event.target.files?.[0];
      if (file) this.cb.onFaceImage?.(file);
      event.target.value = ''; // selecting the same image again is a new source
    });
    $('#fm-face-animate').addEventListener('click', () => this.cb.onAnimateImage?.());
    $('#fm-face-camera').addEventListener('click', () => this.cb.onCameraFace?.());
    $('#fm-face-clear').addEventListener('click', () => this.cb.onClearFace?.());
    $('#fm-skin').addEventListener('click', () => this.cb.onMirrorSkin?.());
    $('#fm-full').addEventListener('click', () => this.cb.onFullscreen?.());
    $('#fm-hudbtn').addEventListener('click', () => this.cb.onHud?.());
    $('#fm-controls').addEventListener('click', () => {
      const closed = $('#fm-panel').classList.toggle('fm-collapsed');
      $('#fm-controls').setAttribute('aria-expanded', String(!closed));
      this.cb.onLayout?.();
    });
    for (const chip of this.root.querySelectorAll('.fm-emo')) {
      const el = /** @type {HTMLElement} */ (chip);
      const id = el.dataset.e;
      el.addEventListener('click', () => this.cb.onEmotion?.(id));
      // Long-press = solo/lock (§10).
      el.addEventListener('pointerdown', () => {
        this._longPressT = window.setTimeout(() => {
          this.cb.onEmotionSolo?.(id);
          el.classList.add('fm-solo');
        }, 550);
      });
      const clear = () => {
        window.clearTimeout(this._longPressT);
        for (const c of this.root.querySelectorAll('.fm-emo')) {
          if (c !== el || !this.cb.isSolo?.()) c.classList.remove('fm-solo');
        }
      };
      el.addEventListener('pointerup', clear);
      el.addEventListener('pointerleave', clear);
      this.chips.set(id, el);
    }
    $('#fm-intensity').addEventListener('input', (e) => this.cb.onIntensity?.(parseFloat(e.target.value)));
    $('#fm-asym').addEventListener('input', (e) => this.cb.onMaster?.('asym', parseFloat(e.target.value)));
    $('#fm-camw').addEventListener('input', (e) => this.cb.onMaster?.('camWeight', parseFloat(e.target.value)));
    $('#fm-blink').addEventListener('input', (e) => this.cb.onMotion?.('blinkRate', parseFloat(e.target.value)));
    $('#fm-gain').addEventListener('input', (e) => this.cb.onMotion?.('globalGain', parseFloat(e.target.value)));
    $('#fm-resp').addEventListener('input', (e) => this.cb.onMotion?.('responsiveness', parseFloat(e.target.value)));
    $('#fm-lag').addEventListener('input', (e) => this.cb.onMotion?.('headLag', parseFloat(e.target.value)));
    for (const k of ['saccades', 'jitter', 'vor', 'pupil']) {
      $(`#fm-eye-${k}`).addEventListener('change', (e) => this.cb.onEyeToggle?.(k, e.target.checked));
    }
    $('#fm-mic').addEventListener('change', (e) => this.cb.onMic?.(e.target.checked));
    $('#fm-rec').addEventListener('click', () => this.cb.onRecord?.());
    $('#fm-play').addEventListener('click', () => this.cb.onPlay?.());
    $('#fm-exjson').addEventListener('click', () => this.cb.onExport?.('json'));
    $('#fm-exanim').addEventListener('click', () => this.cb.onExport?.('anim'));
    $('#fm-expose').addEventListener('click', () => this.cb.onExport?.('pose'));
    $('#fm-cal').addEventListener('click', () => this.cb.onCalibrate?.());
    $('#fm-wiz-hold').addEventListener('pointerdown', () => this.cb.onWizardHold?.(true));
    $('#fm-wiz-hold').addEventListener('pointerup', () => this.cb.onWizardHold?.(false));
    $('#fm-wiz-cancel').addEventListener('click', () => this.cb.onWizardCancel?.());
  }

  /**
   * Keyboard map (§10). Handlers live in main.js via callbacks.
   */
  installKeyboard() {
    window.addEventListener('keydown', (e) => {
      if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const k = e.key.toLowerCase();
      if (k === 'c') this.cb.onCalibrate?.();
      else if (k >= '1' && k <= '7') this.cb.onEmotion?.(EMOTION_ORDER[parseInt(k, 10) - 1]);
      else if (k === '0') this.cb.onReset?.();
      else if (k === 'u') this.cb.onUndo?.();
      else if (k === 'r') this.cb.onRecord?.();
      else if (k === 'm') this.cb.onMirrorSkin?.();
      else if (k === 'v') this.cb.onPip?.();
      else if (k === 'd' && e.shiftKey) this.cb.onHud?.();
      else if (k === 'escape') this.cb.onEscape?.();
      else if (k === ' ') {
        e.preventDefault();
        this.cb.onWizardHold?.(e.type === 'keydown');
      }
    });
    window.addEventListener('keyup', (e) => {
      if (e.key === ' ') this.cb.onWizardHold?.(false);
    });
  }

  /**
   * @param {string} msg
   * @param {'ok'|'warn'|'error'|'debug'} [kind]
   */
  setStatus(msg, kind = 'ok') {
    const el = this.root.querySelector('#fm-status');
    if (el) {
      el.textContent = msg;
      el.className = 'fm-status-' + kind;
    }
  }

  /**
   * @param {string} msg transient toast (tier changes §8)
   */
  toast(msg) {
    const t = document.createElement('div');
    t.className = 'fm-toast';
    t.textContent = msg;
    this.root.appendChild(t);
    setTimeout(() => t.remove(), 3200);
    this.setStatus(msg, 'warn');
  }

  /**
   * Read-back row (§10 "You: joy 62%").
   * @param {string} label
   * @param {number} score 0..1
   */
  setReadback(label, score) {
    const el = this.root.querySelector('#' + CONFIG.readbackId);
    if (el) {
      el.textContent = label === 'none' ? 'You: —' : `You: ${label} ${Math.round(score * 100)}%`;
    }
  }

  /**
   * Highlight the active emotion chip.
   * @param {string} id
   */
  setActiveEmotion(id) {
    for (const [k, el] of this.chips) {
      el.classList.toggle('fm-active', k === id);
    }
  }

  /**
   * Reflect a slider value (playback, undo, reset).
   * @param {string} name
   * @param {number} v
   */
  setSlider(name, v) {
    const s = this.sliders.get(name);
    if (s) s.value = String(v);
  }

  /**
   * @param {string} name
   * @param {boolean} on
   */
  setPin(name, on) {
    const p = this.pinButtons.get(name);
    if (p) {
      p.textContent = on ? '🔒' : '🔓';
      p.classList.toggle('fm-pinned', on);
    }
  }

  /** Identity and webcam driver are separate; no video pixels replace the photo. */
  setCameraMotion(state, imageReady, imageActive = imageReady) {
    const active = ['starting', 'waiting', 'tracking'].includes(state);
    const animate = this.root.querySelector('#fm-face-animate');
    const mirror = this.root.querySelector('#fm-mirror');
    animate.disabled = !active && (!imageReady || this.root.querySelector('#fm-face-source').getAttribute('aria-busy') === 'true');
    const activeLabel = state === 'starting' ? 'Cancel webcam start' : active ? 'Stop webcam motion' : null;
    animate.textContent = activeLabel ?? 'Animate image with webcam';
    mirror.textContent = activeLabel ?? (imageActive ? 'Animate image with webcam' : 'Start webcam motion');
    for (const button of [animate, mirror]) {
      button.setAttribute('aria-pressed', String(active));
      button.classList.toggle('fm-active', active);
    }
    const driver = this.root.querySelector('#fm-face-driver');
    driver.dataset.state = state;
    driver.textContent = state === 'tracking' ? 'Motion: live webcam · Turn, smile, talk and blink'
      : state === 'starting' ? 'Starting camera… the image stays the face source'
        : state === 'waiting' ? 'Webcam on · Looking for your face — image kept'
          : state === 'error' ? 'Camera unavailable · Image kept — allow access and retry'
            : 'Image = appearance · Webcam = movement';
  }

  setFaceAvailable(available) {
    for (const id of ['fm-face-image', 'fm-face-choose', 'fm-face-camera']) this.root.querySelector('#' + id).disabled = !available;
  }

  setFaceBusy(busy) {
    this.root.querySelector('#fm-face-source').setAttribute('aria-busy', String(busy));
    if (busy) this.setFaceMessage('Fitting the current face image…');
  }

  setFaceMessage(message, error = false) {
    const el = this.root.querySelector('#fm-face-message');
    el.textContent = message;
    el.classList.toggle('fm-face-error', error);
  }

  /** Draw immediately; the controller releases the full-resolution image. */
  setFaceReference({ canvas, landmarks, label, kind }) {
    const preview = this.root.querySelector('#fm-face-preview');
    const source = this.root.querySelector('#fm-face-source');
    source.dataset.kind = kind;
    this.root.querySelector('#fm-face-label').textContent = kind === 'generic' ? 'No face image selected' : label;
    this.root.querySelector('#fm-face-clear').disabled = kind === 'generic';
    preview.hidden = !canvas;
    const ctx = preview.getContext('2d');
    ctx.clearRect(0, 0, preview.width, preview.height);
    if (!canvas) return;
    const left = Math.min(landmarks[234].x, landmarks[454].x);
    const right = Math.max(landmarks[234].x, landmarks[454].x);
    const top = landmarks[10].y, bottom = landmarks[152].y;
    const h = (bottom - top) * canvas.height * 1.4;
    const w = Math.max((right - left) * canvas.width * 1.25, h * preview.width / preview.height);
    const cropH = w * preview.height / preview.width;
    const cx = (left + right) / 2 * canvas.width;
    const cy = (top + bottom) / 2 * canvas.height - (bottom - top) * canvas.height * 0.10;
    ctx.fillStyle = '#1a202c'; ctx.fillRect(0, 0, preview.width, preview.height);
    ctx.drawImage(canvas, cx - w / 2, cy - cropH / 2, w, cropH, 0, 0, preview.width, preview.height);
  }

  setFaceMode(message, active) {
    this.root.querySelector('#fm-face-mode').textContent = message;
    this.root.querySelector('#fm-face-source').dataset.active = String(active);
  }

  /** @param {boolean} on */
  setMirrorSkin(on) {
    const el = this.root.querySelector('#fm-skin');
    if (el) el.classList.toggle('fm-active', on);
  }

  /** Webcam texture projection is a fallback feature, not a scan overlay. */
  setMirrorSkinAvailable(available, anatomical = false) {
    const el = this.root.querySelector('#fm-skin');
    if (!el) return;
    el.disabled = !available;
    el.title = available ? (anatomical ? 'Switch to the webcam-textured procedural head (M)'
      : 'Project your webcam texture onto the procedural head (M)')
      : 'Custom heads keep their authored skin. Webcam motion still works.';
    if (!available) el.classList.remove('fm-active');
  }

  /** @param {boolean} on */
  setHudVisible(on) {
    const el = this.root.querySelector('#' + CONFIG.hudId);
    if (el) el.classList.toggle('fm-hidden', !on);
  }

  /**
   * @param {Record<string, string|number>} data
   */
  updateHud(data) {
    const el = this.root.querySelector('#' + CONFIG.hudId);
    if (!el) return;
    el.innerHTML = `
      <b>Shift+D HUD</b><br>
      fps ${data.fps} · p95 ${data.p95} ms · tier ${data.tier}<br>
      capture ${data.capture} · track ${data.track} · solve ${data.solve} · render ${data.render} ms<br>
      heap ${data.heap} MB<br>
      <span class="fm-hud-tier">${data.tierLabel}</span>
    `;
  }

  /** @returns {HTMLVideoElement} */
  videoEl() {
    return /** @type {HTMLVideoElement} */ (this.root.querySelector('#fm-video'));
  }

  /** @param {boolean} on */
  setPipVisible(on) {
    const el = this.root.querySelector('#' + CONFIG.pipId);
    if (el) el.classList.toggle('fm-hidden', !on);
  }

  /**
   * Draw the landmark skeleton on the PiP (toggle V).
   * @param {Array<{x: number, y: number}>} landmarks
   * @param {boolean} show
   */
  drawSkeleton(landmarks, show) {
    const canvas = /** @type {HTMLCanvasElement} */ (this.root.querySelector('#fm-skel'));
    if (!canvas) return;
    const g = canvas.getContext('2d');
    const video = this.videoEl();
    if (!g || !video) return;
    canvas.width = video.clientWidth || 160;
    canvas.height = video.clientHeight || 120;
    g.clearRect(0, 0, canvas.width, canvas.height);
    if (!show || !landmarks.length) return;
    g.fillStyle = 'rgba(80, 220, 160, 0.85)';
    for (let i = 0; i < landmarks.length; i += 4) {
      // Video is mirrored (trap §11.9): mirror x for the overlay too.
      const x = (1 - landmarks[i].x) * canvas.width;
      const y = landmarks[i].y * canvas.height;
      g.fillRect(x - 1, y - 1, 2, 2);
    }
  }

  // --- Calibration wizard (§3/§4): 5-step UI over the phase machine -----

  /**
   * @param {{onHold: Function, onCancel: Function}} wizard callbacks
   */
  openWizard() {
    const el = this.root.querySelector('#fm-wizard');
    if (el) el.classList.remove('fm-hidden');
  }

  /** Close wizard overlay. */
  closeWizard() {
    const el = this.root.querySelector('#fm-wizard');
    if (el) el.classList.add('fm-hidden');
  }

  /**
   * Update wizard prompt/progress; draw live bars (debug view §4).
   * @param {string} prompt
   * @param {number} progress 0..1
   * @param {Float32Array} weights52 live weights for the bars
   * @param {string} [stepLabel]
   */
  updateWizard(prompt, progress, weights52, stepLabel = '') {
    const p = this.root.querySelector('#fm-wiz-prompt');
    const fill = this.root.querySelector('#fm-wiz-fill');
    const title = this.root.querySelector('#fm-wiz-title');
    if (p) p.textContent = prompt;
    if (fill) fill.style.width = `${Math.round(progress * 100)}%`;
    if (title) title.textContent = 'Calibration — ' + stepLabel;
    const bars = /** @type {HTMLCanvasElement} */ (this.root.querySelector('#fm-wiz-bars'));
    if (bars && weights52) {
      const g = bars.getContext('2d');
      if (g) {
        g.clearRect(0, 0, bars.width, bars.height);
        const w = bars.width / 52;
        for (let i = 0; i < 52; i++) {
          const v = Math.min(Math.max(weights52[i], 0), 1);
          const h = v * (bars.height - 6);
          g.fillStyle = v > 0.65 ? '#f7b267' : v > 0.3 ? '#8fd3c1' : '#40566b';
          g.fillRect(i * w + 1, bars.height - h - 2, w - 2, h);
        }
      }
    }
  }
}
