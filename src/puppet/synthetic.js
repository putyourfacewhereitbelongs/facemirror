/**
 * Synthetic 478-point face generator.
 *
 * Two jobs: (1) a deterministic fixture for the geometry tests, and (2) the
 * payload of the page's built-in pipeline self-test, which exercises model
 * building, projection, culling and mouth fitting without a camera. The layout
 * follows the MediaPipe topology closely enough that every index the rig relies
 * on lands where the real detector would put it.
 */

const TAU = Math.PI * 2;
/** deterministic hash noise so tests are reproducible */
const h1 = i => { const x = Math.sin(i * 127.1 + 11.7) * 43758.5453; return x - Math.floor(x); };

import { OVAL, EYE, EYM, BROW, LIP, GROUPS, NOSE_RIDGE, NOSE_WING, IRIS, CHEEK, CHIN, FOREHEAD, TEMPLE } from './landmarks.js';

export function syntheticFace({ W = 720, H = 900, cx = 360, cy = 400, fh = 470, yaw = 0, zScale = 1 } = {}) {
  const fw = fh * 0.74;
  const V = new Array(478);
  const put = (i, x, y, z = 0) => { V[i] = [x, y, z * zScale]; };
  const eyeY = cy - fh * 0.04;

  // face oval: 36 points on an ellipse, top -> image right -> chin -> image left
  const ox = fw * 0.5, oy = fh * 0.52;
  OVAL.forEach((idx, k) => {
    const a = -Math.PI / 2 + (k * TAU) / OVAL.length;
    // a real jaw is narrower than the cheekbones: squash the lower half
    const jaw = a > 0 && a < Math.PI ? 0.86 : 1; // lower face is narrower than the cheekbones
    put(idx, cx + Math.cos(a) * ox, cy + Math.sin(a) * oy * jaw, -Math.cos(a) * 20);
  });

  // eyes
  const ew = fw * 0.30, eh = ew * 0.30;
  [[0, -1], [1, 1]].forEach(([n, side]) => {
    const ex = cx + side * fw * 0.215;
    EYE[n].forEach((idx, k) => {
      const a = (k * TAU) / EYE[n].length + Math.PI * 0.1;
      put(idx, ex + Math.cos(a) * ew * 0.5, eyeY + Math.sin(a) * eh * 0.62, 8 + Math.cos(a) * 2);
    });
    const [iu, il, ic, io] = EYM[n];
    put(iu, ex, eyeY - eh * 0.62, 9); put(il, ex, eyeY + eh * 0.62, 9);
    put(ic, ex - side * ew * 0.5, eyeY, 7); put(io, ex + side * ew * 0.5, eyeY, 6);
    put(IRIS[n], ex + side * ew * 0.02, eyeY, 12);
    for (let k = 1; k < 5; k++) {
      const a = (k * TAU) / 5;
      put(IRIS[n] + k, ex + Math.cos(a) * ew * 0.13, eyeY + Math.sin(a) * ew * 0.13, 12);
    }
  });

  // brows
  BROW.forEach((idx, k) => {
    const n = k < 5 ? 0 : 1, j = k % 5, side = n ? 1 : -1;
    const ex = cx + side * fw * 0.215;
    put(idx, ex + (j - 2) * ew * 0.22, eyeY - eh * 1.9 - Math.cos((j - 2) * 0.5) * eh * 0.35, 12);
  });

  // mouth
  const mw = fw * 0.44, my = cy + fh * 0.235, mh = mw * 0.30;
  LIP.forEach((idx, k) => {
    const a = (k * TAU) / LIP.length + Math.PI * 0.06;
    const upper = Math.cos(a) < 0;
    put(idx, cx + Math.sin(a) * mw * 0.5, my + Math.cos(a) * mh * (upper ? 0.42 : 0.5), 16);
  });
  GROUPS.lips.forEach((idx, k) => {
    const a = (k * TAU) / GROUPS.lips.length;
    put(idx, cx + Math.sin(a) * mw * 0.62, my + Math.cos(a) * mh * 0.85, 15);
  });
  put(61, cx - mw * 0.5, my, 14); put(291, cx + mw * 0.5, my, 14);
  put(13, cx, my - mh * 0.42, 17); put(14, cx, my + mh * 0.5, 17);
  put(0, cx, my - mh * 0.8, 16); put(17, cx, my + mh * 0.85, 16);

  // nose
  NOSE_RIDGE.forEach((idx, k) => {
    const t = k / (NOSE_RIDGE.length - 1);
    put(idx, cx, eyeY - eh * 1.2 + t * (my - mh * 0.9 - (eyeY - eh * 1.2)), 14 + t * 26);
  });
  put(4, cx, my - mh * 1.05, 40); put(1, cx, my - mh * 1.25, 36);
  NOSE_WING.forEach((idx, k) => {
    const side = k % 2 ? 1 : -1, t = Math.floor(k / 2);
    put(idx, cx + side * mw * (0.15 + t * 0.06), my - mh * (0.95 - t * 0.16), 30 - t * 6);
  });

  // chin / cheeks / forehead / temples
  CHIN.forEach((idx, k) => put(idx, cx + (k - 2) * fw * 0.06, cy + fh * 0.47 - Math.abs(k - 2) * 3, 18));
  put(152, cx, cy + fh * 0.5, 20);
  CHEEK.forEach((idx, k) => {
    const side = k % 2 ? 1 : -1, t = Math.floor(k / 2) / 6;
    put(idx, cx + side * fw * (0.2 + t * 0.22), eyeY + fh * (0.08 + t * 0.2), 16 - t * 12);
  });
  FOREHEAD.forEach((idx, k) => put(idx, cx + (h1(idx) - 0.5) * fw * 0.6, eyeY - fh * (0.18 + h1(idx + 5) * 0.16), 22 - h1(idx) * 6));
  put(10, cx, cy - fh * 0.5, 22);
  TEMPLE.forEach((idx, k) => { const side = k % 2 ? 1 : -1; put(idx, cx + side * fw * 0.48, eyeY - fh * 0.05 + Math.floor(k / 2) * fh * 0.06, 2); });
  put(234, cx - fw * 0.49, eyeY + fh * 0.02, 0); put(454, cx + fw * 0.49, eyeY + fh * 0.02, 0);

  // everything else: smooth scatter inside the oval so Delaunay has work to do
  for (let i = 0; i < 478; i++) {
    if (V[i]) continue;
    const a = h1(i * 3.3) * TAU, r = Math.sqrt(h1(i * 7.7)) * 0.94;
    const x = cx + Math.cos(a) * ox * r, y = cy + Math.sin(a) * oy * r * 1.02;
    const rr = Math.hypot((x - cx) / ox, (y - cy) / oy);
    put(i, x, y, 26 * Math.pow(Math.max(0, 1 - rr * rr), 0.45) - 6 * (h1(i) - 0.5));
  }

  // apply an in-plane yaw-ish squash so pose-dependent code paths get exercised
  if (yaw) {
    const c = Math.cos(yaw);
    for (let i = 0; i < 478; i++) {
      const dx = V[i][0] - cx;
      V[i][0] = cx + dx * c - V[i][2] * Math.sin(yaw) * 0.35;
      V[i][2] = (V[i][2] * c + dx * Math.sin(yaw) * 0.6) * zScale;
    }
  }
  return { V, W, H, frame: { cx, cy, fh, fw, eyeY, my } };
}

/** Minimal 2D context double: records calls, returns plausible values. */
export function mockCtx(w = 64, h = 64) {
  const calls = [];
  const log = [];
  const grad = { addColorStop() { } };
  const target = {
    canvas: { width: w, height: h },
    globalAlpha: 1, globalCompositeOperation: 'source-over', filter: 'none', lineWidth: 1,
    fillStyle: '#000', strokeStyle: '#000', lineCap: 'butt', lineJoin: 'miter', shadowBlur: 0, shadowColor: '', shadowOffsetY: 0,
    imageSmoothingQuality: 'high', imageSmoothingEnabled: true, font: '', textAlign: 'left',
  };
  // properties a real context saves/restores - tracked so tests can assert the
  // render path leaves no leaked blend mode, filter or alpha behind
  const STATE = ['globalAlpha', 'globalCompositeOperation', 'filter', 'lineWidth', 'fillStyle',
    'strokeStyle', 'lineCap', 'lineJoin', 'shadowBlur', 'shadowColor', 'shadowOffsetY',
    'imageSmoothingQuality', 'imageSmoothingEnabled', 'font', 'textAlign'];
  const stack = [];
  const handler = {
    get(t, k) {
      if (k in t) return t[k];
      if (k === 'save') return (...a) => { calls.push('save'); log.push(['save', a]); stack.push(STATE.map(p => t[p])); };
      if (k === 'restore') return (...a) => {
        calls.push('restore'); log.push(['restore', a]);
        const s = stack.pop();
        if (s) STATE.forEach((p, i) => { t[p] = s[i]; });
      };
      return (...args) => {
        calls.push(String(k));
        log.push([String(k), args]);
        if (k === 'createRadialGradient' || k === 'createLinearGradient') return grad;
        if (k === 'createPattern') return {};
        if (k === 'getImageData') return { data: new Uint8ClampedArray(Math.max(4, args[2] * args[3] * 4)), width: args[2], height: args[3] };
        if (k === 'createImageData') {
          const iw = args.length > 1 ? args[0] : args[0].width, ih = args.length > 1 ? args[1] : args[0].height;
          return { data: new Uint8ClampedArray(Math.max(4, (iw | 0) * (ih | 0) * 4)), width: iw, height: ih };
        }
        if (k === 'measureText') return { width: 10 };
        if (k === 'isPointInPath') return false;
        return undefined;
      };
    },
    set(t, k, v) { t[k] = v; return true; },
  };
  const ctx = new Proxy(target, handler);
  ctx.__calls = calls;
  ctx.__log = log;
  return ctx;
}

/**
 * Canvas double that hands out a mock 2D context and reports its own size, so
 * the compositing code (which resizes scratch buffers and reads ctx.canvas)
 * can be exercised in Node.
 */
export function mockCanvas(w = 64, h = 64) {
  const cv = {
    width: w, height: h, __ctx: null, __calls: null,
    getContext() {
      if (!this.__ctx) {
        this.__ctx = mockCtx(this.width, this.height);
        this.__ctx.canvas = this;
        this.__calls = this.__ctx.__calls;
        this.__log = this.__ctx.__log;
      }
      return this.__ctx;
    },
    toBlob(cb) { cb && cb({ size: this.width * this.height }); },
  };
  return cv;
}

/**
 * k-nearest-neighbour edge soup standing in for
 * FaceLandmarker.FACE_LANDMARKS_TESSELATION, so buildRig can run in Node.
 */
export function fakeTesselation(V, k = 6, n = 468) {
  const edges = [];
  for (let i = 0; i < n; i++) {
    const d = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = V[j][0] - V[i][0], dy = V[j][1] - V[i][1];
      d.push([dx * dx + dy * dy, j]);
    }
    d.sort((a, b) => a[0] - b[0]);
    for (let q = 0; q < k; q++) edges.push({ start: i, end: d[q][1] });
  }
  return edges;
}
