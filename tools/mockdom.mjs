/* =====================================================================
   FaceMirror · mock DOM + instrumented canvas 2D context  (test only)

   This exists so the puppet engine, renderer and UI can be executed in
   Node exactly as the browser would run them, with two extra powers:

     · every canvas op is recorded, so geometry can be asserted on
     · the classic "flashing" bug class is detected automatically:
       a wide fill / draw issued while the active clip is degenerate
       (a clip path with no real area) covers the whole canvas and makes
       the picture blink.  The mock flags exactly that, and the harness
       includes a negative control proving the detector fires.
   ===================================================================== */
import { parseHTMLIds } from './htmlids.mjs';

class Gradient {
  constructor(kind, args) { this.kind = kind; this.args = args; this.stops = []; }
  addColorStop(o, c) { this.stops.push([o, c]); }
}
class Pattern { constructor(src, rep) { this.src = src; this.rep = rep; } }

class Ctx {
  constructor(canvas) {
    this.canvas = canvas;
    this.W = canvas.width; this.H = canvas.height;
    this.ops = [];              // {op, args, clip, whole}
    this.violations = [];
    this.degenerate = false;
    this.path = [];
    this.sub = [];
    this.stack = [];
    this.cur = { clip: null, degenerate: false, alpha: 1, filter: 'none', gco: 'source-over', mat: [1, 0, 0, 1, 0, 0] };
    this.mat = [1, 0, 0, 1, 0, 0];
    this.globalAlpha = 1; this.filter = 'none'; this.globalCompositeOperation = 'source-over';
    this.imageSmoothingQuality = 'high'; this.lineWidth = 1; this.lineCap = 'butt'; this.lineJoin = 'miter';
    this.strokeStyle = '#000'; this.fillStyle = '#000'; this.font = '10px sans-serif';
    this.textAlign = 'start'; this.textBaseline = 'alphabetic'; this.shadowColor = 'transparent';
    this.shadowBlur = 0; this.shadowOffsetY = 0;
  }
  /* --- state bookkeeping ------------------------------------------- */
  save() {
    this.stack.push({ ...this.cur, mat: this.cur.mat.slice(), props: this.snapshotProps() });
  }
  restore() {
    const s = this.stack.pop();
    if (s) { this.cur = { clip: s.clip, degenerate: s.degenerate, alpha: s.alpha, filter: s.filter, gco: s.gco, mat: s.mat.slice() }; this.restoreProps(s.props); }
  }
  snapshotProps() {
    return { globalAlpha: this.globalAlpha, filter: this.filter, globalCompositeOperation: this.globalCompositeOperation,
             lineWidth: this.lineWidth, strokeStyle: this.strokeStyle, fillStyle: this.fillStyle, imageSmoothingQuality: this.imageSmoothingQuality };
  }
  restoreProps(p) { if (p) Object.assign(this, p); }
  setTransform(a, b, c, d, e, f) {
    if (typeof a === 'object' && a) { const m = a; this.mat = [m.a, m.b, m.c, m.d, m.e, m.f]; }
    else this.mat = [a, b, c, d, e, f];
    this.cur.mat = this.mat.slice();
    this.ops.push({ op: 'setTransform', args: this.mat.slice(), clip: this.cur.clip, degenerate: this.cur.degenerate });
  }
  resetTransform() { this.setTransform(1, 0, 0, 1, 0, 0); }
  translate(x, y) { this.ops.push({ op: 'translate', args: [x, y], clip: this.cur.clip, degenerate: this.cur.degenerate }); }
  rotate(a) { this.ops.push({ op: 'rotate', args: [a], clip: this.cur.clip, degenerate: this.cur.degenerate }); }
  scale(x, y) { this.ops.push({ op: 'scale', args: [x, y], clip: this.cur.clip, degenerate: this.cur.degenerate }); }
  transform(a, b, c, d, e, f) {
    const m = this.mat;                                  // compose like canvas2d
    this.mat = [m[0] * a + m[2] * b, m[1] * a + m[3] * b,
                m[0] * c + m[2] * d, m[1] * c + m[3] * d,
                m[0] * e + m[2] * f + m[4], m[1] * e + m[3] * f + m[5]];
    this.cur.mat = this.mat.slice();
    this.ops.push({ op: 'transform', args: [a, b, c, d, e, f], clip: this.cur.clip, degenerate: this.cur.degenerate });
  }
  /* --- paths --------------------------------------------------------- */
  beginPath() { this.path = []; this.sub = []; }
  moveTo(x, y) { this.sub = [[x, y]]; this.path.push(this.sub); }
  lineTo(x, y) { if (!this.sub.length) { this.sub = [[x, y]]; this.path.push(this.sub); } else this.sub.push([x, y]); }
  quadraticCurveTo(cx, cy, x, y) { this.lineTo(x, y); }
  bezierCurveTo(c1x, c1y, c2x, c2y, x, y) { this.lineTo(x, y); }
  arcTo(x1, y1, x2, y2, r) { this.lineTo(x2, y2); }
  rect(x, y, w, h) { this.moveTo(x, y); this.lineTo(x + w, y); this.lineTo(x + w, y + h); this.lineTo(x, y + h); this.closePath(); }
  closePath() { if (this.sub.length) this.sub.push(this.sub[0].slice()); }
  arc(cx, cy, r) {
    this.arcCount = (this.arcCount || 0) + 1;
    for (let i = 0; i <= 8; i++) {
      const a = i / 8 * Math.PI * 2;
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      if (i === 0) this.moveTo(x, y); else this.lineTo(x, y);
    }
  }
  ellipse(cx, cy, rx, ry) { this.arc(cx, cy, Math.max(rx, ry)); }
  clip() {
    const b = this.pathBBox();
    const area = this.pathArea();
    const inner = this.cur.clip;
    let box = b ? { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 } : null;
    if (box && inner) box = { x0: Math.max(box.x0, inner.x0), y0: Math.max(box.y0, inner.y0), x1: Math.min(box.x1, inner.x1), y1: Math.min(box.y1, inner.y1) };
    const degenerate = Math.abs(area) < 1 || !box;
    this.cur = { ...this.cur, clip: box, degenerate };
    this.ops.push({ op: 'clip', args: [area], clip: box, degenerate });
    this.opCount = (this.opCount || 0) + 1;
    this.totalOps = (this.totalOps || 0) + 1;
    if (this.ops.length > 4096) { this.ops.splice(0, this.ops.length - 4096); this.opDrops = (this.opDrops || 0) + 1; }
  }
  pathBBox() {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const s of this.path) for (const p of s) {
      if (!isFinite(p[0]) || !isFinite(p[1])) return null;
      x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]);
      x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
    }
    return isFinite(x0) ? { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 } : null;
  }
  pathArea() {
    let a = 0;
    for (const s of this.path) {
      if (s.length < 3) continue;
      for (let i = 0; i < s.length; i++) {
        const p = s[i], q = s[(i + 1) % s.length];
        a += p[0] * q[1] - q[0] * p[1];
      }
    }
    return a / 2;
  }
  /* --- painting ------------------------------------------------------ */
  note(op, args, box) {
    const rec = { op, args, clip: this.cur.clip, degenerate: this.cur.degenerate, box: box || null };
    /* The recorder is for per-frame inspection, not for archaeology: keep the
       tail only, otherwise a few thousand frames of tracing eats GBs.      */
    this.ops.push(rec);
    if (this.ops.length > 4096) { this.ops.splice(0, this.ops.length - 4096); this.opDrops = (this.opDrops || 0) + 1; }
    this.opCount = (this.opCount || 0) + 1;
    this.totalOps = (this.totalOps || 0) + 1;
    /* THE flash detector: a fill covering the whole canvas while the
       active clip has no area is exactly what makes the old build blink */
    const w = this.W, h = this.H;
    if (box && this.cur.degenerate && box.w >= w * 0.9 && box.h >= h * 0.9) {
      this.violations.push({ op, args, clip: this.cur.clip, box });
    }
    for (const a of args) {
      if (typeof a === 'number' && !isFinite(a)) {
        this.violations.push({ op: 'nonfinite', args, box });
        break;
      }
    }
  }
  fill() { const b = this.pathBBox(); this.note('fill', [], b); }
  stroke() { const b = this.pathBBox(); this.note('stroke', [], b); }
  fillRect(x, y, w, h) { this.note('fillRect', [x, y, w, h], { x0: x, y0: y, x1: x + w, y1: y + h, w, h }); }
  strokeRect(x, y, w, h) { this.note('strokeRect', [x, y, w, h], { x0: x, y0: y, x1: x + w, y1: y + h, w, h }); }
  clearRect(x, y, w, h) { this.note('clearRect', [x, y, w, h], { x0: x, y0: y, x1: x + w, y1: y + h, w, h }); }
  fillText(t, x, y) { this.note('fillText', [x, y], null); }
  strokeText(t, x, y) { this.note('strokeText', [x, y], null); }
  measureText(t) { return { width: (t || '').length * 6 }; }
  drawImage(src, ...rest) {
    const dims = src && (src.width || (src.naturalWidth || 0)) ? { w: src.width || src.naturalWidth, h: src.height || src.naturalHeight } : { w: 0, h: 0 };
    if (rest.length >= 8) {
      const [sx, sy, sw, sh, dx, dy, dw, dh] = rest;
      this.note('drawImage', [sx, sy, sw, sh, dx, dy, dw, dh], { x0: dx, y0: dy, x1: dx + dw, y1: dy + dh, w: dw, h: dh });
    } else if (rest.length >= 4) {
      const [dx, dy, dw, dh] = rest;
      this.note('drawImage', [dx, dy, dw, dh], { x0: dx, y0: dy, x1: dx + dw, y1: dy + dh, w: dw, h: dh });
    } else {
      const [dx, dy] = rest;
      this.note('drawImage', [dx, dy], { x0: dx, y0: dy, x1: dx + dims.w, y1: dy + dims.h, w: dims.w, h: dims.h });
    }
  }
  /* --- factories ----------------------------------------------------- */
  createLinearGradient(...a) { return new Gradient('linear', a); }
  createRadialGradient(...a) { return new Gradient('radial', a); }
  createPattern(src, rep) { return new Pattern(src, rep); }
  createImageData(w, h) { return { width: w || 1, height: h || 1, data: new Uint8ClampedArray(Math.max(4, (w || 1) * (h || 1) * 4)) }; }
  getImageData(x, y, w, h) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    const data = new Uint8ClampedArray(w * h * 4);
    this.getImageDataCalls = (this.getImageDataCalls || 0) + 1;
    /* a deterministic "photo": skin gradient, darker lips, dark iris */
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const X = (Math.round(x) + i) / Math.max(1, this.W), Y = (Math.round(y) + j) / Math.max(1, this.H);
        const skin = 0.9 - 0.25 * Y;
        const k = (j * w + i) * 4;
        data[k] = 200 * skin + 20;
        data[k + 1] = 160 * skin + 16;
        data[k + 2] = 132 * skin + 12;
        data[k + 3] = 255;
      }
    }
    return { width: w, height: h, data };
  }
  putImageData() { /* no-op */ }
}

class Canvas {
  constructor(w, h) {
    this.width = w || 300; this.height = h || 150;
    this._ctx = null;
    this.style = {};
    this.listeners = {};
    this.parentElement = null;
    this.tagName = 'CANVAS';
  }
  getContext(kind) {
    if (!this._ctx) this._ctx = new Ctx(this);
    this._ctx.W = this.width; this._ctx.H = this.height;
    return this._ctx;
  }
  toDataURL() { this.toDataURLCalls = (this.toDataURLCalls || 0) + 1; return 'data:image/png;base64,MOCK'; }
  captureStream() { return null; }
  getBoundingClientRect() { return { left: 0, top: 0, width: this.width, height: this.height, right: this.width, bottom: this.height }; }
  addEventListener(k, f) { (this.listeners[k] = this.listeners[k] || []).push(f); }
  /** dispatch a synthetic event, for interaction tests */
  fire(k, ev) { for (const f of this.listeners[k] || []) f({ preventDefault() {}, ...ev }); }
}

class El {
  constructor(tag, attrs = {}) {
    this.tagName = (tag || 'div').toUpperCase();
    this.attrs = attrs;
    this.id = attrs.id || '';
    this.dataset = { ...attrs.dataset };
    this.style = {};
    this.children = [];
    this.listeners = {};
    this.value = attrs.value !== undefined ? attrs.value : '';
    this.checked = !!attrs.checked;
    this.disabled = !!attrs.disabled;
    this.textContent = '';
    this.hidden = 'hidden' in attrs;
    this.files = [];
    this.classList = {
      _s: new Set(String(attrs.class || '').split(/\s+/).filter(Boolean)),
      add: c => this.classList._s.add(c),
      remove: c => this.classList._s.delete(c),
      contains: c => this.classList._s.has(c),
      toggle: c => (this.classList._s.has(c) ? (this.classList._s.delete(c), false) : (this.classList._s.add(c), true))
    };
    this.parentElement = null;
    this.clientWidth = attrs.clientWidth || 700;
  }
  addEventListener(k, f) { (this.listeners[k] = this.listeners[k] || []).push(f); }
  fire(k, ev) { for (const f of this.listeners[k] || []) f({ preventDefault() {}, target: this, ...ev }); }
  appendChild(c) { c.parentElement = c.parentElement || this; this.children.push(c); return c; }
  querySelectorAll(sel) {
    const tag = String(sel).replace(/[^a-zA-Z\[\]=]/g, '');
    return this.children.filter(c => {
      if (sel.includes('[')) {
        const m = /\[([\w-]+)(?:=([^\]]+))?\]/.exec(sel);
        if (m) return m[2] === undefined ? c.dataset[m[1]] !== undefined : c.dataset[m[1]] === m[2].replace(/["']/g, '');
      }
      return c.tagName === tag.replace('button', 'BUTTON');
    });
  }
  closest(sel) {
    for (let node = this; node; node = node.parentElement) {
      if (sel === 'button[data-sculpt]' && node.tagName === 'BUTTON' && node.dataset.sculpt !== undefined) return node;
      if (sel === 'button[data-preset]' && node.tagName === 'BUTTON' && node.dataset.preset !== undefined) return node;
    }
    return null;
  }
  setPointerCapture() {}
  releasePointerCapture() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientWidth * 0.75, right: this.clientWidth, bottom: this.clientWidth * 0.75 }; }
  click() { this.fire('click', {}); }
}

export function installMockDom(html, opts = {}) {
  const ids = parseHTMLIds(html);
  const byId = new Map();
  const allCanvases = new Set();
  for (const [id, info] of ids) {
    const tag = info.tag.toLowerCase();
    if (tag === 'input' && info.attrs.type === 'range') {
      // the read-out <output> sibling is faked so numbering code paths run
      const el = new El('input', { id, value: info.attrs.value, type: 'range' });
      el.parentElement = new El('label', {});
      el.parentElement.appendChild(new El('output', {}));
      byId.set(id, el);
      continue;
    }
    if (tag === 'canvas') {
      const c = new Canvas(parseInt(info.attrs.width, 10) || 640, parseInt(info.attrs.height, 10) || 480);
      c.id = id; c.classList = new El('canvas', {}).classList;
      allCanvases.add(c);
      byId.set(id, c);
      continue;
    }
    const el = new El(tag, {
      id,
      value: info.attrs.value,
      checked: 'checked' in info.attrs,
      disabled: 'disabled' in info.attrs,
      class: info.attrs.class
    });
    byId.set(id, el);
  }
  const document = {
    readyState: 'complete',
    getElementById: id => byId.get(id) || null,
    createElement: tag => {
      if (tag !== 'canvas') return new El(tag, {});
      const c = new Canvas(300, 150);
      allCanvases.add(c);
      return c;
    },
    querySelectorAll: () => [],
    addEventListener() {},
    body: new El('body', {}),
    _byId: byId
  };
  const win = {
    document,
    navigator: { mediaDevices: { getUserMedia: async () => { throw new Error('no camera in the harness'); } } },
    performance: { now: () => harnessClock },
    requestAnimationFrame: () => 0,
    addEventListener() {},
    innerWidth: 1280,
    setTimeout: (f, ms) => 0,
    clearTimeout: () => {},
    console
  };
  let harnessClock = 0;
  const setClock = v => { harnessClock = v; };
  return { window: win, document, byId, allCanvases, setClock, ids, El, Canvas, Ctx };
}

export { El, Canvas, Ctx };
