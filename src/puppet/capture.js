/**
 * Output stage: PNG frame capture and WebM/MP4 video recording of the rendered
 * puppet, with optional microphone audio mixed into the recording.
 *
 * The recorder uses captureStream(0) plus an explicit requestFrame() after every
 * rendered frame. That gives the encoder exactly the frames the renderer
 * produced (no duplicated or dropped frames from a fixed-rate capture clock),
 * which removes the stutter you otherwise see in canvas recordings.
 */

export const MIME_CANDIDATES = [
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8,opus',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4',
];

export function pickMime(preferred) {
  if (typeof MediaRecorder === 'undefined') return null;
  const list = preferred ? [preferred, ...MIME_CANDIDATES] : MIME_CANDIDATES;
  for (const m of list) { try { if (MediaRecorder.isTypeSupported(m)) return m; } catch (e) { /* keep looking */ } }
  return '';
}

export function download(blob, name) {
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function saveCanvasPNG(canvas, name = 'trill-face-puppet.png') {
  return new Promise(resolve => {
    canvas.toBlob(b => {
      if (b) { download(b, name); resolve(true); } else resolve(false);
    }, 'image/png');
  });
}

export class Recorder {
  constructor(canvas) {
    this.canvas = canvas;
    this.rec = null; this.chunks = []; this.stream = null; this.track = null;
    this.startedAt = 0; this.mime = ''; this.frames = 0; this.manual = false;
    this.onStop = null; this.onError = null;
  }
  get active() { return !!this.rec && this.rec.state !== 'inactive'; }
  get elapsed() { return this.startedAt ? (performance.now() - this.startedAt) / 1000 : 0; }
  start({ audioTrack = null, fps = 30, bitrate } = {}) {
    if (typeof MediaRecorder === 'undefined') { this.onError && this.onError('MediaRecorder is not supported by this browser'); return false; }
    this.stop(true);
    this.chunks = [];
    this.stream = this.canvas.captureStream(0);
    this.track = this.stream.getVideoTracks()[0];
    this.manual = !!(this.track && typeof this.track.requestFrame === 'function');
    if (!this.manual) { this.stream = this.canvas.captureStream(fps); this.track = this.stream.getVideoTracks()[0]; }
    if (audioTrack) { try { this.stream.addTrack(audioTrack); } catch (e) { /* audio optional */ } }
    this.mime = pickMime();
    const opts = this.mime ? { mimeType: this.mime } : {};
    if (bitrate) opts.videoBitsPerSecond = bitrate;
    try { this.rec = new MediaRecorder(this.stream, opts); }
    catch (e) {
      try { this.rec = new MediaRecorder(this.stream); }
      catch (e2) { this.onError && this.onError(e2.message || String(e2)); return false; }
    }
    this.rec.ondataavailable = e => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.rec.onerror = e => { this.onError && this.onError((e.error && e.error.name) || 'recorder error'); };
    this.rec.onstop = () => {
      const type = (this.rec && this.rec.mimeType) || this.mime || 'video/webm';
      const blob = new Blob(this.chunks, { type });
      this.chunks = [];
      const ext = type.includes('mp4') ? 'mp4' : 'webm';
      this.onStop && this.onStop(blob, ext, this.frames, this.elapsed);
      this.frames = 0; this.startedAt = 0;
      try { this.stream && this.stream.getTracks().forEach(t => t.stop()); } catch (e) { /* done */ }
      this.stream = null; this.rec = null;
    };
    this.rec.start(250);
    this.startedAt = performance.now();
    this.frames = 0;
    this.frame();
    return true;
  }
  /** Hand the freshly rendered frame to the encoder. */
  frame() {
    if (!this.active) return;
    this.frames++;
    if (this.manual) { try { this.track.requestFrame(); } catch (e) { /* track ended */ } }
  }
  stop(silent) {
    if (this.rec && this.rec.state !== 'inactive') { if (silent) this.rec.onstop = null; this.rec.stop(); }
    else if (silent) { this.rec = null; this.stream = null; }
  }
}
