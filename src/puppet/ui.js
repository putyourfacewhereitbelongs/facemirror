/* =====================================================================
   FaceMirror · puppet UI  (DOM side)

   Split of duties, which is what keeps the picture calm:
     runDetection()  fires only when the browser hands us a *new* video
                     frame.  It touches nothing but raw signals.
     smoothSignals() runs every animation frame and turns raw signals
                     into envelopes with slew limits and one-euro filters.
     composeDelta()  sums preset actions + webcam residual + sculpts into
                     one canonical delta, then rate limits the whole thing.
     sessionFrame()  paints.  It never reads a raw threshold.

   Deliberate stability choices
     · no allocation and no getImageData inside the frame loop
     · every signal is filtered; every gate is a float in [0,1]
     · "set neutral" moves the pose target only, so the slew limiter
       turns it into a glide instead of a jump
     · detection loss decays the pose over ~1 s, never snaps to rest
   ===================================================================== */
(function (root) {
  'use strict';
  const FM = root.FM, E = root.FMEngine, RR = root.FMRender;
  const { clamp, clamp01, lerp, slew, envelope, ratioTracker, ratioTrack, eulerFromQuat } = FM;

  const GROUPS = {
    oval: [10, 338, 297, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 152, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 109, 67],
    leye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    reye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    brow: [70, 63, 105, 66, 107, 300, 293, 334, 296, 336],
    lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    inner: [78, 82, 13, 312, 308, 317, 14, 87],
    nose: [4, 168, 6, 197, 195, 5, 1, 2, 98, 327]
  };
  const GROUP_COLOR = { oval: '#60a5fa', leye: '#5eead4', reye: '#5eead4', brow: '#fbbf24', lips: '#f472b6', inner: '#fb7185', nose: '#a78bfa' };
  const CAP = { oval: 0.05, leye: 0.015, reye: 0.015, brow: 0.05, lips: 0.07, inner: 0.07, nose: 0.02 };
  const GROUP_NAME = { oval: 'Face outline', leye: 'Eye', reye: 'Eye', brow: 'Eyebrow', lips: 'Lips', inner: 'Inner mouth', nose: 'Nose' };

  const $ = id => (root.document ? root.document.getElementById(id) : null);
  const mix = (a, b, t) => a + (b - a) * t;
  /* every signal that enters the action mixer goes through nz(): a single
     undefined multiplier (e.g. a blendshape the detector did not report)
     would otherwise poison the whole expression field with NaN.        */
  const nz = v => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

  const A = {
    MP: null, mesh: null, head: null, ses: null, tracker: null, vis: E.createViseme(),
    deltaExpr: new Float32Array(478 * 3),   // canonical delta before click-to-sculpt
    src: null, cam: null, puppetCtx: null,
    faceImg: null, faceVid: null, poseImg: null, poseVid: null,
    camOn: false, stream: null, audioCtx: null, analyser: null, freq: null, micOn: false, micStream: null,
    preset: 'live', quality: 1, lastVideoTime: -1,
    actions: E.zeroActions(), live: E.zeroActions(), target: E.zeroActions(),
    sculpt: new Float32Array(478 * 2), bag: [], drag: -1, hover: -1, dragOff: { x: 0, y: 0 }, sculptMode: 'drag',
    gaze: [[0, 0], [0, 0]], gazeS: [[0, 0], [0, 0]], gazeNeutral: [[0, 0], [0, 0]], sacT: 0, sacV: [0, 0],
    lidRaw: [0, 0], lid: [0, 0], lidN: [null, null], kissRaw: 0, kiss: 0, tongueRaw: 0, tongue: 0,
    liveKiss: 0, liveTongue: 0, liveSmile: 0, liveFrown: 0, liveSquint: 0,
    exprBuf: new Float32Array(478 * 3), resBuf: new Float32Array(478 * 3), tgtBuf: new Float32Array(478 * 3),
    outBuf: new Float32Array(478 * 3),
    perf: { last: 0, dt: 1 / 60, fps: 0, n: 0, t0: 0, hud: 0, t: 0 },
    pose: { cur: null, neu: null, JI: null, tick: 0 },
    pointList: null, partIdx: null, p2: null,
    lastFaceAt: -1e9, err: null, status: '', statusBase: '', rec: null, recStream: null, chunks: [], keyT: false,
    photoName: 'photo', idle: { next: 0, at: 0 }, crash: null
  };

  /* --------------------------------------------------------- utilities */
  function setStatus(msg) { A.status = msg; const el = $('status'); if (el) el.textContent = msg; }
  function flashStatus(msg, ms) {
    const el = $('status');
    if (el) {
      el.textContent = msg;
      clearTimeout(A._st);
      A._st = setTimeout(() => { const e2 = $('status'); if (e2) e2.textContent = A.statusBase || A.status; }, ms || 2800);
    }
  }
  const num = (id, dflt) => { const el = $(id); const v = el ? parseFloat(el.value) : NaN; return isFinite(v) ? v : dflt; };
  const chk = (id) => { const el = $(id); return !!(el && el.checked); };

  /* ------------------------------------------------------------- boot */
  async function boot(vision) {
    A.MP = vision;
    wireControls();
    setStatus('Loading the face model…');
    try {
      const fileset = await vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
      const faceModel = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
      const makeFace = async (mode) => {
        for (const delegate of ['GPU', 'CPU']) {
          try {
            return await vision.FaceLandmarker.createFromOptions(fileset, {
              baseOptions: { modelAssetPath: faceModel, delegate },
              runningMode: mode, numFaces: 1, outputFaceBlendshapes: mode === 'VIDEO'
            });
          } catch (e) { if (delegate === 'CPU') throw e; }
        }
      };
      A.faceImg = await makeFace('IMAGE');
      A.faceVid = await makeFace('VIDEO');
      try {
        const poseModel = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
        const mk = async (mode) => {
          for (const delegate of ['GPU', 'CPU']) {
            try {
              return await vision.PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: poseModel, delegate }, runningMode: mode, numPoses: 1 });
            } catch (e) { if (delegate === 'CPU') throw e; }
          }
        };
        A.poseImg = await mk('IMAGE');
        A.poseVid = await mk('VIDEO');
      } catch (e) { A.poseErr = e && e.message; }
      const cb = $('camBtn');
      if (cb) cb.disabled = false;
      setStatus('Ready. Start the webcam, then choose a photo (or snapshot yourself).');
      requestAnimationFrame(tick);
    } catch (e) {
      A.err = e;
      setStatus('Could not load the face model: ' + (e && e.message) + ' — this page needs cdn.jsdelivr.net and storage.googleapis.com.');
      requestAnimationFrame(tick);
    }
  }

  /* -------------------------------------------------------- source image */
  function loadImage(file) {
    if (!A.faceImg) { setStatus('The face model is still loading — try again in a second.'); return; }
    const img = new Image();
    img.onload = () => {
      try { setPhotoFromElement(img, img.naturalWidth, img.naturalHeight, (file && file.name) || 'photo'); }
      finally { URL.revokeObjectURL(img.src); }
    };
    img.onerror = () => setStatus('That file could not be decoded as an image.');
    img.src = URL.createObjectURL(file);
  }
  function setPhotoFromElement(el, w, h, name) {
    if (!w || !h) { setStatus('That source has no pixels yet — try again.'); return; }
    const s = Math.min(1, 1000 / Math.max(w, h));
    const canvas = RR.makeCanvas(w * s, h * s);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
    let res = null;
    try { res = A.faceImg.detect(canvas); } catch (e) { setStatus('Detection failed: ' + e.message); return; }
    if (!res || !res.faceLandmarks || !res.faceLandmarks.length) {
      setStatus('No face found in that image. Try a clearer, front-facing photo with even lighting.');
      return;
    }
    const tess = (A.MP.FaceLandmarker && A.MP.FaceLandmarker.FACE_LANDMARKS_TESSELATION) || null;
    buildFromLandmarks(res.faceLandmarks[0], canvas, tess);
    A.photoName = name || 'photo';
    /* the photo's own shoulders, if present: drives the pinned skirt */
    A.pose.JI = null;
    if (A.poseImg) {
      try {
        const pr = A.poseImg.detect(canvas);
        const lm = pr && pr.landmarks && pr.landmarks[0];
        if (lm && lm[11] && lm[12] && lm[11].visibility > .4 && lm[12].visibility > .4) {
          const q = lm.map(p => [p.x * canvas.width, p.y * canvas.height, p.visibility]);
          if (Math.hypot(q[11][0] - q[12][0], q[11][1] - q[12][1]) > A.head.fh * 0.35) A.pose.JI = q;
        }
      } catch (e) { /* body is optional */ }
    }
    const pb = $('photoBtn'), rb = $('recordBtn');
    if (pb) pb.disabled = false;
    if (rb) rb.disabled = false;
    setStatus(`Face found — ${A.mesh.n} mesh points, ${Math.round(A.mesh.base.length / 3)} shell triangles`
      + (A.pose.JI ? ', upper-body landmarks detected.' : ' (head-only photo).')
      + ' Drag a dot, or click the face with Smile / Frown / Mouth O.');
    A.statusBase = A.status;
    resizeStage();
  }
  /* shared by the real path and the headless test hook */
  function buildFromLandmarks(landmarks, canvas, tessellation) {
    A.src = canvas;
    A.puppetCtx = null;
    const W = canvas.width, H = canvas.height;
    let adjacency = null;
    if (tessellation) {
      const adj = Array.from({ length: 478 }, () => []);
      const seen = Array.from({ length: 478 }, () => new Set());
      for (const c of tessellation) {
        if (c.start < 478 && c.end < 478 && !seen[c.start].has(c.end)) {
          adj[c.start].push(c.end); adj[c.end].push(c.start);
          seen[c.start].add(c.end); seen[c.end].add(c.start);
        }
      }
      adjacency = adj;
    }
    A.head = E.buildCanonical(landmarks, W, H, { adjacency });
    A.mesh = E.buildMesh(A.head, { delaunay: root.Delaunator, tessellation });
    A.ses = RR.createSession();
    RR.sessionInit(A.ses, { source: canvas, canvas: $('puppet'), head: A.head, mesh: A.mesh });
    A.tracker = E.createTracker(A.head, {});
    A.p2 = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
    A.sculpt = new Float32Array(478 * 2);
    A.outBuf.fill(0);
    A.tgtBuf.fill(0);
    A.bag = [];
    A.drag = -1; A.hover = -1;
    A.lidN = [null, null];
    A.gazeNeutral = [[0, 0], [0, 0]];
    buildPoints();
    buildPartners();
  }
  function buildPoints() {
    const list = [];
    for (const g in GROUPS) {
      const seen = new Set();
      for (const v of GROUPS[g]) {
        if (seen.has(v)) continue;
        seen.add(v);
        list.push({ v, k: v, c: GROUP_COLOR[g], g });
      }
    }
    A.pointList = list;
  }
  /* mirror partner of every landmark, for symmetric dragging */
  function buildPartners() {
    const P = A.head.P, mid = A.head.mid[0], fh = A.head.fh;
    A.partIdx = new Int16Array(478).fill(-1);
    for (let i = 0; i < 478; i++) {
      const mx = 2 * mid - P[i][0], my = P[i][1];
      let bi = -1, bd = (0.05 * fh) ** 2;
      for (let j = 0; j < 478; j++) {
        if (j === i) continue;
        const d = (P[j][0] - mx) ** 2 + (P[j][1] - my) ** 2;
        if (d < bd) { bd = d; bi = j; }
      }
      A.partIdx[i] = bi;
    }
  }

  /* ------------------------------------------------------------- webcam */
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false
      });
      A.stream = stream;
      const vid = $('vid');
      vid.srcObject = stream;
      await vid.play();
      A.cam = RR.makeCanvas(vid.videoWidth || 640, vid.videoHeight || 480);
      A.cam.getContext('2d', { willReadFrequently: true });
      A.camOn = true;
      const cb = $('camBtn');
      if (cb) cb.disabled = true;
      ['snapBtn', 'calBtn', 'photoBtn', 'recordBtn'].forEach(id => { const b = $(id); if (b) b.disabled = false; });
      resizeStage();
      setStatus('Webcam running. Sit so head and shoulders are in frame and relax your face — that becomes the neutral pose.');
      A.statusBase = A.status;
    } catch (e) {
      const denied = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
      setStatus(denied
        ? 'Camera blocked. Allow camera access for this page — if the preview is embedded in a frame, open it in its own tab and press Start webcam again.'
        : 'Webcam unavailable: ' + (e && e.message ? e.message : e) + ' — this needs https:// or localhost plus camera permission.');
    }
  }
  function snapshotFromWebcam() {
    const vid = $('vid');
    if (!A.camOn || !vid || !vid.videoWidth || !vid.videoHeight) { flashStatus('Start the webcam first.'); return; }
    /* Tracking is performed on a horizontally mirrored camera frame.  Use
       that same convention for the still source, otherwise neutral pose and
       the snapshot have opposite handedness and yaw appears to pull the face
       across the image.  Refresh the reusable tracking canvas at click time
       so the captured photo is not one decoded frame behind. */
    const W = vid.videoWidth, H = vid.videoHeight;
    if (!A.cam || A.cam.width !== W || A.cam.height !== H) A.cam = RR.makeCanvas(W, H);
    const ctx = A.cam.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.setTransform(-1, 0, 0, 1, W, 0);
    ctx.drawImage(vid, 0, 0, W, H);
    ctx.restore();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    setPhotoFromElement(A.cam, W, H, 'webcam snapshot');
  }
  function neutralSet() {
    if (!A.tracker || !A.tracker.have) { flashStatus('No tracked face yet — look at the camera, then press it again.'); return; }
    E.trackerSetNeutral(A.tracker, A.cam ? A.cam.width : 640, A.cam ? A.cam.height : 480);
    if (syncP2()) A.gazeNeutral = [E.irisOffset(A.p2, 'L'), E.irisOffset(A.p2, 'R')];
    if (A.pose.cur) A.pose.neu = A.pose.cur.map(p => p.slice());
    flashStatus('Neutral pose stored — the pose glides to it.');
  }

  /* --------------------------------------------------- detection frame */
  function runDetection(t) {
    const vid = $('vid');
    if (!A.camOn || !vid || vid.readyState < 2) return false;
    if (vid.currentTime === A.lastVideoTime) return false;
    A.lastVideoTime = vid.currentTime;
    const cv = A.cam, ctx = cv.getContext('2d');
    ctx.save();
    ctx.setTransform(-1, 0, 0, 1, cv.width, 0);
    ctx.drawImage(vid, 0, 0, cv.width, cv.height);
    ctx.restore();
    let res = null;
    try { res = A.faceVid.detectForVideo(cv, t); } catch (e) { A.err = e; return false; }
    const L = res && res.faceLandmarks && res.faceLandmarks[0];
    if (L && E.trackerPush(A.tracker, L, cv.width, cv.height, t)) {
      A.lastFaceAt = t;
      if (!A.tracker.neutral) neutralSet();
      readSignals(res);
    }
    if (A.poseVid && ((A.pose.tick = (A.pose.tick + 1) & 1) === 0)) {
      try {
        const pr = A.poseVid.detectForVideo(cv, t);
        const lm = pr && pr.landmarks && pr.landmarks[0];
        if (lm) {
          const q = lm.map(p => [p.x * cv.width, p.y * cv.height, p.visibility]);
          const prev = A.pose.cur;
          A.pose.cur = prev ? q.map((p, i) => [lerp(prev[i][0], p[0], .45), lerp(prev[i][1], p[1], .45), p[2]]) : q;
          if (!A.pose.neu && q[11] && q[11][2] > .5 && q[12] && q[12][2] > .5) A.pose.neu = A.pose.cur.map(p => p.slice());
        }
      } catch (e) { /* body is a bonus */ }
    }
    return true;
  }

  /* raw signals out of a detection result (persistent, per video frame) */
  function readSignals(res) {
    const bs = (res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories) || [];
    const bv = E.blendGetter(bs);
    if (!syncP2()) return;
    const p2 = A.p2;
    const live = A.preset === 'live' || chk('pmix');
    /* eyelids: blendshape and geometry agree most of the time */
    for (const which of ['L', 'R']) {
      const e = which === 'L' ? 0 : 1;
      const ratio = E.eyeOpeningRatio(p2, which);
      if (!A.lidN[e]) A.lidN[e] = ratioTracker(ratio);
      ratioTrack(A.lidN[e], ratio);
      const geo = clamp01((A.lidN[e].v * 0.84 - ratio) / (A.lidN[e].v * 0.6));
      const bl = clamp01((bv(which === 'L' ? 'eyeBlinkLeft' : 'eyeBlinkRight') - 0.12) / 0.62);
      let m = mix(geo, bl, 0.4);
      const other = e === 0 ? 1 : 0;
      const otherV = A.lidRaw[other];
      if (Math.abs((A.lidRaw[e] || 0) - otherV) > 0.45) m = Math.min(m, otherV * 0.3 + 0.02);  // a wink keeps the far eye open
      A.lidRaw[e] = clamp01(m);
    }
    const L = A.live;
    for (const k in L) L[k] = 0;
    if (live) {
      const smile = (bv('mouthSmileLeft') + bv('mouthSmileRight')) / 2;
      L.j = clamp01((bv('jawOpen') - 0.04) / 0.68);
      L.sL = L.sR = clamp01(smile * 1.2);
      L.fr = clamp01((bv('mouthFrownLeft') + bv('mouthFrownRight')) / 2 * 1.2);
      L.bu = clamp01(bv('browInnerUp') * 0.7 + (bv('browOuterUpLeft') + bv('browOuterUpRight')) / 2 * 0.6);
      L.buR = clamp01(bv('browOuterUpRight') * 0.9) - 0.45;
      L.bd = clamp01(((bv('browDownLeft') + bv('browDownRight')) / 2) * 1.15);
      L.sq = clamp01(((bv('eyeSquintLeft') + bv('eyeSquintRight')) / 2) * 1.1);
      L.nz = clamp01((bv('noseSneerLeft') + bv('noseSneerRight')) / 2);
      L.ur = clamp01((bv('mouthUpperUpLeft') + bv('mouthUpperUpRight')) / 2);
      L.ck = clamp01(((bv('cheekPuff') + bv('cheekSquintLeft') + bv('cheekSquintRight')) / 3) * 1.2);
      A.liveKiss = clamp01(Math.max(nz(bv('mouthPucker')), nz(bv('mouthFunnel')) * 0.7) * 1.5 - 0.08);
      A.liveTongue = clamp01((nz(bv('tongueOut')) * num('tsens', 1.4) - 0.05) / 0.5);
    } else {
      A.liveKiss = A.liveTongue = 0;
    }
  }

  /* ------------------------------------------------- per-frame smoothing */
  const syncP2 = () => {
    const src = A.tracker && A.tracker.src;
    if (!src || !src[0]) return false;
    const p2 = A.p2 || (A.p2 = Array.from({ length: 478 }, () => ({ x: 0, y: 0 })));
    for (let i = 0; i < 478; i++) { p2[i].x = src[i][0]; p2[i].y = src[i][1]; }
    return true;
  };
  function smoothSignals(dt) {
    const live = A.tracker && A.tracker.trust > 0.25;
    const sens = num('bl', 1);
    if (!live && chk('idleBlink') && !A.camOn) {
      const t = A.perf.t * 1000;
      if (t > A.idle.next) { A.idle.next = t + 2200 + Math.random() * 3200; A.idle.at = t; }
      const k = (t - A.idle.at) / 190;
      const v = k < 1 ? Math.sin(Math.PI * clamp01(k)) : 0;
      A.lidRaw[0] = A.lidRaw[1] = v;
    } else if (!live && chk('idleBlink')) {
      const t = A.perf.t * 1000;
      if (t > A.idle.next) { A.idle.next = t + 2200 + Math.random() * 3200; A.idle.at = t; }
      const k = (t - A.idle.at) / 190;
      const v = k < 1 ? Math.sin(Math.PI * clamp01(k)) : 0;
      A.lidRaw[0] = Math.max(A.lidRaw[0], v);
      A.lidRaw[1] = Math.max(A.lidRaw[1], v);
    }
    for (let e = 0; e < 2; e++) {
      const target = live ? clamp01((A.lidRaw[e] * sens - 0.08) / 0.74) : slew(A.lid[e], 0, dt, 3.2, 3.2);
      A.lid[e] = live ? envelope(A.lid[e], target, dt, 26, 17) : slew(A.lid[e], 0, dt, 3.2, 3.2);
    }
    /* actions: preset layer and webcam layer, then one rate limit */
    const presetActs = A.preset === 'Talking' ? E.talkAt(A.perf.t) : (E.PRESETS[A.preset] || null);
    const amt = num('pi', 1);
    for (const k of E.ACTIONS) {
      let v = 0;
      if (presetActs && presetActs[k]) v = nz(presetActs[k]) * amt;
      if (live && A.live[k]) {
        const gain = k === 'j' ? 0.8 : (A.preset === 'live' ? 0.35 : 0.5);
        v = Math.max(v, nz(A.live[k]) * gain);
      }
      A.target[k] = v;
      A.actions[k] = slew(nz(A.actions[k]), v, dt, 2.2, 1.9);
    }
    A.kiss = envelope(A.kiss, clamp01(nz(A.liveKiss) * (live ? 1 : 0)), dt, 14, 10);
    A.tongue = envelope(A.tongue, clamp01(Math.max((live ? nz(A.liveTongue) : 0), nz(A.actions.tg), (chk('forceTongue') || A.keyT) ? 1 : 0)), dt, 16, 9);
    /* gaze: tracked iris offset, or idle saccades when nothing drives it */
    if (live) syncP2();
    const gg = num('gg', 1.9);
    const now = A.perf.t * 1000;
    if (now > A.sacT) { A.sacT = now + 700 + Math.random() * 2400; A.sacV = [(Math.random() - .5) * .07, (Math.random() - .5) * .045]; }
    for (let e = 0; e < 2; e++) {
      let gx, gy;
      if (live && A.p2 && A.p2[468]) {
        const r = E.irisOffset(A.p2, e ? 'R' : 'L');
        gx = (r[0] - A.gazeNeutral[e][0]) * gg;
        gy = (r[1] - A.gazeNeutral[e][1]) * gg;
      } else { gx = A.sacV[0]; gy = A.sacV[1]; }
      gx += (A.actions.gx || 0) * 0.22;
      gy += (A.actions.gy || 0) * 0.11;
      A.gazeS[e][0] += (gx - A.gazeS[e][0]) * clamp01(dt * 26);
      A.gazeS[e][1] += (gy - A.gazeS[e][1]) * clamp01(dt * 26);
      A.gaze[e] = [clamp(A.gazeS[e][0], -.30, .30), clamp(A.gazeS[e][1], -.14, .14)];
    }
    /* visemes from the microphone */
    if (A.micOn && A.analyser && A.freq) {
      const n = A.analyser.frequencyBinCount;
      A.analyser.getByteFrequencyData(A.freq);
      let sum = 0;
      for (let i = 0; i < n; i++) sum += A.freq[i] * A.freq[i];
      const rms = Math.sqrt(sum / n) / 255;
      const va = E.visemeActions(E.visemeUpdate(A.vis, A.freq, rms, dt), 1);
      if (va) for (const k in va) {
        const v = va[k];
        A.target[k] = Math.max(nz(A.target[k]), v);
        A.actions[k] = slew(nz(A.actions[k]), A.target[k], dt, 4.2, 3.6);
      }
    }
  }

  /* ------------------------------------------------------ expression mix */
  function composeDelta(dt) {
    const strength = num('exs', 0.8);
    const acts = { ...A.actions, pk: Math.max(nz(A.actions.pk), nz(A.kiss)), tg: Math.max(nz(A.actions.tg), nz(A.tongue)) };
    const d1 = E.expressionDelta(A.head, acts, A.exprBuf);
    for (let i = 0; i < 478 * 3; i++) A.tgtBuf[i] = d1[i] * strength;
    const useRes = A.tracker && A.tracker.neutral && (A.preset === 'live' ? true : chk('pmix'));
    if (useRes) {
      const g = clamp01(strength * 1.15) * A.tracker.trust;
      E.retargetResidual(A.head, null, A.tracker.residual, g, A.resBuf);
      for (let i = 0; i < 478 * 3; i++) A.tgtBuf[i] += A.resBuf[i];
    }
    A.deltaExpr.set(A.tgtBuf.subarray(0, 478 * 3));   // expression without the sculpt
    for (let i = 0; i < 478; i++) {
      A.tgtBuf[i * 3] += A.sculpt[i * 2];
      A.tgtBuf[i * 3 + 1] += A.sculpt[i * 2 + 1];
    }
    /* one global rate limit on the finished delta */
    const cap = A.head.fw * 3.0;
    for (let i = 0; i < 478 * 3; i++) A.outBuf[i] = slew(A.outBuf[i], A.tgtBuf[i], dt, cap, cap);
    return A.outBuf;
  }

  /* ---------------------------------------------------------------- loop */
  function tick(now) {
    requestAnimationFrame(tick);
    try { step(now); } catch (e) {
      if (!A.crash) { A.crash = e; console.error(e); setStatus('Render error: ' + e.message); }
    }
  }
  function step(now, renderFrame = true) {
    const t = now || (root.performance ? performance.now() : Date.now());
    const P = A.perf;
    if (!P.last) P.last = t;
    P.dt = clamp((t - P.last) / 1000, 1 / 240, 0.06);
    P.last = t; P.t += P.dt;
    P.n++;
    if (t - P.t0 > 500) { P.fps = Math.round(P.n * 1000 / Math.max(1, t - P.t0)); P.n = 0; P.t0 = t; }
    if (A.ses && A.ses.ready) {
      if (A.camOn) runDetection(t);
      if (A.tracker && A.tracker.neutral) {
        E.trackerUpdate(A.tracker, P.dt, {
          yawGain: num('turn', 1), pitchGain: num('turn', 1) * 0.75, rollGain: num('turn', 1) * 0.6,
          yawLimit: 46, pitchLimit: 26, rollLimit: 22, exprGain: 1,
          poseRate: 24, poseSpeed: 5.6, dropoutMs: 420
        });
      } else if (A.tracker) E.trackerIdle(A.tracker, P.dt);
      smoothSignals(P.dt);
      if (!A.puppetCtx) A.puppetCtx = $('puppet').getContext('2d');
      const q = A.tracker ? A.tracker.q : [0, 0, 0, 1];
      const trust = A.tracker ? A.tracker.trust : 0;
      const fw = A.head.fw;
      const st = {
        t: P.t, dt: P.dt, ctx: A.puppetCtx, quality: A.quality,
        q,
        ox: trust > 0.02 ? A.tracker.ox * fw * 0.5 : 0,
        oy: trust > 0.02 ? A.tracker.oy * fw * 0.4 : 0,
        scale: trust > 0.02 ? A.tracker.scale : 1,
        focal: num('focal', 3.1),
        depthGain: num('dp', 1.3) - 1,
        deltaCanonical: composeDelta(P.dt),
        jawOpen: clamp01(nz(A.actions.j) * 1.4),
        jawMeshGain: 1,
        limits: chk('lim'), shade: num('fs', 0.35), hairGain: num('hs', 0.6),
        gaze: A.gaze, gazeGain: num('gg', 1.9), lid: A.lid, tongue: A.tongue,
        vis: num('tv', 0.4), bright: num('tb', 1.15), warmth: num('tw', 0),
        showPoints: chk('showPts'), showMesh: chk('showMesh'), origOnly: chk('showOrig'),
        points: A.pointList, dragIndex: A.drag >= 0 ? A.drag : -1, hoverIndex: A.hover >= 0 ? A.hover : -1,
        bodyAt: (A.pose.JI && A.pose.cur && A.pose.neu) ? bodyAt : null
      };
      if (renderFrame) {
        RR.sessionFrame(A.ses, st);
        if (P.t - P.hud > 0.25) { P.hud = P.t; updateHud(A.ses.lastDebug || {}); }
      } else {
        /* Headless harnesses can exercise the pose/expression/deformation
           chain without issuing thousands of Canvas2D triangle calls. */
        RR.deform(A.ses, st);
      }
    } else if (A.src) {
      const ctx = $('puppet').getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, $('puppet').width, $('puppet').height);
      ctx.drawImage(A.src, 0, 0);
    }
  }
  function updateHud(dbg) {
    const el = $('meters');
    if (!el) return;
    const Y = A.tracker && A.tracker.trust > 0.05 ? eulerFromQuat(A.tracker.q) : { yaw: 0, pitch: 0, roll: 0 };
    const deg = 180 / Math.PI;
    el.textContent = `${A.perf.fps} fps · ${dbg.tris || 0} tris${dbg.culled ? ' (' + dbg.culled + ' mirrored)' : ''}`
      + ` · yaw ${(Y.yaw * deg).toFixed(0)}° pitch ${(Y.pitch * deg).toFixed(0)}° roll ${(Y.roll * deg).toFixed(0)}°`
      + ` · mouth ${(dbg.mouth ? dbg.mouth.h * 100 : 0).toFixed(0)}% · jaw ${((A.ses.jaw || 0) * deg).toFixed(0)}°`
      + (A.camOn ? (A.tracker && A.tracker.trust > 0.35 ? ' · tracking' : ' · searching for a face') : ' · webcam off')
      + (A.micOn ? ' · mic on' : '');
  }

  /* body motion (rigid-ish shoulders) mapped into the photo's pixel scale */
  function bodyAt(x, y) {
    const JI = A.pose.JI, cur = A.pose.cur, neu = A.pose.neu;
    const B = num('bm', 1) * 0.9;
    if (B <= 0.001) return null;
    const pn0 = (neu[11] && neu[12]) ? Math.hypot(neu[11][0] - neu[12][0], neu[11][1] - neu[12][1]) || 1 : 1;
    const webcamFW = A.tracker && A.tracker.have
      ? Math.hypot(A.tracker.src[454][0] - A.tracker.src[234][0], A.tracker.src[454][1] - A.tracker.src[234][1]) : 0;
    const k = webcamFW > 1 ? A.head.fw / webcamFW : 1;
    const sw = Math.max(1, pn0 * 0.85);
    const br = Math.sin(A.perf.t * 1.6) * A.head.fh * 0.004 * B;
    let dx = 0, dy = 0, wsum = 0;
    for (let j = 11; j <= 16; j++) {
      if (!cur || !cur[j] || !neu[j] || cur[j][2] < 0.4 || !JI[j]) continue;
      const dist = Math.hypot(x - JI[j][0], y - JI[j][1]);
      const w = Math.exp(-(dist * dist) / (2 * sw * sw));
      dx += (cur[j][0] - neu[j][0]) * k * B * w;
      dy += ((cur[j][1] - neu[j][1]) * k * B + (j < 13 ? br : 0)) * w;
      wsum += w;
    }
    if (wsum < 1e-4) return [0, br];
    return [dx / (wsum + 0.25), dy / (wsum + 0.25)];
  }

  /* ------------------------------------------------------ interaction */
  function toCanvas(e) {
    const pc = $('puppet');
    const r = pc.getBoundingClientRect();
    return { x: (e.clientX - r.left) * pc.width / r.width, y: (e.clientY - r.top) * pc.height / r.height };
  }
  function resizeStage() {
    const pc = $('puppet');
    if (!pc || !A.src) return;
    const wrap = pc.parentElement;
    const maxW = wrap ? Math.max(280, wrap.clientWidth - 16) : 640;
    const scale = Math.min(1, maxW / A.src.width);
    pc.style.width = Math.round(A.src.width * scale) + 'px';
    pc.style.height = 'auto';
  }
  function nearestPoint(m, r) {
    if (!A.ses) return -1;
    let bi = -1, bd = (r || 34) ** 2;
    for (const p of A.pointList) {
      const i = p.v;
      const dx = A.ses.DST[i * 2] - m.x, dy = A.ses.DST[i * 2 + 1] - m.y;
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; bi = p.k; }
    }
    return bi;
  }
  /* screen px -> canonical px.  Solves the *actual* mapping the renderer
     uses,  dst = S2 + w·(proj(base + delta) − proj(base)),  with a numerical
     Jacobian — so the dot lands under the pointer even while the head is
     turned, the face is scaled and an expression is running.            */
  function screenToCanonical(i, target) {
    const S = A.ses, head = A.head;
    if (!S || !S.proj || !head) return [0, 0, 0];
    const w = A.mesh ? A.mesh.w[i] : 1;
    const base = head.P[i];
    const p0 = S.proj(base);
    const sx = S.S2[i * 2], sy = S.S2[i * 2 + 1];
    const ex = A.deltaExpr[i * 3], ey = A.deltaExpr[i * 3 + 1];
    const at = (x, y) => {
      const p = S.proj([base[0] + ex + x, base[1] + ey + y, base[2]]);
      return [sx + (p[0] - p0[0]) * w, sy + (p[1] - p0[1]) * w];
    };
    let x = A.sculpt[i * 2], y = A.sculpt[i * 2 + 1];
    const h = Math.max(0.25, head.fw * 0.002);
    for (let it = 0; it < 4; it++) {
      const c = at(x, y);
      const ax = at(x + h, y), ay = at(x, y + h);
      const j00 = (ax[0] - c[0]) / h, j01 = (ay[0] - c[0]) / h;
      const j10 = (ax[1] - c[1]) / h, j11 = (ay[1] - c[1]) / h;
      const r0 = target.x - c[0], r1 = target.y - c[1];
      if (Math.hypot(r0, r1) < 0.25) break;
      const det = j00 * j11 - j01 * j10;
      if (!isFinite(det) || Math.abs(det) < 1e-6) break;
      x += (j11 * r0 - j01 * r1) / det;
      y += (-j10 * r0 + j00 * r1) / det;
    }
    return [x, y, 0];
  }
  function groupOf(k) { const p = A.pointList && A.pointList.find(q => q.k === k); return p ? p.g : 'oval'; }
  function pushUndo() { A.bag.push({ s: A.sculpt.slice(), o: A.outBuf.slice() }); if (A.bag.length > 40) A.bag.shift(); }
  function undo() {
    const b = A.bag.pop();
    if (!b) return;
    A.sculpt.set(b.s); A.outBuf.set(b.o);
  }
  function setDragPoint(k, target) {
    const head = A.head;
    const d = screenToCanonical(k, target);
    const cap = (CAP[groupOf(k)] || 0.05) * head.fh * num('dr', 1);
    const L = Math.hypot(d[0], d[1]);
    if (L > cap) { d[0] *= cap / L; d[1] *= cap / L; }
    A.sculpt[k * 2] = d[0];
    A.sculpt[k * 2 + 1] = d[1];
    const partner = A.partIdx ? A.partIdx[k] : -1;
    if (chk('sym') && partner >= 0 && partner !== k) {
      A.sculpt[partner * 2] = -d[0];
      A.sculpt[partner * 2 + 1] = d[1];
    }
  }
  function insideFace(screen) {
    if (!A.ses || !screen || !Number.isFinite(screen.x) || !Number.isFinite(screen.y)) return false;
    const poly = E.CONTOUR.map(i => {
      const k = i * 2;
      return A.ses.frames > 0
        ? { x: A.ses.DST[k], y: A.ses.DST[k + 1] }
        : { x: A.ses.S2[k], y: A.ses.S2[k + 1] };
    });
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const a = poly[i], b = poly[j];
      const crosses = (a.y > screen.y) !== (b.y > screen.y);
      if (crosses && screen.x < (b.x - a.x) * (screen.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
    }
    return inside;
  }
  /* click-to-sculpt: only accept a real hit on the projected face contour;
     edits remain in canonical space, so the shape stays on the face as it turns. */
  function sculptShape(kind, screen) {
    if (!A.head || !insideFace(screen)) return false;
    const u = A.head.fw * 0.1;
    const move = (i, dx, dy) => {
      A.sculpt[i * 2] += dx;
      A.sculpt[i * 2 + 1] += dy;
      const p = chk('sym') && A.partIdx ? A.partIdx[i] : -1;
      if (p >= 0 && p !== i) { A.sculpt[p * 2] += -dx; A.sculpt[p * 2 + 1] += dy; }
    };
    pushUndo();
    if (kind === 'smile') { move(61, -u * .12, -u * .62); move(291, u * .12, -u * .62); move(13, 0, -u * .1); move(14, 0, u * .18); move(48, 0, -u * .16); move(278, 0, -u * .16); move(205, 0, -u * .1); move(425, 0, -u * .1); }
    else if (kind === 'frown') { move(61, -u * .08, u * .56); move(291, u * .08, u * .56); move(13, 0, u * .08); move(14, 0, -u * .1); }
    else if (kind === 'o') { move(61, u * .72, 0); move(291, -u * .72, 0); move(13, 0, -u * .46); move(14, 0, u * .72); move(0, 0, -u * .26); move(17, 0, u * .26); }
    else if (kind === 'kiss') { move(61, u * .3, 0); move(291, -u * .3, 0); move(13, 0, -u * .14); move(14, 0, u * .2); move(0, 0, -u * .18); move(17, 0, u * .22); }
    else if (kind === 'sad') { move(61, -u * .06, u * .46); move(291, u * .06, u * .46); move(70, 0, u * .26); move(300, 0, u * .26); }
    else if (kind === 'brows') { move(105, 0, -u * .62); move(334, 0, -u * .62); move(107, 0, -u * .5); move(336, 0, -u * .5); }
    else if (kind === 'wink') { move(159, 0, u * .5); move(145, 0, -u * .44); }
    else if (kind === 'puff') { move(205, u * .45, 0); move(425, -u * .45, 0); move(50, u * .3, 0); move(280, -u * .3, 0); }
    else if (kind === 'reset') { A.sculpt.fill(0); A.outBuf.fill(0); flashStatus('Face sculpt cleared.'); return true; }
    flashStatus(kind === 'o'
      ? 'Mouth O applied — drag the inner-mouth dots for fine control.'
      : (kind === 'brows' ? 'Brows raised.' : kind.charAt(0).toUpperCase() + kind.slice(1) + ' applied — drag dots to refine.'));
    return true;
  }

  /* ---------------------------------------------------------- controls */
  function wireControls() {
    const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
    on('camBtn', 'click', startCamera);
    on('file', 'change', e => { const f = e.target.files && e.target.files[0]; if (f) loadImage(f); });
    on('snapBtn', 'click', snapshotFromWebcam);
    on('photoBtn', 'click', savePhoto);
    on('recordBtn', 'click', toggleRecording);
    on('calBtn', 'click', neutralSet);
    on('resetBtn', 'click', () => { pushUndo(); A.sculpt.fill(0); A.outBuf.fill(0); flashStatus('Drags reset.'); });
    on('undoBtn', 'click', undo);
    on('micBtn', 'click', toggleMic);
    on('quality', 'change', e => { A.quality = parseInt(e.target.value, 10) || 0; });
    on('minCamBtn', 'click', () => {
      const card = $('webcamCard');
      if (!card) return;
      const mini = card.classList.toggle('minimized');
      const b = $('minCamBtn');
      if (b) b.textContent = mini ? 'Expand' : 'Minimize';
      resizeStage();
    });
    /* populate the two button racks */
    const presets = $('presets');
    if (presets && !presets.children.length) {
      ['Live webcam', ...Object.keys(E.PRESETS)].forEach(nm => {
        const b = document.createElement('button');
        b.type = 'button'; b.textContent = nm; b.dataset.preset = nm;
        if (nm === 'Live webcam') b.classList.add('on');
        presets.appendChild(b);
      });
      presets.addEventListener('click', e => {
        const b = e.target.closest('button[data-preset]');
        if (!b) return;
        A.preset = b.dataset.preset === 'Live webcam' ? 'live' : b.dataset.preset;
        [...presets.querySelectorAll('button')].forEach(x => x.classList.toggle('on', x === b));
        flashStatus(A.preset === 'live' ? 'Following your webcam.' : 'Preset: ' + b.dataset.preset);
      });
    }
    const sculpt = $('sculptTools');
    if (sculpt && !sculpt.children.length) {
      [['drag', 'Drag landmarks'], ['smile', 'Smile'], ['frown', 'Frown'], ['o', 'Mouth O'],
       ['kiss', 'Pucker'], ['brows', 'Brow raise'], ['sad', 'Sad'], ['wink', 'Wink'],
       ['puff', 'Puff cheeks'], ['reset', 'Clear sculpt']].forEach(([k, label]) => {
        const b = document.createElement('button');
        b.type = 'button'; b.textContent = label; b.dataset.sculpt = k;
        if (k === 'drag') b.classList.add('on');
        sculpt.appendChild(b);
      });
      sculpt.addEventListener('click', e => {
        const b = e.target.closest('button[data-sculpt]');
        if (!b) return;
        A.sculptMode = b.dataset.sculpt;
        [...sculpt.querySelectorAll('button')].forEach(x => x.classList.toggle('on', x === b));
        const pc = $('puppet');
        if (pc) pc.style.cursor = A.sculptMode === 'drag' ? 'crosshair' : 'pointer';
      });
    }
    const pc = $('puppet');
    if (pc) {
      pc.addEventListener('pointerdown', e => {
        if (!A.ses || chk('showOrig')) return;
        const m = toCanvas(e);
        const mode = A.sculptMode || 'drag';
        if (mode !== 'drag') { if (sculptShape(mode, m)) e.preventDefault(); return; }
        const k = nearestPoint(m);
        if (k < 0) return;
        pushUndo();
        A.drag = k;
        A.dragOff = { x: A.ses.DST[k * 2] - m.x, y: A.ses.DST[k * 2 + 1] - m.y };
        try { pc.setPointerCapture(e.pointerId); } catch (err) {}
        pc.style.cursor = 'grabbing';
        e.preventDefault();
      });
      pc.addEventListener('pointermove', e => {
        if (!A.ses) return;
        const m = toCanvas(e);
        if (A.drag >= 0) setDragPoint(A.drag, { x: m.x + A.dragOff.x, y: m.y + A.dragOff.y });
        else {
          A.hover = nearestPoint(m);
          pc.style.cursor = A.hover >= 0 ? 'grab' : (A.sculptMode && A.sculptMode !== 'drag' ? 'pointer' : 'crosshair');
          if (A.hover >= 0 && A.perf.t - (A._hovT || 0) > 0.6) {
            A._hovT = A.perf.t;
            setStatus(GROUP_NAME[groupOf(A.hover)] + ': drag to move · double-click to reset · Ctrl+Z to undo');
          }
        }
      });
      const stop = () => { A.drag = -1; pc.style.cursor = 'crosshair'; };
      pc.addEventListener('pointerup', stop);
      pc.addEventListener('pointercancel', stop);
      pc.addEventListener('dblclick', e => {
        const k = nearestPoint(toCanvas(e));
        if (k >= 0) { pushUndo(); A.sculpt[k * 2] = 0; A.sculpt[k * 2 + 1] = 0; }
      });
    }
    root.addEventListener('keydown', e => {
      if (!e.target || !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) {
        if (e.key === 't' || e.key === 'T') A.keyT = true;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); undo(); }
    });
    root.addEventListener('keyup', e => { if (e.key === 't' || e.key === 'T') A.keyT = false; });
    root.addEventListener('resize', resizeStage);
  }

  /* ----------------------------------------------------------- capture */
  function savePhoto() {
    const pc = $('puppet');
    if (!A.ses || !pc) return;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const a = document.createElement('a');
    a.download = `trill-face-puppet-${stamp}.png`;
    a.href = pc.toDataURL('image/png');
    a.click();
    flashStatus('Rendered frame saved as a PNG.');
  }
  function stopCapturedVideoTracks(stream) {
    /* captureStream() owns only the canvas video track.  Do not stop the
       borrowed microphone audio tracks added below — mic lip-sync may still
       be active for the next recording. */
    const tracks = stream && (stream.getVideoTracks
      ? stream.getVideoTracks()
      : (stream.getTracks ? stream.getTracks().filter(t => t.kind === 'video') : []));
    for (const tr of tracks || []) { try { tr.stop(); } catch (e) {} }
  }
  function toggleRecording() {
    const btn = $('recordBtn'), pc = $('puppet');
    if (A.rec) { try { A.rec.stop(); } catch (e) { flashStatus('Could not stop the recording: ' + e.message); } return; }
    if (!pc || !pc.captureStream || !root.MediaRecorder) { flashStatus('This browser cannot record the canvas.'); return; }
    let stream;
    try { stream = pc.captureStream(30); }
    catch (e) { flashStatus('Could not capture the canvas: ' + e.message); return; }
    if (!stream) { flashStatus('This browser cannot capture the canvas.'); return; }
    if (A.micOn && A.micStream) for (const tr of A.micStream.getAudioTracks()) { try { stream.addTrack(tr); } catch (e) {} }
    A.recStream = stream;
    A.chunks = [];
    let rec = null;
    for (const type of ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp9', 'video/webm']) {
      try { rec = new root.MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 8000000 }); break; } catch (e) {}
    }
    if (!rec) { try { rec = new root.MediaRecorder(stream); } catch (e) {
      stopCapturedVideoTracks(stream); A.recStream = null;
      flashStatus('Recording is not supported here.'); return;
    } }
    A.rec = rec;
    rec.ondataavailable = e => { if (e.data && e.data.size) A.chunks.push(e.data); };
    rec.onstop = () => {
      let saved = false;
      try {
        const blob = new Blob(A.chunks, { type: rec.mimeType || 'video/webm' });
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const a = document.createElement('a');
        a.download = `trill-face-puppet-${stamp}.webm`;
        a.href = URL.createObjectURL(blob);
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        saved = true;
      } catch (e) { flashStatus('The recording could not be saved: ' + e.message); }
      stopCapturedVideoTracks(stream);
      if (A.rec === rec) A.rec = null;
      if (A.recStream === stream) A.recStream = null;
      if (btn) btn.textContent = 'Record video';
      if (saved) flashStatus('Recording saved.');
    };
    try { rec.start(250); }
    catch (e) {
      A.rec = null; A.recStream = null; stopCapturedVideoTracks(stream);
      if (btn) btn.textContent = 'Record video';
      flashStatus('Could not start recording: ' + e.message); return;
    }
    if (btn) btn.textContent = 'Stop recording';
    flashStatus('Recording the rendered puppet…');
  }
  async function toggleMic() {
    const btn = $('micBtn');
    if (A.micOn) {
      A.micOn = false;
      if (A.micStream) A.micStream.getTracks().forEach(t => t.stop());
      A.micStream = null;
      if (btn) btn.textContent = 'Mic lip-sync';
      flashStatus('Microphone off.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      A.micStream = stream;
      const Ctx = root.AudioContext || root.webkitAudioContext;
      if (!A.audioCtx) A.audioCtx = new Ctx();
      if (A.audioCtx.state === 'suspended') await A.audioCtx.resume();
      const src = A.audioCtx.createMediaStreamSource(stream);
      A.analyser = A.audioCtx.createAnalyser();
      A.analyser.fftSize = 1024;
      A.analyser.smoothingTimeConstant = 0.55;
      A.freq = new Uint8Array(A.analyser.frequencyBinCount);
      src.connect(A.analyser);
      A.micOn = true;
      if (btn) btn.textContent = 'Mic on (click to stop)';
      flashStatus('Say something — the mouth follows your voice.');
    } catch (e) {
      flashStatus('Microphone unavailable: ' + e.message);
    }
  }

  /* --------------------------------------------------------- test hooks */
  const __test = {
    boot,
    buildFromLandmarks,
    setCamera: (lm, w, h, t) => {
      const ok = E.trackerPush(A.tracker, lm, w, h, t);
      if (ok && !A.tracker.neutral) E.trackerSetNeutral(A.tracker, w, h);
      return ok;
    },
    signals: readSignals,
    neutral: neutralSet,
    frame: (t, dt, renderFrame = true) => {
      if (dt) A.perf.last = t - dt * 1000;
      step(t, renderFrame !== false);
      return {
        DST: A.ses.DST, depth: A.ses.depth, dbg: A.ses.lastDebug, out: A.outBuf,
        jaw: A.ses.jaw, q: A.tracker ? A.tracker.q : null, sculpt: A.sculpt,
        trust: A.tracker ? A.tracker.trust : 0, actions: A.actions, gaze: A.gaze, lid: A.lid,
        perf: A.perf
      };
    },
    preset: nm => { A.preset = nm; },
    sculptShape: (kind, m) => sculptShape(kind, m || { x: 0, y: 0 }),
    lid: v => { A.lidRaw[0] = A.lidRaw[1] = v; },
    state: () => A,
    session: () => A.ses,
    head: () => A.head,
    mesh: () => A.mesh
  };

  root.PuppetApp = { boot, __test };
})(typeof globalThis !== 'undefined' ? globalThis : this);
