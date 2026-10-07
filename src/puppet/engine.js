/* =====================================================================
   FaceMirror · puppet engine  (DOM-free; needs only globalThis.FM + the
   landmark tables below).  Everything that decides *where* the photo's
   pixels go lives here, so tools/harness.mjs can drive it head-less.

   Pipeline
     detect  478 3D landmarks (photo)  -> lift into a canonical head space
     track   webcam 478 3D landmarks   -> weighted Kabsch rigid fit
     separate rigid pose from expression residual   (FaceVid2Vid split)
     retarget expression into the photo's canonical space, region gains
     rigid mandible (TMJ hinge) + linear-blend skinning for hair / neck
     rotate + weak-perspective project -> 2D targets for every mesh point

   Anti-flash invariants enforced here
     · every dynamic value is rate limited (slew / one-euro / envelope)
     · tracker dropouts decay toward the last good pose, never snap
     · per-landmark expression deltas are clamped, so one bad blend shape
       can never tear the mesh
     · booleans that gate drawing passes are hysteresis-free envelopes
       that live in [0,1] and are never compared against a raw threshold
   ===================================================================== */
(function (root) {
  'use strict';
  const FM = root.FM;
  if (!FM) throw new Error('FaceMirror engine: src/math.js must load first');
  const { clamp, clamp01, lerp, smoothstep, softLimit, slew, envelope, oneEuro, oneEuroRun, oneEuroReset,
          kabsch, v3, m3, quatFromMat, matFromQuat, quatFromYPR, eulerFromQuat, quatSlerp, quatAngle,
          quatSlew, quatFollow, quatNorm, ratioTracker, ratioTrack, splineClosed, polyArea, bounds } = FM;

  /* ===================================================================
     1. Landmark tables (MediaPipe Face Mesh, 478 points)
     =================================================================== */
  const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
  const CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
  const LEYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
  const REYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
  const LBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
  const RBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
  const LIPS_O = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
  const LIPS_I = [78, 82, 13, 312, 308, 317, 14, 87];
  const NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358, 219, 439, 218, 438, 115, 344,
    49, 279, 48, 278, 64, 294, 102, 331, 128, 357, 114, 343, 217, 437];
  /* upper / lower inner lip run (used to hang teeth and the tongue) */
  const UP_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
  const LO_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
  const LIDS = { L: { up: [159, 158, 160, 161, 157], lo: [145, 153, 144, 154, 163], ic: 133, oc: 33 },
                 R: { up: [386, 385, 387, 388, 384], lo: [374, 380, 373, 381, 390], ic: 362, oc: 263 } };
  const EYE_RING = { L: LEYE, R: REYE };
  /* 3D eyeball centres come from the mesh's inner-corner + iris ring */
  const IRIS = { L: { c: 468, ring: [469, 470, 471, 472] }, R: { c: 473, ring: [474, 475, 476, 477] } };
  /* rigid subset — forehead, brow ridge, nose bridge, temples.  These
     barely change with expression, so they carry the pose estimate.  */
  const RIGID = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 9, 8, 107, 336, 296, 334, 293, 300, 285, 276, 283, 282, 295,
    55, 65, 52, 53, 46, 285, 300, 293, 334, 296, 336, 70, 63, 105, 66, 107, 9, 151, 108, 337, 299, 333,
    298, 301, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377];
  const RIGID_W = i => {
    if (i <= 467) {
      if (i === 152 || i === 148 || i === 176 || i === 149 || i === 150 || i === 136 || i === 377 || i === 400 ||
          i === 378 || i === 379 || i === 365 || i === 397 || i === 288 || i === 361 || i === 323 || i === 454) return 0.15; // jaw line
      if (LIPS_I.includes(i) || LIPS_O.includes(i) || i === 13 || i === 14 || i === 17 || i === 0) return 0.05;
      if (i === 468 || i >= 469) return 0;
    }
    return 1;
  };
  /* landmarks whose depth the detector is least sure about (temples, ears,
     jaw, hairline) — these lean on the analytic ellipsoid prior instead  */
  const SIDEY = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
    361, 323, 454, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356];

  const EXPR_GAIN = i => {
    if (i >= 468) return 0;                              // irises are rigid
    if (LIPS_I.includes(i) || LIPS_O.includes(i)) return 1.0;
    if (UP_LIP.includes(i) || LO_LIP.includes(i)) return 1.0;
    /* Webcam residuals are a detail layer, not a second full face warp.
       MediaPipe's cheek / nose coordinates carry enough identity and lighting
       noise to turn a sharp portrait into a field of triangular patches. */
    if (LIDS.L.up.includes(i) || LIDS.L.lo.includes(i) || LIDS.R.up.includes(i) || LIDS.R.lo.includes(i)) return 0.72;
    if (LEYE.includes(i) || REYE.includes(i)) return 0.36;
    if (LBROW.includes(i) || RBROW.includes(i)) return 0.42;
    return 0;                                             // cheeks, nose, jaw: actions only
  };
  /* per-landmark anatomical cap for a single expression delta, in face widths */
  const EXPR_CAP = i => {
    if (LIPS_I.includes(i) || LIPS_O.includes(i)) return 0.16;
    if (UP_LIP.includes(i) || LO_LIP.includes(i)) return 0.13;
    if (LBROW.includes(i) || RBROW.includes(i)) return 0.12;
    return 0.07;
  };

  /* ===================================================================
     2. Canonical head space
     A photo is 2.5D: MediaPipe's z is relative and noisy.  We blend it
     with an analytic head prior (ellipsoid + nose wedge) and then smooth
     the depth over the mesh, which keeps the surface C1 enough that the
     triangle pass cannot fold on a turn.
     =================================================================== */
  /* MediaPipe hands back NormalizedLandmark objects ({x,y,z}); some of our
     own tests feed [x,y,z] tuples.  Accept both, everywhere.            */
  const lx = p => (p.x !== undefined ? p.x : p[0]);
  const ly = p => (p.y !== undefined ? p.y : p[1]);
  const lz = p => (p.z !== undefined ? p.z : p[2]);

  function faceCenterOf(P) {
    let cx = 0, cy = 0, n = 0;
    for (const i of [4, 168, 6, 197, 152, 10, 33, 133, 362, 263]) {
      cx += P[i][0]; cy += P[i][1]; n++;
    }
    return [cx / n, cy / n];
  }
  function buildCanonical(lm, W, H, opts) {
    const o = opts || {};
    const P = new Array(478);
    for (let i = 0; i < 478; i++) {
      const p = lm[i];
      if (!p) return null;
      const z = lz(p);
      P[i] = [lx(p) * W, ly(p) * H, -(isFinite(z) ? z : 0) * W];   // +z = toward the viewer
    }
    const fw = Math.hypot(P[454][0] - P[234][0], P[454][1] - P[234][1]) || W * 0.2;
    const fh = Math.hypot(P[152][0] - P[10][0], P[152][1] - P[10][1]) || fw * 1.35;
    const mid = [(P[234][0] + P[454][0]) / 2, (P[10][1] + P[152][1]) / 2];
    const ctr = faceCenterOf(P);
    /* --- analytic depth prior: an ellipsoid whose pole is the nose --- */
    const rx = fw * 0.78, ry = fh * 0.92, rz = fw * 0.62;
    const noseY = [4, 5, 6].reduce((a, i) => a + P[i][1], 0) / 3;
    const prior = i => {
      const u = (P[i][0] - ctr[0]) / rx, v = (P[i][1] - (ctr[1] + fh * 0.06)) / ry;
      const q = 1 - u * u - v * v;
      let z = q > 0 ? rz * Math.sqrt(q) : 0;
      /* nose ridge: raise the midline from noseY up to the brow */
      const nearMid = 1 - Math.min(1, Math.abs(P[i][0] - mid[0]) / (fw * 0.34));
      const up = clamp01((noseY + fh * 0.34 - P[i][1]) / (fh * 0.5));
      z += rz * 0.30 * nearMid * up * up;
      return z;
    };
    /* --- blend detector depth with the prior, then Laplacian-smooth --- */
    const sideSet = new Set(SIDEY);
    const z = new Float32Array(478);
    let zSum = 0;
    for (let i = 0; i < 478; i++) zSum += P[i][2];
    const zMean = zSum / 478;
    for (let i = 0; i < 478; i++) {
      const w = sideSet.has(i) ? (o.priorSide === undefined ? 0.62 : o.priorSide) : (o.priorMid === undefined ? 0.22 : o.priorMid);
      z[i] = lerp(P[i][2] - zMean, prior(i), w);
    }
    /* neighbour graph from the official tessellation, if present */
    const adj = o.adjacency;
    if (adj) {
      const tmp = new Float32Array(478);
      for (let it = 0; it < (o.smoothIters === undefined ? 6 : o.smoothIters); it++) {
        for (let i = 0; i < 478; i++) {
          const nb = adj[i];
          if (!nb || !nb.length) { tmp[i] = z[i]; continue; }
          let s = 0;
          for (let k = 0; k < nb.length; k++) s += z[nb[k]];
          tmp[i] = z[i] * 0.45 + (s / nb.length) * 0.55;
        }
        z.set(tmp);
      }
    }
    for (let i = 0; i < 478; i++) P[i][2] = z[i];
    return { P, fw, fh, mid, ctr, noseY, W, H };
  }
  /* convenience: head-local normalised coords of a canonical point */
  function localUV(head, p) {
    return [(p[0] - head.ctr[0]) / head.fw, (p[1] - head.ctr[1]) / head.fh, p[2] / head.fw];
  }

  /* ===================================================================
     3. Mesh: face + hair rings + body skirt, one static triangulation
     =================================================================== */
  function buildMesh(head, opts) {
    const o = opts || {};
    const P = head.P.map(p => p.slice());          // canonical 3D
    const n0 = P.length;                           // 478
    const w = [];                                  // shell weight per point
    const kind = [];                               // 0 face, 1 ring, 2 skirt
    for (let i = 0; i < n0; i++) { w.push(1); kind.push(0); }
    /* ---- hair rings: a shell that wraps around the skull -------------
       ring 1 hugs the head (moves with it, hair lag), ring 3 sits well
       behind the head centre so a 45° turn reveals a plausible cranium
       instead of a paper edge.                                        */
    const ringDefs = [[1.16, 0.92, 0.10], [1.52, 0.45, 0.42], [2.10, 0.16, 0.95]];
    const rings = [];
    for (let r = 0; r < ringDefs.length; r++) {
      const [f, wr, back] = ringDefs[r];
      const idxs = [];
      for (const c of CONTOUR) {
        const p = head.P[c];
        let x = head.ctr[0] + (p[0] - head.ctr[0]) * f;
        let y = head.ctr[1] + (p[1] - head.ctr[1]) * f;
        const R = Math.max(head.fw, head.fh) * 0.5;
        let z = lerp(p[2], -back * R, clamp01((f - 1) / 1.1));
        /* keep the top of each ring a little forward (skull, not sphere) */
        const up = clamp01((head.ctr[1] - y) / (head.fh * 0.9) + 0.5);
        z += R * 0.18 * (1 - up) * 0;
        x = clamp(x, 0, head.W);
        y = clamp(y, 0, head.H);
        const id = P.length;
        P.push([x, y, z]);
        w.push(wr);
        kind.push(1);
        idxs.push(id);
      }
      rings.push(idxs);
    }
    /* ---- body skirt: pinned background banner below the chin --------- */
    const skirt = [];
    const chin = head.P[152], chinY = chin[1], botY = head.H;
    if (botY > chinY + 8) {
      for (let r = 0; r < 2; r++) {
        const y = chinY + head.fh * 0.35 + (botY - (chinY + head.fh * 0.35)) * (r ? 0.75 : 0.25);
        for (let q = 0; q < 7; q++) {
          const x = head.W * (q + 0.5) / 7;
          const im = P.length; P.push([x, y, -head.fw * 0.2]); w.push(0); kind.push(2); skirt.push(im);
        }
      }
    }
    /* ---- corners: pin the frame so nothing detaches ------------------ */
    const corners = [];
    const W = head.W, H = head.H;
    for (const [x, y] of [[0, 0], [W, 0], [W, H], [0, H], [W / 2, 0], [W / 2, H], [0, H / 2], [W, H / 2]]) {
      const im = P.length; P.push([x, y, -head.fw * 0.4]); w.push(0); kind.push(2); corners.push(im);
    }
    /* ---- triangulate (Delaunator over the source-plane layout) ------- */
    const D = o.delaunay || root.Delaunator;
    if (!D) throw new Error('engine: Delaunator missing');
    const tri = D.from(P.map(p => [p[0], p[1]])).triangles;
    const base = [];
    for (let i = 0; i < tri.length; i += 3) base.push(tri[i], tri[i + 1], tri[i + 2]);
    /* ---- official face tessellation -> triangles, drawn over the base.
       MediaPipe ships the mesh as an edge list, so triangles are the
       cliques of three mutually connected landmarks (as in the original
       face-mesh demo).  Emitting each triangle once keeps the pass
       deterministic, which matters for painter ordering.              */
    const face = [];
    if (o.tessellation) {
      const adjSets = Array.from({ length: 478 }, () => new Set());
      for (const c of o.tessellation) {
        if (c.start < 478 && c.end < 478) { adjSets[c.start].add(c.end); adjSets[c.end].add(c.start); }
      }
      for (let a = 0; a < 478; a++) {
        for (const b of adjSets[a]) {
          if (b <= a) continue;
          for (const c of adjSets[b]) {
            if (c <= b) continue;
            if (adjSets[a].has(c)) face.push(a, b, c);
          }
        }
      }
    }
    /* adjacency for the depth smoother + nearest-neighbour drag falloff */
    const nbr = Array.from({ length: P.length }, () => new Set());
    for (let i = 0; i < base.length; i += 3) {
      const a = base[i], b = base[i + 1], c = base[i + 2];
      nbr[a].add(b); nbr[b].add(a); nbr[b].add(c); nbr[c].add(b); nbr[c].add(a); nbr[a].add(c);
    }
    /* ring index -> face contour index (for the silhouette + hair lag) */
    const ringOf = [];
    for (let r = 0; r < rings.length; r++) for (let k = 0; k < rings[r].length; k++) ringOf.push({ ring: r, contour: k });
    return { P, w: Float32Array.from(w), kind: Uint8Array.from(kind), rings, skirt, corners, base, face, nbr, ringOf, n: P.length };
  }

  /* ===================================================================
     4. Expression field — parametric deltas in canonical space
     Defined in head-local coordinates, so a smile rides the face while
     the head turns instead of smearing across the screen.
     =================================================================== */
  const ACTIONS = ['sL', 'sR', 'j', 'bu', 'buR', 'bi', 'bd', 'eo', 'sq', 'dr', 'pk', 'fr', 'nz', 'ur',
                   'st', 'bL', 'bR', 'tg', 'gx', 'gy', 'ck', 'wv'];
  const zeroActions = () => { const o = {}; for (const k of ACTIONS) o[k] = 0; return o; };
  const PRESETS = {
    'Neutral': {},
    'Smile': { sL: 0.55, sR: 0.55, j: 0.12, sq: 0.10, ck: 0.25 },
    'Big smile': { sL: 0.92, sR: 0.92, j: 0.16, sq: 0.30, bu: 0.12, ck: 0.55 },
    'Laugh': { sL: 1, sR: 1, j: 0.52, sq: 0.5, bu: 0.18, nz: 0.18, ck: 0.7 },
    'Smirk': { sR: 0.78, bu: 0.10, ck: 0.2 },
    'Sad': { fr: 0.72, bi: 0.9, dr: 0.3, j: 0.04, gy: 0.8 },
    'Angry': { bd: 0.9, fr: 0.5, sq: 0.3, nz: 0.35, ur: 0.15 },
    'Surprised': { bu: 0.9, eo: 0.85, j: 0.5 },
    'Fear': { bu: 0.55, bi: 0.7, eo: 0.75, j: 0.28, st: 0.7 },
    'Disgust': { nz: 0.9, ur: 0.7, bd: 0.35, fr: 0.35, sq: 0.25 },
    'Skeptical': { buR: 0.8, bd: 0.25, sR: 0.28, dr: 0.15 },
    'Sleepy': { dr: 0.68, j: 0.1 },
    'Wink': { bL: 1, sL: 0.45, sR: 0.45, sq: 0.1 },
    'Kiss': { pk: 1, bu: 0.1 },
    'Tongue out': { tg: 1, j: 0.3 },
    'Look left': { gx: -1 }, 'Look right': { gx: 1 }, 'Look up': { gy: -1 }, 'Look down': { gy: 1 },
    'Talking': null
  };
  /* Deterministic pseudo-speech for the preset "Talking" (no audio) */
  function talkAt(t) {
    const s = (k, ph) => Math.abs(Math.sin(k * 127.1 + ph) * 43758.5453) % 1;
    const e = clamp01(Math.sin(t * 0.72) * 3 + 1.15);
    const g = t * 4.3, k = Math.floor(g), f = g - k;
    const o = Math.pow(Math.sin(Math.PI * f), 0.7) * e;
    const h = s(k, 0), h2 = s(k, 3.1);
    const round = h > 0.68;
    return {
      j: o * (0.10 + 0.42 * h),                       // jaw open, vowel height
      pk: round ? o * 0.85 : o * 0.12,                // rounded vowels
      st: h2 < 0.26 ? o * 0.6 : 0,                    // tongue tip for dentals
      sL: 0.10 + 0.12 * h, sR: 0.10 + 0.12 * h,       // mouth corners spread
      bu: e * 0.10 * h2, ck: 0.2
    };
  }
  /* Build the per-landmark 3D expression delta (canonical px units). */
  function expressionDelta(head, A, out) {
    const d = out || new Float32Array(478 * 3);
    d.fill(0);
    const { fw, fh, ctr, mid, noseY } = head;
    const cx = mid[0], yMouth = (head.P[13][1] + head.P[14][1]) / 2;
    const eyeY = (head.P[159][1] + head.P[386][1]) / 2;
    const SD = v => (v >= 0 ? 1 : -1);
    for (let i = 0; i < 478; i++) {
      const p = head.P[i];
      const u = clamp((p[0] - cx) / (fw * 0.5), -1.7, 1.7);
      const au = Math.min(1, Math.abs(u));
      const side = u < 0 ? 0 : 1;
      const sm = side ? A.sR : A.sL;
      const v = (p[1] - eyeY) / fh;
      let dx = 0, dy = 0, dz = 0;
      /* ---- mouth ---------------------------------------------------- */
      const mouthMask = Math.exp(-(((p[0] - cx) / (fw * 0.55)) ** 2) - ((p[1] - yMouth) / (fh * 0.22)) ** 2);
      if (mouthMask > 0.004) {
        const lift = clamp01((v - 0.05) / 0.4);          // 0 at the vermilion, 1 at the chin
        const up = 1 - lift;
        const smileAmt = sm * mouthMask;
        dy += (-smileAmt * fh * 0.20 * au * au * up - A.fr * fh * 0.16 * au * au * up);
        dx += u * (smileAmt * 0.09 + A.st * 0.14) * fw * au * mouthMask;
        dy += A.ur * fh * 0.10 * up * (1 - au * 0.5);
        dz += (A.eo * 0.35 + A.pk * 0.55) * fw * 0.06 * mouthMask;
        /* pucker: lips gather toward the mouth axis and curl outward */
        const pk = A.pk * mouthMask;
        dx -= u * fw * 0.30 * pk * 0.55;
        dy -= 0;
      }
      /* ---- cheeks / lower lids -------------------------------------- */
      const cheek = Math.exp(-(((Math.abs(p[0] - cx) - fw * 0.34) / (fw * 0.3)) ** 2) -
                             (((p[1] - (eyeY + fh * 0.30)) / (fh * 0.22)) ** 2));
      if (cheek > 0.004) {
        dy -= sm * fh * 0.07 * cheek + A.ck * fh * 0.05 * cheek;
        dx += SD(p[0] - cx) * sm * fw * 0.035 * cheek;
        dz += A.ck * fw * 0.05 * cheek;
      }
      /* ---- brows ---------------------------------------------------- */
      if (LBROW.includes(i) || RBROW.includes(i)) {
        const inner = Math.abs(p[0] - cx) < fw * 0.14;
        dy -= (A.bu + (side ? A.buR : 0)) * fh * 0.075;
        dy -= inner ? A.bi * fh * 0.075 : -A.bi * fh * 0.02;
        dy += A.bd * fh * (inner ? 0.06 : 0.035);
        dx += inner ? -SD(p[0] - cx) * A.bd * fh * 0.024 : 0;
      }
      /* ---- nose scrunch / disgust ----------------------------------- */
      if (NOSE.includes(i)) {
        dy -= A.nz * fh * 0.03 * (Math.abs(p[0] - cx) < fw * 0.3 ? 1 : 0.5);
        dx += (p[0] > cx ? 1 : -1) * A.ur * fw * 0.02;
      }
      /* ---- eyes: squint narrows the aperture ------------------------ */
      const eyeMask = Math.exp(-(((Math.abs(p[0] - cx) - fw * 0.24) / (fw * 0.17)) ** 2) - ((p[1] - eyeY) / (fh * 0.09)) ** 2);
      if (eyeMask > 0.004) {
        const isUpper = p[1] < eyeY;
        dy += (isUpper ? 1 : -0.55) * A.sq * fh * 0.05 * eyeMask;
      }
      const g3 = i * 3;
      d[g3] += dx; d[g3 + 1] += dy; d[g3 + 2] += dz;
    }
    return d;
  }
  /* mouth-shape readout from the current photo mesh (drives teeth/tongue) */
  function mouthState(head, D, idx) {
    const g = (a, b) => Math.hypot(D[a].x - D[b].x, D[a].y - D[b].y);
    const mw = g(61, 291) || head.fw * 0.4;
    const gap = g(13, 14);
    return { mw, gap, open: clamp01((gap / mw - 0.045) / 0.30), wide: clamp01((mw / head.fw - 0.30) / 0.25) };
  }

  /* ===================================================================
     5. Expression transfer from the webcam (FaceVid2Vid style split)
     =================================================================== */
  function makeTransfer(head) {
    return { residual: new Float32Array(478 * 3), mix: new Float32Array(478) };
  }
  /* residual (canonical units) -> photo delta, region gains + caps.
     `out` lets the caller reuse one buffer (no per-frame allocation).  */
  function retargetResidual(head, T, resid, gain, out) {
    const d = out || new Float32Array(478 * 3);
    d.fill(0);
    const cap0 = head.fw * 0.25;
    for (let i = 0; i < 478; i++) {
      const g = EXPR_GAIN(i) * gain;
      if (g <= 0) continue;
      const g3 = i * 3;
      let x = resid[g3] * g, y = resid[g3 + 1] * g, z = resid[g3 + 2] * g;
      const n = Math.hypot(x, y, z);
      const cap = Math.max(cap0, head.fw * EXPR_CAP(i) * 3);
      if (n > cap) { const k = cap / n; x *= k; y *= k; z *= k; }
      d[g3] = x; d[g3 + 1] = y; d[g3 + 2] = z;
    }
    return d;
  }
  /* expression energy of a residual — used for the tracker trust gate */
  function residualEnergy(resid, n) {
    let s = 0;
    for (let i = 0; i < 478; i++) {
      if (EXPR_GAIN(i) <= 0) continue;
      const g3 = i * 3;
      s += resid[g3] * resid[g3] + resid[g3 + 1] * resid[g3 + 1];
    }
    return Math.sqrt(s / (n || 1));
  }

  /* ===================================================================
     6. Rigid mandible (TMJ hinge) + linear blend skinning
     The jaw, lower teeth, tongue and chin share one bone, so an open
     mouth never looks like a vertical smear.
     =================================================================== */
  function tmjAxis(head) {
    /* hinge roughly 6% of face width above the mouth, passing through the
       ear canal line — the classic TMJ approximation                     */
    const a = head.P[234], b = head.P[454];
    const y = (head.P[13][1] + head.P[14][1]) / 2 - head.fh * 0.10;
    const zc = (head.P[234][2] + head.P[454][2]) / 2 - head.fw * 0.18;
    return { p: [(a[0] + b[0]) / 2, y, zc], dir: v3.norm([b[0] - a[0], (b[1] - a[1]) * 0.35, (b[2] - a[2]) * 0.9]) };
  }
  const jawWeight = (i, head) => {
    if (i >= 468) return 0;
    const upper = UP_LIP.includes(i), lower = LO_LIP.includes(i);
    if (upper || lower || LIPS_O.includes(i) || LIPS_I.includes(i)) {
      if (upper && lower) return 0.42;               // mouth corners bridge both lips
      if (upper) return 0.06;                        // maxilla / upper lip stays nearly rigid
      if (lower) return 0.94;                        // lower lip follows the mandible
      const midY = (head.P[13][1] + head.P[14][1]) * 0.5;
      return clamp(0.5 + (head.P[i][1] - midY) / (head.fh * 0.02) * 0.45, 0.05, 0.95);
    }
    if (i === 152 || i === 148 || i === 176 || i === 149 || i === 150 || i === 136 || i === 377 || i === 400 ||
        i === 378 || i === 379 || i === 365 || i === 397 || i === 288 || i === 361 || i === 323 || i === 454 ||
        i === 17 || i === 0 || i === 18 || i === 200 || i === 199 || i === 175) return 1;
    return 0.15;
  };
  function applyJaw(head, pts, angle, axis) {
    if (Math.abs(angle) < 1e-4) return;
    const R = m3.fromAxisAngle(axis.dir, angle);
    for (let i = 0; i < 478; i++) {
      const w = jawWeight(i, head);
      if (w <= 0) continue;
      const rel = v3.sub(head.P[i], axis.p);
      const rot = m3.mv(R, rel);
      const p = pts[i];
      p[0] += (axis.p[0] + rot[0] - head.P[i][0]) * w;
      p[1] += (axis.p[1] + rot[1] - head.P[i][1]) * w;
      p[2] += (axis.p[2] + rot[2] - head.P[i][2]) * w;
    }
  }

  /* ===================================================================
     7. Tracker: 478 3D landmarks -> filtered pose + expression residual
     =================================================================== */
  function createTracker(head, opts) {
    const o = opts || {};
    const S = {
      src: new Array(478),            // smoothed webcam canonical points
      neutral: null,                  // webcam neutral (canonical, faceWidth = 1)
      eo: new Array(478 * 3),         // one-euro filters per coordinate
      knee: o.knee || 1.0,
      q: [0, 0, 0, 1],                // smoothed head rotation
      qTarget: [0, 0, 0, 1],
      have: false,
      lastSeen: -1e9,
      trust: 0,                       // 0..1 tracker confidence envelope
      residual: new Float32Array(478 * 3),
      residPrev: new Float32Array(478 * 3),
      scale: 1, ox: 0, oy: 0,
      ypr: { yaw: 0, pitch: 0, roll: 0 },
      yprF: {
        yaw: oneEuro(o.yawFc || 1.5, o.yawBeta || 0.055, 1),
        pitch: oneEuro(o.pitchFc || 1.2, o.pitchBeta || 0.045, 1),
        roll: oneEuro(o.rollFc || 1.1, o.rollBeta || 0.04, 1)
      },
      tFx: oneEuro(2.2, 0.09, 1), tFy: oneEuro(2.2, 0.09, 1), sFx: oneEuro(0.9, 0.03, 1)
    };
    for (let i = 0; i < S.eo.length; i++) S.eo[i] = oneEuro(i >= 468 * 3 ? 2.0 : 2.6, 0.03, 1);
    S.fast = new Set();                                  // lips get a faster cutoff
    for (const i of LIPS_I) { S.fast.add(i * 3); S.fast.add(i * 3 + 1); S.fast.add(i * 3 + 2); }
    for (const i of LIPS_O) { S.fast.add(i * 3); S.fast.add(i * 3 + 1); S.fast.add(i * 3 + 2); }
    return S;
  }
  /* convert raw normalised landmarks into canonical px + one-euro smooth */
  function trackerPush(S, lm, W, H, t) {
    if (!lm || lm.length < 478) return false;
    if (!S.src[0]) for (let i = 0; i < 478; i++) S.src[i] = [0, 0, 0];
    const k = S.knee;
    for (let i = 0; i < 478; i++) {
      const p = lm[i];
      if (!p) return false;
      const X = lx(p), Y = ly(p), Z = lz(p);
      if (!isFinite(X) || !isFinite(Y)) return false;
      const raw = [X * W, Y * H, -(isFinite(Z) ? Z : 0) * W];
      const g3 = i * 3;
      S.src[i][0] = oneEuroRun(S.eo[g3], raw[0] / k, t);
      S.src[i][1] = oneEuroRun(S.eo[g3 + 1], raw[1] / k, t);
      /* lips move fast while speaking: a faster filter keeps consonants */
      S.src[i][2] = oneEuroRun(S.eo[g3 + 2], raw[2] / k, t);
    }
    S.lastSeen = t;
    S.have = true;
    return true;
  }
  function trackerSetNeutral(S, W, H) {
    const ctr = faceCenterOf(S.src);
    const fw = Math.hypot(S.src[454][0] - S.src[234][0], S.src[454][1] - S.src[234][1]) || (W || 1) * 0.2;
    const k = fw || 1;
    S.neutral = S.src.map(p => [(p[0] - ctr[0]) / k, (p[1] - ctr[1]) / k, p[2] / k]);
    S.knee = k;
    S.neutralCtr = ctr;
    S.neutralAt = performance_now();
    /* A new neutral pose invalidates the previous expression baseline. */
    S.residual.fill(0);
    S.residPrev.fill(0);
  }
  function performance_now() { return (root.performance && root.performance.now) ? root.performance.now() : Date.now(); }
  /* One tracker frame. Returns true when a fresh pose was produced. */
  function trackerUpdate(S, dt, cfg) {
    const c = cfg || {};
    const t = performance_now();
    const live = S.have && (t - S.lastSeen) < (c.dropoutMs || 420);
    /* trust envelope: ramps in fast, decays over ~1 s (never snaps) */
    S.trust = envelope(S.trust, live ? 1 : 0, dt, 9, 2.2);
    if (!S.neutral) return false;
    /* No fresh detection: decay pose/expression toward neutral instead of
       freezing the last frame (a frozen pose reads as a hard cut).      */
    if (!live) { trackerIdle(S, dt); return false; }
    const n = S.neutral;
    const Cf = S.src.map(p => [p[0], p[1], p[2]]);
    const src = [], dst = [], wt = [];
    for (const i of RIGID) {
      const w = RIGID_W(i);
      if (w <= 0) continue;
      src.push(n[i]); dst.push([Cf[i][0], Cf[i][1], Cf[i][2]]); wt.push(w);
    }
    const f = kabsch(src, dst, wt);
    if (!f) return false;
    /* ---- expression residual = rigid-removed difference --------------- */
    const Ri = m3.t(f.R);
    const k = c.exprGain === undefined ? 1 : c.exprGain;
    /* The rigid fit removes most pose, but the remaining per-landmark
       residual still contains detector noise.  Sending that raw field to the
       triangle warp makes a photograph get re-sampled differently on every
       webcam frame, which looks like blur and shimmer.  Keep an explicit
       filtered residual (the buffer existed before but was never used), with
       a small pixel-scale deadband and a quicker response for lips. */
    for (let i = 0; i < 478; i++) {
      const a = v3.sub(Cf[i], f.cb);
      const b = m3.mv(Ri, a);
      const g3 = i * 3;
      const raw = [
        (b[0] / f.s - n[i][0]) * k,
        (b[1] / f.s - n[i][1]) * k,
        (b[2] / f.s - n[i][2]) * k
      ];
      const fast = LIPS_I.includes(i) || LIPS_O.includes(i) || UP_LIP.includes(i) || LO_LIP.includes(i);
      const rate = fast ? 15 : 8;
      const dead = fast ? 0.0012 : 0.0022; // roughly 0.3–0.6 px at a 257 px face
      const alpha = 1 - Math.exp(-rate * Math.max(dt || 1 / 60, 1 / 240));
      for (let ax = 0; ax < 3; ax++) {
        const target = Math.abs(raw[ax]) < dead ? 0 : raw[ax];
        const prev = S.residPrev[g3 + ax];
        const next = prev + (target - prev) * alpha;
        S.residPrev[g3 + ax] = next;
        S.residual[g3 + ax] = next;
      }
    }
    /* ---- pose -> yaw/pitch/roll, per-axis gain, soft limits ---------- */
    const e = eulerFromQuat(quatFromMat(f.R));
    const rad = 180 / Math.PI;
    const gY = (c.yawGain === undefined ? 1 : c.yawGain);
    const gP = (c.pitchGain === undefined ? 1 : c.pitchGain);
    const gR = (c.rollGain === undefined ? 1 : c.rollGain);
    const limY = (c.yawLimit === undefined ? 48 : c.yawLimit) / rad;
    const limP = (c.pitchLimit === undefined ? 26 : c.pitchLimit) / rad;
    const limR = (c.rollLimit === undefined ? 22 : c.rollLimit) / rad;
    const yaw = softLimit(e.yaw * gY * rad, limY * rad) / rad;
    const pitch = softLimit(e.pitch * gP * rad, limP * rad) / rad;
    const roll = softLimit(e.roll * gR * rad, limR * rad) / rad;
    S.ypr.yaw = oneEuroRun(S.yprF.yaw, yaw, t);
    S.ypr.pitch = oneEuroRun(S.yprF.pitch, pitch, t);
    S.ypr.roll = oneEuroRun(S.yprF.roll, roll, t);
    S.qTarget = quatFromYPR(S.ypr.yaw, S.ypr.pitch, S.ypr.roll);
    /* slew-limited slerp: bounds angular acceleration as well as speed */
    S.q = quatSlew(S.q, quatFollow(S.q, S.qTarget, dt, c.poseRate || 26), dt, (c.poseSpeed || 5.4));
    /* translation + scale ride the same trust envelope */
    const tx = (f.cb[0] - f.ca[0]) / (S.knee || 1);
    const ty = (f.cb[1] - f.ca[1]) / (S.knee || 1);
    S.ox = oneEuroRun(S.tFx, tx, t) * S.trust;
    S.oy = oneEuroRun(S.tFy, ty, t) * S.trust;
    S.scale = 1 + (oneEuroRun(S.sFx, (f.s - 1) * (c.dolly === undefined ? 0.5 : c.dolly), t)) * S.trust;
    return true;
  }
  /* when the tracker has no face, fade the pose to neutral instead of
     freezing it (no visible pop when detection blinks)                  */
  function trackerIdle(S, dt) {
    S.q = quatSlew(S.q, quatSlerp(S.q, [0, 0, 0, 1], clamp01(dt * 0.9)), dt, 4);
    S.ox = slew(S.ox, 0, dt, 0.5, 0.5);
    S.oy = slew(S.oy, 0, dt, 0.5, 0.5);
    S.scale = slew(S.scale, 1, dt, 0.35, 0.35);
    for (let i = 0; i < S.residual.length; i++) {
      S.residual[i] = slew(S.residual[i], 0, dt, 0.6, 0.6);
      S.residPrev[i] = S.residual[i];
    }
  }

  /* ===================================================================
     8. Viseme / audio driver  (mic amplitude + spectral shape)
     A miniature of the audio->viseme front ends (SadTalker/Wav2Lip use a
     learned encoder): we band-limit the spectrum into four viseme
     classes and drive jaw, pucker, spread and tongue with envelopes.
     =================================================================== */
  function createViseme() {
    return {
      env: 0, open: 0, round: 0, wide: 0, tip: 0, tone: 0,
      bands: [0, 0, 0, 0], t: 0, active: false
    };
  }
  function visemeUpdate(V, spec, rms, dt) {
    if (!spec || !spec.length) return V;
    const n = spec.length;
    const band = (f0, f1) => {
      const i0 = Math.max(0, Math.floor(f0 * n / 24000)), i1 = Math.min(n, Math.ceil(f1 * n / 24000));
      let s = 0;
      for (let i = i0; i < i1; i++) s += spec[i];
      return s / Math.max(1, i1 - i0);
    };
    const lo = band(120, 700);      // F1 zone -> jaw / open vowels
    const mid = band(700, 1800);    // F2 zone -> spread vs round
    const hi = band(1800, 5000);    // sibilants -> tongue tip / spread
    const tot = lo + mid + hi + 1e-6;
    const lvl = clamp01((rms - 0.006) * 11);
    V.env = envelope(V.env, lvl, dt, 22, 7);
    V.open = envelope(V.open, V.env * clamp01(lo / tot * 1.9), dt, 18, 6);
    V.round = envelope(V.round, V.env * clamp01(mid / tot * 2.1) * (1 - clamp01(hi / tot * 2.4)), dt, 13, 6);
    V.wide = envelope(V.wide, V.env * clamp01(hi / tot * 3.2), dt, 20, 7);
    V.tip = envelope(V.tip, V.env * clamp01(hi / tot * 2.2), dt, 20, 6);
    V.active = V.env > 0.02;
    return V;
  }
  function visemeActions(V, amt) {
    const a = amt === undefined ? 1 : amt;
    if (!V.active) return null;
    return {
      j: V.open * 0.62 * a,
      pk: V.round * 0.7 * a,
      sL: 0.08 + V.wide * 0.5 * a, sR: 0.08 + V.wide * 0.5 * a,
      st: V.tip * 0.75 * a,
      bu: V.env * 0.08, ck: V.env * 0.3
    };
  }

  /* ===================================================================
     9. Signal extraction from blendshapes + geometry (photo & webcam)
     =================================================================== */
  function blendGetter(bs) {
    const map = {};
    if (bs) for (const c of bs) map[c.categoryName] = c.score;
    return n => map[n] || 0;
  }
  function eyeOpeningRatio(P, which) {
    const d = LIDS[which];
    const up = d.up.reduce((a, i) => a + P[i].y, 0) / d.up.length;
    const lo = d.lo.reduce((a, i) => a + P[i].y, 0) / d.lo.length;
    const ew = Math.hypot(P[d.oc].x - P[d.ic].x, P[d.oc].y - P[d.ic].y) || 1;
    return (lo - up) / ew;
  }
  function irisOffset(P, which, S) {
    const I = IRIS[which], d = LIDS[which];
    const c = S ? S[I.c] : P[I.c];
    const ic = S ? S[d.ic] : P[d.ic], oc = S ? S[d.oc] : P[d.oc];
    const upP = d.up.map(i => (S ? S[i] : P[i])), loP = d.lo.map(i => (S ? S[i] : P[i]));
    const ex = oc.x - ic.x, ey = oc.y - ic.y, ew = Math.hypot(ex, ey) || 1;
    const ax = ex / ew, ay = ey / ew;
    const dx = c.x - (ic.x + oc.x) / 2;
    const dy = c.y - (upP[2].y + loP[2].y) / 2;
    return [(dx * ax + dy * ay) / ew, (-dx * ay + dy * ax) / ew];
  }

  const eulerYaw = q => eulerFromQuat(q).yaw;
  const eulerYPR = q => eulerFromQuat(q);

  root.FMEngine = {
    eulerYaw, eulerYPR,
    /* tables + accessors */
    lx, ly, lz,
    OVAL, CONTOUR, LEYE, REYE, LBROW, RBROW, LIPS_O, LIPS_I, NOSE, UP_LIP, LO_LIP, LIDS, IRIS, EYE_RING,
    RIGID, RIGID_W, EXPR_GAIN, EXPR_CAP, ACTIONS,
    /* geometry */
    buildCanonical, buildMesh, faceCenterOf, localUV, tmjAxis, jawWeight, applyJaw,
    /* expression */
    zeroActions, PRESETS, talkAt, expressionDelta, mouthState, makeTransfer, retargetResidual, residualEnergy,
    /* tracking */
    createTracker, trackerPush, trackerSetNeutral, trackerUpdate, trackerIdle,
    /* audio */
    createViseme, visemeUpdate, visemeActions,
    /* signals */
    blendGetter, eyeOpeningRatio, irisOffset
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
