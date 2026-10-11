/* =====================================================================
   FaceMirror · puppet renderer
   The still photo becomes a closed 3D shell (face + three hair rings +
   a pinned frame).  Every frame the shell is rotated about the head
   centre, weak-perspective projected and painted far -> near as
   per-triangle affines.

   Anti-artifact design goals (not a guarantee of photorealistic turns)
     · stable far-to-near depth-key sorting and back-face handling reduce
       paint-order smearing
     · culled / far-side triangles re-sample a mirrored source patch; this
       mitigates silhouette collapse but cannot reconstruct unseen details
     · every clip goes through FM.polyOK and every wide fill is clipped
       to a computed bounding box -> a degenerate clip can never flash
       the whole canvas
     · no getImageData in the frame loop; blur filters are local to small feature passes
     · the shell is composed on an offscreen layer that is blitted once,
       so a frame is never shown half-finished
     · every feature pass runs every frame, gated by smooth envelopes
   ===================================================================== */
(function (root) {
  'use strict';
  const FM = root.FM, E = root.FMEngine;
  const { clamp, clamp01, lerp, envelope, polyOK, bounds, m3, v3, matFromQuat } = FM;

  function makeCanvas(w, h) {
    const c = root.document.createElement('canvas');
    c.width = Math.max(1, Math.round(w));
    c.height = Math.max(1, Math.round(h));
    return c;
  }

  /* --------------------------------------------------------- textures */
  function enamelTexture() {
    const c = makeCanvas(128, 160), g = c.getContext('2d');
    const grd = g.createLinearGradient(0, 0, 0, 160);
    [[0, '#cdbd97'], [.2, '#e9e0c8'], [.55, '#f7f2e6'], [.85, '#f1f2ee'], [1, '#cfd7dc']].forEach(s => grd.addColorStop(s[0], s[1]));
    g.fillStyle = grd; g.fillRect(0, 0, 128, 160);
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 46; i++) {
      const x = rnd() * 128, y = rnd() * 160, r = 18 + rnd() * 34;
      const rg = g.createRadialGradient(x, y, 0, x, y, r);
      rg.addColorStop(0, rnd() < .5 ? 'rgba(255,252,240,.18)' : 'rgba(190,160,100,.13)');
      rg.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = rg; g.fillRect(0, 0, 128, 160);
    }
    for (let i = 0; i < 90; i++) {
      g.fillStyle = rnd() < .5 ? 'rgba(255,255,255,.07)' : 'rgba(120,95,60,.06)';
      g.fillRect(rnd() * 128, rnd() * 40, 1 + rnd() * 2, 60 + rnd() * 100);
    }
    return c;
  }
  function noiseTexture(size) {
    const c = makeCanvas(size, size), g = c.getContext('2d');
    const d = g.createImageData(size, size);
    let s = 99;
    for (let i = 0; i < d.data.length; i += 4) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      const v = 128 + ((s >> 16) & 63) - 32;
      d.data[i] = d.data[i + 1] = d.data[i + 2] = v;
      d.data[i + 3] = 255;
    }
    g.putImageData(d, 0, 0);
    return c;
  }

  /* A light unsharp pass of the source photo.  Drawn once over the finished
     shell in `overlay`, it restores the pore-level detail the triangle pass
     loses, which is most of what makes the render read as a photograph.  */
  function buildSharpen(S) {
    const { W, H } = S;
    const out = makeCanvas(W, H);
    const g = out.getContext('2d');
    g.drawImage(S.src, 0, 0);
    /* blur a copy, then subtract it from the original in a difference-like
       composite and push the result through overlay */
    const blur = makeCanvas(W, H);
    const bg = blur.getContext('2d');
    bg.filter = 'blur(1.6px)';
    bg.drawImage(S.src, 0, 0);
    bg.filter = 'none';
    g.globalCompositeOperation = 'difference';
    g.globalAlpha = 0.55;
    g.drawImage(blur, 0, 0);
    g.globalAlpha = 1;
    g.globalCompositeOperation = 'source-over';
    /* keep only the skin: the photo's own lips and eyes must not be
       sharpened into hard rings */
    S.cache.sharpen = out;
  }

  /* ---------------------------------------------------------- session */
  function createSession() {
    return {
      head: null, mesh: null, src: null, canvas: null, layer: null, lctx: null,
      W: 0, H: 0, ready: false,
      S2: null, DST: null, BASE: null, depth: null, normals: null, normalsFace: null,
      passes: null, proj: null, arches: null,
      cache: { enamel: enamelTexture(), noise: noiseTexture(96), iris: [null, null], sclera: ['235,230,225', '235,230,225'], plate: null, lum: 0.7, warm: 0.5 },
      hair: { x: 0, y: 0, vx: 0, vy: 0 },
      lidSkin: [null, null], jaw: 0, openEnv: 0,
      frames: 0, stats: { tris: 0, culled: 0 }, lastDebug: null
    };
  }

  const readAt = (S, x, y, n) => {
    const g = S.src.getContext('2d');
    const x0 = clamp(Math.round(x) - (n >> 1), 0, Math.max(0, S.W - n));
    const y0 = clamp(Math.round(y) - (n >> 1), 0, Math.max(0, S.H - n));
    try {
      const d = g.getImageData(x0, y0, n, n).data;
      let r = 0, gg = 0, b = 0, c = 0;
      for (let i = 0; i < d.length; i += 4) { r += d[i]; gg += d[i + 1]; b += d[i + 2]; c++; }
      return [r / c, gg / c, b / c];
    } catch (e) { return [120, 90, 70]; }
  };

  function prepIris(S) {
    const { head } = S;
    let lum = 0, warm = 0, n = 0;
    for (const i of [205, 425, 50, 280, 36, 266, 116, 345, 129, 358]) {
      const p = head.P[i], c = readAt(S, p[0], p[1], 5);
      lum += (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
      warm += (c[0] - c[2]) / 255;
      n++;
    }
    S.cache.lum = n ? clamp(lum / n, 0.1, 1) : 0.7;
    S.cache.warm = n ? clamp(warm / n * 2 + 0.5, 0, 1) : 0.5;
    for (let e = 0; e < 2; e++) {
      const which = e ? 'R' : 'L';
      const I = E.IRIS[which], ring = I.ring, c = head.P[I.c];
      let rs = 0;
      for (const k of ring) rs += Math.hypot(head.P[k][0] - c[0], head.P[k][1] - c[1]);
      rs = Math.max(2.5, rs / ring.length);
      const R = Math.ceil(rs * 1.35);
      const s1 = readAt(S, c[0] + (head.P[E.LIDS[which].ic][0] - c[0]) * .6, c[1] + (head.P[E.LIDS[which].ic][1] - c[1]) * .6, 5);
      const s2 = readAt(S, c[0] + (head.P[E.LIDS[which].oc][0] - c[0]) * .6, c[1] + (head.P[E.LIDS[which].oc][1] - c[1]) * .6, 5);
      const brighter = (s1[0] + s1[1] + s1[2] > s2[0] + s2[1] + s2[2]) ? s1 : s2;
      S.cache.sclera[e] = brighter.map(v => Math.round(clamp(v * 1.04 * (0.6 + S.cache.lum * 0.6), 0, 255))).join(',');
      const patch = makeCanvas(R * 2, R * 2), g = patch.getContext('2d');
      const ir = [0, 1, 2].reduce((acc, k) => {
        const q = readAt(S, c[0] + (k === 0 ? -rs * .6 : k === 1 ? rs * .6 : 0), c[1] + (k === 2 ? rs * .6 : 0), 5);
        return [acc[0] + q[0] / 3, acc[1] + q[1] / 3, acc[2] + q[2] / 3];
      }, [0, 0, 0]);
      const gr = g.createRadialGradient(R, R, 0, R, R, rs);
      gr.addColorStop(0, 'rgb(14,10,8)'); gr.addColorStop(.3, 'rgb(14,10,8)');
      gr.addColorStop(.42, `rgb(${ir.map(Math.round)})`);
      gr.addColorStop(1, `rgb(${ir.map(v => Math.round(v * .6))})`);
      g.fillStyle = gr; g.beginPath(); g.arc(R, R, rs, 0, 7); g.fill();
      g.save();
      g.beginPath();
      E.EYE_RING[which].forEach((i, k) => {
        const p = head.P[i], x = p[0] - c[0] + R, y = p[1] - c[1] + R;
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.closePath(); g.clip();
      try { g.drawImage(S.src, c[0] - R, c[1] - R, R * 2, R * 2, 0, 0, R * 2, R * 2); } catch (e) {}
      g.restore();
      g.globalCompositeOperation = 'destination-in';
      const m = g.createRadialGradient(R, R, rs * .9, R, R, rs * 1.24);
      m.addColorStop(0, 'rgba(0,0,0,1)'); m.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = m; g.fillRect(0, 0, R * 2, R * 2);
      S.cache.iris[e] = { cv: patch, R, r: rs };
    }
  }

  /* plate: the photo with the silhouette band radially edge-extended, so
     pixels revealed by a turn continue hair / skin instead of showing a
     stale frontal cheek.  Built once per image.                       */
  function buildPlate(S) {
    const { W, H, head } = S;
    const plate = makeCanvas(W, H), g = plate.getContext('2d');
    g.drawImage(S.src, 0, 0);
    const cx = head.ctr[0], cy = head.ctr[1];
    const R = Math.max(head.fw, head.fh) * 0.56;
    const N = 56;
    g.save();
    g.filter = 'blur(3px)';
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2;
      const dx = Math.cos(a), dy = Math.sin(a);
      for (const k of [0.99, 1.09, 1.19, 1.29]) {
        const inx = cx + dx * R * k, iny = cy + dy * R * k;
        const sz = 22;
        const x0 = clamp(inx - dx * 8 - sz / 2, 0, W - sz), y0 = clamp(iny - dy * 8 - sz / 2, 0, H - sz);
        g.globalAlpha = 0.34;
        try { g.drawImage(S.src, x0, y0, sz, sz, x0 + dx * 9, y0 + dy * 9, sz, sz); } catch (e) {}
      }
    }
    g.restore();
    S.cache.plate = plate;
    const bw = Math.max(8, Math.round(W / 8)), bh = Math.max(8, Math.round(H / 8));
    const bl = makeCanvas(bw, bh);
    bl.getContext('2d').drawImage(S.src, 0, 0, bw, bh);
    S.cache.blurSmall = bl;
  }

  /* ------------------------------------------------------ sessionInit */
  function sessionInit(S, spec) {
    S.src = spec.source;
    S.canvas = spec.canvas;
    S.head = spec.head;
    S.mesh = spec.mesh;
    S.W = S.src.width; S.H = S.src.height;
    if (S.canvas.width !== S.W || S.canvas.height !== S.H) { S.canvas.width = S.W; S.canvas.height = S.H; }
    S.layer = makeCanvas(S.W, S.H);
    S.lctx = S.layer.getContext('2d');
    const n = spec.mesh.n;
    S.S2 = new Float32Array(n * 2);
    S.DST = new Float32Array(n * 2);
    S.BASE = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = spec.mesh.P[i];
      S.S2[i * 2] = p[0]; S.S2[i * 2 + 1] = p[1];
      S.BASE[i * 3] = p[0]; S.BASE[i * 3 + 1] = p[1]; S.BASE[i * 3 + 2] = p[2];
    }
    S.depth = new Float32Array(n);
    const mkNormals = (list) => {
      const out = new Float32Array(list.length);
      for (let t = 0; t < list.length; t += 3) {
        const a = list[t], b = list[t + 1], c = list[t + 2];
        const pa = spec.mesh.P[a], pb = spec.mesh.P[b], pc = spec.mesh.P[c];
        let nr = v3.cross(v3.sub(pb, pa), v3.sub(pc, pa));
        const l = v3.len(nr) || 1;
        nr = [nr[0] / l, nr[1] / l, nr[2] / l];
        const area2 = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
        if (area2 < 0) nr = [-nr[0], -nr[1], -nr[2]];
        out[t] = nr[0]; out[t + 1] = nr[1]; out[t + 2] = nr[2];
      }
      return out;
    };
    S.normals = mkNormals(spec.mesh.base);
    S.normalsFace = mkNormals(spec.mesh.face);
    const mkPass = (list, normals, o) => ({
      list, normals, n: list.length / 3,
      order: new Int32Array(list.length / 3),
      dkey: new Float32Array(list.length / 3),
      tmp: new Int32Array(list.length / 3),
      mergeFallbacks: 0, initialized: false,
      pad: o.pad, cull: o.cull, mirror: o.mirror !== false
    });
    S.triA = mkPass(spec.mesh.base, S.normals, { pad: 0.9, cull: -0.02, mirror: true });
    S.triB = mkPass(spec.mesh.face, S.normalsFace, { pad: 0.95, cull: -0.05, mirror: true });
    S.mid = spec.head.mid.slice();
    S.axis = E.tmjAxis(spec.head);
    S.arches = buildArches();
    buildSharpen(S);
    prepIris(S);
    buildPlate(S);
    buildDental(S);
    S.ready = true;
    return S;
  }
  /* ------------------------------------------------------------ deform */
  function deform(S, st) {
    const { head, mesh } = S;
    const n = mesh.n;
    const delta = st.deltaCanonical;
    const jawGain = clamp01(st.jawMeshGain === undefined ? 1 : st.jawMeshGain);
    const C = new Array(478);
    for (let i = 0; i < 478; i++) {
      const p = head.P[i];
      C[i] = [p[0], p[1], p[2]];
      if (delta) { C[i][0] += delta[i * 3]; C[i][1] += delta[i * 3 + 1]; C[i][2] += delta[i * 3 + 2]; }
    }
    /* The mandible is a bone, so its angle is an input, not something to be
       guessed from the lips: `st.jawAngle` (radians) arrives already
       smoothed from whichever driver opened the mouth — the webcam's jaw,
       a preset or the microphone.  Only when nobody supplies one does the
       aperture of the lip landmarks stand in for it.                  */
    const expressionOpen = clamp01(st.jawOpen || 0);
    let want;
    if (st.jawAngle !== undefined && st.jawAngle !== null && isFinite(st.jawAngle)) {
      want = clamp(st.jawAngle, -0.05, E.JAW_MAX * 1.35);
    } else {
      const gapNow = Math.hypot(C[13][0] - C[14][0], C[13][1] - C[14][1]);
      want = clamp01(Math.max(gapNow / (head.fh * 0.17), expressionOpen)) * 0.36;
    }
    S.jaw = FM.slew(S.jaw, want, st.dt || 1 / 60, (st.jawRate || 6.5), 7.5);
    if (jawGain > 0.01) E.applyJaw(head, C, S.jaw * jawGain, S.axis);
    /* hair spring: chases the head, lags, capped */
    const H = S.hair;
    const yaw = E.eulerYaw(st.q);
    const tgtX = -(st.ox || 0) - head.fw * 0.75 * Math.sin(yaw);
    const tgtY = -(st.oy || 0) - head.fh * 0.05 * head.P[10][2] / Math.max(1, head.fw) + Math.sin(st.t * 1.3) * head.fh * 0.004;
    H.vx = (H.vx + (tgtX - H.x) * 0.10) * 0.87;
    H.vy = (H.vy + (tgtY - H.y) * 0.10) * 0.87;
    H.x += H.vx; H.y += H.vy;
    const hg = st.hairGain === undefined ? 0.6 : st.hairGain;
    const hm = Math.hypot(H.x, H.y), cap = head.fh * 0.16;
    const hlx = (hm > cap ? H.x * cap / hm : H.x) * hg;
    const hly = (hm > cap ? H.y * cap / hm : H.y) * hg;
    /* rotation + weak perspective about the head centre */
    const q = st.q, R = matFromQuat(q);
    const f = st.focal * head.fw;
    const cx = head.ctr[0], cy = head.ctr[1];
    const ox = st.ox || 0, oy = st.oy || 0, sc = st.scale || 1;
    const zc = st.depthGain || 0;
    const pivotZ = (head.P[10][2] + head.P[152][2]) * 0.5 * zc;
    const proj = (p) => {
      const rx = p[0] - cx, ry = p[1] - cy, rz = p[2] - pivotZ;
      const mx = R[0] * rx + R[1] * ry + R[2] * rz;
      const my = R[3] * rx + R[4] * ry + R[5] * rz;
      const mz = R[6] * rx + R[7] * ry + R[8] * rz;
      let d = f - mz * sc;
      if (Math.abs(d) < f * 0.25) d = d < 0 ? -f * 0.25 : f * 0.25;
      const k = f / d;
      return [cx + mx * sc * k + ox, cy + my * sc * k + oy, mz];
    };
    S.proj = proj;
    const body = st.bodyAt || null;
    for (let i = 0; i < n; i++) {
      const w = mesh.w[i], sx = S.S2[i * 2], sy = S.S2[i * 2 + 1];
      if (w <= 0) {
        let dx = 0, dy = 0;
        if (body && mesh.kind[i] === 2) { const b = body(sx, sy); if (b) { dx = b[0]; dy = b[1]; } }
        S.DST[i * 2] = sx + dx; S.DST[i * 2 + 1] = sy + dy; S.depth[i] = S.BASE[i * 3 + 2];
        continue;
      }
      const isFace = i < 478;
      const cur = isFace ? C[i] : [S.BASE[i * 3], S.BASE[i * 3 + 1], S.BASE[i * 3 + 2]];
      const p1 = isFace ? proj([cur[0], cur[1], cur[2] * (1 + zc)]) : proj(cur);
      const p0 = isFace ? proj([head.P[i][0], head.P[i][1], head.P[i][2] * (1 + zc)])
                        : proj([S.BASE[i * 3], S.BASE[i * 3 + 1], S.BASE[i * 3 + 2]]);
      let dx = (p1[0] - p0[0]) * w, dy = (p1[1] - p0[1]) * w;
      if (mesh.kind[i] === 1 && hg > 0) {
        const info = mesh.ringOf[i];
        const hw = info ? [0.5, 0.85, 0.4][info.ring] : 0;
        const top = clamp01((head.ctr[1] - sy) / head.fh + 0.75);
        dx += hlx * hw * top; dy += hly * hw * top;
      }
      S.DST[i * 2] = sx + dx;
      S.DST[i * 2 + 1] = sy + dy;
      S.depth[i] = p1[2] * w + S.BASE[i * 3 + 2] * (1 - w) * 0.4;
    }
    if (st.limits !== false) {
      limits(head, C, (out) => {
        for (let i = 0; i < 478; i++) {
          const w = mesh.w[i];
          if (w <= 0) continue;
          const d = out[i];
          const p1 = proj([d[0], d[1], d[2] * (1 + zc)]);
          const p0 = proj([C[i][0], C[i][1], C[i][2] * (1 + zc)]);
          S.DST[i * 2] += (p1[0] - p0[0]) * w;
          S.DST[i * 2 + 1] += (p1[1] - p0[1]) * w;
        }
      });
    }
    return C;
  }
  /* lip / lid collision: smooth projection, never a hard snap */
  function limits(head, C, apply) {
    const groups = [
      [E.LIDS.L.up, E.LIDS.L.lo], [E.LIDS.R.up, E.LIDS.R.lo]
    ];
    const out = C.map(p => p.slice());
    let touched = false;
    for (const [ups, los] of groups) {
      for (let k = 0; k < Math.min(ups.length, los.length); k++) {
        const u = out[ups[k]], l = out[los[k]];
        const pen = (u[1] + 0.4) - l[1];
        if (pen > 0) {
          const corr = Math.min(pen * 0.85, head.fh * 0.06);
          out[ups[k]] = [u[0], u[1] - corr, u[2]];
          out[los[k]] = [l[0], l[1] + corr * 0.9, l[2]];
          touched = true;
        }
      }
    }
    /* Lips.  They are no longer compared index by index (MediaPipe's two
       lip runs are not evenly spaced, so pairing them by index made the
       corners collide and the middle of the lip stay open).  Instead the
       inner upper run and the inner lower run are resampled by *arc
       length* and compared where they actually are, and the correction is
       a smooth field over the whole mouth so a contact reads as lips
       pressing together, not as two dots jumping.                     */
    const up = out.slice(0, 478), lo = out.slice(0, 478);
    const a = E.mouthFrame3(head, up), b = E.mouthFrame3(head, lo);
    if (a && b) {
      const N = 13;
      const cu = FM.resampleCurve(FM.splineOpen(E.LIP_SETS.upIn.map(i => up[i]), 40), N);
      const cl = FM.resampleCurve(FM.splineOpen(E.LIP_SETS.loIn.map(i => lo[i]), 40), N);
      const push = new Map();
      for (let k = 0; k < N; k++) {
        /* contact: the lower lip may not pass through the upper one.  A
           small overlap is fine (that is what "closed" looks like); only
           the excess is removed, and only where it is excess.        */
        const slack = -(a.w) * 0.004;
        const pen = (cu[k][1] + slack) - cl[k][1];
        if (pen <= 0) continue;
        const corr = Math.min(pen * 0.9, a.w * 0.10);
        const t = k / (N - 1);
        push.set(t, { c: corr });
      }
      if (push.size) {
        /* distribute the correction along each lip run by parameter */
        const applyRun = (run, sign) => {
          const idx = run.map(i => i);
          for (let k = 0; k < idx.length; k++) {
            const t = k / (idx.length - 1);
            let corr = 0;
            for (const [tt, v] of push) corr = Math.max(corr, v.c * Math.max(0, 1 - Math.abs(tt - t) * (N - 1) / 1.6));
            if (corr > 0) {
              const p = out[idx[k]];
              out[idx[k]] = [p[0], p[1] + sign * corr * 0.5, p[2]];
              touched = true;
            }
          }
        };
        applyRun(E.LIP_SETS.upIn, -1);
        applyRun(E.LIP_SETS.loIn, 1);
      }
    }
    if (touched) apply(out);
  }

  /* ----------------------------------------------------- triangle pass */
  function drawShell(S, ctx, st, pass, opts) {
    const o = opts || {};
    const DST = S.DST, S2 = S.S2, W = S.W, H = S.H;
    const p = S.proj;
    /* Depth key + temporally coherent ordering.  Start from last frame's
       order (insertion sort is O(m) when the pose barely moved), but if the
       budget blows up during a fast turn, fall back to a stable merge sort.
       Bucket order alone is insufficient: a small within-bucket inversion
       is still enough to smear two overlapping face triangles.             */
    const list = pass.list, m = pass.n, dk = pass.dkey, order = pass.order;
    if (o.resetOrder || !pass.initialized) {
      for (let i = 0; i < m; i++) order[i] = i;
      pass.initialized = true;
    }
    const tmp = pass.tmp;
    tmp.set(order.subarray(0, m)); // backup: insertion sort can stop mid-shift
    for (let t = 0; t < m; t++) {
      const i = t * 3;
      const a = list[i], b = list[i + 1], c = list[i + 2];
      dk[t] = (S.depth[a] + S.depth[b] + S.depth[c]) / 3;
    }
    {
      let work = 0, exact = true;
      const budget = m * 8;
      for (let i = 1; i < m; i++) {
        const v = order[i];
        let j = i - 1;
        while (j >= 0 && dk[order[j]] > dk[v]) {
          order[j + 1] = order[j]; j--;
          if (++work > budget) { exact = false; break; }
        }
        if (!exact) break;
        order[j + 1] = v;
      }
      if (!exact) {
        pass.mergeFallbacks++;
        /* Stable bottom-up merge sort: exact mean-depth-key order, O(m log m),
           allocation-free via the pass-owned scratch array. Restore the
           pre-sort order first: ties then keep their frame-to-frame order. */
        order.set(tmp.subarray(0, m)); // recover the intact list after a partial insertion sort
        for (let width = 1; width < m; width <<= 1) {
          for (let left = 0; left < m; left += width << 1) {
            const mid = Math.min(left + width, m), right = Math.min(left + (width << 1), m);
            let i = left, j = mid, k = left;
            while (i < mid && j < right) tmp[k++] = dk[order[i]] <= dk[order[j]] ? order[i++] : order[j++];
            while (i < mid) tmp[k++] = order[i++];
            while (j < right) tmp[k++] = order[j++];
          }
          order.set(tmp.subarray(0, m));
        }
      }
    }
    const R = matFromQuat(st.q);
    const midX = S.mid[0];
    const pad = pass.pad * (o.padScale || 1);
    let drawn = 0, culled = 0;
    for (let t = 0; t < m; t++) {
      const tri = order[t] * 3;
      const a = list[tri], b = list[tri + 1], c = list[tri + 2];
      const nz = R[6] * pass.normals[tri] + R[7] * pass.normals[tri + 1] + R[8] * pass.normals[tri + 2];
      const behind = nz < pass.cull;
      const ax = DST[a * 2], ay = DST[a * 2 + 1];
      const bx = DST[b * 2], by = DST[b * 2 + 1];
      const cx = DST[c * 2], cy = DST[c * 2 + 1];
      const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (!isFinite(area) || Math.abs(area) < 0.10) continue;
      const minX = Math.min(ax, bx, cx), maxX = Math.max(ax, bx, cx);
      const minY = Math.min(ay, by, cy), maxY = Math.max(ay, by, cy);
      if (maxX < -3 || maxY < -3 || minX > W + 3 || minY > H + 3) continue;
      let s0x = S2[a * 2], s0y = S2[a * 2 + 1];
      let s1x = S2[b * 2], s1y = S2[b * 2 + 1];
      let s2x = S2[c * 2], s2y = S2[c * 2 + 1];
      if (behind && pass.mirror) {
        /* Mirrored source approximation; swap the sample winding so the
           affine remains non-degenerate. This is not learned inpainting. */
        s0x = 2 * midX - s0x;
        const tmpX = 2 * midX - s1x, tmpY = s1y;
        s1x = 2 * midX - s2x; s1y = s2y;
        s2x = tmpX; s2y = tmpY;
      }
      const det = (s1x - s0x) * (s2y - s0y) - (s1y - s0y) * (s2x - s0x);
      if (!isFinite(det) || Math.abs(det) < 1e-6) continue;
      const k = 1 / det;
      const ma = ((bx - ax) * (s2y - s0y) - (by - ay) * (s2x - s0x)) * k;
      const mb = ((by - ay) * (s1x - s0x) - (bx - ax) * (s1y - s0y)) * k;
      const mc = ((cx - ax) * (s2y - s0y) - (cy - ay) * (s2x - s0x)) * k;
      const md = ((cy - ay) * (s1x - s0x) - (cx - ax) * (s1y - s0y)) * k;
      const me = ax - ma * s0x - mc * s0y;
      const mf = ay - mb * s0x - md * s0y;
      const gx = (ax + bx + cx) / 3, gy = (ay + by + cy) / 3;
      ctx.save();
      ctx.beginPath();
      const V = [[ax, ay], [bx, by], [cx, cy]];
      for (let i = 0; i < 3; i++) {
        const vx = V[i][0] - gx, vy = V[i][1] - gy, l = Math.hypot(vx, vy) || 1;
        const X = V[i][0] + vx / l * pad, Y = V[i][1] + vy / l * pad;
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
      ctx.closePath();
      ctx.clip();
      ctx.setTransform(ma, mb, mc, md, me, mf);
      const x0 = clamp(Math.floor(Math.min(s0x, s1x, s2x)) - 1, 0, W);
      const y0 = clamp(Math.floor(Math.min(s0y, s1y, s2y)) - 1, 0, H);
      const x1 = clamp(Math.ceil(Math.max(s0x, s1x, s2x)) + 1, 0, W);
      const y1 = clamp(Math.ceil(Math.max(s0y, s1y, s2y)) + 1, 0, H);
      const sw = x1 - x0, sh = y1 - y0;
      if (sw > 0 && sh > 0) {
        try { ctx.drawImage(behind ? S.cache.plate : S.src, x0, y0, sw, sh, x0, y0, sw, sh); } catch (e) {}
      }
      ctx.restore();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      drawn++; if (behind) culled++;
    }
    S.stats.tris += drawn; S.stats.culled = (S.stats.culled || 0) + culled;
    return drawn;
  }

  /* ------------------------------------------------------- accessories */
  function drawEyelid(S, ctx, which, c, D) {
    if (c < 0.015) return;
    const d = E.LIDS[which];
    const p = D[d.ic], q = D[d.oc];
    const upP = d.up.map(i => D[i]), loP = d.lo.map(i => D[i]);
    const ew = Math.hypot(p.x - q.x, p.y - q.y) || 1;
    const top = Math.min(...upP.map(v => v.y)), bot = Math.max(...loP.map(v => v.y));
    const eh = Math.max(bot - top, ew * 0.13);
    const u = upP[2], l = loP[2];
    const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
    const m = { x: u.x + (l.x - u.x) * c, y: u.y + (l.y - u.y) * c };
    const k = { x: m.x * 2 - mx, y: m.y * 2 - my };
    const ty = top - eh * 0.5, m0 = ew * 0.07;
    const x0 = Math.floor(Math.min(p.x, q.x) - m0 - 8), y0 = Math.floor(ty - 8);
    const W = Math.ceil(ew + m0 * 2 + 16), Hh = Math.ceil(bot - ty + 16);
    const sy = top - eh * 0.95;
    if (W < 6 || Hh < 6 || x0 < 0 || y0 < 0 || x0 + W > S.W || y0 + Hh > S.H || sy < 0) return;
    const slot = which === 'L' ? 0 : 1;
    let fc = S.lidSkin[slot];
    if (!fc || fc.width < W || fc.height < Hh) {
      fc = makeCanvas(Math.max(W, S.W), Math.max(Hh, S.H));
      S.lidSkin[slot] = fc;
    }
    const fx = fc.getContext('2d');
    fx.setTransform(1, 0, 0, 1, 0, 0);
    fx.clearRect(0, 0, fc.width, fc.height);
    fx.save();
    fx.translate(-x0, -y0);
    fx.filter = 'blur(1.2px)';
    fx.beginPath();
    fx.moveTo(p.x, p.y);
    fx.quadraticCurveTo(k.x, k.y, q.x, q.y);
    fx.lineTo(q.x + (q.x > p.x ? m0 : -m0), ty);
    fx.lineTo(p.x + (q.x > p.x ? -m0 : m0), ty);
    fx.closePath();
    fx.fill();
    fx.filter = 'none';
    fx.globalCompositeOperation = 'source-in';
    const skin = S.layer;
    try { fx.drawImage(skin, x0, sy, W, eh * 0.85, x0, ty, W, Math.max(2, bot - ty + 6)); } catch (e) {}
    fx.globalCompositeOperation = 'source-over';
    fx.restore();
    ctx.save();
    ctx.globalAlpha = clamp01(c * 3);
    try { ctx.drawImage(fc, 0, 0, W, Hh, x0, y0, W, Hh); } catch (e) {}
    ctx.restore();
    ctx.save();
    const edge = () => {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.quadraticCurveTo(k.x, k.y, q.x, q.y);
    };
    ctx.lineCap = 'round';
    ctx.filter = 'blur(1.6px)';
    ctx.strokeStyle = 'rgba(58,30,26,.30)';
    ctx.lineWidth = Math.max(1.2, ew * 0.085);
    edge(); ctx.stroke();
    ctx.filter = 'none';
    ctx.strokeStyle = 'rgba(43,26,22,.85)';
    ctx.lineWidth = Math.max(1.1, ew * 0.032);
    edge(); ctx.stroke();
    ctx.restore();
  }

  function drawIris(S, ctx, e, D, gaze, gain) {
    const I = S.cache.iris[e];
    if (!I) return;
    const which = e ? 'R' : 'L';
    const d = E.LIDS[which], ring = E.IRIS[which].ring;
    const pc = D[E.IRIS[which].c];
    const ex = D[d.oc].x - D[d.ic].x, ey = D[d.oc].y - D[d.ic].y;
    const ew = Math.hypot(ex, ey) || 1;
    let r = 0;
    for (const k of ring) r += Math.hypot(D[k].x - pc.x, D[k].y - pc.y);
    r = (r / ring.length) || I.r;
    const ax = ex / ew, ay = ey / ew;
    const off = [(gaze[0] * ax - gaze[1] * ay) * ew * gain, (gaze[0] * ay + gaze[1] * ax) * ew * gain];
    const X = pc.x + off[0], Y = pc.y + off[1];
    const RR = Math.max(1.2, I.R * (r / I.r));
    const poly = E.EYE_RING[which].map(i => D[i]);
    if (!polyOK(poly, 3)) return;
    const bb = bounds(poly);
    ctx.save();
    ctx.beginPath();
    poly.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath();
    ctx.clip();
    const a = clamp01(Math.hypot(off[0], off[1]) / (ew * 0.02));
    if (a > 0.01) {
      const grd = ctx.createRadialGradient(pc.x, pc.y, Math.max(1, r * 0.25), pc.x, pc.y, Math.max(2, r * 1.5));
      grd.addColorStop(0, `rgba(${S.cache.sclera[e]},${a})`);
      grd.addColorStop(1, `rgba(${S.cache.sclera[e]},0)`);
      ctx.fillStyle = grd;
      ctx.fillRect(bb.x0 - 2, bb.y0 - 2, bb.w + 4, bb.h + 4);
    }
    try { ctx.drawImage(I.cv, X - RR, Y - RR, RR * 2, RR * 2); } catch (err) {}
    const lg = ctx.createRadialGradient(X, Y, Math.max(1, r * 0.8), X, Y, Math.max(1.2, r * 1.08));
    lg.addColorStop(0, 'rgba(15,10,8,0)'); lg.addColorStop(1, 'rgba(15,10,8,.34)');
    ctx.fillStyle = lg;
    ctx.beginPath(); ctx.arc(X, Y, r * 1.1, 0, 7); ctx.fill();
    const sg = ctx.createLinearGradient(0, bb.y0, 0, bb.y0 + bb.h * 0.55);
    sg.addColorStop(0, 'rgba(28,14,10,.36)'); sg.addColorStop(1, 'rgba(28,14,10,0)');
    ctx.fillStyle = sg;
    ctx.fillRect(bb.x0, bb.y0, bb.w, bb.h * 0.6);
    ctx.restore();
  }

  function drawLips(S, ctx, D, mw, gap) {
    const O = E.LIPS_O.map(i => D[i]), In = E.LIPS_I.map(i => D[i]);
    const a = D[61], b = D[291];
    if (!polyOK(O, 12) || !polyOK(In, 2.5)) return false;
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    const dn = { x: -Math.sin(ang), y: Math.cos(ang) };
    const bb = bounds(O);
    const seam = clamp01(1 - gap / (mw * 0.075));
    const lc = { x: (D[17].x + D[14].x) / 2, y: (D[17].y + D[14].y) / 2 };
    const uc = { x: (D[0].x + D[13].x) / 2, y: (D[0].y + D[13].y) / 2 };
    ctx.save();
    ctx.beginPath();
    O.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath();
    In.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath();
    ctx.clip('evenodd');
    ctx.fillStyle = 'rgba(186,66,82,.13)';
    ctx.fillRect(bb.x0 - 2, bb.y0 - 2, bb.w + 4, bb.h + 4);
    ctx.lineWidth = 0.8;
    for (let i = 0; i < 30; i++) {
      const t = (i + .5) / 30, bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t, h = mw * 0.19;
      ctx.strokeStyle = `rgba(84,28,36,${.06 + .05 * (i % 3)})`;
      ctx.beginPath();
      ctx.moveTo(bx - dn.x * h, by - dn.y * h);
      ctx.lineTo(bx + dn.x * h, by + dn.y * h);
      ctx.stroke();
    }
    const gl = (c, rw, rh, al) => {
      if (!(rw > 0.5 && rh > 0.2)) return;
      ctx.save();
      ctx.translate(c.x, c.y); ctx.rotate(ang); ctx.scale(1, rh / rw);
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rw);
      g.addColorStop(0, `rgba(255,240,238,${al})`);
      g.addColorStop(1, 'rgba(255,240,238,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(0, 0, rw, 0, 7); ctx.fill();
      ctx.restore();
    };
    gl(lc, mw * 0.26, mw * 0.05, .36 * (1 - seam * 0.3));
    gl(uc, mw * 0.16, mw * 0.035, .15);
    ctx.restore();
    if (seam > 0.02) {
      ctx.save();
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      const path = () => {
        ctx.beginPath();
        E.UP_LIP.forEach((i, k) => k ? ctx.lineTo(D[i].x, D[i].y) : ctx.moveTo(D[i].x, D[i].y));
      };
      ctx.filter = 'blur(1.8px)';
      ctx.strokeStyle = `rgba(52,15,22,${.28 * seam})`;
      ctx.lineWidth = Math.max(1, mw * 0.035);
      path(); ctx.stroke();
      ctx.filter = 'blur(.6px)';
      ctx.strokeStyle = `rgba(44,10,16,${.55 * seam})`;
      ctx.lineWidth = Math.max(1, mw * 0.011);
      path(); ctx.stroke();
      ctx.filter = 'none';
      const r = Math.max(1, mw * 0.07);
      for (const P of [a, b]) {
        const g = ctx.createRadialGradient(P.x, P.y, 0, P.x, P.y, r);
        g.addColorStop(0, `rgba(40,8,14,${.36 * seam})`);
        g.addColorStop(1, 'rgba(40,8,14,0)');
        ctx.fillStyle = g;
        ctx.fillRect(P.x - r, P.y - r, r * 2, r * 2);
      }
      ctx.restore();
    }
    return true;
  }
  /* ====================================================================
     ORAL CAVITY
     A photo gives the puppet one mouth: the one it was photographed with,
     usually closed.  Everything behind the lips therefore has to be
     drawn, and drawn in a fixed order or it reads as stickers on a face:

        back of the mouth  ->  tongue  ->  lower teeth  ->  upper teeth  ->
        lips

     The cavity itself is measured from the puppet's own lips every frame:
     the aperture rim (and, where the photo happens to show the mouth open
     — a smile, a laugh, a speaking photo — the *real* teeth the detector
     found), then everything is drawn as a small oriented volume behind
     that rim with per-vertex Lambert shading.  That shading is the whole
     difference between "teeth drawn on a photo" and teeth standing in a
     mouth: the upper arch is lit from above, the tongue catches light at
     the front, and the throat behind is nearly black.
     ==================================================================== */

  /* ------------------------------------------------ the photo's own mouth */
  function buildDental(S) {
    const { head } = S;
    const mouth = head.mouth || (head.mouth = E.mouthModel(head.P));
    const a = head.P[61], b = head.P[291], ui = head.P[13], lo = head.P[14];
    const w = mouth.w;
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const ax = [Math.cos(ang), Math.sin(ang), 0];
    /* `ay` points down the face inside the mouth frame */
    let ay = v3.sub(lo, ui);
    ay = v3.norm([ay[0], ay[1], ay[2] || 0]);
    if (ay[1] < 0) ay = [-ay[0], -ay[1], -ay[2]];
    const az = v3.norm(v3.cross(ax, ay));                 // out of the face
    const org = [(ui[0] + lo[0]) / 2, (ui[1] + lo[1]) / 2, (ui[2] + lo[2]) / 2];
    /* how much of the photo's own mouth interior is visible: a wide smile
       with a dark gap shows the puppet's real teeth, and where it does the
       drawn arches step back instead of covering them               */
    const openNorm = mouth.deep / w;
    const seen = clamp01((openNorm - 0.03) / 0.12);
    /* a sample of the photo inside the mouth tells us how bright the real
       interior is, so drawn enamel can be graded to match the photo */
    let inner = 0.62;
    try {
      const c = readAt(S, org[0], org[1], 7);
      inner = clamp((0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255, 0.05, 1);
    } catch (e) { /* the photo is not readable: keep the default */ }
    /* the photo's own lip colour, sampled at the middle of the lower lip:
       the drawn mucosa is painted from it, so the mouth keeps the colour
       of the face around it instead of turning into lipstick           */
    try {
      const p = head.P[17];
      S.lipTone = readAt(S, p[0], p[1] - head.fh * 0.006, 5);
    } catch (e) { S.lipTone = null; }
    /* the lower inner lip at rest, in the mouth frame: the lower arch is
       offset from it by however far the lip has moved since */
    const loMid = [(head.P[17][0] + head.P[314][0]) / 2, (head.P[17][1] + head.P[314][1]) / 2];
    const rx = loMid[0] - org[0], ry = loMid[1] - org[1];
    S.dental = { org, ax, ay, az, w, ang, seen, inner, driverTeeth: seen > 0.35,
                 restLoV: rx * ay[0] + ry * ay[1] };
  }
  /* upper/lower inner lip runs, resampled to the same parameter so index k
     of either run is the same place on the mouth                       */
  function lipRuns(D, n) {
    const res = ids => FM.resampleCurve(FM.splineOpen(ids.map(i => D[i]), 48), n);
    return { up: res(E.LIP_SETS.upIn), lo: res(E.LIP_SETS.loIn),
             upOut: res(E.LIP_SETS.upOut), loOut: res(E.LIP_SETS.loOut) };
  }

  function buildArches() {
    return {
      /* incisal width of each tooth, in mouth widths; the five per side add
         up to just under half a mouth, so the arches close exactly at the
         corners instead of hanging outside the lips */
      TW: [.140, .096, .094, .076, .050],
      TH: [.92, .82, .88, .70, .58],          // crown length, relative
      DEPTH: [0, .22, .55, .95, 1.25]         // how far back each tooth sits
    };
  }
  const archPoint = (d, u, v, z) => [
    d.org[0] + d.ax[0] * u + d.ay[0] * v + d.az[0] * z,
    d.org[1] + d.ax[1] * u + d.ay[1] * v + d.az[1] * z,
    d.org[2] + d.ax[2] * u + d.ay[2] * v + d.az[2] * z
  ];

  /* ------------------------------------------------------------ the mouth */
  /* One pass that paints the entire mouth interior in one go: back of the
     mouth, tongue, both dental arches, then the lips over the top.  Each
     vertex carries its own depth and normal, so the surface turns and
     foreshortens with the head exactly like the rest of the face — this is
     the "animated like the rest of the face" requirement, and it is why
     nothing here is drawn as a flat sticker.                          */
  function drawMouth(S, ctx, st, C, D, mf) {
    const head = S.head;
    /* `mf` is the mouth as it is right now (after the jaw and the lips have
       been posed); head.mouth is the rest pose used to place the arches. */
    const M = mf || head.mouth;
    const zc = st.depthGain || 0;
    const proj3 = p => S.proj([p[0], p[1], p[2] * (1 + zc)]);
    const runs = lipRuns(D, 13);
    /* the aperture rim, closed by the two corner points */
    const rim = runs.up.concat(runs.lo.slice().reverse());
    const rimW = bounds(rim);
    const open = clamp01((M.deep / M.w - 0.03) / 0.28);
    S.openEnv = FM.envelope(S.openEnv, open, st.dt || 1 / 60, 17, 9);
    const o = S.openEnv;
    /* One scalar carries everything that has to fade in and out with the
       mouth: never a threshold, so there is no frame where a mouth is
       suddenly "open enough" and a photograph's teeth appear.         */
    const seen = clamp01((S.dental ? S.dental.seen : 0) * (1 - 0.55 * o));
    const teeth = st.teeth || {};
    const vis = clamp01(teeth.vis === undefined ? 0.4 : teeth.vis);
    const enamel = clamp01((0.55 + S.cache.lum * 0.62) * (teeth.bright === undefined ? 1.1 : teeth.bright));
    const warmth = clamp01(teeth.warm || 0);

    const tongue = clamp01(st.tongue || 0);
    const tongueOut = clamp01((tongue - 0.25) / 0.75);
    const rimOK = rimW.w > 2 && rimW.h > 0.6 && o > 0.004;
    const pad = Math.max(1, M.w * 0.04);
    if (!rimOK) {
      /* Lips together.  Nothing belongs inside the aperture, but a tongue
         that is deliberately pushed out still has to leave the mouth, so
         that one pass is kept and everything else is skipped.         */
      if (tongueOut > 0.12) drawTongueSurface(S, ctx, C, runs, o, tongue, true);
      return { h: o, drawn: 0, upper: 0, lower: 0, seen, tongueOut };
    }
    ctx.save();
    ctx.beginPath();
    rim.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
    ctx.clip();
    const box = { x0: rimW.x0 - pad, y0: rimW.y0 - pad, w: rimW.w + pad * 2, h: rimW.h + pad * 2 };

    /* ---- 1. the puppet's own mouth interior, when the photo shows one -- */
    if (seen > 0.02) {
      ctx.save();
      ctx.globalAlpha = seen * 0.92;
      try {
        const src = S.cache.sharpen || S.src;
        ctx.drawImage(src, box.x0, box.y0, box.w, box.h, box.x0, box.y0, box.w, box.h);
      } catch (e) { /* not fatal: the procedural cavity below still covers it */ }
      ctx.restore();
    }

    /* ---- 2. back of the mouth ----------------------------------------
       Not a black disc: the soft palate is a warm dark red, the throat
       behind the tongue goes nearly black, and the inner cheeks catch a
       little bounce light at the corners.  The three together are what
       make the aperture read as a cavity with a back to it.          */
    const rx = bow(runs.up, 0.5), rl = bowAt(runs.lo, 0.5);
    const mid = [(rx[0] + rl[0]) / 2, (rx[1] + rl[1]) / 2];
    const bg = ctx.createLinearGradient(0, rimW.y0, 0, rimW.y0 + rimW.h);
    bg.addColorStop(0, shade(58, 18, 20, 1));
    bg.addColorStop(0.20, shade(34, 10, 12, 1));
    bg.addColorStop(0.46, shade(14, 4, 5, 1));
    bg.addColorStop(0.72, shade(20, 6, 7, 1));
    bg.addColorStop(1, shade(34, 10, 12, 1));
    ctx.fillStyle = bg;
    ctx.fillRect(box.x0, box.y0, box.w, box.h);
    /* the arch of the soft palate, just inside the upper lip */
    const pg = ctx.createLinearGradient(0, rimW.y0, 0, rimW.y0 + rimW.h * 0.42);
    pg.addColorStop(0, `rgba(96,30,34,${0.45 * (0.4 + o * 0.6)})`);
    pg.addColorStop(1, 'rgba(96,30,34,0)');
    ctx.fillStyle = pg;
    ctx.fillRect(box.x0, rimW.y0, box.w, rimW.h * 0.5);
    /* the throat: a soft well behind the middle of the aperture */
    const vr = Math.max(M.w * 0.20, M.deep * 0.55);
    const vg = ctx.createRadialGradient(mid[0], mid[1] - vr * 0.15, 1, mid[0], mid[1], vr);
    vg.addColorStop(0, `rgba(3,0,1,${0.50 + 0.42 * o})`);
    vg.addColorStop(0.6, `rgba(3,0,1,${0.22 + 0.25 * o})`);
    vg.addColorStop(1, 'rgba(3,0,1,0)');
    ctx.fillStyle = vg;
    ctx.fillRect(box.x0, box.y0, box.w, box.h);
    /* the corners: the commissure always creases, so a pool of shadow
       sits at each end of the aperture and keeps it from reading as a
       cut-out                                                            */
    for (const k of [0, 1]) {
      const p = bow(runs.up, k === 0 ? 0.02 : 0.98);
      const rr = M.w * 0.11;
      const cg = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], rr);
      cg.addColorStop(0, 'rgba(40,6,12,.75)');
      cg.addColorStop(1, 'rgba(40,6,12,0)');
      ctx.fillStyle = cg;
      ctx.fillRect(p[0] - rr, p[1] - rr, rr * 2, rr * 2);
    }
    /* bounce light off the inner cheeks, so the corners are not holes */
    for (const k of [0.10, 0.90]) {
      const p = bow(runs.up, k);
      const rr = M.w * 0.16;
      const cg = ctx.createRadialGradient(p[0], (p[1] + mid[1]) / 2, 0, p[0], (p[1] + mid[1]) / 2, rr);
      cg.addColorStop(0, `rgba(150,74,64,${0.16 + 0.16 * o})`);
      cg.addColorStop(1, 'rgba(150,74,64,0)');
      ctx.fillStyle = cg;
      ctx.fillRect(p[0] - rr, (p[1] + mid[1]) / 2 - rr, rr * 2, rr * 2);
    }

    /* ---- 3. tongue --------------------------------------------------- */
    drawTongueSurface(S, ctx, C, runs, o, tongue, false);

    /* ---- 4. dental arches --------------------------------------------
       The maxilla does not move, so the upper arch stays where the photo
       put it; the mandible does, so the lower arch is offset by however far
       the lower lip has travelled from the rest pose (which already
       includes the hinge).  That is why the teeth no longer grow with the
       mouth: opening moves one arch, not both.                        */
    const lift = clamp01(teeth.lift === undefined ? 0.35 : teeth.lift);
    /* Once the mouth is open the arches are the brightest thing inside it,
       so they are painted at full opacity: `vis` and `seen` only decide
       how much of them the photo already shows, never how ghostly they
       end up looking.                                                  */
    const tv = teeth.vis === undefined ? 0.9 : clamp01(0.55 + teeth.vis * 0.45);
    const upper = clamp01((0.10 + 0.90 * o) * tv * (1 - seen * 0.62));
    const lower = clamp01((0.06 + 0.94 * o) * tv * (1 - seen * 0.72));
    const restLo = S.dental ? S.dental.restLoV : 0;
    const loV = frameUV(S.dental, runs.loOut[6]).v;
    const drop = clamp((loV - restLo) * (S.jaw >= 0 ? 1 : 0.4), -M.w * 0.1, M.w * 0.65);
    const arch = drawArchNew(S, ctx, proj3, S.dental, 'up', { alpha: upper, enamel, warmth, o, lift, M,
      bite: M.w * 0.022 });                       // a little lower than the rim, so the crowns show
    const archLo = drawArchNew(S, ctx, proj3, S.dental, 'lo', { alpha: lower, enamel, warmth, o, lift, M, bite: drop });

    /* ---- 5. contact shadow under the upper lip, and the wet line ----- */
    if (arch.alphaSum > 0.05 || o > 0.05) {
      const sh = ctx.createLinearGradient(0, rimW.y0 - M.w * 0.02, 0, rimW.y0 + M.w * 0.10);
      sh.addColorStop(0, `rgba(20,4,6,${0.5 * (0.4 + o)})`);
      sh.addColorStop(1, 'rgba(20,4,6,0)');
      ctx.fillStyle = sh;
      ctx.fillRect(box.x0, rimW.y0 - M.w * 0.03, box.w, M.w * 0.16);
    }
    /* saliva: a thin specular line where the lower lip meets the teeth */
    if (o > 0.05) {
      ctx.save();
      ctx.globalAlpha = 0.30 * o;
      ctx.strokeStyle = 'rgba(255,240,238,.9)';
      ctx.lineWidth = Math.max(0.7, M.w * 0.012);
      ctx.beginPath();
      runs.lo.forEach((p, i) => i ? ctx.lineTo(p[0], p[1] - M.w * 0.012) : ctx.moveTo(p[0], p[1] - M.w * 0.012));
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();
    /* a tongue that is out past the lips is drawn over the lower lip, with
       the shadow the lip casts on it, so "tongue out" reads as protrusion
       rather than as a red shape that stops at the lip line */
    if (tongueOut > 0.12) drawTongueSurface(S, ctx, C, runs, o, tongue, true);
    return { h: o, drawn: arch.drawn + archLo.drawn, upper: upper, lower: lower, seen, tongueOut };
  }
  const shade = (r, g, b, a) => `rgba(${r | 0},${g | 0},${b | 0},${clamp01(a).toFixed(3)})`;
  function bow(run, t) {
    return run[Math.round(clamp01(t) * (run.length - 1))];
  }
  const bowAt = bow;

  /* ------------------------------------------------------------- arches */
  /* One rigid dental arch, placed in `d` (built from the puppet's own
     mouth) with depth-aware shading.  Teeth are quads with a real world
     position: the near incisors are drawn last and larger, the molars
     disappear around the corner of the arch.                          */
  function drawArchNew(S, ctx, proj3, d, which, cfg) {
    const { TW, TH, DEPTH } = S.arches;
    const isUp = which === 'up';
    const M = cfg.M;
    /* the occlusal line comes from the caller: the maxilla is fixed to the
       skull and the mandible rides the jaw, and that is the whole reason
       the teeth stay the same size while the mouth opens               */
    const bite = cfg.bite;
    const crownH = M.w * (isUp ? 0.112 : 0.100);
    const list = [];
    for (let sd = 0; sd < 2; sd++) {
      let cum = 0;
      for (let j = 0; j < 5; j++) {
        const crown = crownH * TH[j];
        const su = sd ? 1 : -1;
        const uIn = cum * su, uOut = (cum + TW[j]) * su;
        const z = M.w * 0.02 - M.w * 0.34 * (DEPTH[j] + 0.15);
        const curve = M.w * 0.06 * DEPTH[j];
        const vInc = bite;
        const vGum = bite + (isUp ? -crown : crown);
        const c0 = archPoint(d, uIn * M.w, vGum, z - curve * 0.4);
        const c1 = archPoint(d, uOut * M.w, vGum, z - curve * 1.4);
        const c2 = archPoint(d, uIn * M.w, vInc, z);
        const c3 = archPoint(d, uOut * M.w, vInc, z - curve);
        const q = [proj3(c0), proj3(c1), proj3(c2), proj3(c3)];
        if (!q.every(v => isFinite(v[0]) && isFinite(v[1]))) { cum += TW[j]; continue; }
        /* outward normal of this tooth's labial face, in world space */
        let n = v3.cross(v3.sub(c1, c0), v3.sub(c2, c0));
        n = v3.norm(n);
        if (v3.dot(n, d.az) < 0) n = [-n[0], -n[1], -n[2]];
        list.push({ q, j, sd, back: DEPTH[j], n, crown, mid: [(q[0][0] + q[1][0]) / 2, (q[0][1] + q[1][1]) / 2] });
        cum += TW[j];
      }
    }
    /* back teeth first, so the arch reads as a curve and not as a fan */
    list.sort((a, b) => b.back - a.back);
    const ivory = v => `rgb(${Math.round(clamp(248 * v, 0, 255))},${Math.round(clamp(238 * v, 0, 255))},${Math.round(clamp(219 * v, 0, 255))})`;
    let alphaSum = 0, drawn = 0, frontY = null, frontZ = Infinity;
    for (const t of list) {
      const [A0, B0, C0, D0] = t.q;                    // A,B: gum line.  C,D: incisal edge
      const alpha = clamp01(cfg.alpha * (1 - 0.10 * t.back));
      if (alpha <= 0.02) continue;
      alphaSum += alpha;
      /* Lambert on the tooth's own face, so the arch turns away with the
         head instead of being a flat sticker */
      const lam = 0.40 + 0.60 * clamp01(0.50 + 0.55 * v3.dot(t.n, NORM_LIGHT));
      const lum0 = lam * cfg.enamel * (1 - t.back * 0.16);
      const tone = clamp(lum0, 0.30, 1.16);
      /* a couple of percent of inset in `u` separates each tooth from its
         neighbour; that gap, not a drawn outline, is the contact           */
      const ins = (p, q, k) => [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k];
      const gA = ins(A0, B0, 0.012), gB = ins(B0, A0, 0.012);
      const iA = ins(C0, D0, 0.022), iB = ins(D0, C0, 0.022);
      const im = [(iA[0] + iB[0]) / 2, (iA[1] + iB[1]) / 2];
      /* the incisal edge bows away from the gum line: a rounded biting
         edge, the way a real incisor's mesioincisal corners are rounded */
      const ctl = [im[0], im[1] + (isUp ? 1 : -1) * t.crown * 0.07];
      const path = () => {
        ctx.beginPath();
        ctx.moveTo(gA[0], gA[1]);
        ctx.lineTo(iA[0], iA[1]);
        ctx.quadraticCurveTo(ctl[0], ctl[1], iB[0], iB[1]);
        ctx.lineTo(gB[0], gB[1]);
        ctx.closePath();
      };
      ctx.save();
      ctx.globalAlpha = alpha;
      path();
      /* crown shading: the gum end is tucked in shade under the lip, the
         biting edge is the brightest part of the tooth, as it is in life */
      const gg = ctx.createLinearGradient(A0[0], A0[1], C0[0], C0[1]);
      gg.addColorStop(0, ivory(tone * 0.60));
      gg.addColorStop(0.30, ivory(tone * 0.90));
      gg.addColorStop(0.86, ivory(tone * 1.05));
      gg.addColorStop(1, ivory(tone * 1.12));
      ctx.fillStyle = gg;
      ctx.fill();
      /* and across its width, so a crown reads as round rather than flat */
      const ug = ctx.createLinearGradient(A0[0], A0[1], B0[0], B0[1]);
      ug.addColorStop(0, 'rgba(66,44,30,.30)');
      ug.addColorStop(0.26, 'rgba(66,44,30,0)');
      ug.addColorStop(0.70, 'rgba(255,252,244,.16)');
      ug.addColorStop(1, 'rgba(66,44,30,.32)');
      ctx.fillStyle = ug;
      ctx.fill();
      /* the enamel grain, still in the tooth's own frame so it turns with
         it — kept faint, because a texture magnified four times is mud  */
      if (t.j < 3) {
        const ex = [B0[0] - A0[0], B0[1] - A0[1]];
        const ey = [C0[0] - A0[0], C0[1] - A0[1]];
        ctx.save();
        ctx.clip();
        ctx.transform(ex[0] / 40, ex[1] / 40, ey[0] / 90, ey[1] / 90, A0[0], A0[1]);
        ctx.globalAlpha = alpha * 0.20;
        try { ctx.drawImage(S.cache.enamel, (t.j * 29 + (t.sd ? 13 : 0)) % 88, 0, 40, 90, 0, 0, 40, 90); } catch (e) {}
        ctx.restore();
      }
      /* the lip above the arch throws its shadow over the top of the
         crown — without this the teeth look pasted into the mouth */
      const cast = isUp ? clamp01(0.10 + 0.26 * (cfg.o || 0)) : clamp01(0.08 + 0.14 * (cfg.o || 0));
      if (cast > 0.02) {
        const cg = ctx.createLinearGradient(A0[0], A0[1], C0[0], C0[1]);
        cg.addColorStop(0, `rgba(34,8,10,${cast})`);
        cg.addColorStop(isUp ? 0.42 : 0.34, 'rgba(34,8,10,0)');
        cg.addColorStop(1, 'rgba(34,8,10,0)');
        ctx.fillStyle = cg;
        ctx.fill();
      }
      /* a warm tone in the mouth's ambient light, and the finest dark
         outline that still reads as an edge                                  */
      if (cfg.warmth > 0.01) { ctx.fillStyle = `rgba(198,152,74,${cfg.warmth * 0.16})`; ctx.fill(); }
      ctx.strokeStyle = `rgba(84,54,40,${0.18 * clamp01(0.5 + t.back)})`;
      ctx.lineWidth = Math.max(0.4, M.w * 0.0035);
      path();
      ctx.stroke();
      /* a thin specular along the biting edge: enamel is the wettest,
         glossiest surface in the picture apart from the eyes          */
      ctx.strokeStyle = `rgba(255,255,252,${0.20 * clamp01(lum0 - 0.25)})`;
      ctx.lineWidth = Math.max(0.5, M.w * 0.006);
      ctx.beginPath();
      ctx.moveTo(iA[0], iA[1]);
      ctx.quadraticCurveTo(ctl[0], ctl[1], iB[0], iB[1]);
      ctx.stroke();
      ctx.restore();
      drawn++;
      /* the incisal edge of the nearest tooth in this arch: the mouth's
         own geometry uses it, so it is reported back to the caller      */
      const tipY = isUp ? Math.max(C0[1], D0[1]) : Math.min(C0[1], D0[1]);
      if (frontY === null || t.back < frontZ) { frontY = tipY; frontZ = t.back; }
    }
    return { alphaSum, drawn, frontY };
  }

  /* the light the mouth is lit by: up, slightly in front and to the left,
     matching the key light used on the face */
  const NORM_LIGHT = v3.norm([-0.32, -0.86, 0.4]);

  /* --------------------------------------------------------------- tongue */
  function drawTongueSurface(S, ctx, C, runs, o, tg, protrude) {
    const M = S.head.mouth;
    const d = S.dental;
    if (!d) return;
    const out = clamp01((tg - 0.25) / 0.75);            // how far it leaves the mouth
    const rise = tg;                                    // general tongue activity
    /* inside the mouth the tongue stops at the aperture; a protruding
       tongue is drawn from the floor of the mouth out past the lower lip */
    const len = protrude ? M.w * (0.30 + 0.75 * out) : M.w * (0.45 + 0.55 * out);
    const halfW = M.w * (0.30 + 0.05 * out);
    /* The tongue rides the mandible, so its surface is measured against
       the lower lip as it is *now*: rows run from under the biting line
       down to just short of the lower lip, which is the floor of the
       mouth.  Anchoring them to the rest pose instead made the tongue a
       strip at the top of the opening that never grew with the jaw.   */
    const loV = runs && runs.lo ? frameUV(d, runs.lo[Math.floor(runs.lo.length / 2)]).v : M.deep * 0.5;
    const vTop = Math.max(1.5, M.deep * 0.10);
    const vBot = Math.max(vTop + 2.5, loV - M.w * 0.020);
    const seg = 8;
    const rows = [];
    for (let s = 0; s <= seg; s++) {
      const t = s / seg;
      const w = halfW * (0.52 + 0.52 * Math.sin(Math.PI * Math.min(1, 0.30 + t * 0.70)));
      const v = lerp(vTop, vBot, t) - rise * M.w * 0.03
        + (protrude ? t * (M.deep * 0.90 + M.w * 0.34 * out) : 0);
      const z = M.w * 0.02 + t * len;
      const l = archPoint(d, -w, v, z), r = archPoint(d, w, v, z), m = archPoint(d, 0, v, z);
      rows.push({ l: S.proj(l), r: S.proj(r), m: S.proj(m), w, t, z });
    }
    if (!rows.every(s => isFinite(s.l[0]) && isFinite(s.m[0]) && isFinite(s.r[0]))) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(rows[0].l[0], rows[0].l[1]);
    for (let i = 1; i < rows.length; i++) ctx.lineTo(rows[i].l[0], rows[i].l[1]);
    for (let i = rows.length - 1; i >= 0; i--) ctx.lineTo(rows[i].r[0], rows[i].r[1]);
    ctx.closePath();
    const first = rows[0].m, last = rows[rows.length - 1].m;
    const g = ctx.createLinearGradient(first[0], first[1], last[0], last[1]);
    g.addColorStop(0, 'rgba(86,32,42,1)');           // root, under the palate
    g.addColorStop(0.30, 'rgba(136,58,64,1)');
    g.addColorStop(0.68, 'rgba(172,88,90,1)');
    g.addColorStop(1, 'rgba(188,108,104,1)');        // the tip catches the light
    ctx.fillStyle = g;
    ctx.fill();
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(rows[0].l[0], rows[0].l[1]);
    for (let i = 1; i < rows.length; i++) ctx.lineTo(rows[i].l[0], rows[i].l[1]);
    for (let i = rows.length - 1; i >= 0; i--) ctx.lineTo(rows[i].r[0], rows[i].r[1]);
    ctx.closePath();
    ctx.clip();
    /* the sides of the tongue fall away into the floor of the mouth, so
       they are darker than the crest: that curvature is most of what
       makes it read as a tongue and not as a pink shape             */
    const side = ctx.createLinearGradient(first[0] - halfW, first[1], first[0] + halfW, first[1]);
    side.addColorStop(0, 'rgba(58,14,24,.55)');
    side.addColorStop(0.28, 'rgba(58,14,24,0)');
    side.addColorStop(0.72, 'rgba(58,14,24,0)');
    side.addColorStop(1, 'rgba(58,14,24,.55)');
    ctx.fillStyle = side;
    ctx.fillRect(first[0] - halfW * 1.2, Math.min(first[1], last[1]) - halfW,
                 halfW * 2.4, Math.abs(last[1] - first[1]) + halfW * 2);
    /* the midline groove + papilla: two soft strokes, not a drawn line */
    ctx.strokeStyle = 'rgba(104,26,40,.42)';
    ctx.lineWidth = Math.max(1, halfW * 0.16);
    ctx.beginPath();
    ctx.moveTo(first[0], first[1]);
    for (let i = 1; i < rows.length; i++) ctx.lineTo(rows[i].m[0], rows[i].m[1]);
    ctx.stroke();
    /* the papillae: two soft lateral blooms, never dots — at life size a
       stipple reads as dirt on the tongue, not as texture              */
    ctx.save();
    ctx.filter = 'blur(3px)';
    for (const sgn of [-1, 1]) {
      const x = lerp(first[0], last[0], 0.42) + sgn * halfW * 0.42;
      const y = lerp(first[1], last[1], 0.42) + halfW * 0.18;
      const pg = ctx.createRadialGradient(x, y, 0, x, y, halfW * 0.7);
      pg.addColorStop(0, 'rgba(255,214,214,.16)');
      pg.addColorStop(1, 'rgba(255,214,214,0)');
      ctx.fillStyle = pg;
      ctx.fillRect(x - halfW, y - halfW, halfW * 2, halfW * 2);
    }
    ctx.restore();
    /* a wet specular band down the middle */
    const sg = ctx.createLinearGradient(first[0], first[1] - halfW * 0.6, first[0], first[1] + halfW * 0.6);
    sg.addColorStop(0, 'rgba(255,235,235,0)');
    sg.addColorStop(0.46, `rgba(255,242,242,${0.24 + 0.20 * o})`);
    sg.addColorStop(1, 'rgba(255,235,235,0)');
    ctx.fillStyle = sg;
    ctx.fillRect(Math.min(first[0], last[0]) - halfW * 2, Math.min(first[1], last[1]) - halfW,
                 Math.abs(last[0] - first[0]) + halfW * 4, Math.abs(last[1] - first[1]) + halfW * 2);
    /* the lower lip's shadow on the part of the tongue that is outside */
    if (protrude) {
      const lr = runs.lo[Math.floor(runs.lo.length / 2)];
      const sg2 = ctx.createLinearGradient(lr[0], lr[1], lr[0], lr[1] + M.w * 0.16);
      sg2.addColorStop(0, 'rgba(46,10,16,.42)');
      sg2.addColorStop(1, 'rgba(46,10,16,0)');
      ctx.fillStyle = sg2;
      ctx.fillRect(Math.min(first[0], last[0]) - halfW * 2, lr[1], Math.abs(last[0] - first[0]) + halfW * 4, M.w * 0.2);
    }
    /* deepest shadow where the tongue meets the back of the mouth */
    const dh = ctx.createLinearGradient(first[0], first[1], last[0], last[1]);
    dh.addColorStop(0, 'rgba(30,4,8,.55)');
    dh.addColorStop(0.5, 'rgba(30,4,8,0)');
    ctx.fillStyle = dh;
    ctx.fillRect(Math.min(first[0], last[0]) - halfW * 2, Math.min(first[1], last[1]) - halfW,
                 Math.abs(last[0] - first[0]) + halfW * 4, Math.abs(last[1] - first[1]) + halfW * 2);
    ctx.restore();
    ctx.restore();
  }

  /* ------------------------------------------------------------- lips */
  /* The photo already has lips, and they are the most recognisable thing
     in a portrait — so they are never repainted.  What is drawn is what a
     photograph cannot contain: the inner surface of the lips once they
     part, rolled back to meet the teeth, plus the shading that makes the
     aperture read as a hole rather than a dark shape.                 */
  function drawLipShell(S, ctx, C, D, runs, o, mouthNow) {
    const M = mouthNow || S.mouth || S.head.mouth;
    const d = S.dental;
    const zc = S.depthGain || 0;
    const proj3 = p => S.proj([p[0], p[1], p[2] * (1 + zc)]);
    if (o < 0.01 || !d) return 0;
    const N = runs.up.length;
    /* The mucosa of each lip is only a few millimetres deep and it rolls
       back INTO the mouth: away from the viewer (-z) and inward past the
       rim.  Anything drawn in front of the rim reads as lipstick floating
       over the face, which is exactly what this used to look like.  */
    const depth = M.w * (0.035 + 0.055 * o);
    /* the mucosa fills the vermilion the deform opened and rolls only a
       hair past the rim — roll further in and it covers the teeth, which
       belong to the mouth, not to the lip                              */
    const band = M.w * (0.010 + 0.016 * o);
    const top = [], bot = [];
    for (let k = 0; k < N; k++) {
      const t = k / (N - 1);
      const taper = 0.30 + 0.70 * Math.sin(Math.PI * clamp01(0.06 + t * 0.88));
      const up = runs.up[k], lo = runs.lo[k];
      const uo = runs.upOut[k], oo = runs.loOut[k];
      const uu = frameUV(d, up), ul = frameUV(d, lo);
      const qi = proj3(archPoint(d, uu.u * 0.97, uu.v + band * taper, -depth * taper));
      top.push([uo[0], uo[1], qi[0], qi[1]]);
      const qj = proj3(archPoint(d, ul.u * 0.97, ul.v - band * taper, -depth * taper));
      bot.push([oo[0], oo[1], qj[0], qj[1]]);
    }
    const strip = (row, dir) => {
      ctx.beginPath();
      ctx.moveTo(row[0][0], row[0][1]);
      for (let k = 1; k < row.length; k++) ctx.lineTo(row[k][0], row[k][1]);
      for (let k = row.length - 1; k >= 0; k--) ctx.lineTo(row[k][2], row[k][3]);
      ctx.closePath();
      const mid = row[Math.floor(row.length / 2)];
      /* the outer edge keeps whatever colour the photograph's own lip has
         there, so the drawn mucosa is the person's lip colour and not a
         lipstick someone else chose                                    */
      const lip0 = S.lipTone;
      const outer = lip0
        ? `rgb(${clamp(lip0[0] * 0.94, 0, 255) | 0},${clamp(lip0[1] * 0.86, 0, 255) | 0},${clamp(lip0[2] * 0.86, 0, 255) | 0})`
        : 'rgba(150,56,62,1)';
      const g = ctx.createLinearGradient(mid[0], mid[1], mid[2], mid[3]);
      g.addColorStop(0, outer);
      g.addColorStop(0.55, 'rgba(118,36,44,1)');
      g.addColorStop(1, 'rgba(64,16,22,1)');
      ctx.fillStyle = g;
      ctx.fill();
      /* the mucosa is wet: a thin highlight along its crest */
      ctx.strokeStyle = `rgba(255,205,205,${0.10 + 0.10 * o})`;
      ctx.lineWidth = Math.max(0.6, M.w * 0.010);
      ctx.beginPath();
      row.forEach((p, k) => k ? ctx.lineTo(p[0], p[1] - dir * M.w * 0.006) : ctx.moveTo(p[0], p[1] - dir * M.w * 0.006));
      ctx.stroke();
    };
    /* lower lip first (its inner surface is nearer), then the upper */
    strip(bot, -1);
    strip(top, 1);
    return 1;
  }
  /* a screen point expressed in the mouth's own frame (u across, v down) */
  function frameUV(d, p) {
    const rx = p[0] - d.org[0], ry = p[1] - d.org[1];
    return { u: rx * d.ax[0] + ry * d.ax[1], v: rx * d.ay[0] + ry * d.ay[1] };
  }

  /* ------------------------------------------------------- shading the lips */
  function lipShading(S, ctx, C, D, runs, o, mouthNow) {
    const M = mouthNow || S.mouth || S.head.mouth;
    const teeth = S.dental ? S.dental : null;
    /* 1. the seam: a dark line between the lips when they are together */
    const seam = clamp01(1 - o * 3.2);
    if (seam > 0.02) {
      ctx.save();
      ctx.lineCap = 'round';
      ctx.filter = 'blur(1.1px)';
      ctx.strokeStyle = `rgba(58,14,20,${0.42 * seam})`;
      ctx.lineWidth = Math.max(0.9, M.w * 0.013);
      ctx.beginPath();
      runs.up.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.stroke();
      ctx.filter = 'blur(.5px)';
      ctx.strokeStyle = `rgba(30,6,10,${0.32 * seam})`;
      ctx.lineWidth = Math.max(0.6, M.w * 0.007);
      ctx.stroke();
      ctx.restore();
    }
    /* 2. the corners: a small pool of shadow, deeper when the mouth is
          stretched into a smile, which is where a real corner creases */
    const corner = M.w * (0.075 + 0.05 * o);
    for (const k of [0, runs.up.length - 1]) {
      const p = runs.up[k];
      const g = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], corner);
      g.addColorStop(0, `rgba(48,10,16,${0.42 + 0.16 * o})`);
      g.addColorStop(1, 'rgba(48,10,16,0)');
      ctx.fillStyle = g;
      ctx.fillRect(p[0] - corner, p[1] - corner, corner * 2, corner * 2);
    }
    /* 3. the shadow the upper lip casts on the lower one, and the contact
          shadow under the lower lip — these are what make parted lips look
          like they have depth instead of being a cut-out */
    if (o > 0.05 && teeth) {
      ctx.save();
      ctx.globalAlpha = clamp01(o * 1.2);
      const g = ctx.createLinearGradient(0, runs.up[6][1], 0, runs.up[6][1] + M.w * 0.11);
      g.addColorStop(0, 'rgba(40,8,12,.30)');
      g.addColorStop(1, 'rgba(40,8,12,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, runs.up[6][1], S.W, M.w * 0.11);
      const g2 = ctx.createLinearGradient(0, runs.lo[6][1] - M.w * 0.09, 0, runs.lo[6][1]);
      g2.addColorStop(0, 'rgba(40,8,12,0)');
      g2.addColorStop(1, 'rgba(40,8,12,.26)');
      ctx.fillStyle = g2;
      ctx.fillRect(0, runs.lo[6][1] - M.w * 0.09, S.W, M.w * 0.09);
      ctx.restore();
    }
  }

  /* the wet gloss on the existing photograph lips: a restrained highlight,
     present when the mouth is closed and fading as it opens            */
  function drawLips(S, ctx, D, mw, gap, mouthNow) {
    const M = mouthNow || S.mouth || S.head.mouth;
    const a = D[61], b = D[291];
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    const o = clamp01((gap / (mw || 1) - 0.045) / 0.30);
    const seam = clamp01(1 - o * 2.6);
    const bb = bounds(E.MOUTH_RING.map(i => D[i]));
    if (!(bb.w > 2 && bb.h > 1) || !FM.polyOK(E.MOUTH_RING.map(i => D[i]), 4)) return false;
    ctx.save();
    ctx.beginPath();
    E.MOUTH_RING.forEach((i, k) => k ? ctx.lineTo(D[i].x, D[i].y) : ctx.moveTo(D[i].x, D[i].y));
    ctx.closePath();
    ctx.clip();
    const gl = (cx, cy, rw, rh, al) => {
      if (!(rw > 0.6 && rh > 0.2 && al > 0.01)) return;
      ctx.save();
      ctx.translate(cx, cy); ctx.rotate(ang); ctx.scale(1, rh / rw);
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rw);
      g.addColorStop(0, `rgba(255,236,232,${al})`);
      g.addColorStop(1, 'rgba(255,236,232,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(0, 0, rw, 0, 7); ctx.fill();
      ctx.restore();
    };
    const lc = { x: (D[17].x + D[14].x) / 2, y: (D[17].y + D[14].y) / 2 };
    const uc = { x: (D[0].x + D[13].x) / 2, y: (D[0].y + D[13].y) / 2 };
    gl(lc.x, lc.y, M.w * 0.24, M.w * 0.045, 0.26 * seam + 0.05);
    gl(uc.x, uc.y, M.w * 0.15, M.w * 0.030, 0.10 * seam);
    ctx.restore();
    return true;
  }

  function drawShading(S, ctx, D, st) {
    const yaw = E.eulerYaw(st.q);
    const amt = clamp01(Math.abs(yaw) * 0.9 * (st.shade || 0));
    if (amt < 0.012) return;
    const poly = E.CONTOUR.map(i => D[i]);
    if (!polyOK(poly, 20)) return;
    const bb = bounds(poly);
    ctx.save();
    ctx.beginPath();
    poly.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath();
    ctx.clip();
    const left = yaw < 0;
    const g = ctx.createLinearGradient(left ? bb.x0 : bb.x1, 0, left ? bb.x1 : bb.x0, 0);
    g.addColorStop(0, `rgba(22,12,14,${amt})`);
    g.addColorStop(.45, 'rgba(22,12,14,0)');
    g.addColorStop(1, `rgba(255,244,236,${amt * 0.32})`);
    ctx.fillStyle = g;
    ctx.fillRect(bb.x0 - 2, bb.y0 - 2, bb.w + 4, bb.h + 4);
    ctx.restore();
  }

  /* ------------------------------------------------------------- frame */
  function sessionFrame(S, st) {
    if (!S.ready) return null;
    S.frames++;
    const ctx = st.ctx || S.canvas.getContext('2d');
    const lc = S.lctx;
    const C = deform(S, st);
    const D = new Array(478);
    for (let i = 0; i < 478; i++) D[i] = { x: S.DST[i * 2], y: S.DST[i * 2 + 1], z: S.depth[i] };
    S.stats.tris = 0; S.stats.culled = 0;
    /* ---- compose the shell on the layer, then blit once ------------ */
    lc.setTransform(1, 0, 0, 1, 0, 0);
    lc.clearRect(0, 0, S.W, S.H);
    lc.drawImage(S.cache.plate || S.src, 0, 0);
    lc.imageSmoothingQuality = st.quality > 0 ? 'high' : 'low';
    drawShell(S, lc, st, S.triA, { resetOrder: S.frames < 3, padScale: st.quality > 1 ? 1.4 : 1 });
    if (st.quality > 0) drawShell(S, lc, st, S.triB, { resetOrder: S.frames < 3, padScale: st.quality > 1 ? 1.4 : 1 });
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, S.W, S.H);
    let mw = 0, gap = 0, h = S.openEnv, mouthM = null;
    const oral = { cavity: false, lowerTeeth: 0, lowerAlpha: 0, tongue: false, upperTeeth: 0, upperAlpha: 0, lips: false };
    if (st.origOnly) {
      ctx.drawImage(S.src, 0, 0);
    } else {
      ctx.drawImage(S.layer, 0, 0);
      /* ---- face detail: the triangle shell resamples the photo, which
             softens skin and smears the fine detail around the eyes and
             mouth; a light unsharp mask of the same source puts it back so
             the render still looks like a photograph and not a painting */
      if (st.quality > 0 && S.cache.sharpen) {
        ctx.save();
        ctx.globalAlpha = st.quality > 1 ? 0.42 : 0.30;
        ctx.globalCompositeOperation = 'overlay';
        try { ctx.drawImage(S.cache.sharpen, 0, 0); } catch (e) {}
        ctx.restore();
      }
      /* ---- the mouth ------------------------------------------------
             One pass owns the whole mouth: cavity, tongue, both arches and
             the inner lips are ordered and lit from the same frame, so the
             mouth turns, foreshortens and shades like the rest of the face.
             It runs every frame, gated by smooth envelopes.            */
      mw = Math.hypot(S.DST[61 * 2] - S.DST[291 * 2], S.DST[61 * 2 + 1] - S.DST[291 * 2 + 1]) || S.head.fw * 0.4;
      const rawGap = Math.hypot(S.DST[13 * 2] - S.DST[14 * 2], S.DST[13 * 2 + 1] - S.DST[14 * 2 + 1]);
      gap = Math.max(0, rawGap - mw * 0.035);
      mouthM = E.mouthModel(C);
      S.mouth = mouthM;
      const runs = lipRuns(D, 13);
      const mm = drawMouth(S, ctx, st, C, D, mouthM);
      h = mm.h;
      oral.cavity = true;
      oral.upperAlpha = mm.upper; oral.lowerAlpha = mm.lower;
      oral.upperTeeth = oral.lowerTeeth = Math.round(mm.drawn / 2);
      oral.tongue = mm.tongueOut > 0.02;
      drawLipShell(S, ctx, C, D, runs, h, mouthM);
      oral.lips = drawLips(S, ctx, D, mw, gap, mouthM) === true;
      lipShading(S, ctx, C, D, runs, h, mouthM);
      drawIris(S, ctx, 0, D, st.gaze[0], st.gazeGain);
      drawIris(S, ctx, 1, D, st.gaze[1], st.gazeGain);
      drawEyelid(S, ctx, 'L', st.lid[0], D);
      drawEyelid(S, ctx, 'R', st.lid[1], D);
      drawShading(S, ctx, D, st);
    }
    /* ---- overlays -------------------------------------------------- */
    if (st.showPoints && st.points) {
      ctx.save();
      for (let k = 0; k < st.points.length; k++) {
        const def = st.points[k];
        const px = st.origOnly ? S.S2[def.v * 2] : S.DST[def.v * 2];
        const py = st.origOnly ? S.S2[def.v * 2 + 1] : S.DST[def.v * 2 + 1];
        const hot = def.k === st.dragIndex || def.k === st.hoverIndex;
        ctx.beginPath();
        ctx.arc(px, py, hot ? 8 : 4.2, 0, 7);
        ctx.fillStyle = 'rgba(0,0,0,.55)';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, py, hot ? 6 : 2.8, 0, 7);
        ctx.fillStyle = def.c;
        ctx.fill();
      }
      ctx.restore();
    }
    if (st.showMesh) {
      ctx.save();
      ctx.strokeStyle = 'rgba(94,234,212,.30)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      const list = S.triA.list;
      for (let t = 0; t < list.length; t += 3) {
        const a = list[t], b = list[t + 1], c = list[t + 2];
        if (S.mesh.kind[a] === 2 && S.mesh.kind[b] === 2 && S.mesh.kind[c] === 2) continue;
        ctx.moveTo(S.DST[a * 2], S.DST[a * 2 + 1]);
        ctx.lineTo(S.DST[b * 2], S.DST[b * 2 + 1]);
        ctx.lineTo(S.DST[c * 2], S.DST[c * 2 + 1]);
        ctx.closePath();
      }
      ctx.stroke();
      ctx.restore();
    }
    const dbg = { tris: S.stats.tris, culled: S.stats.culled,
      mouth: { mw, gap, h, ...oral, open: mouthM ? mouthM.open : 0, teethAlpha: oral.upperAlpha + oral.lowerAlpha,
               seen: S.dental ? clamp01(S.dental.seen * (1 - 0.55 * h)) : 0, tongue: h > 0.02 ? 1 : 0 },
      jaw: S.jaw };
    S.lastDebug = dbg;
    return dbg;
  }

  root.FMRender = { createSession, sessionInit, sessionFrame, deform, buildPlate, prepIris, makeCanvas,
                    buildArches, buildDental, drawMouth, drawLipShell, lipRuns, buildSharpen,
                    archPoint, drawTongueSurface };
})(typeof globalThis !== 'undefined' ? globalThis : this);
