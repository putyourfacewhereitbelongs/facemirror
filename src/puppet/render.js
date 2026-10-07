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
    /* Only a fraction of a pixel is needed to hide antialiased seams.
       Large padded clips overlap and repeatedly resample the same skin,
       which is especially visible when an emotion moves many triangles. */
    S.triA = mkPass(spec.mesh.base, S.normals, { pad: 0.35, cull: -0.02, mirror: true });
    S.triB = mkPass(spec.mesh.face, S.normalsFace, { pad: 0.28, cull: -0.05, mirror: true });
    S.mid = spec.head.mid.slice();
    S.axis = E.tmjAxis(spec.head);
    S.arches = buildArches();
    prepIris(S);
    buildPlate(S);
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
    const gapNow = Math.hypot(C[13][0] - C[14][0], C[13][1] - C[14][1]);
    const expressionOpen = clamp01(st.jawOpen || 0);
    const want = clamp01(Math.max(gapNow / (head.fh * 0.17), expressionOpen)) * 0.36;
    S.jaw = FM.slew(S.jaw, want, st.dt || 1 / 60, 3.2, 3.6);
    if (jawGain > 0.01) {
      /* Canvas y grows downward.  With the TMJ axis used here, a positive
         mouth-open amount needs the negative right-hand rotation; the old
         sign lifted the lower lip into the upper lip, so webcam jawOpen and
         speaking presets appeared to keep the mouth closed. */
      E.applyJaw(head, C, -S.jaw * jawGain, S.axis);
    }
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
      [E.LIDS.L.up, E.LIDS.L.lo], [E.LIDS.R.up, E.LIDS.R.lo],
      [E.UP_LIP, E.LO_LIP]
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
    /* Frontal frames can stay on the original photograph.  This is the
       important quality guard: a 2.5D triangle warp is never allowed to
       repaint a straight-on face just because the tracker emitted a noisy
       landmark residual.  We still sort/prepare the pass above so its state
       is ready if the head later turns. */
    if (o.skipPaint) return 0;
    const R = matFromQuat(st.q);
    const midX = S.mid[0];
    const pad = pass.pad * (o.padScale || 1);
    let drawn = 0, culled = 0;
    const passAlpha = clamp01(o.alpha === undefined ? 1 : o.alpha);
    for (let t = 0; t < m; t++) {
      const tri = order[t] * 3;
      const a = list[tri], b = list[tri + 1], c = list[tri + 2];
      if (o.onlyChanged) {
        const threshold = o.changeThreshold === undefined ? 0.18 : o.changeThreshold;
        const moved = Math.max(
          Math.hypot(DST[a * 2] - S2[a * 2], DST[a * 2 + 1] - S2[a * 2 + 1]),
          Math.hypot(DST[b * 2] - S2[b * 2], DST[b * 2 + 1] - S2[b * 2 + 1]),
          Math.hypot(DST[c * 2] - S2[c * 2], DST[c * 2 + 1] - S2[c * 2 + 1])
        );
        /* In a frontal expression frame the untouched photograph is already
           the sharpest possible background.  Re-sample only triangles that
           actually moved; repainting every identity triangle is a blur pass. */
        if (!(moved > threshold)) continue;
      }
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
      /* A local sculpt must not fold a source triangle over itself.  The
         affine below can faithfully paint an inverted triangle, but on a
         textured portrait that reads as a large random block.  The source
         winding is unchanged by the far-side mirror plus vertex swap, so a
         sign change here is a real fold, not a legitimate head turn.  The
         plate underneath supplies a calm fallback for that tiny triangle. */
      if (area * det <= 0) continue;
      const k = 1 / det;
      const ma = ((bx - ax) * (s2y - s0y) - (by - ay) * (s2x - s0x)) * k;
      const mb = ((by - ay) * (s1x - s0x) - (bx - ax) * (s1y - s0y)) * k;
      const mc = ((cx - ax) * (s2y - s0y) - (cy - ay) * (s2x - s0x)) * k;
      const md = ((cy - ay) * (s1x - s0x) - (cx - ax) * (s1y - s0y)) * k;
      const me = ax - ma * s0x - mc * s0y;
      const mf = ay - mb * s0x - md * s0y;
      const gx = (ax + bx + cx) / 3, gy = (ay + by + cy) / 3;
      ctx.save();
      ctx.globalAlpha = passAlpha;
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

  /* ------------------------------------------------------- oral cavity */
  function buildArches() {
    return {
      TW: [.17, .12, .115, .085, .05],       // incisal width, in mouth widths
      TH: [.95, .82, .9, .68, .55],          // crown length, relative
      DEPTH: [0, .18, .5, .95, 1.3]          // how far back each tooth sits
    };
  }
  function mouthFrame(head, C) {
    const a = C[61], b = C[291], up = C[13], lo = C[14];
    const ax = v3.norm(v3.sub(b, a));
    let ay = v3.sub(lo, up);
    const pr = v3.dot(ay, ax);
    ay = v3.norm([ay[0] - ax[0] * pr, ay[1] - ax[1] * pr, ay[2] - ax[2] * pr]);
    const az = v3.norm(v3.cross(ax, ay));
    const mw = v3.dist(a, b) || head.fw * 0.4;
    return { ax, ay, az, org: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], mw,
             upY: v3.dot(up, ay), loY: v3.dot(lo, ay) };
  }
  const mfp = (fr, u, v, w) => [
    fr.org[0] + fr.ax[0] * u + fr.ay[0] * v + fr.az[0] * w,
    fr.org[1] + fr.ax[1] * u + fr.ay[1] * v + fr.az[1] * w,
    fr.org[2] + fr.ax[2] * u + fr.ay[2] * v + fr.az[2] * w
  ];
  /* Draw one rigid dental arch and return visibility diagnostics. The
     upper arch is painted after the tongue to restore front-to-back order. */
  function drawArch(S, ctx, proj3, fr, which, openAmt, cfg) {
    /* cfg = { vis, bright, warmth } straight from the sliders */
    const { TW, TH, DEPTH } = S.arches;
    const isUp = which === 'up';
    const line = isUp ? fr.upY + openAmt * 0.16 : fr.loY - openAmt * 0.16;
    const crown = fr.mw * (isUp ? 0.30 : 0.26);
    const teeth = [];
    for (let sd = 0; sd < 2; sd++) {
      let cum = 0;
      for (let j = 0; j < 5; j++) {
        const w = TW[j] * (isUp ? 1 : 0.94);
        const u0 = cum * (sd ? 1 : -1) * fr.mw, u1 = (cum + w) * (sd ? 1 : -1) * fr.mw;
        const z = -fr.mw * 0.13 - fr.mw * 0.26 * DEPTH[j];
        const hh = crown * TH[j] * (1 + (((j * 13 + (sd ? 5 : 0)) % 7) - 3) * 0.015);
        const vBite = line + (isUp ? 0 : 0);
        const vGum = line + (isUp ? -hh : hh);
        const a3 = mfp(fr, u0, vGum, z), b3 = mfp(fr, u1, vGum, z);
        const c3 = mfp(fr, u0, vBite, z), d3 = mfp(fr, u1, vBite, z);
        const q = [proj3(a3), proj3(b3), proj3(c3), proj3(d3)];
        if (!q.every(v => isFinite(v[0]) && isFinite(v[1]))) continue;
        teeth.push({ q, j, sd, back: DEPTH[j], hh, z, gumY: vGum, biteY: vBite });
        cum += w;
      }
    }
    /* back teeth first: the arch reads as a curve */
    teeth.sort((a, b) => b.back - a.back);
    const vis = clamp01(cfg.vis) * clamp01(openAmt * 2.8);
    const bright = clamp01((0.62 + S.cache.lum * 0.5) * cfg.bright);
    let frontY = null, drawnTeeth = 0, alphaSum = 0;
    for (const t of teeth) {
      const [A0, B0, C0, D0] = t.q;
      const alpha = clamp01(vis * (1 - Math.min(.5, t.back * 0.32)));
      if (alpha <= 0.02) continue;
      alphaSum += alpha;
      const quad = () => {
        ctx.beginPath();
        ctx.moveTo(A0[0], A0[1]); ctx.lineTo(B0[0], B0[1]);
        ctx.lineTo(D0[0], D0[1]); ctx.lineTo(C0[0], C0[1]);
        ctx.closePath();
      };
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      quad();
      ctx.clip();
      ctx.save();
      ctx.transform((B0[0] - A0[0]) / 48, (B0[1] - A0[1]) / 48,
                    (C0[0] - A0[0]) / 120, (C0[1] - A0[1]) / 120, A0[0], A0[1]);
      try { ctx.drawImage(S.cache.enamel, (t.j * 29 + (t.sd ? 13 : 0)) % 80, 0, 48, 120, 0, 0, 48, 120); } catch (e) {}
      ctx.restore();
      ctx.save();
      quad();
      const side = ctx.createLinearGradient(A0[0], A0[1], B0[0], B0[1]);
      side.addColorStop(0, 'rgba(106,82,58,.34)');
      side.addColorStop(.28, 'rgba(106,82,58,0)');
      side.addColorStop(.72, 'rgba(106,82,58,0)');
      side.addColorStop(1, 'rgba(106,82,58,.34)');
      ctx.fillStyle = side;
      ctx.fill();
      if (cfg.warmth > 0.01) {
        ctx.fillStyle = `rgba(196,150,70,${cfg.warmth * 0.22})`;
        ctx.fill();
      }
      const eg = ctx.createLinearGradient(C0[0], C0[1], A0[0], A0[1]);
      eg.addColorStop(0, `rgba(255,255,255,${0.20 * bright})`);
      eg.addColorStop(.42, 'rgba(255,255,255,0)');
      eg.addColorStop(1, `rgba(232,226,210,${0.12 * bright})`);
      ctx.fillStyle = eg;
      ctx.fill();
      /* biting edge reads translucent, like real enamel */
      const tgd = ctx.createLinearGradient(C0[0], C0[1], (C0[0] + A0[0]) / 2, (C0[1] + A0[1]) / 2);
      tgd.addColorStop(0, `rgba(255,252,245,${0.22 * bright})`);
      tgd.addColorStop(1, 'rgba(255,252,245,0)');
      ctx.fillStyle = tgd;
      ctx.fill();
      ctx.restore();
      /* interproximal shade where neighbours meet */
      ctx.strokeStyle = `rgba(52,26,22,${0.35 * clamp01(0.35 + t.back)})`;
      ctx.lineWidth = 1.05;
      ctx.beginPath();
      ctx.moveTo(A0[0], A0[1]);
      ctx.lineTo(C0[0], C0[1]);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(B0[0], B0[1]);
      ctx.lineTo(D0[0], D0[1]);
      ctx.stroke();
      ctx.restore();
      if (frontY === null || (isUp ? C0[1] > frontY : C0[1] < frontY)) frontY = C0[1];
      drawnTeeth++;
    }
    return { frontY, drawnTeeth, alphaSum };
  }

  function drawCavity(S, ctx, D, h) {
    const a = D[61], b = D[291], up = D[13], lo = D[14];
    const mw = Math.hypot(b.x - a.x, b.y - a.y) || S.head.fw * 0.4;
    const gap = Math.hypot(lo.x - up.x, lo.y - up.y);
    if (![a, b, up, lo].every(p => p && Number.isFinite(p.x) && Number.isFinite(p.y)) || !(mw > 4 && gap > 0.5)) return false;
    const ax = (b.x - a.x) / mw, ay = (b.y - a.y) / mw;
    let nx = -ay, ny = ax;
    if ((lo.x - up.x) * nx + (lo.y - up.y) * ny < 0) { nx = -nx; ny = -ny; }
    const ocx = (up.x + lo.x) * 0.5, ocy = (up.y + lo.y) * 0.5;
    /* A rounded four-point mouth opening is much more stable than filling
       the raw eight-point inner-lip polygon after a jaw turn.  The latter
       can self-cross and paint a black triangular wedge. */
    const halfH = clamp(Math.max(gap * 0.5, mw * 0.018), mw * 0.018, mw * 0.18);
    const tx = ocx - nx * halfH, ty = ocy - ny * halfH;
    const bx = ocx + nx * halfH, by = ocy + ny * halfH;
    const poly = [{ x: a.x, y: a.y }, { x: b.x, y: b.y }, { x: bx, y: by }, { x: tx, y: ty }];
    if (!polyOK(poly, 2)) return false;
    const bb = bounds(poly);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.quadraticCurveTo(tx, ty, b.x, b.y);
    ctx.quadraticCurveTo(bx, by, a.x, a.y);
    ctx.closePath();
    const g = ctx.createLinearGradient(tx, ty, bx, by);
    g.addColorStop(0, `rgba(92,28,34,${0.72 * h})`);
    g.addColorStop(.48, `rgba(25,7,10,${0.86 * h})`);
    g.addColorStop(1, `rgba(52,14,20,${0.78 * h})`);
    ctx.fillStyle = g;
    ctx.fill();
    const rg = ctx.createRadialGradient(ocx, ocy, 1, ocx, ocy, Math.max(3, mw * 0.24));
    rg.addColorStop(0, `rgba(4,1,2,${0.42 * h})`);
    rg.addColorStop(1, 'rgba(4,1,2,0)');
    ctx.fillStyle = rg;
    ctx.fill();
    ctx.restore();
    return true;
  }

  function drawTongue(S, ctx, fr, proj3, D, st, h) {
    const tg = clamp01(st.tongue);
    if (tg < 0.02) return false;
    const In = E.LIPS_I.map(i => D[i]);
    if (!polyOK(In, 2.5)) return false;
    const mw = fr.mw, gap = Math.max(0, st.gap || 0);
    const len = mw * (0.20 + 0.62 * tg) + gap * 0.30;
    const half = mw * (0.13 + 0.09 * tg);
    const sec = [];
    const S5 = 5;
    for (let s = 0; s <= S5; s++) {
      const t = s / S5;
      const w = half * (0.5 + 0.6 * Math.sin(Math.PI * Math.min(1, 0.18 + t * 0.82)));
      const v = fr.loY + gap * 0.06 + t * (gap * 0.26 + mw * 0.06) * (0.6 + 0.4 * tg);
      const z = -mw * 0.10 + t * len;
      sec.push({ l: proj3(mfp(fr, -w, v, z)), r: proj3(mfp(fr, w, v, z)), m: proj3(mfp(fr, 0, v, z)), t, w });
    }
    if (!sec.every(s => isFinite(s.l[0]) && isFinite(s.r[0]) && isFinite(s.m[0]))) return false;
    const bb = bounds(sec.flatMap(s => [s.l, s.r]));
    if (!(bb.w > 1 && bb.h > 1)) return false;
    ctx.save();
    ctx.beginPath();
    In.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath();
    ctx.clip();
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(sec[0].l[0], sec[0].l[1]);
      for (let i = 1; i < sec.length; i++) ctx.lineTo(sec[i].l[0], sec[i].l[1]);
      for (let i = sec.length - 1; i >= 0; i--) ctx.lineTo(sec[i].r[0], sec[i].r[1]);
      ctx.closePath();
    };
    const g = ctx.createLinearGradient(sec[0].m[0], sec[0].m[1], sec[sec.length - 1].m[0], sec[sec.length - 1].m[1]);
    g.addColorStop(0, 'rgba(140,54,68,1)');
    g.addColorStop(.5, 'rgba(200,100,112,1)');
    g.addColorStop(1, 'rgba(222,132,140,1)');
    ctx.globalAlpha = clamp01(0.4 + 0.6 * h);
    path();
    ctx.fillStyle = g;
    ctx.fill();
    ctx.save();
    path(); ctx.clip();
    let s2 = 12345;
    const rnd = () => (s2 = (s2 * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const first = sec[0].m, last = sec[sec.length - 1].m;
    for (let i = 0; i < 80; i++) {
      const t = rnd(), u = rnd();
      const x = lerp(first[0], last[0], t) + (u - .5) * half * 1.7;
      const y = lerp(first[1], last[1], t);
      ctx.beginPath();
      ctx.arc(x, y, i % 3 ? 0.9 : 1.4, 0, 7);
      ctx.fillStyle = 'rgba(255,206,212,.15)';
      ctx.fill();
    }
    ctx.strokeStyle = 'rgba(120,26,42,.42)';
    ctx.lineWidth = Math.max(1.2, half * 0.16);
    ctx.beginPath();
    ctx.moveTo(first[0], first[1]);
    for (let i = 1; i < sec.length; i++) ctx.lineTo(sec[i].m[0], sec[i].m[1]);
    ctx.stroke();
    const tip = sec[sec.length - 1].m;
    const sg = ctx.createRadialGradient(tip[0], tip[1], 1, tip[0], tip[1], Math.max(3, half * 1.5));
    sg.addColorStop(0, 'rgba(255,222,226,.32)');
    sg.addColorStop(1, 'rgba(255,222,226,0)');
    ctx.fillStyle = sg;
    ctx.fillRect(bb.x0 - 4, bb.y0 - 4, bb.w + 8, bb.h + 8);
    const sh = ctx.createLinearGradient(0, bb.y0, 0, bb.y0 + bb.h * 0.5);
    sh.addColorStop(0, 'rgba(58,12,20,.5)');
    sh.addColorStop(1, 'rgba(58,12,20,0)');
    ctx.fillStyle = sh;
    ctx.fillRect(bb.x0 - 4, bb.y0 - 4, bb.w + 8, bb.h * 0.55);
    ctx.restore();
    ctx.globalAlpha = 1;
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
    const q = st.q || [0, 0, 0, 1];
    const qAngle = 2 * Math.acos(clamp(Math.abs(q[3]), 0, 1));
    /* Translation/scale from the webcam are not reasons to repaint a
       straight-on photo; they are especially noisy after model inference.
       Hold the original source until there is a real, unmistakable turn. */
    const frontal = qAngle < 0.28;
    /* A frontal still is already a pixel-perfect render.  Keep it as the
       base and repaint only moved triangles for expressions.  The full
       shell remains available once the head actually turns. */
    lc.drawImage(frontal ? S.src : (S.cache.plate || S.src), 0, 0);
    lc.imageSmoothingQuality = st.quality > 0 ? 'high' : 'low';
    const shellOpts = {
      resetOrder: S.frames < 3,
      padScale: st.quality > 1 ? 1.4 : 1,
      /* Never paint triangle patches over a straight-on source portrait. */
      skipPaint: frontal
    };
    drawShell(S, lc, st, S.triA, shellOpts);
    if (st.quality > 0) drawShell(S, lc, st, S.triB, {
      ...shellOpts,
      /* The Delaunay shell is only used on a real turn. The finer pass
         restores facial detail there without affecting frontal sharpness. */
      alpha: st.quality > 1 ? 0.72 : 0.48
    });
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, S.W, S.H);
    let mw = 0, gap = 0, h = S.openEnv;
    const oral = { cavity: false, lowerTeeth: 0, lowerAlpha: 0, tongue: false, upperTeeth: 0, upperAlpha: 0, lips: false };
    if (st.origOnly) {
      ctx.drawImage(S.src, 0, 0);
    } else {
      ctx.drawImage(S.layer, 0, 0);
      /* ---- accessories -------------------------------------------- */
      const fr = mouthFrame(S.head, C);
      mw = Math.hypot(S.DST[61 * 2] - S.DST[291 * 2], S.DST[61 * 2 + 1] - S.DST[291 * 2 + 1]) || S.head.fw * 0.4;
      const rawGap = Math.hypot(S.DST[13 * 2] - S.DST[14 * 2], S.DST[13 * 2 + 1] - S.DST[14 * 2 + 1]);
      gap = Math.max(0, rawGap - mw * 0.035);
      S.openEnv = envelope(S.openEnv, clamp01(gap / (mw * 0.30)), st.dt || 1 / 60, 14, 9);
      h = S.openEnv;
      const zc = st.depthGain || 0;
      const proj3 = (p) => S.proj([p[0], p[1], p[2] * (1 + zc)]);
      if (h > 0.004) oral.cavity = drawCavity(S, ctx, D, h) === true;
      const openAmt = clamp01(h * 1.1);
      const lower = drawArch(S, ctx, proj3, fr, 'lo', openAmt, st);
      oral.lowerTeeth = lower.drawnTeeth; oral.lowerAlpha = lower.alphaSum;
      oral.tongue = drawTongue(S, ctx, fr, proj3, D, { tongue: st.tongue, gap: gap + mw * 0.035 }, h) === true;
      const upper = drawArch(S, ctx, proj3, fr, 'up', openAmt, st);
      oral.upperTeeth = upper.drawnTeeth; oral.upperAlpha = upper.alphaSum;
      oral.lips = drawLips(S, ctx, D, mw, gap) === true;
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
    const dbg = { tris: S.stats.tris, culled: S.stats.culled, mouth: { mw, gap, h, ...oral }, jaw: S.jaw };
    S.lastDebug = dbg;
    return dbg;
  }

  root.FMRender = { createSession, sessionInit, sessionFrame, deform, buildPlate, prepIris, makeCanvas,
                    mouthFrame, mfp, buildArches };
})(typeof globalThis !== 'undefined' ? globalThis : this);
