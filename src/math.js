/* =====================================================================
   FaceMirror · pure math core  (no DOM, no canvas)
   Loaded as a classic script in the browser and inlined into puppet.html.
   Exposed on globalThis.FM so both the app module and the Node test
   harness (tools/verify.mjs) can exercise exactly the same code.

   Contents
     · scalars      clamp / lerp / smoothstep / softLimit / slew / schmitt
     · vec3 + mat3  small row-major linear algebra
     · eigen3/svd3  cyclic Jacobi eigen decomposition of a symmetric 3x3
     · kabsch       weighted orthogonal Procrustes (rigid pose) w/ scale
     · quaternion   mat<->quat, slerp, slew-limited rotation
     · filters      one-euro filter, running-max decay, envelope follower
     · geometry     catmull-rom closed spline, polygon area/validity
   ===================================================================== */
(function (root) {
  'use strict';

  /* ------------------------------ scalars ----------------------------- */
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (e0, e1, x) => {
    const t = clamp01((x - e0) / ((e1 - e0) || 1e-9));
    return t * t * (3 - 2 * t);
  };
  /* Soft saturation.  Exactly linear out to `kneeFrac` of the bound so that
     ordinary head turns are NOT attenuated (a plain tanh squashes even a
     30° turn), then a C1-continuous exponential knee that approaches the
     bound asymptotically and can never cross it.                        */
  const softLimit = (x, m, kneeFrac) => {
    if (m <= 1e-9) return 0;
    const a = m * (kneeFrac === undefined ? 0.70 : clamp01(kneeFrac));
    const s = x < 0 ? -1 : 1, ax = Math.abs(x);
    if (ax <= a) return x;
    const r = m - a;
    return s * (a + r * (1 - Math.exp(-(ax - a) / r)));
  };
  const sign = v => (v < 0 ? -1 : 1);
  const fin = v => Number.isFinite(v);
  /* Geometry consumers use both tuple points [x,y] and renderer points
     {x,y}; normalize the latter at the boundary instead of silently
     rejecting every overlay polygon. Typed arrays stay tuple-compatible. */
  const xy2 = p => p && p.x !== undefined ? [p.x, p.y] : p;
  const fin2 = p => { const q = xy2(p); return !!q && fin(q[0]) && fin(q[1]); };
  const fin3 = p => !!p && fin(p[0]) && fin(p[1]) && fin(p[2]);

  /* slew limiter: approach a target at a bounded rate (units / second).
     This is the anti-pop primitive: nothing in this app jumps, every
     discrete state change is turned into a rate-limited envelope.     */
  function slew(cur, target, dt, upRate, downRate) {
    if (!fin(cur) || !fin(target)) return fin(target) ? target : 0;
    const step = (target > cur ? upRate : downRate) * dt;
    const d = target - cur;
    return Math.abs(d) <= step ? target : cur + (d < 0 ? -step : step);
  }

  /* exponential-ish follower with a dead-band: ignores sub-pixel noise,
     used to keep a resting face perfectly still (no shimmer).         */
  function follow(cur, target, dt, rate, deadband) {
    if (!fin(cur)) cur = 0;
    const d = target - cur;
    if (Math.abs(d) < (deadband || 0)) return cur;
    const a = 1 - Math.exp(-(rate || 8) * (dt || 1 / 60));
    return cur + d * a;
  }

  /* Schmitt trigger — hysteresis for boolean-ish states so a threshold
     sitting exactly on a signal's average cannot strobe.             */
  function schmitt(state, x, onAt, offAt) {
    if (state) return x > offAt;
    return x > onAt;
  }

  /* envelope follower with attack/release (audio RMS, jaw, lids, ...)  */
  function envelope(cur, x, dt, attack, release) {
    /* 4-arg form means "same speed both ways" — an undefined release used to
       leak NaN into every signal that decayed (tracker trust did exactly
       that on the first dropped frame).                              */
    const rate = x > cur ? attack : (release === undefined ? attack : release);
    return cur + (x - cur) * (1 - Math.exp(-rate * (dt || 1 / 60)));
  }

  /* ------------------------------- vec3 ------------------------------- */
  const v3 = {
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ],
    len: a => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]),
    norm: a => {
      const l = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) || 1;
      return [a[0] / l, a[1] / l, a[2] / l];
    },
    dist: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
    mix: (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
  };

  /* ---------------------------- mat3 (row major) ---------------------- */
  const m3 = {
    id: () => [1, 0, 0, 0, 1, 0, 0, 0, 1],
    t: A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]],
    mul(A, B) {
      const C = new Array(9);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++)
          C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
      return C;
    },
    mv(A, v) {
      return [
        A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
        A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
        A[6] * v[0] + A[7] * v[1] + A[8] * v[2]
      ];
    },
    mtv(A, v) {
      return [
        A[0] * v[0] + A[3] * v[1] + A[6] * v[2],
        A[1] * v[0] + A[4] * v[1] + A[7] * v[2],
        A[2] * v[0] + A[5] * v[1] + A[8] * v[2]
      ];
    },
    det: A =>
      A[0] * (A[4] * A[8] - A[5] * A[7]) -
      A[1] * (A[3] * A[8] - A[5] * A[6]) +
      A[2] * (A[3] * A[7] - A[4] * A[6]),
    inv(A) {
      const d = m3.det(A);
      if (!fin(d) || Math.abs(d) < 1e-12) return m3.id();
      const i = 1 / d;
      return [
        (A[4] * A[8] - A[5] * A[7]) * i, (A[2] * A[7] - A[1] * A[8]) * i, (A[1] * A[5] - A[2] * A[4]) * i,
        (A[5] * A[6] - A[3] * A[8]) * i, (A[0] * A[8] - A[2] * A[6]) * i, (A[2] * A[3] - A[0] * A[5]) * i,
        (A[3] * A[7] - A[4] * A[6]) * i, (A[1] * A[6] - A[0] * A[7]) * i, (A[0] * A[4] - A[1] * A[3]) * i
      ];
    },
    trace: A => A[0] + A[4] + A[8],
    /* Higham's polar iteration: matrix -> nearest rotation (fallback path) */
    ortho(A, iters) {
      let R = A.slice();
      for (let k = 0; k < (iters || 24); k++) {
        const Ri = m3.inv(R);
        if (!Ri.every(fin)) break;
        const Rn = new Array(9);
        for (let j = 0; j < 9; j++) Rn[j] = 0.5 * (R[j] + Ri[[0, 3, 6, 1, 4, 7, 2, 5, 8][j]]);
        R = Rn;
        // renormalise so the iteration cannot drift
        const sc = Math.cbrt(Math.abs(m3.det(R))) || 1;
        for (let j = 0; j < 9; j++) R[j] /= sc;
      }
      return R.every(fin) ? R : m3.id();
    },
    fromAxisAngle(ax, ang) {
      const [x, y, z] = v3.norm(ax), c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
      return [
        c + x * x * t, x * y * t - z * s, x * z * t + y * s,
        y * x * t + z * s, c + y * y * t, y * z * t - x * s,
        z * x * t - y * s, z * y * t + x * s, c + z * z * t
      ];
    },
    axisAngle(R) {
      const th = Math.acos(clamp((m3.trace(R) - 1) / 2, -1, 1));
      const s = Math.sin(th);
      if (Math.abs(s) < 1e-6) return { axis: [0, 0, 1], angle: 0 };
      return {
        axis: [(R[7] - R[5]) / (2 * s), (R[2] - R[6]) / (2 * s), (R[3] - R[1]) / (2 * s)],
        angle: th
      };
    }
  };

  /* ------------- symmetric 3x3 eigen (cyclic Jacobi, ~8 sweeps) ------- */
  function eigenSym3(Ain, sweeps) {
    const A = [Ain[0], Ain[1], Ain[2], Ain[3], Ain[4], Ain[5], Ain[6], Ain[7], Ain[8]];
    let V = m3.id();
    const pairs = [[0, 1], [0, 2], [1, 2]];
    for (let sweep = 0; sweep < (sweeps || 12); sweep++) {
      let off = 0;
      for (const [p, q] of pairs) {
        const apq = A[p * 3 + q];
        off += Math.abs(apq);
        if (Math.abs(apq) < 1e-15) continue;
        const app = A[p * 3 + p], aqq = A[q * 3 + q];
        const tau = (aqq - app) / (2 * apq);
        const t = sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t), s = t * c;
        for (let k = 0; k < 3; k++) {          // A <- Jᵀ A J
          const akp = A[k * 3 + p], akq = A[k * 3 + q];
          A[k * 3 + p] = c * akp - s * akq;
          A[k * 3 + q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = A[p * 3 + k], aqk = A[q * 3 + k];
          A[p * 3 + k] = c * apk - s * aqk;
          A[q * 3 + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {          // V <- V J
          const vkp = V[k * 3 + p], vkq = V[k * 3 + q];
          V[k * 3 + p] = c * vkp - s * vkq;
          V[k * 3 + q] = s * vkp + c * vkq;
        }
      }
      if (off < 1e-13) break;
    }
    return {
      vals: [A[0], A[4], A[8]],
      vecs: V,                                  // column k = eigenvector k
      col: k => [V[k], V[3 + k], V[6 + k]]
    };
  }

  /* ------------------------------- SVD3 ------------------------------- */
  /* A = U·diag(s)·Vᵀ with U and V *proper* rotations (det = +1).  When A
     contains a reflection (det A < 0) that is expressed as a negative
     smallest singular value, so the identity still holds exactly and no
     caller can ever pick up a mirrored "rotation".                     */
  function svd3(A) {
    const At = m3.t(A);
    const e = eigenSym3(m3.mul(At, A));
    const order = [0, 1, 2].sort((i, j) => e.vals[j] - e.vals[i]);
    const s = order.map(i => Math.sqrt(Math.max(e.vals[i], 0)));
    if (!s.every(fin) || !(s[0] > 0)) return { U: m3.id(), V: m3.id(), s: [1, 1, 1] };
    const v = order.map(i => e.col(i));
    /* make the right-hand factor a rotation first */
    let V = toMat(v);
    if (m3.det(V) < 0) { v[2] = [-v[2][0], -v[2][1], -v[2][2]]; V = toMat(v); }
    /* left-hand factor from the columns we can trust */
    const u = [[], [], []];
    const tol = s[0] * 1e-6;
    let rank = 0;
    for (let k = 0; k < 3; k++) {
      if (s[k] > tol) { u[k] = v3.scale(m3.mv(A, v[k]), 1 / s[k]); rank++; }
      else u[k] = null;
    }
    if (rank === 0) return { U: m3.id(), V, s: [0, 0, 0] };
    /* complete an orthonormal basis for the null directions */
    if (!u[0]) u[0] = [1, 0, 0];
    u[0] = v3.norm(u[0]);
    if (!u[1] || rank < 2) {
      const helper = Math.abs(u[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      u[1] = v3.norm(v3.cross(u[0], helper));
    } else {
      u[1] = v3.norm(v3.sub(u[1], v3.scale(u[0], v3.dot(u[0], u[1]))));
    }
    u[2] = v3.cross(u[0], u[1]);
    let U = toMat(u);
    /* if A is a reflection, push the sign into the smallest singular value */
    if (m3.det(U) < 0) {
      u[2] = [-u[2][0], -u[2][1], -u[2][2]];
      U = toMat(u);
      s[2] = -s[2];
    }
    return { U, V, s, rank };
  }
  function toMat(cols) {                       // [c0, c1, c2] -> row-major
    return [
      cols[0][0], cols[1][0], cols[2][0],
      cols[0][1], cols[1][1], cols[2][1],
      cols[0][2], cols[1][2], cols[2][2]
    ];
  }

  /* ------------------------------ kabsch ------------------------------ */
  /* Weighted similarity fit  dst ≈ s·R·(src − ca) + cb,  R a proper
     rotation (no reflections). This is the rigid head-pose solver and
     also the identity bridge between the webcam canonical space and the
     photo canonical space.                                            */
  function kabsch(src, dst, weights) {
    const n = Math.min(src.length, dst.length);
    let sw = 0;
    const ca = [0, 0, 0], cb = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const w = weights ? weights[i] : 1;
      if (!(w > 0) || !fin3(src[i]) || !fin3(dst[i])) continue;
      sw += w;
      for (let k = 0; k < 3; k++) { ca[k] += w * src[i][k]; cb[k] += w * dst[i][k]; }
    }
    if (!(sw > 0)) return null;
    for (let k = 0; k < 3; k++) { ca[k] /= sw; cb[k] /= sw; }
    const H = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let na = 0, nb = 0, used = 0;
    for (let i = 0; i < n; i++) {
      const w = weights ? weights[i] : 1;
      if (!(w > 0) || !fin3(src[i]) || !fin3(dst[i])) continue;
      const a = v3.sub(src[i], ca), b = v3.sub(dst[i], cb);
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) H[r * 3 + c] += w * b[r] * a[c];
      na += w * v3.dot(a, a);
      nb += w * v3.dot(b, b);
      used++;
    }
    if (used < 3 || na <= 1e-9) return null;
    let { U, V } = svd3(H);
    let R = m3.mul(U, m3.t(V));
    if (!R.every(fin) || m3.det(R) < 0) {
      U = [U[0], U[1], U[2], U[3], U[4], U[5], -U[6], -U[7], -U[8]];
      R = m3.mul(U, m3.t(V));
    }
    if (!R.every(fin)) R = m3.id();
    // optimal isotropic scale for the chosen rotation
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const w = weights ? weights[i] : 1;
      if (!(w > 0) || !fin3(src[i]) || !fin3(dst[i])) continue;
      const a = v3.sub(src[i], ca), b = v3.sub(dst[i], cb);
      num += w * v3.dot(b, m3.mv(R, a));
      den += w * v3.dot(a, a);
    }
    let s = den > 1e-9 ? num / den : 1;
    if (!fin(s) || s <= 1e-4) s = Math.sqrt(nb / na) || 1;
    const t = [cb[0] - s * (R[0] * ca[0] + R[1] * ca[1] + R[2] * ca[2]),
               cb[1] - s * (R[3] * ca[0] + R[4] * ca[1] + R[5] * ca[2]),
               cb[2] - s * (R[6] * ca[0] + R[7] * ca[1] + R[8] * ca[2])];
    let err = 0;
    for (let i = 0; i < n; i++) {
      const w = weights ? weights[i] : 1;
      if (!(w > 0) || !fin3(src[i]) || !fin3(dst[i])) continue;
      const p = v3.add(v3.scale(m3.mv(R, v3.sub(src[i], ca)), s), cb);
      err += w * v3.dist(p, dst[i]) ** 2;
    }
    return { R, t, s, ca, cb, rms: Math.sqrt(err / sw), n: used };
  }

  /* ---------------------------- quaternion ---------------------------- */
  function quatFromMat(R) {
    const tr = m3.trace(R);
    let q;
    if (tr > 0) {
      const s = Math.sqrt(tr + 1) * 2;
      q = [(R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s, 0.25 * s];
    } else if (R[0] > R[4] && R[0] > R[8]) {
      const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
      q = [0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s, (R[7] - R[5]) / s];
    } else if (R[4] > R[8]) {
      const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
      q = [(R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s, (R[2] - R[6]) / s];
    } else {
      const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
      q = [(R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s, (R[3] - R[1]) / s];
    }
    return quatNorm(q);
  }
  function quatNorm(q) {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
  }
  function matFromQuat(q) {
    const [x, y, z, w] = quatNorm(q);
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    ];
  }
  function quatMul(a, b) {
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    ];
  }
  const quatConj = q => [-q[0], -q[1], -q[2], q[3]];
  function quatSlerp(a, b, t) {
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let bb = b;
    if (d < 0) { bb = [-b[0], -b[1], -b[2], -b[3]]; d = -d; }
    if (d > 0.9995) return quatNorm([lerp(a[0], bb[0], t), lerp(a[1], bb[1], t), lerp(a[2], bb[2], t), lerp(a[3], bb[3], t)]);
    const th = Math.acos(clamp(d, -1, 1)), s = Math.sin(th);
    const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
    return quatNorm([a[0] * wa + bb[0] * wb, a[1] * wa + bb[1] * wb, a[2] * wa + bb[2] * wb, a[3] * wa + bb[3] * wb]);
  }
  const quatAngle = q => 2 * Math.acos(clamp(Math.abs(q[3]), -1, 1));
  /* rate-of-change limiter for rotations: never turn faster than maxRad/s */
  function quatSlew(cur, target, dt, maxRadPerSec) {
    const maxStep = maxRadPerSec * dt;
    const q = quatNorm(cur || [0, 0, 0, 1]), t = quatNorm(target);
    const d = q[0] * t[0] + q[1] * t[1] + q[2] * t[2] + q[3] * t[3];
    const tt = d < 0 ? [-t[0], -t[1], -t[2], -t[3]] : t;
    const ang = 2 * Math.acos(clamp(Math.abs(d), -1, 1));
    if (!fin(ang) || ang <= maxStep) return tt;
    return quatSlerp(q, tt, maxStep / ang);
  }
  /* small-angle smoothing in quaternion space (dt aware) */
  function quatFollow(cur, target, dt, rate) {
    const a = 1 - Math.exp(-(rate || 12) * (dt || 1 / 60));
    return quatSlerp(quatNorm(cur || [0, 0, 0, 1]), quatNorm(target), clamp01(a));
  }
  /* yaw / pitch / roll with R = Ry(yaw)·Rx(pitch)·Rz(roll), exact inverse
     of eulerFromQuat below. Angles are radians, image convention
     (x right, y down, z toward the viewer).                          */
  function quatFromYPR(yaw, pitch, roll) {
    const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
    const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
    const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
    return quatNorm([
      cy * sp * cr + sy * cp * sr,
      sy * cp * cr - cy * sp * sr,
      cy * cp * sr - sy * sp * cr,
      cy * cp * cr + sy * sp * sr
    ]);
  }
  /* Inverse of quatFromYPR. Conventions, in image space
     (x right, y down, z toward the viewer):
       yaw   > 0  nose swings toward +x   (head turns to image-right)
       pitch > 0  nose swings toward -y   (head looks up)
       roll  > 0  +x rotates toward +y, i.e. the image-right side dips */
  function eulerFromQuat(q) {
    const [x, y, z, w] = quatNorm(q);
    const R = matFromQuat([x, y, z, w]);
    // R = Ry(yaw)·Rx(pitch)·Rz(roll)  →
    //   R[2]=sin(yaw)cos(pitch)  R[8]=cos(yaw)cos(pitch)
    //   R[5]=-sin(pitch)         R[3]=cos(pitch)sin(roll)  R[4]=cos(pitch)cos(roll)
    const pitch = Math.asin(clamp(-R[5], -1, 1));
    const yaw = Math.atan2(R[2], R[8]);
    const roll = Math.atan2(R[3], R[4]);
    return { yaw, pitch, roll };
  }

  /* ------------------------------ filters ----------------------------- */
  const alphaFor = (cut, dt) => {
    const r = 2 * Math.PI * Math.max(cut, 0.01) * Math.max(dt, 1e-4);
    return r / (r + 1);
  };
  /* One-euro filter (Casiez et al. 2012): jitter-free when still, low
     latency when moving. fc = base cutoff Hz, beta = speed coupling.   */
  function oneEuro(fc, beta, dcut) {
    return { x: null, dx: 0, t: -1, fc: fc === undefined ? 1.6 : fc, beta: beta === undefined ? 0.02 : beta, dcut: dcut === undefined ? 1 : dcut };
  }
  function oneEuroRun(f, x, t) {
    if (!fin(x)) return f.x === null ? 0 : f.x;
    if (f.x === null || f.t < 0) { f.x = x; f.t = t; return x; }
    let dt = (t - f.t) / 1000;
    if (!(dt > 0)) dt = 1 / 60;
    if (dt > 0.25) dt = 0.25;
    f.t = t;
    const dx = (x - f.x) / dt;
    f.dx = f.dx + alphaFor(f.dcut, dt) * (dx - f.dx);
    const cut = f.fc + f.beta * Math.abs(f.dx);
    const a = alphaFor(cut, dt);
    f.x = f.x + a * (x - f.x);
    if (!fin(f.x)) f.x = x;
    return f.x;
  }
  const oneEuroReset = f => { f.x = null; f.dx = 0; f.t = -1; };
  /* running extreme with slow decay: normalises "eyes wide open" without
     the unbounded drift of a plain running max.                        */
  function ratioTracker(init, decay) {
    return { v: init, decay: decay === undefined ? 0.9985 : decay };
  }
  function ratioTrack(rt, x) {
    if (!fin(x)) return rt.v;
    rt.v = Math.max(rt.v * rt.decay, x);
    rt.v = Math.max(rt.v, 1e-3);
    return rt.v;
  }

  /* ----------------------------- geometry ----------------------------- */
  function polyArea(pts) {
    let a = 0;
    for (let i = 0, n = pts.length; i < n; i++) {
      const p = xy2(pts[i]), q = xy2(pts[(i + 1) % n]);
      a += p[0] * q[1] - q[0] * p[1];
    }
    return a / 2;
  }
  /* A polygon is safe to use as a canvas clip only if every point is
     finite and it encloses real area. Degenerate clips are the classic
     source of full-canvas flashes, so every clip goes through this.   */
  function polyOK(pts, minArea) {
    if (!pts || pts.length < 3) return false;
    let area = 0, ok = true;
    for (let i = 0, n = pts.length; i < n; i++) {
      const p = xy2(pts[i]), q = xy2(pts[(i + 1) % n]);
      if (!fin2(p) || !fin2(q)) { ok = false; break; }
      area += p[0] * q[1] - q[0] * p[1];
    }
    return ok && Math.abs(area / 2) >= (minArea === undefined ? 1.5 : minArea);
  }
  function bounds(pts) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const raw of pts) {
      const p = xy2(raw);
      if (!fin2(p)) continue;
      if (p[0] < x0) x0 = p[0];
      if (p[1] < y0) y0 = p[1];
      if (p[0] > x1) x1 = p[0];
      if (p[1] > y1) y1 = p[1];
    }
    return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
  }
  /* closed Catmull-Rom through pts, resampled to `samples` points */
  function splineClosed(pts, samples) {
    const n = pts.length, out = [];
    const at = i => pts[((i % n) + n) % n];
    const segs = n, per = Math.max(2, Math.round((samples || n * 4) / segs));
    for (let i = 0; i < segs; i++) {
      const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
      for (let k = 0; k < per; k++) {
        const t = k / per, t2 = t * t, t3 = t2 * t;
        out.push([
          0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
          0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        ]);
      }
    }
    return out;
  }
  /* Catmull-Rom arc between three points, used by the tongue/teeth paths */
  function quadPoint(a, b, c, t) {
    const u = 1 - t;
    return [u * u * a[0] + 2 * u * t * b[0] + t * t * c[0], u * u * a[1] + 2 * u * t * b[1] + t * t * c[1]];
  }
  function triArea2(a, b, c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  }
  function pointInPoly(p, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
      if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / ((yj - yi) || 1e-9) + xi) inside = !inside;
    }
    return inside;
  }
  /* blend weight along a polyline: 1 near `pts`, falling off with radius */
  function polylineWeight(p, poly, radius) {
    let best = Infinity;
    for (let i = 0; i + 1 < poly.length; i++) {
      const a = poly[i], b = poly[i + 1];
      const vx = b[0] - a[0], vy = b[1] - a[1];
      const wx = p[0] - a[0], wy = p[1] - a[1];
      const L2 = vx * vx + vy * vy || 1e-9;
      const t = clamp((wx * vx + wy * vy) / L2, 0, 1);
      best = Math.min(best, Math.hypot(wx - vx * t, wy - vy * t));
    }
    return smoothstep(radius, radius * 0.35, best);
  }

  root.FM = {
    clamp, clamp01, lerp, smoothstep, softLimit, sign, fin, fin2, fin3,
    slew, follow, schmitt, envelope,
    v3, m3, eigenSym3, svd3, kabsch,
    quatFromMat, matFromQuat, quatMul, quatConj, quatSlerp, quatAngle, quatSlew, quatFollow,
    quatFromYPR, eulerFromQuat, quatNorm,
    alphaFor, oneEuro, oneEuroRun, oneEuroReset, ratioTracker, ratioTrack,
    polyArea, polyOK, bounds, splineClosed, quadPoint, triArea2, pointInPoly, polylineWeight
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
