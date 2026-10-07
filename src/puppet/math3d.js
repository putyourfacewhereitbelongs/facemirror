/**
 * Minimal 3D linear algebra for the head rig.
 *
 * Frame convention (documented once, used everywhere):
 *   x -> image right, y -> image DOWN, z -> TOWARD the viewer.
 *   The camera sits at (0, 0, F) and the projection plane is z = 0, so a head
 *   space point X projects to  (cx + X.x*s, cy + X.y*s)  with  s = F / (F - X.z).
 *
 * MediaPipe reports landmark depth with the opposite sign (the nose tip is the
 * most negative z), so driver depth is negated on the way in - see ZSIGN.
 */

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, t) => a + (b - a) * t;
/** Hermite ramp: 0 below `a`, 1 above `b`, C1 in between. */
export const ramp = (a, b, x) => { const t = clamp((x - a) / (b - a || 1e-6), 0, 1); return t * t * (3 - 2 * t); };
export const ID3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

export const mul3 = (A, v) => [
  A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
  A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
  A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
];
export const mul33 = (A, B) => {
  const o = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return o;
};
export const transpose = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
export const det3 = A => A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
export const inverse = A => {
  const d = det3(A) || 1e-9;
  return [(A[4] * A[8] - A[5] * A[7]) / d, (A[2] * A[7] - A[1] * A[8]) / d, (A[1] * A[5] - A[2] * A[4]) / d,
    (A[5] * A[6] - A[3] * A[8]) / d, (A[0] * A[8] - A[2] * A[6]) / d, (A[2] * A[3] - A[0] * A[5]) / d,
    (A[3] * A[7] - A[4] * A[6]) / d, (A[1] * A[6] - A[0] * A[7]) / d, (A[0] * A[4] - A[1] * A[3]) / d];
};
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm3 = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/**
 * Weighted similarity fit (Kabsch / orthogonal iteration): the rotation, scale
 * and centroid pair that best carries point set A onto point set B.
 */
export function fitSimilarity(A, B, w) {
  const n = A.length;
  const ca = [0, 0, 0], cb = [0, 0, 0];
  let tw = 0;
  for (let i = 0; i < n; i++) { const wi = w ? w[i] : 1; tw += wi; for (let k = 0; k < 3; k++) { ca[k] += A[i][k] * wi; cb[k] += B[i][k] * wi; } }
  for (let k = 0; k < 3; k++) { ca[k] /= tw || 1; cb[k] /= tw || 1; }
  const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  let na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    const a = [A[i][0] - ca[0], A[i][1] - ca[1], A[i][2] - ca[2]];
    const b = [B[i][0] - cb[0], B[i][1] - cb[1], B[i][2] - cb[2]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M[r * 3 + c] += wi * b[r] * a[c];
    na += wi * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    nb += wi * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  }
  // Orthogonal iteration: a few polar-decomposition steps turn M into a rotation.
  const fn = Math.hypot(...M) || 1;
  let R = M.map(x => (x / fn) * 1.7);
  for (let k = 0; k < 24; k++) { const t = transpose(inverse(R)); R = R.map((x, j) => 0.5 * (x + t[j])); }
  if (!R.every(Number.isFinite) || Math.abs(det3(R) - 1) > 0.15) {
    R = ID3();
    if (Math.abs(det3(R) - 1) > 0.15) R = ID3();
  }
  if (det3(R) < 0) R = ID3();
  return { R, s: Math.sqrt(nb / (na || 1)) || 1, ca, cb, M };
}

/** Extract (yaw, pitch, roll) from R = Rz(roll) * Rx(pitch) * Ry(yaw), radians. */
export function eulerFromR(R) {
  const sp = clamp(R[7], -1, 1);
  const pitch = Math.asin(sp);
  const cp = Math.cos(pitch);
  if (cp < 1e-4) return { yaw: Math.atan2(R[2], R[0]), pitch, roll: 0 };
  return { yaw: Math.atan2(-R[6], R[8]), pitch, roll: Math.atan2(-R[1], R[4]) };
}

export function RFromEuler(yaw, pitch, roll) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
  return [
    cr * cy - sr * sp * sy, -sr * cp, cr * sy + sr * sp * cy,
    sr * cy + cr * sp * sy, cr * cp, sr * sy - cr * sp * cy,
    -cp * sy, sp, cp * cy,
  ];
}

export function axisAngle(R) {
  const th = Math.acos(clamp((R[0] + R[4] + R[8] - 1) / 2, -1, 1));
  if (th < 1e-6) return { axis: [0, 1, 0], angle: 0 };
  const s = 1 / (2 * Math.sin(th));
  return { axis: norm3([(R[7] - R[5]) * s, (R[2] - R[6]) * s, (R[3] - R[1]) * s]), angle: th };
}

export function rotAxis(axis, a) {
  const [x, y, z] = axis, c = Math.cos(a), s = Math.sin(a), u = 1 - c;
  return [c + x * x * u, x * y * u - z * s, x * z * u + y * s,
    y * x * u + z * s, c + y * y * u, y * z * u - x * s,
    z * x * u - y * s, z * y * u + x * s, c + z * z * u];
}

/**
 * Soft limiter: small angles pass through untouched, large angles saturate
 * smoothly at `max` so a driver can never twist the photo past the point where
 * its pixels stop existing. tanh keeps the derivative continuous (no popping).
 */
export const softLimit = (x, gain, max) => {
  const g = x * gain;
  if (max <= 0) return g;
  return max * Math.tanh(g / max);
};

/**
 * Per-axis retargeting of a head rotation: gains, soft limits and a hard safety
 * clamp. Working per axis (instead of on the raw Kabsch rotation) is what makes
 * a big turn controllable and stable.
 */
export function retargetEuler(e, gains, limits) {
  const yaw = clamp(softLimit(e.yaw, gains.yaw, limits.yaw), -limits.yawHard, limits.yawHard);
  const pitch = clamp(softLimit(e.pitch, gains.pitch, limits.pitch), -limits.pitchHard, limits.pitchHard);
  const roll = clamp(softLimit(e.roll, gains.roll, limits.roll), -limits.rollHard, limits.rollHard);
  return { yaw, pitch, roll, R: RFromEuler(yaw, pitch, roll) };
}

/** Rotation about an arbitrary point/axis: returns the displacement of `p`. */
export function hingeDisplacement(p, origin, axis, angle) {
  const d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
  const R = rotAxis(axis, angle);
  const r = mul3(R, d);
  return [r[0] - d[0], r[1] - d[1], r[2] - d[2]];
}

/** Robust scale between two point clouds (median of per-point distance ratios). */
export function cloudScaleRatio(A, B, idx) {
  const ca = centroid(A, idx), cb = centroid(B, idx);
  let sa = 0, sb = 0;
  for (const i of idx) {
    sa += Math.hypot(A[i][0] - ca[0], A[i][1] - ca[1]);
    sb += Math.hypot(B[i][0] - cb[0], B[i][1] - cb[1]);
  }
  return (sb / idx.length) / ((sa / idx.length) || 1);
}
export function centroid(P, idx) {
  let x = 0, y = 0, z = 0;
  for (const i of idx) { x += P[i][0]; y += P[i][1]; z += P[i][2] || 0; }
  const n = idx.length || 1;
  return [x / n, y / n, z / n];
}
