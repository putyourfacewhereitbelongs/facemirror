/* Synthetic 478-point face used by the harness and the inspector.
   Face width (234↔454) is 1.0 in these units; the layout follows the
   real MediaPipe index groups closely enough that every region test in
   the engine (lids, lips, brows, irises, contour, jaw) lands where it
   would on a real photo. */
/* =====================================================================
   1. synthetic face model in face units (face width 234↔454 == 1.0)
   ===================================================================== */
const R = { rx: 0.50, ry: 0.64, rz: 0.42 };
const surf = (x, y) => {
  const q = 1 - (x / R.rx) ** 2 - (y / R.ry) ** 2;
  let z = q > 0 ? R.rz * Math.sqrt(q) : 0;
  z += R.rz * 0.30 * Math.exp(-((x / 0.07) ** 2)) * Math.max(0, 1 - Math.abs(y - 0.02) / 0.35); // nose ridge
  return z;
};
export const CONT = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
export const LIPS_O = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
export const LIPS_I = [78, 82, 13, 312, 308, 317, 14, 87];
export const LEYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const REYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const LBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const RBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
const NOSEB = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2];
const NOSEA = [98, 327, 129, 358, 219, 439, 218, 438, 115, 344, 49, 279, 48, 278, 64, 294];
const EXTRA = { 205: [-0.30, 0.07], 425: [0.30, 0.07], 50: [-0.26, 0.20], 280: [0.26, 0.20], 36: [-0.13, -0.30], 266: [0.13, -0.30],
  116: [-0.24, 0.20], 345: [0.24, 0.20], 118: [-0.19, 0.14], 347: [0.19, 0.14], 100: [-0.16, 0.26], 329: [0.16, 0.26],
  18: [0, 0.375], 199: [0, 0.42], 200: [0, 0.45], 175: [0, 0.33], 151: [0, -0.46], 108: [-0.10, -0.46], 337: [0.10, -0.46],
  9: [0, -0.40], 8: [-0.05, -0.38], 299: [-0.09, -0.40], 333: [0.09, -0.40], 298: [-0.13, -0.38], 301: [-0.07, -0.35] };

export function synthFace() {
  const P = new Array(478).fill(null).map(() => [0, 0, 0]);
  const put = (i, x, y, z) => { P[i] = [x, y, z === undefined ? surf(x, y) : z]; };
  /* contour: top -> +x side -> chin -> -x side -> back to top */
  CONT.forEach((idx, k) => {
    const a = -Math.PI / 2 + 2 * Math.PI * k / CONT.length;
    const x = Math.cos(a) * R.rx * 1.02, y = Math.sin(a) * R.ry * 1.02;
    put(idx, x, y, Math.max(0, surf(x, y)) * 0.25);
  });
  /* eyes: index 0 is the outer corner (33 / 263), index 8 the inner one */
  const eye = (list, cx) => {
    const inner = cx < 0 ? 1 : -1;
    list.forEach((idx, k) => {
      const a = k <= 8 ? Math.PI * (1 - k / 8) : -Math.PI * (k - 8) / 8;
      const up = Math.sin(a) < -0.2;
      const rx = 0.088, ry = up ? 0.034 : 0.026;
      const x = cx + Math.cos(a) * rx * (inner > 0 ? 1 : 1);
      const y = -0.145 + Math.sin(a) * ry - (up ? 0.006 : 0);
      put(idx, x, y, surf(x, y) * 0.98);
    });
  };
  eye(LEYE, -0.205);
  eye(REYE, 0.205);
  /* irises: 468 belongs to the image-left eye, 473 to the image-right one */
  [[468, -0.205], [473, 0.205]].forEach(([c, cx], n) => {
    put(c, cx, -0.145, surf(cx, -0.145) * 1.0);
    for (let k = 0; k < 4; k++) {
      const a = k * Math.PI / 2;
      put(469 + n * 5 + k, cx + Math.cos(a) * 0.030, -0.145 + Math.sin(a) * 0.030, surf(cx, -0.145));
    }
  });
  const brow = (list, x0, x1, y0) => list.forEach((idx, k) => {
    const t = k / (list.length - 1);
    const x = x0 + (x1 - x0) * t, y = y0 - Math.sin(Math.PI * t) * 0.02 - t * 0.012;
    put(idx, x, y, surf(x, y) * 0.95);
  });
  brow(LBROW, -0.07, -0.36, -0.245);
  brow(RBROW, 0.07, 0.36, -0.245);
  /* nose */
  NOSEB.forEach((idx, k) => {
    const t = k / (NOSEB.length - 1);
    const y = -0.23 + t * 0.30;
    put(idx, 0, y, surf(0, y));
  });
  NOSEA.forEach((idx, k) => {
    const side = k % 2 ? 1 : -1;
    const t = Math.floor(k / 2) / (NOSEA.length / 2 - 1);
    const x = side * (0.045 + t * 0.03), y = 0.055 + t * 0.055;
    put(idx, x, y, surf(x, y) * 0.98);
  });
  /* lips: outer ring then inner ring (holes in the shell) */
  LIPS_O.forEach((idx, k) => {
    const a = k <= 10 ? Math.PI * (1 - k / 10) : -Math.PI * (k - 10) / 10;
    const x = Math.cos(a) * 0.155, y = 0.285 + Math.sin(a) * 0.058;
    put(idx, x, y, surf(x, y) * 0.86);
  });
  LIPS_I.forEach((idx, k) => {
    const angs = [Math.PI, 1.38 * Math.PI, 1.5 * Math.PI, 1.62 * Math.PI, 0, 0.38 * Math.PI, 0.5 * Math.PI, 0.62 * Math.PI];
    const a = angs[k];
    const x = Math.cos(a) * 0.100, y = 0.285 + Math.sin(a) * 0.004;
    put(idx, x, y, surf(x, y) * 0.80);
  });
  for (const k in EXTRA) put(+k, EXTRA[k][0], EXTRA[k][1]);
  /* everything else: deterministic scatter inside the oval */
  let s = 20261007;
  const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const taken = new Set([...CONT, ...LIPS_O, ...LIPS_I, ...LEYE, ...REYE, ...LBROW, ...RBROW, ...NOSEB, ...NOSEA,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477, ...Object.keys(EXTRA).map(Number)]);
  for (let i = 0; i < 478; i++) {
    if (taken.has(i)) continue;
    const a = rnd() * Math.PI * 2, r = Math.sqrt(rnd()) * 0.94;
    const x = Math.cos(a) * R.rx * r, y = Math.sin(a) * R.ry * r;
    put(i, x, y, surf(x, y) * 0.97);
  }
  return P;
}
/* nearest-neighbour edge list, standing in for MediaPipe's tessellation */
export function synthTessellation(P) {
  const edges = new Set();
  const out = [];
  const add = (a, b) => { const k = a < b ? a + ':' + b : b + ':' + a; if (!edges.has(k)) { edges.add(k); out.push({ start: a, end: b }); } };
  for (let i = 0; i < P.length; i++) {
    const d = [];
    for (let j = 0; j < P.length; j++) {
      if (i === j) continue;
      d.push([(P[j][0] - P[i][0]) ** 2 + (P[j][1] - P[i][1]) ** 2 + (P[j][2] - P[i][2]) ** 2, j]);
    }
    d.sort((a, b) => a[0] - b[0]);
    for (let k = 0; k < 5; k++) add(i, d[k][1]);
  }
  return out;
}
/* rotate about the origin then translate / scale, image convention */
export function poseModel(P, yaw, pitch, roll, tx = 0, ty = 0, tz = 0, sc = 1) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
  const M = [
    cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp,
    cp * sr, cp * cr, -sp,
    -sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp
  ];
  return P.map(([x, y, z]) => {
    const X = x * sc, Y = y * sc, Z = z * sc;
    return [
      M[0] * X + M[1] * Y + M[2] * Z + tx,
      M[3] * X + M[4] * Y + M[5] * Z + ty,
      M[6] * X + M[7] * Y + M[8] * Z + tz
    ];
  });
}
export const toLandmarks = (P, W, H, scalePx) => P.map(([x, y, z]) => ({
  x: 0.5 + (x * scalePx) / W,
  y: 0.48 + (y * scalePx) / H,
  z: -(z * scalePx) / W
}));


export const FACE_PX_DEFAULT = 256;
export const MODEL_W = 640, MODEL_H = 480;
