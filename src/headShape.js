/**
 * headShape.js — shared, DOM-free ARKit displacement fields. The anatomical
 * head masks these to facial tissue, so expressions never pull on the ears.
 */

export const MORPH_CONFIG = {
  jawPivot: [0, -0.18, 0.18],
  jawOpenRad: 0.5,
  morphRadius: 0.34,
};
const CONFIG = MORPH_CONFIG;

/**
 * Gaussian falloff for morph displacement fields.
 * @param {number} dx @param {number} dy @param {number} dz
 * @param {number} radius
 * @returns {number}
 */
function gauss(dx, dy, dz, radius) {
  const d2 = dx * dx + dy * dy + dz * dz;
  const r2 = radius * radius;
  return Math.exp(-d2 / (2 * r2 * 0.55));
}

/**
 * Displacement field for one procedural morph, evaluated per vertex.
 * Returns delta [dx,dy,dz] into `out3`.
 * @param {string} name ARKit channel name
 * @param {number} x @param {number} y @param {number} z vertex (head-local)
 * @param {Float32Array} out3
 */
export function proceduralMorphDelta(name, x, y, z, out3) {
  let dx = 0;
  let dy = 0;
  let dz = 0;
  const R = CONFIG.morphRadius;
  const g = (ax, ay, az, r = R) => gauss(x - ax, y - ay, z - az, r);
  const mouthY = -0.30;
  const mouthZ = 0.72;
  const jawGate = Math.max(0, Math.min(1, (-0.12 - y) / 0.55)) * Math.max(0, Math.min(1, (z + 0.25) / 0.6));

  switch (name) {
    case 'jawOpen': {
      // Jaw rotation about the TMJ-like pivot (§6 jaw rotation via morphs).
      const [px, py, pz] = CONFIG.jawPivot;
      const w = jawGate;
      const ang = CONFIG.jawOpenRad * w;
      const ry = y - py;
      const rz = z - pz;
      dy += (ry * (Math.cos(ang) - 1) - rz * Math.sin(ang)) * w;
      dz += (ry * Math.sin(ang) + rz * (Math.cos(ang) - 1)) * w;
      break;
    }
    case 'jawForward': dz += jawGate * 0.09; break;
    case 'jawLeft': dx -= jawGate * 0.10; break;
    case 'jawRight': dx += jawGate * 0.10; break;
    case 'mouthClose': {
      const w = g(0, mouthY - 0.10, mouthZ, R * 0.8);
      dy += w * 0.12;
      break;
    }
    case 'mouthSmileLeft': {
      const w = g(-0.30, mouthY, mouthZ * 0.92, R * 0.85);
      dy += w * 0.115; dx -= w * 0.055; dz -= w * 0.02;
      break;
    }
    case 'mouthSmileRight': {
      const w = g(0.30, mouthY, mouthZ * 0.92, R * 0.85);
      dy += w * 0.115; dx += w * 0.055; dz -= w * 0.02;
      break;
    }
    case 'mouthDimpleLeft': { const w = g(-0.33, mouthY + 0.03, mouthZ * 0.85); dx += w * 0.035; dz -= w * 0.03; break; }
    case 'mouthDimpleRight': { const w = g(0.33, mouthY + 0.03, mouthZ * 0.85); dx -= w * 0.035; dz -= w * 0.03; break; }
    case 'mouthFrownLeft': { const w = g(-0.28, mouthY, mouthZ * 0.9); dy -= w * 0.10; break; }
    case 'mouthFrownRight': { const w = g(0.28, mouthY, mouthZ * 0.9); dy -= w * 0.10; break; }
    case 'mouthLeft': { const w = g(0, mouthY, mouthZ); dx -= w * 0.12; break; }
    case 'mouthRight': { const w = g(0, mouthY, mouthZ); dx += w * 0.12; break; }
    case 'mouthPucker': { const w = g(0, mouthY, mouthZ, R * 0.75); dz += w * 0.10; dx *= 1; break; }
    case 'mouthFunnel': { const w = g(0, mouthY, mouthZ, R * 0.8); dz += w * 0.13; dy -= w * 0.015; break; }
    case 'mouthUpperUpLeft': { const w = g(-0.12, mouthY + 0.10, mouthZ * 0.95); dy += w * 0.075; break; }
    case 'mouthUpperUpRight': { const w = g(0.12, mouthY + 0.10, mouthZ * 0.95); dy += w * 0.075; break; }
    case 'mouthLowerDownLeft': { const w = g(-0.12, mouthY - 0.11, mouthZ * 0.95); dy -= w * 0.075; break; }
    case 'mouthLowerDownRight': { const w = g(0.12, mouthY - 0.11, mouthZ * 0.95); dy -= w * 0.075; break; }
    case 'mouthPressLeft': { const w = g(-0.16, mouthY, mouthZ * 0.98, R * 0.7); dz -= w * 0.035; dy *= 1; break; }
    case 'mouthPressRight': { const w = g(0.16, mouthY, mouthZ * 0.98, R * 0.7); dz -= w * 0.035; break; }
    case 'mouthRollLower': { const w = g(0, mouthY - 0.10, mouthZ * 0.97, R * 0.7); dy += w * 0.05; dz -= w * 0.02; break; }
    case 'mouthRollUpper': { const w = g(0, mouthY + 0.10, mouthZ * 0.97, R * 0.7); dy -= w * 0.05; dz -= w * 0.02; break; }
    case 'mouthShrugLower': { const w = g(0, mouthY - 0.12, mouthZ * 0.95); dy += w * 0.05; break; }
    case 'mouthShrugUpper': { const w = g(0, mouthY + 0.12, mouthZ * 0.95); dy += w * 0.05; break; }
    case 'mouthStretchLeft': { const w = g(-0.34, mouthY, mouthZ * 0.85); dx -= w * 0.10; break; }
    case 'mouthStretchRight': { const w = g(0.34, mouthY, mouthZ * 0.85); dx += w * 0.10; break; }
    case 'browInnerUp': {
      const wl = g(-0.14, 0.30, mouthZ * 0.78);
      const wr = g(0.14, 0.30, mouthZ * 0.78);
      dy += (wl + wr) * 0.10;
      break;
    }
    case 'browDownLeft': { const w = g(-0.20, 0.27, mouthZ * 0.75); dy -= w * 0.075; break; }
    case 'browDownRight': { const w = g(0.20, 0.27, mouthZ * 0.75); dy -= w * 0.075; break; }
    case 'browOuterUpLeft': { const w = g(-0.42, 0.28, mouthZ * 0.55); dy += w * 0.11; break; }
    case 'browOuterUpRight': { const w = g(0.42, 0.28, mouthZ * 0.55); dy += w * 0.11; break; }
    case 'cheekPuff': {
      const wl = g(-0.52, -0.05, 0.42);
      const wr = g(0.52, -0.05, 0.42);
      dx -= wl * 0.10; dx += wr * 0.10;
      dz += (wl + wr) * 0.04;
      break;
    }
    case 'cheekSquintLeft': { const w = g(-0.34, 0.02, 0.62); dy += w * 0.05; break; }
    case 'cheekSquintRight': { const w = g(0.34, 0.02, 0.62); dy += w * 0.05; break; }
    case 'noseSneerLeft': { const w = g(-0.13, 0.02, 0.82, R * 0.55); dy += w * 0.055; break; }
    case 'noseSneerRight': { const w = g(0.13, 0.02, 0.82, R * 0.55); dy += w * 0.055; break; }
    case 'eyeWideLeft': { const w = g(-0.30, 0.12, 0.68, R * 0.62); dy += w * 0.05; break; }
    case 'eyeWideRight': { const w = g(0.30, 0.12, 0.68, R * 0.62); dy += w * 0.05; break; }
    case 'eyeSquintLeft': { const w = g(-0.30, 0.05, 0.68, R * 0.58); dy += w * 0.035; break; }
    case 'eyeSquintRight': { const w = g(0.30, 0.05, 0.68, R * 0.58); dy += w * 0.035; break; }
    case 'eyeBlinkLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.55); dy -= w * 0.045; break; }
    case 'eyeBlinkRight': { const w = g(0.30, 0.10, 0.70, R * 0.55); dy -= w * 0.045; break; }
    case 'eyeLookUpLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dy += w * 0.025; break; }
    case 'eyeLookUpRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dy += w * 0.025; break; }
    case 'eyeLookDownLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dy -= w * 0.025; break; }
    case 'eyeLookDownRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dy -= w * 0.025; break; }
    case 'eyeLookInLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dx += w * 0.02; break; }
    case 'eyeLookInRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dx -= w * 0.02; break; }
    case 'eyeLookOutLeft': { const w = g(-0.30, 0.10, 0.70, R * 0.5); dx -= w * 0.02; break; }
    case 'eyeLookOutRight': { const w = g(0.30, 0.10, 0.70, R * 0.5); dx += w * 0.02; break; }
    case 'tongueOut': {
      const w = g(0, mouthY - 0.05, mouthZ * 0.95, R * 0.6);
      dz += w * 0.16;
      break;
    }
    default: break; // jawForward etc. handled above; unknown → zero
  }
  out3[0] = dx;
  out3[1] = dy;
  out3[2] = dz;
}
