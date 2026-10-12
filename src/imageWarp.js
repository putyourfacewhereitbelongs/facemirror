/**
 * imageWarp.js — landmark-anchored 2D/2.5D image mesh warping math.
 * Pure math over Float32Arrays and plain objects: no DOM, no THREE (§9.1).
 *
 * Maps 478 MediaPipe normalized image landmarks (or a centered canonical
 * fallback when a photo has no detectable face) into plane coordinates,
 * extracts the 16 interactive drag-control anchors, and evaluates all 52
 * ARKit blendshape displacement fields directly on the 2D image mesh.
 */

import { LM } from './channels.js';

/** @type {Record<string, number>} */
export const CONFIG = {
  planeMaxW: 1.42,
  planeMaxH: 1.06,
  planeCenterY: 0.06,
  planeBaseZ: 0.35,
  gridSegmentsW: 72,
  gridSegmentsH: 72,
  edgeMargin: 0.065,
  reliefScale: 0.22,
  defaultFaceScaleFrac: 0.38,
};

/**
 * Compute world-space plane dimensions that preserve the image aspect ratio
 * and fit comfortably inside the studio camera frustum.
 *
 * @param {number} imgW pixel width
 * @param {number} imgH pixel height
 * @param {number} [maxW]
 * @param {number} [maxH]
 * @returns {{planeW: number, planeH: number, aspect: number, centerY: number, baseZ: number}}
 */
export function computeImagePlaneSize(
  imgW,
  imgH,
  maxW = CONFIG.planeMaxW,
  maxH = CONFIG.planeMaxH
) {
  const w = imgW > 0 ? imgW : 1;
  const h = imgH > 0 ? imgH : 1;
  const aspect = w / h;
  let planeW;
  let planeH;
  if (aspect >= maxW / maxH) {
    planeW = maxW;
    planeH = maxW / aspect;
  } else {
    planeH = maxH;
    planeW = maxH * aspect;
  }
  return {
    planeW,
    planeH,
    aspect,
    centerY: CONFIG.planeCenterY,
    baseZ: CONFIG.planeBaseZ,
  };
}

/**
 * Smooth boundary weight that vanishes at the outer edges of the image
 * (u ∈ {0,1} or v ∈ {0,1}) so facial warps never tear or pull the frame border.
 *
 * @param {number} u normalized horizontal coord 0..1
 * @param {number} v normalized vertical coord 0..1
 * @param {number} [margin] border fade width
 * @returns {number} 0..1
 */
export function edgePinWeight(u, v, margin = CONFIG.edgeMargin) {
  const m = Math.max(margin, 1e-4);
  const su = Math.min(u / m, (1 - u) / m, 1);
  const sv = Math.min(v / m, (1 - v) / m, 1);
  const cu = su < 0 ? 0 : su * su * (3 - 2 * su);
  const cv = sv < 0 ? 0 : sv * sv * (3 - 2 * sv);
  return cu * cv;
}

/**
 * Build a centered, well-proportioned 478-landmark array in normalized
 * [0..1] image coordinates for images where no human face is detected.
 *
 * @returns {Array<{x: number, y: number, z: number}>}
 */
export function buildCanonicalImageLandmarks() {
  const L = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const set = (i, x, y, z = 0) => {
    if (i >= 0 && i < L.length) {
      L[i].x = x;
      L[i].y = y;
      L[i].z = z;
    }
  };
  // Left eye (image left, set A)
  set(LM.eyeAOuter, 0.345, 0.415, -0.01);
  set(LM.eyeAInner, 0.435, 0.420, -0.01);
  set(LM.eyeATop, 0.390, 0.394, -0.01);
  set(LM.eyeABot, 0.390, 0.436, -0.01);
  set(LM.irisA, 0.390, 0.415, -0.01);
  // Right eye (image right, set B)
  set(LM.eyeBInner, 0.565, 0.420, -0.01);
  set(LM.eyeBOuter, 0.655, 0.415, -0.01);
  set(LM.eyeBTop, 0.610, 0.394, -0.01);
  set(LM.eyeBBot, 0.610, 0.436, -0.01);
  set(LM.irisB, 0.610, 0.415, -0.01);
  // Eyebrows
  set(LM.browAInner, 0.435, 0.348, -0.01);
  set(105, 0.382, 0.332, -0.01);
  set(LM.browAOuter, 0.320, 0.355, -0.01);
  set(LM.browBInner, 0.565, 0.348, -0.01);
  set(334, 0.618, 0.332, -0.01);
  set(LM.browBOuter, 0.680, 0.355, -0.01);
  // Nose
  set(LM.noseBridgeHigh, 0.500, 0.385, -0.02);
  set(LM.noseBridgeLow, 0.500, 0.445, -0.03);
  set(LM.noseTip, 0.500, 0.510, -0.045);
  set(LM.noseWingA, 0.452, 0.525, -0.02);
  set(LM.noseWingB, 0.548, 0.525, -0.02);
  set(LM.noseCreaseA, 0.440, 0.520, -0.015);
  set(LM.noseCreaseB, 0.560, 0.520, -0.015);
  // Mouth & lips
  set(LM.mouthCornerA, 0.405, 0.625, -0.015);
  set(LM.mouthCornerB, 0.595, 0.625, -0.015);
  set(LM.upperLip, 0.500, 0.598, -0.025);
  set(LM.innerUpperLip, 0.500, 0.618, -0.02);
  set(LM.innerLowerLip, 0.500, 0.630, -0.02);
  set(LM.lowerLip, 0.500, 0.654, -0.022);
  set(LM.lowerLipArcA, 0.448, 0.646, -0.018);
  set(LM.lowerLipArcB, 0.552, 0.646, -0.018);
  // Cheeks, jaw, chin
  set(LM.cheekA, 0.320, 0.535, -0.01);
  set(LM.cheekB, 0.680, 0.535, -0.01);
  set(172, 0.335, 0.635, 0.0);
  set(397, 0.665, 0.635, 0.0);
  set(LM.chinMid, 0.500, 0.705, -0.015);
  set(LM.chinBottom, 0.500, 0.748, -0.01);
  return L;
}

/**
 * Anisotropic 2D Gaussian weight around (ax, ay).
 * @param {number} x @param {number} y
 * @param {number} ax @param {number} ay
 * @param {number} rx @param {number} [ry]
 * @returns {number}
 */
function gauss2D(x, y, ax, ay, rx, ry = rx) {
  const dx = (x - ax) / Math.max(rx, 1e-4);
  const dy = (y - ay) / Math.max(ry, 1e-4);
  return Math.exp(-0.5 * (dx * dx + dy * dy));
}

/**
 * Subtle 2.5D relief depth z(x, y) over the face region so head rotations
 * produce natural parallax while keeping image borders flat at baseZ.
 *
 * @param {number} x @param {number} y
 * @param {number} u @param {number} v
 * @param {ReturnType<typeof extractImageFeatures>} features
 * @param {number} [reliefScale]
 * @returns {number}
 */
export function faceReliefZ(x, y, u, v, features, reliefScale = CONFIG.reliefScale) {
  const S = features.faceScale;
  const edge = edgePinWeight(u, v);
  const [fcx, fcy] = features.faceCenter;
  const [ntx, nty] = features.nose.tip;
  const dome = gauss2D(x, y, fcx, fcy, S * 0.68, S * 0.82);
  const noseBump = gauss2D(x, y, ntx, nty, S * 0.16, S * 0.20);
  const chinBump = gauss2D(x, y, features.chinMid[0], features.chinMid[1], S * 0.24, S * 0.18);
  return features.baseZ + edge * reliefScale * S * (dome * 0.65 + noseBump * 0.30 + chinBump * 0.12);
}

/**
 * Extract plane-space facial feature positions, scale, and the 16 drag-control
 * anchors (matching CONTROLS in drag.js) from 478 normalized landmarks.
 *
 * @param {Array<{x: number, y: number, z?: number}>|null|undefined} landmarks478
 * @param {{planeW: number, planeH: number, centerY: number, baseZ: number}} planeSpec
 */
export function extractImageFeatures(landmarks478, planeSpec) {
  const fallback = buildCanonicalImageLandmarks();
  const src = landmarks478 && landmarks478.length >= 468 ? landmarks478 : fallback;
  const { planeW, planeH, centerY, baseZ } = planeSpec;

  const pt = (idx) => {
    const L = src[idx] ?? fallback[idx] ?? { x: 0.5, y: 0.5 };
    const u = Number.isFinite(L.x) ? L.x : fallback[idx].x;
    const v = Number.isFinite(L.y) ? L.y : fallback[idx].y;
    return [
      (u - 0.5) * planeW,
      (0.5 - v) * planeH + centerY,
      u,
      v,
    ];
  };

  // Ensure left (-x) vs right (+x) ordering matches screen space.
  const rawEyeA = pt(LM.eyeAOuter);
  const rawEyeB = pt(LM.eyeBOuter);
  const swapLR = rawEyeA[0] > rawEyeB[0];
  const pick = (idxA, idxB) => (swapLR ? pt(idxB) : pt(idxA));

  const eyeLOuter = pick(LM.eyeAOuter, LM.eyeBOuter);
  const eyeLInner = pick(LM.eyeAInner, LM.eyeBInner);
  const eyeLTop = pick(LM.eyeATop, LM.eyeBTop);
  const eyeLBot = pick(LM.eyeABot, LM.eyeBBot);
  const eyeLIris = src.length > LM.irisB ? pick(LM.irisA, LM.irisB) : null;
  const eyeLCenter = eyeLIris ?? [
    (eyeLOuter[0] + eyeLInner[0]) * 0.5,
    (eyeLTop[1] + eyeLBot[1]) * 0.5,
  ];
  const eyeLWidth = Math.max(Math.hypot(eyeLInner[0] - eyeLOuter[0], eyeLInner[1] - eyeLOuter[1]), 0.04);
  const eyeLHeight = Math.max(Math.hypot(eyeLTop[0] - eyeLBot[0], eyeLTop[1] - eyeLBot[1]), eyeLWidth * 0.30);

  const eyeROuter = pick(LM.eyeBOuter, LM.eyeAOuter);
  const eyeRInner = pick(LM.eyeBInner, LM.eyeAInner);
  const eyeRTop = pick(LM.eyeBTop, LM.eyeATop);
  const eyeRBot = pick(LM.eyeBBot, LM.eyeABot);
  const eyeRIris = src.length > LM.irisB ? pick(LM.irisB, LM.irisA) : null;
  const eyeRCenter = eyeRIris ?? [
    (eyeROuter[0] + eyeRInner[0]) * 0.5,
    (eyeRTop[1] + eyeRBot[1]) * 0.5,
  ];
  const eyeRWidth = Math.max(Math.hypot(eyeROuter[0] - eyeRInner[0], eyeROuter[1] - eyeRInner[1]), 0.04);
  const eyeRHeight = Math.max(Math.hypot(eyeRTop[0] - eyeRBot[0], eyeRTop[1] - eyeRBot[1]), eyeRWidth * 0.30);

  const rawSpan = Math.hypot(eyeROuter[0] - eyeLOuter[0], eyeROuter[1] - eyeLOuter[1]);
  const faceScale = Math.min(
    Math.max(rawSpan, Math.min(planeW, planeH) * 0.14),
    Math.max(planeW, planeH) * 0.85
  );

  const browLInner = pick(LM.browAInner, LM.browBInner);
  const browLMid = pick(105, 334);
  const browLOuter = pick(LM.browAOuter, LM.browBOuter);
  const browRInner = pick(LM.browBInner, LM.browAInner);
  const browRMid = pick(334, 105);
  const browROuter = pick(LM.browBOuter, LM.browAOuter);

  const noseTip = pt(LM.noseTip);
  const noseBridge = pt(LM.noseBridgeLow);
  const noseWingL = pick(LM.noseWingA, LM.noseWingB);
  const noseWingR = pick(LM.noseWingB, LM.noseWingA);
  const noseCreaseL = pick(LM.noseCreaseA, LM.noseCreaseB);
  const noseCreaseR = pick(LM.noseCreaseB, LM.noseCreaseA);

  const mouthCornerL = pick(LM.mouthCornerA, LM.mouthCornerB);
  const mouthCornerR = pick(LM.mouthCornerB, LM.mouthCornerA);
  const upperLip = pt(LM.upperLip);
  const innerUpperLip = pt(LM.innerUpperLip);
  const innerLowerLip = pt(LM.innerLowerLip);
  const lowerLip = pt(LM.lowerLip);
  const lowerLipArcL = pick(LM.lowerLipArcA, LM.lowerLipArcB);
  const lowerLipArcR = pick(LM.lowerLipArcB, LM.lowerLipArcA);
  const mouthCenter = [
    (mouthCornerL[0] + mouthCornerR[0]) * 0.5,
    (innerUpperLip[1] + innerLowerLip[1]) * 0.5,
  ];
  const mouthWidth = Math.max(
    Math.hypot(mouthCornerR[0] - mouthCornerL[0], mouthCornerR[1] - mouthCornerL[1]),
    faceScale * 0.35
  );

  const cheekL = pick(LM.cheekA, LM.cheekB);
  const cheekR = pick(LM.cheekB, LM.cheekA);
  const jawSideL = pick(172, 397);
  const jawSideR = pick(397, 172);
  const chinMid = pt(LM.chinMid);
  const chinBottom = pt(LM.chinBottom);

  const faceCenter = [
    (eyeLCenter[0] + eyeRCenter[0] + mouthCenter[0]) / 3,
    (eyeLCenter[1] + eyeRCenter[1] + mouthCenter[1]) / 3,
  ];

  const features = {
    planeW,
    planeH,
    centerY,
    baseZ,
    faceScale,
    faceCenter,
    eyeL: {
      outer: eyeLOuter,
      inner: eyeLInner,
      top: eyeLTop,
      bot: eyeLBot,
      center: eyeLCenter,
      width: eyeLWidth,
      height: eyeLHeight,
    },
    eyeR: {
      outer: eyeROuter,
      inner: eyeRInner,
      top: eyeRTop,
      bot: eyeRBot,
      center: eyeRCenter,
      width: eyeRWidth,
      height: eyeRHeight,
    },
    browL: { inner: browLInner, mid: browLMid, outer: browLOuter },
    browR: { inner: browRInner, mid: browRMid, outer: browROuter },
    nose: {
      tip: noseTip,
      bridge: noseBridge,
      wingL: noseWingL,
      wingR: noseWingR,
      creaseL: noseCreaseL,
      creaseR: noseCreaseR,
    },
    mouth: {
      cornerL: mouthCornerL,
      cornerR: mouthCornerR,
      upper: upperLip,
      innerUpper: innerUpperLip,
      innerLower: innerLowerLip,
      lower: lowerLip,
      arcL: lowerLipArcL,
      arcR: lowerLipArcR,
      center: mouthCenter,
      width: mouthWidth,
    },
    cheekL,
    cheekR,
    jawSideL,
    jawSideR,
    chinMid,
    chinBottom,
    /** @type {Array<[number, number, number]>} */
    controlAnchors: [],
  };

  const anchor3 = (p) => {
    const u = p[2] ?? (p[0] / planeW + 0.5);
    const v = p[3] ?? (0.5 - (p[1] - centerY) / planeH);
    return /** @type {[number, number, number]} */ ([
      p[0],
      p[1],
      faceReliefZ(p[0], p[1], u, v, features) + 0.015,
    ]);
  };

  // Order matches CONTROLS in src/drag.js (0..15).
  features.controlAnchors = [
    anchor3(mouthCornerL), // 0: mouthCornerL
    anchor3(mouthCornerR), // 1: mouthCornerR
    anchor3(upperLip),     // 2: lipUpperMid
    anchor3(lowerLip),     // 3: lipLowerMid
    anchor3(browLInner),   // 4: browInnerL
    anchor3(browRInner),   // 5: browInnerR
    anchor3(browLOuter),   // 6: browOuterL
    anchor3(browROuter),   // 7: browOuterR
    anchor3(chinBottom),   // 8: chinMid
    anchor3(noseTip),      // 9: noseTip
    anchor3(eyeLTop),      // 10: lidMidL
    anchor3(eyeRTop),      // 11: lidMidR
    anchor3(jawSideL),     // 12: jawSideL
    anchor3(jawSideR),     // 13: jawSideR
    anchor3(cheekL),       // 14: cheekL
    anchor3(cheekR),       // 15: cheekR
  ];

  return features;
}

/**
 * Evaluate the 2D/2.5D morph displacement [dx, dy, dz] for one ARKit channel
 * at mesh vertex (x, y) with normalized image coords (u, v).
 *
 * @param {string} name ARKit channel name
 * @param {number} x vertex x in plane coords
 * @param {number} y vertex y in plane coords
 * @param {number} u normalized image u ∈ [0,1]
 * @param {number} v normalized image v ∈ [0,1]
 * @param {ReturnType<typeof extractImageFeatures>} features
 * @param {Float32Array} out3
 */
export function imageMorphDelta(name, x, y, u, v, features, out3) {
  let dx = 0;
  let dy = 0;
  let dz = 0;
  const S = features.faceScale;
  const { eyeL, eyeR, browL, browR, nose, mouth, cheekL, cheekR, jawSideL, jawSideR, chinBottom } = features;

  const lipSeamY = mouth.center[1];
  const chinSpanY = Math.max(lipSeamY - chinBottom[1], S * 0.22);
  // Smooth lower-jaw mask: 0 above lip seam, ramps quickly below lip seam,
  // and tapers smoothly past the chin.
  let jawGate = 0;
  if (y < lipSeamY + S * 0.025) {
    const tBelow = Math.min(Math.max((lipSeamY + S * 0.02 - y) / (S * 0.045), 0), 1);
    const sBelow = tBelow * tBelow * (3 - 2 * tBelow);
    const yDecay = Math.exp(-0.5 * ((y - (lipSeamY - chinSpanY * 0.55)) / (chinSpanY * 1.05)) ** 2);
    const xDecay = Math.exp(-0.5 * ((x - mouth.center[0]) / (S * 0.56)) ** 2);
    jawGate = sBelow * yDecay * xDecay;
  }

  // Helper for eye lid & iris warps.
  const applyEyeBlink = (eye, sign) => {
    const rx = Math.max(eye.width * 0.68, S * 0.13);
    const openH = Math.max(eye.top[1] - eye.bot[1], eye.height, S * 0.055);
    const ry = Math.max(openH * 1.55, S * 0.095);
    const wUpper = gauss2D(x, y, eye.center[0], eye.top[1] + openH * 0.08, rx, ry);
    const wLower = gauss2D(x, y, eye.center[0], eye.bot[1] - openH * 0.08, rx * 0.9, ry * 0.75);
    if (sign > 0) {
      // Blink closure: upper lid pulls down across eye opening, lower lid rises slightly.
      const upperBias = y >= eye.bot[1] - openH * 0.15 ? 1 : 0.35;
      dy -= wUpper * upperBias * openH * 1.12;
      if (y < eye.center[1]) dy += wLower * openH * 0.22;
    } else {
      // Eye wide: upper lid rises, lower lid drops slightly.
      dy += wUpper * openH * 0.62;
      dy -= wLower * openH * 0.20;
    }
  };

  const applyEyeLook = (eye, dirX, dirY) => {
    const rx = Math.max(eye.width * 0.45, S * 0.075);
    const openH = Math.max(eye.top[1] - eye.bot[1], eye.height, S * 0.055);
    const ry = Math.max(openH * 0.95, S * 0.06);
    const wIris = gauss2D(x, y, eye.center[0], eye.center[1], rx, ry);
    dx += wIris * dirX * eye.width * 0.30;
    dy += wIris * dirY * openH * 0.50;
  };

  switch (name) {
    // --- Jaw ---
    case 'jawOpen': {
      dy -= jawGate * S * 0.25;
      dx -= (x - mouth.center[0]) * jawGate * 0.08;
      // Slight upper-lip stabilize/lift around mouth center
      const wUp = gauss2D(x, y, mouth.upper[0], mouth.upper[1], mouth.width * 0.45, S * 0.07);
      dy += wUp * S * 0.015;
      break;
    }
    case 'jawForward': {
      dy += jawGate * S * 0.035;
      dz += jawGate * S * 0.09;
      break;
    }
    case 'jawLeft': {
      const wSide = gauss2D(x, y, jawSideL[0], jawSideL[1], S * 0.28, S * 0.24);
      dx -= Math.max(jawGate, wSide * 0.85) * S * 0.12;
      break;
    }
    case 'jawRight': {
      const wSide = gauss2D(x, y, jawSideR[0], jawSideR[1], S * 0.28, S * 0.24);
      dx += Math.max(jawGate, wSide * 0.85) * S * 0.12;
      break;
    }
    case 'mouthClose': {
      const wLow = gauss2D(x, y, mouth.lower[0], mouth.lower[1], mouth.width * 0.55, S * 0.11);
      const wUp = gauss2D(x, y, mouth.upper[0], mouth.upper[1], mouth.width * 0.55, S * 0.09);
      dy += wLow * S * 0.09 - wUp * S * 0.025;
      break;
    }

    // --- Mouth Smile / Frown / Dimple / Stretch ---
    case 'mouthSmileLeft': {
      const wC = gauss2D(x, y, mouth.cornerL[0], mouth.cornerL[1], S * 0.21, S * 0.17);
      const wCh = gauss2D(x, y, cheekL[0], cheekL[1], S * 0.25, S * 0.20);
      dy += wC * S * 0.115 + wCh * S * 0.035;
      dx -= wC * S * 0.065 + wCh * S * 0.020;
      break;
    }
    case 'mouthSmileRight': {
      const wC = gauss2D(x, y, mouth.cornerR[0], mouth.cornerR[1], S * 0.21, S * 0.17);
      const wCh = gauss2D(x, y, cheekR[0], cheekR[1], S * 0.25, S * 0.20);
      dy += wC * S * 0.115 + wCh * S * 0.035;
      dx += wC * S * 0.065 + wCh * S * 0.020;
      break;
    }
    case 'mouthDimpleLeft': {
      const w = gauss2D(x, y, mouth.cornerL[0], mouth.cornerL[1], S * 0.18, S * 0.15);
      dx -= w * S * 0.045;
      dy += w * S * 0.035;
      dz -= w * S * 0.03;
      break;
    }
    case 'mouthDimpleRight': {
      const w = gauss2D(x, y, mouth.cornerR[0], mouth.cornerR[1], S * 0.18, S * 0.15);
      dx += w * S * 0.045;
      dy += w * S * 0.035;
      dz -= w * S * 0.03;
      break;
    }
    case 'mouthFrownLeft': {
      const w = gauss2D(x, y, mouth.cornerL[0], mouth.cornerL[1], S * 0.21, S * 0.17);
      dy -= w * S * 0.105;
      dx += w * S * 0.015;
      break;
    }
    case 'mouthFrownRight': {
      const w = gauss2D(x, y, mouth.cornerR[0], mouth.cornerR[1], S * 0.21, S * 0.17);
      dy -= w * S * 0.105;
      dx -= w * S * 0.015;
      break;
    }
    case 'mouthStretchLeft': {
      const w = gauss2D(x, y, mouth.cornerL[0], mouth.cornerL[1], S * 0.22, S * 0.16);
      dx -= w * S * 0.105;
      break;
    }
    case 'mouthStretchRight': {
      const w = gauss2D(x, y, mouth.cornerR[0], mouth.cornerR[1], S * 0.22, S * 0.16);
      dx += w * S * 0.105;
      break;
    }
    case 'mouthLeft': {
      const w = gauss2D(x, y, mouth.center[0], mouth.center[1], mouth.width * 0.75, S * 0.19);
      dx -= w * S * 0.11;
      break;
    }
    case 'mouthRight': {
      const w = gauss2D(x, y, mouth.center[0], mouth.center[1], mouth.width * 0.75, S * 0.19);
      dx += w * S * 0.11;
      break;
    }

    // --- Mouth Pucker / Funnel / Lips ---
    case 'mouthPucker': {
      const w = gauss2D(x, y, mouth.center[0], mouth.center[1], mouth.width * 0.72, S * 0.19);
      dx -= (x - mouth.center[0]) * w * 0.45;
      dy -= (y - mouth.center[1]) * w * 0.12;
      dz += w * S * 0.08;
      break;
    }
    case 'mouthFunnel': {
      const w = gauss2D(x, y, mouth.center[0], mouth.center[1], mouth.width * 0.70, S * 0.19);
      dx -= (x - mouth.center[0]) * w * 0.30;
      const vDir = y >= lipSeamY ? 1 : -1;
      dy += vDir * w * S * 0.045;
      dz += w * S * 0.09;
      break;
    }
    case 'mouthUpperUpLeft': {
      const ax = (mouth.upper[0] + mouth.cornerL[0]) * 0.5;
      const w = Math.max(
        gauss2D(x, y, ax, mouth.upper[1], S * 0.16, S * 0.11),
        gauss2D(x, y, mouth.upper[0], mouth.upper[1], S * 0.16, S * 0.11) * 0.75
      );
      dy += w * S * 0.075;
      break;
    }
    case 'mouthUpperUpRight': {
      const ax = (mouth.upper[0] + mouth.cornerR[0]) * 0.5;
      const w = Math.max(
        gauss2D(x, y, ax, mouth.upper[1], S * 0.16, S * 0.11),
        gauss2D(x, y, mouth.upper[0], mouth.upper[1], S * 0.16, S * 0.11) * 0.75
      );
      dy += w * S * 0.075;
      break;
    }
    case 'mouthLowerDownLeft': {
      const ax = (mouth.lower[0] + mouth.cornerL[0]) * 0.5;
      const w = Math.max(
        gauss2D(x, y, ax, mouth.lower[1], S * 0.16, S * 0.12),
        gauss2D(x, y, mouth.lower[0], mouth.lower[1], S * 0.16, S * 0.12) * 0.75
      );
      dy -= w * S * 0.08;
      break;
    }
    case 'mouthLowerDownRight': {
      const ax = (mouth.lower[0] + mouth.cornerR[0]) * 0.5;
      const w = Math.max(
        gauss2D(x, y, ax, mouth.lower[1], S * 0.16, S * 0.12),
        gauss2D(x, y, mouth.lower[0], mouth.lower[1], S * 0.16, S * 0.12) * 0.75
      );
      dy -= w * S * 0.08;
      break;
    }
    case 'mouthPressLeft': {
      const ax = (mouth.center[0] + mouth.cornerL[0]) * 0.5;
      const w = gauss2D(x, y, ax, mouth.center[1], S * 0.16, S * 0.10);
      dy -= (y - mouth.center[1]) * w * 0.35;
      dz -= w * S * 0.03;
      break;
    }
    case 'mouthPressRight': {
      const ax = (mouth.center[0] + mouth.cornerR[0]) * 0.5;
      const w = gauss2D(x, y, ax, mouth.center[1], S * 0.16, S * 0.10);
      dy -= (y - mouth.center[1]) * w * 0.35;
      dz -= w * S * 0.03;
      break;
    }
    case 'mouthRollLower': {
      const w = gauss2D(x, y, mouth.lower[0], mouth.lower[1], mouth.width * 0.50, S * 0.09);
      dy += w * S * 0.045;
      dz -= w * S * 0.025;
      break;
    }
    case 'mouthRollUpper': {
      const w = gauss2D(x, y, mouth.upper[0], mouth.upper[1], mouth.width * 0.50, S * 0.09);
      dy -= w * S * 0.045;
      dz -= w * S * 0.025;
      break;
    }
    case 'mouthShrugLower': {
      const w = gauss2D(x, y, mouth.lower[0], mouth.lower[1], mouth.width * 0.55, S * 0.12);
      dy += w * S * 0.055;
      break;
    }
    case 'mouthShrugUpper': {
      const w = gauss2D(x, y, mouth.upper[0], mouth.upper[1], mouth.width * 0.55, S * 0.12);
      dy += w * S * 0.055;
      break;
    }

    // --- Eyebrows ---
    case 'browInnerUp': {
      const wL = gauss2D(x, y, browL.inner[0], browL.inner[1], S * 0.21, S * 0.15);
      const wR = gauss2D(x, y, browR.inner[0], browR.inner[1], S * 0.21, S * 0.15);
      dy += (wL + wR) * S * 0.095;
      break;
    }
    case 'browDownLeft': {
      const wI = gauss2D(x, y, browL.inner[0], browL.inner[1], S * 0.21, S * 0.15);
      const wM = gauss2D(x, y, browL.mid[0], browL.mid[1], S * 0.21, S * 0.15);
      const w = Math.max(wI, wM * 0.85);
      dy -= w * S * 0.085;
      dx += wI * S * 0.03;
      break;
    }
    case 'browDownRight': {
      const wI = gauss2D(x, y, browR.inner[0], browR.inner[1], S * 0.21, S * 0.15);
      const wM = gauss2D(x, y, browR.mid[0], browR.mid[1], S * 0.21, S * 0.15);
      const w = Math.max(wI, wM * 0.85);
      dy -= w * S * 0.085;
      dx -= wI * S * 0.03;
      break;
    }
    case 'browOuterUpLeft': {
      const w = Math.max(
        gauss2D(x, y, browL.outer[0], browL.outer[1], S * 0.22, S * 0.15),
        gauss2D(x, y, browL.mid[0], browL.mid[1], S * 0.20, S * 0.14) * 0.7
      );
      dy += w * S * 0.10;
      break;
    }
    case 'browOuterUpRight': {
      const w = Math.max(
        gauss2D(x, y, browR.outer[0], browR.outer[1], S * 0.22, S * 0.15),
        gauss2D(x, y, browR.mid[0], browR.mid[1], S * 0.20, S * 0.14) * 0.7
      );
      dy += w * S * 0.10;
      break;
    }

    // --- Cheeks & Nose ---
    case 'cheekPuff': {
      const wL = gauss2D(x, y, cheekL[0], cheekL[1], S * 0.25, S * 0.22);
      const wR = gauss2D(x, y, cheekR[0], cheekR[1], S * 0.25, S * 0.22);
      dx -= wL * S * 0.10;
      dx += wR * S * 0.10;
      dz += (wL + wR) * S * 0.05;
      break;
    }
    case 'cheekSquintLeft': {
      const w = gauss2D(x, y, cheekL[0], cheekL[1] + S * 0.05, S * 0.24, S * 0.18);
      dy += w * S * 0.06;
      break;
    }
    case 'cheekSquintRight': {
      const w = gauss2D(x, y, cheekR[0], cheekR[1] + S * 0.05, S * 0.24, S * 0.18);
      dy += w * S * 0.06;
      break;
    }
    case 'noseSneerLeft': {
      const w = Math.max(
        gauss2D(x, y, nose.wingL[0], nose.wingL[1], S * 0.15, S * 0.12),
        gauss2D(x, y, nose.tip[0], nose.tip[1], S * 0.15, S * 0.12) * 0.65
      );
      dy += w * S * 0.055;
      break;
    }
    case 'noseSneerRight': {
      const w = Math.max(
        gauss2D(x, y, nose.wingR[0], nose.wingR[1], S * 0.15, S * 0.12),
        gauss2D(x, y, nose.tip[0], nose.tip[1], S * 0.15, S * 0.12) * 0.65
      );
      dy += w * S * 0.055;
      break;
    }

    // --- Eyes: Blink / Wide / Squint / Look ---
    case 'eyeBlinkLeft': applyEyeBlink(eyeL, 1); break;
    case 'eyeBlinkRight': applyEyeBlink(eyeR, 1); break;
    case 'eyeWideLeft': applyEyeBlink(eyeL, -1); break;
    case 'eyeWideRight': applyEyeBlink(eyeR, -1); break;
    case 'eyeSquintLeft': {
      const openH = Math.max(eyeL.top[1] - eyeL.bot[1], eyeL.height, S * 0.055);
      const wLow = gauss2D(x, y, eyeL.center[0], eyeL.bot[1] - openH * 0.15, eyeL.width * 0.75, openH * 1.3);
      const wUp = gauss2D(x, y, eyeL.center[0], eyeL.top[1], eyeL.width * 0.70, openH * 1.1);
      dy += wLow * openH * 0.55 - wUp * openH * 0.22;
      break;
    }
    case 'eyeSquintRight': {
      const openH = Math.max(eyeR.top[1] - eyeR.bot[1], eyeR.height, S * 0.055);
      const wLow = gauss2D(x, y, eyeR.center[0], eyeR.bot[1] - openH * 0.15, eyeR.width * 0.75, openH * 1.3);
      const wUp = gauss2D(x, y, eyeR.center[0], eyeR.top[1], eyeR.width * 0.70, openH * 1.1);
      dy += wLow * openH * 0.55 - wUp * openH * 0.22;
      break;
    }
    case 'eyeLookOutLeft': applyEyeLook(eyeL, -1, 0); break;
    case 'eyeLookInLeft': applyEyeLook(eyeL, 1, 0); break;
    case 'eyeLookOutRight': applyEyeLook(eyeR, 1, 0); break;
    case 'eyeLookInRight': applyEyeLook(eyeR, -1, 0); break;
    case 'eyeLookUpLeft': applyEyeLook(eyeL, 0, 1); break;
    case 'eyeLookUpRight': applyEyeLook(eyeR, 0, 1); break;
    case 'eyeLookDownLeft': applyEyeLook(eyeL, 0, -1); break;
    case 'eyeLookDownRight': applyEyeLook(eyeR, 0, -1); break;

    // --- Tongue ---
    case 'tongueOut': {
      const w = gauss2D(x, y, mouth.innerLower[0], mouth.innerLower[1], mouth.width * 0.36, S * 0.12);
      dy -= w * S * 0.09;
      dz += w * S * 0.11;
      break;
    }
    default:
      break;
  }

  const edge = edgePinWeight(u, v);
  out3[0] = (dx * edge) || 0;
  out3[1] = (dy * edge) || 0;
  out3[2] = (dz * edge) || 0;
}

/**
 * Evaluate the morphed position of a control anchor (0..15) given current
 * 52 channel weights (and optional warpScale).
 *
 * @param {number} controlIdx 0..15
 * @param {Float32Array|number[]} weights52
 * @param {readonly string[]} channels
 * @param {ReturnType<typeof extractImageFeatures>} features
 * @param {Float32Array} out3
 * @param {number} [warpScale]
 * @returns {Float32Array} out3
 */
export function evaluateControlAnchorMorphed(
  controlIdx,
  weights52,
  channels,
  features,
  out3,
  warpScale = 1
) {
  const base = features.controlAnchors[controlIdx] ?? [0, 0, features.baseZ];
  const u = Math.min(Math.max(base[0] / features.planeW + 0.5, 0), 1);
  const v = Math.min(Math.max(0.5 - (base[1] - features.centerY) / features.planeH, 0), 1);
  let x = base[0];
  let y = base[1];
  let z = base[2];
  const d = new Float32Array(3);
  for (let k = 0; k < channels.length; k++) {
    const w = (weights52[k] ?? 0) * warpScale;
    if (w === 0) continue;
    imageMorphDelta(channels[k], base[0], base[1], u, v, features, d);
    x += d[0] * w;
    y += d[1] * w;
    z += d[2] * w;
  }
  out3[0] = x;
  out3[1] = y;
  out3[2] = z;
  return out3;
}
