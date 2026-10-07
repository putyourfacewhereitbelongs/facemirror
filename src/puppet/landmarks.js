/**
 * MediaPipe FaceLandmarker index tables.
 *
 * The landmarker emits 478 points: 468 face-mesh vertices plus 10 iris points
 * (468..472 left eye, 473..477 right eye). Every group below is a stable,
 * hand-checked subset of that topology; they drive the rig, the depth prior,
 * the shading mesh and the overlay colours.
 */

/** Face oval, ordered clockwise starting at the forehead (10). */
export const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

/** Overlay groups: only these ~100 points are drawn as draggable landmarks. */
export const GROUPS = {
  oval: [10, 338, 297, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 152, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 109, 67],
  leye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  reye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  brow: [70, 63, 105, 66, 107, 300, 293, 334, 296, 336],
  lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
  inner: [78, 82, 13, 312, 308, 317, 14, 87],
  nose: [4, 168],
};

export const COLORS = { oval: '#60a5fa', leye: '#5eead4', reye: '#5eead4', brow: '#fbbf24', lips: '#f472b6', inner: '#fb7185', nose: '#a78bfa' };

/** Per-group drag cap, as a fraction of face height: lips travel further than eyelids. */
export const DRAG_CAP = { oval: 0.05, leye: 0.015, reye: 0.015, brow: 0.05, lips: 0.07, inner: 0.07, nose: 0.02 };

export const GROUP_NAMES = { oval: 'Face outline', leye: 'Eye', reye: 'Eye', brow: 'Eyebrow', lips: 'Lips', inner: 'Inner mouth', nose: 'Nose' };

/** Flattened, parallel arrays describing every draggable landmark. */
export const SHOWN = (() => {
  const idx = [], color = [], cap = [], group = [];
  for (const key in GROUPS) for (const i of GROUPS[key]) { idx.push(i); color.push(COLORS[key]); cap.push(DRAG_CAP[key]); group.push(key); }
  return { idx, color, cap, group, count: idx.length };
})();

/** Landmarks whose expression delta is treated as "nose" (extra flexibility slider). */
export const NOSE_PTS = new Set([49, 279, 64, 294, 48, 278, 115, 344, 98, 327, 129, 358, 219, 439, 218, 438]);
/** Nose bridge: gets the "nose flexibility" gain. */
export const NOSE_BRIDGE = new Set([1, 2, 4, 5, 6, 19, 94, 168, 195, 197]);
export const NOSE_RIDGE = [168, 6, 197, 195, 5, 4, 1];
export const NOSE_TIP = [1, 4];
export const NOSE_WING = [49, 279, 98, 327, 205, 425];

/** Rigid set used for the Kabsch head-pose fit: skull outline + bridge + eye corners. */
export const RIG = [10, 338, 297, 284, 251, 389, 356, 454, 323, 93, 234, 127, 162, 21, 54, 103, 67, 109, 168, 6, 197, 195, 5, 4, 33, 133, 362, 263];
/** Weights for the rigid fit: the skull outline dominates pose, the nose is expression-polluted. */
export const RIG_W = RIG.map(i => (i === 4 || i === 5 || i === 6 || i === 197 || i === 195 || i === 168 ? 0.35 : i === 33 || i === 133 || i === 362 || i === 263 ? 0.8 : 1));

/** Eye rings (16 points each) and the four anchor points used for gaze/eyelid maths. */
export const EYE = [
  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
];
/** [upper lid, lower lid, inner corner, outer corner] per eye. */
export const EYM = [[159, 145, 33, 133], [386, 374, 362, 263]];
export const IRIS = [468, 473];
export const UPPER_LID = new Set([159, 158, 160, 157, 161, 246, 386, 385, 387, 384, 388, 466]);
export const LOWER_LID = new Set([145, 153, 144, 154, 163, 7, 374, 380, 373, 381, 390, 249]);

/** Inner lip contour (the wet/dry border) as an ordered loop. */
export const LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
/** Upper and lower halves of that loop, both running corner -> corner. */
export const UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
export const LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
export const LIPSET = new Set([...GROUPS.lips, ...GROUPS.inner, ...LIP]);
/** Upper lip is anchored to the maxilla: it must not translate when the jaw opens. */
export const MAXILLA = new Set([...UP, 61, 291, 0, 37, 39, 40, 185, 267, 269, 270, 409]);
/** Mandible-carried points: they follow the jaw hinge (chin, jaw line, lower lip). */
export const MANDIBLE = [152, 175, 148, 176, 149, 150, 136, 172, 58, 200, 199, 419, 215, 435, 214, 434, 178, 88, 95, 87, 14];

export const CHIN = [152, 175, 148, 176, 149];
export const CHEEK = [205, 425, 101, 330, 213, 433, 50, 280, 116, 345, 216, 436, 117, 346, 118, 347, 206, 426, 209, 438, 49, 279];
export const FOREHEAD = [10, 151, 290, 338, 297, 284, 251, 109, 67, 107, 336, 8, 80, 168];
export const BROW = GROUPS.brow;
export const INNER_BROW = new Set([66, 107, 336, 296, 55, 285]);
export const TEMPLE = [234, 454, 162, 389, 127, 356, 21, 54];

/** dlib/iBUG 68-point topology expressed as MediaPipe indices (classic-map overlay). */
export const IBUG68 = [
  // jaw 0-16
  234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 389,
  // right brow 17-21
  70, 63, 105, 66, 107,
  // left brow 22-26
  300, 293, 334, 296, 336,
  // nose bridge 27-30
  168, 6, 197, 195,
  // nostrils 31-35
  4, 49, 279, 205, 425,
  // right eye 36-41
  33, 160, 158, 133, 153, 144,
  // left eye 42-47
  362, 385, 387, 263, 373, 380,
  // outer lip 48-59
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
  // inner lip 60-67
  78, 191, 80, 81, 82, 13, 312, 311,
];

/** Facial Action Coding System style labels over the MediaPipe blendshape names. */
export const AU_MAP = [
  ['AU1 inner brow raiser', ['browInnerUp']],
  ['AU2 outer brow raiser', ['browOuterUp']],
  ['AU4 brow lowerer', ['browDownLeft', 'browDownRight']],
  ['AU5 upper lid raiser', ['eyeWideLeft', 'eyeWideRight']],
  ['AU6 cheek raiser', ['cheekSquintLeft', 'cheekSquintRight']],
  ['AU9 nose wrinkler', ['noseSneerLeft', 'noseSneerRight']],
  ['AU10 upper lip raiser', ['mouthUpperUpLeft', 'mouthUpperUpRight']],
  ['AU12 lip corner puller', ['mouthSmileLeft', 'mouthSmileRight']],
  ['AU15 lip corner depressor', ['mouthFrownLeft', 'mouthFrownRight']],
  ['AU18 lip puckerer', ['mouthPucker', 'mouthFunnel']],
  ['AU20 lip stretcher', ['mouthStretchLeft', 'mouthStretchRight']],
  ['AU25 lips part', ['mouthClose', 'jawOpen']],
  ['AU26 jaw drop', ['jawOpen']],
  ['AU43 eye closure', ['eyeBlinkLeft', 'eyeBlinkRight']],
  ['AU45 blink', ['eyeBlinkLeft', 'eyeBlinkRight']],
];
