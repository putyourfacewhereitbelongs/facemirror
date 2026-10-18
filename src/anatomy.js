/**
 * anatomy.js — local scan-based head geometry and its facial rig.
 *
 * The auricles are part of the SAME opaque mesh as the skull. No billboards,
 * landmark-derived ear positions, negative scales, or yaw-dependent hiding:
 * the head transform and the depth buffer handle every viewing angle.
 * Import the pinned core directly so geometry regressions also run in Node,
 * without a DOM, WebGL, or an npm install (the browser uses this same core).
 */
import * as THREE from '../vendor/three.core.js';
import { CHANNELS, CH } from './rig.js';
import { proceduralMorphDelta } from './headShape.js';

export const ANATOMY = Object.freeze({
  sourceScale: 0.4,
  sourceOffsetY: -0.56,
  neckY: -1.12,
  eyeX: 0.28,
  eyeY: 0.105,
  eyeZ: 0.665,
  eyeRadius: 0.145,
  eyeRadiusY: 0.105,
  eyeRadiusZ: 0.12,
  eyeWidth: 0.139,
  upperLid: 0.052,
  lowerLid: 0.033,
  lidSegments: 64,
  lidRings: 5,
});

function smoothstep(lo, hi, v) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
}

/** Facial tissue only: the ears, rear cranium, and neck remain rigid. */
export function faceDeformationMask(x, y, z) {
  return smoothstep(0.24, 0.55, z)
    * (1 - smoothstep(0.50, 0.64, Math.abs(x)))
    * smoothstep(-1.05, -0.62, y);
}

/** Shared morph fields, smoothly pinned before they reach an ear root. */
export function anatomicalMorphDelta(name, x, y, z, out) {
  // These channels are authored on the curved lid collars below. Moving the
  // surrounding scan as well opens a crack at the collar's fixed outer edge.
  if (/^eye(Blink|Wide|Squint)/.test(name)) {
    out.fill(0);
    return out;
  }
  proceduralMorphDelta(name, x, y, z, out);
  const mask = faceDeformationMask(x, y, z);
  for (let c = 0; c < 3; c++) out[c] = mask === 0 ? 0 : out[c] * mask;
  return out;
}

/**
 * Clip off the scan's shoulders, interpolating attributes at the neck plane.
 * A separate downward-facing cap closes the neck, including on upward nods.
 * The head/ears above the plane keep their original topology and normals.
 */
export function clipNeck(source, planeY = ANATOMY.neckY) {
  const p = source.getAttribute('position');
  const n = source.getAttribute('normal');
  const uv = source.getAttribute('uv');
  const positions = Array.from(p.array);
  const normals = Array.from(n.array);
  const uvs = Array.from(uv.array);
  const indices = [];
  const border = new Map();
  const intersections = new Map();
  const index = source.index;

  const cut = (a, b) => {
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    if (intersections.has(key)) return intersections.get(key);
    const t = (planeY - positions[a * 3 + 1]) / (positions[b * 3 + 1] - positions[a * 3 + 1]);
    const i = positions.length / 3;
    for (let c = 0; c < 3; c++) {
      positions.push(positions[a * 3 + c] + (positions[b * 3 + c] - positions[a * 3 + c]) * t);
      normals.push(normals[a * 3 + c] + (normals[b * 3 + c] - normals[a * 3 + c]) * t);
    }
    positions[i * 3 + 1] = planeY;
    for (let c = 0; c < 2; c++) uvs.push(uvs[a * 2 + c] + (uvs[b * 2 + c] - uvs[a * 2 + c]) * t);
    intersections.set(key, i);
    border.set(`${positions[i * 3].toFixed(5)}:${positions[i * 3 + 2].toFixed(5)}`, i);
    return i;
  };

  for (let i = 0; i < index.count; i += 3) {
    const face = [index.getX(i), index.getX(i + 1), index.getX(i + 2)];
    const polygon = [];
    for (let j = 0; j < 3; j++) {
      const a = face[j];
      const b = face[(j + 1) % 3];
      const inA = positions[a * 3 + 1] >= planeY;
      const inB = positions[b * 3 + 1] >= planeY;
      if (inA) polygon.push(a);
      if (inA !== inB) polygon.push(cut(a, b));
    }
    for (let j = 1; j + 1 < polygon.length; j++) indices.push(polygon[0], polygon[j], polygon[j + 1]);
  }

  if (border.size >= 3) {
    const ring = [...border.values()];
    const cx = ring.reduce((s, i) => s + positions[i * 3], 0) / ring.length;
    const cz = ring.reduce((s, i) => s + positions[i * 3 + 2], 0) / ring.length;
    ring.sort((a, b) => Math.atan2(positions[a * 3 + 2] - cz, positions[a * 3] - cx)
      - Math.atan2(positions[b * 3 + 2] - cz, positions[b * 3] - cx));
    const center = positions.length / 3;
    positions.push(cx, planeY, cz);
    normals.push(0, -1, 0);
    uvs.push(0.5, 0.5);
    for (const i of ring) {
      positions.push(positions[i * 3], planeY, positions[i * 3 + 2]);
      normals.push(0, -1, 0);
      uvs.push(uvs[i * 2], uvs[i * 2 + 1]);
    }
    for (let i = 0; i < ring.length; i++) indices.push(center, center + 1 + i, center + 1 + (i + 1) % ring.length);
  }

  // Compact away the unused shoulder vertices as well as their GPU storage.
  const used = [...new Set(indices)];
  const remap = new Map(used.map((i, j) => [i, j]));
  const geo = new THREE.BufferGeometry();
  for (const [name, data, size] of [['position', positions, 3], ['normal', normals, 3], ['uv', uvs, 2]]) {
    geo.setAttribute(name, new THREE.Float32BufferAttribute(used.flatMap((i) => data.slice(i * size, i * size + size)), size));
  }
  geo.setIndex(indices.map((i) => remap.get(i)));
  geo.normalizeNormals();
  return geo;
}

function eyeSeam(side, dx) {
  return ANATOMY.eyeY - side * dx * 0.10;
}

/**
 * Subtract smooth, convex eye apertures from the scan. Centroid-only triangle
 * removal leaves sawtooth holes around the lids, particularly in profile.
 * Split crossing triangles and interpolate their UVs/normals instead.
 */
export function cutEyeSockets(source) {
  const p = source.getAttribute('position');
  const n = source.getAttribute('normal');
  const uv = source.getAttribute('uv');
  const positions = Array.from(p.array);
  const normals = Array.from(n.array);
  const uvs = Array.from(uv.array);
  const indices = [];
  const cache = new Map();
  const vertex = (i) => ({ i, p: [p.getX(i), p.getY(i), p.getZ(i)], n: [n.getX(i), n.getY(i), n.getZ(i)], uv: [uv.getX(i), uv.getY(i)] });
  const append = (v) => {
    if (v.i !== undefined) return v.i;
    const key = [...v.p, ...v.uv].map((x) => x.toFixed(6)).join(':');
    if (cache.has(key)) return cache.get(key);
    const i = positions.length / 3;
    positions.push(...v.p); normals.push(...v.n); uvs.push(...v.uv);
    cache.set(key, i);
    return i;
  };
  const triangulate = (polygon) => {
    for (let j = 1; j + 1 < polygon.length; j++) indices.push(append(polygon[0]), append(polygon[j]), append(polygon[j + 1]));
  };
  const outlines = [-1, 1].map((side) => Array.from({ length: 64 }, (_, i) => {
    const angle = i / 64 * Math.PI * 2;
    const dx = Math.cos(angle) * 0.190;
    return [side * ANATOMY.eyeX + dx, eyeSeam(side, dx) + Math.sin(angle) * (Math.sin(angle) > 0 ? 0.118 : 0.093)];
  }));
  const split = (polygon, a, b) => {
    const distances = polygon.map((v) => (b[0] - a[0]) * (v.p[1] - a[1]) - (b[1] - a[1]) * (v.p[0] - a[0]));
    if (distances.every((d) => d >= 0)) return [polygon, []];
    if (distances.every((d) => d < 0)) return [[], polygon];
    const inner = [], outer = [];
    for (let j = 0; j < polygon.length; j++) {
      const v = polygon[j], w = polygon[(j + 1) % polygon.length];
      const d = distances[j], e = distances[(j + 1) % polygon.length];
      (d >= 0 ? inner : outer).push(v);
      if ((d >= 0) !== (e >= 0)) {
        const t = d / (d - e);
        const cut = {};
        for (const attr of ['p', 'n', 'uv']) cut[attr] = v[attr].map((x, c) => x + (w[attr][c] - x) * t);
        inner.push(cut); outer.push(cut);
      }
    }
    return [inner, outer];
  };
  for (let i = 0; i < source.index.count; i += 3) {
    const ids = [source.index.getX(i), source.index.getX(i + 1), source.index.getX(i + 2)];
    const xs = ids.map((j) => p.getX(j)), ys = ids.map((j) => p.getY(j));
    const side = (xs[0] + xs[1] + xs[2]) < 0 ? -1 : 1;
    const cx = side * ANATOMY.eyeX;
    if (Math.max(...xs) < cx - 0.195 || Math.min(...xs) > cx + 0.195
      || Math.max(...ys) < ANATOMY.eyeY - 0.12 || Math.min(...ys) > ANATOMY.eyeY + 0.15
      || ids.every((j) => p.getZ(j) < 0.46)) {
      indices.push(...ids);
      continue;
    }
    let remaining = ids.map(vertex);
    const outline = outlines[side < 0 ? 0 : 1];
    for (let j = 0; j < outline.length && remaining.length; j++) {
      const [inner, outer] = split(remaining, outline[j], outline[(j + 1) % outline.length]);
      triangulate(outer);
      remaining = inner;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.normalizeNormals();
  return geo;
}

/** Read skin depth + UV from the uncut surface, for a seamless eyelid collar. */
function skinSampler(geometry) {
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  mesh.updateMatrixWorld(true);
  const ray = new THREE.Raycaster();
  const down = new THREE.Vector3(0, 0, -1);
  return (x, y) => {
    ray.set(new THREE.Vector3(x, y, 2), down);
    const hit = ray.intersectObject(mesh, false)[0];
    return hit ? { z: hit.point.z, u: hit.uv.x, v: hit.uv.y, normal: hit.normal.toArray() }
      : { z: 0.58, u: 0.5, v: 0.5, normal: [0, 0, 1] };
  };
}

/** A true 3D lid: closed and open edges both follow the eyeball's curvature. */
export function eyelidPoint(side, angle, ring, blink, outerZ, out) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const innerX = c * ANATOMY.eyeWidth;
  const outerX = c * 0.195;
  const innerY = eyeSeam(side, innerX) + s * (s > 0 ? ANATOMY.upperLid : ANATOMY.lowerLid) * (1 - blink);
  const outerY = eyeSeam(side, outerX) + s * (s > 0 ? 0.123 : 0.098);
  const innerDepth = 1 - (innerX / ANATOMY.eyeRadius) ** 2 - ((innerY - ANATOMY.eyeY) / ANATOMY.eyeRadiusY) ** 2;
  const innerZ = ANATOMY.eyeZ + ANATOMY.eyeRadiusZ * Math.sqrt(Math.max(0, innerDepth)) + 0.004;
  out[0] = side * ANATOMY.eyeX + innerX * (1 - ring) + outerX * ring;
  out[1] = innerY * (1 - ring) + outerY * ring;
  out[2] = innerZ * (1 - ring) + outerZ * ring;
  // A straight interpolation cuts through a spherical eyeball. Keep the
  // entire lid outside it, not just the inner edge (including mid-blink).
  const dx = out[0] - side * ANATOMY.eyeX;
  const dy = out[1] - ANATOMY.eyeY;
  const sphere = 1 - (dx / ANATOMY.eyeRadius) ** 2 - (dy / ANATOMY.eyeRadiusY) ** 2;
  if (sphere > 0) out[2] = Math.max(out[2], ANATOMY.eyeZ + ANATOMY.eyeRadiusZ * Math.sqrt(sphere) + 0.006);
  return out;
}

function wireMorphs(mesh, morphMap, sampleDelta) {
  const geometry = mesh.geometry;
  const position = geometry.getAttribute('position');
  const normalBase = new THREE.BufferGeometry();
  normalBase.setAttribute('position', position);
  normalBase.setIndex(geometry.index);
  normalBase.computeVertexNormals();
  const baseNormal = normalBase.getAttribute('normal');
  const d = new Float32Array(3);
  geometry.morphAttributes.position = [];
  geometry.morphAttributes.normal = [];
  for (const [k, name] of CHANNELS.entries()) {
    const deltas = new Float32Array(position.count * 3);
    const posed = position.clone();
    for (let i = 0; i < position.count; i++) {
      sampleDelta(name, i, d);
      deltas.set(d, i * 3);
      posed.setXYZ(i, position.getX(i) + d[0], position.getY(i) + d[1], position.getZ(i) + d[2]);
    }
    geometry.morphAttributes.position.push(new THREE.Float32BufferAttribute(deltas, 3));
    // Morph normals too; otherwise a rotating, smiling face keeps the old
    // lighting and reads like rubber. No normal recomputation in the loop.
    const probe = new THREE.BufferGeometry();
    probe.setAttribute('position', posed);
    probe.setIndex(geometry.index);
    probe.computeVertexNormals();
    const normalDeltas = new Float32Array(position.count * 3);
    for (let i = 0; i < position.count; i++) {
      for (let c = 0; c < 3; c++) normalDeltas[i * 3 + c] = probe.attributes.normal.array[i * 3 + c] - baseNormal.array[i * 3 + c];
    }
    geometry.morphAttributes.normal.push(new THREE.Float32BufferAttribute(normalDeltas, 3));
    probe.dispose();
    (morphMap[name] ??= []).push({ mesh, index: k });
  }
  normalBase.dispose();
  geometry.morphTargetsRelative = true;
  mesh.updateMorphTargets();
  mesh.morphTargetDictionary = Object.fromEntries(CHANNELS.map((name, i) => [name, i]));
  geometry.computeBoundingSphere();
}

function makeLid(side, material, sampleSkin, morphMap) {
  const { lidSegments: S, lidRings: R } = ANATOMY;
  const positions = [];
  const uvs = [];
  const indices = [];
  const outerDepth = [];
  const skinNormals = [];
  const point = new Float32Array(3);
  for (let j = 0; j <= R; j++) {
    for (let i = 0; i <= S; i++) {
      const a = i / S * Math.PI * 2;
      eyelidPoint(side, a, 1, 0, 0, point);
      const depth = sampleSkin(point[0], point[1]).z + 0.003;
      outerDepth.push(depth);
      eyelidPoint(side, a, j / R, 0, depth, point);
      positions.push(...point);
      const uv = sampleSkin(point[0], point[1]);
      uvs.push(uv.u, uv.v);
      skinNormals.push(...uv.normal);
    }
  }
  for (let j = 0; j < R; j++) {
    for (let i = 0; i < S; i++) {
      const a = j * (S + 1) + i;
      const b = a + S + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const normals = geometry.getAttribute('normal');
  for (let i = 0; i < normals.count; i++) {
    const t = Math.floor(i / (S + 1)) / R;
    const blend = smoothstep(0, 1, t);
    for (let c = 0; c < 3; c++) normals.array[i * 3 + c] = normals.array[i * 3 + c] * (1 - blend) + skinNormals[i * 3 + c] * blend;
  }
  geometry.normalizeNormals();
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = side < 0 ? 'leftLids' : 'rightLids';
  const own = side < 0 ? 'Left' : 'Right';
  wireMorphs(mesh, morphMap, (name, i, out) => {
    const blink = name === `eyeBlink${own}` ? 1 : name === `eyeSquint${own}` ? 0.45 : name === `eyeWide${own}` ? -0.30 : 0;
    if (blink) {
      eyelidPoint(side, i % (S + 1) / S * Math.PI * 2, Math.floor(i / (S + 1)) / R, blink, outerDepth[i], out);
      for (let c = 0; c < 3; c++) out[c] -= positions[i * 3 + c];
    } else {
      anatomicalMorphDelta(name, ...positions.slice(i * 3, i * 3 + 3), out);
      if (/^eye(Blink|Wide|Squint)/.test(name)) out.fill(0);
    }
  });
  return mesh;
}

function makeEye(side, irisTexture) {
  const eye = new THREE.Group();
  eye.name = side < 0 ? 'leftEye' : 'rightEye';
  eye.position.set(side * ANATOMY.eyeX, ANATOMY.eyeY, ANATOMY.eyeZ);
  const sclera = new THREE.MeshPhysicalMaterial({
    color: 0xe2d9cf, roughness: 0.28, metalness: 0, clearcoat: 0.7,
    clearcoatRoughness: 0.09, envMapIntensity: 0.65,
  });
  const globe = new THREE.SphereGeometry(ANATOMY.eyeRadius, 32, 24);
  globe.scale(1, ANATOMY.eyeRadiusY / ANATOMY.eyeRadius, ANATOMY.eyeRadiusZ / ANATOMY.eyeRadius);
  eye.add(new THREE.Mesh(globe, sclera));
  // A curved iris cap, not a flat disc or an equirectangular iris painted on
  // the back of a sphere. UVs select the iris region of makeIrisTexture().
  const positions = [0, 0, ANATOMY.eyeRadiusZ + 0.001];
  const uvs = [0.5, 0.5];
  const indices = [];
  const segments = 48;
  const rings = 6;
  for (let j = 1; j <= rings; j++) {
    const r = j / rings * 0.057;
    for (let i = 0; i < segments; i++) {
      const a = i / segments * Math.PI * 2;
      const x = Math.cos(a) * r, y = Math.sin(a) * r;
      const depth = 1 - (x / ANATOMY.eyeRadius) ** 2 - (y / ANATOMY.eyeRadiusY) ** 2;
      positions.push(x, y, ANATOMY.eyeRadiusZ * Math.sqrt(depth) + 0.001);
      uvs.push(0.5 + Math.cos(a) * j / rings * 0.20, 0.5 + Math.sin(a) * j / rings * 0.20);
      const b = 1 + (j - 1) * segments + i;
      const next = 1 + (j - 1) * segments + (i + 1) % segments;
      if (j === 1) indices.push(0, b, next);
      else indices.push(b - segments, b, next, b - segments, next, next - segments);
    }
  }
  const irisGeo = new THREE.BufferGeometry();
  irisGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  irisGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  irisGeo.setIndex(indices);
  irisGeo.computeVertexNormals();
  const irisMat = new THREE.MeshPhysicalMaterial({
    map: irisTexture, roughness: 0.24, metalness: 0, clearcoat: 1,
    clearcoatRoughness: 0.06, envMapIntensity: 0.8,
  });
  eye.add(new THREE.Mesh(irisGeo, irisMat));
  return eye;
}

/**
 * Build an animated head from the locally bundled scan. Ears are never
 * assigned facial morph deltas; only the common parent rotates them.
 * Materials/textures are supplied by the browser loader, not this module.
 */
export function buildAnatomicalRig(source, skinMaterial, irisTexture) {
  const normalized = source.clone();
  normalized.scale(ANATOMY.sourceScale, ANATOMY.sourceScale, ANATOMY.sourceScale);
  normalized.translate(0, ANATOMY.sourceOffsetY, 0);
  const geometry = clipNeck(normalized);
  normalized.dispose();
  const sampleSkin = skinSampler(geometry);
  const morphMap = {};
  const lidL = makeLid(-1, skinMaterial, sampleSkin, morphMap);
  const lidR = makeLid(1, skinMaterial, sampleSkin, morphMap);
  const cut = cutEyeSockets(geometry);
  geometry.dispose();
  const position = cut.getAttribute('position');
  const head = new THREE.Mesh(cut, skinMaterial);
  head.name = 'anatomicalHeadAndEars';
  wireMorphs(head, morphMap, (name, i, out) => anatomicalMorphDelta(name, position.getX(i), position.getY(i), position.getZ(i), out));
  const eyeL = makeEye(-1, irisTexture);
  const eyeR = makeEye(1, irisTexture);
  const root = new THREE.Group();
  root.name = 'AnatomicalHead';
  root.add(head, lidL, lidR, eyeL, eyeR);
  const lidChannels = [
    { lid: lidL, blink: CH.eyeBlinkLeft, squint: CH.eyeSquintLeft, wide: CH.eyeWideLeft },
    { lid: lidR, blink: CH.eyeBlinkRight, squint: CH.eyeSquintRight, wide: CH.eyeWideRight },
  ];
  return {
    root, meshes: [head, lidL, lidR], morphMap,
    warnings: [], missingChannels: [],
    eyeNodes: { left: eyeL, right: eyeR },
    // The scan is a closed-mouth surface, not an ARKit scan with oral parts.
    // Preserve the existing rig's no-teeth jaw cap rather than promising teeth.
    hasTeeth: false, hasTongue: false, isProcedural: false,
    isAnatomical: true, hash: 'anatomical-head-v1',
    eyeTexture: irisTexture,
    updateLids: (weights) => {
      for (const { lid, blink, squint, wide } of lidChannels) {
        // Additive morphs must not pull a closed lid through its partner when
        // a camera blink and an emotion-driven squint happen together.
        lid.morphTargetInfluences[blink] = Math.max(-0.30, Math.min(1, weights[blink] + 0.45 * weights[squint] - 0.30 * weights[wide]));
        lid.morphTargetInfluences[squint] = lid.morphTargetInfluences[wide] = 0;
      }
    },
    setPupil: (s) => {
      const scale = 1 / Math.max(0.65, Math.min(1.3, s));
      irisTexture.repeat.set(scale, scale);
      irisTexture.offset.set((1 - scale) / 2, (1 - scale) / 2);
    },
  };
}
