/**
 * faceImage.js — the single runtime face-image → likeness pipeline.
 * Files, dropped portraits, rendered canvases and webcam snapshots all use
 * this path. Inference/baking are local and one-time, not part of the rAF loop.
 * A bad/replaced/cancelled input cannot partially overwrite the current face.
 */
import * as THREE from 'three';
import { CONFIG as VISION } from './tracker.js';
import {
  FIT_LANDMARKS, LatestFaceSource, validatePortrait,
  makeIdentityControls, makeFaceWarp, deformIdentityGeometry,
} from './likeness.js';
import { mapPortraitSurface, sampleSkinTone, sampleScalpTone, samplePortraitPixel, bakePortraitAtlas } from './portraitTexture.js';

export const PHOTO_LIMITS = Object.freeze({ maxBytes: 12 * 1024 * 1024, maxDimension: 2048 });
let filesetPromise = null;
let topologyPromise = null;

/** Separate IMAGE task: never reset the running VIDEO tracker/timestamps. */
export async function createPhotoLandmarker() {
  if (!filesetPromise) {
    filesetPromise = import(VISION.visionModule).then(async ({ FilesetResolver, FaceLandmarker }) => ({
      FaceLandmarker, fileset: await FilesetResolver.forVisionTasks(VISION.wasmLocal),
    })).catch((err) => { filesetPromise = null; throw err; });
  }
  const { fileset, FaceLandmarker } = await filesetPromise;
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: VISION.modelLocal, delegate: 'CPU' },
    runningMode: 'IMAGE', numFaces: 2, // refuse ambiguous multi-person photos
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5,
  });
}

async function getTopology() {
  if (!topologyPromise) {
    topologyPromise = fetch(new URL('../assets/face-topology.json', import.meta.url)).then(async (res) => {
      if (!res.ok) throw new Error('Local face topology is unavailable. Reload the app assets');
      return res.json();
    }).catch((err) => { topologyPromise = null; throw err; });
  }
  return topologyPromise;
}

/** Copy *now*, not after async inference: a rendered canvas/video can change. */
export function snapshotFaceImage(image) {
  const width = image.videoWidth || image.naturalWidth || image.width;
  const height = image.videoHeight || image.naturalHeight || image.height;
  if (!width || !height) throw new Error('The face image is not ready yet');
  const scale = Math.min(1, PHOTO_LIMITS.maxDimension / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#bfc3c8';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height); // EXIF is baked by the decoder
  ctx.getImageData(0, 0, 1, 1); // reject tainted cross-origin canvases before touching the rig
  return canvas;
}

async function decodeFaceImage(file) {
  if (file.size > PHOTO_LIMITS.maxBytes) throw new Error('Choose a portrait smaller than 12 MB');
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)
    && !(file.type === '' && /\.(jpe?g|png|webp)$/i.test(file.name ?? ''))) {
    throw new Error('Choose a JPEG, PNG or WebP portrait (or drop a .glb head)');
  }
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    return snapshotFaceImage(bitmap);
  } catch (err) {
    if (err?.name === 'SecurityError') throw new Error('Use a local image or a same-origin rendered face');
    throw new Error('This image could not be read. Try a JPEG, PNG or WebP portrait');
  } finally { bitmap?.close(); }
}

function canvasTexture(canvas) {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 2;
  return tex;
}

function makeIrisTexture(canvas, landmarks, index, pixels) {
  const center = landmarks[index];
  if (!center) return null;
  const edges = landmarks.slice(index + 1, index + 5);
  const radius = edges.reduce((sum, p) => sum + Math.hypot((p.x - center.x) * canvas.width, (p.y - center.y) * canvas.height), 0) / 4 * 0.90;
  if (!Number.isFinite(radius) || radius < 1) return null;
  const ring = (index === 468
    ? [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    : [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]).map((i) => landmarks[i]);
  const insideEye = (x, y) => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const a = ring[i], b = ring[j];
      if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside;
    }
    return inside;
  };
  const iris = document.createElement('canvas');
  iris.width = iris.height = 256;
  const ctx = iris.getContext('2d'), image = ctx.createImageData(256, 256);
  const rgb = [0, 0, 0];
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
    const dx = (x - 127.5) / 51.2, dy = (y - 127.5) / 51.2, r = Math.hypot(dx, dy);
    const i = (y * 256 + x) * 4;
    if (r > 1.02) { image.data.set([210, 205, 198, 255], i); continue; }
    let u = center.x + dx * radius / canvas.width, v = center.y + dy * radius / canvas.height;
    if (!insideEye(u, v)) {
      // An iris is often partly hidden by a smiling eyelid. Fill its hidden
      // arcs from the visible horizontal iris band, not copied skin/sclera.
      u = center.x + (dx < 0 ? -1 : 1) * Math.min(0.95, r) * radius / canvas.width;
      v = center.y;
    }
    samplePortraitPixel(pixels, canvas.width, canvas.height, u, v, rgb);
    for (let k = 0; k < 3; k++) image.data[i + k] = rgb[k];
    image.data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvasTexture(iris);
}

/** Snapshot the template once. Keep its CPU data for replacement/reset. */
function rememberAsset(asset) {
  const geometries = new Map(), materials = new Map(), eyes = new Map();
  asset.root.traverse((obj) => {
    if (obj.isMesh) {
      geometries.set(obj, obj.geometry);
      for (const mat of Array.isArray(obj.material) ? obj.material : [obj.material]) {
        materials.set(mat, { map: mat.map, normalMap: mat.normalMap, roughness: mat.roughness, color: mat.color.clone() });
      }
    }
  });
  for (const eye of Object.values(asset.eyeNodes)) if (eye) eyes.set(eye, eye.position.toArray());
  return { geometries, materials, eyes, hash: asset.hash, setPupil: asset.setPupil };
}

function buildReference(asset, capture, landmarks, topology, rotation) {
  const { camera, canvas, head } = capture;
  const raycaster = new THREE.Raycaster();
  const anchors = [];
  head.updateMatrixWorld(true);
  for (const index of FIT_LANDMARKS) {
    const lm = landmarks[index];
    raycaster.setFromCamera(new THREE.Vector2(lm.x * 2 - 1, 1 - lm.y * 2), camera);
    const hit = raycaster.intersectObject(head, true)[0];
    if (hit) anchors.push({ index, point: hit.point.toArray() });
  }
  if (anchors.length < 40) throw new Error('Could not map the template face. Try reloading the app');
  const reference = { landmarks, anchors, rotation, width: canvas.width, height: canvas.height, surfaces: new Map(), topology };
  const point = new THREE.Vector3();
  for (const mesh of asset.meshes) {
    const geo = mesh.geometry, p = geo.attributes.position;
    const projected = new Float32Array(p.count * 2);
    for (let i = 0; i < p.count; i++) {
      point.fromBufferAttribute(p, i).project(camera);
      projected[i * 2] = point.x * 0.5 + 0.5;
      projected[i * 2 + 1] = 0.5 - point.y * 0.5;
    }
    reference.surfaces.set(mesh, {
      projected, positions: p.array, normals: geo.attributes.normal.array,
      atlasUV: geo.attributes.uv.array, indices: geo.index.array,
    });
  }
  return reference;
}

/**
 * Runtime source controller. Inference tasks close in finally; old GPU buffers
 * and private textures are disposed on replace/reset. Portraits/features are
 * NEVER uploaded or persisted to localStorage. Only explicit head export keeps
 * the fitted mesh/atlas. A thumbnail is drawn synchronously by onChange.
 */
export class FaceImageSource {
  constructor({ getAsset, captureReference, onBusy = () => {}, onChange = () => {}, onError = () => {}, detectorFactory = createPhotoLandmarker }) {
    Object.assign(this, { getAsset, captureReference, onBusy, onChange, onError, detectorFactory });
    this.jobs = new LatestFaceSource();
    this.reference = null;
    this.original = null;
    this.textures = [];
    this.state = { kind: 'generic', label: 'Generic preview' };
  }

  cancelPending() { this.jobs.cancel(); this.onBusy(false); }

  /**
   * image: File/Blob or CanvasImageSource (the active rendered source image).
   * landmarks are optional for an already-detected webcam snapshot, not required
   * for normal images/canvases. Every source uses the same bake/fit/commit path.
   */
  setImage(image, { label = image?.name || 'Current face image', kind = 'image', landmarks = null, matrix = null } = {}) {
    this.jobs.cancel(); // even an invalid newest source cancels older pending work
    let snapshot = null;
    try { if (!(image instanceof Blob)) snapshot = snapshotFaceImage(image); }
    catch (err) { this.onBusy(false); this.onError(err); return Promise.resolve(false); }
    const known = landmarks?.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    matrix = matrix ? Array.from(matrix) : null; // freeze webcam pose with the frame
    this.onBusy(true);
    return this.jobs.run(async (current) => {
      let detector = null, canvas = snapshot, product = null, committed = false;
      try {
        const asset = this.getAsset();
        if (!asset?.isAnatomical) throw new Error('Photo likeness needs the bundled anatomical head. Reload the app assets');
        if (!canvas) canvas = await decodeFaceImage(image);
        if (!current()) return false;
        // Let the loading state paint before the one-time CPU work.
        await new Promise((resolve) => requestAnimationFrame(resolve));
        if (!current()) return false;
        if (!known || !this.reference) detector = await this.detectorFactory();
        if (!current()) return false;
        const result = known ? { faceLandmarks: [known], facialTransformationMatrixes: matrix ? [{ data: matrix }] : [] } : detector.detect(canvas);
        const lm = validatePortrait(result, canvas.width, canvas.height).map((p) => ({ x: p.x, y: p.y, z: p.z }));
        // Own these values before the detector is called again for the template
        // or closed; some runtimes expose borrowed typed-array result storage.
        const rotation = result.facialTransformationMatrixes?.[0]?.data;
        const portraitRotation = rotation ? Array.from(rotation) : null;
        if (!this.reference) {
          const topology = await getTopology();
          if (!current()) return false;
          const capture = this.captureReference(asset);
          try {
            const detected = detector.detect(capture.canvas);
            const base = detected.faceLandmarks?.[0]?.map((p) => ({ x: p.x, y: p.y, z: p.z }));
            if (!base) throw new Error('Could not detect the template face. Reload the app and try again');
            this.reference = buildReference(asset, capture, base, topology, Array.from(detected.facialTransformationMatrixes?.[0]?.data ?? new THREE.Matrix4().elements));
            this.original = rememberAsset(asset);
          } finally { capture.canvas.width = capture.canvas.height = 1; }
        }
        if (!current()) return false;
        product = this._build(asset, canvas, lm, portraitRotation);
        if (!current()) return false;
        this._commit(asset, product);
        this.state = { kind, label, version: this.jobs.version };
        asset.likeness = this.state;
        asset.hash = `${this.original.hash}:local-face-${this.jobs.version}`;
        committed = true;
        this.onChange({ asset, canvas, landmarks: lm, ...this.state });
        return true;
      } catch (err) {
        if (current()) this.onError(err);
        return false;
      } finally {
        detector?.close();
        if (!committed && product) {
          for (const geo of product.geometries.values()) geo.dispose();
          for (const tex of product.textures) tex.dispose();
        }
        if (canvas) canvas.width = canvas.height = 1;
        if (current()) this.onBusy(false);
      }
    }).finally(() => { if (snapshot) snapshot.width = snapshot.height = 1; });
  }

  _build(asset, canvas, landmarks, rotation) {
    const reference = this.reference, original = this.original;
    const controls = makeIdentityControls(reference, landmarks, canvas.width, canvas.height, rotation);
    const warp = makeFaceWarp(controls);
    const fittedEyes = new Map();
    for (const [eye, position] of original.eyes) {
      const next = new Float64Array(3); warp(...position, next);
      fittedEyes.set(eye, Array.from(next));
    }
    const geometries = new Map();
    const textures = [];
    try {
      for (const [mesh, source] of original.geometries) {
        const eye = original.eyes.has(mesh.parent) ? mesh.parent : null;
        geometries.set(mesh, deformIdentityGeometry(source, warp,
          eye ? original.eyes.get(eye) : undefined, eye ? fittedEyes.get(eye) : undefined));
      }
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      const skin = sampleSkinTone(pixels, canvas.width, canvas.height, landmarks);
      const scalp = sampleScalpTone(pixels, canvas.width, canvas.height, landmarks, reference, skin);
      const surfaces = [...reference.surfaces.values()].map((surface) => mapPortraitSurface(surface, reference, landmarks, reference.topology, canvas.width, canvas.height));
      const baked = bakePortraitAtlas(surfaces, pixels, canvas.width, canvas.height, skin, scalp);
      if (baked.photoPixels < 1000) throw new Error('Could not project this portrait onto the face');
      const atlas = document.createElement('canvas');
      atlas.width = baked.width; atlas.height = baked.height;
      atlas.getContext('2d').putImageData(new ImageData(baked.data, baked.width, baked.height), 0, 0);
      const texture = canvasTexture(atlas); textures.push(texture);
      const irises = new Map();
      for (const [eye, index] of [[asset.eyeNodes.left, 468], [asset.eyeNodes.right, 473]]) {
        const iris = makeIrisTexture(canvas, landmarks, index, pixels);
        if (eye && iris) { irises.set(eye.children[1].material, iris); textures.push(iris); }
      }
      return { geometries, fittedEyes, texture, irises, textures, warp };
    } catch (err) {
      for (const geo of geometries.values()) geo.dispose();
      for (const tex of textures) tex.dispose();
      throw err;
    }
  }

  _commit(asset, product) {
    asset.fitPoint = product.warp;
    for (const [mesh, geo] of product.geometries) { mesh.geometry.dispose(); mesh.geometry = geo; }
    for (const [eye, position] of product.fittedEyes) eye.position.fromArray(position);
    for (const mesh of asset.meshes) {
      const mat = mesh.material;
      mat.map = product.texture;
      mat.normalMap = null; // no other person's pores/brows/stubble from the scan
      mat.color.set(0xffffff);
      mat.roughness = 0.68;
      mat.needsUpdate = true;
    }
    for (const [mat, texture] of product.irises) { mat.map = texture; mat.needsUpdate = true; }
    asset.setPupil = (s) => {
      const scale = 1 / Math.max(0.65, Math.min(1.3, s));
      for (const tex of product.irises.values()) {
        tex.repeat.set(scale, scale); tex.offset.set((1 - scale) / 2, (1 - scale) / 2);
      }
    };
    for (const texture of this.textures) texture.dispose();
    this.textures = product.textures;
  }

  /** Explicit reset only: normal expression Reset does not forget the person. */
  clear() {
    this.cancelPending();
    const asset = this.getAsset();
    if (this.original && asset) {
      for (const [mesh, geo] of this.original.geometries) { if (mesh.geometry !== geo) mesh.geometry.dispose(); mesh.geometry = geo; }
      for (const [eye, position] of this.original.eyes) eye.position.fromArray(position);
      for (const [material, saved] of this.original.materials) {
        Object.assign(material, saved); material.color = saved.color.clone(); material.needsUpdate = true;
      }
      asset.hash = this.original.hash;
      asset.setPupil = this.original.setPupil;
      delete asset.likeness;
      delete asset.fitPoint;
    }
    for (const texture of this.textures) texture.dispose();
    this.textures = [];
    this.state = { kind: 'generic', label: 'Generic preview' };
    this.onChange({ asset, canvas: null, ...this.state });
  }
}
