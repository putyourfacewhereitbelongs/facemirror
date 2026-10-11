/**
 * Vision runtime loader.
 *
 * The MediaPipe Tasks bundle (~4 MB of JS + WASM) and the landmarker weights
 * are pulled from the jsDelivr CDN and Google's model bucket at runtime, the
 * same way the standalone page always did. The dynamic import is marked
 * @vite-ignore so the bundler leaves the absolute URL alone.
 */

export const VISION_VERSION = '0.10.14';
export const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}`;
export const MODELS = {
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
};

let vision = null;

export async function loadVision() {
  if (vision) return vision;
  const url = `${CDN}/vision_bundle.mjs`;
  vision = await import(/* @vite-ignore */ url);
  return vision;
}
