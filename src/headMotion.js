/** Apply webcam/manual/recorded motion without touching an image identity. */
import { CHANNELS, CH } from './rig.js';

/**
 * Apply 52 weights to a loaded head (GLB or procedural).
 * Eye-look channels rotate eye bones when present (§6), else morphs apply.
 * @param {import('./headAsset.js').HeadAsset} asset
 * @param {Float32Array} weights52
 * @param {number} [jawCap] optional jawOpen cap (no-teeth assets)
 */
export function applyWeights(asset, weights52, jawCap = 1) {
  for (const name of CHANNELS) {
    const targets = asset.morphMap[name];
    if (!targets) continue;
    let w = weights52[CH[name]];
    if (name === 'jawOpen' && w > jawCap) w = jawCap;
    for (const t of targets) {
      const inf = t.mesh.morphTargetInfluences;
      if (inf) inf[t.index] = w;
    }
  }
  asset.updateLids?.(weights52);
  // Eye bones (trap §11.3 naming resolved upstream; look weights are read
  // from the eyeLook* channels).
  const lookYaw = (weights52[CH.eyeLookOutLeft] - weights52[CH.eyeLookInLeft])
    - (weights52[CH.eyeLookOutRight] - weights52[CH.eyeLookInRight]);
  const lookPitch = weights52[CH.eyeLookUpLeft] - weights52[CH.eyeLookDownLeft];
  if (asset.eyeNodes.left) {
    asset.eyeNodes.left.rotation.y = lookYaw * 0.45;
    asset.eyeNodes.left.rotation.x = -lookPitch * 0.45;
  }
  if (asset.eyeNodes.right) {
    asset.eyeNodes.right.rotation.y = lookYaw * 0.45;
    asset.eyeNodes.right.rotation.x = -lookPitch * 0.45;
  }
}
