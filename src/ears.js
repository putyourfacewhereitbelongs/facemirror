/**
 * ears.js — lightweight, closed auricles for the asset-free fallback.
 * The anatomical default has scanned ears in its skull mesh; these keep the
 * same rigid attachment/depth behavior if that local asset cannot be read.
 */
import * as THREE from '../vendor/three.core.js';

export function buildFallbackEars() {
  const material = new THREE.MeshPhysicalMaterial({
    color: 0xc99e86, roughness: 0.55, metalness: 0,
    vertexColors: true, envMapIntensity: 0.9,
  });
  material.userData.skin = true;
  const makeEar = (side) => {
    const segments = 48;
    const rings = 18;
    const positions = [], colors = [], indices = [];
    const front = [], back = [];
    const add = (r, angle, rear) => {
      const v = Math.cos(angle) * 0.265 * r;
      const u = Math.sin(angle) * 0.14 * r + v * 0.07;
      const thickness = Math.sqrt(Math.max(0, 1 - r * r));
      const concha = Math.exp(-1 * ((u + 0.025) / 0.065) ** 2 - ((v + 0.025) / 0.11) ** 2);
      const helix = 0.035 * Math.exp(-1 * ((r - 0.83) / 0.10) ** 2);
      const antihelix = 0.018 * Math.exp(-1 * ((r - 0.48) / 0.13) ** 2) * Math.max(0, Math.min(1, (v + 0.14) / 0.12));
      const tragus = 0.023 * Math.exp(-1 * ((u + 0.095) / 0.025) ** 2 - ((v + 0.025) / 0.055) ** 2);
      const lobe = 0.016 * Math.exp(-1 * ((v + 0.20) / 0.075) ** 2);
      const depth = rear ? -0.035 * thickness : (0.032 - concha * 0.022 + helix + antihelix + tragus + lobe) * thickness;
      const i = positions.length / 3;
      positions.push(side * (0.77 + u * 0.42 + depth * 0.9075), 0.015 + v, -0.015 - u * 0.9075 + depth * 0.42);
      const crease = rear ? 0.08 : concha * 0.18;
      colors.push(1 - crease * 0.30, 1 - crease, 1 - crease * 1.05);
      return i;
    };
    const triangle = (a, b, c, rear) => {
      // Reflection changes winding. Reverse indices, not the object's scale:
      // FrontSide culling then remains correct on BOTH ears during turns.
      if ((side < 0) !== rear) indices.push(a, c, b);
      else indices.push(a, b, c);
    };
    for (const rear of [false, true]) {
      const surface = rear ? back : front;
      surface.push([add(0, 0, rear)]);
      for (let j = 1; j <= rings; j++) {
        // Share the perimeter between the front and rear sheets: no open
        // rim and no paper-thin back face revealed when the head rotates.
        surface.push(Array.from({ length: segments }, (_, i) => rear && j === rings
          ? front[rings][i] : add(j / rings, i / segments * Math.PI * 2, rear)));
        for (let i = 0; i < segments; i++) {
          const next = (i + 1) % segments;
          if (j === 1) triangle(surface[0][0], surface[j][next], surface[j][i], rear);
          else {
            triangle(surface[j - 1][i], surface[j - 1][next], surface[j][i], rear);
            triangle(surface[j - 1][next], surface[j][next], surface[j][i], rear);
          }
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const ear = new THREE.Mesh(geometry, material);
    ear.name = side < 0 ? 'leftEar' : 'rightEar';
    return ear;
  };
  return { left: makeEar(-1), right: makeEar(1) };
}
