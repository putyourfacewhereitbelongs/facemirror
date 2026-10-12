/**
 * renderer.js — three.js scene, studio lighting, skin material with the
 * cheap SSS approximation, tone mapping (spec §7).
 *
 * Trap §11.10: eyes never share the skin material (one material reads
 * plastic). The SSS is ONE onBeforeCompile patch, commented below — true SSS
 * is out (no post passes, no raymarch).
 */

import * as THREE from 'three';
import { applyWeights } from './headAsset.js';

/** @type {Record<string, number|number[]>} */
export const CONFIG = {
  fov: 32,
  camZ: 2.35,
  camY: 0.06,
  pixelRatioCap: 1.25,        // §7
  skinRoughness: 0.45,        // 0.35–0.55 band (§7)
  lipsClearcoat: 0.12,        // lips region only (§7)
  skinClearcoat: 0.0,
  sssWrap: 0.45,              // N·L wrap (§7): clamp((N·L+0.45)/1.45)
  sssFresnel: 0.18,           // (1−N·V)²·0.18·albedo·(1,0.35,0.25)
  sssBlurMix: 0.25,           // blurred-normal mip-4 diffuse trick (§7)
  keyIntensity: 0.95,         // warm key, 55° cone
  keyAngleDeg: 55,
  fillIntensity: 0.35,        // cool fill
  rimIntensity: 0.6,          // tight rim, upper back
  breathAmp: 0.002,           // ±2 mm y breath, 0.18 Hz (§7)
  breathHz: 0.18,
  envSize: 256,               // procedural equirect gradient (§7)
  antiParallax: 0.012,        // anti-phase camera counter-rotate (§5.4)
  clearColor: 0x0b0d12,
};

/**
 * Apply the ONE cheap-SSS shader patch (§7) to a physical material.
 * Comments cite the three trap numbers this protects against.
 * @param {THREE.MeshPhysicalMaterial} material
 */
export function applySkinSSS(material) {
  material.onBeforeCompile = (shader) => {
    // 1) Wrap lighting: diffuse N·L → clamp((N·L+0.45)/1.45, 0, 1).
    //    Half-Lambert wrap fakes shallow light bleed under the skin with no
    //    extra passes. NOTE: onBeforeCompile runs BEFORE #include expansion,
    //    so we patch the chunk template itself and inline it (a plain string
    //    replace on the main shader would silently no-op).
    const physicalChunk = THREE.ShaderChunk.lights_physical_pars_fragment;
    const wrappedChunk = physicalChunk.replace(
      'float dotNL = saturate( dot( geometryNormal, directLight.direction ) );',
      `float dotNL = clamp( ( dot( geometryNormal, directLight.direction ) + ${CONFIG.sssWrap} ) / ${(1 + CONFIG.sssWrap).toFixed(2)}, 0.0, 1.0 );`
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <lights_physical_pars_fragment>',
      wrappedChunk
    );
    // 2) Blurred-normal diffuse trick: re-sample the same normal map at
    //    explicit mip 4 and gently mix the shading normal toward it —
    //    approximates multi-scatter blur (only meaningful with a normal map).
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <normal_fragment_maps>',
      `#include <normal_fragment_maps>
      #ifdef USE_NORMALMAP
        vec3 sssBlurN = normalize( textureLod( normalMap, vNormalMapUv, 4.0 ).xyz * 2.0 - 1.0 );
        normal = normalize( mix( normal, sssBlurN, ${CONFIG.sssBlurMix} ) );
      #endif`
    );
    // 3) Red-tint fresnel glow: (1−N·V)²·0.18·albedo·(1,0.35,0.25) — the
    //    ear/nose-lit-up look. `outgoingLight` is in scope right before
    //    <opaque_fragment> in the main() assembly.
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <opaque_fragment>',
      `float sssFres = pow( 1.0 - saturate( dot( normal, normalize( vViewPosition ) ) ), 2.0 );
      outgoingLight += sssFres * ${CONFIG.sssFresnel} * diffuseColor.rgb * vec3( 1.0, 0.35, 0.25 );
      #include <opaque_fragment>`
    );
  };
  material.customProgramCacheKey = () => 'facemirror-sss-v1';
}

/**
 * Build the procedural/studio environment: 256px equirect gradient with two
 * painted "area rect" highlights, run through PMREM once (§7).
 * @param {THREE.WebGLRenderer} renderer
 * @returns {THREE.Texture}
 */
export function buildEnvironment(renderer) {
  const S = CONFIG.envSize;
  const data = new Uint8Array(S * (S / 2) * 4);
  const W = S;
  const H = S / 2;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = y / H; // 0 top … 1 bottom
      // Sky → horizon → floor gradient.
      let r = 28 + 60 * (1 - v);
      let g = 32 + 66 * (1 - v);
      let b = 42 + 80 * (1 - v);
      // Two soft area rects (studio softboxes), baked into the gradient.
      const box1 = Math.exp(-(((x - W * 0.22) / (W * 0.10)) ** 2 + ((y - H * 0.30) / (H * 0.16)) ** 2));
      const box2 = Math.exp(-(((x - W * 0.72) / (W * 0.13)) ** 2 + ((y - H * 0.36) / (H * 0.20)) ** 2));
      r += box1 * 175 + box2 * 95;
      g += box1 * 168 + box2 * 105;
      b += box1 * 150 + box2 * 125;
      const i = (y * W + x) * 4;
      data[i] = Math.min(r, 255);
      data[i + 1] = Math.min(g, 255);
      data[i + 2] = Math.min(b, 255);
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const env = pmrem.fromEquirectangular(tex).texture;
  pmrem.dispose();
  tex.dispose();
  return env;
}

/**
 * Studio scene wrapper.
 */
export class Stage {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    /** @type {THREE.WebGLRenderer|null} */
    this.renderer = null;
    this.webglFailed = false;
    try {
      this.renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: false,             // §7
        powerPreference: 'high-performance',
        alpha: false,
      });
    } catch (err) {
      // Graceful: no WebGL at all → 2D fallback canvas stays (§0.5).
      this.webglFailed = true;
      this.ctx2d = canvas.getContext('2d');
    }
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(CONFIG.clearColor);
    this.camera = new THREE.PerspectiveCamera(CONFIG.fov, 16 / 9, 0.1, 40);
    this.camera.position.set(0, CONFIG.camY, CONFIG.camZ);
    this.cameraTarget = new THREE.Vector3(0, 0, 0);
    this.headNode = new THREE.Group(); // pose.js quaternion lands here (§7)
    this.scene.add(this.headNode);

    if (this.renderer) {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping; // §7
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer.shadowMap.enabled = false;                 // §7
      this.setPixelRatio(Math.min(typeof devicePixelRatio === 'number' ? devicePixelRatio : 1, CONFIG.pixelRatioCap));
      this.env = buildEnvironment(this.renderer);
      this.scene.environment = this.env;
    }

    // Lights (§7): warm key spot 55°, cool fill, tight rim upper back.
    const key = new THREE.SpotLight(0xffe8d2, CONFIG.keyIntensity * 12);
    key.angle = (CONFIG.keyAngleDeg * Math.PI) / 180;
    key.penumbra = 0.45;
    key.position.set(1.6, 2.1, 2.6);
    const fill = new THREE.DirectionalLight(0xb9c8ff, CONFIG.fillIntensity * 2.2);
    fill.position.set(-2.2, 0.6, 1.8);
    const rim = new THREE.DirectionalLight(0xfff2e0, CONFIG.rimIntensity * 2.6);
    rim.position.set(0.4, 2.2, -2.6);
    const amb = new THREE.AmbientLight(0x2a2f3a, 1.1);
    this.scene.add(key, fill, rim, amb);

    // Skin materials (§7 / trap §11.10): eyes get their own glossy material
    // from headAsset; skull gets this physical material + SSS patch.
    this.skinMat = new THREE.MeshPhysicalMaterial({
      color: 0xc99e86,
      roughness: CONFIG.skinRoughness,
      metalness: 0,
      clearcoat: CONFIG.skinClearcoat,
      envMapIntensity: 0.9,
    });
    applySkinSSS(this.skinMat);
    this.lipsMat = new THREE.MeshPhysicalMaterial({
      color: 0xb06e62,
      roughness: 0.38,
      metalness: 0,
      clearcoat: CONFIG.lipsClearcoat, // §7: clearcoat on the lips region only
      envMapIntensity: 1.0,
    });
    applySkinSSS(this.lipsMat);

    /** @type {import('./headAsset.js').HeadAsset|null} */
    this.asset = null;
    this.renderScale = 1;
    this._t = 0;
  }

  /**
   * @param {number} pr
   */
  setPixelRatio(pr) {
    if (this.renderer) this.renderer.setPixelRatio(pr);
  }

  /**
   * Perf governor quality tier (§8): 1.25 → 1.0 → 0.85 then detection cuts.
   * @param {number} scale
   */
  setRenderScale(scale) {
    this.renderScale = scale;
    if (this.renderer) {
      this.renderer.setPixelRatio(Math.min(typeof devicePixelRatio === 'number' ? devicePixelRatio : 1, CONFIG.pixelRatioCap) * scale);
    }
  }

  /**
   * Install a head asset (GLB or procedural).
   * @param {import('./headAsset.js').HeadAsset} asset
   */
  setHead(asset) {
    if (this.asset) this.headNode.remove(this.asset.root);
    this.asset = asset;
    // Assign skin/lips materials to meshes that come with none (procedural)
    // or a neutral default (GLB without materials).
    for (const mesh of asset.meshes) {
      if (!mesh.material || (mesh.material.isMeshStandardMaterial && !mesh.material.map)) {
        mesh.material = Array.isArray(mesh.material) ? mesh.material : this.skinMat;
      }
      if (mesh.material && mesh.material.isMeshStandardMaterial && mesh.userData?.lips) {
        mesh.material = this.lipsMat;
      }
    }
    this.headNode.add(asset.root);
  }

  /**
   * Per-substep pose application (§7): head quaternion + breath sine.
   * @param {Float32Array} quat4 [x,y,z,w] from pose.js
   * @param {Float32Array} pos3
   * @param {number} dtSec
   * @param {boolean} idle breathing only when idle (§7)
   */
  applyPose(quat4, pos3, dtSec, idle) {
    this._t += dtSec;
    this.headNode.quaternion.set(quat4[0], quat4[1], quat4[2], quat4[3]);
    const breath = idle ? Math.sin(this._t * CONFIG.breathHz * 2 * Math.PI) * CONFIG.breathAmp : 0;
    this.headNode.position.set(pos3[0], pos3[1] + breath, pos3[2]);
    // Tiny anti-phase counter-rotate of the camera target for parallax feel
    // (§5.4): makes the head feel attached to a neck, not floating.
    this.cameraTarget.set(
      -quat4[1] * CONFIG.antiParallax,
      -quat4[0] * CONFIG.antiParallax * 0.6,
      0
    );
    this.camera.lookAt(this.cameraTarget);
  }

  /**
   * Resize.
   * @param {number} w @param {number} h
   */
  resize(w, h) {
    if (this.renderer) this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  /**
   * Draw one frame. Returns false in 2D fallback mode.
   * @param {Float32Array} weights52
   * @param {number} jawCap
   * @returns {boolean}
   */
  render(weights52, jawCap = 1) {
    if (!this.renderer) {
      // 2D graceful fallback (§0.5): draw a liveliness meter so the app
      // still "runs" without WebGL.
      const g = this.ctx2d;
      if (g) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        g.fillStyle = '#0b0d12';
        g.fillRect(0, 0, w, h);
        g.fillStyle = '#e8c8a8';
        g.beginPath();
        g.ellipse(w / 2, h / 2, h * 0.22, h * 0.3, 0, 0, Math.PI * 2);
        g.fill();
        const jaw = weights52[24] ?? 0; // jawOpen
        g.fillStyle = '#5a3a32';
        g.beginPath();
        g.ellipse(w / 2, h / 2 + h * 0.12, h * 0.07 * (1 + jaw * 1.4), h * 0.03 + h * 0.05 * jaw, 0, 0, Math.PI * 2);
        g.fill();
        const b = weights52[9] ?? 0; // eyeBlinkRight-ish column
        g.fillStyle = '#222';
        const eyeY = h / 2 - h * 0.06;
        g.fillRect(w / 2 - h * 0.13, eyeY - h * 0.012 * (1 - b), h * 0.07, h * 0.024 * (1 - b) + 1);
        g.fillRect(w / 2 + h * 0.06, eyeY - h * 0.012 * (1 - b), h * 0.07, h * 0.024 * (1 - b) + 1);
        g.fillStyle = '#8fa';
        g.fillRect(0, h - 6, w * Math.min(weights52[43] + weights52[44], 1), 6); // smile meter
      }
      return false;
    }
    if (this.asset) {
      applyWeights(this.asset, weights52, jawCap);
      // Procedural head extras: lid shells + pupil scale from eye channels.
      if (this.asset.isProcedural && this.asset.setBlink) {
        this.asset.setBlink(weights52[8], weights52[9]);
      }
    }
    this.renderer.render(this.scene, this.camera);
    return true;
  }
}
