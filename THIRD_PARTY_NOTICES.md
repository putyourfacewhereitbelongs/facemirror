# Third-party notices

The repository vendors the runtime dependencies used by both the default
2.5D puppet and FaceMirror-3D Studio. Core face tracking can therefore start
without relying on a live CDN; the puppet's optional shoulder/body tracker is
loaded separately and may be unavailable without stopping face animation.

## three.js (0.186.1) — MIT License

Vendored at `vendor/three.module.js`, `vendor/three.core.js`,
`vendor/examples/jsm/` (GLTFLoader, GLTFExporter, OrbitControls,
BufferGeometryUtils, SkeletonUtils).
Copyright © 2010-2026 three.js authors. <https://github.com/mrdoob/three.js>

## MediaPipe Tasks Vision (1.1.0) — Apache License 2.0

Vendored at `vendor/tasks-vision/` (`vision_bundle.mjs` + wasm).
Copyright © Google LLC. <https://github.com/google-ai-edge/mediapipe>

## MediaPipe Face Landmarker model (float16/1) — Apache License 2.0

`assets/face_landmarker.task` (and `vendor/face_landmarker.task`),
3,758,596 bytes, sha256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>

## Delaunator 5.1.0 — ISC License

Vendored at `vendor/delaunator.min.js` for the default puppet page.
The matching license is included at `vendor/DELAUNATOR-LICENSE.txt`.
Copyright © 2026 Mapbox. <https://github.com/mapbox/delaunator>

## stabilizeBlink math

The blink-stabilization math in `src/eyes.js` is a clean-room port of the
idea from KalidoKit (MIT) — the library itself is **not** a dependency and
its pre-blendshape face solver is deliberately not used.
