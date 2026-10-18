# Third-party notices

FaceMirror-3D vendors its runtime dependencies so the app runs offline after
the first load and never depends on a live download.

## three.js (0.186.1) — MIT License

Vendored at `vendor/three.module.js`, `vendor/three.core.js`,
`vendor/examples/jsm/` (GLTFLoader, GLTFExporter, OrbitControls,
BufferGeometryUtils, SkeletonUtils).
Copyright © 2010-2026 three.js authors. <https://github.com/mrdoob/three.js>

## MediaPipe Tasks Vision (1.1.0) — Apache License 2.0

Vendored at `vendor/tasks-vision/` (`vision_bundle.mjs` + wasm).
Copyright © Google LLC. <https://github.com/google-ai-edge/mediapipe>

FaceMirror modification: the bundle's usage-telemetry sink constructor is
replaced with a disabled sink. It does not start a logging timer, queue usage
events, or send the automatic log POST when an IMAGE/VIDEO task is closed.
The original license header is retained and the modification is marked at
the top of the bundle. Inference/model code is unchanged.

## MediaPipe Face Landmarker model (float16/1) — Apache License 2.0

`assets/face_landmarker.task` (and `vendor/face_landmarker.task`),
3,758,596 bytes, sha256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>

## stabilizeBlink math

The blink-stabilization math in `src/eyes.js` is a clean-room port of the
idea from KalidoKit (MIT) — the library itself is **not** a dependency and
its pre-blendshape face solver is deliberately not used.

## Infinite head scan — Creative Commons Attribution 3.0 Unported

**Infinite, 3D Head Scan by Lee Perry-Smith**, based on a work at
<https://www.triplegangers.com/>, supplied by Infinite Realities
(<http://www.ir-ltd.net/>).
License: <https://creativecommons.org/licenses/by/3.0/>.

Mesh, albedo, and tangent-space normal map are vendored under
`assets/anatomical-head/` from three.js r180, commit
`9e8635e2031c25859dc47ba07e72230dccb2682a`,
`examples/models/gltf/LeePerrySmith/`.
Original notice: `assets/anatomical-head/LICENSE.txt`.

FaceMirror adapts the scan at runtime: scale/offset, cropped and capped neck,
opened eye surfaces, added eyeballs and animated lids, approximate facial
morphs, and physical skin shading. The source files themselves are unchanged
apart from local filenames. See that directory's README for provenance and
checksums. Credit is also visible in the app's Help panel.

## MediaPipe canonical face topology — Apache License 2.0

`assets/face-topology.json` contains only the 898 triangles' zero-based vertex
indices derived from `mediapipe/modules/face_geometry/data/canonical_face_model.obj`
in <https://github.com/google-ai-edge/mediapipe>, pinned to commit
`a908d668c730da128dfa8d9f6bd25d519d006692` (source blob
`0e666d1c4e75949d1639c2bcf347a38da4834164`). Vertex positions, UVs and normals
are omitted; the indices are used for landmark-to-portrait correspondence.
Copyright © Google LLC. Full license: `assets/face-topology.LICENSE.txt`.
Derived JSON SHA-256:
`db2a680eb2e70e477f3dbfeba1c04344531083bc4df811928f86edc452faf5d7`.
