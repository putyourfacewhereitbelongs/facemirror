# FaceMirror Studio

A real-time facial tracking and neural static-portrait reenactment studio. Its production path uses FasterLivePortrait with TensorRT on a persistent NVIDIA GPU and streams frames through a backpressured WebSocket. The client automatically falls back to LivePortrait ONNX through browser WebGPU when no GPU service is available.

## What runs for real

- **FasterLivePortrait on TensorRT**: warm, persistent GPU inference over a low-latency WebSocket; see [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md)
- **LivePortrait ONNX on WebGPU fallback**: appearance feature extraction, motion extraction, 3D implicit keypoint transfer, stitching, and neural frame generation
- Approximately 326 MB is downloaded on the first neural run and retained in the browser Cache Storage
- Model inference and webcam frames remain on-device; model hosting only serves static weight files
- **MediaPipe Face Mesh** (self-hosted WASM + packed model assets; no runtime CDN)
- 468-point webcam face tracking (478 with refined iris landmarks)
- Live jaw, smile, blink, brow, yaw, and roll signals derived from geometry
- Static portrait retargeting through an 850+ triangle identity mesh using source-image pixels
- Procedurally shaded oral cavity with individual tooth divisions, tongue depth, shadows, and wet highlights
- Click-to-pose Smile, Frown, and O-mouth landmark presets
- PNG frame capture and 30 FPS WebM recording of the rendered output
- Continuous full-speed tracking while the webcam preview is minimized
- Local image upload, configurable motion gains, temporal controls, and display grading

Research systems whose official weights are unavailable, server-sized, or incompatible with browser inference are clearly labeled as interactive visualizations/analogues in the UI. The app does not pretend to run unreleased models such as VASA-1 or EMO.

## Trill Face Puppet (`/puppet.html`)

A second, independent workspace: a single still photograph turned into a rigid 3D
puppet and driven live from a webcam. It shares nothing with the neural path above
except the build. No model weights are downloaded per frame and nothing leaves the
machine, so the steady-state cost is canvas work only; the **Quality** slider
(0/1/2) trades triangle padding, shading resolution and ROI restoration for frame
time on weaker hardware.

Open it from the studio index (**Trill Face Puppet** card) or go straight to
`/puppet.html`.

### Pipeline

1. **Detect** - MediaPipe `FaceLandmarker` (478 points with irises) plus
   `PoseLandmarker` for shoulders, loaded from the jsDelivr CDN at runtime
   (`@mediapipe/tasks-vision@0.10.14`, ~4 MB, marked `@vite-ignore`).
2. **Reconstruct** - the 2.5D photo is lifted into a canonical head space:
   an analytic ellipsoid depth prior blended with the detector's own smoothed
   relative z, a focal length of `4.2 x face width`, and a Delaunay triangulation
   merged with MediaPipe's official tessellation. Three scalp rings and a body
   skirt are welded around the head so the silhouette has something to pull on.
3. **Retarget** - weighted Kabsch over the rigid subset gives yaw/pitch/roll,
   which are re-gained per axis, soft-limited to about +/-49 deg yaw, and smoothed
   by dt-aware scalars with rate limits. Tracker scale becomes a camera dolly, not
   a zoom; tracker confidence becomes the rotation mix.
4. **Deform** - expression deltas, presets, visemes and manual sculpts are applied
   in *canonical* space before rotation (the FaceVid2Vid decomposition), so a smile
   stays on the face while the head turns. The mandible is a rigid bone hinged on
   the TMJ axis, hair and neck follow by linear blend skinning with per-ring
   weights, and the frame border is pinned.
5. **Repair** - a stretch limiter relaxes only the edges the solver owns, then the
   disoccluded background is filled from an inpainted plate computed once per
   source image.
6. **Paint** - one triangle pass, painter-sorted far to near, each triangle warped
   by its own affine and clipped with a sub-pixel expansion, then relighting,
   oral composite, eyes and local ROI restoration on top.

### Head turning

The turn is a real perspective projection of a 3D point cloud, not a shear:

- the silhouette narrows *more* than `cos(yaw)` because the near side is magnified;
- the jaw hinges about a 3D TMJ axis, so the chin swings back and down instead of
  sliding in the image plane;
- the far eye fades with measured visibility instead of popping;
- hair rings dome with azimuth, so the shell rotates as a body;
- a same-region edge solver prevents the swinging hair shell from dragging the face
  outline 17-100 px off its own projection;
- the stretch limiter caps elongation. It cannot reach 1.0 - perspective alone
  magnifies the near cheek by 1.5-1.6x - but at the app's yaw limit it cuts the
  worst visible skin stretch from about 3.7x to about 2.0x. The HUD reports both
  numbers.

### Flicker removal

The original single-file page drew the mesh twice per frame (Delaunay plus the
official tessellation) and gated every feature on a hard threshold. Both are gone:

- **one** triangle pass per frame, byte-identical for identical input (asserted in
  `frame.test.js`);
- every binary gate replaced by a hysteresis ramp with asymmetric attack/release,
  which cut threshold-straddling movement by ~6x in `temporal.test.js`;
- One-Euro filtering over all 478 landmarks, with a per-vertex cutoff boost on the
  lips;
- landmark dropout holds the last good samples while a confidence scalar decays and
  cross-fades back, instead of snapping the face to the origin;
- the dental arch is smoothed in its own local frame with angle unwrapping, so teeth
  cannot strobe or spin through the -pi/pi branch;
- eye openness uses a peak-hold tracker (slow rise, fast fall, hard floor);
- every FX stage is individually guarded, so a failure cannot blank the frame.

### Mouth, teeth, tongue

Draw order is cavity -> tongue (inside) -> lower teeth -> gum -> upper teeth ->
ambient occlusion -> specular -> lips -> kiss -> tongue (outside, cross-faded).
Crowns sit at fixed arc-length fractions of the smoothed dental arch (10 per arch,
20 total) with normals taken from the arch tangent, foreshortened by the projected
arch depth, and given translucent incisal edges, interdental gaps, yaw-dependent
specular, corner AO, a saliva strand and a wet lower-lip highlight. The tongue is
viseme-aware. Every amount is a continuous gate; nothing pops.

### Audio

`audio.js` is honest DSP, not a trained lip-sync network: six contiguous bands
(F0 60-280, F1, F2, F2-high, F3, air 6.5-11 kHz), RMS band energies, a normalised
autocorrelation F0 estimate, and rules that map those onto jaw / spread / pucker /
tongue-tip / closure, then onto one of ten named visemes. Microphone or file input;
the mic track can be muxed into the recording.

### Controls

- **Record** - `MediaRecorder` over `captureStream(0)` with `requestFrame()` after
  each render, VP9 -> VP8 -> webm -> mp4 fallback, timestamped filename, live badge.
- **Photo** - full-resolution PNG of the rendered canvas.
- **Sculpt** - click and drag landmarks; smile / frown / O-mouth presets; symmetric
  partners; edits glide rather than snap.
- **Minimize webcam** - the preview collapses, tracking continues at full speed.
- **Show landmarks** - checkbox; iBUG-68 and FACS-AU relabellings are visualisations
  only, listed as such in the technique map.
- **Self-test** - runs the whole pipeline against a synthetic face in-page and
  prints per-stage timings. The same stages run headlessly in
  `src/puppet/selftest.test.js`.

### Technique map

The page carries a 21-row panel stating, for every item in the reference brief
(FOMM, TPS, FaceVid2Vid, MegaPortraits, LivePortrait, FLAME/DECA/EMOCA, Wav2Lip,
SadTalker, VideoReTalking, GeneFace, AniPortrait, EMO, Hallo, VASA-1), whether the
implementation is `real`, an `analogue`, or `not present`. GeneFace/NeRF, EMO,
Hallo and VASA-1 are **not present** - they need trained weights and a GPU.

## Run

```bash
npm install
npm run dev
```

Open the shown URL, choose LivePortrait, and select **Start neural mirror**. The first run downloads and caches roughly 326 MB of ONNX weights. A current Chromium browser with WebGPU and hardware acceleration is required. Camera access requires HTTPS or localhost; the Arena preview provides HTTPS automatically.

## Build

```bash
npm run build
npm run preview
```

## Privacy and safety

Camera frames and selected portraits stay in the browser. There is no backend, upload endpoint, analytics SDK, face recognition, identity swap, or watermarking system. The bundled default portrait is a fictional synthetic identity. Use only images you have permission to animate.

## Architecture

- React + TypeScript + Vite, two build inputs: `index.html` (studio) and
  `puppet.html` (puppet workspace)
- `@mediapipe/face_mesh` with assets under `public/face_mesh` for the React studio;
  `@mediapipe/tasks-vision` from jsDelivr for the puppet page
- One persistent capture/inference loop shared by every workspace
- Hash routes, allowing each demo to open and reload independently
- Preferences persisted in local storage
- Minimizable driver camera that does not pause the inference loop
- `src/puppet/` is plain ES modules with no DOM access outside `main.js`:
  `model.js` (rig), `pose.js` (retarget + limiter), `frame.js` (composite),
  `mouth.js` / `eyes.js` / `plate.js` / `audio.js` / `temporal.js` / `render.js` /
  `capture.js`, with `synthetic.js` providing a DOM-free face, canvas and
  tessellation so all of it is testable in Node

## Tests

```bash
npm test        # 127 tests, 8 files
npm run verify  # tests + tsc + vite build + backend syntax
```

The puppet suites assert the things that are easy to regress and hard to see:
rest-pose projection identity, perspective-vs-shear under yaw, stretch-limiter
effectiveness, jaw hinge direction, single-pass painting, byte-identical output for
identical input, the `warpTri` affine recovered from the recorded `setTransform`,
arc-length tooth placement against a polyline ground truth, gate hysteresis and
One-Euro roughness reduction, plate inpainting convergence, viseme discrimination,
and a full import + `runSelfTest()` run of `main.js` against a stubbed DOM.

## Rendering limits

A single photograph does not contain the hidden side of a head or unseen oral anatomy. The local renderer uses region-preserving triangular deformation and conservative pose motion to avoid flat-card stretching, but it does not claim to reconstruct genuinely unseen geometry. Procedural teeth/tongue rendering is used when the source has no visible mouth interior. Production-grade large-pose novel views require a trained 3D or generative model and substantially more compute.
