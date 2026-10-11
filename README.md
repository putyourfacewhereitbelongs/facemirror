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
# FaceMirror · Trill Face Puppet

A browser-based **2.5D** portrait puppet: MediaPipe FaceLandmarker finds 478 landmarks in a still
photo and webcam frame, then a perspective-projected facial shell transfers head pose and expression
onto the photo. It includes a rigid jaw, procedural dental arches and tongue, click-to-sculpt
expressions, PNG capture and WebM recording.

This is a classical mesh-warp renderer, **not** a diffusion/video-generation model. A single still
cannot reveal the true far side of a head; mirrored source patches and a blurred edge-extended plate
approximate those newly visible areas. The supported yaw is deliberately limited to about 46°.

```bash
npm run dev       # build and serve on 0.0.0.0:4173 (camera works on a secure/local origin)
npm run check     # fail if puppet.html / index.html are stale vs src/
npm test          # check + 38 math checks + 46 behavioural checks
```

`puppet.html` is the distributable page with the local modules inlined. The page still needs internet
access for MediaPipe Tasks Vision and its model files. For webcam access, use `npm run dev` or another
HTTPS/localhost server rather than opening the file from disk.

## Controls

| Control | What it does |
| --- | --- |
| **Choose image** | load a portrait; a webcam snapshot can also become the source image |
| **Use webcam snapshot** | capture the mirrored webcam frame and use it as the puppet's source portrait |
| **Start webcam / Set neutral** | track the face and record the pose/expression to neutralize against |
| **Save rendered photo** | download the current rendered puppet frame as PNG |
| **Record video** | record the rendered canvas as WebM; mic audio is included when mic lip-sync is on |
| **Mic lip-sync** | microphone spectrum drives smoothed jaw/viseme actions (a heuristic, not phoneme recognition) |
| **Minimize** (webcam card) | hide the webcam panel while leaving tracking active |
| **Landmarks / Mesh / Original** | toggle landmark dots, shell wireframe, or the untouched source image |
| **Sculpt tools** | choose Drag, Smile, Frown, Mouth O, Pucker, Brow raise, Sad, Wink, Puff cheeks, or Clear; expression tools only act on a click inside the face |
| **Presets** | 21 expression presets, including Talking and Tongue out |
| **Settings** | tune pose gain, smoothing, depth, mouth limits, teeth visibility/brightness/warmth, gaze and hair |

Keyboard: `Ctrl+Z` undo, `T` tongue out, double-click a landmark to reset it. The **◀ Back to
FaceMirror · Continue ▶** link returns to `index.html#back`.

## Head-turn and anti-flicker pipeline

* **Canonical facial space.** Landmark coordinates accept MediaPipe `{x,y,z}` objects and tuple
  points. Depth blends detector z with an analytic head prior and a smoothed tessellation.
* **Rigid pose / expression split.** A weighted Kabsch fit estimates yaw, pitch, roll, scale and
  translation from relatively rigid landmarks. Expression residuals are retargeted separately so a
  smile does not become head motion. Pose is projected with weak perspective and a soft yaw cap.
* **Triangle shell.** The source photo is mapped over a Delaunay frame/background shell plus the
  detector's facial tessellation. Painter order is carried between frames with insertion sort; if a
  turn exceeds the work budget, an allocation-free stable merge sort restores the complete order
  by each triangle's mean-depth key. The harness checks both passes for depth-key inversions.
* **Far-side approximation.** Back-facing triangles sample the mirrored side of the source rather
  than being dropped or stretched to the silhouette. An edge-extended blurred plate covers small
  disocclusions. This reduces obvious collapse, but it does not invent identity-specific details
  that the photograph never captured.
* **Temporal continuity.** Per-coordinate one-euro landmark filters, bounded pose slew, trust
  envelopes on tracking loss, smoothed expression actions and mouth/teeth visibility envelopes
  avoid hard state switches. Polygon guards accept both tuple and `{x,y}` point formats and reject
  non-finite/degenerate clips before drawing.

These controls address common causes of popping and paint-order smearing. The test suite is a
headless mock-DOM test, not a substitute for visually checking a real webcam/photo in a browser.

## Mouth, teeth and tongue

The mandible rotates around an estimated TMJ axis with different skinning weights for upper and
lower lip landmarks. Jaw opening can come from the webcam `jawOpen` blendshape, a preset, or the
microphone viseme driver. Upper/lower dental arches are separate parametric surfaces with varied
crown proportions, a generated enamel texture, shading and interproximal lines. The tongue is a
procedural, clipped surface with a groove, papillae and wet highlights; the oral cavity and lip
passes are clipped to the detected inner/outer lip contours and fade with the mouth-opening envelope.

These are procedural approximations. They are not a fitted FLAME mouth, identity-trained tooth
prior, learned mouth inpainting network, or physically based translucent dental renderer. Quality
will depend on the photo, its mouth visibility and the fit of the 478-point detector.

## Coverage of the supplied research topics

The table distinguishes implemented browser techniques from research systems that are **not**
shipped as model weights. Similar-looking behavior is not the same as implementing the cited model.

| Topic from the write-up | What this page implements | What is not implemented |
| --- | --- | --- |
| dlib 68-point, MediaPipe Face Mesh, OpenFace AUs, FAN | MediaPipe FaceLandmarker 478-point mesh and its blendshape scores; MediaPipe PoseLandmarker is optional for shoulders/arms | dlib/OpenFace/FAN detectors or interchangeable detector adapters |
| FOMM, TPS, FaceVid2Vid, MegaPortraits, LivePortrait | 3D pose/expression separation, canonical landmark transfer, triangulated affine warping, depth ordering and a mirrored far-side source patch | The learned keypoint/motion fields, TPS generator, neural free-view synthesis, high-resolution portrait generators or LivePortrait model weights |
| FLAME / EMOCA / DECA and oral geometry | Analytic depth prior, a rigid jaw hinge, explicit upper/lower parametric dental arches and procedural tongue | FLAME fitting, EMOCA/DECA coefficients, person-specific scanned teeth or a trained oral-cavity prior |
| Wav2Lip, SadTalker, VideoReTalking, GeneFace/GeneFace++, AniPortrait, MuseTalk | Mic amplitude/spectral bands plus MediaPipe jaw/blendshape signals feed smoothed jaw, pucker, spread and tongue actions | Phoneme/forced alignment, learned audio-to-expression models, lip-sync discriminators, NeRF, learned lower-face generation or super-resolution |
| Teeth priors, mouth inpainting, compositing and enhancement | Procedural enamel/crown variation, clipped oral layers, gradients and local canvas compositing | Learned tooth identity priors, conditional neural inpainting, GFPGAN/CodeFormer/RestoreFormer, or Poisson/learned blending |
| Temporal flicker countermeasures | One-euro filters, pose/expression slew, envelope gating, carried ordering and exact merge-sort fallback | Temporal discriminator or optical-flow network |
| EMO, Hallo/Hallo2, AniPortrait diffusion, VASA-1 | No diffusion model; the page exposes a replaceable render stage at `FMRender.sessionFrame` | These models, their checkpoints, GPU inference, and their temporal-attention/video-generation pipelines |

The cited neural systems need their model architecture, trained weights and a supported inference
runtime (often a GPU). They cannot be truthfully represented by this single-page classical renderer.
A production neural version would connect a real inference service/model at the render stage while
keeping the current tracking, sculpting, capture and UI around it.

## Verification and limits

`npm test` verifies the generated page is current, runs 38 pure-math checks, and exercises the app
modules against a mock DOM and synthetic 478-point face (48 behavioural assertions). The harness
checks pose recovery, held-pose stability, bounded sweep/dropout, zero NaNs and flash-class clips,
complete depth-key permutations at startup and a known yaw, forced merge-sort recovery from adversarial
prior order, full shell coverage via mirrored samples, mouth/cavity/teeth/tongue passes, jawOpen
blendshape routing, mirrored webcam snapshots, click hit-testing, PNG capture,
recording lifecycle/track cleanup and the landmark/minimize controls. It also reports tracked
clip/fill/image calls per rendered frame; this is a canvas-command proxy, **not an FPS benchmark**.

No browser/GPU visual pass is available in the headless test environment. Before treating the hard
visual requirement as satisfied for a particular portrait, open the live page with a real camera and
photo, inspect both yaw directions and mouth shapes, and report any remaining artifact. Large turns
or a source with a hidden far cheek still require a true 3D/neural portrait model for reliable
photorealism.
