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
