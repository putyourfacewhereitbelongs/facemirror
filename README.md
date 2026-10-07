# FaceMirror · Trill Face Puppet

A still photograph becomes a live puppet. MediaPipe FaceLandmarker finds 478 landmarks in the
photo **and** in your webcam feed; the photo is lifted into a canonical 3D head space, warped by a
painter-sorted triangle shell, and re-rendered every frame — perspective head rotation, rigid jaw
with real teeth and tongue, audio visemes, drag/smile/frown/mouth-O sculpting, PNG capture and
WebM recording.

```
open index.html          # hub page, links into the puppet
open puppet.html         # the app itself (no server needed — everything is inlined)
```

Or serve it (recommended: camera APIs want a proper origin):

```bash
npm run build            # src/ -> puppet.html + index.html
npm run check            # fail if the committed page is stale vs src/
npm run serve            # http://0.0.0.0:4173/
npm run verify           # 37 pure-math checks (geometry, quaternions, filters)
npm run harness          # 39 behavioural checks against a mock DOM + synthetic face
npm test                 # check + verify + harness
```

## Controls

| Control | What it does |
| --- | --- |
| **Photo / Start webcam** | pick the still image, then drive it from the camera |
| **Set neutral** | records the relaxed pose/expression that everything is measured against |
| **Snapshot as image** | downloads the current rendered frame as PNG |
| **Record video** | records the rendered canvas (plus mic audio when enabled) to WebM |
| **Mic lip-sync** | microphone → jaw/visemes, spectral shape drives vowels and rounded sounds |
| **Minimise** (webcam card) | shrinks the camera card; tracking keeps running |
| **Landmarks / Mesh / Original** | overlay the 478 dots, the shell wireframe, or the untouched photo |
| **Sculpt tools** | *Drag landmarks* moves single dots; *Smile*, *Frown*, *Mouth O*, *Pucker*, *Brow raise*, *Sad*, *Wink*, *Puff cheeks* apply a controlled local shape on click |
| **Presets** | 21 canned expressions, from Smile to Tongue out |
| **Sliders** | turn gain, expression gain/presets, depth, focal, shading, hair, gaze, viseme |

Keyboard: `Ctrl+Z` undo, `T` tongue out, double-click a dot to reset it. The **◀ Back to FaceMirror**
link returns to `index.html#back`.

## How the head turn avoids stretching

* **Canonical 3D space.** Each landmark is a 3D point in the photo's own metric
  (`x·W, y·H, −z·W`), so expressions are authored on the face, not on the screen.
* **Rigid pose by orthogonal Procrustes.** The webcam's 478 landmarks are fitted to the tracked
  neutral with weighted Kabsch (SVD with proper-rotation factors, reflection folded into the
  singular values), then decomposed to yaw/pitch/roll with per-axis gain and a soft limit that is
  exactly linear inside ±32° — the old build silently multiplied your head turn by ~0.6.
* **Perspective projection with depth ordering.** Every triangle is drawn far-to-near; the order is
  carried across frames (insertion sort, O(n) while the pose moves smoothly) and re-sorted exactly
  with a bucket sort when a fast turn blows the budget — leftover inversions were the smeared
  triangles.
* **Culling + mirrored far side.** Faces pointing away from the camera are dropped and re-sampled
  from the mirrored side of the face, so the far cheek wraps instead of stretching to the silhouette.
* **Disocclusion fill.** Areas the photo never saw are filled from the pre-blurred skin plate, so a
  hard turn cannot leave a hole.
* **Depth shading** keyed to yaw keeps the turned side readable.

## Why it does not flash

Every state change is a rate-limited envelope, never a switch: one-euro filters per landmark
coordinate with a faster cutoff for the lips, an attack/release envelope on tracker trust (detection
loss decays the pose toward neutral instead of freezing), slew limits on the finished expression
delta, all feature passes gated by smooth `[0,1]` weights, and full-bleed fills clipped to computed
bounding boxes. The mouth interior is the case in point: the oral cavity ramps in as an envelope,
so the lip gap cannot cross a threshold and pop.

## Mouth, teeth and tongue

A rigid mandible hinge (TMJ axis from the ear landmarks) rotates the jaw bone with linearly-blended
skinning weights; upper and lower dental arches are separate shells with their own enamel gradient,
specular banding and gum line, the tongue is a procedural surface with a midline groove and
papillae, and the cavity behind them is shaded by an envelope-driven gradient. The jaw angle is
derived from the lip gap and the jawOpen blendshape, so it never opens further than the face
actually does.

## Coverage of the reenactment literature

The classical half of the pipeline is implemented in full and runs at 60 fps in one HTML file:
landmark detection (MediaPipe), rigid pose fitting, expression transfer, local warping with
occlusion-aware depth ordering, disocclusion inpainting, Poisson-style blending for the composite,
parametric oral cavity with learned tooth proportions, an amplitude+spectrum → viseme front end,
and a full temporal-filter chain against flicker.

The neural half (FOMM / FaceVid2Vid / MegaPortraits / LivePortrait, and the diffusion frontier —
EMO, Hallo, VASA-1) needs a GPU and model weights that cannot ship inside this single page. The
app implements their *observable behaviour* with the classical budget: the same
warping + occlusion + relighting stack, driven by the audio/viseme front end instead of a learned
renderer. Swapping in a neural renderer means replacing `sessionFrame` with a model call — the
tracking, sculpting, recording and UI layers stay as they are.

## Verification

`npm run verify` checks the math the whole thing rests on (quaternion/euler round-trips, SVD,
Kabsch at four scales plus 400 adversarial point sets, one-euro/slew/schmitt/envelope, spline and
polygon guards). `npm run harness` boots the real modules in Node against a mock DOM and a
synthetic 478-point face, then asserts: pose recovery within 3°, a held pose that is pixel-perfect
still, a bounded fast-sweep step, smooth dropout decay and re-acquire, zero NaN over 240 randomised
frames, no flash-class canvas op, far-to-near paint order over 1182 triangles, the oral cavity
appearing only with an open mouth and always clipped inside the lips, sculpt offsets that do not
drift while tracking, and a drag that lands on the pointer within 3.5 px even with the head turned.
