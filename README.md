# FaceMirror-3D

A realistic, webcam-driven 3D face avatar that mirrors your face — pose, mouth,
eyes, blinks, emotions — and can also be sculpted by click-dragging its features
and picking emotions. Built for a machine with **4 GB RAM and CPU-only graphics**:
motion reads as *alive* (real blinks, saccades, skin, head inertia), and nothing
ever depends on a download beyond the app's own pinned assets.

- **100% local.** Face images and video frames never leave the page. Usage
  telemetry is disabled; photos and fitted geometry are not persisted.
- **Zero install.** Static site, vanilla ES modules, no build step.
- **Anatomical by default.** Locally bundled scan-based skin, 3D ears attached
  to the skull, separate eyes, and animated lids. No webcam → full manual mode.
- **Image-driven likeness.** Choose or drop the active portrait to fit facial
  proportions, skin texture and iris appearance onto the animated head.
- **Graceful everywhere.** A custom `head.glb` overrides the built-in head.
  Missing bundled scan → procedural head with closed 3D ears + optional webcam
  "Mirror Skin". Adaptive quality reduces work on slower machines.

## Run it

```bash
cd facemirror/
python3 -m http.server 8010
# or: npx serve
```

Open <http://localhost:8010> in Chrome or Edge. Press **Mirror (webcam) ⏻** and
allow the camera. (Webcam needs http/localhost — not `file://`.)

First load is fully self-contained: three.js, MediaPipe tasks-vision wasm, and
the 3.7 MB face-landmarker model are vendored under `vendor/` + `assets/`.
The anatomical mesh and its skin maps add about 684 KiB, all local too.
After that it runs offline (settings live in `localStorage`).

Drag away from a facial control to turn the head: the near ear foreshortens
and the skull naturally hides the far ear. **Controls** collapses the panel;
it starts collapsed on narrow screens so the complete portrait stays visible.
**Mirror Skin (M)** switches to the webcam-textured procedural head and back
to anatomical skin; it never pastes a flat camera patch over the scan's ears.
A custom GLB keeps its authored skin while still following webcam motion.

## Likeness from the current face image

Use **Face image…** or drop a JPEG, PNG or WebP onto the head (up to 12 MB).
Choose one clear, front-facing face, with the forehead, chin and both eyes
visible. The reference thumbnail shows the image currently driving likeness.
Images without a usable face, multiple faces, clipped faces or strong profile
turns are rejected without changing the last successfully fitted person.

The local IMAGE detector calibrates against a neutral rendering of the bundled
head, fits bounded facial proportions, and bakes the portrait into the existing
UV atlas. Rest geometry, expression morphs, eyes and lids use the same fit;
ears remain rigid. Replacements start from the original template rather than
accumulating the previous person's shape. Head export embeds the fitted atlas.

Other in-page renderers can supply their active image/canvas through the same
pipeline, with an immediate snapshot so later redraws cannot change the source:

```js
window.dispatchEvent(new CustomEvent('facemirror:face-image', {
  detail: { image: currentFaceCanvas, label: 'Current rendered face' },
}));
```

Webcam motion does not overwrite a chosen image identity. **Use webcam face**
requests a new, relaxed frontal snapshot instead. **Clear likeness** forgets
the fitted person; normal expression reset keeps it. Images, landmarks and
photo-derived Jacobians are session-only; reloading returns to the generic
preview. Only an explicit head export saves the image-derived appearance.

This is approximate single-view fitting, not a recovered 3D scan. Unseen
profile, ears and full hair geometry cannot be determined from one portrait;
source lighting/expression can remain in the texture. Use a custom head GLB
for a production identity/expression rig.

## What's inside

| Path | What |
| --- | --- |
| `index.html`, `styles.css` | UI shell: dark studio, one collapsible right panel, full-bleed canvas |
| `src/main.js` | bootstrap, fixed 60 Hz solver loop, producer wiring |
| `src/tracker.js` | `getUserMedia` + MediaPipe FaceLandmarker (`numFaces:1`, strictly-increasing timestamps) |
| `src/calibrate.js` | 5-step wizard: 60-frame neutral, gesture ranges, wink L/R map, neutral depth |
| `src/rig.js` | ONE source of truth: `weights = Float32Array(52)`, explicit producer merge + 🔒 pins |
| `src/channels.js` | dead-channel revival from 478 landmarks in canonical (pose-removed) space |
| `src/smoothing.js` | One-Euro + critically-damped springs, 60 Hz fixed substeps, hysteresis |
| `src/eyes.js` | Poisson blink scheduler, saccades, microsaccades, Listing's law, VOR clamp-and-reset, stabilizeBlink |
| `src/pose.js` | head quaternion from the transformation matrix, clamps, inertia, rubber-band drag |
| `src/emotions.js` | 7 FACS preset vectors + cosine-similarity read-back ("Emotion read-back (local)") |
| `src/drag.js` | click-drag editing via per-control morph Jacobian solve, undo stack |
| `src/headAsset.js` | local anatomical asset + GLB loader (ARKit-52 name wiring) + procedural head + Mirror Skin |
| `src/anatomy.js`, `src/ears.js`, `src/headShape.js` | scan adaptation, rigid ear masking, closed fallback auricles, shared facial morphs |
| `src/faceImage.js`, `src/likeness.js`, `src/portraitTexture.js` | local image ingestion, latest-source-wins fitting, morph rebasing and exportable photo atlas |
| `src/renderer.js` | three.js scene, studio lighting, cheap-SSS skin patch, tone mapping |
| `src/recorder.js` | 52-ch 30 fps ring buffer, playback, `.json` / `.anim.glb` / head `.glb` export |
| `src/perf.js` | adaptive quality governor + `Shift+D` HUD |
| `src/ui.js` | chips, 52 sliders with 🔒 + gain drags, wizard, PiP preview |
| `vendor/`, `assets/` | pinned three.js 0.186.1, tasks-vision 1.1.0 + wasm, face_landmarker.task |

Every producer (camera, drag, emotion, record-playback, manual sliders, idle)
writes its own layer; `rig.update` merges them explicitly:

```
w_final = pin ? manual : clamp01(max(cam · camW, emotion · intensity) + dragAdd)
```

## Calibrate (2 minutes, makes the app feel right)

Press **C** (or the Help panel button). Hold **SPACE** (or the hold button)
while each prompt shows; cancel any time with Esc.

1. **Neutral** — relax, 60 frames (median baseline + noise floor per channel).
2. **Mouth range** — wide-open mouth, then a big smile (20 frames each).
3. **Eyes/brows** — squeeze eyes shut, then raise brows high.
4. **Cheeks + wink** — puff cheeks (sealed lips), then close your **left** eye
   only (this maps MediaPipe's L/R naming — bilateral pairs are remapped at
   load, never by flipping sliders).
5. **Save** — profile stored in `localStorage`; re-run any time.

The wizard's live 52-bar view doubles as the debug display.

## Keyboard

`C` calibrate · `1..7` emotions · `0` reset all · `U` undo drag edit ·
`R` record · `M` Mirror Skin · `V` PiP skeleton · `Shift+D` perf HUD ·
`Esc` release drag / close wizard · wheel on head = dolly.

Long-press an emotion chip = solo/lock. Drag face parts to sculpt (mouth
corners, lips, brows, chin, nose, lids, jaw sides, cheeks — 16 control
points); edits pin 🔒 so the camera stops overriding them.

## Export

- **.json** — 52 channels × 30 fps clip (portable, up to 300 s).
- **.anim.glb** — morph-animated glTF clip (52 weights animated on a head
  proxy; opens in any glTF viewer).
- **head .glb** — the current head with morphs baked in (a usable 3D image to
  keep). Drop any ARKit-52 head onto the canvas (or into `assets/head.glb`)
  any time; name variants like `mouth_smile_left` are auto-corrected with a
  warning, missing channels disable instead of crashing.

## Tests

```bash
node --test src/tests/     # pure-math suites + headless FakeDetector smoke
```

114 tests across image-source validation, pose-normalized identity fitting,
texture baking, source replacement/cancellation, ear rigidity, anatomical
geometry and near/far-ear occlusion, One-Euro
lag/jitter, spring stability, blink-scheduler statistics (2-minute sim), saccade main sequence, emotion classification
(contempt ≠ joy), symmetrize, Jacobian-solver convergence, stabilizeBlink,
dead-channel anti-gating (head rotation must not fake a jaw shift), calibration
math, and GLB export parse-back. The smoke harness runs the full solve chain
with `window.__FACEMIRROR_FAKE = 'happy|blink|turn|nosecret'` — no webcam, no
DOM.

## Manual checklist (2 minutes, in the browser)

1. **Mirror on** → talk, nod, smile: avatar follows, no jitter, no frozen lids.
2. **Look 45° away**: tracking holds (up to ~80°).
3. **Cover the camera 3 s**: settles into breathing idle + occasional blinks —
   not a frozen stare. Uncover: snaps back to live.
4. **Blink at profile**: never a one-eyed blink (stabilizeBlink).
5. **Drag a mouth corner**: smile tracks the cursor, 🔒 pins appear on the
   edited channels, `U` undoes, `0` resets everything.
6. **Emotion chips** (keys 1–7): ~0.8 s spring in with slight overshoot; your
   mouth still talks through the preset; long-press = lock; read-back row shows
   "You: joy 62%".
7. **Shift+D**: HUD shows fps, per-stage ms, heap, tier; abuse the CPU and
   watch the tier toast down (and back up after 5 s of headroom).
8. **R** to record a few seconds → ▶ plays back as the camera producer →
   export `.json` and `.anim.glb`; open the `.anim.glb` at
   <https://gltf-viewer.donmccurdy.com> (or Blender) — morphs + clip intact.
9. Drop a head `.glb` on the canvas: warnings list any non-ARKit morph names;
   all 52 sliders still drive the head.
10. **Turn ±70–85°**, using the webcam or holding a whole-head drag: ears stay
    attached, the far ear disappears behind the skull, and the near ear shows
    its bowl/rim rather than a flat oval. Smile, puff cheeks, and blink while
    turned: neither ear stretches. Check both anatomical and Mirror Skin modes.

## Performance notes (4 GB target)

- Whole tab ≤ ~600 MB; detection ≤ 30 ms/frame at 640×480; render ~60 fps on
  Intel UHD-class GPUs. The governor (p95 of 120 frame times) steps quality
  1.25 → 1.0 → 0.85 render scale → every-2nd-frame detection → no detail
  normal → 30 fps cap, and back up when there's headroom.
- The anatomical head is about 20k triangles. Position/normal morph targets
  are generated once at load; no ear repositioning, per-frame remeshing, or
  extra rendering passes are needed.
- All hot arrays are preallocated — no per-frame allocations in the tracker or
  solver loops (GC pauses read as face stutter on 4 GB machines).
- Startup prints a self-check report to the console; the HUD's heap readout
  warns above 700 MB.

## Known honesty limits

1. **>80° turns lose tracking** — that is the model's spec, not a bug in this
   app; the face-landmarker simply has no face to track past there.
2. **Glasses, beards, and motion blur soften expressions** — landmark and
   blendshape quality degrade with heavy occlusion and blur.
3. **Without a face source, the preview is a generic neutral scan.** Image
   fitting derives approximate likeness, not unseen anatomy or full hair.
   Runtime-generated expressions are approximate, with no modeled teeth/oral
   cavity (the no-teeth jaw cap applies). Supply an ARKit-52 `assets/head.glb`
   for a production rig. The template scan is credited to Lee Perry-Smith /
   Infinite Realities under CC BY 3.0; see `assets/anatomical-head/README.md`.
