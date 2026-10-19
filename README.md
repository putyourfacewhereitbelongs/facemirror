# FaceMirror

FaceMirror's main page is the **2.5D photo puppet**: animate a still portrait
with webcam-driven expressions and head turns, sculpt facial features, then
save a PNG or record a WebM. The FaceMirror-3D experience is still included as
a separate studio at [`studio.html`](studio.html).

- **Your face stays local.** Webcam frames are processed in the browser and are
  never uploaded. The puppet uses its bundled MediaPipe runtime, WASM, and face
  model first, with a pinned CDN fallback if a deployment is missing an asset.
- **Static app.** No backend or build tool is needed to run the committed pages.
- **Works without a webcam.** Choose a portrait and use the sculpting and
  expression tools manually.

## Run the main app

```bash
npm run dev
```

This builds the puppet page from its source and serves the site on
<http://localhost:4173>. Open it in Chrome or Edge, choose a portrait, then
press **Start webcam** and allow camera access. Webcam access requires `http://`
or `https://` (not `file://`).

The main page is generated from [`src/puppet/puppet.src.html`](src/puppet/puppet.src.html)
and the scripts in `src/puppet/`. The generated `index.html` is the default
entry point; `puppet.html` remains as a direct-link alias. Rebuild after
editing the puppet sources with:

```bash
npm run build:puppet
```

For a quick static serve without rebuilding, use `python3 -m http.server 8010`
and open <http://localhost:8010>.

## Main app: 2.5D photo puppet

The 478-point face mesh warps a portrait through head pose and expressions,
with sculpt tools, procedural teeth, PNG capture, and WebM recording.
Head turns use a geometrically derived ear band recessed behind the cheek; the
far ear tucks behind the jaw and fades into an edge-extended plate.

The puppet page loads the vendored MediaPipe Tasks Vision runtime, WASM, face
model, and Delaunay triangulator locally, so webcam expressions and head turns
do not depend on a live CDN. Pinned CDN fallbacks remain for deployments with
missing assets; the optional shoulder/body tracker loads in the background and
cannot block face tracking from starting.

## FaceMirror-3D Studio

Open [`studio.html`](studio.html) for the webcam-driven 3D face avatar and face
editor. It tracks pose, mouth, eyes, blinks, and emotions, and supports manual
sculpting, 52-channel controls, recordings, and GLB export. Its three.js,
MediaPipe runtime, and face-landmarker model are vendored locally under
`vendor/` and `assets/`.

The 3D studio targets **4 GB RAM and CPU-only graphics**: it includes a
procedural head fallback, webcam "Mirror Skin", manual mode when no webcam is
available, and adaptive rendering quality.

## What's inside

| Path | What |
| --- | --- |
| `index.html` | Generated main 2.5D puppet page (default site entry point) |
| `puppet.html` | Generated direct-link alias of the main puppet page |
| `src/puppet/puppet.src.html`, `src/puppet/`, `src/math.js` | Puppet HTML source, engine, renderer, UI, and shared math core |
| `tools/build.mjs` | Builds and verifies both generated puppet pages |
| `studio.html`, `styles.css` | FaceMirror-3D Studio page and its UI styles |
| `src/main.js` | 3D studio bootstrap, fixed 60 Hz solver loop, producer wiring |
| `src/tracker.js` | `getUserMedia` + MediaPipe FaceLandmarker (`numFaces:1`, strictly-increasing timestamps) |
| `src/calibrate.js` | 5-step wizard: 60-frame neutral, gesture ranges, wink L/R map, neutral depth |
| `src/rig.js` | One source of truth: `weights = Float32Array(52)`, producer merge + 🔒 pins |
| `src/channels.js`, `src/smoothing.js` | Landmark revival, One-Euro filters, critically-damped springs, fixed substeps |
| `src/eyes.js`, `src/pose.js` | Blink scheduler, saccades, gaze, head quaternion, inertia, drag |
| `src/emotions.js`, `src/drag.js` | FACS presets/read-back and click-drag editing |
| `src/headAsset.js`, `src/renderer.js` | GLB loader, procedural head, Mirror Skin, three.js renderer |
| `src/recorder.js`, `src/perf.js`, `src/ui.js` | Recording/export, adaptive quality, 3D studio controls |
| `vendor/`, `assets/` | Pinned three.js, MediaPipe Tasks Vision/WASM, and face-landmarker model |
| `tools/` | Puppet build, verification, mock-DOM harness, and static dev server |

### 3D studio controls and exports

Press **C** (or the Help panel button) to calibrate. Hold **SPACE** while each
prompt shows; cancel with Esc.

1. **Neutral** — relax for 60 frames (median baseline + noise floor per channel).
2. **Mouth range** — wide-open mouth, then a big smile.
3. **Eyes/brows** — squeeze eyes shut, then raise brows high.
4. **Cheeks + wink** — puff cheeks, then close your left eye to map MediaPipe's
   L/R naming.
5. **Save** — the profile is stored in `localStorage`.

Keyboard: `C` calibrate · `1..7` emotions · `0` reset · `U` undo · `R` record ·
`M` Mirror Skin · `V` PiP skeleton · `Shift+D` perf HUD · wheel on the head to
dolly. Long-press an emotion chip to solo/lock. Drag face parts to sculpt;
edits pin 🔒 so the camera stops overriding them.

Exports include `.json` clips, morph-animated `.anim.glb` clips, and the
current head as `.glb`. Drop an ARKit-52 head GLB onto the studio canvas (or
into `assets/head.glb`) to use it instead of the procedural fallback.

## Tests

Install the test dependency once, then run the full suite:

```bash
npm install
npm test
```

- `npm run test:3d` — pure-math suites and headless FakeDetector smoke test.
- `npm run test:puppet` — verifies `index.html` and `puppet.html` are current,
  runs 38 math checks, and executes the main page in a mock-DOM harness (56
  behavioral assertions, including ear recess/swing/tuck and far-ear fade).

The 3D suites cover One-Euro lag/jitter, spring stability, blink scheduling,
saccades, emotion classification, Jacobian-solver convergence, calibration,
and GLB export. The smoke harness runs the solve chain with
`window.__FACEMIRROR_FAKE = 'happy|blink|turn|nosecret'` — no webcam or DOM.

## Manual browser checks

**Main puppet:** choose a portrait → enable webcam → nod, turn, smile, and
speak; drag landmarks or use expression presets; save a PNG and record a short
WebM. Try minimizing the webcam preview while tracking continues.

**3D studio:** enable Mirror and test expressions, blinks, and head turns; cover
the camera briefly to check idle behavior; calibrate, drag a mouth corner,
record and export a `.json` / `.anim.glb`, then drop in a head `.glb`.

## 3D studio notes and limits

The 3D renderer adapts quality to CPU/GPU headroom, targeting about 60 fps on
Intel UHD-class hardware and a 4 GB machine. Frames never leave the browser.
Glasses, beards, motion blur, and turns beyond roughly 80° can reduce tracking
quality; photoreal scans require supplying an ARKit-52 head GLB.
