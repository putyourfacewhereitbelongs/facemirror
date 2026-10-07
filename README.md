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

- React + TypeScript + Vite
- `@mediapipe/face_mesh` with assets under `public/face_mesh`
- One persistent capture/inference loop shared by every workspace
- Hash routes, allowing each demo to open and reload independently
- Preferences persisted in local storage
- Minimizable driver camera that does not pause the inference loop

## Rendering limits

A single photograph does not contain the hidden side of a head or unseen oral anatomy. The local renderer uses region-preserving triangular deformation and conservative pose motion to avoid flat-card stretching, but it does not claim to reconstruct genuinely unseen geometry. Procedural teeth/tongue rendering is used when the source has no visible mouth interior. Production-grade large-pose novel views require a trained 3D or generative model and substantially more compute.
