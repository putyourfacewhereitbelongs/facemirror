# assets/

- `face_landmarker.task` — MediaPipe Face Landmarker (float16 v1, 3,758,596 B,
  sha256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`,
  Apache-2.0). Loaded locally first; CDN is only a fallback.
- `head.glb` — *(optional, user-supplied)* **drop head.glb here** — must contain
  morph targets named per ARKit-52 exactly (`mouthSmileLeft`, not
  `mouth_smile_left`). Common variants (`_l`, `_L`, snake_case) are
  auto-corrected at load with a warning. Missing channels are disabled — no
  crash. Eye-look channels drive eye bones when nodes like `leftEye/rightEye/
  L_eyes` exist. The app runs fully without this file using the procedural
  fallback head + webcam "Mirror Skin".
- `hdri.ktx2|env.jpg` — *(optional)* small 256–512 px env map. If absent, a
  procedural gradient studio env is generated at boot (see renderer.js).
