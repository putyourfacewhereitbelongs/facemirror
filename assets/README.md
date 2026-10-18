# assets/

- `face_landmarker.task` — MediaPipe Face Landmarker (float16 v1, 3,758,596 B,
  sha256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`,
  Apache-2.0). Loaded locally first; CDN is only a fallback.
- `anatomical-head/` — bundled generic head scan + skin/normal maps (about
  684 KiB), credited to Lee Perry-Smith / Infinite Realities, CC BY 3.0.
  See its README for the pinned source, checksums, and runtime adaptations.
  Ears are part of the skull mesh and excluded from all facial morphs.
- `face-topology.json` — 898 canonical MediaPipe face triangles, used only
  for landmark-to-image UV correspondence. Derived from the pinned canonical
  OBJ at commit `a908d668c730da128dfa8d9f6bd25d519d006692`, Apache-2.0;
  see `face-topology.LICENSE.txt` and `THIRD_PARTY_NOTICES.md` for provenance.
- `head.glb` — *(optional, user-supplied)* **drop head.glb here** — must contain
  morph targets named per ARKit-52 exactly (`mouthSmileLeft`, not
  `mouth_smile_left`). Common variants (`_l`, `_L`, snake_case) are
  auto-corrected at load with a warning. Missing channels are disabled — no
  crash. Eye-look channels drive eye bones when nodes like `leftEye/rightEye/
  L_eyes` exist. The app runs fully without this file using the bundled
  anatomical head, falling back to procedural geometry if the scan is missing.
  Mirror Skin switches to the procedural webcam head; it does not overlay a
  camera plane on the anatomical/custom head.
- `hdri.ktx2|env.jpg` — *(optional)* small 256–512 px env map. If absent, a
  procedural gradient studio env is generated at boot (see renderer.js).
