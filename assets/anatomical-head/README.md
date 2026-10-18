# Bundled anatomical head

**Infinite, 3D Head Scan by Lee Perry-Smith**, based on work at
[triplegangers.com](https://www.triplegangers.com/), supplied by
[Infinite Realities](http://www.ir-ltd.net/).
Licensed under [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/).
The original license notice is retained in `LICENSE.txt`. Credit is also shown
in the app's Help panel and in `THIRD_PARTY_NOTICES.md`.

## Pinned source

Vendored from [three.js r180, commit
`9e8635e2031c25859dc47ba07e72230dccb2682a`](https://github.com/mrdoob/three.js/tree/9e8635e2031c25859dc47ba07e72230dccb2682a/examples/models/gltf/LeePerrySmith).
The files are unchanged except for their local filenames:

| Local file | Original | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `head.glb` | `LeePerrySmith.glb` | 404976 | `402b8a8ac9f03232e6d64b5962929703a069daf99d3c49ac8eb0e48bedc9c576` |
| `albedo.jpg` | `Map-COL.jpg` | 148269 | `e976d73b31407f8d0967412bf468019ed26a5d5a32cf5811aabff7e816458a65` |
| `normal.jpg` | `Infinite-Level_02_Tangent_SmoothUV.jpg` | 146793 | `36925e51ad9b324b94e8faf4692da1b4132809f2762bb8d5bd549ffd215d4ca6` |

All resources load locally; there is no runtime request to the upstream site.

## FaceMirror adaptations

`src/anatomy.js` normalizes the scan into head-local space, clips/caps the
shoulders, opens the closed-eye surfaces, and adds curved animated lids and
separate eyeballs. It generates 52 approximate facial morphs and their normal
deltas at load time rather than checking in a much larger generated GLB.

The scanned ears remain in the skull mesh, completely outside the facial
morph mask. The common head quaternion provides their rotation; opaque depth
testing handles near/far-ear occlusion. There are no ear landmarks to track,
no camera-facing ear planes, and no view-angle visibility thresholds.

This is a generic neutral scan, not a scan of the webcam user. It has no
oral cavity/teeth; the rig retains its no-teeth jaw cap. A purpose-built ARKit
head can still be supplied as `assets/head.glb` or dropped onto the canvas.
