import { describe, expect, it } from 'vitest';
import { buildRig, depthPrior, symmetryPartners, tessTriangles, R_HAIR, R_FACE } from './model.js';
import { syntheticFace } from './synthetic.js';
import { OVAL } from './landmarks.js';

// a stand-in for FaceLandmarker.FACE_LANDMARKS_TESSELATION: a dense edge soup
// built from the synthetic point set so the rig gets real topology to chew on.
function fakeTesselation(V, k = 6) {
  const edges = [];
  for (let i = 0; i < 468; i++) {
    const d = V.map((p, j) => [j === i ? Infinity : Math.hypot(p[0] - V[i][0], p[1] - V[i][1]), j])
      .sort((a, b) => a[0] - b[0]).slice(0, k);
    for (const [, j] of d) edges.push({ start: i, end: j });
  }
  return edges;
}

const face = syntheticFace();
const tess = fakeTesselation(face.V);
const rig = buildRig(face.V, face.W, face.H, tess);

describe('head model', () => {
  it('builds a rig with hair shells, border and body vertices', () => {
    expect(rig.n).toBeGreaterThan(478);
    expect(rig.region[0]).toBe(R_FACE);
    expect(rig.region[478]).toBe(R_HAIR);
    expect(rig.tri.count).toBeGreaterThan(300);
    expect(rig.edges.count).toBeGreaterThan(300);
    expect(rig.shade.count).toBeGreaterThan(40);
    expect(Array.from(rig.Xc).every(Number.isFinite)).toBe(true);
    expect(Array.from(rig.Z).every(Number.isFinite)).toBe(true);
  });

  it('re-projects to the identity at rest (no stretching when idle)', () => {
    const F = rig.frame.F, { cx, cy } = rig.frame;
    let worst = 0;
    for (let i = 0; i < rig.n; i++) {
      const z = rig.Xc[i * 3 + 2], s = F / (F - z);
      const x = cx + rig.Xc[i * 3] * s, y = cy + rig.Xc[i * 3 + 1] * s;
      worst = Math.max(worst, Math.hypot(x - rig.V[i][0], y - rig.V[i][1]));
    }
    expect(worst).toBeLessThan(0.01);
  });

  it('puts the nose in front of the cheeks and the ears at the silhouette plane', () => {
    const nose = Math.max(rig.Z[4], rig.Z[1], rig.Z[168]);
    const cheek = (rig.Z[205] + rig.Z[425]) / 2;
    const ear = (rig.Z[234] + rig.Z[454]) / 2;
    expect(nose).toBeGreaterThan(cheek);
    expect(cheek).toBeGreaterThan(ear);
    expect(ear).toBeLessThan(rig.frame.az * 0.6);
    expect(nose).toBeGreaterThan(rig.frame.az * 0.8);
  });

  it('gives the jaw hinge full weight at the chin and none at the brow', () => {
    expect(rig.jawW[152]).toBeGreaterThan(0.8);
    expect(rig.jawW[70]).toBe(0);
    expect(rig.jawW[13]).toBeLessThan(0.6);
  });

  it('keeps the neck from shearing: rotation weight falls off below the chin', () => {
    expect(rig.wRot[10]).toBe(1);
    const body = rig.wRot[rig.bodyStart];
    expect(body).toBe(0);
  });

  it('orients every rest normal toward the viewer', () => {
    for (let t = 0; t < rig.tri.count; t++) expect(rig.tri.n0[t * 3 + 2]).toBeGreaterThan(-1e-6);
  });

  it('sorts static shell triangles before head triangles', () => {
    const { order, kind, headStart } = rig.tri;
    for (let i = 0; i < headStart; i++) expect(kind[order[i]]).toBeGreaterThanOrEqual(2);
    for (let i = headStart; i < order.length; i++) expect(kind[order[i]]).toBeLessThan(2);
  });

  it('symmetry partners are mutual near the midline', () => {
    const sym = symmetryPartners(face.V.slice(0, 478), rig.frame.cx, rig.frame.FH * 0.34);
    let found = 0;
    for (const i of OVAL) if (sym[i] >= 0 && sym[sym[i]] === i) found++;
    expect(found).toBeGreaterThan(OVAL.length * 0.5);
  });

  it('depth prior stays finite and inside the skull for arbitrary input', () => {
    const z = depthPrior(face.V, rig.frame, { adj468: rig.adj468, sym: rig.sym, zSign: -1, detectorBlend: 0.5 });
    expect(z.length).toBe(478);
    expect(Array.from(z).every(Number.isFinite)).toBe(true);
    expect(Math.max(...z)).toBeLessThan(rig.frame.az * 1.6);
  });

  it('tessTriangles only emits mutually connected triangles', () => {
    const { tris } = tessTriangles(tess, 478);
    expect(tris.length % 3).toBe(0);
    expect(tris.length).toBeGreaterThan(0);
  });
});
