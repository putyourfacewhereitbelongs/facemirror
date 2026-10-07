#!/usr/bin/env node
/* =====================================================================
   FaceMirror · build
   Inlines src/math.js and src/puppet/*.js into a single self-contained
   page (puppet.html) plus a small studio landing page (index.html).
   No bundler, no dependencies: the output is readable, view-source-able
   and works from file:// as long as the CDN is reachable.

   usage:  node tools/build.mjs [--check]
   ===================================================================== */
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(here, '..');
const args = new Set(process.argv.slice(2));
const checkOnly = args.has('--check');

const SCRIPTS = [
  'src/math.js',
  'src/puppet/engine.js',
  'src/puppet/render.js',
  'src/puppet/ui.js'
];

async function inlineOne(rel) {
  const code = await readFile(join(rootDir, rel), 'utf8');
  if (code.includes('</script')) throw new Error(`${rel} contains a literal </script — it cannot be inlined safely`);
  return `<script data-src="${rel}">\n${code}\n</script>\n`;
}

/* render the two pages in memory — the single source of truth for both
   `build` and `check`, so a stale page can never pass as current */
export async function render() {
  const src = await readFile(join(rootDir, 'src/puppet/puppet.src.html'), 'utf8');
  const blobs = [];
  for (const rel of SCRIPTS) blobs.push(await inlineOne(rel));
  const out = src.replace(/<!--@include[^>]*-->/g, () => blobs.shift());
  if (out.includes('@include')) throw new Error('not every @include marker was replaced');
  return { puppet: out, index: landingPage() };
}
export async function build() {
  const { puppet, index } = await render();
  await writeFile(join(rootDir, 'puppet.html'), puppet, 'utf8');
  await writeFile(join(rootDir, 'index.html'), index, 'utf8');
  return { bytes: Buffer.byteLength(puppet), files: SCRIPTS.length };
}
/* --check: is the committed page byte-identical to what src/ renders?
   returns the list of stale files (empty when everything is current)   */
export async function stale() {
  const { puppet, index } = await render();
  const bad = [];
  for (const [name, want] of [['puppet.html', puppet], ['index.html', index]]) {
    const have = await readFile(join(rootDir, name), 'utf8').catch(() => null);
    if (have !== want) bad.push(name);
  }
  return bad;
}

function landingPage() {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FaceMirror · Trill Face Puppet</title>
<style>
  :root{--bg:#0d0f14;--card:#161a22;--fg:#e8eaf0;--mut:#8b93a7;--acc:#5eead4;--line:#272c38}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
       display:flex;min-height:100vh;align-items:center;justify-content:center;padding:24px}
  main{max-width:760px}
  h1{font-size:30px;margin:0 0 6px}h1 em{color:var(--acc);font-style:normal}
  p{color:var(--mut)}
  a.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0;text-decoration:none;color:var(--fg)}
  a.card:hover{border-color:var(--acc)}
  a.card b{color:var(--acc)}
  ul{color:var(--mut);font-size:14px}
  code{background:#0b0e13;border:1px solid var(--line);border-radius:6px;padding:1px 6px}
</style>
</head>
<body id="back">
<main>
  <h1>Face<em>Mirror</em></h1>
  <p>A still photo turned into a live 2.5D puppet. In-browser MediaPipe Tasks Vision finds the 478-point face mesh;
  a weighted Kabsch fit drives head pose, while a painter-sorted triangle shell warps the source photo. The mouth uses
  a rigid jaw with procedural dental arches and tongue (not a neural portrait generator).</p>
  <a class="card" href="puppet.html">
    <b>Trill Face Puppet →</b><br>
    Webcam-driven photo puppet: perspective head turn with mirrored-source patches approximating the unseen far side,
    drag-to-sculpt landmarks, click-to-sculpt Smile / Frown / Mouth O, procedural teeth and tongue, mic lip-sync, PNG capture and WebM recording.
  </a>
  <ul>
    <li>Start the webcam, then choose a photo (or snapshot yourself) — a chest-up photo also gets shoulder/arm tracking.</li>
    <li>Press <code>Set neutral pose</code> to re-zero your own face at any time.</li>
    <li>Minimize the webcam card to keep tracking while reclaiming screen space.</li>
  </ul>
  <p style="font-size:13px">For webcam access, serve with <code>npm run dev</code> or use HTTPS; opening the file directly may restrict camera APIs.
  The page also needs network access to cdn.jsdelivr.net and storage.googleapis.com for MediaPipe and its models.</p>
</main>
</body>
</html>
`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  if (checkOnly) {
    const bad = await stale();
    if (bad.length) {
      console.error(`check: STALE — ${bad.join(', ')} does not match src/. Run \`npm run build\`.`);
      process.exit(1);
    }
    console.log('check: puppet.html + index.html match src/ exactly');
  } else {
    const r = await build();
    console.log(`built puppet.html (${(r.bytes / 1024).toFixed(1)} kB, ${r.files} modules inlined) + index.html`);
  }
}
