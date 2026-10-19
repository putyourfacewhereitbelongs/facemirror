#!/usr/bin/env node
/* =====================================================================
   FaceMirror · legacy puppet build
   Inlines src/math.js and src/puppet/*.js into a single self-contained
   page (puppet.html).  The repository root index.html belongs to
   FaceMirror-3D and is deliberately not touched by this build.
   No bundler: the output is readable, view-source-able and works from
   file:// as long as the CDN is reachable.

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

/* Render the legacy puppet page in memory — the single source of truth
   for both `build` and `check`, so a stale page can never pass current. */
export async function render() {
  const src = await readFile(join(rootDir, 'src/puppet/puppet.src.html'), 'utf8');
  const blobs = [];
  for (const rel of SCRIPTS) blobs.push(await inlineOne(rel));
  const out = src.replace(/<!--@include[^>]*-->/g, () => blobs.shift());
  if (out.includes('@include')) throw new Error('not every @include marker was replaced');
  return { puppet: out };
}
export async function build() {
  const { puppet } = await render();
  await writeFile(join(rootDir, 'puppet.html'), puppet, 'utf8');
  return { bytes: Buffer.byteLength(puppet), files: SCRIPTS.length };
}
/* --check: is the committed puppet page byte-identical to what the
   legacy sources render?  Returns the list of stale files.            */
export async function stale() {
  const { puppet } = await render();
  const have = await readFile(join(rootDir, 'puppet.html'), 'utf8').catch(() => null);
  return have === puppet ? [] : ['puppet.html'];
}

if (import.meta.url === `file://${process.argv[1]}`) {
  if (checkOnly) {
    const bad = await stale();
    if (bad.length) {
      console.error(`check: STALE — ${bad.join(', ')} does not match src/. Run \`npm run build:puppet\`.`);
      process.exit(1);
    }
    console.log('check: puppet.html matches the legacy puppet sources exactly');
  } else {
    const r = await build();
    console.log(`built puppet.html (${(r.bytes / 1024).toFixed(1)} kB, ${r.files} modules inlined)`);
  }
}
