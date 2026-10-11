#!/usr/bin/env node
/* Static dev server. Binds 0.0.0.0 (never 127.0.0.1) so the sandbox live
   preview and phones on the LAN can both reach it.
   usage: node tools/serve.mjs [port]                       */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, normalize, resolve } from 'node:path';

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.argv[2] || process.env.PORT || 4173);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.webp': 'image/webp', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.task': 'application/octet-stream',
  '.data': 'application/octet-stream', '.binarypb': 'application/octet-stream', '.md': 'text/markdown; charset=utf-8'
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let path = decodeURIComponent(url.pathname);
    if (path.endsWith('/')) path += 'index.html';
    const file = join(rootDir, normalize(path).replace(/^([/\\])+/, ''));
    if (!file.startsWith(rootDir)) { res.writeHead(403).end('forbidden'); return; }
    const s = await stat(file).catch(() => null);
    if (!s || !s.isFile()) { res.writeHead(404, { 'content-type': 'text/plain' }).end('not found: ' + path); return; }
    const body = await readFile(file);
    const ext = file.slice(file.lastIndexOf('.'));
    res.writeHead(200, {
      'content-type': TYPES[ext] || 'application/octet-stream',
      'cache-control': 'no-cache',
      'content-length': body.length
    });
    res.end(body);
  } catch (e) {
    res.writeHead(500, { 'content-type': 'text/plain' }).end(String(e && e.message));
  }
});
server.listen(port, '0.0.0.0', () => {
  console.log(`FaceMirror dev server on http://0.0.0.0:${port}/  (puppet.html, index.html)`);
});
