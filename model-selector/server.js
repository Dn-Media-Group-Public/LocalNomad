'use strict';
const http  = require('http');
const fs    = require('fs');
const path  = require('path');
const { execSync, spawn } = require('child_process');
const os    = require('os');

const OPENCODE_CONFIG = path.join(os.homedir(), '.config', 'opencode', 'opencode.json');
const PROXY_PORT  = 11435;   // OpenCode talks here
const UI_PORT     = 4000;    // CodeNomad SideCar status page
const OLLAMA_HOST = '127.0.0.1';
const OLLAMA_PORT = 11434;

// ── Model catalogue ───────────────────────────────────────────────────────────
const MODELS = [
  { id: 'qwen2.5-coder:1.5b',    name: 'Qwen2.5-Coder 1.5B',     ramGB:  2, desc: 'Tiny & fast' },
  { id: 'qwen2.5-coder:3b',      name: 'Qwen2.5-Coder 3B',        ramGB:  3, desc: 'Good for simple tasks' },
  { id: 'llama3.2:3b',           name: 'Llama 3.2 3B',            ramGB:  3, desc: 'Small general-purpose' },
  { id: 'codellama:7b',          name: 'CodeLlama 7B',            ramGB:  6, desc: 'Meta code model' },
  { id: 'mistral:7b',            name: 'Mistral 7B',              ramGB:  6, desc: 'General purpose' },
  { id: 'qwen2.5-coder:7b',      name: 'Qwen2.5-Coder 7B',        ramGB:  6, desc: 'Recommended starting point' },
  { id: 'llama3.1:8b',           name: 'Llama 3.1 8B',            ramGB:  7, desc: 'Strong general model' },
  { id: 'codellama:13b',         name: 'CodeLlama 13B',           ramGB: 10, desc: 'Larger code model' },
  { id: 'qwen2.5-coder:14b',     name: 'Qwen2.5-Coder 14B',       ramGB: 11, desc: 'Very capable coder' },
  { id: 'phi4:14b',              name: 'Phi-4 14B',               ramGB: 11, desc: 'Strong reasoning' },
  { id: 'deepseek-coder-v2:16b', name: 'DeepSeek-Coder V2 16B',   ramGB: 13, desc: 'Excellent at code' },
  { id: 'qwen2.5-coder:32b',     name: 'Qwen2.5-Coder 32B',       ramGB: 22, desc: 'Best local coder' },
  { id: 'llama3.1:70b',          name: 'Llama 3.1 70B',           ramGB: 48, desc: 'Needs ≥64 GB' },
];

// ── Server-side state ─────────────────────────────────────────────────────────
// downloads: modelId → { status: 'downloading'|'done'|'error', pct, line, error }
const downloads = new Map();
const sseClients = new Set();

function broadcast(obj) {
  const msg = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of sseClients) {
    try { res.write(msg); } catch (_) { sseClients.delete(res); }
  }
}

// ── System helpers ────────────────────────────────────────────────────────────
function memGB(field) {
  try {
    const m = fs.readFileSync('/proc/meminfo', 'utf8')
      .match(new RegExp(field + ':\\s+(\\d+)\\s+kB'));
    if (m) return parseInt(m[1]) / 1048576;
  } catch (_) {}
  return 8;
}

// Models on disk
function downloadedModels() {
  try {
    return execSync('ollama list 2>/dev/null', { encoding: 'utf8' })
      .split('\n').slice(1).map(l => l.split(/\s+/)[0]).filter(Boolean);
  } catch (_) { return []; }
}

// Models currently loaded in RAM
function runningModels() {
  try {
    return execSync('ollama ps 2>/dev/null', { encoding: 'utf8' })
      .split('\n').slice(1).map(l => l.split(/\s+/)[0]).filter(Boolean);
  } catch (_) { return []; }
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', d => chunks.push(d));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

// ── Pull manager ──────────────────────────────────────────────────────────────
// Returns a Promise that resolves when the model is on disk.
// If a pull is already in progress for that model, waits for it rather than
// spawning a second process.
function ensureDownloaded(modelId) {
  const existing = downloads.get(modelId);
  if (existing) {
    if (existing.status === 'done')        return Promise.resolve();
    if (existing.status === 'downloading') return existing._promise;
  }

  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });

  const dl = { status: 'downloading', pct: 0, line: '', error: null, _promise: promise };
  downloads.set(modelId, dl);
  broadcast({ type: 'start', model: modelId });

  const child = spawn('ollama', ['pull', modelId]);

  function handleChunk(data) {
    const text = data.toString();
    const pctMatch = text.match(/(\d+)%/);
    if (pctMatch) {
      const newPct = parseInt(pctMatch[1]);
      dl.pct = newPct;
    }
    const cleaned = text.replace(/\r/g, '').split('\n').filter(s => s.trim()).pop() || '';
    if (cleaned) dl.line = cleaned;
    broadcast({ type: 'progress', model: modelId, pct: dl.pct, line: dl.line });
  }

  child.stdout.on('data', handleChunk);
  child.stderr.on('data', handleChunk);
  child.on('close', code => {
    if (code === 0) {
      dl.status = 'done'; dl.pct = 100;
      broadcast({ type: 'done', model: modelId });
      resolve();
    } else {
      dl.status = 'error';
      dl.error = `ollama pull exited with code ${code}`;
      broadcast({ type: 'error', model: modelId, error: dl.error });
      reject(new Error(dl.error));
    }
  });

  return promise;
}

// ── Proxy helpers ─────────────────────────────────────────────────────────────
function makeOllamaOptions(req, bodyLen) {
  const headers = Object.assign({}, req.headers, {
    host: `${OLLAMA_HOST}:${OLLAMA_PORT}`,
  });
  if (bodyLen !== undefined) headers['content-length'] = bodyLen;
  delete headers['transfer-encoding'];
  return { hostname: OLLAMA_HOST, port: OLLAMA_PORT,
           path: req.url, method: req.method, headers };
}

// Forward a request to ollama, piping the response back.
// bodyBuffer must be a Buffer (pass empty Buffer for requests with no body).
function forwardToOllama(req, res, bodyBuffer) {
  const opts = makeOllamaOptions(req, bodyBuffer.length);
  const proxy = http.request(opts, proxyRes => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  proxy.on('error', e => {
    if (!res.headersSent) res.writeHead(502);
    res.end(`Ollama proxy error: ${e.message}`);
  });
  proxy.write(bodyBuffer);
  proxy.end();
}

// Send one OpenAI-format SSE content chunk (headers must already be sent).
function sendSSEChunk(res, content) {
  const payload = JSON.stringify({
    id: 'chatcmpl-locallm',
    object: 'chat.completion.chunk',
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });
  res.write(`data: ${payload}\n\n`);
}

// Forward an already-started streaming response to ollama (headers already sent).
function pipeOllamaStream(req, res, bodyBuffer) {
  const opts = makeOllamaOptions(req, bodyBuffer.length);
  const proxy = http.request(opts, proxyRes => {
    // Don't re-send headers — just pipe the body (SSE chunks from ollama)
    proxyRes.pipe(res);
  });
  proxy.on('error', e => {
    sendSSEChunk(res, `\n❌ Ollama error: ${e.message}`);
    res.write('data: [DONE]\n\n');
    res.end();
  });
  proxy.write(bodyBuffer);
  proxy.end();
}

// ── Proxy server (port 11435) ─────────────────────────────────────────────────
const proxyServer = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];

  // /v1/models — return all RAM-fitting models so OpenCode's /models shows them all,
  // not just the subset ollama currently has on disk.
  if (req.method === 'GET' && url === '/v1/models') {
    const total   = memGB('MemTotal');
    const fitting = MODELS.filter(m => m.ramGB <= total * 0.85);
    const data = fitting.map(m => ({
      id: m.id, object: 'model', created: 0, owned_by: 'ollama',
    }));
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data }));
    return;
  }

  // /v1/chat/completions — auto-download if needed, then forward
  if (req.method === 'POST' && url === '/v1/chat/completions') {
    const body = await readBody(req);
    let parsed;
    try { parsed = JSON.parse(body); }
    catch (_) { res.writeHead(400); res.end('Bad JSON'); return; }

    const modelId  = parsed.model || '';
    const isStream = parsed.stream !== false;
    const onDisk   = downloadedModels().includes(modelId);

    if (onDisk) {
      // Already downloaded — forward directly, no overhead
      forwardToOllama(req, res, body);
      return;
    }

    // Not on disk — need to pull first
    if (isStream) {
      // Start streaming response immediately so OpenCode shows activity
      res.writeHead(200, {
        'Content-Type':  'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
      });
      sendSSEChunk(res, `⬇ Model **${modelId}** is not downloaded yet. Downloading now — this may take several minutes…\n`);

      // Pipe live progress into the chat stream
      function onProgress(ev) {
        if (ev.model !== modelId) return;
        if (ev.type === 'progress' && ev.pct > 0) {
          // Overwrite the previous line with \r so progress isn't a wall of text
          sendSSEChunk(res, `\r⬇ ${ev.pct}%  ${ev.line || ''}`);
        }
      }
      // Temporarily hook into broadcast by adding a fake SSE client
      const progressSink = { write: data => {
        try {
          const obj = JSON.parse(data.replace(/^data: /, '').trim());
          onProgress(obj);
        } catch (_) {}
      }};
      sseClients.add(progressSink);

      try {
        await ensureDownloaded(modelId);
        sseClients.delete(progressSink);
        sendSSEChunk(res, `\n✓ Download complete. Running inference…\n\n`);
        // Now pipe ollama's actual streaming response into the open SSE stream
        pipeOllamaStream(req, res, body);
      } catch (e) {
        sseClients.delete(progressSink);
        sendSSEChunk(res, `\n❌ Download failed: ${e.message}\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
    } else {
      // Non-streaming — block until downloaded, then forward
      try {
        await ensureDownloaded(modelId);
        forwardToOllama(req, res, body);
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: `Download failed: ${e.message}` } }));
      }
    }
    return;
  }

  // Everything else (embeddings, tags, etc.) — forward verbatim
  const body = await readBody(req);
  forwardToOllama(req, res, body);
});

proxyServer.listen(PROXY_PORT, '127.0.0.1', () =>
  console.log(`[ollama-proxy] listening on 127.0.0.1:${PROXY_PORT}`)
);

// ── Status UI (port 4000) ─────────────────────────────────────────────────────
// Polls ollama state every 5 s and broadcasts changes so the UI stays current.
let lastSnapshot = { downloaded: [], running: [] };
setInterval(() => {
  const snap = { downloaded: downloadedModels(), running: runningModels() };
  const changed =
    snap.downloaded.join() !== lastSnapshot.downloaded.join() ||
    snap.running.join()    !== lastSnapshot.running.join();
  if (changed) {
    lastSnapshot = snap;
    broadcast({ type: 'snapshot', ...snap });
  }
}, 5000);

function buildStatusHTML(total, avail) {
  const fitting    = MODELS.filter(m => m.ramGB <= total * 0.85);
  const downloaded = downloadedModels();
  const running    = runningModels();

  const rows = fitting.map(m => {
    const isRunning    = running.includes(m.id);
    const isDownloaded = downloaded.includes(m.id);
    const dl           = downloads.get(m.id);
    const isDownloading = dl && dl.status === 'downloading';

    let statusBadge;
    if (isDownloading)    statusBadge = `<span class="badge dl" id="badge-${m.id}">⬇ ${dl.pct}%</span>`;
    else if (isRunning)   statusBadge = `<span class="badge running">running</span>`;
    else if (isDownloaded) statusBadge = `<span class="badge downloaded">downloaded</span>`;
    else                  statusBadge = `<span class="badge none">—</span>`;

    let progressRow = '';
    if (isDownloading) {
      progressRow = `
        <tr id="progress-row-${m.id}">
          <td colspan="4" style="padding:0 .75rem .75rem">
            <div class="pbar-track"><div class="pbar-fill" id="pbar-${m.id}" style="width:${dl.pct}%"></div></div>
            <div class="pbar-line" id="pline-${m.id}">${dl.line}</div>
          </td>
        </tr>`;
    }

    return `
      <tr data-model="${m.id}">
        <td><strong>${m.name}</strong><br><small>${m.desc}</small></td>
        <td class="num">${m.ramGB} GB</td>
        <td id="status-${m.id}">${statusBadge}</td>
      </tr>${progressRow}`;
  }).join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Status</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #11111b; color: #cdd6f4;
         margin: 0; padding: 1.5rem; }
  h1 { color: #cba6f7; margin-top: 0; font-size: 1.2rem; }
  .stats { background: #1e1e2e; border-radius: 8px; padding: .6rem 1rem;
           margin-bottom: 1rem; display: flex; gap: 1.5rem; font-size: .85rem; }
  .stats span { color: #a6e3a1; }
  .hint { font-size: .82rem; color: #6c7086; margin-bottom: 1rem; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #1e1e2e; color: #89b4fa; padding: .55rem .75rem; text-align: left;
       font-size: .82rem; }
  td { padding: .55rem .75rem; border-bottom: 1px solid #313244; vertical-align: middle;
       font-size: .88rem; }
  .num { text-align: right; }
  .badge { padding: 2px 9px; border-radius: 99px; font-size: .72rem; font-weight: 700;
           display: inline-block; }
  .badge.running    { background: #89b4fa; color: #1e1e2e; }
  .badge.downloaded { background: #a6e3a1; color: #1e1e2e; }
  .badge.dl         { background: #f9e2af; color: #1e1e2e; font-variant-numeric: tabular-nums; }
  .badge.none       { color: #45475a; font-weight: 400; }
  .pbar-track { background: #313244; border-radius: 99px; height: 8px; overflow: hidden;
                margin-bottom: .3rem; }
  .pbar-fill  { background: #cba6f7; height: 100%; border-radius: 99px;
                transition: width .4s ease; }
  .pbar-line  { font-family: monospace; font-size: .75rem; color: #a6adc8;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .legend { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;
            font-size: .8rem; }
  .legend span { display: flex; align-items: center; gap: .4rem; }
</style>
</head>
<body>
<h1>Model Status</h1>
<div class="stats">
  Total RAM: <span>${total.toFixed(1)} GB</span>
  Available: <span>${avail.toFixed(1)} GB</span>
  Threshold: <span>${(total * 0.85).toFixed(1)} GB</span>
</div>
<div class="legend">
  <span><span class="badge running">running</span> loaded in RAM</span>
  <span><span class="badge downloaded">downloaded</span> on disk, not in RAM</span>
  <span><span class="badge dl">⬇ N%</span> downloading now</span>
  <span><span class="badge none">—</span> not downloaded</span>
</div>
<p class="hint">Select a model inside OpenCode using <code>/models</code>. If not yet downloaded, it will be pulled automatically when first used.</p>
<table>
  <thead><tr><th>Model</th><th class="num">Min RAM</th><th>Status</th></tr></thead>
  <tbody id="tbody">${rows}</tbody>
</table>

<script>
const es = new EventSource('events');

function setStatus(model, html) {
  const el = document.getElementById('status-' + model);
  if (el) el.innerHTML = html;
}
function setProgress(model, pct, line) {
  const badge = document.getElementById('badge-' + model);
  if (badge) badge.textContent = '⬇ ' + pct + '%';
  const bar   = document.getElementById('pbar-' + model);
  if (bar)   bar.style.width = pct + '%';
  const pline = document.getElementById('pline-' + model);
  if (pline && line) pline.textContent = line;
}
function removeProgressRow(model) {
  const row = document.getElementById('progress-row-' + model);
  if (row) row.remove();
}
function insertProgressRow(model) {
  const row = document.querySelector('[data-model="' + model + '"]');
  if (!row || document.getElementById('progress-row-' + model)) return;
  const pr = document.createElement('tr');
  pr.id = 'progress-row-' + model;
  pr.innerHTML =
    '<td colspan="3" style="padding:0 .75rem .75rem">' +
    '<div class="pbar-track"><div class="pbar-fill" id="pbar-' + model + '" style="width:0%"></div></div>' +
    '<div class="pbar-line" id="pline-' + model + '"></div></td>';
  row.after(pr);
}

es.onmessage = function(e) {
  const ev = JSON.parse(e.data);
  if (ev.type === 'start') {
    insertProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge dl" id="badge-' + ev.model + '">⬇ 0%</span>');
  } else if (ev.type === 'progress') {
    setProgress(ev.model, ev.pct, ev.line);
  } else if (ev.type === 'done') {
    removeProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge downloaded">downloaded</span>');
  } else if (ev.type === 'error') {
    removeProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge none" title="' + (ev.error||'') + '">error</span>');
  } else if (ev.type === 'snapshot') {
    // Full refresh of downloaded/running badges for all rows
    document.querySelectorAll('[data-model]').forEach(row => {
      const id = row.getAttribute('data-model');
      const dl = downloads && downloads[id];
      if (dl && dl.status === 'downloading') return; // don't overwrite live progress
      const isRunning    = ev.running.includes(id);
      const isDownloaded = ev.downloaded.includes(id);
      if (isRunning)        setStatus(id, '<span class="badge running">running</span>');
      else if (isDownloaded) setStatus(id, '<span class="badge downloaded">downloaded</span>');
      else                   setStatus(id, '<span class="badge none">—</span>');
    });
  }
};
</script>
</body></html>`;
}

const uiServer = http.createServer((req, res) => {
  const url = req.url.split('?')[0].replace(/\/+$/, '') || '/';

  if (req.method === 'GET' && (url === '/' || url === '/index.html')) {
    const total = memGB('MemTotal');
    const avail = memGB('MemAvailable');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(buildStatusHTML(total, avail));
    return;
  }

  if (req.method === 'GET' && url === '/events') {
    res.writeHead(200, {
      'Content-Type':      'text/event-stream',
      'Cache-Control':     'no-cache',
      'Connection':        'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.write(': connected\n\n');

    // Replay in-progress downloads on reconnect
    for (const [model, dl] of downloads) {
      if (dl.status === 'downloading') {
        res.write(`data: ${JSON.stringify({ type: 'progress', model, pct: dl.pct, line: dl.line })}\n\n`);
      }
    }

    sseClients.add(res);
    const ping = setInterval(() => {
      try { res.write(': ping\n\n'); }
      catch (_) { clearInterval(ping); sseClients.delete(res); }
    }, 15000);
    req.on('close', () => { clearInterval(ping); sseClients.delete(res); });
    return;
  }

  res.writeHead(404); res.end('Not found');
});

uiServer.listen(UI_PORT, '0.0.0.0', () =>
  console.log(`[model-status] http://localhost:${UI_PORT}`)
);
