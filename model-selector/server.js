'use strict';
const http  = require('http');
const fs    = require('fs');
const path  = require('path');
const { execSync, exec, spawn } = require('child_process');
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
let activeModel  = null;   // last model sent to ollama for inference
const loadingModels = new Set(); // models currently being loaded into RAM by ollama
let activeOllamaReq = null;  // the in-flight ollama inference request (if any)

const INFERENCE_TIMEOUT_MS = 5 * 60 * 1000;  // 5 min hard cap per request

function broadcast(obj) {
  const msg = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of sseClients) {
    try { res.write(msg); } catch (_) { sseClients.delete(res); }
  }
}

// Ask ollama to immediately unload a model from RAM.
function unloadModel(modelId) {
  return new Promise(resolve => {
    const body = Buffer.from(JSON.stringify({ model: modelId, keep_alive: 0 }));
    const req = http.request(
      { hostname: OLLAMA_HOST, port: OLLAMA_PORT, path: '/api/generate',
        method: 'POST', headers: { 'content-type': 'application/json',
                                   'content-length': body.length } },
      res => { res.resume(); res.on('end', resolve); }
    );
    req.on('error', resolve);   // ignore errors — best-effort
    req.write(body);
    req.end();
  });
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

// Models on disk (async — avoids blocking the event loop)
function downloadedModels() {
  return new Promise(resolve => {
    exec('ollama list 2>/dev/null', { timeout: 5000 }, (err, stdout) => {
      if (err) return resolve([]);
      resolve(stdout.split('\n').slice(1).map(l => l.split(/\s+/)[0]).filter(Boolean));
    });
  });
}

// Models currently loaded in RAM (async)
function runningModels() {
  return new Promise(resolve => {
    exec('ollama ps 2>/dev/null', { timeout: 5000 }, (err, stdout) => {
      if (err) return resolve([]);
      resolve(stdout.split('\n').slice(1).map(l => l.split(/\s+/)[0]).filter(Boolean));
    });
  });
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
// onFirstChunk (optional) fires once when the first response byte arrives.
function forwardToOllama(req, res, bodyBuffer, onFirstChunk) {
  if (activeOllamaReq) {
    try { activeOllamaReq.destroy(); } catch (_) {}
    activeOllamaReq = null;
  }
  const opts = makeOllamaOptions(req, bodyBuffer.length);
  const proxy = http.request(opts, proxyRes => {
    activeOllamaReq = null;
    const headers = Object.assign({}, proxyRes.headers, { 'x-accel-buffering': 'no' });
    res.writeHead(proxyRes.statusCode, headers);
    if (onFirstChunk) {
      proxyRes.once('data', () => { onFirstChunk(); });
    }
    proxyRes.pipe(res);
  });
  activeOllamaReq = proxy;
  const timeoutHandle = setTimeout(() => {
    try { proxy.destroy(); } catch (_) {}
    activeOllamaReq = null;
    if (!res.headersSent) res.writeHead(504);
    res.end(JSON.stringify({ error: { message: 'Inference timed out' } }));
  }, INFERENCE_TIMEOUT_MS);
  proxy.on('error', e => {
    clearTimeout(timeoutHandle);
    activeOllamaReq = null;
    if (!res.headersSent) res.writeHead(502);
    res.end(`Ollama proxy error: ${e.message}`);
  });
  proxy.on('response', () => clearTimeout(timeoutHandle));
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
// Sends SSE heartbeats until first byte arrives so clients don't show "queued".
// Enforces INFERENCE_TIMEOUT_MS and aborts any previous in-flight request.
// onFirstChunk (optional) fires once when the first response byte arrives.
function pipeOllamaStream(req, res, bodyBuffer, onFirstChunk) {
  // Abort any previous in-flight inference so ollama isn't blocked.
  if (activeOllamaReq) {
    try { activeOllamaReq.destroy(); } catch (_) {}
    activeOllamaReq = null;
  }

  const opts = makeOllamaOptions(req, bodyBuffer.length);
  const proxy = http.request(opts, proxyRes => {
    clearInterval(heartbeat);
    clearTimeout(timeoutHandle);
    activeOllamaReq = null;
    if (onFirstChunk) {
      proxyRes.once('data', () => { onFirstChunk(); });
    }
    proxyRes.pipe(res);
  });

  activeOllamaReq = proxy;

  // Heartbeat: send SSE comments every 5 s while waiting for first byte.
  // This keeps the connection alive and prevents "queued" / stale detection.
  const heartbeat = setInterval(() => {
    try { res.write(': processing\n\n'); } catch (_) { clearInterval(heartbeat); }
  }, 5000);

  // Hard timeout: if ollama takes too long, abort and tell the client.
  const timeoutHandle = setTimeout(() => {
    clearInterval(heartbeat);
    try { proxy.destroy(); } catch (_) {}
    activeOllamaReq = null;
    sendSSEChunk(res, `\n⏱ Inference timed out after ${INFERENCE_TIMEOUT_MS / 60000} min. Try a shorter prompt or smaller model.\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  }, INFERENCE_TIMEOUT_MS);

  proxy.on('error', e => {
    clearInterval(heartbeat);
    clearTimeout(timeoutHandle);
    activeOllamaReq = null;
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
    const [onDiskList, runningList] = await Promise.all([downloadedModels(), runningModels()]);
    const onDisk   = onDiskList.includes(modelId);

    // Evict the previous model from RAM before loading a new one
    if (activeModel && activeModel !== modelId) {
      await unloadModel(activeModel);
      broadcast({ type: 'unloaded', model: activeModel });
    }
    activeModel = modelId;

    // Helper: signal loading→running transitions to the sidecar
    function signalLoading() {
      loadingModels.add(modelId);
      broadcast({ type: 'loading', model: modelId });
    }
    function signalRunning() {
      loadingModels.delete(modelId);
      broadcast({ type: 'running', model: modelId });
    }
    function statusImgTag() {
      return `<img src="http://localhost:${UI_PORT}/status.svg?t=${Date.now()}" ` +
        `onload="setTimeout(()=>{this.src='http://localhost:${UI_PORT}/status.svg?t='+Date.now()},1000)" ` +
        `style="display:block;max-width:420px;border-radius:8px" alt="Model Status"/>`;
    }

    if (onDisk) {
      const alreadyRunning = runningList.includes(modelId);
      if (!alreadyRunning && isStream) {
        // Model on disk but not in RAM — show loading image then pipe inference
        res.writeHead(200, {
          'Content-Type':  'text/event-stream',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no',
        });
        sendSSEChunk(res, `⏳ Loading **${modelId}** into RAM…\n\n${statusImgTag()}\n`);
        signalLoading();
        pipeOllamaStream(req, res, body, signalRunning);
      } else if (alreadyRunning && isStream) {
        // Model already in RAM, streaming — set SSE headers explicitly so the
        // AI SDK sees text/event-stream before Ollama's first byte arrives.
        res.writeHead(200, {
          'Content-Type':  'text/event-stream',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no',
        });
        signalLoading();
        pipeOllamaStream(req, res, body, signalRunning);
      } else {
        // Non-streaming request
        signalLoading();
        forwardToOllama(req, res, body, signalRunning);
      }
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
      sendSSEChunk(res, `⬇ Model **${modelId}** not downloaded. Pulling now…\n\n` +
        `${statusImgTag()}\n`);

      try {
        await ensureDownloaded(modelId);
        sendSSEChunk(res, `✓ Download complete. Running inference…\n\n`);
        signalLoading();
        pipeOllamaStream(req, res, body, signalRunning);
      } catch (e) {
        sendSSEChunk(res, `❌ Download failed: ${e.message}\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
    } else {
      // Non-streaming — block until downloaded, then forward
      try {
        await ensureDownloaded(modelId);
        signalLoading();
        forwardToOllama(req, res, body, signalRunning);
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
setInterval(async () => {
  const [downloaded, running] = await Promise.all([downloadedModels(), runningModels()]);
  const snap = { downloaded, running };
  const changed =
    snap.downloaded.join() !== lastSnapshot.downloaded.join() ||
    snap.running.join()    !== lastSnapshot.running.join();
  if (changed) {
    lastSnapshot = snap;
    broadcast({ type: 'snapshot', ...snap });
  }
}, 5000);

function buildStatusSVG(cachedRunning) {
  // Collect active models: downloading > loading > running (show at most one running)
  const rows = [];
  for (const [model, dl] of downloads) {
    if (dl.status === 'downloading') rows.push({ model, phase: 'downloading', pct: dl.pct, line: dl.line });
  }
  for (const model of loadingModels) {
    rows.push({ model, phase: 'loading', pct: 0, line: 'Loading weights into RAM\u2026' });
  }
  if (rows.length === 0) {
    const running = cachedRunning || lastSnapshot.running;
    if (running.length > 0) rows.push({ model: running[0], phase: 'running', pct: 100, line: 'Ready' });
    else rows.push({ model: '\u2014', phase: 'idle', pct: 0, line: 'No model active' });
  }

  const W = 420, ROW = 72, PAD = 10;
  const H = rows.length * ROW + PAD * 2;

  const COLORS = {
    bg: '#1e1e2e', track: '#313244', text: '#cdd6f4', sub: '#a6adc8',
    downloading: '#f9e2af', loading: '#fab387', running: '#89b4fa',
    bar_dl: '#cba6f7', bar_load: '#fab387', bar_run: '#89b4fa',
  };

  function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

  const svgRows = rows.map((r, i) => {
    const y = PAD + i * ROW;
    const barY = y + 40;
    const BAR_W = W - PAD * 2;

    // Badge colours
    const badgeColor  = COLORS[r.phase] || COLORS.sub;
    const badgeText   = r.phase === 'downloading' ? `\u2b07 ${r.pct}%`
                      : r.phase === 'loading'     ? 'loading\u2026'
                      : r.phase === 'running'     ? 'running'
                      :                             'idle';

    // Progress bar fill / indeterminate
    let barContent;
    if (r.phase === 'downloading') {
      const fillW = Math.round(BAR_W * r.pct / 100);
      barContent = `<rect x="${PAD}" y="${barY}" width="${fillW}" height="8" rx="4" fill="${COLORS.bar_dl}"/>`;
    } else if (r.phase === 'loading') {
      // Animated shimmer via SMIL
      const shimW = Math.round(BAR_W * 0.3);
      barContent = `
        <defs>
          <clipPath id="clip${i}">
            <rect x="${PAD}" y="${barY}" width="${BAR_W}" height="8" rx="4"/>
          </clipPath>
        </defs>
        <rect x="${PAD}" y="${barY}" width="${BAR_W}" height="8" rx="4" fill="${COLORS.track}"/>
        <rect x="${PAD}" y="${barY}" width="${shimW}" height="8" fill="${COLORS.bar_load}" clip-path="url(#clip${i})">
          <animateTransform attributeName="transform" type="translate"
            values="${-shimW} 0; ${BAR_W + shimW} 0"
            dur="1.4s" repeatCount="indefinite"/>
        </rect>`;
    } else if (r.phase === 'running') {
      barContent = `<rect x="${PAD}" y="${barY}" width="${BAR_W}" height="8" rx="4" fill="${COLORS.bar_run}"/>`;
    } else {
      barContent = '';
    }

    return `
      <text x="${PAD}" y="${y + 16}" font-family="monospace" font-size="13" font-weight="bold" fill="${COLORS.text}">${esc(r.model)}</text>
      <rect x="${W - 80}" y="${y + 2}" width="70" height="18" rx="9" fill="${badgeColor}"/>
      <text x="${W - 45}" y="${y + 15}" font-family="monospace" font-size="11" font-weight="bold" fill="#1e1e2e" text-anchor="middle">${esc(badgeText)}</text>
      <rect x="${PAD}" y="${barY}" width="${BAR_W}" height="8" rx="4" fill="${COLORS.track}"/>
      ${barContent}
      <text x="${PAD}" y="${barY + 22}" font-family="monospace" font-size="10" fill="${COLORS.sub}">${esc(r.line)}</text>`;
  }).join('');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <rect width="${W}" height="${H}" rx="8" fill="${COLORS.bg}"/>
  ${svgRows}
</svg>`;
}

function buildStatusHTML(total, avail, downloaded, running) {
  const fitting    = MODELS.filter(m => m.ramGB <= total * 0.85);

  const rows = fitting.map(m => {
    const isRunning     = running.includes(m.id);
    const isDownloaded  = downloaded.includes(m.id);
    const dl            = downloads.get(m.id);
    const isDownloading = dl && dl.status === 'downloading';
    const isLoading     = loadingModels.has(m.id);

    let statusBadge;
    if (isDownloading)    statusBadge = `<span class="badge dl" id="badge-${m.id}">⬇ ${dl.pct}%</span>`;
    else if (isLoading)   statusBadge = `<span class="badge loading" id="badge-${m.id}">loading…</span>`;
    else if (isRunning)   statusBadge = `<span class="badge running">running</span>`;
    else if (isDownloaded) statusBadge = `<span class="badge downloaded">downloaded</span>`;
    else                  statusBadge = `<span class="badge none">—</span>`;

    let progressRow = '';
    if (isDownloading) {
      progressRow = `
        <tr id="progress-row-${m.id}">
          <td colspan="3" style="padding:0 .75rem .75rem">
            <div class="pbar-track"><div class="pbar-fill" id="pbar-${m.id}" style="width:${dl.pct}%"></div></div>
            <div class="pbar-line" id="pline-${m.id}">${dl.line}</div>
          </td>
        </tr>`;
    } else if (isLoading) {
      progressRow = `
        <tr id="progress-row-${m.id}">
          <td colspan="3" style="padding:0 .75rem .75rem">
            <div class="pbar-track"><div class="pbar-indeterminate" id="pbar-${m.id}"></div></div>
            <div class="pbar-line" id="pline-${m.id}">Loading weights into RAM…</div>
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
  .badge.loading    { background: #fab387; color: #1e1e2e; animation: pulse .9s ease-in-out infinite alternate; }
  .badge.none       { color: #45475a; font-weight: 400; }
  .pbar-track { background: #313244; border-radius: 99px; height: 8px; overflow: hidden;
                margin-bottom: .3rem; }
  .pbar-fill  { background: #cba6f7; height: 100%; border-radius: 99px;
                transition: width .4s ease; }
  @keyframes shimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
  }
  @keyframes pulse { from { opacity: 1; } to { opacity: .55; } }
  .pbar-indeterminate { position: relative; height: 100%; width: 25%;
                        background: #fab387; border-radius: 99px;
                        animation: shimmer 1.4s ease-in-out infinite; }
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
  <span><span class="badge loading">loading…</span> loading into RAM</span>
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
const localLoading = new Set(); // models we know are loading, to guard snapshot

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
function insertProgressRow(model, indeterminate) {
  const row = document.querySelector('[data-model="' + model + '"]');
  if (!row) return;
  removeProgressRow(model); // replace if already present
  const pr = document.createElement('tr');
  pr.id = 'progress-row-' + model;
  if (indeterminate) {
    pr.innerHTML =
      '<td colspan="3" style="padding:0 .75rem .75rem">' +
      '<div class="pbar-track"><div class="pbar-indeterminate" id="pbar-' + model + '"></div></div>' +
      '<div class="pbar-line" id="pline-' + model + '">Loading weights into RAM\u2026</div></td>';
  } else {
    pr.innerHTML =
      '<td colspan="3" style="padding:0 .75rem .75rem">' +
      '<div class="pbar-track"><div class="pbar-fill" id="pbar-' + model + '" style="width:0%"></div></div>' +
      '<div class="pbar-line" id="pline-' + model + '"></div></td>';
  }
  row.after(pr);
}

es.onmessage = function(e) {
  const ev = JSON.parse(e.data);
  if (ev.type === 'start') {
    insertProgressRow(ev.model, false);
    setStatus(ev.model, '<span class="badge dl" id="badge-' + ev.model + '">⬇ 0%</span>');
  } else if (ev.type === 'progress') {
    setProgress(ev.model, ev.pct, ev.line);
  } else if (ev.type === 'done') {
    removeProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge downloaded">downloaded</span>');
  } else if (ev.type === 'loading') {
    localLoading.add(ev.model);
    insertProgressRow(ev.model, true);
    setStatus(ev.model, '<span class="badge loading" id="badge-' + ev.model + '">loading\u2026</span>');
  } else if (ev.type === 'running') {
    localLoading.delete(ev.model);
    removeProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge running">running</span>');
  } else if (ev.type === 'unloaded') {
    localLoading.delete(ev.model);
    removeProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge downloaded">downloaded</span>');
  } else if (ev.type === 'error') {
    removeProgressRow(ev.model);
    setStatus(ev.model, '<span class="badge none" title="' + (ev.error||'') + '">error</span>');
  } else if (ev.type === 'snapshot') {
    // Full refresh of downloaded/running badges — skip models with live state
    document.querySelectorAll('[data-model]').forEach(row => {
      const id = row.getAttribute('data-model');
      if (localLoading.has(id)) return;
      if (document.getElementById('progress-row-' + id)) return; // downloading
      const isRunning    = ev.running.includes(id);
      const isDownloaded = ev.downloaded.includes(id);
      if (isRunning)         setStatus(id, '<span class="badge running">running</span>');
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
    res.end(buildStatusHTML(total, avail, lastSnapshot.downloaded, lastSnapshot.running));
    return;
  }

  if (req.method === 'GET' && url === '/status.svg') {
    res.writeHead(200, {
      'Content-Type':  'image/svg+xml',
      'Cache-Control': 'no-store',
    });
    res.end(buildStatusSVG(lastSnapshot.running));
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
        res.write(`data: ${JSON.stringify({ type: 'start', model })}\n\n`);
        res.write(`data: ${JSON.stringify({ type: 'progress', model, pct: dl.pct, line: dl.line })}\n\n`);
      }
    }
    // Replay in-progress loads on reconnect
    for (const model of loadingModels) {
      res.write(`data: ${JSON.stringify({ type: 'loading', model })}\n\n`);
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
