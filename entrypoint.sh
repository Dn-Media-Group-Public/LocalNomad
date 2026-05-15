#!/bin/bash
set -e

# All config lives under /workspace as hidden dirs — no separate mounts needed.
export HOME=/workspace

# ── 1. Start Ollama ──────────────────────────────────────────────────────────
ollama serve &

echo "[locallm] Waiting for ollama..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "[locallm] Ollama ready."

# ── 2. Start proxy + status UI (must be up before opencode reads config) ─────
node /opt/model-selector/server.js &
echo "[locallm] Ollama proxy on 127.0.0.1:11435, status UI on 0.0.0.0:4000"

# ── 3. Bootstrap OpenCode config (first-run) ─────────────────────────────────
# Writes opencode.json with:
#   - provider pointing to the proxy (port 11435)
#   - every model that fits in ≤85% of total RAM pre-listed
#   - no default model set (user picks via /models in OpenCode)
mkdir -p ~/.config/opencode

node - <<'JSEOF'
const fs   = require('fs');
const path = require('path');
const os   = require('os');

const CFG = path.join(os.homedir(), '.config', 'opencode', 'opencode.json');

const MODELS = [
  { id: 'qwen2.5-coder:1.5b',    ramGB:  2 },
  { id: 'qwen2.5-coder:3b',      ramGB:  3 },
  { id: 'llama3.2:3b',           ramGB:  3 },
  { id: 'codellama:7b',          ramGB:  6 },
  { id: 'mistral:7b',            ramGB:  6 },
  { id: 'qwen2.5-coder:7b',      ramGB:  6 },
  { id: 'llama3.1:8b',           ramGB:  7 },
  { id: 'codellama:13b',         ramGB: 10 },
  { id: 'qwen2.5-coder:14b',     ramGB: 11 },
  { id: 'phi4:14b',              ramGB: 11 },
  { id: 'deepseek-coder-v2:16b', ramGB: 13 },
  { id: 'qwen2.5-coder:32b',     ramGB: 22 },
  { id: 'llama3.1:70b',          ramGB: 48 },
];

function totalRAMgb() {
  try {
    const m = fs.readFileSync('/proc/meminfo', 'utf8').match(/MemTotal:\s+(\d+)\s+kB/);
    if (m) return parseInt(m[1]) / 1048576;
  } catch (_) {}
  return 8;
}

const totalGB  = totalRAMgb();
const fitting  = MODELS.filter(m => m.ramGB <= totalGB * 0.85);
const models   = {};
for (const m of fitting) models[m.id] = { name: m.id };

let cfg = {};
if (fs.existsSync(CFG)) {
  try { cfg = JSON.parse(fs.readFileSync(CFG, 'utf8')); } catch (_) {}
}

// Always keep provider config pointing to proxy with current RAM-filtered model list.
// Preserve any other keys the user may have set (e.g. model, rules, theme).
cfg.provider = cfg.provider || {};
cfg.provider.ollama = {
  npm:     '@ai-sdk/openai-compatible',
  name:    'Ollama (local)',
  options: { baseURL: 'http://localhost:11435/v1' },
  models,
};

// Only allow the local ollama provider — suppress all cloud providers.
cfg.enabled_providers = ['ollama'];

fs.writeFileSync(CFG, JSON.stringify(cfg, null, 2));
console.log(`[locallm] opencode.json updated — ${fitting.length} models listed (RAM threshold: ${(totalGB * 0.85).toFixed(1)} GB)`);
JSEOF

# ── 4. Bootstrap CodeNomad config with status sidecar (first-run) ────────────
mkdir -p ~/.config/codenomad

node - <<'JSEOF'
const fs   = require('fs');
const path = require('path');
const os   = require('os');

const CFG = path.join(os.homedir(), '.config', 'codenomad', 'config.yaml');

const SIDECAR = `\
server:
  sidecars:
    - id: model-status
      kind: port
      name: Model Status
      port: 4000
      insecure: true
      prefixMode: strip
      createdAt: "${new Date().toISOString()}"
      updatedAt: "${new Date().toISOString()}"
`;

if (!fs.existsSync(CFG)) {
  fs.writeFileSync(CFG, SIDECAR);
  console.log('[locallm] Created codenomad config.yaml with Model Status sidecar.');
} else if (!fs.readFileSync(CFG, 'utf8').includes('id: model-status')) {
  console.log('[locallm] codenomad config.yaml exists — skipping sidecar injection.');
} else {
  console.log('[locallm] Model Status sidecar already present.');
}
JSEOF

# ── 5. Init workspace git repo (CodeNomad requires git) ──────────────────────
if [ ! -d /workspace/.git ]; then
    git init /workspace
    git -C /workspace config user.email 'local@localhost'
    git -C /workspace config user.name 'Local'
    echo "[locallm] Initialised git repo in /workspace"
fi

# ── 6. Start CodeNomad ────────────────────────────────────────────────────────
echo "[locallm] Starting CodeNomad on http://0.0.0.0:3000"
exec codenomad \
    --https=false \
    --http=true \
    --http-port=3000 \
    --host=0.0.0.0 \
    --workspace-root=/workspace
