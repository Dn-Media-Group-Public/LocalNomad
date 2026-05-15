# locallm

A self-contained Docker container that runs a fully local, CPU-based AI coding environment. No cloud API keys required. Everything persists in your project directory.

**Stack:**
- [Ollama](https://ollama.com) — local LLM inference server
- [OpenCode](https://opencode.ai) — AI coding agent CLI
- [CodeNomad](https://github.com/NeuralNomadsAI/CodeNomad) — web UI cockpit for OpenCode
- Ollama proxy — sits between OpenCode and Ollama; auto-downloads models on first use
- Model Status sidecar — read-only status page showing what is downloaded, running, or downloading

---

## Architecture

```
Your browser
    │
    └─► CodeNomad  (HTTP, port 3000)
              │
              ├─► OpenCode sessions
              │         │
              │         └─► Ollama proxy  (port 11435, localhost only)
              │                   │   auto-pulls model if not on disk,
              │                   │   then forwards to real Ollama
              │                   └─► Ollama  (port 11434, localhost only)
              │                               └─► /workspace/.ollama/
              │
              └─► Model Status sidecar  (port 4000, localhost only)
                        shows: downloading / downloaded / running in RAM
```

Only port 3000 is exposed outside the container. Ollama, the proxy, and the status page are localhost-only.

---

## How model selection works

1. In an OpenCode session, run `/models` and pick any model from the list.
2. All models that fit within 85% of the container's total RAM are shown — including ones not yet downloaded.
3. If the chosen model is not on disk, the proxy intercepts the first request and streams `ollama pull` progress directly into the chat as a message, then continues with the actual inference once the download completes.
4. On subsequent uses the model is already on disk and there is no delay.
5. Ollama keeps the most recently used model loaded in RAM; it is evicted when another model is loaded or when ollama is restarted.

The **Model Status** tab inside CodeNomad shows all models with live status badges at a glance.

---

## File layout

```
locallm/
├── Dockerfile
├── entrypoint.sh
└── model-selector/
    └── server.js        # ollama proxy (port 11435) + status UI (port 4000)
```

### Dockerfile

- Base: `ubuntu:24.04`
- Installs: Node.js 20, `curl`, `zstd` (required by the Ollama installer), Ollama, `opencode-ai` (npm global), `@neuralnomads/codenomad` (npm global)
- Bakes in `CODENOMAD_SKIP_AUTH=true`
- Only `EXPOSE 3000`

### entrypoint.sh — startup sequence

1. `export HOME=/workspace` — all `~/…` paths resolve inside the workspace mount
2. `ollama serve` starts in background; script polls until ready
3. `model-selector/server.js` starts (proxy on 11435, status UI on 4000)
4. `opencode.json` is written (or updated) with:
   - Provider pointing to the proxy at `http://localhost:11435/v1`
   - Every model that fits in ≤85% of total RAM pre-listed so OpenCode's `/models` shows them all
   - Any existing keys the user has set are preserved
5. `codenomad config.yaml` is created on first run with the Model Status sidecar pre-registered
6. `codenomad` starts in foreground on port 3000

### model-selector/server.js

Two HTTP servers in one process.

#### Ollama proxy (port 11435)

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/v1/models` | Returns all RAM-fitting models (not just downloaded ones) so OpenCode's `/models` picker shows the full list |
| `POST` | `/v1/chat/completions` | If model is on disk: forward directly to Ollama. If not: stream download progress as chat content chunks, then forward the real request once complete |
| anything else | `*` | Forward verbatim to Ollama on port 11434 |

When a model needs downloading the proxy sends OpenAI-format streaming chunks into the open chat stream so the user sees live progress inside the OpenCode session rather than a silent hang.

#### Model Status UI (port 4000)

Read-only HTML page. Served as a CodeNomad SideCar tab.

| Badge | Meaning |
|---|---|
| `running` (blue) | Loaded in RAM — `ollama ps` |
| `downloaded` (green) | On disk, not in RAM — `ollama list` |
| `⬇ N%` (yellow) | Download in progress |
| `—` | Not downloaded |

The page subscribes to a Server-Sent Events stream (`/events`) for live updates. The server polls `ollama list` and `ollama ps` every 5 seconds and broadcasts changes to all connected clients.

#### Model catalogue (Q4_K_M estimates, shown filtered by RAM)

| Model | Min RAM | Notes |
|---|---|---|
| qwen2.5-coder:1.5b | 2 GB | Tiny, fast |
| qwen2.5-coder:3b | 3 GB | Simple tasks |
| llama3.2:3b | 3 GB | Small general |
| codellama:7b | 6 GB | Meta code model |
| mistral:7b | 6 GB | General purpose |
| qwen2.5-coder:7b | 6 GB | Recommended starting point |
| llama3.1:8b | 7 GB | Strong general |
| codellama:13b | 10 GB | |
| qwen2.5-coder:14b | 11 GB | Very capable |
| phi4:14b | 11 GB | Strong reasoning |
| deepseek-coder-v2:16b | 13 GB | |
| qwen2.5-coder:32b | 22 GB | Best local coder |
| llama3.1:70b | 48 GB | Needs ≥64 GB |

---

## Data persistence

`HOME` is set to `/workspace` at runtime, so all persistent data lands inside the bind-mounted directory as hidden subdirectories. No named Docker volumes are needed.

| Path | Contents |
|---|---|
| `/workspace/.ollama/` | Downloaded model weights |
| `/workspace/.config/opencode/` | OpenCode config and session history |
| `/workspace/.config/codenomad/` | CodeNomad config, sidecar definitions, instance history |

---

## Usage

### Build

```bash
docker build -t locallm .
```

### Run

```bash
docker run -d \
  --name locallm \
  -p 3000:3000 \
  -v /path/to/your/project:/workspace \
  locallm
```

Open **http://localhost:3000**.

### Select a model

Inside any OpenCode session in CodeNomad:

```
/models
```

Pick a model. If it is not yet downloaded, the proxy will pull it and show progress in the chat. The model is then used for that message and all subsequent ones in the session.

### Check model status

Open the **Model Status** tab in CodeNomad's SideCar panel. It shows which models are downloaded, which is currently loaded in RAM, and any active downloads with a live progress bar.

### Stopping and restarting

```bash
docker stop locallm
docker start locallm
```

Downloaded models and all config are preserved in the workspace directory.

---

## Security notes

- **Authentication is disabled** (`CODENOMAD_SKIP_AUTH=true`). Do not expose port 3000 to the internet without a reverse proxy with authentication in front.
- Ollama, the proxy, and the status UI bind to localhost inside the container and are not reachable from outside.
- The workspace bind mount gives the container full read/write access to your project directory.
