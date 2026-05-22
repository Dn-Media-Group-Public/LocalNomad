# ── Stage 1: build llama-server with native ARM flags (dotprod, bf16, etc.) ──
FROM ubuntu:24.04 AS llama-builder

RUN echo Version 1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp

# GGML_NATIVE=ON detects all CPU extensions at compile time (dotprod, bf16, sve…)
RUN cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON && \
    cmake --build /tmp/llama.cpp/build --target llama-server -j$(nproc)

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# Skip CodeNomad auth — always on, no need to pass at runtime
ENV CODENOMAD_SKIP_AUTH=true

# Copy llama-server binary and its shared libs from builder stage
COPY --from=llama-builder /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /tmp/llama.cpp/build/bin/libllama-common.so.0 /usr/local/lib/
COPY --from=llama-builder /tmp/llama.cpp/build/bin/libmtmd.so.0 /usr/local/lib/
COPY --from=llama-builder /tmp/llama.cpp/build/bin/libllama.so.0 /usr/local/lib/
COPY --from=llama-builder /tmp/llama.cpp/build/bin/libggml.so.0 /usr/local/lib/
COPY --from=llama-builder /tmp/llama.cpp/build/bin/libggml-base.so.0 /usr/local/lib/
COPY --from=llama-builder /tmp/llama.cpp/build/bin/libggml-cpu.so.0 /usr/local/lib/
RUN ldconfig

# Runtime deps for llama-server (libgomp) + base tools
RUN apt-get update && apt-get install -y \
    curl ca-certificates gnupg zstd git libgomp1 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Ollama — kept for model download management (pull/list) only
RUN curl -fsSL https://ollama.com/install.sh | sh

# OpenCode CLI
RUN npm install -g opencode-ai

# CodeNomad server
RUN npm install -g @neuralnomads/codenomad

# Model selector sidecar (no external deps — pure Node built-ins)
COPY model-selector/server.js /opt/model-selector/server.js

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

# Only CodeNomad is exposed; model selector is internal (SideCar only)
EXPOSE 3000

ENTRYPOINT ["/entrypoint.sh"]
