FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# Skip CodeNomad auth — always on, no need to pass at runtime
ENV CODENOMAD_SKIP_AUTH=true

# Node.js 20 + base tools
RUN apt-get update && apt-get install -y \
    curl ca-certificates gnupg zstd git && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Ollama
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
