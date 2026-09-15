# syntax=docker/dockerfile:1.7
# Scriba — GPU container: FastAPI + React web app.
#
#   docker compose up -d                     # http://localhost:8000
#
# CUDA runtime libraries ship inside the PyTorch wheels, so the base image only needs
# the NVIDIA driver from the host (NVIDIA Container Toolkit / Docker Desktop WSL2 GPU).

ARG PYTHON_VERSION=3.12
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------- frontend build
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund --loglevel=error
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------- python runtime
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG TORCH_INDEX
ARG INSTALL_DIARIZATION=true
ARG INSTALL_VLLM=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYANNOTE_METRICS_ENABLED=false \
    SCRIBA_APP__OUTPUT_ROOT=/data/outputs \
    SCRIBA_ENV_FILE=/data/config/.env \
    SCRIBA_CACHE_DIR=/data/config \
    SCRIBA_FRONTEND_DIR=/opt/scriba/frontend

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scriba

# PyTorch first (large, rarely changes) from the CUDA wheel index.
RUN pip install torch torchaudio --index-url ${TORCH_INDEX}

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[asr,ffmpeg,monitor]" \
 && if [ "$INSTALL_DIARIZATION" = "true" ]; then pip install pyannote.audio; fi \
 && if [ "$INSTALL_VLLM" = "true" ]; then pip install "qwen-asr[vllm]"; fi \
 && python -c "import torch, qwen_asr; print('torch', torch.__version__)"

COPY --from=frontend /build/dist ${SCRIBA_FRONTEND_DIR}

RUN useradd --create-home --uid 1000 scriba \
 && mkdir -p /data/outputs /data/config /models \
 && chown -R scriba:scriba /data /models
USER scriba

VOLUME ["/models", "/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

ENTRYPOINT ["tini", "--", "scriba"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
