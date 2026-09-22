# SecondPass - Vocal Coaching Pipeline
#
# Base: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#   - Matches local dev environment (PyTorch 2.6.0 + CUDA 12.4)
#   - "runtime" variant (~4.5 GB) vs "devel" (~9 GB) -- we don't need nvcc
#
# Build (from repo root):
#   docker build -t secondpass:latest .
#
# Run locally (GPU):
#   docker run --gpus all -p 8000:8000 --env-file .env secondpass:latest
#
# Run locally (CPU, slower):
#   docker run -p 8000:8000 -e SECONDPASS_DEVICE=cpu --env-file .env secondpass:latest
#
# Or use docker compose (see docker-compose.yml):
#   docker compose --profile gpu up
#   docker compose --profile cpu up

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# libsndfile1  - soundfile / librosa WAV I/O
# ffmpeg       - torchaudio backend for non-WAV formats (MP3 reference vocals)
# git          - needed by some pip installs (e.g. g2p_en data download)
# build-essential / swig - webrtcvad C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        git \
        build-essential \
        swig \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies (layered for cache efficiency)
# ---------------------------------------------------------------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# STARS extras (pyworld, praat-parselmouth, webrtcvad, etc.)
COPY third_party/stars/requirements.txt ./stars_requirements.txt
RUN pip install --no-cache-dir -r stars_requirements.txt

# ---------------------------------------------------------------------------
# Model weights (large, rarely change -- own layer for cache efficiency)
# ---------------------------------------------------------------------------

# STARS student checkpoint (~7 MB, tracked in git)
COPY data/student_v6/ ./data/student_v6/

# NanoPitch: vendored model.py + best.pth checkpoint (~1.7 MB, tracked in git)
COPY nanopitch/ ./nanopitch/

# ---------------------------------------------------------------------------
# Application source code
# ---------------------------------------------------------------------------
COPY vocal_coach/   ./vocal_coach/
COPY web/           ./web/
COPY config/        ./config/
COPY scripts/       ./scripts/
COPY third_party/   ./third_party/

# ---------------------------------------------------------------------------
# Song bundles (audio + precomputed reference artifacts)
# Note: ~525 MB -- keep this layer last so code changes don't invalidate it.
# ---------------------------------------------------------------------------
COPY data/songs/ ./data/songs/

# RAG store: ChromaDB index + playbook YAMLs + pedagogy source Markdown.
COPY data/rag/ ./data/rag/

# ---------------------------------------------------------------------------
# Wire STARS phone set into the path its loader expects.
#
# Production uses stars_profile=fast (student only), so the 700 MB bilingual
# teacher checkpoint and RMVPE weights are not needed here.
# ---------------------------------------------------------------------------
RUN mkdir -p third_party/stars/data/processed/bilingual \
    && cp third_party/stars/chinese_and_english_phone_set.json \
          third_party/stars/data/processed/bilingual/phone_set.json

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Tell vocal_coach/pitch.py where NanoPitch lives (vendored in /app/nanopitch)
ENV NANOPITCH_DIR=/app/nanopitch

# Prevents Python output buffering (shows uvicorn logs immediately)
ENV PYTHONUNBUFFERED=1

# Default to GPU; override with -e SECONDPASS_DEVICE=cpu for CPU-only runs
ENV SECONDPASS_DEVICE=cuda

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Bind to 0.0.0.0 so the container port is reachable from outside.
CMD ["python", "-m", "uvicorn", "web.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "120"]
