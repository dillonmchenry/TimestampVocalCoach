# SecondPass - Vocal Coaching Pipeline
#
# Base: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#   - Matches local dev environment (PyTorch 2.6.0 + CUDA 12.4, RTX 3070 / sm_86)
#   - "runtime" variant (~4.5 GB) vs "devel" (~9 GB) -- we don't need nvcc
#
# Build-time prerequisites (files that must exist before `docker build`):
#   - rmvpe/model.pt          (~368 MB, gitignored -- download from verstar/STARS on HF)
#   - data/student_v6/        (tracked in git, already present)
#   - ../NanoPitch/           (sibling repo, cloned locally)
#
# Build:
#   docker build -t secondpass:latest .
#
# Run locally (GPU):
#   docker run --gpus all -p 8000:8000 --env-file .env secondpass:latest
#
# Run locally (CPU, slower):
#   docker run -p 8000:8000 --env-file .env secondpass:latest

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# libsndfile1  - soundfile / librosa WAV I/O
# ffmpeg       - torchaudio backend for non-WAV formats
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

# Sprint 3 extras not in the above files
RUN pip install --no-cache-dir \
    chromadb \
    openai \
    python-dotenv

# ---------------------------------------------------------------------------
# Model weights (large, rarely change -- own layer for cache efficiency)
# ---------------------------------------------------------------------------

# RMVPE pitch encoder (~368 MB).
# This file is gitignored. Run scripts/fetch_rmvpe.py or download manually:
#   huggingface-cli download verstar/STARS rmvpe/model.pt --local-dir .
COPY rmvpe/model.pt ./rmvpe/model.pt

# STARS student checkpoint (~7 MB, tracked in git)
COPY data/student_v6/ ./data/student_v6/

# NanoPitch: only model.py (runtime import) + the best checkpoint (~1.7 MB)
# We copy just the training/ source and the single best.pth to keep the layer small.
COPY nanopitch/training/model.py ./nanopitch/training/model.py
COPY nanopitch/training/runs/best_150+late_clean_112gru_model/checkpoints/best.pth \
     ./nanopitch/training/runs/best_150+late_clean_112gru_model/checkpoints/best.pth

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
# Note: ~1.5 GB -- keep this layer last so code changes don't invalidate it.
# ---------------------------------------------------------------------------
COPY data/songs/ ./data/songs/

# RAG store: ChromaDB index + playbook YAMLs + pedagogy source Markdown.
# Built by scripts/build_rag.py. Needed for Sprint 3 LLM coaching features.
COPY data/rag/ ./data/rag/

# ---------------------------------------------------------------------------
# Wire RMVPE into the path that STARS's checkpoint loader expects.
#
# setup_stars_runtime.py also tries to link the 700 MB bilingual teacher;
# we skip that here since production uses stars_profile=fast (student only).
# The student runner only needs rmvpe/model.pt, which we link below.
# ---------------------------------------------------------------------------
RUN mkdir -p third_party/stars/checkpoints/rmvpe \
    && ln -sf /app/rmvpe/model.pt /app/third_party/stars/checkpoints/rmvpe/model.pt \
    && mkdir -p third_party/stars/data/processed/bilingual \
    && cp third_party/stars/chinese_and_english_phone_set.json \
          third_party/stars/data/processed/bilingual/phone_set.json

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Tell vocal_coach/pitch.py where NanoPitch lives
ENV NANOPITCH_DIR=/app/nanopitch

# Prevents Python output buffering (shows uvicorn logs immediately)
ENV PYTHONUNBUFFERED=1

# Default to GPU; override with -e SECONDPASS_DEVICE=cpu for CPU-only runs
ENV SECONDPASS_DEVICE=cuda

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

# Bind to 0.0.0.0 so the container port is reachable from outside.
# (The tunnel/cloud platform handles external TLS termination.)
CMD ["python", "-m", "uvicorn", "web.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "120"]
