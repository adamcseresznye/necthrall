# ============================================================================
# Necthrall - Production-Ready Multi-Stage Dockerfile for Heroku
# ============================================================================
# Optimized for Heroku Basic dyno (512MB RAM, 500MB slug limit)
# ============================================================================

# ============================================================================
# STAGE 1: Builder - Install dependencies and prepare models
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# --- 1. SETUP BUILD VENV (Heavy Tools) ---
# We use this separate venv to run the ONNX export script with torch/transformers
RUN python -m venv /opt/venv_build
ENV PATH="/opt/venv_build/bin:$PATH"

RUN echo "Installing CPU-only PyTorch for build..." && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN echo "Installing build dependencies..." && \
    pip install -r /tmp/requirements.txt && \
    pip install optimum[onnxruntime]

# Create and populate model cache
RUN mkdir -p /opt/onnx_model_cache
COPY scripts/setup_onnx.py /tmp/setup_onnx.py
WORKDIR /opt
RUN echo "Exporting and quantizing ONNX model..." && \
    python /tmp/setup_onnx.py && \
    echo "ONNX model baked into image"
RUN mv /opt/onnx_model_cache /opt/onnx_model_cache_final

# --- 2. SETUP RUNTIME VENV (Lightweight App) ---
# CRITICAL FIX: We create this at /opt/venv so paths match the final stage
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Filter out heavy ML libs and install app dependencies
RUN grep -vE "^torch|^transformers|^sentence-transformers|^optimum" /tmp/requirements.txt > /tmp/requirements_runtime.txt && \
    echo "Installing runtime requirements..." && \
    pip install -r /tmp/requirements_runtime.txt

# Install lightweight inference engine
RUN echo "Installing lightweight inference dependencies..." && \
    pip install onnxruntime tokenizers

# ============================================================================
# STAGE 2: Final - Minimal runtime image
# ============================================================================
FROM python:3.11-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# Setup Environment for Caches
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# Install minimal runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# CRITICAL FIX: Copy from /opt/venv to /opt/venv (Matching Paths!)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy baked models
COPY --from=builder /opt/onnx_model_cache_final /app/onnx_model_cache

# Create directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers /app/.cache/torch

# Copy App
COPY . /app

# User Security
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Sanity Check
RUN python -c "import onnxruntime; print(f'Runtime Ready: {onnxruntime.__version__}')"

EXPOSE $PORT

# Start
CMD ["sh", "-c", "python -c 'import gc; gc.set_threshold(50, 5, 5);' && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30"]
