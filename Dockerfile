# ============================================================================
# Necthrall - Optimized Dockerfile for Hugging Face Spaces (Docker SDK)
# ============================================================================
# - Base: python:3.11-slim
# - Pre-downloads sentence-transformers/all-MiniLM-L6-v2
# - Runs as non-root user (UID 1000) on port 7860
# ============================================================================

FROM python:3.11-slim

# Environment variables for Python and ML optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# ============================================================================
# STEP 1: Install system dependencies
# ============================================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ============================================================================
# STEP 2: Install Python dependencies
# ============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ============================================================================
# STEP 3: Pre-download and convert embedding model to ONNX (cached in image layer)
# ============================================================================
# Copy the setup script first
COPY scripts/setup_onnx.py /tmp/setup_onnx.py

# Run the ONNX setup script to download and quantize the model
RUN python /tmp/setup_onnx.py && rm /tmp/setup_onnx.py

# ============================================================================
# STEP 4: Create non-root user (HF Spaces requirement: UID 1000)
# ============================================================================
RUN useradd -m -u 1000 user

# Set home and working directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ============================================================================
# STEP 5: Copy application code
# ============================================================================
# Copy with ownership set to user
COPY --chown=user:user . $HOME/app

# Copy the pre-built ONNX model cache to the app directory
RUN cp -r /onnx_model_cache $HOME/app/onnx_model_cache && \
    chown -R user:user $HOME/app/onnx_model_cache

# Switch to non-root user
USER user

# ============================================================================
# STEP 6: Expose port and run application
# ============================================================================
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
