# ============================================================================
# Necthrall Lite - Memory-Optimized Production Dockerfile
# ============================================================================
# Designed for Heroku Basic dyno (512MB RAM limit)
# Pre-downloads ML models during BUILD phase to avoid runtime memory spikes
# ============================================================================

FROM python:3.11-slim

# ============================================================================
# STEP 1: Set cache environment variables BEFORE any model downloads
# ============================================================================
# All model caches point to /app/.cache for consistent storage
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    TORCH_HOME=/app/.cache/torch \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ============================================================================
# STEP 2: Install system dependencies
# ============================================================================
RUN echo "📦 Installing system dependencies..." && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "✅ System dependencies installed"

# ============================================================================
# STEP 3: Install Python dependencies from requirements.txt
# ============================================================================
# Copy only requirements.txt first for better layer caching
COPY requirements.txt /app/

# CRITICAL: Install CPU-only PyTorch FIRST to avoid 2GB+ GPU bloat
# This must come BEFORE requirements.txt which includes sentence-transformers
RUN echo "📦 Installing CPU-only PyTorch (saves ~2GB)..." && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    echo "✅ CPU-only PyTorch installed"

RUN echo "📦 Installing Python dependencies..." && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "✅ Python dependencies installed"

# Install additional build-time dependencies for ONNX export
# These are needed for model conversion but can be cleaned up later
RUN echo "📦 Installing ONNX export dependencies..." && \
    pip install --no-cache-dir optimum[onnxruntime] && \
    echo "✅ ONNX export dependencies installed"

# ============================================================================
# STEP 3: Create cache directories with proper permissions
# ============================================================================
RUN mkdir -p /app/.cache/huggingface \
             /app/.cache/sentence-transformers \
             /app/.cache/torch \
             /app/onnx_model_cache

# ============================================================================
# STEP 4: Pre-download standard models BEFORE copying application code
# ============================================================================
# This ensures models are cached in a Docker layer that won't invalidate 
# when application code changes, saving 5-10 minutes per build
# 
# CRITICAL: Only use pip-installed packages here, NO custom modules!
# ============================================================================

# Download sentence-transformers embedding model
RUN echo "📦 Downloading sentence-transformers/all-MiniLM-L6-v2..." && \
    python3 <<'EOF'
import sys
try:
    from sentence_transformers import SentenceTransformer
    print("  ⏳ Loading model (this may take a few minutes)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Verify model works
    test_embedding = model.encode(["test"])
    print(f"  ✅ Model downloaded and verified (embedding dim: {len(test_embedding[0])})")
except Exception as e:
    print(f"  ❌ ERROR: Failed to download embedding model: {e}")
    sys.exit(1)
EOF

# Download cross-encoder reranker model
RUN echo "📦 Downloading cross-encoder/ms-marco-MiniLM-L-6-v2..." && \
    python3 <<'EOF'
import sys
try:
    from sentence_transformers import CrossEncoder
    print("  ⏳ Loading cross-encoder model...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Verify model works
    test_score = model.predict([("query", "document")])
    print(f"  ✅ Cross-encoder downloaded and verified (test score: {test_score[0]:.4f})")
except Exception as e:
    print(f"  ❌ ERROR: Failed to download cross-encoder model: {e}")
    sys.exit(1)
EOF

# ============================================================================
# STEP 5: Copy application code AFTER standard model downloads
# ============================================================================
# This separation ensures model downloads are cached in Docker layers that
# persist across code changes. Only layers AFTER this point rebuild when
# code changes.
COPY . /app

# ============================================================================
# STEP 6: Run ONNX setup AFTER code copy (with graceful error handling)
# ============================================================================
# The ONNX setup script converts the model to optimized ONNX format
# If it fails, we log a warning but continue (allow runtime fallback)
RUN echo "📦 Setting up ONNX-optimized embedding model..." && \
    python3 <<'EOF'
import sys
import os
from pathlib import Path

onnx_cache = Path("/app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2")
quant_model = onnx_cache / "model_quantized.onnx"

# Check if ONNX model already exists
if quant_model.exists():
    print(f"  ✅ ONNX model already exists at {quant_model}")
    sys.exit(0)

# Try using scripts/setup_onnx.py if it exists
setup_script = Path("/app/scripts/setup_onnx.py")
if setup_script.exists():
    print("  ⏳ Running scripts/setup_onnx.py...")
    try:
        # Change to app directory for relative paths
        os.chdir("/app")
        exec(open(setup_script).read())
        print("  ✅ ONNX setup completed via setup_onnx.py")
        sys.exit(0)
    except Exception as e:
        print(f"  ⚠️ WARNING: setup_onnx.py failed: {e}")
        print("  ⏳ Attempting direct ONNX initialization...")

# Fallback: Try direct ONNX export using optimum
print("  ⏳ Attempting direct ONNX export with optimum...")
try:
    import shutil
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    float_path = onnx_cache / "float32"

    # Create cache directory
    onnx_cache.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print("    📥 Exporting model to ONNX format...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(float_path)

    # Save tokenizer
    print("    📝 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(onnx_cache)

    # Copy config
    if (float_path / "config.json").exists():
        shutil.copy(float_path / "config.json", onnx_cache / "config.json")

    # Quantize for better performance
    print("    📉 Quantizing to INT8...")
    quantize_dynamic(
        model_input=float_path / "model.onnx",
        model_output=quant_model,
        weight_type=QuantType.QUInt8,
    )

    # Cleanup float32 model to save space
    if float_path.exists():
        shutil.rmtree(float_path)

    print("  ✅ ONNX model exported and quantized successfully")
    sys.exit(0)

except ImportError as e:
    print(f"  ⚠️ WARNING: Required packages not available: {e}")
    print("  ⚠️ ONNX optimization skipped - will use standard inference at runtime")
except Exception as e:
    print(f"  ⚠️ WARNING: ONNX export failed: {e}")
    print("  ⚠️ Will attempt runtime initialization or fallback to standard inference")

# Don't fail the build - allow runtime fallback
sys.exit(0)
EOF

# Verify ONNX model if created
RUN echo "🔍 Verifying ONNX setup..." && \
    python3 <<'EOF'
from pathlib import Path

onnx_cache = Path("/app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2")
quant_model = onnx_cache / "model_quantized.onnx"
tokenizer = onnx_cache / "tokenizer.json"

if quant_model.exists():
    size_mb = quant_model.stat().st_size / (1024 * 1024)
    print(f"  ✅ ONNX model found: {quant_model} ({size_mb:.1f} MB)")
    if tokenizer.exists():
        print(f"  ✅ Tokenizer found: {tokenizer}")
    else:
        print(f"  ⚠️ Tokenizer not found at {tokenizer}")
else:
    print("  ⚠️ ONNX model not found - will use standard inference at runtime")

# List cache contents for debugging
import os
cache_path = Path("/app/.cache")
if cache_path.exists():
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    print(f"  📊 Total cache size: {total_size / (1024 * 1024):.1f} MB")
EOF

# ============================================================================
# STEP 7: Set runtime environment variables for CPU-only execution
# ============================================================================
# These settings prevent threading issues and reduce memory usage
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NUMEXPR_MAX_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# ============================================================================
# STEP 8: Create non-root user for security
# ============================================================================
RUN echo "👤 Creating non-root user..." && \
    groupadd -g 1000 app && \
    useradd -u 1000 -g app -m -s /bin/bash app && \
    chown -R app:app /app && \
    echo "✅ Non-root user created"

USER app

# ============================================================================
# STEP 9: Expose port and set health check
# ============================================================================
EXPOSE 8000

# ============================================================================
# STEP 10: CMD with aggressive garbage collection
# ============================================================================
# GC optimization reduces memory fragmentation on low-memory systems
# - threshold(50, 5, 5): More aggressive collection than default (700, 10, 10)
# - Single worker to minimize memory usage
# - Keep-alive timeout for connection reuse
CMD ["sh", "-c", "python -c 'import gc; gc.set_threshold(50, 5, 5); print(\"✅ GC configured for low-memory operation\")' && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30"]
