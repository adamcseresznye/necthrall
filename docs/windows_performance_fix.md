# ONNX Performance - Threading Configuration

**Status**: ✅ **SOLVED** - Achieved 7x speedup (27 → 194 chunks/sec)

## Results

| Configuration | Throughput | 10k chunks | Status |
|--------------|------------|------------|--------|
| PyTorch FP32 | 16 chunks/sec | 641s | ❌ Too slow |
| ONNX INT8 (broken) | 27 chunks/sec | 370s | ❌ Single-threaded |
| **ONNX INT8 (fixed)** | **194 chunks/sec** | **52s** | ✅ **Production ready** |
| Sanity check baseline | 361 chunks/sec | 28s | Reference |

## Windows ONNX Performance Fix

## Problem

ONNX Runtime with CPU execution delivers ~20-30 chunks/sec on Windows, even with INT8 quantization. Expected performance is 200-400 chunks/sec with multi-core utilization.

**Root Cause**: Windows OpenBLAS/MKL libraries lock their thread count when first loaded and ignore `OMP_NUM_THREADS` set in-process. This forces single-core operation.

## Solution

Environment variables **MUST be set at the very top of your script** before any imports (including config):

```python
import os

# CRITICAL: Set threading BEFORE any imports
num_cores = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["TORCH_NUM_THREADS"] = str(num_cores)

# NOW safe to import config and other libraries
from config.onnx_embedding import ONNXEmbeddingModel
```

### Alternative: Set before starting Python

### PowerShell

```powershell
$env:OMP_NUM_THREADS="16"; $env:MKL_NUM_THREADS="16"; $env:TORCH_NUM_THREADS="16"; python scripts/test_onnx_performance.py
```

### Command Prompt

```cmd
set OMP_NUM_THREADS=16 && set MKL_NUM_THREADS=16 && set TORCH_NUM_THREADS=16 && python scripts/test_onnx_performance.py
```

### FastAPI/Uvicorn (Production)

```powershell
$env:OMP_NUM_THREADS="16"
$env:MKL_NUM_THREADS="16"
$env:TORCH_NUM_THREADS="16"
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker/Heroku (Production)

Add to `Dockerfile` or environment configuration:

```dockerfile
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4
```

## Verification

Check threading is working:

```powershell
# Should see "Threading: 16 cores configured" at startup
$env:OMP_NUM_THREADS="16"; python -c "from config.onnx_embedding import ONNXEmbeddingModel; print('Config loaded')"
```

Expected performance:
- **Single-core (broken)**: ~20-30 chunks/sec
- **Multi-core (working)**: ~200-400 chunks/sec

## Why In-Process Config Doesn't Work

1. Python starts without threading env vars set
2. First import of numpy/torch/onnxruntime causes OpenBLAS to initialize
3. OpenBLAS checks `OMP_NUM_THREADS` **once** during DLL load
4. After initialization, OpenBLAS ignores all changes to environment variables
5. Setting env vars in code (even before imports) is too late on Windows

## Production Deployment Checklist

- [ ] Set `OMP_NUM_THREADS` in process manager (systemd, Docker, Heroku config)
- [ ] Verify with test script showing >200 chunks/sec
- [ ] Monitor CPU usage (should see 80-100% utilization across cores)
- [ ] Adjust thread count based on available CPU cores (recommend = core count)

## References

- OpenBLAS threading: https://github.com/OpenMathLib/OpenBLAS/wiki/faq#multi-threaded
- ONNX Runtime CPU: https://onnxruntime.ai/docs/execution-providers/CPU-ExecutionProvider.html
- Windows DLL initialization order: https://github.com/pytorch/pytorch/issues/91966
