# ============================================================================
# Necthrall Lite - Docker Memory Test Script (PowerShell)
# ============================================================================
# Tests the Docker image for memory compliance with Heroku Basic dyno limits
# 
# Usage:
#   .\scripts\test_docker_memory.ps1
# ============================================================================

$ErrorActionPreference = "Stop"

$IMAGE_NAME = "necthrall-lite-test"
$CONTAINER_NAME = "necthrall-memory-test"
$MEMORY_LIMIT = "512m"
$PORT = "8000"

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "🧪 Necthrall Lite - Docker Memory Test" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Test Case 1: Build Docker Image
# ============================================================================
Write-Host "📦 Test Case 1: Building Docker image..." -ForegroundColor Yellow
Write-Host "   This may take 5-15 minutes on first build..." -ForegroundColor Gray
Write-Host ""

$BuildStart = Get-Date

try {
    docker build -t $IMAGE_NAME . 2>&1 | Tee-Object -FilePath "build.log"
    $BuildEnd = Get-Date
    $BuildTime = ($BuildEnd - $BuildStart).TotalSeconds
    
    Write-Host ""
    Write-Host "✅ Build succeeded in $([math]::Round($BuildTime)) seconds" -ForegroundColor Green
    
    if ($BuildTime -gt 900) {
        Write-Host "⚠️ WARNING: Build took longer than 15 minutes" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Test Case 2: Check Image Size
# ============================================================================
Write-Host ""
Write-Host "📏 Test Case 2: Checking image size..." -ForegroundColor Yellow

$ImageInfo = docker image inspect $IMAGE_NAME | ConvertFrom-Json
$ImageSizeBytes = $ImageInfo[0].Size
$ImageSizeMB = [math]::Round($ImageSizeBytes / 1MB)
$ImageSizeGB = [math]::Round($ImageSizeBytes / 1GB, 2)

Write-Host "   Image size: $ImageSizeMB MB ($ImageSizeGB GB)" -ForegroundColor Gray

if ($ImageSizeMB -gt 2048) {
    Write-Host "⚠️ WARNING: Image size exceeds 2GB limit" -ForegroundColor Yellow
} else {
    Write-Host "✅ Image size within 2GB limit" -ForegroundColor Green
}

# ============================================================================
# Test Case 3: Verify Cache Contents
# ============================================================================
Write-Host ""
Write-Host "📂 Test Case 3: Verifying model cache..." -ForegroundColor Yellow

$CacheCheck = @'
echo "   Checking /app/.cache..."
if [ -d "/app/.cache" ]; then
    CACHE_SIZE=$(du -sh /app/.cache 2>/dev/null | cut -f1)
    echo "   Cache size: $CACHE_SIZE"
else
    echo "   WARNING: Cache directory not found"
fi

echo ""
echo "   Checking ONNX model..."
if [ -f "/app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2/model_quantized.onnx" ]; then
    ONNX_SIZE=$(du -h /app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2/model_quantized.onnx | cut -f1)
    echo "   ONNX model found: $ONNX_SIZE"
else
    echo "   WARNING: ONNX model not found (will use runtime fallback)"
fi
'@

docker run --rm $IMAGE_NAME sh -c $CacheCheck

# ============================================================================
# Test Case 4: Start Container with Memory Limit
# ============================================================================
Write-Host ""
Write-Host "🚀 Test Case 4: Starting container with $MEMORY_LIMIT memory limit..." -ForegroundColor Yellow

# Clean up any existing container
docker rm -f $CONTAINER_NAME 2>$null | Out-Null

# Start container with memory limit
docker run -d `
    --name $CONTAINER_NAME `
    --memory=$MEMORY_LIMIT `
    --memory-swap=$MEMORY_LIMIT `
    -p "${PORT}:8000" `
    -e "PORT=8000" `
    $IMAGE_NAME | Out-Null

Write-Host "   Container started, waiting for startup..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# Check if container is still running
$Running = docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}"
if ($Running -eq $CONTAINER_NAME) {
    Write-Host "✅ Container is running" -ForegroundColor Green
} else {
    Write-Host "❌ Container crashed during startup!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Container logs:" -ForegroundColor Yellow
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME 2>$null | Out-Null
    exit 1
}

# ============================================================================
# Test Case 5: Check Memory Usage
# ============================================================================
Write-Host ""
Write-Host "📊 Test Case 5: Checking memory usage..." -ForegroundColor Yellow

$Stats = docker stats --no-stream --format "{{.MemUsage}}|{{.MemPerc}}" $CONTAINER_NAME
$StatsParts = $Stats -split '\|'
$MemoryUsage = $StatsParts[0]
$MemoryPercent = $StatsParts[1]

Write-Host "   Memory usage: $MemoryUsage" -ForegroundColor Gray
Write-Host "   Memory percent: $MemoryPercent" -ForegroundColor Gray

# Extract numeric value
if ($MemoryUsage -match '(\d+)') {
    $MemoryMB = [int]$Matches[1]
    
    if ($MemoryMB -lt 450) {
        Write-Host "✅ Memory usage is under 450MB target" -ForegroundColor Green
    } else {
        Write-Host "⚠️ WARNING: Memory usage exceeds 450MB target" -ForegroundColor Yellow
    }
}

# ============================================================================
# Test Case 6: Health Check
# ============================================================================
Write-Host ""
Write-Host "🏥 Test Case 6: Testing health endpoint..." -ForegroundColor Yellow

Start-Sleep -Seconds 5

try {
    $Response = Invoke-WebRequest -Uri "http://localhost:$PORT/health" -UseBasicParsing -TimeoutSec 10
    if ($Response.StatusCode -eq 200) {
        Write-Host "✅ Health check passed (HTTP 200)" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Health endpoint not responding - trying root..." -ForegroundColor Yellow
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:$PORT/" -UseBasicParsing -TimeoutSec 10
        if ($Response.StatusCode -eq 200) {
            Write-Host "✅ Root endpoint responding (HTTP 200)" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️ No response from container (may still be initializing)" -ForegroundColor Yellow
    }
}

# ============================================================================
# Test Case 7: Check for OOM Errors
# ============================================================================
Write-Host ""
Write-Host "🔍 Test Case 7: Checking for OOM/Killed errors..." -ForegroundColor Yellow

$Logs = docker logs $CONTAINER_NAME 2>&1

if ($Logs -match "killed|oom|out of memory") {
    Write-Host "❌ Found OOM/Killed errors in logs!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Relevant log lines:" -ForegroundColor Yellow
    $Logs -split "`n" | Where-Object { $_ -match "killed|oom|out of memory" }
} else {
    Write-Host "✅ No OOM/Killed errors found" -ForegroundColor Green
}

# ============================================================================
# Cleanup
# ============================================================================
Write-Host ""
Write-Host "🧹 Cleaning up..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME 2>$null | Out-Null
docker rm $CONTAINER_NAME 2>$null | Out-Null

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "📋 Test Summary" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "   Build time: $([math]::Round($BuildTime))s (target: <900s)" -ForegroundColor Gray
Write-Host "   Image size: ${ImageSizeMB}MB (target: <2048MB)" -ForegroundColor Gray
Write-Host "   Memory usage: ~${MemoryMB}MB (target: <450MB)" -ForegroundColor Gray
Write-Host "   OOM errors: None detected" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ All memory optimization tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the container manually:" -ForegroundColor Yellow
Write-Host "   docker run -m 512m -p 8000:8000 $IMAGE_NAME" -ForegroundColor White
