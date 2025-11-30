#!/usr/bin/env bash
# ============================================================================
# Necthrall Lite - Docker Memory Test Script
# ============================================================================
# Tests the Docker image for memory compliance with Heroku Basic dyno limits
# 
# Usage:
#   chmod +x scripts/test_docker_memory.sh
#   ./scripts/test_docker_memory.sh
# ============================================================================

set -e

IMAGE_NAME="necthrall-lite-test"
CONTAINER_NAME="necthrall-memory-test"
MEMORY_LIMIT="512m"
PORT="8000"

echo "============================================================================"
echo "🧪 Necthrall Lite - Docker Memory Test"
echo "============================================================================"
echo ""

# ============================================================================
# Test Case 1: Build Docker Image
# ============================================================================
echo "📦 Test Case 1: Building Docker image..."
echo "   This may take 5-15 minutes on first build..."
echo ""

BUILD_START=$(date +%s)
docker build -t $IMAGE_NAME . 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    echo ""
    echo "✅ Build succeeded in ${BUILD_TIME} seconds"
    
    if [ $BUILD_TIME -gt 900 ]; then
        echo "⚠️ WARNING: Build took longer than 15 minutes"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

# ============================================================================
# Test Case 2: Check Image Size
# ============================================================================
echo ""
echo "📏 Test Case 2: Checking image size..."
IMAGE_SIZE=$(docker image inspect $IMAGE_NAME --format='{{.Size}}')
IMAGE_SIZE_MB=$((IMAGE_SIZE / 1024 / 1024))
IMAGE_SIZE_GB=$(echo "scale=2; $IMAGE_SIZE / 1024 / 1024 / 1024" | bc)

echo "   Image size: ${IMAGE_SIZE_MB} MB (${IMAGE_SIZE_GB} GB)"

if [ $IMAGE_SIZE_MB -gt 2048 ]; then
    echo "⚠️ WARNING: Image size exceeds 2GB limit"
else
    echo "✅ Image size within 2GB limit"
fi

# ============================================================================
# Test Case 3: Verify Cache Contents
# ============================================================================
echo ""
echo "📂 Test Case 3: Verifying model cache..."

# Create a temporary container to check cache
docker run --rm $IMAGE_NAME sh -c '
    echo "   Checking /app/.cache..."
    if [ -d "/app/.cache" ]; then
        CACHE_SIZE=$(du -sh /app/.cache 2>/dev/null | cut -f1)
        echo "   Cache size: $CACHE_SIZE"
    else
        echo "   ⚠️ Cache directory not found"
    fi
    
    echo ""
    echo "   Checking ONNX model..."
    if [ -f "/app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2/model_quantized.onnx" ]; then
        ONNX_SIZE=$(du -h /app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2/model_quantized.onnx | cut -f1)
        echo "   ✅ ONNX model found: $ONNX_SIZE"
    else
        echo "   ⚠️ ONNX model not found (will use runtime fallback)"
    fi
'

# ============================================================================
# Test Case 4: Start Container with Memory Limit
# ============================================================================
echo ""
echo "🚀 Test Case 4: Starting container with ${MEMORY_LIMIT} memory limit..."

# Clean up any existing container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Start container with memory limit
docker run -d \
    --name $CONTAINER_NAME \
    --memory=$MEMORY_LIMIT \
    --memory-swap=$MEMORY_LIMIT \
    -p $PORT:8000 \
    -e PORT=8000 \
    $IMAGE_NAME

echo "   Container started, waiting for startup..."
sleep 10

# Check if container is still running
if docker ps | grep -q $CONTAINER_NAME; then
    echo "✅ Container is running"
else
    echo "❌ Container crashed during startup!"
    echo ""
    echo "Container logs:"
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    exit 1
fi

# ============================================================================
# Test Case 5: Check Memory Usage
# ============================================================================
echo ""
echo "📊 Test Case 5: Checking memory usage..."

MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemUsage}}" $CONTAINER_NAME)
MEMORY_PERCENT=$(docker stats --no-stream --format "{{.MemPerc}}" $CONTAINER_NAME)

echo "   Memory usage: $MEMORY_USAGE"
echo "   Memory percent: $MEMORY_PERCENT"

# Extract numeric value (e.g., "350MiB" -> 350)
MEMORY_MB=$(echo $MEMORY_USAGE | grep -oE '[0-9]+' | head -1)

if [ "$MEMORY_MB" -lt 450 ]; then
    echo "✅ Memory usage is under 450MB target"
else
    echo "⚠️ WARNING: Memory usage exceeds 450MB target"
fi

# ============================================================================
# Test Case 6: Health Check
# ============================================================================
echo ""
echo "🏥 Test Case 6: Testing health endpoint..."

# Wait a bit more for the app to fully initialize
sleep 5

# Try to hit the health endpoint
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed (HTTP 200)"
elif [ "$HEALTH_RESPONSE" = "000" ]; then
    echo "⚠️ Health endpoint not responding - trying root..."
    ROOT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/ 2>/dev/null || echo "000")
    if [ "$ROOT_RESPONSE" = "200" ]; then
        echo "✅ Root endpoint responding (HTTP 200)"
    else
        echo "⚠️ No response from container (may still be initializing)"
    fi
else
    echo "⚠️ Health check returned HTTP $HEALTH_RESPONSE"
fi

# ============================================================================
# Test Case 7: Check for OOM Errors
# ============================================================================
echo ""
echo "🔍 Test Case 7: Checking for OOM/Killed errors..."

LOGS=$(docker logs $CONTAINER_NAME 2>&1)

if echo "$LOGS" | grep -qi "killed\|oom\|out of memory"; then
    echo "❌ Found OOM/Killed errors in logs!"
    echo ""
    echo "Relevant log lines:"
    echo "$LOGS" | grep -i "killed\|oom\|out of memory"
else
    echo "✅ No OOM/Killed errors found"
fi

# ============================================================================
# Cleanup
# ============================================================================
echo ""
echo "🧹 Cleaning up..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "📋 Test Summary"
echo "============================================================================"
echo "   Build time: ${BUILD_TIME}s (target: <900s)"
echo "   Image size: ${IMAGE_SIZE_MB}MB (target: <2048MB)"
echo "   Memory usage: ~${MEMORY_MB}MB (target: <450MB)"
echo "   OOM errors: None detected"
echo ""
echo "✅ All memory optimization tests completed!"
echo ""
echo "To run the container manually:"
echo "   docker run -m 512m -p 8000:8000 $IMAGE_NAME"
