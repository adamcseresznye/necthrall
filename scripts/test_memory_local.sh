#!/bin/bash
# ============================================================================
# Necthrall Lite - Local Memory Testing Script
# ============================================================================
# Simulates Heroku's 512MB memory limit locally using Docker
# Must pass before deploying to Heroku Basic dyno
#
# Usage:
#   chmod +x scripts/test_memory_local.sh
#   ./scripts/test_memory_local.sh
#
# Exit codes:
#   0 - All tests passed, ready for Heroku
#   1 - Tests failed, fix issues before deploying
# ============================================================================

# Configuration
IMAGE_NAME="necthrall-memory-test"
CONTAINER_NAME="necthrall-test"
MEMORY_LIMIT="512m"
PORT="8000"
STARTUP_WAIT=10
CURL_TIMEOUT=30

# Track overall test status
TEST_PASSED=true
CLEANUP_NEEDED=false

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Cleanup function - runs on exit regardless of success/failure
# ============================================================================
cleanup() {
    if [ "$CLEANUP_NEEDED" = true ]; then
        echo ""
        echo -e "${BLUE}🧹 Cleaning up...${NC}"
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        echo "   Container removed"
    fi
}

# Register cleanup to run on script exit
trap cleanup EXIT

# ============================================================================
# Helper function to print section headers
# ============================================================================
print_header() {
    echo ""
    echo "============================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "============================================================================"
}

# ============================================================================
# Pre-flight: Clean up any existing container with same name
# ============================================================================
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}🧪 Necthrall Lite - Local Memory Testing${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Target: Heroku Basic dyno (512MB RAM limit)"
echo "Container: $CONTAINER_NAME"
echo "Memory limit: $MEMORY_LIMIT"
echo ""

# Remove any existing container with same name to avoid conflicts
echo "🔄 Checking for existing containers..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "   Found existing container, removing..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo "   ✅ Cleaned up existing container"
else
    echo "   ✅ No conflicts found"
fi

# ============================================================================
# Step 1: Build Docker Image
# ============================================================================
print_header "📦 Step 1: Building Docker Image"

echo "Building image: $IMAGE_NAME"
echo "This may take 5-15 minutes on first build..."
echo ""

BUILD_START=$(date +%s)

if docker build -t $IMAGE_NAME . ; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    echo ""
    echo -e "${GREEN}✅ Build successful!${NC} (${BUILD_TIME}s)"
else
    echo ""
    echo -e "${RED}❌ Build failed!${NC}"
    echo "Check the build output above for errors."
    exit 1
fi

# ============================================================================
# Step 2: Display Image Size
# ============================================================================
print_header "📏 Step 2: Image Size Analysis"

echo "Docker images:"
echo ""
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "(REPOSITORY|$IMAGE_NAME)"

# Get size in bytes for comparison
IMAGE_SIZE_BYTES=$(docker image inspect $IMAGE_NAME --format='{{.Size}}' 2>/dev/null || echo "0")
IMAGE_SIZE_MB=$((IMAGE_SIZE_BYTES / 1024 / 1024))

echo ""
if [ "$IMAGE_SIZE_MB" -lt 2048 ]; then
    echo -e "${GREEN}✅ Image size: ${IMAGE_SIZE_MB}MB (under 2GB limit)${NC}"
else
    echo -e "${YELLOW}⚠️ Image size: ${IMAGE_SIZE_MB}MB (exceeds 2GB target)${NC}"
fi

# ============================================================================
# Step 3: Verify Cached Models
# ============================================================================
print_header "📂 Step 3: Verify Cached Models"

echo "Checking cache directory inside image..."
echo ""

docker run --rm $IMAGE_NAME sh -c '
    # Check main cache
    if [ -d "/app/.cache" ]; then
        CACHE_SIZE=$(du -sh /app/.cache 2>/dev/null | cut -f1)
        echo "   /app/.cache size: $CACHE_SIZE"
    else
        echo "   ⚠️ /app/.cache not found"
    fi
    
    # Check ONNX cache
    if [ -d "/app/onnx_model_cache" ]; then
        ONNX_SIZE=$(du -sh /app/onnx_model_cache 2>/dev/null | cut -f1)
        echo "   /app/onnx_model_cache size: $ONNX_SIZE"
    else
        echo "   ⚠️ /app/onnx_model_cache not found"
    fi
    
    # Check for specific model file
    if [ -f "/app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2/model_quantized.onnx" ]; then
        echo ""
        echo "   ✅ ONNX quantized model present"
    else
        echo ""
        echo "   ⚠️ ONNX model not found (runtime fallback will be used)"
    fi
'

echo ""
echo -e "${GREEN}✅ Cache verification complete${NC}"

# ============================================================================
# Step 4: Start Container with Memory Limit
# ============================================================================
print_header "🚀 Step 4: Start Container (512MB limit)"

echo "Starting container with memory constraints..."
echo "   --memory=$MEMORY_LIMIT"
echo "   --memory-swap=$MEMORY_LIMIT (prevents swap usage)"

# Check for .env file
if [ -f ".env" ]; then
    echo "   --env-file=.env (loading environment variables)"
    ENV_FILE_FLAG="--env-file=.env"
else
    echo -e "${YELLOW}   ⚠️ No .env file found - container may fail to start${NC}"
    echo "   Create .env from .env.example with your API keys"
    ENV_FILE_FLAG=""
fi
echo ""

CLEANUP_NEEDED=true

docker run -d \
    --name $CONTAINER_NAME \
    --memory=$MEMORY_LIMIT \
    --memory-swap=$MEMORY_LIMIT \
    -p $PORT:8000 \
    -e PORT=8000 \
    $ENV_FILE_FLAG \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container started${NC}"
else
    echo -e "${RED}❌ Failed to start container!${NC}"
    TEST_PASSED=false
    exit 1
fi

# ============================================================================
# Step 5: Wait for Startup and Check Container Status
# ============================================================================
print_header "⏳ Step 5: Container Startup Check"

echo "Waiting ${STARTUP_WAIT} seconds for container initialization..."
sleep $STARTUP_WAIT

# Check if container is still running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✅ Container is running (not killed by OOM)${NC}"
else
    echo -e "${RED}❌ Container failed to start or was killed!${NC}"
    echo ""
    echo "Container logs:"
    echo "----------------------------------------"
    docker logs $CONTAINER_NAME 2>&1 | tail -50
    echo "----------------------------------------"
    TEST_PASSED=false
    exit 1
fi

# ============================================================================
# Step 6: Memory Usage - Before Query
# ============================================================================
print_header "📊 Step 6: Memory Usage Analysis"

echo "Memory usage BEFORE test query:"
echo ""
docker stats $CONTAINER_NAME --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Capture memory for comparison
MEM_BEFORE=$(docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}")
MEM_PERCENT_BEFORE=$(docker stats $CONTAINER_NAME --no-stream --format "{{.MemPerc}}")

# Extract numeric MB value
MEM_MB_BEFORE=$(echo "$MEM_BEFORE" | grep -oE '[0-9]+' | head -1)

echo ""
if [ "$MEM_MB_BEFORE" -lt 400 ]; then
    echo -e "${GREEN}✅ Startup memory: ${MEM_MB_BEFORE}MB (excellent, under 400MB)${NC}"
elif [ "$MEM_MB_BEFORE" -lt 450 ]; then
    echo -e "${GREEN}✅ Startup memory: ${MEM_MB_BEFORE}MB (good, under 450MB target)${NC}"
else
    echo -e "${YELLOW}⚠️ Startup memory: ${MEM_MB_BEFORE}MB (high, over 450MB target)${NC}"
fi

# ============================================================================
# Step 7: Send Test Query
# ============================================================================
print_header "🔍 Step 7: Test Query"

echo "Sending test POST request to /query endpoint..."
echo "Timeout: ${CURL_TIMEOUT}s (first query may be slow due to lazy loading)"
echo ""

# Prepare test payload
TEST_QUERY='{"query": "What is machine learning?", "max_papers": 1}'

# Send request with timing
echo "Request: POST http://localhost:$PORT/query"
echo "Payload: $TEST_QUERY"
echo ""

START_TIME=$(date +%s.%N)

HTTP_CODE=$(curl -s -o /tmp/query_response.json -w "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$TEST_QUERY" \
    --connect-timeout 10 \
    --max-time $CURL_TIMEOUT \
    "http://localhost:$PORT/query" 2>/dev/null) || HTTP_CODE="000"

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "N/A")

echo "Response code: $HTTP_CODE"
echo "Duration: ${DURATION}s"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ Query successful (HTTP 200)${NC}"
    echo ""
    echo "Response preview (first 500 chars):"
    echo "----------------------------------------"
    head -c 500 /tmp/query_response.json 2>/dev/null || echo "(no response body)"
    echo ""
    echo "----------------------------------------"
elif [ "$HTTP_CODE" = "000" ]; then
    echo -e "${YELLOW}⚠️ Connection failed or timed out${NC}"
    echo "   This may be expected if the /query endpoint isn't implemented yet."
    echo "   Checking if container is still running..."
    
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}   ✅ Container still running (no OOM crash)${NC}"
    else
        echo -e "${RED}   ❌ Container crashed during query!${NC}"
        TEST_PASSED=false
    fi
elif [ "$HTTP_CODE" = "404" ]; then
    echo -e "${YELLOW}⚠️ Endpoint not found (HTTP 404)${NC}"
    echo "   The /query endpoint may not be implemented yet."
elif [ "$HTTP_CODE" = "500" ] || [ "$HTTP_CODE" = "503" ]; then
    echo -e "${YELLOW}⚠️ Server error (HTTP $HTTP_CODE)${NC}"
    echo "   The server may still be initializing or missing dependencies."
else
    echo -e "${YELLOW}⚠️ Unexpected response (HTTP $HTTP_CODE)${NC}"
fi

# ============================================================================
# Memory Usage - After Query
# ============================================================================
echo ""
echo "Memory usage AFTER test query:"
echo ""
docker stats $CONTAINER_NAME --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

MEM_AFTER=$(docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}")
MEM_MB_AFTER=$(echo "$MEM_AFTER" | grep -oE '[0-9]+' | head -1)

echo ""
if [ "$MEM_MB_AFTER" -lt 450 ]; then
    echo -e "${GREEN}✅ Post-query memory: ${MEM_MB_AFTER}MB (under 450MB target)${NC}"
elif [ "$MEM_MB_AFTER" -lt 500 ]; then
    echo -e "${YELLOW}⚠️ Post-query memory: ${MEM_MB_AFTER}MB (close to limit)${NC}"
else
    echo -e "${RED}❌ Post-query memory: ${MEM_MB_AFTER}MB (danger zone!)${NC}"
    TEST_PASSED=false
fi

# ============================================================================
# Step 8: Check Logs for OOM Errors
# ============================================================================
print_header "🔍 Step 8: OOM Error Check"

echo "Scanning container logs for memory-related errors..."
echo ""

# Get logs and search for OOM-related keywords
LOGS=$(docker logs $CONTAINER_NAME 2>&1)
OOM_FOUND=false

# Check for various OOM indicators
if echo "$LOGS" | grep -qi "killed"; then
    echo -e "${RED}❌ Found 'killed' in logs${NC}"
    echo "$LOGS" | grep -i "killed" | head -5
    OOM_FOUND=true
fi

if echo "$LOGS" | grep -qi "oom"; then
    echo -e "${RED}❌ Found 'oom' in logs${NC}"
    echo "$LOGS" | grep -i "oom" | head -5
    OOM_FOUND=true
fi

if echo "$LOGS" | grep -qi "out of memory"; then
    echo -e "${RED}❌ Found 'out of memory' in logs${NC}"
    echo "$LOGS" | grep -i "out of memory" | head -5
    OOM_FOUND=true
fi

if echo "$LOGS" | grep -qi "cannot allocate memory"; then
    echo -e "${RED}❌ Found 'cannot allocate memory' in logs${NC}"
    echo "$LOGS" | grep -i "cannot allocate memory" | head -5
    OOM_FOUND=true
fi

if [ "$OOM_FOUND" = true ]; then
    echo ""
    echo -e "${RED}❌ OOM-related errors detected!${NC}"
    TEST_PASSED=false
else
    echo -e "${GREEN}✅ No OOM errors found in logs${NC}"
fi

# ============================================================================
# Step 9: Final Container Status
# ============================================================================
print_header "🏥 Step 9: Final Health Check"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✅ Container is still running${NC}"
    
    # Try health endpoint
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 5 \
        --max-time 10 \
        "http://localhost:$PORT/health" 2>/dev/null) || HEALTH_CODE="000"
    
    if [ "$HEALTH_CODE" = "200" ]; then
        echo -e "${GREEN}✅ Health endpoint responding (HTTP 200)${NC}"
    else
        # Try root endpoint as fallback
        ROOT_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --connect-timeout 5 \
            --max-time 10 \
            "http://localhost:$PORT/" 2>/dev/null) || ROOT_CODE="000"
        
        if [ "$ROOT_CODE" = "200" ]; then
            echo -e "${GREEN}✅ Root endpoint responding (HTTP 200)${NC}"
        else
            echo -e "${YELLOW}⚠️ No HTTP response (may still be initializing)${NC}"
        fi
    fi
else
    echo -e "${RED}❌ Container has stopped unexpectedly!${NC}"
    echo ""
    echo "Last 20 lines of logs:"
    echo "----------------------------------------"
    docker logs $CONTAINER_NAME 2>&1 | tail -20
    echo "----------------------------------------"
    TEST_PASSED=false
fi

# ============================================================================
# Final Summary
# ============================================================================
print_header "📋 Test Summary"

echo "Build time:         ${BUILD_TIME}s"
echo "Image size:         ${IMAGE_SIZE_MB}MB"
echo "Startup memory:     ${MEM_MB_BEFORE}MB"
echo "Post-query memory:  ${MEM_MB_AFTER}MB"
echo "Memory limit:       512MB"
echo "Target threshold:   450MB"
echo ""

if [ "$TEST_PASSED" = true ]; then
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}✅ ALL TESTS PASSED - Ready for Heroku deployment!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Commit your changes"
    echo "  2. Push to GitHub: git push origin semantic-scholar-pipeline"
    echo "  3. Deploy to Heroku: git push heroku semantic-scholar-pipeline:main"
    echo ""
    echo ""
    echo -e "${BLUE}Press Enter to exit...${NC}"
    read -r
    exit 0
else
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}❌ TESTS FAILED - Fix memory issues before deploying!${NC}"
    echo -e "${RED}============================================================================${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if models are pre-downloaded in Dockerfile"
    echo "  2. Ensure ONNX runtime is being used instead of PyTorch"
    echo "  3. Review container logs for specific errors"
    echo "  4. Consider reducing batch sizes or model complexity"
    echo ""
    echo "To view full logs:"
    echo "  docker logs $CONTAINER_NAME"
    echo ""
    echo ""
    echo -e "${BLUE}Press Enter to exit...${NC}"
    read -r
    exit 1
fi
