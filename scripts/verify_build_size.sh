#!/bin/bash
# ============================================================================
# verify_build_size.sh - Verify Docker image size for Heroku deployment
# ============================================================================
# Heroku Basic dyno has a 500MB slug size limit.
# This script builds the image and checks if it exceeds the limit.
# ============================================================================

set -e

IMAGE_NAME="necthrall-heroku"
MAX_SIZE_MB=500

echo "============================================"
echo "ðŸ³ Building Docker image: ${IMAGE_NAME}"
echo "============================================"

# Build the Docker image
docker build -t "${IMAGE_NAME}" .

echo ""
echo "============================================"
echo "ðŸ“Š Checking image size..."
echo "============================================"

# Get image size in bytes and convert to MB
SIZE_BYTES=$(docker image inspect "${IMAGE_NAME}" --format='{{.Size}}')
SIZE_MB=$((SIZE_BYTES / 1024 / 1024))

# Display image info
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.ID}}"

echo ""
echo "--------------------------------------------"
echo "Image size: ${SIZE_MB} MB"
echo "Heroku limit: ${MAX_SIZE_MB} MB"
echo "--------------------------------------------"

# Check if size exceeds limit
if [ "${SIZE_MB}" -gt "${MAX_SIZE_MB}" ]; then
    echo ""
    echo "âš ï¸  WARNING: Image size (${SIZE_MB} MB) exceeds Heroku's ${MAX_SIZE_MB} MB limit!"
    echo ""
    echo "Suggestions to reduce size:"
    echo "  - Review multi-stage build to exclude unnecessary files"
    echo "  - Check for large dependencies that can be removed"
    echo "  - Ensure .dockerignore excludes test files, docs, etc."
    echo "  - Consider using alpine base image (may require additional setup)"
    echo ""
    exit 1
else
    echo ""
    echo "âœ… Image size is within Heroku's ${MAX_SIZE_MB} MB limit."
    echo "   Headroom: $((MAX_SIZE_MB - SIZE_MB)) MB"
    echo ""
fi

echo "============================================"
echo "ðŸ” Layer breakdown (top 10 by size):"
echo "============================================"
docker history "${IMAGE_NAME}" --format "table {{.Size}}\t{{.CreatedBy}}" | head -12

echo ""
echo "âœ… Build verification complete!"
