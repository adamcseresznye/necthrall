# ============================================================================
# verify_build_size.ps1 - Verify Docker image size for Heroku deployment
# ============================================================================
# Heroku Basic dyno has a 500MB slug size limit.
# This script builds the image and checks if it exceeds the limit.
# ============================================================================

$ErrorActionPreference = "Stop"

$IMAGE_NAME = "necthrall-heroku"
$MAX_SIZE_MB = 500

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Building Docker image: $IMAGE_NAME" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Build the Docker image
docker build -t $IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Checking image size..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Get image size in bytes and convert to MB
$SIZE_BYTES = docker image inspect $IMAGE_NAME --format='{{.Size}}' | Out-String
$SIZE_BYTES = [long]$SIZE_BYTES.Trim()
$SIZE_MB = [math]::Floor($SIZE_BYTES / 1024 / 1024)

# Display image info
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.ID}}"

Write-Host ""
Write-Host "--------------------------------------------"
Write-Host "Image size: $SIZE_MB MB"
Write-Host "Heroku limit: $MAX_SIZE_MB MB"
Write-Host "--------------------------------------------"

# Check if size exceeds limit
if ($SIZE_MB -gt $MAX_SIZE_MB) {
    Write-Host ""
    Write-Host "WARNING: Image size ($SIZE_MB MB) exceeds Heroku's $MAX_SIZE_MB MB limit!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Suggestions to reduce size:" -ForegroundColor Yellow
    Write-Host "  - Review multi-stage build to exclude unnecessary files"
    Write-Host "  - Check for large dependencies that can be removed"
    Write-Host "  - Ensure .dockerignore excludes test files, docs, etc."
    Write-Host "  - Consider using alpine base image (may require additional setup)"
    Write-Host ""
    exit 1
} else {
    Write-Host ""
    Write-Host "Image size is within Heroku's $MAX_SIZE_MB MB limit." -ForegroundColor Green
    $HEADROOM = $MAX_SIZE_MB - $SIZE_MB
    Write-Host "Headroom: $HEADROOM MB" -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Layer breakdown (top 10 by size):" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
docker history $IMAGE_NAME --format "table {{.Size}}\t{{.CreatedBy}}" | Select-Object -First 12

Write-Host ""
Write-Host "Build verification complete!" -ForegroundColor Green
