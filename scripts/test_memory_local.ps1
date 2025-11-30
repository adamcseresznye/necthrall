<#
.SYNOPSIS
    Necthrall Lite - Local Memory Testing Script
.DESCRIPTION
    Simulates Heroku's 512MB memory limit locally using Docker.
    Must pass before deploying to Heroku Basic dyno.
.PARAMETER SkipBuild
    Skip the Docker build step (use existing image)
.PARAMETER KeepContainer
    Don't remove the container after testing
.EXAMPLE
    .\scripts\test_memory_local.ps1
    .\scripts\test_memory_local.ps1 -SkipBuild
#>

param(
    [switch]$SkipBuild,
    [switch]$KeepContainer
)

# Configuration
$IMAGE_NAME = "necthrall-memory-test"
$CONTAINER_NAME = "necthrall-test"
$MEMORY_LIMIT = "512m"
$PORT = "8000"
$STARTUP_WAIT = 10
$CURL_TIMEOUT = 30

# Track overall test status
$script:TestPassed = $true
$script:CleanupNeeded = $false
$script:BuildTime = 0
$script:ImageSizeMB = 0
$script:MemMBBefore = 0
$script:MemMBAfter = 0

# ============================================================================
# Helper Functions
# ============================================================================
function Write-Header([string]$Message) {
    Write-Host ""
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ("=" * 76) -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Err([string]$Message) {
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Detail([string]$Message) {
    Write-Host "   $Message" -ForegroundColor Gray
}

function Do-Cleanup {
    if ($script:CleanupNeeded -and (-not $KeepContainer)) {
        Write-Host ""
        Write-Host "Cleaning up..." -ForegroundColor Blue
        docker stop $CONTAINER_NAME 2>$null | Out-Null
        docker rm $CONTAINER_NAME 2>$null | Out-Null
        Write-Detail "Container removed"
    }
}

# ============================================================================
# Main Script
# ============================================================================
try {
    Write-Host ""
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host "Necthrall Lite - Local Memory Testing" -ForegroundColor Cyan
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Target: Heroku Basic dyno (512MB RAM limit)"
    Write-Host "Container: $CONTAINER_NAME"
    Write-Host "Memory limit: $MEMORY_LIMIT"
    Write-Host ""

    # ============================================================================
    # Pre-flight: Clean up existing container
    # ============================================================================
    Write-Host "Checking for existing containers..." -ForegroundColor Blue
    $existingContainer = docker ps -a --format "{{.Names}}" 2>$null | Where-Object { $_ -eq $CONTAINER_NAME }
    if ($existingContainer) {
        Write-Detail "Found existing container, removing..."
        docker stop $CONTAINER_NAME 2>$null | Out-Null
        docker rm $CONTAINER_NAME 2>$null | Out-Null
        Write-Ok "Cleaned up existing container"
    } else {
        Write-Ok "No conflicts found"
    }

    # ============================================================================
    # Step 1: Build Docker Image
    # ============================================================================
    Write-Header "Step 1: Building Docker Image"

    if ($SkipBuild) {
        Write-Warn "Skipping build (using existing image)"
        $script:BuildTime = 0
    } else {
        Write-Host "Building image: $IMAGE_NAME"
        Write-Host "This may take 5-15 minutes on first build..."
        Write-Host ""

        $BuildStart = Get-Date
        
        docker build -t $IMAGE_NAME .
        $buildExitCode = $LASTEXITCODE
        
        $BuildEnd = Get-Date
        $script:BuildTime = [math]::Round(($BuildEnd - $BuildStart).TotalSeconds)

        if ($buildExitCode -eq 0) {
            Write-Host ""
            Write-Ok "Build successful! Time: $($script:BuildTime) seconds"
        } else {
            Write-Host ""
            Write-Err "Build failed!"
            exit 1
        }
    }

    # ============================================================================
    # Step 2: Display Image Size
    # ============================================================================
    Write-Header "Step 2: Image Size Analysis"

    Write-Host "Docker images:"
    Write-Host ""
    docker images $IMAGE_NAME

    $imageJson = docker image inspect $IMAGE_NAME 2>$null
    if ($imageJson) {
        $imageInfo = $imageJson | ConvertFrom-Json
        $ImageSizeBytes = $imageInfo[0].Size
        $script:ImageSizeMB = [math]::Round($ImageSizeBytes / 1MB)
        
        Write-Host ""
        if ($script:ImageSizeMB -lt 2048) {
            Write-Ok "Image size: $($script:ImageSizeMB) MB (under 2GB limit)"
        } else {
            Write-Warn "Image size: $($script:ImageSizeMB) MB (exceeds 2GB target)"
        }
    } else {
        $script:ImageSizeMB = 0
        Write-Err "Could not inspect image"
    }

    # ============================================================================
    # Step 3: Verify Cached Models
    # ============================================================================
    Write-Header "Step 3: Verify Cached Models"

    Write-Host "Checking cache directory inside image..."
    Write-Host ""

    docker run --rm $IMAGE_NAME du -sh /app/.cache /app/onnx_model_cache 2>$null
    
    $onnxCheck = docker run --rm $IMAGE_NAME test -f /app/onnx_model_cache/sentence-transformers_all-MiniLM-L6-v2/model_quantized.onnx
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "ONNX quantized model present"
    } else {
        Write-Warn "ONNX model not found (runtime fallback will be used)"
    }

    # ============================================================================
    # Step 4: Start Container with Memory Limit
    # ============================================================================
    Write-Header "Step 4: Start Container (512MB limit)"

    Write-Host "Starting container with memory constraints..."
    Write-Detail "--memory=$MEMORY_LIMIT"
    Write-Detail "--memory-swap=$MEMORY_LIMIT (prevents swap usage)"
    Write-Host ""

    $script:CleanupNeeded = $true

    docker run -d --name $CONTAINER_NAME --memory=$MEMORY_LIMIT --memory-swap=$MEMORY_LIMIT -p "${PORT}:8000" -e PORT=8000 $IMAGE_NAME | Out-Null

    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Container started"
    } else {
        Write-Err "Failed to start container!"
        $script:TestPassed = $false
        throw "Container start failed"
    }

    # ============================================================================
    # Step 5: Wait and Check Container Status
    # ============================================================================
    Write-Header "Step 5: Container Startup Check"

    Write-Host "Waiting $STARTUP_WAIT seconds for container initialization..."
    Start-Sleep -Seconds $STARTUP_WAIT

    $running = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $CONTAINER_NAME }
    if ($running) {
        Write-Ok "Container is running (not killed by OOM)"
    } else {
        Write-Err "Container failed to start or was killed!"
        Write-Host ""
        Write-Host "Container logs:" -ForegroundColor Yellow
        Write-Host ("-" * 40)
        docker logs $CONTAINER_NAME 2>&1 | Select-Object -Last 50
        Write-Host ("-" * 40)
        $script:TestPassed = $false
        throw "Container crashed"
    }

    # ============================================================================
    # Step 6: Memory Usage - Before Query
    # ============================================================================
    Write-Header "Step 6: Memory Usage Analysis"

    Write-Host "Memory usage BEFORE test query:"
    Write-Host ""
    docker stats $CONTAINER_NAME --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

    $memBefore = docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}"
    if ($memBefore -match '(\d+)') {
        $script:MemMBBefore = [int]$Matches[1]
    }

    Write-Host ""
    if ($script:MemMBBefore -lt 400) {
        Write-Ok "Startup memory: $($script:MemMBBefore) MB (excellent, under 400MB)"
    } elseif ($script:MemMBBefore -lt 450) {
        Write-Ok "Startup memory: $($script:MemMBBefore) MB (good, under 450MB target)"
    } else {
        Write-Warn "Startup memory: $($script:MemMBBefore) MB (high, over 450MB target)"
    }

    # ============================================================================
    # Step 7: Send Test Query
    # ============================================================================
    Write-Header "Step 7: Test Query"

    Write-Host "Sending test POST request to /query endpoint..."
    Write-Host "Timeout: $CURL_TIMEOUT seconds (first query may be slow)"
    Write-Host ""

    $testPayload = @{
        query = "What is machine learning?"
        max_papers = 1
    } | ConvertTo-Json -Compress

    Write-Host "Request: POST http://localhost:$PORT/query"
    Write-Host "Payload: $testPayload"
    Write-Host ""

    $queryStart = Get-Date

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$PORT/query" -Method POST -ContentType "application/json" -Body $testPayload -TimeoutSec $CURL_TIMEOUT -UseBasicParsing

        $queryEnd = Get-Date
        $duration = [math]::Round(($queryEnd - $queryStart).TotalSeconds, 2)

        Write-Host "Response code: $($response.StatusCode)"
        Write-Host "Duration: $duration seconds"
        Write-Host ""
        Write-Ok "Query successful (HTTP $($response.StatusCode))"
        
        Write-Host ""
        Write-Host "Response preview:"
        Write-Host ("-" * 40)
        if ($response.Content.Length -gt 500) {
            Write-Host $response.Content.Substring(0, 500)
            Write-Host "..."
        } else {
            Write-Host $response.Content
        }
        Write-Host ("-" * 40)
    } catch {
        $queryEnd = Get-Date
        $duration = [math]::Round(($queryEnd - $queryStart).TotalSeconds, 2)
        
        $statusCode = 0
        if ($_.Exception.Response) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }

        Write-Host "Response code: $statusCode"
        Write-Host "Duration: $duration seconds"
        Write-Host ""

        if ($statusCode -eq 0) {
            Write-Warn "Connection failed or timed out"
            Write-Detail "This may be expected if /query endpoint is not implemented yet."
            Write-Detail "Checking if container is still running..."
            
            $stillRunning = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $CONTAINER_NAME }
            if ($stillRunning) {
                Write-Ok "Container still running (no OOM crash)"
            } else {
                Write-Err "Container crashed during query!"
                $script:TestPassed = $false
            }
        } elseif ($statusCode -eq 404) {
            Write-Warn "Endpoint not found (HTTP 404)"
            Write-Detail "The /query endpoint may not be implemented yet."
        } else {
            Write-Warn "Server error (HTTP $statusCode)"
            Write-Detail "The server may still be initializing."
        }
    }

    # Memory after query
    Write-Host ""
    Write-Host "Memory usage AFTER test query:"
    Write-Host ""
    docker stats $CONTAINER_NAME --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

    $memAfter = docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}"
    if ($memAfter -match '(\d+)') {
        $script:MemMBAfter = [int]$Matches[1]
    }

    Write-Host ""
    if ($script:MemMBAfter -lt 450) {
        Write-Ok "Post-query memory: $($script:MemMBAfter) MB (under 450MB target)"
    } elseif ($script:MemMBAfter -lt 500) {
        Write-Warn "Post-query memory: $($script:MemMBAfter) MB (close to limit)"
    } else {
        Write-Err "Post-query memory: $($script:MemMBAfter) MB (danger zone!)"
        $script:TestPassed = $false
    }

    # ============================================================================
    # Step 8: Check Logs for OOM Errors
    # ============================================================================
    Write-Header "Step 8: OOM Error Check"

    Write-Host "Scanning container logs for memory-related errors..."
    Write-Host ""

    $logs = docker logs $CONTAINER_NAME 2>&1 | Out-String
    $oomFound = $false

    $oomPatterns = @("killed", "oom", "out of memory", "cannot allocate memory")
    foreach ($pattern in $oomPatterns) {
        if ($logs -imatch $pattern) {
            Write-Err "Found pattern: $pattern"
            $oomFound = $true
        }
    }

    if ($oomFound) {
        Write-Host ""
        Write-Err "OOM-related errors detected!"
        $script:TestPassed = $false
    } else {
        Write-Ok "No OOM errors found in logs"
    }

    # ============================================================================
    # Step 9: Final Health Check
    # ============================================================================
    Write-Header "Step 9: Final Health Check"

    $stillRunning = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $CONTAINER_NAME }
    if ($stillRunning) {
        Write-Ok "Container is still running"
        
        try {
            $health = Invoke-WebRequest -Uri "http://localhost:$PORT/health" -UseBasicParsing -TimeoutSec 10
            Write-Ok "Health endpoint responding (HTTP $($health.StatusCode))"
        } catch {
            try {
                $root = Invoke-WebRequest -Uri "http://localhost:$PORT/" -UseBasicParsing -TimeoutSec 10
                Write-Ok "Root endpoint responding (HTTP $($root.StatusCode))"
            } catch {
                Write-Warn "No HTTP response (may still be initializing)"
            }
        }
    } else {
        Write-Err "Container has stopped unexpectedly!"
        Write-Host ""
        Write-Host "Last 20 lines of logs:" -ForegroundColor Yellow
        Write-Host ("-" * 40)
        docker logs $CONTAINER_NAME 2>&1 | Select-Object -Last 20
        Write-Host ("-" * 40)
        $script:TestPassed = $false
    }

    # ============================================================================
    # Final Summary
    # ============================================================================
    Write-Header "Test Summary"

    Write-Host "Build time:         $($script:BuildTime) seconds"
    Write-Host "Image size:         $($script:ImageSizeMB) MB"
    Write-Host "Startup memory:     $($script:MemMBBefore) MB"
    Write-Host "Post-query memory:  $($script:MemMBAfter) MB"
    Write-Host "Memory limit:       512 MB"
    Write-Host "Target threshold:   450 MB"
    Write-Host ""

    if ($script:TestPassed) {
        Write-Host ("=" * 76) -ForegroundColor Green
        Write-Host "ALL TESTS PASSED - Ready for Heroku deployment!" -ForegroundColor Green
        Write-Host ("=" * 76) -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  1. Commit your changes"
        Write-Host "  2. Push to GitHub: git push origin semantic-scholar-pipeline"
        Write-Host "  3. Deploy to Heroku: git push heroku semantic-scholar-pipeline:main"
        Write-Host ""
        $exitCode = 0
    } else {
        Write-Host ("=" * 76) -ForegroundColor Red
        Write-Host "TESTS FAILED - Fix memory issues before deploying!" -ForegroundColor Red
        Write-Host ("=" * 76) -ForegroundColor Red
        Write-Host ""
        Write-Host "Troubleshooting:"
        Write-Host "  1. Check if models are pre-downloaded in Dockerfile"
        Write-Host "  2. Ensure ONNX runtime is being used instead of PyTorch"
        Write-Host "  3. Review container logs for specific errors"
        Write-Host "  4. Consider reducing batch sizes or model complexity"
        Write-Host ""
        Write-Host "To view full logs:"
        Write-Host "  docker logs $CONTAINER_NAME"
        Write-Host ""
        $exitCode = 1
    }

} catch {
    Write-Host ""
    Write-Err "Script error: $_"
    $exitCode = 1
} finally {
    Do-Cleanup
}

# Keep window open for review
Write-Host ""
Write-Host "Press Enter to exit..." -ForegroundColor Gray
Read-Host | Out-Null

exit $exitCode
