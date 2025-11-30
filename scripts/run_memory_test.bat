@echo off
REM ============================================================================
REM Necthrall Lite - Memory Test Launcher for Windows
REM ============================================================================
REM Double-click this file to run the memory test
REM The window will stay open so you can see the results
REM ============================================================================

cd /d "%~dp0.."
echo Starting memory test...
echo.

REM Add Docker Desktop to PATH
set "PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin"
set "PATH=%PATH%;%LOCALAPPDATA%\Docker\wsl\docker-desktop\bin"
set "PATH=%PATH%;%ProgramFiles%\Docker\Docker\resources\bin"

REM Verify Docker is available
docker --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not available!
    echo.
    echo Please make sure:
    echo   1. Docker Desktop is installed
    echo   2. Docker Desktop is RUNNING ^(check system tray^)
    echo.
    echo If Docker Desktop is running, try running this from PowerShell instead:
    echo   .\scripts\test_memory_local.ps1
    echo.
    pause
    exit /b 1
)

echo Docker found: 
docker --version
echo.

REM Try common Git Bash locations
set "GITBASH="

if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" (
    set "GITBASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"
) else if exist "C:\Program Files\Git\bin\bash.exe" (
    set "GITBASH=C:\Program Files\Git\bin\bash.exe"
) else if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    set "GITBASH=C:\Program Files (x86)\Git\bin\bash.exe"
)

if defined GITBASH (
    echo Found Git Bash at: %GITBASH%
    echo.
    "%GITBASH%" --login -c "export PATH=\"$PATH:/c/Program Files/Docker/Docker/resources/bin\"; ./scripts/test_memory_local.sh; echo; echo 'Test complete. Press Enter to close...'; read"
) else (
    echo ERROR: Git Bash not found!
    echo.
    echo Please install Git for Windows from: https://git-scm.com/download/win
)

echo.
pause
