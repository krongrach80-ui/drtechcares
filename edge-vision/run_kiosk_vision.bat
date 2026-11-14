@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo ============================================================================
echo   Dr.Tech.Care - Clinical 3D Edge Vision Server (RTMW3D-l + InsightFace)
echo ============================================================================

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required and must be on PATH.
    pause
    exit /b 1
  )
  echo [SETUP] Installing dependencies from requirements.txt...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

set "MODE=%~1"
if /I "%MODE%"=="--mock" (
  echo [MODE] Starting Edge Vision Server in SYNTHETIC MOCK VIDEO mode...
  set "MOCK_VIDEO_PATH=SYNTHETIC"
) else if /I "%MODE%"=="--video" (
  if "%~2"=="" (
    echo [ERROR] Missing video path: run_kiosk_vision.bat --video path\to\video.mp4
    exit /b 1
  )
  set "MOCK_VIDEO_PATH=%~2"
  echo [MODE] Starting Edge Vision Server using video file: !MOCK_VIDEO_PATH!
) else (
  echo [MODE] Starting Edge Vision Server with live physical webcam ^(1080p 60fps locked focus^)...
)

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
endlocal

