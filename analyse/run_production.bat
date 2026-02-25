@echo off
REM Production Deployment Batch Script for Windows
REM Runs full consolidation analysis on all available GCS data

echo ========================================
echo   PRODUCTION ANALYSIS DEPLOYMENT
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo.
echo Installing required packages...
pip install -r requirements.txt --quiet

REM Set environment variables
echo.
echo Setting environment variables...
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json
set PROJECT_ID=ignition-ki-csv-storage
set GCS_BUCKET_NAME=ignition-ki-csv-data-2025-user123

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "production_analysis" mkdir production_analysis

REM Run production deployment
echo.
echo ========================================
echo   Starting Production Analysis...
echo ========================================
echo.

python deploy_full_analysis.py

REM Check if successful
if errorlevel 1 (
    echo.
    echo ========================================
    echo   DEPLOYMENT FAILED
    echo ========================================
    echo Check logs for details.
) else (
    echo.
    echo ========================================
    echo   DEPLOYMENT SUCCESSFUL
    echo ========================================
    echo Results saved in production_analysis folder
)

echo.
pause