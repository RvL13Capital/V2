@echo off
REM =====================================================
REM AIv3 System - Install Dependencies
REM =====================================================

echo.
echo ========================================
echo AIv3 System - Dependency Installation
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing required dependencies...
echo.

REM Core dependencies (always needed)
echo [1/3] Installing core dependencies...
pip install pandas numpy pyarrow google-cloud-storage google-auth tqdm python-dateutil

echo.
echo [2/3] Installing visualization dependencies...
pip install plotly matplotlib scipy statsmodels

echo.
echo [3/3] Installing optional PDF generation dependencies...
echo Note: This may take a few minutes...
pip install reportlab

REM Try to install kaleido (may fail on some systems)
echo.
echo Attempting to install kaleido for image export...
pip install kaleido 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: kaleido installation failed (not critical)
    echo Some chart export features may not be available.
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.

REM Test imports
echo Testing installations...
python -c "import pandas; print('✓ pandas installed')"
python -c "import numpy; print('✓ numpy installed')"
python -c "import pyarrow; print('✓ pyarrow installed (parquet support available)')"
python -c "import plotly; print('✓ plotly installed')"
python -c "from google.cloud import storage; print('✓ google-cloud-storage installed')"
python -c "import reportlab; print('✓ reportlab installed (PDF generation available)')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ! reportlab not installed (PDF generation will not be available)
)

echo.
echo You can now run the AIv3 System!
echo Use: AIv3_System.bat or python main.py
echo.

pause