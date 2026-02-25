@echo off
REM Test Pipeline - Validate feature extraction and training data preparation
REM Tests on 10 patterns before scaling to full GCS dataset

echo ======================================================================
echo TEST PIPELINE - Phase 1 Validation
echo ======================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    if exist "..\AI Infra\venv\Scripts\activate.bat" (
        echo Using AI Infra virtual environment...
        call "..\AI Infra\venv\Scripts\activate.bat"
    ) else (
        echo Warning: Virtual environment not found, using system Python
    )
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo ======================================================================
echo STEP 1: Feature Extraction (10 patterns)
echo ======================================================================
echo.

python test_feature_extraction.py
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo X TEST FAILED - Feature extraction failed
    echo ======================================================================
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo STEP 2: Training Data Preparation
echo ======================================================================
echo.

cd "AI Infra\hybrid_model"
python test_prepare_training.py
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo X TEST FAILED - Training data preparation failed
    echo ======================================================================
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo.
echo ======================================================================
echo TEST PIPELINE COMPLETE
echo ======================================================================
echo.
echo √ Phase 1 validation successful!
echo √ Feature extraction working
echo √ Training data preparation working
echo.
echo Next steps:
echo   1. Review test outputs in AI Infra\data\
echo   2. Run full pipeline: run_full_pipeline.bat
echo.
echo ======================================================================

pause
