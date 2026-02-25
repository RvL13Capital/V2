@echo off
REM OPTIMIZED Full Pipeline - 5-10x Faster with Parallel Processing
REM Expected time: 30-90 minutes (vs 4-8 hours sequential)

echo ======================================================================
echo OPTIMIZED PARALLEL PIPELINE - 5-10x Faster
echo ======================================================================
echo.
echo Estimated time: 30-90 minutes (vs 4-8 hours sequential)
echo Using parallel processing with auto-detected worker count
echo.
echo Processing ALL tickers from GCS (3,548+)
echo.
echo Press Ctrl+C to cancel, or
pause

REM Activate virtual environment
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
echo OPTIMIZED PRODUCTION RUN START
echo ======================================================================
echo Start time: %date% %time%
echo.

echo.
echo ======================================================================
echo STEP 1: Parallel Pattern Detection + Feature Extraction
echo ======================================================================
echo This will:
echo   - Process ALL 3,548+ tickers in parallel
echo   - Auto-detect optimal worker count (typically 10-20)
echo   - Detect 10,000+ consolidation patterns
echo   - Extract 28+ volume features per pattern
echo.
echo Estimated time: 30-90 minutes (with parallel processing)
echo.

python extract_all_features_parallel.py --min-patterns 10000
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo X PRODUCTION FAILED - Parallel extraction failed
    echo ======================================================================
    echo Check log file: extract_features_parallel_*.log
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo STEP 2: Training Data Preparation
echo ======================================================================
echo.

REM Find most recent features file
for /f "delims=" %%i in ('dir /b /o-d "AI Infra\data\features\parallel_pattern_features_*.parquet" 2^>nul') do (
    set FEATURES_FILE=%%i
    goto :found_file
)

echo [ERROR] No features file found!
pause
exit /b 1

:found_file
echo Found features file: %FEATURES_FILE%
echo.

cd "AI Infra\hybrid_model"

REM Create temporary script
echo import sys > temp_prepare_prod.py
echo from pathlib import Path >> temp_prepare_prod.py
echo sys.path.insert(0, str(Path(__file__).parent)) >> temp_prepare_prod.py
echo from test_prepare_training import prepare_training_data >> temp_prepare_prod.py
echo. >> temp_prepare_prod.py
echo result = prepare_training_data( >> temp_prepare_prod.py
echo     features_file=Path("../data/features/%FEATURES_FILE%"), >> temp_prepare_prod.py
echo     output_dir=Path("../data/raw"), >> temp_prepare_prod.py
echo     test_mode=False >> temp_prepare_prod.py
echo ) >> temp_prepare_prod.py
echo. >> temp_prepare_prod.py
echo if not result or not result['success']: >> temp_prepare_prod.py
echo     sys.exit(1) >> temp_prepare_prod.py

python temp_prepare_prod.py
set PREP_EXIT=%errorlevel%
del temp_prepare_prod.py

if %PREP_EXIT% NEQ 0 (
    echo.
    echo ======================================================================
    echo X PRODUCTION FAILED - Training data preparation failed
    echo ======================================================================
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo.
echo ======================================================================
echo OPTIMIZED PIPELINE COMPLETE
echo ======================================================================
echo End time: %date% %time%
echo.
echo Output files:
echo   - Patterns: output\patterns_parallel_*.parquet
echo   - Features: AI Infra\data\features\parallel_pattern_features_*.parquet
echo   - Training: AI Infra\data\raw\production_training_data.parquet
echo   - Metadata: AI Infra\data\raw\production_training_metadata.json
echo   - Log:      extract_features_parallel_*.log
echo.
echo Performance: 5-10x faster than sequential version
echo.
echo Next steps:
echo   1. Review production_training_metadata.json
echo   2. Train model: cd AI Infra\hybrid_model ^&^& python integrated_self_training.py train
echo.
echo ======================================================================

pause
