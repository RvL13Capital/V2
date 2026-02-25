@echo off
REM =====================================================
REM AIv3 System - Enhanced Pattern Detection with ML
REM Integrates core pattern detection with volume ML
REM =====================================================

echo.
echo ==========================================
echo AIv3 Enhanced Pattern Detection System
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_environment.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set default parameters
set TICKERS=ALL
set LIMIT=100
set OUTPUT_DIR=output
set USE_ML=true
set HIGH_CONFIDENCE=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto run_detection
if /i "%~1"=="--tickers" (
    set TICKERS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--limit" (
    set LIMIT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--no-ml" (
    set USE_ML=false
    shift
    goto parse_args
)
if /i "%~1"=="--high-confidence" (
    set HIGH_CONFIDENCE=true
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    goto show_help
)
shift
goto parse_args

:run_detection
echo Starting Enhanced Pattern Detection...
echo.
echo Configuration:
echo   Tickers: %TICKERS%
echo   Limit: %LIMIT%
echo   Use ML: %USE_ML%
echo   High Confidence Only: %HIGH_CONFIDENCE%
echo.

REM Create timestamp for output files
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"

REM Create output filenames
set PATTERNS_FILE=%OUTPUT_DIR%\patterns_%timestamp%.parquet
set ENHANCED_FILE=%OUTPUT_DIR%\enhanced_patterns_%timestamp%.parquet

REM Step 1: Run core pattern detection
echo [Step 1/3] Detecting consolidation patterns...
python main.py detect --tickers %TICKERS% --limit %LIMIT% --output %PATTERNS_FILE%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Pattern detection failed!
    pause
    exit /b %ERRORLEVEL%
)

REM Step 2: Apply ML enhancement if enabled
if "%USE_ML%"=="true" (
    echo.
    echo [Step 2/3] Applying ML enhancement...

    REM Check if AI Infra system exists
    if exist "AI Infra\integration_bridge.py" (
        cd "AI Infra"
        python integration_bridge.py --patterns "..\%PATTERNS_FILE%" --data "data\raw" --output "..\%ENHANCED_FILE%"
        cd ..

        if exist %ENHANCED_FILE% (
            set FINAL_FILE=%ENHANCED_FILE%
            echo ML enhancement complete!
        ) else (
            echo Warning: ML enhancement failed, using base patterns
            set FINAL_FILE=%PATTERNS_FILE%
        )
    ) else (
        echo Warning: AI Infra ML system not found, using base patterns
        set FINAL_FILE=%PATTERNS_FILE%
    )
) else (
    echo [Step 2/3] Skipping ML enhancement (disabled)
    set FINAL_FILE=%PATTERNS_FILE%
)

REM Step 3: Filter and summarize results
echo.
echo [Step 3/3] Processing results...

REM Apply high-confidence filter if requested
if "%HIGH_CONFIDENCE%"=="true" (
    echo Filtering for high-confidence patterns...
    python -c "import pandas as pd; df=pd.read_parquet('%FINAL_FILE%'); high=df[df['ml_probability']>=0.2] if 'ml_probability' in df.columns else df; high.to_parquet('high_conf_%FINAL_FILE%'); print(f'Filtered to {len(high)} high-confidence patterns')"
    set FINAL_FILE=high_conf_%FINAL_FILE%
)

echo.
echo ==========================================
echo Pattern Detection Complete!
echo Results saved to: %FINAL_FILE%
echo ==========================================
echo.

REM Display summary
python -c "import pandas as pd; df=pd.read_parquet('%FINAL_FILE%'); print(f'Total patterns: {len(df)}'); print('\nTop 10 patterns by ML probability:' if 'ml_probability' in df.columns else '\nTop 10 patterns:'); print(df.nlargest(10, 'ml_probability')[['ticker','start_date','ml_probability','ml_signal']] if 'ml_probability' in df.columns else df.head(10)[['ticker','start_date','duration_days']])"

echo.
REM Ask if user wants to run analysis
choice /C YN /M "Do you want to run detailed analysis?"
if %ERRORLEVEL%==1 (
    echo.
    call run_analysis.bat --input "%FINAL_FILE%"
)

pause
exit /b 0

:show_help
echo.
echo Usage: run_pattern_detection_enhanced.bat [options]
echo.
echo Options:
echo   --tickers [LIST]      Comma-separated ticker list or ALL (default: ALL)
echo   --limit [N]           Limit number of tickers to process (default: 100)
echo   --no-ml               Disable ML enhancement
echo   --high-confidence     Only keep high confidence patterns (ML prob >= 20%%)
echo   --help                Show this help message
echo.
echo Examples:
echo   run_pattern_detection_enhanced.bat
echo   run_pattern_detection_enhanced.bat --tickers AAPL,GOOGL,MSFT
echo   run_pattern_detection_enhanced.bat --limit 50 --high-confidence
echo   run_pattern_detection_enhanced.bat --no-ml --limit 100
echo.
echo This enhanced version integrates:
echo   1. Core AIv3 pattern detection (consolidation patterns)
echo   2. Volume-based ML predictions (34%% win rate on high confidence)
echo   3. Signal generation with confidence levels
echo.
pause
exit /b 0