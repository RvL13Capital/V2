@echo off
REM =====================================================
REM AIv3 System - Master Control Panel
REM Consolidation Pattern Detection & Analysis System
REM =====================================================

color 0A
cls

:main_menu
echo.
echo ============================================================
echo                    AIv3 SYSTEM v3.0
echo            Pattern Detection for Explosive Moves
echo ============================================================
echo.
echo  Main Menu:
echo.
echo  [1] Quick Start (Detect + Analyze + Report)
echo  [2] Pattern Detection
echo  [3] Pattern Analysis
echo  [4] Backtesting
echo  [5] Generate Reports
echo  [6] System Configuration
echo  [7] Run Tests
echo  [8] Help / Documentation
echo  [9] Exit
echo.
echo ============================================================
echo.

choice /C 123456789 /N /M "Select option (1-9): "

if %ERRORLEVEL%==1 goto quick_start
if %ERRORLEVEL%==2 goto pattern_detection
if %ERRORLEVEL%==3 goto pattern_analysis
if %ERRORLEVEL%==4 goto backtesting
if %ERRORLEVEL%==5 goto reports
if %ERRORLEVEL%==6 goto configuration
if %ERRORLEVEL%==7 goto run_tests
if %ERRORLEVEL%==8 goto help
if %ERRORLEVEL%==9 goto exit_system

goto main_menu

:quick_start
cls
echo.
echo ============================================================
echo                    QUICK START MODE
echo ============================================================
echo.
echo This will:
echo  1. Detect patterns in top 50 tickers
echo  2. Analyze the patterns
echo  3. Generate a comprehensive report
echo.
choice /C YN /M "Continue with Quick Start?"
if %ERRORLEVEL%==2 goto main_menu

echo.
echo [Step 1/3] Detecting patterns...
call run_pattern_detection.bat --limit 50

echo.
echo [Step 2/3] Analyzing patterns...
call run_analysis.bat --type performance

echo.
echo [Step 3/3] Generating report...
python main.py report --input output\patterns_*.parquet --type pattern

echo.
echo Quick Start complete!
pause
goto main_menu

:pattern_detection
cls
echo.
echo ============================================================
echo                  PATTERN DETECTION
echo ============================================================
echo.
echo Select detection mode:
echo.
echo  [1] Quick Scan (10 tickers)
echo  [2] Standard Scan (100 tickers)
echo  [3] Full Market Scan (ALL tickers)
echo  [4] Custom Ticker List
echo  [5] Back to Main Menu
echo.

choice /C 12345 /N /M "Select option (1-5): "

if %ERRORLEVEL%==1 (
    call run_pattern_detection.bat --limit 10 --report
) else if %ERRORLEVEL%==2 (
    call run_pattern_detection.bat --limit 100 --report
) else if %ERRORLEVEL%==3 (
    echo WARNING: This will scan ALL tickers and may take a long time!
    choice /C YN /M "Continue?"
    if %ERRORLEVEL%==1 call run_pattern_detection.bat --tickers ALL
) else if %ERRORLEVEL%==4 (
    set /p TICKERS="Enter comma-separated ticker list: "
    call run_pattern_detection.bat --tickers %TICKERS% --report
) else (
    goto main_menu
)

pause
goto main_menu

:pattern_analysis
cls
echo.
echo ============================================================
echo                   PATTERN ANALYSIS
echo ============================================================
echo.
echo Select analysis type:
echo.
echo  [1] Statistical Analysis
echo  [2] Performance Analysis
echo  [3] Quality Analysis
echo  [4] Comprehensive Analysis (All types)
echo  [5] Back to Main Menu
echo.

choice /C 12345 /N /M "Select option (1-5): "

if %ERRORLEVEL%==5 goto main_menu

if %ERRORLEVEL%==1 set ANALYSIS_TYPE=statistical
if %ERRORLEVEL%==2 set ANALYSIS_TYPE=performance
if %ERRORLEVEL%==3 set ANALYSIS_TYPE=quality
if %ERRORLEVEL%==4 goto comprehensive_analysis

call run_analysis.bat --type %ANALYSIS_TYPE% --report
pause
goto main_menu

:comprehensive_analysis
echo.
echo Running comprehensive analysis...
call run_analysis.bat --type statistical
call run_analysis.bat --type performance
call run_analysis.bat --type quality
echo.
echo Comprehensive analysis complete!
pause
goto main_menu

:backtesting
cls
echo.
echo ============================================================
echo                     BACKTESTING
echo ============================================================
echo.
echo Select backtest period:
echo.
echo  [1] Last Month
echo  [2] Last 3 Months
echo  [3] Last 6 Months
echo  [4] Last Year
echo  [5] Custom Date Range
echo  [6] Back to Main Menu
echo.

choice /C 123456 /N /M "Select option (1-6): "

if %ERRORLEVEL%==6 goto main_menu

if %ERRORLEVEL%==1 (
    call run_backtest.bat --start 2024-08-01
) else if %ERRORLEVEL%==2 (
    call run_backtest.bat --start 2024-06-01
) else if %ERRORLEVEL%==3 (
    call run_backtest.bat --start 2024-03-01
) else if %ERRORLEVEL%==4 (
    call run_backtest.bat --start 2023-09-01
) else if %ERRORLEVEL%==5 (
    set /p START_DATE="Enter start date (YYYY-MM-DD): "
    set /p END_DATE="Enter end date (YYYY-MM-DD): "
    call run_backtest.bat --start %START_DATE% --end %END_DATE%
)

pause
goto main_menu

:reports
cls
echo.
echo ============================================================
echo                   REPORT GENERATION
echo ============================================================
echo.
echo Select report type:
echo.
echo  [1] Pattern Analysis Report
echo  [2] Backtest Results Report
echo  [3] Executive Summary
echo  [4] Back to Main Menu
echo.

choice /C 1234 /N /M "Select option (1-4): "

if %ERRORLEVEL%==4 goto main_menu

echo.
set /p INPUT_FILE="Enter input file path (or press Enter for latest): "

if "%INPUT_FILE%"=="" (
    for /f "delims=" %%i in ('dir /b /od output\patterns_*.parquet 2^>nul') do set "INPUT_FILE=output\%%i"
    if "%INPUT_FILE%"=="" set INPUT_FILE=historical_patterns.parquet
)

if %ERRORLEVEL%==1 (
    python main.py report --input "%INPUT_FILE%" --type pattern --title "AIv3 Pattern Analysis Report"
) else if %ERRORLEVEL%==2 (
    python main.py report --input "%INPUT_FILE%" --type backtest --title "AIv3 Backtest Results"
) else if %ERRORLEVEL%==3 (
    python main.py report --input "%INPUT_FILE%" --type analysis --title "AIv3 Executive Summary"
)

echo.
echo Report generated in ./reports folder
pause
goto main_menu

:configuration
cls
echo.
echo ============================================================
echo                 SYSTEM CONFIGURATION
echo ============================================================
echo.
echo  [1] View Current Configuration
echo  [2] Test GCS Connection
echo  [3] Setup Virtual Environment
echo  [4] Install/Update Dependencies
echo  [5] Back to Main Menu
echo.

choice /C 12345 /N /M "Select option (1-5): "

if %ERRORLEVEL%==1 (
    echo.
    python -c "from core import get_config; c=get_config(); print(f'GCS Project: {c.gcs.project_id}'); print(f'GCS Bucket: {c.gcs.bucket_name}'); print(f'Min Pattern Duration: {c.pattern.min_duration_days} days'); print(f'Explosive Threshold: {c.outcome.explosive_threshold}%%')"
    echo.
    pause
) else if %ERRORLEVEL%==2 (
    echo.
    python -c "from core import get_data_loader; loader=get_data_loader(); tickers=loader.list_available_tickers(); print(f'GCS Connection OK - Found {len(tickers)} tickers')"
    echo.
    pause
) else if %ERRORLEVEL%==3 (
    echo.
    if exist "setup_environment.bat" (
        call setup_environment.bat
    ) else (
        python -m venv venv
        call venv\Scripts\activate.bat
        pip install -r requirements.txt
    )
    pause
) else if %ERRORLEVEL%==4 (
    echo.
    call venv\Scripts\activate.bat
    pip install --upgrade -r requirements.txt
    pause
)

goto main_menu

:run_tests
cls
echo.
echo ============================================================
echo                    RUNNING TESTS
echo ============================================================
echo.

call venv\Scripts\activate.bat
python test_refactored.py

echo.
pause
goto main_menu

:help
cls
echo.
echo ============================================================
echo                 AIv3 SYSTEM HELP
echo ============================================================
echo.
echo OVERVIEW:
echo The AIv3 System detects consolidation patterns in stock
echo price data that often precede explosive moves (40%+ gains).
echo.
echo KEY CONCEPTS:
echo - Consolidation: Period of low volatility and tight range
echo - Pattern Duration: Typically 5-100 days
echo - Outcome Classes: K0-K5 based on gain percentage
echo - Expected Value: Strategic value calculation
echo.
echo WORKFLOW:
echo 1. Pattern Detection: Scan tickers for consolidations
echo 2. Analysis: Evaluate pattern quality and outcomes
echo 3. Backtesting: Test historical performance
echo 4. Reports: Generate comprehensive PDF reports
echo.
echo FILES:
echo - historical_patterns.parquet: Past pattern data
echo - output/: Detection results and reports
echo - core/: Refactored system modules
echo - main.py: Command-line interface
echo.
echo For more information, see CLAUDE.md
echo.
pause
goto main_menu

:exit_system
cls
echo.
echo ============================================================
echo.
echo   Thank you for using AIv3 System!
echo.
echo   Pattern Detection for Explosive Moves
echo.
echo ============================================================
echo.
timeout /t 2 >nul
exit /b 0