@echo off
title Market Consolidation Analysis Tool
color 0A

echo ============================================================
echo     COMPREHENSIVE MARKET CONSOLIDATION ANALYSIS TOOL
echo ============================================================
echo.
echo This tool analyzes stock market consolidation patterns using
echo 4 different detection methods on real GCS market data.
echo.
echo Minimum Price Filter: $0.01
echo Data Source: ignition-ki-csv-data-2025-user123
echo.
echo ============================================================
echo.
echo Select the number of stocks to analyze:
echo.
echo   [1] Quick Test    - 20 stocks
echo   [2] Small Sample  - 100 stocks  
echo   [3] Medium Sample - 500 stocks
echo   [4] Large Sample  - 1000 stocks
echo   [5] COMPLETE      - ALL stocks (2931+ tickers)
echo   [6] Custom Number - Enter your own number
echo   [Q] Quit
echo.
echo ============================================================
echo.

set /p choice="Enter your choice [1-6 or Q]: "

if /i "%choice%"=="Q" goto :quit

set num_stocks=0

if "%choice%"=="1" (
    set num_stocks=20
    echo.
    echo Selected: Quick Test - 20 stocks
) else if "%choice%"=="2" (
    set num_stocks=100
    echo.
    echo Selected: Small Sample - 100 stocks
) else if "%choice%"=="3" (
    set num_stocks=500
    echo.
    echo Selected: Medium Sample - 500 stocks
) else if "%choice%"=="4" (
    set num_stocks=1000
    echo.
    echo Selected: Large Sample - 1000 stocks
) else if "%choice%"=="5" (
    set num_stocks=0
    echo.
    echo Selected: COMPLETE ANALYSIS - ALL stocks
    echo WARNING: This may take several hours!
) else if "%choice%"=="6" (
    echo.
    set /p num_stocks="Enter number of stocks to analyze: "
    echo.
    echo Selected: Custom - %num_stocks% stocks
) else (
    echo.
    echo Invalid choice! Please try again.
    pause
    goto :restart
)

echo.
echo ============================================================
echo.
echo Select output format:
echo.
echo   [1] JSON Only (Recommended for large datasets)
echo   [2] Full Output (JSON + CSV + Excel)
echo.
set /p output_choice="Enter your choice [1-2]: "

set output_format=json_only
if "%output_choice%"=="2" (
    set output_format=full
    echo Selected: Full Output Format
) else (
    echo Selected: JSON Only Format
)

echo.
echo ============================================================
echo.
echo Configuration Summary:
echo   - Number of stocks: %num_stocks%
if "%num_stocks%"=="0" echo   - Analyzing ALL available stocks
echo   - Output format: %output_format%
echo   - Minimum price filter: $0.01
echo   - Data completeness threshold: 65%%
echo.
echo ============================================================
echo.

set /p confirm="Do you want to proceed? [Y/N]: "
if /i not "%confirm%"=="Y" goto :quit

echo.
echo Starting analysis...
echo.

REM Create a timestamp for unique filenames
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)
set mytime=%mytime: =0%
set timestamp=%mydate%_%mytime%

REM Run the Python script with parameters
if "%num_stocks%"=="0" (
    python run_market_analysis_configurable.py --complete --output-format %output_format% --timestamp %timestamp%
) else (
    python run_market_analysis_configurable.py --num-stocks %num_stocks% --output-format %output_format% --timestamp %timestamp%
)

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Analysis failed! Check the error messages above.
    echo ============================================================
    pause
    goto :quit
)

echo.
echo ============================================================
echo ANALYSIS COMPLETE!
echo.
echo Output files have been saved with timestamp: %timestamp%
echo.
if "%output_format%"=="json_only" (
    echo Files generated:
    echo   - market_consolidation_complete_%timestamp%.json
) else (
    echo Files generated:
    echo   - market_consolidation_complete_%timestamp%.json
    echo   - market_consolidation_full_data_%timestamp%.csv
    echo   - market_consolidation_summary_%timestamp%.xlsx
)
echo.
echo ============================================================
echo.
pause
goto :end

:restart
cls
call "%~f0"
goto :end

:quit
echo.
echo Exiting...
echo.
pause

:end