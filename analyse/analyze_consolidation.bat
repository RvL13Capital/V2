@echo off
setlocal enabledelayedexpansion

:: Set console colors
color 0A

echo.
echo ============================================================
echo      CONSOLIDATION PATTERN ANALYSIS FOR PARQUET FILES
echo          Analyze BigQuery consolidation results
echo ============================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

:MAIN_MENU
cls
echo.
echo ============================================================
echo            CONSOLIDATION ANALYSIS MENU
echo ============================================================
echo.
echo   [1] ANALYZE SINGLE PARQUET FILE
echo   [2] ANALYZE ALL PARQUET FILES
echo   [3] ANALYZE WITH CUSTOM PARAMETERS
echo   [4] VIEW LATEST RESULTS
echo   [5] GENERATE COMPARISON REPORT
echo   [0] EXIT
echo.
echo ============================================================
echo.

set /p choice="Select option [0-5]: "

if "%choice%"=="1" goto ANALYZE_SINGLE
if "%choice%"=="2" goto ANALYZE_ALL
if "%choice%"=="3" goto CUSTOM_PARAMS
if "%choice%"=="4" goto VIEW_RESULTS
if "%choice%"=="5" goto COMPARE_RESULTS
if "%choice%"=="0" goto EXIT_SCRIPT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 >nul
goto MAIN_MENU

:ANALYZE_SINGLE
cls
echo.
echo ============================================================
echo         ANALYZE SINGLE PARQUET FILE
echo ============================================================
echo.
echo Available parquet files:
echo.
dir /b *.parquet 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] No parquet files found in current directory!
)
echo.
set /p filename="Enter parquet filename: "

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto MAIN_MENU
)

echo.
echo [RUNNING] Analyzing %filename%...
python analyze_consolidation_parquet.py "%filename%"
echo.
pause
goto MAIN_MENU

:ANALYZE_ALL
cls
echo.
echo ============================================================
echo         ANALYZE ALL PARQUET FILES
echo ============================================================
echo.
echo Finding all parquet files...
echo.

for %%f in (*.parquet) do (
    echo.
    echo Processing: %%f
    echo ----------------------------------------
    python analyze_consolidation_parquet.py "%%f" --output-dir "analysis_output\%%~nf"
    echo.
)

echo.
echo [COMPLETE] All parquet files analyzed!
pause
goto MAIN_MENU

:CUSTOM_PARAMS
cls
echo.
echo ============================================================
echo        ANALYZE WITH CUSTOM PARAMETERS
echo ============================================================
echo.
set /p filename="Enter parquet filename: "

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto MAIN_MENU
)

echo.
echo Set custom thresholds:
echo.
set /p min_gain_20="Minimum 20-day gain percentage (default 20): "
if "%min_gain_20%"=="" set min_gain_20=20

set /p min_gain_40="Minimum 40-day gain percentage (default 30): "
if "%min_gain_40%"=="" set min_gain_40=30

set /p output_dir="Output directory (default analysis_output): "
if "%output_dir%"=="" set output_dir=analysis_output

echo.
echo [RUNNING] Analyzing with custom parameters...
echo   - 20-day min gain: %min_gain_20%%%
echo   - 40-day min gain: %min_gain_40%%%
echo   - Output: %output_dir%
echo.

python analyze_consolidation_parquet.py "%filename%" --min-gain-20d 0.%min_gain_20% --min-gain-40d 0.%min_gain_40% --output-dir "%output_dir%"

echo.
pause
goto MAIN_MENU

:VIEW_RESULTS
cls
echo.
echo ============================================================
echo             VIEW LATEST ANALYSIS RESULTS
echo ============================================================
echo.

:: Find the latest JSON report
set latest_report=
for /f "delims=" %%f in ('dir /b /o-d analysis_output\analysis_report_*.json 2^>nul') do (
    set latest_report=analysis_output\%%f
    goto :found_report
)

:found_report
if "%latest_report%"=="" (
    echo [ERROR] No analysis reports found!
    echo Run an analysis first.
    pause
    goto MAIN_MENU
)

echo Latest report: %latest_report%
echo.
echo Opening report...
type "%latest_report%" | more

echo.
echo.
echo [1] View best patterns CSV
echo [2] View method comparison CSV
echo [3] Return to menu
echo.
set /p view_choice="Select option: "

if "%view_choice%"=="1" (
    for /f "delims=" %%f in ('dir /b /o-d analysis_output\best_patterns_*.csv 2^>nul') do (
        start analysis_output\%%f
        goto MAIN_MENU
    )
)

if "%view_choice%"=="2" (
    for /f "delims=" %%f in ('dir /b /o-d analysis_output\method_comparison_*.csv 2^>nul') do (
        start analysis_output\%%f
        goto MAIN_MENU
    )
)

goto MAIN_MENU

:COMPARE_RESULTS
cls
echo.
echo ============================================================
echo          GENERATE COMPARISON REPORT
echo ============================================================
echo.

:: Create comparison Python script inline
echo import pandas as pd > compare_results.py
echo import json >> compare_results.py
echo from pathlib import Path >> compare_results.py
echo. >> compare_results.py
echo # Find all analysis reports >> compare_results.py
echo reports = list(Path('analysis_output').glob('analysis_report_*.json')) >> compare_results.py
echo. >> compare_results.py
echo if not reports: >> compare_results.py
echo     print("No reports found!") >> compare_results.py
echo     exit() >> compare_results.py
echo. >> compare_results.py
echo print(f"Found {len(reports)} reports\n") >> compare_results.py
echo. >> compare_results.py
echo # Load and compare >> compare_results.py
echo for report_path in reports: >> compare_results.py
echo     with open(report_path) as f: >> compare_results.py
echo         data = json.load(f) >> compare_results.py
echo     print(f"\n{report_path.name}:") >> compare_results.py
echo     print(f"  Total rows: {data['data_summary']['total_rows']}") >> compare_results.py
echo     print(f"  Best patterns: {data['best_patterns']['count']}") >> compare_results.py
echo     print(f"  Date range: {data['data_summary']['date_range']['start']} to {data['data_summary']['date_range']['end']}") >> compare_results.py
echo. >> compare_results.py
echo     # Print method effectiveness >> compare_results.py
echo     print("\n  Method Performance:") >> compare_results.py
echo     for method, stats in data['method_effectiveness'].items(): >> compare_results.py
echo         if stats['total_signals'] ^> 0: >> compare_results.py
echo             print(f"    {method}: {stats.get('avg_gain_40d', 0):.1f}%% (40d gain)") >> compare_results.py

python compare_results.py
del compare_results.py

echo.
pause
goto MAIN_MENU

:EXIT_SCRIPT
cls
echo.
echo ============================================================
echo   Thank you for using Consolidation Pattern Analysis!
echo ============================================================
echo.
echo Your analysis results are saved in: analysis_output\
echo.
timeout /t 3 >nul
exit /b 0