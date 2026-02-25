@echo off
setlocal enabledelayedexpansion

:: Set console colors
color 0A

echo.
echo ============================================================
echo        PATTERN ANALYSIS SUITE - COMPLETE TOOLKIT
echo     Comprehensive Analysis & Visualization for Parquet Data
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
echo                  PATTERN ANALYSIS SUITE
echo ============================================================
echo.
echo   === ANALYSIS TOOLS ===
echo   [1] CONSOLIDATION ANALYZER - Statistical analysis
echo   [2] PATTERN DNA EXPLORER - Interactive sunburst explorer
echo   [3] INTERACTIVE DASHBOARD - Complete metrics dashboard
echo.
echo   === QUICK ACTIONS ===
echo   [4] RUN COMPLETE ANALYSIS - All tools on one file
echo   [5] BATCH PROCESS - Analyze multiple parquet files
echo   [6] VIEW LATEST RESULTS - Open recent analysis
echo.
echo   === UTILITIES ===
echo   [7] PARQUET VIEWER - Examine parquet file contents
echo   [8] CLEAN OUTPUT - Remove old analysis files
echo   [9] HELP - User guide and documentation
echo.
echo   [0] EXIT
echo.
echo ============================================================
echo.

set /p choice="Select option [0-9]: "

if "%choice%"=="1" goto CONSOLIDATION_ANALYZER
if "%choice%"=="2" goto DNA_EXPLORER
if "%choice%"=="3" goto INTERACTIVE_DASHBOARD
if "%choice%"=="4" goto COMPLETE_ANALYSIS
if "%choice%"=="5" goto BATCH_PROCESS
if "%choice%"=="6" goto VIEW_RESULTS
if "%choice%"=="7" goto PARQUET_VIEWER
if "%choice%"=="8" goto CLEAN_OUTPUT
if "%choice%"=="9" goto HELP
if "%choice%"=="0" goto EXIT_SCRIPT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 >nul
goto MAIN_MENU

:CONSOLIDATION_ANALYZER
cls
echo.
echo ============================================================
echo           CONSOLIDATION PATTERN ANALYZER
echo ============================================================
echo.
echo This tool provides:
echo   - Method effectiveness comparison
echo   - Method agreement analysis
echo   - Best pattern identification
echo   - Temporal pattern analysis
echo   - Risk/reward metrics
echo.
echo Available parquet files:
echo.
dir /b *.parquet 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] No parquet files found!
)
echo.
set /p filename="Enter parquet filename (or 'back' to return): "

if /i "%filename%"=="back" goto MAIN_MENU

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto CONSOLIDATION_ANALYZER
)

echo.
echo [RUNNING] Analyzing consolidation patterns...
python analyze_consolidation_parquet.py "%filename%"

echo.
echo [COMPLETE] Analysis saved to analysis_output folder
pause
goto MAIN_MENU

:DNA_EXPLORER
cls
echo.
echo ============================================================
echo             PATTERN DNA EXPLORER
echo ============================================================
echo.
echo This tool creates:
echo   - Interactive sunburst visualization
echo   - Success recipe analysis
echo   - Feature combination insights
echo   - Hierarchical pattern exploration
echo.
echo Available parquet files:
echo.
dir /b *.parquet 2>nul
echo.
set /p filename="Enter parquet filename (or 'back' to return): "

if /i "%filename%"=="back" goto MAIN_MENU

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto DNA_EXPLORER
)

echo.
echo Select visualization type:
echo   [1] Complete DNA Explorer (all features)
echo   [2] Simple Sunburst only
echo   [3] Success Recipe Analysis
echo.
set /p viz_type="Select [1-3]: "

echo.
echo [RUNNING] Creating Pattern DNA visualization...

if "%viz_type%"=="1" (
    python pattern_dna_explorer_fixed.py "%filename%" --output "pattern_dna_complete.html"
)
if "%viz_type%"=="2" (
    python pattern_dna_explorer.py "%filename%" --output-dir "." --auto-open
)
if "%viz_type%"=="3" (
    python pattern_dna_explorer.py "%filename%" --output-dir "."
)

echo.
echo [COMPLETE] Visualization created and opened in browser
pause
goto MAIN_MENU

:INTERACTIVE_DASHBOARD
cls
echo.
echo ============================================================
echo            INTERACTIVE DASHBOARD
echo ============================================================
echo.
echo This creates a comprehensive dashboard with:
echo   - 12 interactive charts
echo   - Method comparisons
echo   - Risk/reward analysis
echo   - Temporal patterns
echo   - Performance metrics
echo.
echo Available parquet files:
echo.
dir /b *.parquet 2>nul
echo.
set /p filename="Enter parquet filename (or 'back' to return): "

if /i "%filename%"=="back" goto MAIN_MENU

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto INTERACTIVE_DASHBOARD
)

echo.
echo [RUNNING] Creating interactive dashboard...
python consolidation_dashboard.py "%filename%" --auto-open

echo.
echo [COMPLETE] Dashboard created and opened in browser
pause
goto MAIN_MENU

:COMPLETE_ANALYSIS
cls
echo.
echo ============================================================
echo             COMPLETE ANALYSIS PIPELINE
echo ============================================================
echo.
echo This will run ALL analysis tools on a single file:
echo   1. Consolidation statistical analysis
echo   2. Pattern DNA Explorer
echo   3. Interactive Dashboard
echo   4. Visualization suite
echo.
echo Available parquet files:
echo.
dir /b *.parquet 2>nul
echo.
set /p filename="Enter parquet filename (or 'back' to return): "

if /i "%filename%"=="back" goto MAIN_MENU

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto COMPLETE_ANALYSIS
)

:: Create timestamp for output folder
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"
set "output_dir=complete_analysis_%timestamp%"

echo.
echo Creating output directory: %output_dir%
mkdir "%output_dir%" 2>nul

echo.
echo [1/4] Running consolidation analysis...
python analyze_consolidation_parquet.py "%filename%" --output-dir "%output_dir%\consolidation"

echo.
echo [2/4] Creating Pattern DNA Explorer...
python pattern_dna_explorer_fixed.py "%filename%" --output "%output_dir%\pattern_dna.html"

echo.
echo [3/4] Building Interactive Dashboard...
python consolidation_dashboard.py "%filename%" --output "%output_dir%\dashboard.html"

echo.
echo [4/4] Generating visualizations...
python visualize_consolidation_parquet.py "%filename%" --output-dir "%output_dir%\visualizations" --interactive

echo.
echo ============================================================
echo            COMPLETE ANALYSIS FINISHED!
echo ============================================================
echo.
echo Results saved to: %output_dir%
echo.
echo Files created:
echo   - Consolidation analysis reports (CSV, JSON)
echo   - Pattern DNA Explorer (HTML)
echo   - Interactive Dashboard (HTML)
echo   - Visualization charts (PNG, HTML)
echo.

set /p open="Open results folder? (Y/N): "
if /i "%open%"=="Y" start "" "%output_dir%"

pause
goto MAIN_MENU

:BATCH_PROCESS
cls
echo.
echo ============================================================
echo              BATCH PROCESS MULTIPLE FILES
echo ============================================================
echo.
echo Process all parquet files in current directory
echo.

set file_count=0
for %%f in (*.parquet) do set /a file_count+=1

if %file_count%==0 (
    echo [ERROR] No parquet files found!
    pause
    goto MAIN_MENU
)

echo Found %file_count% parquet files to process
echo.
echo Select analysis type:
echo   [1] Consolidation Analysis only
echo   [2] Pattern DNA Explorer only
echo   [3] Interactive Dashboard only
echo   [4] Complete Analysis (all tools)
echo.
set /p batch_type="Select [1-4]: "

echo.
set /p confirm="Process %file_count% files? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
for %%f in (*.parquet) do (
    echo.
    echo Processing: %%f
    echo ========================================

    if "%batch_type%"=="1" (
        python analyze_consolidation_parquet.py "%%f" --output-dir "batch_output\%%~nf\consolidation"
    )
    if "%batch_type%"=="2" (
        python pattern_dna_explorer_fixed.py "%%f" --output "batch_output\%%~nf\pattern_dna.html"
    )
    if "%batch_type%"=="3" (
        python consolidation_dashboard.py "%%f" --output "batch_output\%%~nf\dashboard.html"
    )
    if "%batch_type%"=="4" (
        python analyze_consolidation_parquet.py "%%f" --output-dir "batch_output\%%~nf\consolidation"
        python pattern_dna_explorer_fixed.py "%%f" --output "batch_output\%%~nf\pattern_dna.html"
        python consolidation_dashboard.py "%%f" --output "batch_output\%%~nf\dashboard.html"
    )
)

echo.
echo [COMPLETE] Batch processing finished!
echo Results saved to: batch_output\
pause
goto MAIN_MENU

:VIEW_RESULTS
cls
echo.
echo ============================================================
echo              VIEW LATEST RESULTS
echo ============================================================
echo.

echo Select result type to view:
echo   [1] Latest Dashboard (HTML)
echo   [2] Latest Pattern DNA (HTML)
echo   [3] Latest Analysis Report (JSON)
echo   [4] Latest CSV Results
echo   [5] Open output folder
echo.
set /p view_type="Select [1-5]: "

if "%view_type%"=="1" (
    for /f "delims=" %%f in ('dir /b /o-d *dashboard*.html 2^>nul') do (
        start "" "%%f"
        goto view_done
    )
    echo No dashboard files found!
)

if "%view_type%"=="2" (
    for /f "delims=" %%f in ('dir /b /o-d pattern_dna*.html 2^>nul') do (
        start "" "%%f"
        goto view_done
    )
    echo No Pattern DNA files found!
)

if "%view_type%"=="3" (
    for /f "delims=" %%f in ('dir /b /o-d analysis_output\*.json 2^>nul') do (
        start "" "analysis_output\%%f"
        goto view_done
    )
    echo No JSON reports found!
)

if "%view_type%"=="4" (
    for /f "delims=" %%f in ('dir /b /o-d analysis_output\*.csv 2^>nul') do (
        start "" "analysis_output\%%f"
        goto view_done
    )
    echo No CSV files found!
)

if "%view_type%"=="5" (
    if exist analysis_output (
        start "" "analysis_output"
    ) else (
        echo Output folder not found!
    )
)

:view_done
pause
goto MAIN_MENU

:PARQUET_VIEWER
cls
echo.
echo ============================================================
echo              PARQUET FILE VIEWER
echo ============================================================
echo.
echo Available parquet files:
echo.
dir /b *.parquet 2>nul
echo.
set /p filename="Enter parquet filename to examine (or 'back'): "

if /i "%filename%"=="back" goto MAIN_MENU

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto PARQUET_VIEWER
)

echo.
python -c "import pandas as pd; df = pd.read_parquet('%filename%'); print('Shape:', df.shape); print('\nColumns:', list(df.columns)); print('\nFirst 5 rows:'); print(df.head()); print('\nData types:'); print(df.dtypes); print('\nSummary statistics:'); print(df.describe())"

echo.
pause
goto MAIN_MENU

:CLEAN_OUTPUT
cls
echo.
echo ============================================================
echo              CLEAN OUTPUT FILES
echo ============================================================
echo.
echo This will remove:
echo   - analysis_output folder
echo   - batch_output folder
echo   - All .html files in current directory
echo   - Pattern CSV files
echo.
set /p confirm="Are you sure you want to clean output files? (Y/N): "

if /i "%confirm%"=="Y" (
    echo.
    echo Cleaning files...

    if exist analysis_output (
        rmdir /s /q analysis_output
        echo [OK] Removed analysis_output
    )

    if exist batch_output (
        rmdir /s /q batch_output
        echo [OK] Removed batch_output
    )

    del /q *dashboard*.html 2>nul
    del /q pattern_dna*.html 2>nul
    del /q *analysis_report*.json 2>nul
    echo [OK] Removed HTML and JSON files

    echo.
    echo Cleanup complete!
) else (
    echo Cleanup cancelled.
)

pause
goto MAIN_MENU

:HELP
cls
echo.
echo ============================================================
echo           PATTERN ANALYSIS SUITE - HELP
echo ============================================================
echo.
echo === OVERVIEW ===
echo This suite provides comprehensive analysis tools for
echo consolidation patterns in parquet files from BigQuery.
echo.
echo === TOOLS DESCRIPTION ===
echo.
echo 1. CONSOLIDATION ANALYZER
echo    - Analyzes method effectiveness (Bollinger, Range, Volume, ATR)
echo    - Finds best performing patterns
echo    - Calculates risk/reward metrics
echo    - Outputs: JSON reports, CSV files
echo.
echo 2. PATTERN DNA EXPLORER
echo    - Interactive sunburst chart showing pattern hierarchy
echo    - Reveals successful feature combinations
echo    - Color-coded by performance (green=gains, red=losses)
echo    - Click segments to zoom and explore
echo.
echo 3. INTERACTIVE DASHBOARD
echo    - 12 comprehensive charts
echo    - Method comparisons and correlations
echo    - Temporal analysis
echo    - Real-time hover information
echo.
echo === WORKFLOW RECOMMENDATIONS ===
echo.
echo For single file analysis:
echo   - Use option 4 (Complete Analysis) for full insights
echo.
echo For multiple files:
echo   - Use option 5 (Batch Process) to analyze all at once
echo.
echo === FILE REQUIREMENTS ===
echo.
echo Parquet files must contain:
echo   - ticker, date, price/close, volume
echo   - method1_bollinger, method2_range_based, etc.
echo   - max_gain_20d, max_gain_40d
echo   - bbw, volume_ratio, atr_pct (optional)
echo.
echo === OUTPUT FILES ===
echo.
echo - HTML files: Open in browser for interactive viewing
echo - CSV files: Open in Excel for data manipulation
echo - JSON files: Complete analysis data
echo.
pause
goto MAIN_MENU

:EXIT_SCRIPT
cls
echo.
echo ============================================================
echo      Thank you for using Pattern Analysis Suite!
echo ============================================================
echo.
echo Your analysis results are saved in:
echo   - analysis_output\ (individual analyses)
echo   - batch_output\ (batch processing)
echo   - HTML files (interactive visualizations)
echo.
timeout /t 3 >nul
exit /b 0