@echo off
setlocal enabledelayedexpansion

:: Set console colors and title
color 0A
title BigQuery Ticker Analysis - 3000+ Stocks

:: Set credentials path
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json

echo.
echo ================================================================================
echo           BIGQUERY MARKET CONSOLIDATION ANALYSIS
echo                    Powered by Storage API
echo ================================================================================
echo.
echo [INFO] Checking system requirements...

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

:: Check if credentials exist
if not exist "%GOOGLE_APPLICATION_CREDENTIALS%" (
    color 0C
    echo [ERROR] GCS credentials not found!
    echo Expected at: %GOOGLE_APPLICATION_CREDENTIALS%
    pause
    exit /b 1
)

:: Check required Python files
if not exist "process_all_tickers.py" (
    color 0C
    echo [ERROR] process_all_tickers.py not found!
    pause
    exit /b 1
)

echo [OK] All requirements met
echo.

:MAIN_MENU
cls
echo.
echo ================================================================================
echo                    BIGQUERY TICKER ANALYSIS MENU
echo ================================================================================
echo.
echo   [1] QUICK TEST - Process 100 tickers (Recommended first)
echo   [2] PROCESS ALL - Analyze all available tickers
echo   [3] CUSTOM RANGE - Specify number of tickers
echo   [4] DATE FILTER - Process with date range
echo   [5] CHECK PERMISSIONS - Verify GCP setup
echo   [6] VIEW LAST RESULTS - Open latest parquet file info
echo   [7] CLEAN UP - Remove old log files
echo   [0] EXIT
echo.
echo ================================================================================
echo.

set /p choice="Select option [0-7]: "

if "%choice%"=="1" goto QUICK_TEST
if "%choice%"=="2" goto PROCESS_ALL
if "%choice%"=="3" goto CUSTOM_RANGE
if "%choice%"=="4" goto DATE_FILTER
if "%choice%"=="5" goto CHECK_PERMISSIONS
if "%choice%"=="6" goto VIEW_RESULTS
if "%choice%"=="7" goto CLEANUP
if "%choice%"=="0" goto EXIT_SCRIPT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 >nul
goto MAIN_MENU

:QUICK_TEST
cls
echo.
echo ================================================================================
echo                         QUICK TEST MODE
echo ================================================================================
echo.
echo This will process the first 100 tickers as a test.
echo Estimated time: ~5-10 seconds
echo Quota usage: Minimal (< 0.01 GB)
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [STARTING] Processing 100 tickers in test mode...
echo.

python process_all_tickers.py --test-mode

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Test completed successfully!
    echo Check the generated parquet file for results.
) else (
    echo.
    echo [ERROR] Test failed. Check the log files for details.
)

echo.
pause
goto MAIN_MENU

:PROCESS_ALL
cls
echo.
echo ================================================================================
echo                     PROCESS ALL TICKERS
echo ================================================================================
echo.
echo This will process ALL available tickers from GCS.
echo.
echo [CHECKING] Getting ticker count...

:: Get ticker count first
python -c "import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'%GOOGLE_APPLICATION_CREDENTIALS%'; from cloud_market_analysis import CloudMarketAnalyzer; a=CloudMarketAnalyzer('ignition-ki-csv-storage', r'%GOOGLE_APPLICATION_CREDENTIALS%'); t=a.load_tickers_from_gcs(); print(f'Found {len(t)} tickers')" 2>nul

echo.
echo Estimated time: 1-5 minutes (depending on ticker count)
echo Quota usage: ~0.2-1 GB (within free tier)
echo.
color 0E
echo [WARNING] This will process a large amount of data!
color 0A
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [STARTING] Processing all tickers...
echo [INFO] This may take several minutes. Please wait...
echo.

python process_all_tickers.py

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] All tickers processed successfully!
    echo.

    :: Show latest parquet file
    for /f "delims=" %%i in ('dir /b /od all_tickers_analysis_*.parquet 2^>nul') do set latest_file=%%i
    if defined latest_file (
        echo Output saved to: !latest_file!

        :: Get file size
        for %%A in (!latest_file!) do set size=%%~zA
        set /a size_mb=!size!/1048576
        echo File size: !size_mb! MB
    )
) else (
    echo.
    echo [ERROR] Processing failed. Check the log files for details.
)

echo.
pause
goto MAIN_MENU

:CUSTOM_RANGE
cls
echo.
echo ================================================================================
echo                      CUSTOM TICKER RANGE
echo ================================================================================
echo.
echo Specify how many tickers to process.
echo.
set /p num_tickers="Enter number of tickers (e.g., 500): "

:: Validate input
echo %num_tickers%| findstr /r "^[0-9][0-9]*$" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Invalid number. Please enter a valid integer.
    timeout /t 2 >nul
    goto CUSTOM_RANGE
)

echo.
echo Will process %num_tickers% tickers
echo Estimated time: ~%num_tickers%/20 seconds
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [STARTING] Processing %num_tickers% tickers...
echo.

python process_all_tickers.py --limit %num_tickers%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Processing completed!
) else (
    echo.
    echo [ERROR] Processing failed. Check the log files.
)

echo.
pause
goto MAIN_MENU

:DATE_FILTER
cls
echo.
echo ================================================================================
echo                    DATE RANGE FILTERING
echo ================================================================================
echo.
echo Process tickers with a specific date range.
echo.
echo Enter dates in YYYY-MM-DD format (e.g., 2024-01-01)
echo.
set /p start_date="Start date: "
set /p end_date="End date: "
set /p num="Number of tickers (or press Enter for all): "

echo.
echo Configuration:
echo   Start: %start_date%
echo   End: %end_date%
if defined num (
    echo   Tickers: %num%
    set limit_arg=--limit %num%
) else (
    echo   Tickers: ALL
    set limit_arg=
)
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [STARTING] Processing with date filter...
echo.

python process_all_tickers.py --start-date %start_date% --end-date %end_date% %limit_arg%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Date-filtered processing completed!
) else (
    echo.
    echo [ERROR] Processing failed.
)

echo.
pause
goto MAIN_MENU

:CHECK_PERMISSIONS
cls
echo.
echo ================================================================================
echo                  GCP PERMISSIONS CHECK
echo ================================================================================
echo.
echo Checking BigQuery and Storage API permissions...
echo.

if exist check_gcp_permissions.py (
    python check_gcp_permissions.py
) else (
    echo [WARNING] Permission checker not found.
    echo.
    echo Running basic connection test...
    python -c "from google.cloud import bigquery, bigquery_storage; print('[OK] BigQuery modules loaded'); client = bigquery.Client(); print('[OK] BigQuery client created'); storage_client = bigquery_storage.BigQueryReadClient(); print('[OK] Storage API client created')" 2>&1
)

echo.
pause
goto MAIN_MENU

:VIEW_RESULTS
cls
echo.
echo ================================================================================
echo                      ANALYSIS RESULTS
echo ================================================================================
echo.
echo Recent analysis files:
echo.

:: List parquet files
echo PARQUET FILES:
echo --------------
dir /b /od all_tickers_analysis_*.parquet 2>nul
if %errorlevel% neq 0 (
    echo   No parquet files found
)

echo.
echo LOG FILES:
echo ----------
dir /b /od process_all_tickers_*.log 2>nul | tail -5 2>nul
if %errorlevel% neq 0 (
    dir /b /od process_all_tickers_*.log 2>nul
)

echo.
echo SUMMARY FILES:
echo --------------
dir /b /od processing_summary_*.json 2>nul | tail -5 2>nul
if %errorlevel% neq 0 (
    dir /b /od processing_summary_*.json 2>nul
)

echo.

:: Get latest parquet file info
for /f "delims=" %%i in ('dir /b /od all_tickers_analysis_*.parquet 2^>nul') do set latest_file=%%i
if defined latest_file (
    echo.
    echo LATEST RESULT: !latest_file!
    echo ----------------------------------------

    :: Show basic info
    for %%A in (!latest_file!) do (
        echo Size: %%~zA bytes
        echo Modified: %%~tA
    )

    :: Try to show row count
    python -c "import pyarrow.parquet as pq; t=pq.read_table('!latest_file!'); print(f'Rows: {t.num_rows:,}'); print(f'Columns: {t.num_columns}')" 2>nul
)

echo.
set /p view_choice="Open latest summary JSON? (Y/N): "
if /i "%view_choice%"=="Y" (
    for /f "delims=" %%i in ('dir /b /od processing_summary_*.json 2^>nul') do set latest_json=%%i
    if defined latest_json (
        type !latest_json!
    )
)

echo.
pause
goto MAIN_MENU

:CLEANUP
cls
echo.
echo ================================================================================
echo                         CLEANUP
echo ================================================================================
echo.
echo This will remove old log files and temporary files.
echo.
echo Files to be removed:
echo   - Log files older than 7 days
echo   - Temporary processing files
echo   - Cache files
echo.
color 0E
echo [WARNING] Parquet result files will NOT be deleted
color 0A
echo.
set /p confirm="Continue with cleanup? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [CLEANING] Removing old files...

:: Remove old log files (older than 7 days)
forfiles /m "process_all_tickers_*.log" /d -7 /c "cmd /c del @file" 2>nul
if %errorlevel% equ 0 (
    echo [OK] Old log files removed
) else (
    echo [INFO] No old log files to remove
)

:: Remove temp files
del /q *.tmp 2>nul
del /q *.temp 2>nul

:: Remove Python cache
if exist __pycache__ (
    rmdir /s /q __pycache__
    echo [OK] Python cache cleared
)

echo.
echo [COMPLETE] Cleanup finished
echo.
pause
goto MAIN_MENU

:EXIT_SCRIPT
cls
echo.
echo ================================================================================
echo           Thank you for using BigQuery Ticker Analysis!
echo ================================================================================
echo.
echo Your results are saved in parquet format.
echo Use Python/Pandas to analyze the data:
echo.
echo   import pandas as pd
echo   df = pd.read_parquet('all_tickers_analysis_*.parquet')
echo.
timeout /t 5 >nul
exit /b 0