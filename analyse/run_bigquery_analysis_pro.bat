@echo off
setlocal enabledelayedexpansion

:: Advanced BigQuery Analysis Tool
:: Version 2.0 - Professional Edition

:: Set console properties
color 0A
title BigQuery Analysis Pro - 3000+ Tickers with Storage API
mode con: cols=100 lines=40

:: Configuration
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json
set PROJECT_ID=ignition-ki-csv-storage
set BUCKET_NAME=ignition-ki-csv-data-2025-user123

:: Create timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"

:STARTUP
cls
echo.
echo    ____  _       ____
echo   / __ )(_)___ _/ __ \__  _____  _______  __
echo  / __  / / __ `/ / / / / / / _ \/ ___/ / / /
echo / /_/ / / /_/ / /_/ / /_/ /  __/ /  / /_/ /
echo /_____/_/\__, /\___\_\__,_/\___/_/   \__, /
echo           /_/                       /____/
echo.
echo         MARKET CONSOLIDATION ANALYZER v2.0
echo         Powered by BigQuery Storage API
echo.
echo ================================================================================
echo.

:: System check
echo [SYSTEM CHECK]
echo --------------
echo Checking requirements...

:: Python check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo   [X] Python not found
    echo.
    echo Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [+] Python %PYTHON_VERSION% found

:: Credentials check
if exist "%GOOGLE_APPLICATION_CREDENTIALS%" (
    echo   [+] GCP credentials found
    :: Extract service account email
    for /f "tokens=2 delims=:," %%a in ('type "%GOOGLE_APPLICATION_CREDENTIALS%" ^| findstr "client_email"') do (
        set service_account=%%a
        set service_account=!service_account:"=!
        set service_account=!service_account: =!
    )
    echo       Service Account: !service_account!
) else (
    color 0C
    echo   [X] GCP credentials not found
    echo       Expected: %GOOGLE_APPLICATION_CREDENTIALS%
    pause
    exit /b 1
)

:: BigQuery module check
python -c "import google.cloud.bigquery" 2>nul
if %errorlevel% equ 0 (
    echo   [+] BigQuery SDK installed
) else (
    echo   [!] BigQuery SDK not installed
    set /p install="Install now? (Y/N): "
    if /i "!install!"=="Y" (
        pip install google-cloud-bigquery google-cloud-bigquery-storage
    )
)

:: Storage API check
python -c "from google.cloud import bigquery_storage" 2>nul
if %errorlevel% equ 0 (
    echo   [+] BigQuery Storage API available
) else (
    echo   [!] Storage API not available (will use REST fallback)
)

echo.
echo System check complete. Press any key to continue...
pause >nul

:MAIN_MENU
cls
echo.
echo ================================================================================
echo                         BIGQUERY ANALYSIS PRO
echo ================================================================================
echo.
echo   Project: %PROJECT_ID%
echo   Bucket: %BUCKET_NAME%
echo   Time: %date% %time%
echo.
echo --------------------------------------------------------------------------------
echo.
echo   [1] EXPRESS ANALYSIS
echo       - Quick 100 ticker test
echo       - Verify setup works
echo.
echo   [2] FULL MARKET SCAN
echo       - Process ALL tickers
echo       - Complete consolidation analysis
echo.
echo   [3] INTELLIGENT BATCH
echo       - Smart batching for large datasets
echo       - Automatic quota management
echo.
echo   [4] CUSTOM ANALYSIS
echo       - Date range filtering
echo       - Ticker selection
echo       - Advanced options
echo.
echo   [5] REAL-TIME MONITOR
echo       - Live processing status
echo       - Quota tracking
echo.
echo   [6] RESULTS VIEWER
echo       - Analyze output files
echo       - Generate reports
echo.
echo   [7] PERFORMANCE TEST
echo       - Benchmark speeds
echo       - Compare methods
echo.
echo   [8] MAINTENANCE
echo       - Check permissions
echo       - Clean cache
echo       - Update dependencies
echo.
echo   [9] DOCUMENTATION
echo.
echo   [0] EXIT
echo.
echo ================================================================================
echo.

set /p choice="Select option [0-9]: "

if "%choice%"=="1" goto EXPRESS
if "%choice%"=="2" goto FULL_SCAN
if "%choice%"=="3" goto INTELLIGENT_BATCH
if "%choice%"=="4" goto CUSTOM_ANALYSIS
if "%choice%"=="5" goto MONITOR
if "%choice%"=="6" goto RESULTS
if "%choice%"=="7" goto PERFORMANCE
if "%choice%"=="8" goto MAINTENANCE
if "%choice%"=="9" goto DOCS
if "%choice%"=="0" goto SAFE_EXIT

echo [ERROR] Invalid selection
timeout /t 2 >nul
goto MAIN_MENU

:EXPRESS
cls
echo.
echo ================================================================================
echo                        EXPRESS ANALYSIS
echo ================================================================================
echo.
echo Running quick test with 100 tickers...
echo.
echo Expected duration: 5-10 seconds
echo Quota usage: < 0.01 GB
echo.

python process_all_tickers.py --test-mode

if %errorlevel% equ 0 (
    echo.
    color 0A
    echo [SUCCESS] Express analysis completed!

    :: Show quick stats
    for /f "delims=" %%i in ('dir /b /od all_tickers_analysis_*.parquet 2^>nul') do set result=%%i
    if defined result (
        echo.
        echo Results saved to: !result!
        python -c "import pyarrow.parquet as pq; t=pq.read_table('!result!'); print(f'Total rows: {t.num_rows:,}')" 2>nul
    )
) else (
    color 0C
    echo [ERROR] Analysis failed
)

echo.
pause
goto MAIN_MENU

:FULL_SCAN
cls
echo.
echo ================================================================================
echo                      FULL MARKET SCAN
echo ================================================================================
echo.

:: Get ticker count
echo Scanning available tickers...
python -c "import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'%GOOGLE_APPLICATION_CREDENTIALS%'; from cloud_market_analysis import CloudMarketAnalyzer; a=CloudMarketAnalyzer('%PROJECT_ID%', r'%GOOGLE_APPLICATION_CREDENTIALS%'); tickers=a.load_tickers_from_gcs(); print(f'\nFound {len(tickers)} tickers in GCS'); print(f'Estimated processing time: {len(tickers)//20} seconds'); print(f'Estimated quota usage: {len(tickers)*0.00007:.3f} GB')" 2>nul

echo.
echo --------------------------------------------------------------------------------
echo.
echo Processing options:
echo   [1] Process all at once (fastest)
echo   [2] Process in batches of 500
echo   [3] Process in batches of 1000
echo   [B] Back to menu
echo.

set /p scan_choice="Select option: "

if /i "%scan_choice%"=="B" goto MAIN_MENU

set batch_size=0
if "%scan_choice%"=="2" set batch_size=500
if "%scan_choice%"=="3" set batch_size=1000

echo.
color 0E
echo [WARNING] This will process ALL available tickers
color 0A
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [PROCESSING] Starting full market scan at %time%
echo.

if "%batch_size%"=="0" (
    python process_all_tickers.py
) else (
    python process_all_tickers.py --batch-size %batch_size%
)

if %errorlevel% equ 0 (
    echo.
    echo [COMPLETE] Full scan finished at %time%

    :: Generate summary
    echo.
    echo Generating analysis summary...
    python -c "import pandas as pd; import glob; files=glob.glob('all_tickers_analysis_*.parquet'); df=pd.read_parquet(files[-1]) if files else None; print(f'Tickers analyzed: {df[\"ticker\"].nunique()}' if df is not None else 'No data'); print(f'Total data points: {len(df):,}' if df is not None else 'No data'); print(f'Consolidations found: {df[\"consolidation\"].sum():,}' if df is not None and \"consolidation\" in df.columns else 'N/A')" 2>nul
)

echo.
pause
goto MAIN_MENU

:INTELLIGENT_BATCH
cls
echo.
echo ================================================================================
echo                    INTELLIGENT BATCH PROCESSING
echo ================================================================================
echo.
echo This mode automatically:
echo   - Monitors quota usage in real-time
echo   - Adjusts batch sizes dynamically
echo   - Switches between BigQuery and local processing
echo   - Retries failed tickers
echo.

set /p max_tickers="Maximum tickers to process (Enter for all): "
if not defined max_tickers set max_tickers=99999

echo.
echo Starting intelligent batch processing...
echo.

:: Create monitoring script
echo import os > temp_monitor.py
echo import time >> temp_monitor.py
echo import sys >> temp_monitor.py
echo os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'%GOOGLE_APPLICATION_CREDENTIALS%' >> temp_monitor.py
echo from cloud_market_analysis import CloudMarketAnalyzer >> temp_monitor.py
echo analyzer = CloudMarketAnalyzer('%PROJECT_ID%', r'%GOOGLE_APPLICATION_CREDENTIALS%') >> temp_monitor.py
echo tickers = analyzer.load_tickers_from_gcs(num_stocks=%max_tickers%) >> temp_monitor.py
echo print(f'Processing {len(tickers)} tickers intelligently...') >> temp_monitor.py
echo. >> temp_monitor.py
echo batch_size = 100 >> temp_monitor.py
echo for i in range(0, len(tickers), batch_size): >> temp_monitor.py
echo     batch = tickers[i:i+batch_size] >> temp_monitor.py
echo     print(f'\nBatch {i//batch_size + 1}: Processing {len(batch)} tickers') >> temp_monitor.py
echo     usage = analyzer.get_usage_report() >> temp_monitor.py
echo     print(f'Quota used: {usage["bigquery"]["percentage_used"]:.1f}%%') >> temp_monitor.py
echo     if usage["bigquery"]["percentage_used"] ^> 80: >> temp_monitor.py
echo         print('Approaching quota limit, switching to local processing') >> temp_monitor.py
echo         results = analyzer.process_locally_optimized(batch) >> temp_monitor.py
echo     else: >> temp_monitor.py
echo         results = analyzer.run_hybrid_analysis(batch) >> temp_monitor.py
echo     time.sleep(0.5) >> temp_monitor.py

python temp_monitor.py
del temp_monitor.py

echo.
pause
goto MAIN_MENU

:CUSTOM_ANALYSIS
cls
echo.
echo ================================================================================
echo                      CUSTOM ANALYSIS
echo ================================================================================
echo.
echo Configure your analysis parameters:
echo.

set /p start_date="Start date (YYYY-MM-DD, Enter for 2023-01-01): "
if not defined start_date set start_date=2023-01-01

set /p end_date="End date (YYYY-MM-DD, Enter for today): "
if not defined end_date set end_date=%date:~-4%-%date:~4,2%-%date:~7,2%

set /p ticker_limit="Number of tickers (Enter for all): "

set /p batch_size="Batch size (Enter for 500): "
if not defined batch_size set batch_size=500

echo.
echo Configuration:
echo   Date range: %start_date% to %end_date%
echo   Tickers: %ticker_limit%
echo   Batch size: %batch_size%
echo.

set /p confirm="Start analysis? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

set args=--start-date %start_date% --end-date %end_date% --batch-size %batch_size%
if defined ticker_limit set args=%args% --limit %ticker_limit%

echo.
python process_all_tickers.py %args%

echo.
pause
goto MAIN_MENU

:MONITOR
cls
echo.
echo ================================================================================
echo                     REAL-TIME MONITOR
echo ================================================================================
echo.
echo Starting live monitoring dashboard...
echo Press Ctrl+C to stop
echo.

:: Create monitoring loop
:MONITOR_LOOP
cls
echo ================================================================================
echo                  LIVE MONITORING - %time%
echo ================================================================================
echo.

:: Show quota status
python -c "import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'%GOOGLE_APPLICATION_CREDENTIALS%'; from cloud_market_analysis import CloudMarketAnalyzer; a=CloudMarketAnalyzer('%PROJECT_ID%', r'%GOOGLE_APPLICATION_CREDENTIALS%'); u=a.get_usage_report(); print(f'Quota Usage: {u[\"bigquery\"][\"bytes_processed_today_gb\"]:.2f}/{u[\"bigquery\"][\"daily_limit_gb\"]:.0f} GB ({u[\"bigquery\"][\"percentage_used\"]:.1f}%%)')" 2>nul

echo.
echo Recent Files:
dir /b /od all_tickers_analysis_*.parquet 2>nul | tail -3

echo.
echo Active Processes:
tasklist | findstr python

echo.
echo Refreshing in 5 seconds... (Press Ctrl+C to exit)
timeout /t 5 >nul
goto MONITOR_LOOP

:RESULTS
cls
echo.
echo ================================================================================
echo                      RESULTS VIEWER
echo ================================================================================
echo.

:: List recent results
echo Recent Analysis Results:
echo ------------------------
for /f "delims=" %%f in ('dir /b /od all_tickers_analysis_*.parquet 2^>nul') do (
    set file=%%f
    for %%A in (%%f) do set size=%%~zA
    set /a size_mb=!size!/1048576
    echo   %%f (!size_mb! MB)
)

echo.
set /p result_file="Enter filename to analyze (or Enter for latest): "

if not defined result_file (
    for /f "delims=" %%i in ('dir /b /od all_tickers_analysis_*.parquet 2^>nul') do set result_file=%%i
)

if not exist "%result_file%" (
    echo [ERROR] File not found
    pause
    goto MAIN_MENU
)

echo.
echo Analyzing %result_file%...
echo.

python -c "import pandas as pd; df=pd.read_parquet('%result_file%'); print(f'Shape: {df.shape}'); print(f'Tickers: {df[\"ticker\"].nunique() if \"ticker\" in df else \"N/A\"}'); print(f'Date range: {df[\"date\"].min() if \"date\" in df else \"N/A\"} to {df[\"date\"].max() if \"date\" in df else \"N/A\"}'); print(f'\nColumns: {list(df.columns)}'); print(f'\nSample data:\n{df.head()}')" 2>nul

echo.
set /p export="Export summary to CSV? (Y/N): "
if /i "%export%"=="Y" (
    python -c "import pandas as pd; df=pd.read_parquet('%result_file%'); summary=df.groupby('ticker').agg({'consolidation': 'sum', 'price': ['min', 'max', 'mean']}) if 'consolidation' in df else df.describe(); summary.to_csv('summary_%timestamp%.csv'); print('Exported to summary_%timestamp%.csv')" 2>nul
)

echo.
pause
goto MAIN_MENU

:PERFORMANCE
cls
echo.
echo ================================================================================
echo                     PERFORMANCE TEST
echo ================================================================================
echo.
echo Running performance benchmarks...
echo.

echo Test 1: BigQuery with Storage API
echo ----------------------------------
powershell -Command "Measure-Command {python -c \"import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'%GOOGLE_APPLICATION_CREDENTIALS%'; from cloud_market_analysis import CloudMarketAnalyzer; a=CloudMarketAnalyzer('%PROJECT_ID%', r'%GOOGLE_APPLICATION_CREDENTIALS%'); a.process_with_bigquery(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], '2024-01-01', '2024-12-31')\" | Out-Default}"

echo.
echo Test 2: Local Processing
echo ------------------------
powershell -Command "Measure-Command {python -c \"import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'%GOOGLE_APPLICATION_CREDENTIALS%'; from cloud_market_analysis import CloudMarketAnalyzer; a=CloudMarketAnalyzer('%PROJECT_ID%', r'%GOOGLE_APPLICATION_CREDENTIALS%'); a.process_locally_optimized(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])\" | Out-Default}"

echo.
pause
goto MAIN_MENU

:MAINTENANCE
cls
echo.
echo ================================================================================
echo                       MAINTENANCE
echo ================================================================================
echo.
echo   [1] Check GCP Permissions
echo   [2] Update Dependencies
echo   [3] Clear Cache
echo   [4] Verify Data Integrity
echo   [5] Backup Results
echo   [B] Back
echo.

set /p maint_choice="Select option: "

if /i "%maint_choice%"=="B" goto MAIN_MENU

if "%maint_choice%"=="1" (
    echo.
    python check_gcp_permissions.py
    pause
    goto MAINTENANCE
)

if "%maint_choice%"=="2" (
    echo.
    echo Updating dependencies...
    pip install --upgrade google-cloud-bigquery google-cloud-bigquery-storage google-cloud-storage pandas pyarrow
    pause
    goto MAINTENANCE
)

if "%maint_choice%"=="3" (
    echo.
    echo Clearing cache...
    rmdir /s /q __pycache__ 2>nul
    del /q *.pyc 2>nul
    del /q *.tmp 2>nul
    echo Cache cleared
    pause
    goto MAINTENANCE
)

if "%maint_choice%"=="4" (
    echo.
    echo Verifying data integrity...
    for %%f in (all_tickers_analysis_*.parquet) do (
        echo Checking %%f...
        python -c "import pyarrow.parquet as pq; pq.read_table('%%f'); print('  [OK] %%f')" 2>nul || echo   [ERROR] %%f corrupted
    )
    pause
    goto MAINTENANCE
)

if "%maint_choice%"=="5" (
    echo.
    set backup_dir=backups_%timestamp%
    mkdir !backup_dir! 2>nul
    echo Backing up to !backup_dir!...
    copy all_tickers_analysis_*.parquet !backup_dir!\ >nul 2>&1
    copy processing_summary_*.json !backup_dir!\ >nul 2>&1
    echo Backup complete
    pause
    goto MAINTENANCE
)

goto MAINTENANCE

:DOCS
cls
echo.
echo ================================================================================
echo                      DOCUMENTATION
echo ================================================================================
echo.
echo BIGQUERY ANALYSIS SYSTEM
echo ------------------------
echo.
echo This system processes market data using Google BigQuery with Storage API
echo for ultra-fast analysis of thousands of tickers.
echo.
echo KEY FEATURES:
echo   * BigQuery Storage API for 10x faster data retrieval
echo   * Automatic fallback to REST API if needed
echo   * Smart batching to stay within free tier (1TB/month)
echo   * Hybrid cloud/local processing
echo   * Consolidation pattern detection
echo.
echo FREE TIER LIMITS:
echo   * Daily: 33 GB of data processing
echo   * Monthly: 1 TB total
echo   * ~3000 tickers = ~0.2 GB (well within limits)
echo.
echo OUTPUT FILES:
echo   * all_tickers_analysis_*.parquet - Main results
echo   * processing_summary_*.json - Processing metadata
echo   * process_all_tickers_*.log - Detailed logs
echo.
echo COMMON ISSUES:
echo   1. "Storage API not available" - Need to enable API in GCP Console
echo   2. "Permission denied" - Add BigQuery Read Session User role
echo   3. "Quota exceeded" - Wait until tomorrow or use local processing
echo.
pause
goto MAIN_MENU

:SAFE_EXIT
cls
echo.
echo ================================================================================
echo                    CLOSING BIGQUERY ANALYSIS
echo ================================================================================
echo.

:: Final quota check
echo Final quota status:
python -c "import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'%GOOGLE_APPLICATION_CREDENTIALS%'; from cloud_market_analysis import CloudMarketAnalyzer; a=CloudMarketAnalyzer('%PROJECT_ID%', r'%GOOGLE_APPLICATION_CREDENTIALS%'); u=a.get_usage_report(); print(f'Used today: {u[\"bigquery\"][\"bytes_processed_today_gb\"]:.2f} GB of {u[\"bigquery\"][\"daily_limit_gb\"]:.0f} GB')" 2>nul

echo.
echo Thank you for using BigQuery Analysis Pro!
echo.
timeout /t 3 >nul
exit /b 0