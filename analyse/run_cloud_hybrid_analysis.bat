@echo off
setlocal enabledelayedexpansion

:: Set console colors
color 0A

echo.
echo ============================================================
echo     CLOUD-OPTIMIZED MARKET ANALYSIS WITH HYBRID MODE
echo     Intelligent Cloud/Local Processing with Free Tier Protection
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

:: Check if cloud_market_analysis.py exists
if not exist cloud_market_analysis.py (
    color 0C
    echo [ERROR] cloud_market_analysis.py not found!
    echo Please ensure the script is in the current directory
    pause
    exit /b 1
)

:: Check for required dependencies
python -c "import google.cloud.bigquery" 2>nul
if %errorlevel% neq 0 (
    color 0E
    echo [WARNING] Google Cloud dependencies not installed!
    echo.
    echo Please run: install_cloud_dependencies.bat
    echo Or manually install: pip install google-cloud-bigquery google-cloud-storage pyarrow pandas-gbq
    echo.
    set /p install_now="Install dependencies now? (Y/N): "
    if /i "!install_now!"=="Y" (
        if exist install_cloud_dependencies.bat (
            call install_cloud_dependencies.bat
        ) else (
            echo Installing dependencies...
            pip install google-cloud-bigquery google-cloud-storage pyarrow pandas-gbq
        )
    ) else (
        echo [INFO] Cloud features will not be available. Only local processing will work.
        timeout /t 3 >nul
    )
    color 0A
)

:MAIN_MENU
cls
echo.
echo ============================================================
echo                    MAIN MENU
echo ============================================================
echo.
echo   [1] QUICK ANALYSIS (Recommended for first-time users)
echo   [2] HYBRID MODE - Smart Cloud/Local (Best Performance)
echo   [3] CLOUD ONLY - Maximum Speed (Uses BigQuery Quota)
echo   [4] LOCAL ONLY - No Cloud Usage (Slower)
echo   [5] CHECK QUOTA STATUS
echo   [6] ADVANCED OPTIONS
echo   [7] HELP / DOCUMENTATION
echo   [0] EXIT
echo.
echo ============================================================
echo.

set /p main_choice="Select option [0-7]: "

if "%main_choice%"=="1" goto QUICK_ANALYSIS
if "%main_choice%"=="2" goto HYBRID_MODE
if "%main_choice%"=="3" goto CLOUD_ONLY
if "%main_choice%"=="4" goto LOCAL_ONLY
if "%main_choice%"=="5" goto CHECK_QUOTA
if "%main_choice%"=="6" goto ADVANCED_OPTIONS
if "%main_choice%"=="7" goto HELP
if "%main_choice%"=="0" goto EXIT_SCRIPT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 >nul
goto MAIN_MENU

:QUICK_ANALYSIS
cls
echo.
echo ============================================================
echo                    QUICK ANALYSIS
echo ============================================================
echo.
echo This will run a quick test with optimized settings:
echo   - 10 stocks for testing
echo   - Hybrid mode (cloud with local fallback)
echo   - Automatic quota management
echo.
echo [INFO] Estimated time: 1-2 minutes
echo [INFO] Quota usage: Minimal (~0.1 GB)
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [RUNNING] Starting quick analysis...
python cloud_market_analysis.py --num-stocks 10 --use-bigquery

echo.
echo [COMPLETE] Analysis finished!
pause
goto MAIN_MENU

:HYBRID_MODE
cls
echo.
echo ============================================================
echo              HYBRID MODE CONFIGURATION
echo ============================================================
echo.
echo Hybrid mode intelligently switches between cloud and local:
echo   - Uses BigQuery when within quota limits (FAST)
echo   - Automatically falls back to local if quota exceeded
echo   - Optimizes for best performance/cost ratio
echo.
echo Select dataset size:
echo   [1] Small   - 50 stocks  (~5 min, ~1 GB quota)
echo   [2] Medium  - 100 stocks (~10 min, ~3 GB quota)
echo   [3] Large   - 500 stocks (~30 min, ~15 GB quota)
echo   [4] XLarge  - 1000 stocks (~60 min, ~30 GB quota)
echo   [5] Custom  - Specify number
echo   [0] Back to main menu
echo.

set /p hybrid_choice="Select size [0-5]: "

if "%hybrid_choice%"=="0" goto MAIN_MENU
if "%hybrid_choice%"=="1" set num_stocks=50
if "%hybrid_choice%"=="2" set num_stocks=100
if "%hybrid_choice%"=="3" set num_stocks=500
if "%hybrid_choice%"=="4" set num_stocks=1000
if "%hybrid_choice%"=="5" (
    set /p num_stocks="Enter number of stocks: "
)

:: Calculate estimated quota usage
set /a estimated_gb=!num_stocks!*30/1000

echo.
echo ============================================================
echo                 ANALYSIS CONFIGURATION
echo ============================================================
echo.
echo   Stocks to analyze: !num_stocks!
echo   Mode: HYBRID (Cloud + Local Fallback)
echo   Estimated quota: ~!estimated_gb! GB
echo   Estimated time: ~!num_stocks!/20 minutes
echo.
echo [WARNING] Large datasets may take significant time
echo.

set /p confirm="Start analysis? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [RUNNING] Starting hybrid analysis for !num_stocks! stocks...
echo [INFO] Press Ctrl+C to cancel at any time
echo.

python cloud_market_analysis.py --num-stocks !num_stocks! --use-bigquery

echo.
echo [COMPLETE] Hybrid analysis finished!
echo.

:: Show usage after completion
echo Checking quota usage...
python cloud_market_analysis.py --check-usage

pause
goto MAIN_MENU

:CLOUD_ONLY
cls
echo.
echo ============================================================
echo               CLOUD ONLY MODE (FASTEST)
echo ============================================================
echo.
color 0E
echo [WARNING] This mode uses your BigQuery quota!
echo          Daily free limit: 33 GB
echo          Monthly free limit: 1 TB
color 0A
echo.
echo Benefits:
echo   + 10-100x faster than local processing
echo   + Can handle massive datasets
echo   + Parallel processing of all stocks
echo.
echo Limitations:
echo   - Uses quota (may incur costs if exceeded)
echo   - Requires internet connection
echo   - Stops if quota exceeded (no fallback)
echo.

:: First check current quota
echo Checking current quota status...
python cloud_market_analysis.py --check-usage
echo.

set /p continue="Continue with cloud-only mode? (Y/N): "
if /i not "%continue%"=="Y" goto MAIN_MENU

echo.
echo Select dataset size:
echo   [1] Small   - 50 stocks  (~1 GB quota)
echo   [2] Medium  - 100 stocks (~3 GB quota)
echo   [3] Large   - 500 stocks (~15 GB quota)
echo   [4] Maximum - Use remaining daily quota
echo   [0] Cancel
echo.

set /p cloud_choice="Select size [0-4]: "

if "%cloud_choice%"=="0" goto MAIN_MENU
if "%cloud_choice%"=="1" set num_stocks=50
if "%cloud_choice%"=="2" set num_stocks=100
if "%cloud_choice%"=="3" set num_stocks=500
if "%cloud_choice%"=="4" (
    echo [INFO] Will process maximum stocks within quota limits
    set num_stocks=1000
)

echo.
color 0E
echo [CONFIRM] This will use approximately !num_stocks!*0.03 GB of quota
color 0A
set /p final_confirm="Are you sure? (Y/N): "
if /i not "%final_confirm%"=="Y" goto MAIN_MENU

echo.
echo [RUNNING] Starting cloud-only analysis...
python cloud_market_analysis.py --num-stocks !num_stocks! --use-bigquery --cloud-only

echo.
echo [COMPLETE] Cloud analysis finished!
pause
goto MAIN_MENU

:LOCAL_ONLY
cls
echo.
echo ============================================================
echo              LOCAL ONLY MODE (NO QUOTA)
echo ============================================================
echo.
echo This mode processes everything locally:
echo   + No quota usage
echo   + Works offline
echo   + Predictable performance
echo.
echo   - Slower than cloud (10-100x)
echo   - Limited by local RAM
echo   - Sequential processing
echo.
echo Select dataset size:
echo   [1] Small   - 50 stocks  (~10 min)
echo   [2] Medium  - 100 stocks (~20 min)
echo   [3] Large   - 500 stocks (~90 min)
echo   [4] Custom  - Specify number
echo   [0] Cancel
echo.

set /p local_choice="Select size [0-4]: "

if "%local_choice%"=="0" goto MAIN_MENU
if "%local_choice%"=="1" set num_stocks=50
if "%local_choice%"=="2" set num_stocks=100
if "%local_choice%"=="3" set num_stocks=500
if "%local_choice%"=="4" (
    set /p num_stocks="Enter number of stocks: "
)

echo.
echo [INFO] Processing !num_stocks! stocks locally
echo [INFO] This may take a while. Please be patient...
echo.

set /p confirm="Start local analysis? (Y/N): "
if /i not "%confirm%"=="Y" goto MAIN_MENU

echo.
echo [RUNNING] Starting local-only analysis...
python cloud_market_analysis.py --num-stocks !num_stocks! --local-only

echo.
echo [COMPLETE] Local analysis finished!
pause
goto MAIN_MENU

:CHECK_QUOTA
cls
echo.
echo ============================================================
echo                  QUOTA STATUS CHECK
echo ============================================================
echo.
echo Checking your current BigQuery usage and limits...
echo.

python cloud_market_analysis.py --check-usage

echo.
echo ============================================================
echo                     QUOTA GUIDE
echo ============================================================
echo.
echo FREE TIER LIMITS:
echo   Daily:   33 GB  (resets every 24 hours)
echo   Monthly: 1 TB   (resets on the 1st)
echo.
echo TYPICAL USAGE:
echo   1 stock  = ~0.03 GB
echo   10 stocks = ~0.3 GB
echo   100 stocks = ~3 GB
echo   1000 stocks = ~30 GB
echo.
echo RECOMMENDATIONS:
echo   - Use hybrid mode to stay within limits
echo   - Process large datasets over multiple days
echo   - Monitor usage regularly
echo.
pause
goto MAIN_MENU

:ADVANCED_OPTIONS
cls
echo.
echo ============================================================
echo                  ADVANCED OPTIONS
echo ============================================================
echo.
echo   [1] Custom date range analysis
echo   [2] Specific ticker list
echo   [3] Export to BigQuery table
echo   [4] Configure processing parameters
echo   [5] Benchmark cloud vs local speed
echo   [6] Clear cache and temp files
echo   [0] Back to main menu
echo.

set /p adv_choice="Select option [0-6]: "

if "%adv_choice%"=="0" goto MAIN_MENU

if "%adv_choice%"=="1" (
    echo.
    set /p start_date="Enter start date (YYYY-MM-DD): "
    set /p end_date="Enter end date (YYYY-MM-DD): "
    set /p num="Enter number of stocks: "
    echo.
    echo [RUNNING] Date range analysis...
    python cloud_market_analysis.py --num-stocks !num! --start-date !start_date! --end-date !end_date! --use-bigquery
    pause
    goto ADVANCED_OPTIONS
)

if "%adv_choice%"=="2" (
    echo.
    echo Enter tickers separated by spaces (e.g., AAPL MSFT GOOGL AMZN):
    set /p tickers="Tickers: "
    echo.
    echo [RUNNING] Analyzing specific tickers...
    python cloud_market_analysis.py --tickers !tickers! --use-bigquery
    pause
    goto ADVANCED_OPTIONS
)

if "%adv_choice%"=="3" (
    echo.
    echo This will export results to BigQuery for SQL analysis
    set /p table="Enter table name: "
    set /p num="Enter number of stocks: "
    echo.
    echo [RUNNING] Exporting to BigQuery...
    python cloud_market_analysis.py --num-stocks !num! --use-bigquery --export-bq --table !table!
    pause
    goto ADVANCED_OPTIONS
)

if "%adv_choice%"=="4" (
    echo.
    echo Configuration options:
    echo   [1] Set chunk size (default: 50)
    echo   [2] Set parallel workers (default: 10)
    echo   [3] Set cache TTL (default: 15 min)
    echo.
    set /p config="Select configuration [1-3]: "

    if "!config!"=="1" (
        set /p chunk="Enter chunk size: "
        set CHUNK_SIZE=!chunk!
    )
    if "!config!"=="2" (
        set /p workers="Enter number of workers: "
        set WORKERS=!workers!
    )
    if "!config!"=="3" (
        set /p ttl="Enter cache TTL (minutes): "
        set CACHE_TTL=!ttl!
    )

    echo Configuration updated!
    pause
    goto ADVANCED_OPTIONS
)

if "%adv_choice%"=="5" (
    echo.
    echo [BENCHMARK] Running speed comparison...
    echo.
    echo Testing with 10 stocks...
    echo.
    echo Cloud processing:
    powershell -Command "Measure-Command {python cloud_market_analysis.py --num-stocks 10 --use-bigquery --cloud-only | Out-Default}"
    echo.
    echo Local processing:
    powershell -Command "Measure-Command {python cloud_market_analysis.py --num-stocks 10 --local-only | Out-Default}"
    echo.
    pause
    goto ADVANCED_OPTIONS
)

if "%adv_choice%"=="6" (
    echo.
    echo Clearing cache and temporary files...
    if exist __pycache__ rmdir /s /q __pycache__
    if exist *.tmp del /q *.tmp
    if exist .cache rmdir /s /q .cache
    echo Cache cleared!
    pause
    goto ADVANCED_OPTIONS
)

goto ADVANCED_OPTIONS

:HELP
cls
echo.
echo ============================================================
echo                 HELP / DOCUMENTATION
echo ============================================================
echo.
echo PROCESSING MODES:
echo.
echo 1. HYBRID MODE (Recommended)
echo    - Automatically uses cloud when possible
echo    - Falls back to local if quota exceeded
echo    - Best balance of speed and cost
echo.
echo 2. CLOUD ONLY
echo    - Maximum speed (10-100x faster)
echo    - Uses BigQuery quota
echo    - Stops if quota exceeded
echo.
echo 3. LOCAL ONLY
echo    - No quota usage
echo    - Works offline
echo    - Slower processing
echo.
echo QUOTA MANAGEMENT:
echo    - Free tier: 1TB/month, ~33GB/day
echo    - Check status regularly (Option 5)
echo    - Hybrid mode prevents overages
echo.
echo PERFORMANCE TIPS:
echo    - Start with small datasets to test
echo    - Use hybrid mode for large analyses
echo    - Process over multiple days if needed
echo    - Cloud is best for >100 stocks
echo.
echo FILES GENERATED:
echo    - cloud_analysis_results_*.parquet (main results)
echo    - market_consolidation_*.json (detailed JSON)
echo    - analysis_logs_*.txt (processing logs)
echo.
pause
goto MAIN_MENU

:EXIT_SCRIPT
cls
echo.
echo ============================================================
echo Thank you for using Cloud-Optimized Market Analysis!
echo ============================================================
echo.
echo Your results are saved in the current directory.
echo.
timeout /t 3 >nul
exit /b 0