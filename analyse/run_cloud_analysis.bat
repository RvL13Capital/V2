@echo off
echo ========================================
echo Cloud-Optimized Market Analysis Runner
echo ========================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Display menu
echo Select processing mode:
echo 1. Hybrid Mode (Cloud + Local fallback) - RECOMMENDED
echo 2. Cloud Only (BigQuery) - FASTEST but uses quota
echo 3. Local Only - No quota usage
echo 4. Check Cloud Usage/Quota
echo 5. Small Test (10 stocks)
echo 6. Medium Analysis (100 stocks)
echo 7. Large Analysis (500 stocks)
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Running HYBRID mode analysis...
    set /p num="Enter number of stocks to analyze: "
    python cloud_market_analysis.py --num-stocks %num% --use-bigquery

) else if "%choice%"=="2" (
    echo.
    echo Running CLOUD ONLY mode...
    echo WARNING: This will use your BigQuery quota!
    set /p confirm="Are you sure? (y/n): "
    if /i "%confirm%"=="y" (
        set /p num="Enter number of stocks to analyze: "
        python cloud_market_analysis.py --num-stocks %num% --use-bigquery --cloud-only
    )

) else if "%choice%"=="3" (
    echo.
    echo Running LOCAL ONLY mode...
    set /p num="Enter number of stocks to analyze: "
    python cloud_market_analysis.py --num-stocks %num% --local-only

) else if "%choice%"=="4" (
    echo.
    echo Checking cloud usage and quotas...
    python cloud_market_analysis.py --check-usage

) else if "%choice%"=="5" (
    echo.
    echo Running small test with 10 stocks...
    python cloud_market_analysis.py --num-stocks 10 --use-bigquery

) else if "%choice%"=="6" (
    echo.
    echo Running medium analysis with 100 stocks...
    python cloud_market_analysis.py --num-stocks 100 --use-bigquery

) else if "%choice%"=="7" (
    echo.
    echo Running large analysis with 500 stocks...
    echo This may take a while...
    python cloud_market_analysis.py --num-stocks 500 --use-bigquery

) else (
    echo Invalid choice!
)

echo.
echo ========================================
echo Analysis Complete
echo ========================================
pause