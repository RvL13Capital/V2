@echo off
echo ==========================================
echo Enhanced Pattern Export for EDA Analysis
echo ==========================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Select export option:
echo 1. Quick Test (10 tickers)
echo 2. Small Dataset (50 tickers)
echo 3. Medium Dataset (100 tickers)
echo 4. Large Dataset (500 tickers)
echo 5. Custom ticker list
echo 6. Date range export
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Exporting patterns for 10 tickers...
    python export_patterns_for_eda.py --num-tickers 10

) else if "%choice%"=="2" (
    echo.
    echo Exporting patterns for 50 tickers...
    python export_patterns_for_eda.py --num-tickers 50

) else if "%choice%"=="3" (
    echo.
    echo Exporting patterns for 100 tickers...
    python export_patterns_for_eda.py --num-tickers 100

) else if "%choice%"=="4" (
    echo.
    echo Exporting patterns for 500 tickers...
    echo This may take 10-20 minutes...
    python export_patterns_for_eda.py --num-tickers 500

) else if "%choice%"=="5" (
    echo.
    set /p tickers="Enter tickers separated by spaces (e.g., AAPL MSFT GOOGL): "
    python export_patterns_for_eda.py --tickers %tickers%

) else if "%choice%"=="6" (
    echo.
    set /p start="Enter start date (YYYY-MM-DD): "
    set /p end="Enter end date (YYYY-MM-DD): "
    set /p num="Enter number of tickers: "
    python export_patterns_for_eda.py --num-tickers %num% --start-date %start% --end-date %end%

) else (
    echo Invalid choice!
)

echo.
echo ==========================================
echo Export Complete!
echo ==========================================
echo.
echo The JSON file contains:
echo - All required fields for EDA analysis
echo - Complete qualification metrics (30+ features)
echo - Temporal data for time-based analysis
echo - Outcome classes and performance metrics
echo - Market context and volume profiles
echo.
echo You can now use this file with eda_tool.py
echo.
pause