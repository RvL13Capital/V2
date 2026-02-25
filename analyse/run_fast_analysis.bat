@echo off
setlocal enabledelayedexpansion

:: Fast GCS Analysis - Bypasses BigQuery quota issues
color 0A
title Fast Market Analysis - Direct from GCS

echo.
echo ================================================================================
echo              FAST MARKET ANALYSIS (NO BIGQUERY)
echo           Direct GCS Processing - Avoids Quota Issues
echo ================================================================================
echo.

:: Set credentials
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json

:MENU
echo.
echo [1] Quick Test (100 tickers)
echo [2] Medium Analysis (500 tickers)
echo [3] Large Analysis (1000 tickers)
echo [4] FULL ANALYSIS (ALL ~3000 tickers)
echo [5] Custom number
echo [0] Exit
echo.

set /p choice="Select option: "

if "%choice%"=="1" (
    set limit=100
    goto RUN
)
if "%choice%"=="2" (
    set limit=500
    goto RUN
)
if "%choice%"=="3" (
    set limit=1000
    goto RUN
)
if "%choice%"=="4" (
    set limit=
    goto RUN
)
if "%choice%"=="5" (
    set /p limit="Enter number of tickers: "
    goto RUN
)
if "%choice%"=="0" exit /b 0

goto MENU

:RUN
cls
echo.
echo ================================================================================
echo                         PROCESSING
echo ================================================================================
echo.

if defined limit (
    echo Processing %limit% tickers...
    set args=--limit %limit%
) else (
    echo Processing ALL tickers...
    set args=
)

echo Start time: %time%
echo.

:: Run the analysis
python fast_gcs_analysis.py %args% --batch-size 100

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Analysis completed!
    echo End time: %time%

    :: Show results file
    echo.
    echo Results saved to: gcs_analysis_results.parquet

    :: Try to show summary
    python -c "import pandas as pd; df=pd.read_parquet('gcs_analysis_results.parquet'); print(f'\nSummary:'); print(f'  Total tickers: {len(df)}'); print(f'  In consolidation: {df[\"consolidation\"].sum() if \"consolidation\" in df else 0}'); print(f'\nTop 5 consolidation candidates:'); top5=df[df['consolidation']==True].head(5) if 'consolidation' in df else df.head(5); print(top5[['ticker', 'price', 'consolidation_days_30d']].to_string() if not top5.empty else 'No data')" 2>nul
) else (
    echo.
    echo [ERROR] Analysis failed!
)

echo.
pause
goto MENU