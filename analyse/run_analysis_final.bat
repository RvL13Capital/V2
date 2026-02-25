@echo off
setlocal enabledelayedexpansion

:: FINAL SOLUTION - Optimized GCS Analysis
color 0A
title Market Analysis - Optimized Version

echo.
echo ================================================================================
echo                    MARKET CONSOLIDATION ANALYZER
echo                        Optimized GCS Processing
echo ================================================================================
echo.

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json

:MENU
echo.
echo Select analysis scope:
echo.
echo   [1] Test Run      -    25 tickers (~15 seconds)
echo   [2] Quick Scan    -   100 tickers (~1 minute)
echo   [3] Medium Scan   -   500 tickers (~5 minutes)
echo   [4] Large Scan    -  1000 tickers (~10 minutes)
echo   [5] FULL SCAN     -  ALL ~3000 tickers (~30 minutes)
echo.
echo   [C] Custom number
echo   [R] Show recent results
echo   [0] Exit
echo.

set /p choice="Your choice: "

if "%choice%"=="1" (
    set limit=25
    set batch=25
    goto RUN
)
if "%choice%"=="2" (
    set limit=100
    set batch=25
    goto RUN
)
if "%choice%"=="3" (
    set limit=500
    set batch=50
    goto RUN
)
if "%choice%"=="4" (
    set limit=1000
    set batch=50
    goto RUN
)
if "%choice%"=="5" (
    echo.
    echo [WARNING] This will process ALL ~3000 tickers!
    echo Estimated time: 30-40 minutes
    set /p confirm="Are you sure? (Y/N): "
    if /i "!confirm!"=="Y" (
        set limit=
        set batch=100
        goto RUN
    )
    goto MENU
)
if /i "%choice%"=="C" (
    set /p limit="Enter number of tickers: "
    set batch=50
    goto RUN
)
if /i "%choice%"=="R" goto RESULTS
if "%choice%"=="0" exit /b 0

echo Invalid choice!
timeout /t 2 >nul
goto MENU

:RUN
cls
echo.
echo ================================================================================
echo                            PROCESSING
echo ================================================================================
echo.

if defined limit (
    echo Analyzing %limit% tickers (batch size: %batch%)
    set args=--limit %limit% --batch-size %batch%
) else (
    echo Analyzing ALL tickers (batch size: %batch%)
    set args=--batch-size %batch%
)

echo.
echo Start time: %time%
echo.
echo Processing... (this may take a while)
echo.

:: Use optimized script
if exist optimized_gcs_analysis.py (
    python optimized_gcs_analysis.py %args%
    set result_file=optimized_analysis.parquet
) else if exist fast_gcs_analysis.py (
    python fast_gcs_analysis.py %args%
    set result_file=gcs_analysis_results.parquet
) else (
    echo [ERROR] No analysis script found!
    pause
    goto MENU
)

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo                         ANALYSIS COMPLETE
    echo ================================================================================
    echo.
    echo End time: %time%
    echo Results saved to: %result_file%

    :: Show summary
    python -c "import pandas as pd; df=pd.read_parquet('%result_file%'); print(f'\nSUMMARY:'); print(f'--------'); print(f'Total tickers analyzed: {len(df)}'); cons=df['consolidation'].sum() if 'consolidation' in df else 0; print(f'Tickers in consolidation: {cons}'); print(f'Consolidation rate: {cons/len(df)*100:.1f}%%' if len(df)>0 else 'N/A'); print(f'\nTOP CONSOLIDATION CANDIDATES:'); print('-'*40); top=df[df['consolidation']==True].head(10) if 'consolidation' in df else df.head(10); for i,r in top.iterrows(): print(f'{r[\"ticker\"]:6s} - ${r[\"price\"]:7.2f} - {r[\"consolidation_days_30d\"]} days') if 'price' in r and r['price'] else print(f'{r[\"ticker\"]:6s}')" 2>nul

    echo.
    set /p view="Open full results in Excel/CSV? (Y/N): "
    if /i "!view!"=="Y" (
        python -c "import pandas as pd; df=pd.read_parquet('%result_file%'); df.to_csv('analysis_results.csv', index=False); print('Exported to analysis_results.csv')" 2>nul
        start analysis_results.csv 2>nul
    )
) else (
    echo.
    echo [ERROR] Analysis failed!
    echo Check the error messages above.
)

echo.
pause
goto MENU

:RESULTS
cls
echo.
echo ================================================================================
echo                         RECENT RESULTS
echo ================================================================================
echo.

echo Analysis files:
echo ---------------
dir /b /od *analysis*.parquet 2>nul

echo.
echo CSV exports:
echo ------------
dir /b /od analysis_results*.csv 2>nul

echo.

:: Check latest file
for /f "delims=" %%i in ('dir /b /od *analysis*.parquet 2^>nul') do set latest=%%i
if defined latest (
    echo Latest: %latest%
    python -c "import pandas as pd; df=pd.read_parquet('%latest%'); print(f'Tickers: {len(df)}, In consolidation: {df[\"consolidation\"].sum() if \"consolidation\" in df else 0}')" 2>nul
)

echo.
pause
goto MENU