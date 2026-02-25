@echo off
setlocal enabledelayedexpansion

:: Historical Pattern Analysis - Finds ALL patterns over time
color 0A
title Historical Consolidation Pattern Analyzer

echo.
echo ================================================================================
echo           HISTORICAL CONSOLIDATION PATTERN ANALYZER
echo         Finds ALL patterns in history and their outcomes
echo ================================================================================
echo.
echo This tool analyzes the ENTIRE price history to find:
echo   - ALL consolidation patterns that occurred
echo   - What happened AFTER each pattern (breakout or failure)
echo   - Success rates and pattern characteristics
echo   - Current patterns that might breakout soon
echo.
echo ================================================================================
echo.

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json

:MENU
echo.
echo Select analysis scope:
echo.
echo   [1] Quick Test     -   10 tickers (find all their historical patterns)
echo   [2] Small Analysis -   50 tickers
echo   [3] Medium Analysis -  100 tickers
echo   [4] Large Analysis -   500 tickers
echo   [5] FULL HISTORICAL - ALL ~3000 tickers (finds EVERY pattern ever!)
echo.
echo   [R] View recent results
echo   [V] View pattern statistics
echo   [0] Exit
echo.

set /p choice="Your choice: "

if "%choice%"=="1" (
    set limit=10
    echo.
    echo Analyzing 10 tickers for ALL historical patterns...
    goto RUN
)
if "%choice%"=="2" (
    set limit=50
    echo.
    echo Analyzing 50 tickers for ALL historical patterns...
    goto RUN
)
if "%choice%"=="3" (
    set limit=100
    echo.
    echo Analyzing 100 tickers for ALL historical patterns...
    goto RUN
)
if "%choice%"=="4" (
    set limit=500
    echo.
    echo Analyzing 500 tickers for ALL historical patterns...
    goto RUN
)
if "%choice%"=="5" (
    echo.
    echo [WARNING] This will analyze ALL ~3000 tickers!
    echo This could find 10,000+ patterns and take 30+ minutes
    set /p confirm="Are you sure? (Y/N): "
    if /i "!confirm!"=="Y" (
        set limit=
        goto RUN
    )
    goto MENU
)
if /i "%choice%"=="R" goto RESULTS
if /i "%choice%"=="V" goto STATS
if "%choice%"=="0" exit /b 0

echo Invalid choice!
timeout /t 2 >nul
goto MENU

:RUN
cls
echo.
echo ================================================================================
echo                    ANALYZING HISTORICAL PATTERNS
echo ================================================================================
echo.

if defined limit (
    set args=--limit %limit%
    echo Analyzing %limit% tickers...
) else (
    set args=
    echo Analyzing ALL tickers...
)

echo.
echo This will find:
echo   1. Every consolidation pattern in history
echo   2. What happened after each pattern
echo   3. Which patterns led to explosive moves (40%+)
echo   4. Current patterns ready to breakout
echo.
echo Start time: %time%
echo Processing...
echo.

python historical_pattern_analyzer.py %args% --json

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo                      ANALYSIS COMPLETE!
    echo ================================================================================
    echo End time: %time%
    echo.
    echo Files created:
    echo   - historical_patterns.parquet (all pattern data)
    echo   - historical_patterns.json (readable format)
    echo   - pattern_summary.json (statistics)

    :: Show summary
    if exist pattern_summary.json (
        echo.
        echo Pattern Statistics:
        echo -------------------
        type pattern_summary.json 2>nul
    )

    echo.
    set /p view="Export to Excel for detailed analysis? (Y/N): "
    if /i "!view!"=="Y" (
        python -c "import pandas as pd; df=pd.read_parquet('historical_patterns.parquet'); df.to_csv('historical_patterns.csv', index=False); print('\nExported to historical_patterns.csv')" 2>nul
        start historical_patterns.csv 2>nul
    )

) else (
    echo.
    echo [ERROR] Analysis failed!
)

echo.
pause
goto MENU

:RESULTS
cls
echo.
echo ================================================================================
echo                    RECENT ANALYSIS RESULTS
echo ================================================================================
echo.

dir /b /od historical_patterns*.* 2>nul
dir /b pattern_summary.json 2>nul

echo.

if exist historical_patterns.parquet (
    echo Loading recent patterns...
    echo.
    python -c "import pandas as pd; df=pd.read_parquet('historical_patterns.parquet'); recent=df[df['outcome_class'].str.contains('PENDING|UNKNOWN', na=False)] if 'outcome_class' in df else df.tail(20); print('CURRENT/RECENT PATTERNS:'); print('-'*40); for _,r in recent.head(15).iterrows(): print(f\"{r['ticker']:6s} - Duration: {r['pattern_duration_days']} days - Status: {r.get('outcome_class', 'N/A')}\")" 2>nul
)

echo.
pause
goto MENU

:STATS
cls
echo.
echo ================================================================================
echo                    PATTERN STATISTICS
echo ================================================================================
echo.

if exist historical_patterns.parquet (
    python -c "import pandas as pd; df=pd.read_parquet('historical_patterns.parquet'); print(f'Total patterns found: {len(df)}'); print(f'Unique tickers: {df[\"ticker\"].nunique()}'); print('\nOutcome Distribution:'); print(df['outcome_class'].value_counts() if 'outcome_class' in df else 'No outcome data'); explosive=df[df['outcome_max_gain']>40] if 'outcome_max_gain' in df else None; print(f'\nExplosive patterns (40%+): {len(explosive) if explosive is not None else 0}'); print('\nTop 10 Explosive Patterns:'); if explosive is not None and not explosive.empty: top=explosive.nlargest(10, 'outcome_max_gain')[['ticker','pattern_start_date','outcome_max_gain']]; for _,r in top.iterrows(): print(f'{r[\"ticker\"]:6s} ({r[\"pattern_start_date\"][:10]}) -> +{r[\"outcome_max_gain\"]:.1f}%%')" 2>nul
) else (
    echo No analysis data found. Run an analysis first!
)

echo.
pause
goto MENU