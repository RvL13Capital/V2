@echo off
REM Complete workflow test

echo ============================================================
echo Testing Complete AIv3 Workflow
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo [Step 1] Ensuring output directory exists...
mkdir output 2>nul

echo.
echo [Step 2] Running pattern detection with 2 tickers...
python main.py detect --limit 2

echo.
echo [Step 3] Finding the generated pattern file...
for /f "delims=" %%i in ('dir /b /od output\patterns_*.parquet 2^>nul ^| findstr /r "patterns_.*\.parquet"') do set "PATTERN_FILE=output\%%i"

if "%PATTERN_FILE%"=="" (
    echo ERROR: No pattern file found!
    pause
    exit /b 1
)

echo Found pattern file: %PATTERN_FILE%

echo.
echo [Step 4] Running analysis on the pattern file...
python main.py analyze --input "%PATTERN_FILE%" --type statistical

echo.
echo [Step 5] Running performance analysis...
python main.py analyze --input "%PATTERN_FILE%" --type performance

echo.
echo ============================================================
echo Workflow Test Complete!
echo ============================================================
pause