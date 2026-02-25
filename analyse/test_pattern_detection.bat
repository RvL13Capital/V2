@echo off
REM Test pattern detection with small sample

echo Testing pattern detection with 2 tickers...
echo.

call venv\Scripts\activate.bat

echo Creating output directory...
mkdir output 2>nul

echo.
echo Running detection...
python main.py detect --limit 2

echo.
echo Checking output directory...
dir output\*.parquet

echo.
echo If patterns file exists, running analysis...
for %%f in (output\patterns_*.parquet) do (
    echo Found: %%f
    echo Running analysis...
    python main.py analyze --input "%%f" --type statistical
    goto :done
)

:done
echo.
echo Test complete!
pause