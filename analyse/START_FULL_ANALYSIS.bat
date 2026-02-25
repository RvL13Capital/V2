@echo off
echo ================================================================================
echo                     FULL CONSOLIDATION PATTERN ANALYSIS
echo ================================================================================
echo.
echo This will analyze ALL tickers in your GCS bucket (2900+ files)
echo Estimated time: 10-30 minutes
echo.
echo Results will be saved to: full_analysis_[timestamp]/
echo.
pause

echo.
echo Starting analysis...
python run_full_analysis.py

pause