@echo off
echo ================================================================================
echo           COMPLETE ANALYSIS WITH PDF REPORT
echo ================================================================================
echo.
echo This will:
echo   1. Process ALL tickers from GCS (3000+ files)
echo   2. Apply $0.01 minimum price filter
echo   3. Detect consolidation patterns
echo   4. Calculate strategic values (K0-K5)
echo   5. Generate comprehensive PDF report
echo.
echo Estimated time: 15-45 minutes
echo.
echo ================================================================================
echo.
echo Starting analysis...
echo.

python run_analysis_auto.py

echo.
echo ================================================================================
echo ANALYSIS COMPLETE
echo ================================================================================
echo.
echo Check the output folder for:
echo   - PDF Report (analysis_report_*.pdf)
echo   - Pattern data (patterns_*.json)
echo   - Extended metrics (extended_metrics_*.json)
echo   - Summary report (summary_*.txt)
echo.
pause