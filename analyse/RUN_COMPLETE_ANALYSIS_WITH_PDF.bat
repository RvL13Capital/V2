@echo off
echo ================================================================================
echo           COMPLETE CONSOLIDATION ANALYSIS WITH PDF REPORT
echo ================================================================================
echo.
echo This comprehensive analysis will:
echo   1. Detect patterns in ALL tickers from your GCS bucket
echo   2. Calculate extended metrics (time to targets, false breakouts, etc.)
echo   3. Generate detailed PDF report with visualizations
echo.
echo Estimated time: 15-45 minutes
echo.
echo Output will include:
echo   - PDF report with model explanation and visualizations
echo   - JSON files with all patterns and metrics
echo   - Text summary report
echo.
pause

echo.
echo Installing required packages for PDF generation...
pip install matplotlib seaborn reportlab fpdf2 --quiet

echo.
echo Starting complete analysis...
python run_complete_analysis.py

pause