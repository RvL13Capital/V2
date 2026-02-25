@echo off
setlocal enabledelayedexpansion

:: Advanced Analysis with Full Historical Data
:: ============================================

title AIv3 Advanced Analysis - Full Historical Data

cls
echo ================================================================
echo         AIv3 ADVANCED ANALYSIS WITH FULL HISTORICAL DATA
echo ================================================================
echo.
echo This analysis uses complete historical data from GCS:
echo   - ignition-ki-csv-data-2025-user123/market_data/
echo   - ignition-ki-csv-data-2025-user123/tickers/
echo.
echo ================================================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Main Menu
:MENU
echo SELECT ANALYSIS SCOPE:
echo.
echo   1. Quick Analysis    (25 tickers)
echo   2. Standard Analysis (50 tickers)
echo   3. Extended Analysis (100 tickers)
echo   4. Large Analysis    (200 tickers)
echo   5. Maximum Analysis  (500 tickers)
echo   6. Custom Number     (specify your own)
echo   7. ANALYZE ALL       (all available tickers - may take hours!)
echo.
echo   Q. Quit
echo.
echo ----------------------------------------------------------------

set /p choice="Enter your choice (1-7 or Q): "

:: Process choice
if /i "%choice%"=="Q" goto :EXIT
if "%choice%"=="1" set NUM_TICKERS=25&& goto :RUN_ANALYSIS
if "%choice%"=="2" set NUM_TICKERS=50&& goto :RUN_ANALYSIS
if "%choice%"=="3" set NUM_TICKERS=100&& goto :RUN_ANALYSIS
if "%choice%"=="4" set NUM_TICKERS=200&& goto :RUN_ANALYSIS
if "%choice%"=="5" set NUM_TICKERS=500&& goto :RUN_ANALYSIS
if "%choice%"=="6" goto :CUSTOM_NUMBER
if "%choice%"=="7" set NUM_TICKERS=9999&& goto :RUN_ANALYSIS

:: Invalid choice
echo.
echo Invalid choice! Please try again.
echo.
pause
cls
goto :MENU

:CUSTOM_NUMBER
echo.
set /p NUM_TICKERS="Enter number of tickers to analyze (1-9999): "

:: Validate input is a number
set "valid=true"
for /f "delims=0123456789" %%i in ("%NUM_TICKERS%") do set valid=false
if "%valid%"=="false" (
    echo.
    echo ERROR: Please enter a valid number!
    echo.
    pause
    cls
    goto :MENU
)

:: Check range
if %NUM_TICKERS% LSS 1 (
    echo.
    echo ERROR: Number must be at least 1!
    echo.
    pause
    cls
    goto :MENU
)

if %NUM_TICKERS% GTR 9999 (
    echo.
    echo ERROR: Number too large! Maximum is 9999.
    echo.
    pause
    cls
    goto :MENU
)

:RUN_ANALYSIS
cls
echo ================================================================
echo                    STARTING ANALYSIS
echo ================================================================
echo.
echo Configuration:
echo   - Tickers to analyze: %NUM_TICKERS%
echo   - Data source: Full historical data from GCS
echo   - Analysis type: All 5 advanced analyses
echo   - Output: PDF report with visualizations
echo.

:: Warning for large analyses
if %NUM_TICKERS% GTR 200 (
    echo WARNING: Large analysis requested!
    echo This may take significant time to complete.
    echo.
    set /p confirm="Continue? (Y/N): "
    if /i not "!confirm!"=="Y" (
        echo.
        echo Analysis cancelled.
        pause
        cls
        goto :MENU
    )
)

echo ----------------------------------------------------------------
echo.

:: Create Python script for analysis
echo Creating analysis script...
(
echo import os
echo import sys
echo from datetime import datetime
echo.
echo # Set number of tickers
echo NUM_TICKERS = %NUM_TICKERS%
echo.
echo print^(f"Starting analysis with {NUM_TICKERS} tickers..."^)
echo print^("="*60^)
echo.
echo # Import and run analysis
echo try:
echo     from advanced_analysis_with_gcs import load_real_market_data_for_analysis
echo     from comprehensive_pdf_report import ComprehensivePDFReportGenerator
echo     
echo     # Set GCS credentials if available
echo     cred_files = ['gcs-key.json', 'credentials.json', '../gcs-key.json']
echo     for cred_file in cred_files:
echo         if os.path.exists^(cred_file^):
echo             os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_file
echo             print^(f"Using credentials: {cred_file}"^)
echo             break
echo     
echo     # Load data with full history
echo     print^("\nLoading full historical data from GCS..."^)
echo     print^("This may take several minutes depending on the number of tickers..."^)
echo     
echo     real_data = load_real_market_data_for_analysis^(
echo         num_tickers=NUM_TICKERS,
echo         use_full_history=True  # Use complete available history
echo     ^)
echo     
echo     if real_data.empty:
echo         print^("\nERROR: No patterns found in the data!"^)
echo         sys.exit^(1^)
echo     
echo     # Generate report
echo     print^("\nGenerating comprehensive PDF report..."^)
echo     timestamp = datetime.now^(^).strftime^("%%Y%%m%%d_%%H%%M%%S"^)
echo     
echo     if NUM_TICKERS == 9999:
echo         report_name = f"full_analysis_ALL_TICKERS_{timestamp}.pdf"
echo     else:
echo         report_name = f"full_analysis_{NUM_TICKERS}_tickers_{timestamp}.pdf"
echo     
echo     generator = ComprehensivePDFReportGenerator^(real_data, report_name^)
echo     generator.generate_report^(^)
echo     
echo     # Print summary
echo     print^("\n" + "="*60^)
echo     print^("ANALYSIS COMPLETE!"^)
echo     print^("="*60^)
echo     print^(f"\nAnalyzed {len^(real_data^)} patterns from {real_data['ticker'].nunique^(^)} tickers"^)
echo     
echo     success_rate = len^(real_data[real_data['outcome_class'].isin^(['K2','K3','K4']^)]^) / len^(real_data^) * 100
echo     k4_rate = len^(real_data[real_data['outcome_class'] == 'K4']^) / len^(real_data^) * 100
echo     
echo     print^(f"Success Rate ^(K2-K4^): {success_rate:.1f}%%"^)
echo     print^(f"Exceptional Rate ^(K4^): {k4_rate:.1f}%%"^)
echo     print^(f"\nPDF Report: {report_name}"^)
echo     
echo     # Open the PDF automatically on Windows
echo     import subprocess
echo     try:
echo         subprocess.Popen^(['start', '', report_name], shell=True^)
echo         print^("Report opened automatically."^)
echo     except:
echo         print^("Please open the report manually."^)
echo         
echo except ImportError as e:
echo     print^(f"\nERROR: Missing required module: {e}"^)
echo     print^("Please install requirements: pip install -r requirements.txt"^)
echo     sys.exit^(1^)
echo except Exception as e:
echo     print^(f"\nERROR: {e}"^)
echo     import traceback
echo     traceback.print_exc^(^)
echo     sys.exit^(1^)
) > temp_analysis_script.py

:: Run the analysis
echo Running analysis...
echo.
python temp_analysis_script.py

:: Check if analysis succeeded
if errorlevel 1 (
    echo.
    echo ================================================================
    echo                    ANALYSIS FAILED
    echo ================================================================
    echo.
    echo Please check the error messages above.
    echo Common issues:
    echo   - Missing GCS credentials ^(gcs-key.json^)
    echo   - Missing Python packages ^(run: pip install -r requirements.txt^)
    echo   - No internet connection to access GCS
    echo.
) else (
    echo.
    echo ================================================================
    echo                    ANALYSIS SUCCESSFUL
    echo ================================================================
    echo.
    echo The PDF report has been generated and should open automatically.
    echo Look for the PDF file in the current directory.
    echo.
)

:: Cleanup
del temp_analysis_script.py 2>nul

:ASK_AGAIN
echo.
set /p again="Run another analysis? (Y/N): "
if /i "%again%"=="Y" (
    cls
    goto :MENU
)

:EXIT
echo.
echo Thank you for using AIv3 Advanced Analysis!
echo.
pause
exit /b 0