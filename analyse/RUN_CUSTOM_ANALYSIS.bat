@echo off
setlocal enabledelayedexpansion

:: Custom Advanced Analysis with Filters
:: ======================================

title AIv3 Custom Analysis - Advanced Options

cls
echo ================================================================
echo       AIv3 CUSTOM ANALYSIS WITH ADVANCED FILTERING OPTIONS
echo ================================================================
echo.
echo This tool allows you to customize your analysis parameters:
echo   - Number of tickers to analyze
echo   - Pattern duration filters
echo   - Boundary width thresholds
echo   - Volume contraction requirements
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

:: Initialize default values
set NUM_TICKERS=100
set MIN_DURATION=10
set MAX_DURATION=60
set MAX_BOUNDARY=15
set MAX_VOLUME=0.8
set USE_FILTERS=N

:MAIN_MENU
cls
echo ================================================================
echo                    CUSTOM ANALYSIS CONFIGURATION
echo ================================================================
echo.
echo Current Settings:
echo   1. Number of Tickers:     %NUM_TICKERS%
echo   2. Min Pattern Duration:  %MIN_DURATION% days
echo   3. Max Pattern Duration:  %MAX_DURATION% days
echo   4. Max Boundary Width:    %MAX_BOUNDARY%%%
echo   5. Max Volume Ratio:      %MAX_VOLUME%
echo.
echo   F. Apply Filters:         %USE_FILTERS%
echo.
echo ----------------------------------------------------------------
echo.
echo Options:
echo   1-5. Change setting
echo   A.   Analyze ALL available tickers
echo   F.   Toggle filters ON/OFF
echo   R.   Reset to defaults
echo   S.   START ANALYSIS
echo   Q.   Quit
echo.

set /p choice="Enter your choice: "

:: Process choice
if /i "%choice%"=="Q" goto :EXIT
if /i "%choice%"=="S" goto :START_ANALYSIS
if /i "%choice%"=="A" goto :ANALYZE_ALL
if /i "%choice%"=="F" goto :TOGGLE_FILTERS
if /i "%choice%"=="R" goto :RESET_DEFAULTS
if "%choice%"=="1" goto :SET_TICKERS
if "%choice%"=="2" goto :SET_MIN_DURATION
if "%choice%"=="3" goto :SET_MAX_DURATION
if "%choice%"=="4" goto :SET_BOUNDARY
if "%choice%"=="5" goto :SET_VOLUME

echo.
echo Invalid choice! Please try again.
pause
goto :MAIN_MENU

:SET_TICKERS
echo.
echo Enter number of tickers to analyze (1-9999)
echo Or enter 0 to analyze ALL available tickers
set /p NUM_TICKERS="Number of tickers: "

:: Validate
set "valid=true"
for /f "delims=0123456789" %%i in ("%NUM_TICKERS%") do set valid=false
if "%valid%"=="false" (
    echo Invalid input! Using default: 100
    set NUM_TICKERS=100
    pause
)
if %NUM_TICKERS%==0 set NUM_TICKERS=9999
goto :MAIN_MENU

:SET_MIN_DURATION
echo.
set /p MIN_DURATION="Enter minimum pattern duration in days (5-100): "
:: Simple validation
if %MIN_DURATION% LSS 5 set MIN_DURATION=5
if %MIN_DURATION% GTR 100 set MIN_DURATION=100
goto :MAIN_MENU

:SET_MAX_DURATION
echo.
set /p MAX_DURATION="Enter maximum pattern duration in days (10-200): "
:: Simple validation
if %MAX_DURATION% LSS 10 set MAX_DURATION=10
if %MAX_DURATION% GTR 200 set MAX_DURATION=200
goto :MAIN_MENU

:SET_BOUNDARY
echo.
set /p MAX_BOUNDARY="Enter maximum boundary width in percent (5-50): "
:: Simple validation
if %MAX_BOUNDARY% LSS 5 set MAX_BOUNDARY=5
if %MAX_BOUNDARY% GTR 50 set MAX_BOUNDARY=50
goto :MAIN_MENU

:SET_VOLUME
echo.
echo Enter maximum volume contraction ratio (0.1-1.0)
echo Lower values = stronger volume contraction required
set /p MAX_VOLUME="Volume ratio: "
goto :MAIN_MENU

:TOGGLE_FILTERS
if "%USE_FILTERS%"=="Y" (
    set USE_FILTERS=N
) else (
    set USE_FILTERS=Y
)
goto :MAIN_MENU

:ANALYZE_ALL
set NUM_TICKERS=9999
echo.
echo Set to analyze ALL available tickers!
pause
goto :MAIN_MENU

:RESET_DEFAULTS
set NUM_TICKERS=100
set MIN_DURATION=10
set MAX_DURATION=60
set MAX_BOUNDARY=15
set MAX_VOLUME=0.8
set USE_FILTERS=N
echo.
echo Settings reset to defaults!
pause
goto :MAIN_MENU

:START_ANALYSIS
cls
echo ================================================================
echo                    STARTING CUSTOM ANALYSIS
echo ================================================================
echo.
echo Configuration Summary:
echo   - Tickers to analyze:    %NUM_TICKERS%
if "%USE_FILTERS%"=="Y" (
    echo   - Apply Filters:         YES
    echo   - Duration Range:        %MIN_DURATION% - %MAX_DURATION% days
    echo   - Max Boundary Width:    %MAX_BOUNDARY%%%
    echo   - Max Volume Ratio:      %MAX_VOLUME%
) else (
    echo   - Apply Filters:         NO ^(using default criteria^)
)
echo   - Data Source:           Full historical GCS data
echo   - Output:                Comprehensive PDF report
echo.

:: Confirm for large analyses
if %NUM_TICKERS% GTR 500 (
    echo WARNING: Very large analysis requested!
    echo This may take SEVERAL HOURS to complete.
    echo.
    set /p confirm="Are you sure you want to continue? (Y/N): "
    if /i not "!confirm!"=="Y" (
        echo.
        echo Analysis cancelled.
        pause
        goto :MAIN_MENU
    )
)

echo ----------------------------------------------------------------
echo.

:: Create custom Python script
echo Creating custom analysis script...
(
echo import os
echo import sys
echo from datetime import datetime
echo import pandas as pd
echo.
echo # Configuration from batch file
echo NUM_TICKERS = %NUM_TICKERS%
echo USE_FILTERS = "%USE_FILTERS%" == "Y"
echo MIN_DURATION = %MIN_DURATION%
echo MAX_DURATION = %MAX_DURATION%
echo MAX_BOUNDARY = %MAX_BOUNDARY%
echo MAX_VOLUME = %MAX_VOLUME%
echo.
echo print^("="*70^)
echo print^("CUSTOM ANALYSIS WITH USER-DEFINED PARAMETERS"^)
echo print^("="*70^)
echo print^(f"Tickers: {NUM_TICKERS if NUM_TICKERS < 9999 else 'ALL'}"^)
echo if USE_FILTERS:
echo     print^(f"Duration: {MIN_DURATION}-{MAX_DURATION} days"^)
echo     print^(f"Max Boundary: {MAX_BOUNDARY}%%"^)
echo     print^(f"Max Volume Ratio: {MAX_VOLUME}"^)
echo print^("="*70^)
echo print^(^)
echo.
echo try:
echo     # Set credentials
echo     cred_files = ['gcs-key.json', 'credentials.json', '../gcs-key.json']
echo     for cred_file in cred_files:
echo         if os.path.exists^(cred_file^):
echo             os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_file
echo             print^(f"Using credentials: {cred_file}"^)
echo             break
echo.
echo     # Import modules
echo     from advanced_analysis_with_gcs import ^(
echo         load_real_market_data_for_analysis,
echo         GCSDataLoader,
echo         ConsolidationPatternDetector
echo     ^)
echo     from comprehensive_pdf_report import ComprehensivePDFReportGenerator
echo.
echo     # Load data
echo     print^("\nLoading full historical data from GCS..."^)
echo     print^("This may take several minutes..."^)
echo     
echo     # Load raw data
echo     real_data = load_real_market_data_for_analysis^(
echo         num_tickers=NUM_TICKERS if NUM_TICKERS ^< 9999 else 5000,
echo         use_full_history=True
echo     ^)
echo     
echo     if real_data.empty:
echo         print^("\nERROR: No patterns found!"^)
echo         sys.exit^(1^)
echo     
echo     print^(f"\nLoaded {len^(real_data^)} patterns before filtering"^)
echo     
echo     # Apply custom filters if requested
echo     if USE_FILTERS:
echo         print^("\nApplying custom filters..."^)
echo         original_count = len^(real_data^)
echo         
echo         # Duration filter
echo         real_data = real_data[
echo             ^(real_data['duration'] ^>= MIN_DURATION^) ^& 
echo             ^(real_data['duration'] ^<= MAX_DURATION^)
echo         ]
echo         print^(f"  After duration filter: {len^(real_data^)} patterns"^)
echo         
echo         # Boundary width filter
echo         real_data = real_data[real_data['boundary_width'] ^<= MAX_BOUNDARY]
echo         print^(f"  After boundary filter: {len^(real_data^)} patterns"^)
echo         
echo         # Volume contraction filter
echo         real_data = real_data[real_data['volume_contraction'] ^<= MAX_VOLUME]
echo         print^(f"  After volume filter: {len^(real_data^)} patterns"^)
echo         
echo         filtered_pct = ^(original_count - len^(real_data^)^) / original_count * 100
echo         print^(f"\nFiltered out {filtered_pct:.1f}%% of patterns"^)
echo     
echo     if real_data.empty:
echo         print^("\nERROR: No patterns remain after filtering!"^)
echo         print^("Try relaxing your filter criteria."^)
echo         sys.exit^(1^)
echo     
echo     # Generate report
echo     print^("\nGenerating comprehensive PDF report..."^)
echo     timestamp = datetime.now^(^).strftime^("%%Y%%m%%d_%%H%%M%%S"^)
echo     
echo     if NUM_TICKERS == 9999:
echo         report_name = f"custom_analysis_ALL_{timestamp}.pdf"
echo     else:
echo         report_name = f"custom_analysis_{NUM_TICKERS}t_{timestamp}.pdf"
echo     
echo     generator = ComprehensivePDFReportGenerator^(real_data, report_name^)
echo     generator.generate_report^(^)
echo     
echo     # Summary statistics
echo     print^("\n" + "="*70^)
echo     print^("ANALYSIS COMPLETE!"^)
echo     print^("="*70^)
echo     
echo     print^(f"\nFinal Dataset:"^)
echo     print^(f"  Total Patterns: {len^(real_data^):,}"^)
echo     print^(f"  Unique Tickers: {real_data['ticker'].nunique^(^)}"^)
echo     
echo     # Outcome distribution
echo     print^("\nOutcome Distribution:"^)
echo     for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
echo         count = len^(real_data[real_data['outcome_class'] == outcome]^)
echo         pct = count / len^(real_data^) * 100 if len^(real_data^) ^> 0 else 0
echo         print^(f"  {outcome}: {count:,} ^({pct:.1f}%%^)"^)
echo     
echo     success_rate = len^(real_data[real_data['outcome_class'].isin^(['K2','K3','K4']^)]^) / len^(real_data^) * 100
echo     k4_rate = len^(real_data[real_data['outcome_class'] == 'K4']^) / len^(real_data^) * 100
echo     
echo     print^(f"\nKey Metrics:"^)
echo     print^(f"  Success Rate ^(K2-K4^): {success_rate:.1f}%%"^)
echo     print^(f"  Exceptional Rate ^(K4^): {k4_rate:.1f}%%"^)
echo     print^(f"  Average Max Gain: {real_data['max_gain'].mean^(^):.1f}%%"^)
echo     
echo     print^(f"\nPDF Report: {report_name}"^)
echo     
echo     # Open PDF
echo     import subprocess
echo     try:
echo         subprocess.Popen^(['start', '', report_name], shell=True^)
echo         print^("Report opened automatically."^)
echo     except:
echo         pass
echo         
echo except ImportError as e:
echo     print^(f"\nERROR: Missing module: {e}"^)
echo     print^("Run: pip install -r requirements.txt"^)
echo     sys.exit^(1^)
echo except Exception as e:
echo     print^(f"\nERROR: {e}"^)
echo     import traceback
echo     traceback.print_exc^(^)
echo     sys.exit^(1^)
) > custom_analysis_script.py

:: Run the analysis
echo Starting analysis with custom parameters...
echo.
python custom_analysis_script.py

:: Check result
if errorlevel 1 (
    echo.
    echo ================================================================
    echo                      ANALYSIS FAILED
    echo ================================================================
    echo.
    echo Check the error messages above for details.
    echo.
) else (
    echo.
    echo ================================================================
    echo                   ANALYSIS SUCCESSFUL
    echo ================================================================
    echo.
)

:: Cleanup
del custom_analysis_script.py 2>nul

:: Ask if user wants to run another analysis
echo.
set /p again="Run another analysis with different settings? (Y/N): "
if /i "%again%"=="Y" goto :MAIN_MENU

:EXIT
echo.
echo Thank you for using AIv3 Custom Analysis!
echo.
pause
exit /b 0