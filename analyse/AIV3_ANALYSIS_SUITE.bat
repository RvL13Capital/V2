@echo off
setlocal enabledelayedexpansion

:: AIv3 Complete Analysis Suite - All-in-One
:: ==========================================
:: This script handles everything:
:: - Setup and verification
:: - Credential management  
:: - Package installation
:: - Analysis execution with multiple options
:: ==========================================

title AIv3 Analysis Suite - Complete System

:: Set colors for better UI
color 0A

cls
echo.
echo    ============================================================
echo              AIv3 CONSOLIDATION PATTERN ANALYSIS SUITE
echo    ============================================================
echo.
echo                    Advanced Market Analysis System
echo                     Using Full Historical GCS Data
echo.
echo    ============================================================
echo.

:: Initialize variables
set SYSTEM_READY=NO
set CREDENTIAL_FILE=
set PYTHON_OK=NO
set PACKAGES_OK=NO
set GCS_OK=NO

:: Main Menu Loop
:MAIN_MENU
cls
echo ================================================================
echo                    AIv3 ANALYSIS SUITE
echo ================================================================
echo.
echo System Status:
if "%SYSTEM_READY%"=="YES" (
    echo   [READY]  All systems operational
    echo.
) else (
    echo   [SETUP]  Configuration needed
    echo.
)
echo ----------------------------------------------------------------
echo.
echo   1. Quick Setup     - Automatic setup and verification
echo   2. Run Analysis    - Start pattern analysis
echo   3. Custom Analysis - Advanced options
echo   4. System Check    - Verify configuration
echo   5. Credentials     - Setup GCS credentials
echo   6. Install Deps    - Install Python packages
echo   7. Help            - Documentation
echo   Q. Quit
echo.
echo ================================================================
echo.

set /p MENU_CHOICE="Select option (1-7 or Q): "

if /i "%MENU_CHOICE%"=="Q" goto :EXIT_PROGRAM
if "%MENU_CHOICE%"=="1" goto :QUICK_SETUP
if "%MENU_CHOICE%"=="2" goto :RUN_ANALYSIS
if "%MENU_CHOICE%"=="3" goto :CUSTOM_ANALYSIS
if "%MENU_CHOICE%"=="4" goto :SYSTEM_CHECK
if "%MENU_CHOICE%"=="5" goto :SETUP_CREDENTIALS
if "%MENU_CHOICE%"=="6" goto :INSTALL_PACKAGES
if "%MENU_CHOICE%"=="7" goto :SHOW_HELP

echo.
echo Invalid option! Please try again.
timeout /t 2 >nul
goto :MAIN_MENU

:: ==================== QUICK SETUP ====================
:QUICK_SETUP
cls
echo ================================================================
echo                      QUICK SETUP
echo ================================================================
echo.
echo Running automatic configuration...
echo.

:: 1. Check Python
call :CHECK_PYTHON_FUNC
if "%PYTHON_OK%"=="NO" (
    echo.
    echo ERROR: Python not found. Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    echo.
    pause
    goto :MAIN_MENU
)

:: 2. Setup credentials
call :AUTO_SETUP_CREDENTIALS

:: 3. Install packages
call :AUTO_INSTALL_PACKAGES

:: 4. Test connection
call :TEST_GCS_CONNECTION

:: Check if ready
if "%PYTHON_OK%"=="YES" if "%PACKAGES_OK%"=="YES" if "%GCS_OK%"=="YES" (
    set SYSTEM_READY=YES
    echo.
    echo ================================================================
    echo                    SETUP COMPLETE!
    echo ================================================================
    echo.
    echo System is ready for analysis.
    echo.
    set /p START_NOW="Start analysis now? (Y/N): "
    if /i "!START_NOW!"=="Y" goto :RUN_ANALYSIS
) else (
    echo.
    echo ================================================================
    echo                  SETUP INCOMPLETE
    echo ================================================================
    echo Please check the errors above and try manual setup.
    echo.
    pause
)
goto :MAIN_MENU

:: ==================== RUN ANALYSIS ====================
:RUN_ANALYSIS
cls

:: Check if system is ready
if not "%SYSTEM_READY%"=="YES" (
    call :QUICK_CHECK
    if not "%SYSTEM_READY%"=="YES" (
        echo.
        echo System is not ready. Running setup first...
        echo.
        timeout /t 2 >nul
        goto :QUICK_SETUP
    )
)

echo ================================================================
echo                    RUN ANALYSIS
echo ================================================================
echo.
echo Select analysis scope:
echo.
echo   1. Quick     (25 tickers,  ~5 min)
echo   2. Standard  (50 tickers,  ~10 min)
echo   3. Extended  (100 tickers, ~20 min)
echo   4. Large     (200 tickers, ~45 min)
echo   5. Maximum   (500 tickers, ~2 hours)
echo   6. Custom    (specify number)
echo   7. ALL DATA  (all tickers, several hours!)
echo.
echo   B. Back to main menu
echo.
echo ----------------------------------------------------------------
echo.

set /p ANALYSIS_CHOICE="Select scope (1-7 or B): "

if /i "%ANALYSIS_CHOICE%"=="B" goto :MAIN_MENU

:: Set number of tickers based on choice
if "%ANALYSIS_CHOICE%"=="1" set NUM_TICKERS=25
if "%ANALYSIS_CHOICE%"=="2" set NUM_TICKERS=50
if "%ANALYSIS_CHOICE%"=="3" set NUM_TICKERS=100
if "%ANALYSIS_CHOICE%"=="4" set NUM_TICKERS=200
if "%ANALYSIS_CHOICE%"=="5" set NUM_TICKERS=500
if "%ANALYSIS_CHOICE%"=="6" (
    set /p NUM_TICKERS="Enter number of tickers (1-9999): "
)
if "%ANALYSIS_CHOICE%"=="7" set NUM_TICKERS=9999

:: Validate
if not defined NUM_TICKERS (
    echo Invalid choice!
    timeout /t 2 >nul
    goto :RUN_ANALYSIS
)

:: Run the analysis
call :EXECUTE_ANALYSIS %NUM_TICKERS%

echo.
echo Analysis complete!
echo.
pause
goto :MAIN_MENU

:: ==================== CUSTOM ANALYSIS ====================
:CUSTOM_ANALYSIS
cls
echo ================================================================
echo                   CUSTOM ANALYSIS
echo ================================================================
echo.

:: Check if system is ready
if not "%SYSTEM_READY%"=="YES" (
    echo System not ready. Please run Quick Setup first.
    echo.
    pause
    goto :MAIN_MENU
)

:: Get custom parameters
echo Configure analysis parameters:
echo.

set /p NUM_TICKERS="Number of tickers (default 100): "
if "%NUM_TICKERS%"=="" set NUM_TICKERS=100

set /p MIN_DURATION="Minimum pattern duration in days (default 10): "
if "%MIN_DURATION%"=="" set MIN_DURATION=10

set /p MAX_DURATION="Maximum pattern duration in days (default 60): "
if "%MAX_DURATION%"=="" set MAX_DURATION=60

set /p MAX_BOUNDARY="Maximum boundary width %% (default 15): "
if "%MAX_BOUNDARY%"=="" set MAX_BOUNDARY=15

set /p MAX_VOLUME="Maximum volume ratio (default 0.8): "
if "%MAX_VOLUME%"=="" set MAX_VOLUME=0.8

echo.
echo ----------------------------------------------------------------
echo Configuration:
echo   Tickers:         %NUM_TICKERS%
echo   Duration:        %MIN_DURATION%-%MAX_DURATION% days
echo   Boundary Width:  <=%MAX_BOUNDARY%%%
echo   Volume Ratio:    <=%MAX_VOLUME%
echo ----------------------------------------------------------------
echo.

set /p CONFIRM="Start analysis with these settings? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto :MAIN_MENU

:: Execute custom analysis
call :EXECUTE_CUSTOM_ANALYSIS %NUM_TICKERS% %MIN_DURATION% %MAX_DURATION% %MAX_BOUNDARY% %MAX_VOLUME%

echo.
pause
goto :MAIN_MENU

:: ==================== SYSTEM CHECK ====================
:SYSTEM_CHECK
cls
echo ================================================================
echo                    SYSTEM CHECK
echo ================================================================
echo.

call :CHECK_PYTHON_FUNC
call :CHECK_PACKAGES_FUNC
call :CHECK_CREDENTIALS_FUNC
call :TEST_GCS_CONNECTION

echo.
echo ----------------------------------------------------------------
echo Summary:
echo   Python:      %PYTHON_OK%
echo   Packages:    %PACKAGES_OK%
echo   Credentials: %CREDENTIAL_FILE%
echo   GCS Access:  %GCS_OK%
echo ----------------------------------------------------------------
echo.

if "%PYTHON_OK%"=="YES" if "%PACKAGES_OK%"=="YES" if "%GCS_OK%"=="YES" (
    set SYSTEM_READY=YES
    echo [READY] System is fully configured!
) else (
    set SYSTEM_READY=NO
    echo [NOT READY] Please fix the issues above.
)

echo.
pause
goto :MAIN_MENU

:: ==================== SETUP CREDENTIALS ====================
:SETUP_CREDENTIALS
cls
echo ================================================================
echo                   SETUP CREDENTIALS
echo ================================================================
echo.

:: Check for downloaded credential file
if exist "C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (2) (1).json" (
    echo Found credential file in Downloads!
    copy "C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (2) (1).json" "gcs-key.json" >nul
    set CREDENTIAL_FILE=gcs-key.json
    echo Credentials copied successfully.
    echo.
    pause
    goto :MAIN_MENU
)

echo Please provide your GCS credential JSON file.
echo.
echo You can:
echo   1. Drag and drop the file here and press Enter
echo   2. Type the full path to the file
echo   3. Place it in this directory as 'gcs-key.json'
echo.

set /p CRED_PATH="Credential file path (or Enter to skip): "

if not "%CRED_PATH%"=="" (
    set CRED_PATH=%CRED_PATH:"=%
    if exist "%CRED_PATH%" (
        copy "%CRED_PATH%" "gcs-key.json" >nul
        set CREDENTIAL_FILE=gcs-key.json
        echo.
        echo Credentials saved successfully!
    ) else (
        echo.
        echo ERROR: File not found!
    )
)

echo.
pause
goto :MAIN_MENU

:: ==================== INSTALL PACKAGES ====================
:INSTALL_PACKAGES
cls
echo ================================================================
echo                   INSTALL PACKAGES
echo ================================================================
echo.

call :AUTO_INSTALL_PACKAGES

echo.
echo Package installation complete.
echo.
pause
goto :MAIN_MENU

:: ==================== SHOW HELP ====================
:SHOW_HELP
cls
echo ================================================================
echo                        HELP
echo ================================================================
echo.
echo AIv3 Analysis Suite - Complete Documentation
echo.
echo SETUP REQUIREMENTS:
echo   1. Python 3.8 or higher
echo   2. GCS credential JSON file
echo   3. Internet connection for GCS access
echo.
echo ANALYSIS OPTIONS:
echo   - Quick: 25 tickers for fast results
echo   - Standard: 50 tickers for balanced analysis
echo   - Extended: 100 tickers for comprehensive results
echo   - Large: 200+ tickers for deep analysis
echo   - ALL: Analyze entire dataset (very slow)
echo.
echo CUSTOM PARAMETERS:
echo   - Pattern Duration: Days of consolidation (10-100)
echo   - Boundary Width: Price range percentage (5-50)
echo   - Volume Ratio: Volume contraction threshold (0.1-1.0)
echo.
echo OUTPUT:
echo   - Comprehensive PDF report with 5 analyses
echo   - Charts and visualizations
echo   - Statistical validation
echo   - Strategic recommendations
echo.
echo GCS PATHS:
echo   - market_data/ : Primary market data
echo   - tickers/     : Additional ticker data
echo.
echo ================================================================
echo.
pause
goto :MAIN_MENU

:: ==================== FUNCTIONS ====================

:CHECK_PYTHON_FUNC
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    set PYTHON_OK=NO
    echo   [FAIL] Python not found
) else (
    set PYTHON_OK=YES
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
    echo   [OK] Python !PY_VER!
)
exit /b

:CHECK_PACKAGES_FUNC
echo Checking packages...
python -c "import pandas,numpy,matplotlib,seaborn,sklearn,scipy,google.cloud.storage,reportlab" 2>nul
if errorlevel 1 (
    set PACKAGES_OK=NO
    echo   [FAIL] Some packages missing
) else (
    set PACKAGES_OK=YES
    echo   [OK] All packages installed
)
exit /b

:CHECK_CREDENTIALS_FUNC
echo Checking credentials...
if exist "gcs-key.json" (
    set CREDENTIAL_FILE=gcs-key.json
    echo   [OK] Credentials found
) else if exist "C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (2) (1).json" (
    copy "C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (2) (1).json" "gcs-key.json" >nul
    set CREDENTIAL_FILE=gcs-key.json
    echo   [OK] Credentials copied from Downloads
) else (
    set CREDENTIAL_FILE=
    echo   [FAIL] No credentials found
)
exit /b

:AUTO_SETUP_CREDENTIALS
echo Setting up credentials...
call :CHECK_CREDENTIALS_FUNC
if "%CREDENTIAL_FILE%"=="" (
    echo   Please provide credentials manually
)
exit /b

:AUTO_INSTALL_PACKAGES
echo Installing Python packages...

:: Create requirements.txt
(
    echo pandas>=2.0.0
    echo numpy>=1.24.0
    echo matplotlib>=3.7.0
    echo seaborn>=0.12.0
    echo scikit-learn>=1.3.0
    echo scipy>=1.11.0
    echo google-cloud-storage>=2.10.0
    echo reportlab>=4.0.0
    echo Pillow>=10.0.0
) > requirements.txt

pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo   [WARN] Some packages failed, trying individually...
    pip install pandas numpy matplotlib seaborn scikit-learn scipy google-cloud-storage reportlab Pillow --quiet
)

call :CHECK_PACKAGES_FUNC
exit /b

:TEST_GCS_CONNECTION
echo Testing GCS connection...
if not defined CREDENTIAL_FILE (
    set GCS_OK=NO
    echo   [SKIP] No credentials
    exit /b
)

:: Create test script
(
    echo import os, sys
    echo from google.cloud import storage
    echo os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-key.json'
    echo try:
    echo     client = storage.Client^(project='ignition-ki-csv-storage'^)
    echo     bucket = client.bucket^('ignition-ki-csv-data-2025-user123'^)
    echo     list^(bucket.list_blobs^(max_results=1^)^)
    echo     print^('   [OK] GCS connection successful'^)
    echo     sys.exit^(0^)
    echo except Exception as e:
    echo     print^('   [FAIL] ' + str^(e^)^)
    echo     sys.exit^(1^)
) > test_gcs_temp.py

python test_gcs_temp.py 2>nul
if errorlevel 1 (
    set GCS_OK=NO
) else (
    set GCS_OK=YES
)
del test_gcs_temp.py 2>nul
exit /b

:QUICK_CHECK
:: Quick system check without output
set SYSTEM_READY=NO
python --version >nul 2>&1
if errorlevel 1 exit /b
if not exist "gcs-key.json" exit /b
python -c "import pandas,numpy,matplotlib,seaborn,sklearn,scipy,google.cloud.storage,reportlab" 2>nul
if errorlevel 1 exit /b
set SYSTEM_READY=YES
exit /b

:EXECUTE_ANALYSIS
echo.
echo ================================================================
echo                   EXECUTING ANALYSIS
echo ================================================================
echo.
echo Starting analysis with %1 tickers...
echo Using full historical data from GCS
echo.

:: Create and run analysis script
(
    echo import os, sys
    echo from datetime import datetime
    echo.
    echo os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-key.json'
    echo.
    echo print^("Loading data..."^)
    echo from advanced_analysis_with_gcs import load_real_market_data_for_analysis
    echo from comprehensive_pdf_report import ComprehensivePDFReportGenerator
    echo.
    echo data = load_real_market_data_for_analysis^(num_tickers=%1, use_full_history=True^)
    echo.
    echo if data.empty:
    echo     print^("No patterns found!"^)
    echo     sys.exit^(1^)
    echo.
    echo print^(f"Analyzing {len(data)} patterns..."^)
    echo timestamp = datetime.now^(^).strftime^("%%Y%%m%%d_%%H%%M%%S"^)
    echo report_name = f"analysis_%1_tickers_{timestamp}.pdf"
    echo.
    echo generator = ComprehensivePDFReportGenerator^(data, report_name^)
    echo generator.generate_report^(^)
    echo.
    echo print^("\nReport saved: " + report_name^)
    echo.
    echo # Open PDF
    echo import subprocess
    echo subprocess.Popen^(['start', '', report_name], shell=True^)
) > run_analysis_temp.py

python run_analysis_temp.py
del run_analysis_temp.py 2>nul
exit /b

:EXECUTE_CUSTOM_ANALYSIS
echo.
echo Running custom analysis...
echo.

:: Create custom analysis script with parameters
(
    echo import os, sys
    echo from datetime import datetime
    echo import pandas as pd
    echo.
    echo os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-key.json'
    echo.
    echo # Parameters
    echo NUM_TICKERS = %1
    echo MIN_DURATION = %2
    echo MAX_DURATION = %3
    echo MAX_BOUNDARY = %4
    echo MAX_VOLUME = %5
    echo.
    echo from advanced_analysis_with_gcs import load_real_market_data_for_analysis
    echo from comprehensive_pdf_report import ComprehensivePDFReportGenerator
    echo.
    echo print^("Loading data with custom filters..."^)
    echo data = load_real_market_data_for_analysis^(num_tickers=NUM_TICKERS, use_full_history=True^)
    echo.
    echo # Apply filters
    echo print^("Applying custom filters..."^)
    echo data = data[^(data['duration'] ^>= MIN_DURATION^) ^& ^(data['duration'] ^<= MAX_DURATION^)]
    echo data = data[data['boundary_width'] ^<= MAX_BOUNDARY]
    echo data = data[data['volume_contraction'] ^<= MAX_VOLUME]
    echo.
    echo if data.empty:
    echo     print^("No patterns match criteria!"^)
    echo     sys.exit^(1^)
    echo.
    echo print^("Analyzing " + str^(len^(data^)^) + " filtered patterns..."^)
    echo timestamp = datetime.now^(^).strftime^("%%Y%%m%%d_%%H%%M%%S"^)
    echo report_name = f"custom_analysis_{timestamp}.pdf"
    echo.
    echo generator = ComprehensivePDFReportGenerator^(data, report_name^)
    echo generator.generate_report^(^)
    echo.
    echo print^("\nReport saved: " + report_name^)
    echo.
    echo # Open PDF
    echo import subprocess
    echo subprocess.Popen^(['start', '', report_name], shell=True^)
) > run_custom_temp.py

python run_custom_temp.py
del run_custom_temp.py 2>nul
exit /b

:: ==================== EXIT ====================
:EXIT_PROGRAM
cls
echo.
echo ================================================================
echo           Thank you for using AIv3 Analysis Suite!
echo ================================================================
echo.
echo.
timeout /t 2 >nul
exit /b 0