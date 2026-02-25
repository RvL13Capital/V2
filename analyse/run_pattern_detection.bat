@echo off
REM =====================================================
REM AIv3 System - Pattern Detection
REM =====================================================

echo.
echo ========================================
echo AIv3 Pattern Detection System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_environment.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set default parameters
set TICKERS=ALL
set LIMIT=100
set OUTPUT_DIR=output
set GENERATE_REPORT=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto run_detection
if /i "%~1"=="--tickers" (
    set TICKERS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--limit" (
    set LIMIT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--report" (
    set GENERATE_REPORT=true
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    goto show_help
)
shift
goto parse_args

:run_detection
echo Starting Pattern Detection...
echo.
echo Configuration:
echo   Tickers: %TICKERS%
echo   Limit: %LIMIT%
echo   Generate Report: %GENERATE_REPORT%
echo.

REM Create timestamp for output files
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"

REM Create output filename
set OUTPUT_FILE=%OUTPUT_DIR%\patterns_%timestamp%.parquet

REM Build command
set CMD=python main.py detect --tickers %TICKERS% --limit %LIMIT% --output %OUTPUT_FILE%

if "%GENERATE_REPORT%"=="true" (
    set CMD=%CMD% --report
)

REM Run detection
echo Running command: %CMD%
echo.
%CMD%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Pattern detection failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Pattern Detection Complete!
echo Results saved to: %OUTPUT_FILE%
echo ========================================
echo.

REM Ask if user wants to run analysis
choice /C YN /M "Do you want to analyze these patterns now?"
if %ERRORLEVEL%==1 (
    echo.
    call run_analysis.bat --input "%OUTPUT_FILE%"
)

pause
exit /b 0

:show_help
echo.
echo Usage: run_pattern_detection.bat [options]
echo.
echo Options:
echo   --tickers [LIST]   Comma-separated ticker list or ALL (default: ALL)
echo   --limit [N]        Limit number of tickers to process (default: 100)
echo   --report           Generate PDF report after detection
echo   --help             Show this help message
echo.
echo Examples:
echo   run_pattern_detection.bat
echo   run_pattern_detection.bat --tickers AAPL,GOOGL,MSFT --report
echo   run_pattern_detection.bat --limit 50 --report
echo.
pause
exit /b 0