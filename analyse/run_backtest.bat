@echo off
REM =====================================================
REM AIv3 System - Backtesting
REM =====================================================

echo.
echo ========================================
echo AIv3 Backtesting System
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
set INPUT_FILE=
set START_DATE=
set END_DATE=
set OUTPUT_FILE=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto check_input
if /i "%~1"=="--input" (
    set INPUT_FILE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--start" (
    set START_DATE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--end" (
    set END_DATE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--output" (
    set OUTPUT_FILE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    goto show_help
)
shift
goto parse_args

:check_input
REM If no input file specified, use historical patterns
if "%INPUT_FILE%"=="" (
    if exist "historical_patterns.parquet" (
        set INPUT_FILE=historical_patterns.parquet
        echo Using historical_patterns.parquet
    ) else (
        echo ERROR: No pattern file specified!
        echo Please specify --input file or ensure historical_patterns.parquet exists
        pause
        exit /b 1
    )
)

REM Set default date range if not specified
if "%START_DATE%"=="" (
    REM Default to 1 year ago
    set START_DATE=2023-01-01
)

if "%END_DATE%"=="" (
    REM Default to today
    for /f "tokens=1-3 delims=/" %%a in ('date /t') do (
        set END_DATE=%%c-%%a-%%b
    )
)

REM Create output filename if not specified
if "%OUTPUT_FILE%"=="" (
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"
    set OUTPUT_FILE=output\backtest_results_%timestamp%.json
)

:run_backtest
echo.
echo Starting Backtest...
echo.
echo Configuration:
echo   Input File: %INPUT_FILE%
echo   Start Date: %START_DATE%
echo   End Date: %END_DATE%
echo   Output File: %OUTPUT_FILE%
echo.

REM Build command
set CMD=python main.py backtest --input "%INPUT_FILE%"

if not "%START_DATE%"=="" (
    set CMD=%CMD% --start-date %START_DATE%
)

if not "%END_DATE%"=="" (
    set CMD=%CMD% --end-date %END_DATE%
)

if not "%OUTPUT_FILE%"=="" (
    set CMD=%CMD% --output "%OUTPUT_FILE%"
)

REM Run backtest
echo Running command: %CMD%
echo.
%CMD%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Backtest failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Backtest Complete!
echo Results saved to: %OUTPUT_FILE%
echo ========================================
echo.

REM Ask what to do next
echo What would you like to do next?
echo.
echo 1. Run backtest with different date range
echo 2. Generate backtest report
echo 3. View results
echo 4. Exit
echo.
choice /C 1234 /M "Select option"

if %ERRORLEVEL%==1 (
    echo.
    set /p START_DATE="Enter start date (YYYY-MM-DD): "
    set /p END_DATE="Enter end date (YYYY-MM-DD): "
    goto run_backtest
)

if %ERRORLEVEL%==2 (
    echo.
    echo Generating backtest report...
    python main.py report --input "%INPUT_FILE%" --type backtest --title "Backtest Results %START_DATE% to %END_DATE%"
    pause
)

if %ERRORLEVEL%==3 (
    echo.
    echo Opening results file...
    start notepad "%OUTPUT_FILE%"
)

exit /b 0

:show_help
echo.
echo Usage: run_backtest.bat [options]
echo.
echo Options:
echo   --input [FILE]     Input pattern file (default: historical_patterns.parquet)
echo   --start [DATE]     Start date for backtest (YYYY-MM-DD)
echo   --end [DATE]       End date for backtest (YYYY-MM-DD)
echo   --output [FILE]    Output results file (JSON)
echo   --help             Show this help message
echo.
echo Examples:
echo   run_backtest.bat
echo   run_backtest.bat --input patterns.parquet --start 2023-01-01
echo   run_backtest.bat --start 2022-01-01 --end 2023-12-31
echo.
pause
exit /b 0