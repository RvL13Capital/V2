@echo off
REM =====================================================
REM AIv3 System - Pattern Analysis
REM =====================================================

echo.
echo ========================================
echo AIv3 Pattern Analysis System
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
set ANALYSIS_TYPE=statistical
set GENERATE_REPORT=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto check_input
if /i "%~1"=="--input" (
    set INPUT_FILE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--type" (
    set ANALYSIS_TYPE=%~2
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

:check_input
REM If no input file specified, try to find the latest patterns file
if "%INPUT_FILE%"=="" (
    echo No input file specified. Looking for latest patterns file...
    for /f "delims=" %%i in ('dir /b /od output\patterns_*.parquet 2^>nul') do set "INPUT_FILE=output\%%i"

    if "%INPUT_FILE%"=="" (
        REM Try historical patterns as fallback
        if exist "historical_patterns.parquet" (
            set INPUT_FILE=historical_patterns.parquet
            echo Using historical_patterns.parquet
        ) else (
            echo ERROR: No pattern files found!
            echo Please run pattern detection first or specify --input file
            pause
            exit /b 1
        )
    ) else (
        echo Found: %INPUT_FILE%
    )
)

:run_analysis
echo.
echo Starting Analysis...
echo.
echo Configuration:
echo   Input File: %INPUT_FILE%
echo   Analysis Type: %ANALYSIS_TYPE%
echo   Generate Report: %GENERATE_REPORT%
echo.

REM Build command
set CMD=python main.py analyze --input "%INPUT_FILE%" --type %ANALYSIS_TYPE%

if "%GENERATE_REPORT%"=="true" (
    set CMD=%CMD% --report
)

REM Run analysis
echo Running command: %CMD%
echo.
%CMD%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Analysis failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Analysis Complete!
echo ========================================
echo.

REM Ask what to do next
echo What would you like to do next?
echo.
echo 1. Run different analysis type
echo 2. Generate report
echo 3. Run backtest
echo 4. Exit
echo.
choice /C 1234 /M "Select option"

if %ERRORLEVEL%==1 (
    echo.
    echo Select analysis type:
    echo 1. Statistical
    echo 2. Performance
    echo 3. Quality
    choice /C 123 /M "Select type"
    if %ERRORLEVEL%==1 set ANALYSIS_TYPE=statistical
    if %ERRORLEVEL%==2 set ANALYSIS_TYPE=performance
    if %ERRORLEVEL%==3 set ANALYSIS_TYPE=quality
    goto run_analysis
)

if %ERRORLEVEL%==2 (
    echo.
    python main.py report --input "%INPUT_FILE%" --type pattern
    pause
)

if %ERRORLEVEL%==3 (
    echo.
    call run_backtest.bat --input "%INPUT_FILE%"
)

exit /b 0

:show_help
echo.
echo Usage: run_analysis.bat [options]
echo.
echo Options:
echo   --input [FILE]     Input pattern file (parquet/csv)
echo   --type [TYPE]      Analysis type: statistical, performance, quality
echo   --report           Generate PDF report after analysis
echo   --help             Show this help message
echo.
echo Analysis Types:
echo   statistical - Basic statistics and distributions
echo   performance - Focus on high-performing patterns
echo   quality     - Analyze pattern quality scores
echo.
echo Examples:
echo   run_analysis.bat
echo   run_analysis.bat --input patterns.parquet --type performance
echo   run_analysis.bat --type quality --report
echo.
pause
exit /b 0