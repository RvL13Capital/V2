@echo off
setlocal enabledelayedexpansion

:: Set console colors
color 0A

echo.
echo ============================================================
echo          PARQUET DATA ANALYSIS TOOL
echo     Analyze and Visualize Parquet Data Files
echo ============================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

:: Check for required dependencies
echo [INFO] Checking required dependencies...
python -c "import pandas; import pyarrow" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Required dependencies not installed!
    echo Installing pandas and pyarrow...
    pip install pandas pyarrow plotly
)

:MAIN_MENU
cls
echo.
echo ============================================================
echo               PARQUET ANALYSIS MENU
echo ============================================================
echo.
echo   [1] ANALYZE SINGLE PARQUET FILE
echo   [2] ANALYZE ALL PARQUET FILES IN DIRECTORY
echo   [3] CONVERT PARQUET TO CSV
echo   [4] CONVERT PARQUET TO JSON
echo   [5] MERGE MULTIPLE PARQUET FILES
echo   [6] GENERATE STATISTICS REPORT
echo   [7] VISUALIZE DATA
echo   [8] FILTER AND QUERY PARQUET
echo   [0] EXIT
echo.
echo ============================================================
echo.

set /p choice="Select option [0-8]: "

if "%choice%"=="1" goto ANALYZE_SINGLE
if "%choice%"=="2" goto ANALYZE_ALL
if "%choice%"=="3" goto CONVERT_CSV
if "%choice%"=="4" goto CONVERT_JSON
if "%choice%"=="5" goto MERGE_FILES
if "%choice%"=="6" goto STATS_REPORT
if "%choice%"=="7" goto VISUALIZE
if "%choice%"=="8" goto FILTER_QUERY
if "%choice%"=="0" goto EXIT_SCRIPT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 >nul
goto MAIN_MENU

:ANALYZE_SINGLE
cls
echo.
echo ============================================================
echo           ANALYZE SINGLE PARQUET FILE
echo ============================================================
echo.
echo Available parquet files in current directory:
echo.
dir /b *.parquet 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] No parquet files found in current directory!
)
echo.
set /p filename="Enter parquet filename (or full path): "

if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto MAIN_MENU
)

echo.
echo [RUNNING] Analyzing %filename%...
python analyze_parquet_data.py --file "%filename%" --action analyze
echo.
pause
goto MAIN_MENU

:ANALYZE_ALL
cls
echo.
echo ============================================================
echo         ANALYZE ALL PARQUET FILES IN DIRECTORY
echo ============================================================
echo.
set /p dir_path="Enter directory path (or press Enter for current): "
if "%dir_path%"=="" set dir_path=.

echo.
echo [RUNNING] Analyzing all parquet files in %dir_path%...
python analyze_parquet_data.py --directory "%dir_path%" --action analyze-all
echo.
pause
goto MAIN_MENU

:CONVERT_CSV
cls
echo.
echo ============================================================
echo           CONVERT PARQUET TO CSV
echo ============================================================
echo.
set /p input_file="Enter parquet filename: "
if not exist "%input_file%" (
    echo [ERROR] File not found: %input_file%
    pause
    goto MAIN_MENU
)

set output_file=%input_file:.parquet=.csv%
set /p output_file="Enter output CSV filename [%output_file%]: "

echo.
echo [RUNNING] Converting to CSV...
python analyze_parquet_data.py --file "%input_file%" --action to-csv --output "%output_file%"
echo.
echo [COMPLETE] CSV file saved as: %output_file%
pause
goto MAIN_MENU

:CONVERT_JSON
cls
echo.
echo ============================================================
echo           CONVERT PARQUET TO JSON
echo ============================================================
echo.
set /p input_file="Enter parquet filename: "
if not exist "%input_file%" (
    echo [ERROR] File not found: %input_file%
    pause
    goto MAIN_MENU
)

echo.
echo Select JSON format:
echo   [1] Records (list of dictionaries)
echo   [2] Table (dict with columns and data)
echo   [3] Split (dict with index, columns, data)
echo   [4] Index (dict with index as keys)
echo.
set /p json_format="Select format [1-4]: "

if "%json_format%"=="1" set orient=records
if "%json_format%"=="2" set orient=table
if "%json_format%"=="3" set orient=split
if "%json_format%"=="4" set orient=index

set output_file=%input_file:.parquet=.json%
set /p output_file="Enter output JSON filename [%output_file%]: "

echo.
echo [RUNNING] Converting to JSON...
python analyze_parquet_data.py --file "%input_file%" --action to-json --output "%output_file%" --orient %orient%
echo.
echo [COMPLETE] JSON file saved as: %output_file%
pause
goto MAIN_MENU

:MERGE_FILES
cls
echo.
echo ============================================================
echo           MERGE MULTIPLE PARQUET FILES
echo ============================================================
echo.
echo Enter parquet files to merge (separated by commas):
echo Example: file1.parquet,file2.parquet,file3.parquet
echo.
set /p files="Files to merge: "
set /p output="Output filename: "

echo.
echo [RUNNING] Merging parquet files...
python analyze_parquet_data.py --files "%files%" --action merge --output "%output%"
echo.
echo [COMPLETE] Merged file saved as: %output%
pause
goto MAIN_MENU

:STATS_REPORT
cls
echo.
echo ============================================================
echo           GENERATE STATISTICS REPORT
echo ============================================================
echo.
set /p filename="Enter parquet filename: "
if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto MAIN_MENU
)

echo.
echo Select report type:
echo   [1] Basic Statistics (count, mean, std, min, max)
echo   [2] Detailed Report (including percentiles)
echo   [3] Data Quality Report (nulls, duplicates)
echo   [4] Full Analysis Report (all of the above)
echo.
set /p report_type="Select report type [1-4]: "

echo.
echo [RUNNING] Generating report...
python analyze_parquet_data.py --file "%filename%" --action stats --report-type %report_type%

echo.
set /p save_report="Save report to file? (Y/N): "
if /i "%save_report%"=="Y" (
    set report_file=%filename:.parquet=_report.txt%
    python analyze_parquet_data.py --file "%filename%" --action stats --report-type %report_type% --save "%report_file%"
    echo [SAVED] Report saved to: %report_file%
)

pause
goto MAIN_MENU

:VISUALIZE
cls
echo.
echo ============================================================
echo              VISUALIZE PARQUET DATA
echo ============================================================
echo.
set /p filename="Enter parquet filename: "
if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto MAIN_MENU
)

echo.
echo Select visualization type:
echo   [1] Distribution plots
echo   [2] Correlation heatmap
echo   [3] Time series plots (if date column exists)
echo   [4] Scatter plots
echo   [5] Interactive dashboard
echo.
set /p viz_type="Select visualization [1-5]: "

echo.
echo [RUNNING] Creating visualizations...
python analyze_parquet_data.py --file "%filename%" --action visualize --viz-type %viz_type%
echo.
pause
goto MAIN_MENU

:FILTER_QUERY
cls
echo.
echo ============================================================
echo           FILTER AND QUERY PARQUET DATA
echo ============================================================
echo.
set /p filename="Enter parquet filename: "
if not exist "%filename%" (
    echo [ERROR] File not found: %filename%
    pause
    goto MAIN_MENU
)

echo.
echo Enter filter conditions (SQL-like syntax):
echo Examples:
echo   - column_name > 100
echo   - column_name == 'value'
echo   - column_name.str.contains('pattern')
echo.
set /p query="Enter query: "

set /p output="Output filename (leave empty to display only): "

echo.
echo [RUNNING] Querying data...
if "%output%"=="" (
    python analyze_parquet_data.py --file "%filename%" --action query --query "%query%"
) else (
    python analyze_parquet_data.py --file "%filename%" --action query --query "%query%" --output "%output%"
    echo [SAVED] Filtered data saved to: %output%
)
echo.
pause
goto MAIN_MENU

:EXIT_SCRIPT
cls
echo.
echo ============================================================
echo       Thank you for using Parquet Analysis Tool!
echo ============================================================
echo.
timeout /t 2 >nul
exit /b 0