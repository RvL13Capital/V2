@echo off
echo ============================================================
echo     INSTALLING CLOUD ANALYSIS DEPENDENCIES
echo ============================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

echo This will install the following packages:
echo   - google-cloud-bigquery (for BigQuery processing)
echo   - google-cloud-storage (for GCS access)
echo   - pyarrow (for Parquet file support)
echo   - pandas-gbq (for BigQuery pandas integration)
echo.

set /p confirm="Continue with installation? (Y/N): "
if /i not "%confirm%"=="Y" exit /b 0

echo.
echo [1/5] Installing google-cloud-bigquery...
pip install google-cloud-bigquery --quiet

echo [2/5] Installing google-cloud-storage...
pip install google-cloud-storage --quiet

echo [3/5] Installing pyarrow...
pip install pyarrow --quiet

echo [4/5] Installing pandas-gbq...
pip install pandas-gbq --quiet

echo [5/5] Installing db-dtypes (BigQuery data types)...
pip install db-dtypes --quiet

echo.
echo ============================================================
echo Verifying installation...
echo ============================================================
echo.

python -c "import google.cloud.bigquery; print('✓ BigQuery module: OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ BigQuery module: FAILED
    set failed=1
) else (
    echo ✓ BigQuery module: OK
)

python -c "import google.cloud.storage; print('✓ Storage module: OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Storage module: FAILED
    set failed=1
) else (
    echo ✓ Storage module: OK
)

python -c "import pyarrow; print('✓ PyArrow module: OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ PyArrow module: FAILED
    set failed=1
) else (
    echo ✓ PyArrow module: OK
)

python -c "import pandas_gbq; print('✓ Pandas-GBQ module: OK')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Pandas-GBQ module: FAILED
    set failed=1
) else (
    echo ✓ Pandas-GBQ module: OK
)

echo.
if defined failed (
    echo ============================================================
    echo INSTALLATION INCOMPLETE - Some modules failed
    echo ============================================================
    echo.
    echo Try running: pip install --upgrade google-cloud-bigquery google-cloud-storage pyarrow pandas-gbq
    echo.
) else (
    echo ============================================================
    echo INSTALLATION SUCCESSFUL!
    echo ============================================================
    echo.
    echo All cloud dependencies are installed.
    echo You can now run the cloud-optimized analysis.
    echo.
)

pause