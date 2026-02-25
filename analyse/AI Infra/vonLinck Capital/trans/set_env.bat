@echo off
REM Set GCS credentials for TRANS system
REM Run this before using the system: call set_env.bat

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\vonLinck Capital\trans\gcs_credentials.json
set PROJECT_ID=ignition-ki-csv-storage
set GCS_BUCKET_NAME=ignition-ki-csv-data-2025-user123

echo.
echo ========================================
echo TRANS Environment Configured
echo ========================================
echo GCS Credentials: %GOOGLE_APPLICATION_CREDENTIALS%
echo Project ID: %PROJECT_ID%
echo Bucket: %GCS_BUCKET_NAME%
echo.
echo Environment variables set for this session.
echo You can now run Python scripts.
echo.
