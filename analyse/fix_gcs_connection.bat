@echo off
REM Fix GCS SSL Connection Issues
REM This sets environment variables to bypass proxy issues

echo Setting environment variables to fix GCS connection...

REM Disable proxy for Google APIs
set NO_PROXY=googleapis.com,*.googleapis.com,metadata.google.internal
set HTTPS_PROXY=
set HTTP_PROXY=

REM Force use of system certificates
set REQUESTS_CA_BUNDLE=
set CURL_CA_BUNDLE=

REM Use alternative OAuth flow that might work better
set GOOGLE_AUTH_DISABLE_MTLS=1

echo.
echo [OK] Environment configured
echo.
echo Now running the multi-source analysis...
echo.

REM Call the actual analysis script
call run_multi_source_analysis.bat
