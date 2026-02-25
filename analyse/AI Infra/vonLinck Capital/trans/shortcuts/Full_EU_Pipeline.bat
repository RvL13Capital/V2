@echo off
title Full EU Pipeline
cd /d "%~dp0.."
echo ============================================================
echo                 FULL EU PIPELINE
echo ============================================================
echo.
echo WARNING: This runs the complete pipeline:
echo   - Pattern detection
echo   - Sequence generation
echo   - Model training
echo.
echo This may take 30+ minutes!
echo.
set /p CONFIRM="Continue? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto :end
echo.
echo Starting full EU pipeline...
echo.
python scripts/run_external.py --name full_eu "python pipeline/run_robust.py --eu --both"
echo.
echo ============================================================
echo Pipeline complete! Check output/tasks/ for results.
echo ============================================================
:end
pause
