@echo off
title Run Test Suite
cd /d "%~dp0.."
echo ============================================================
echo                    RUN TEST SUITE
echo ============================================================
echo.
echo Running all pytest tests...
echo.
python -m pytest tests/ -v --tb=short
echo.
echo ============================================================
echo Tests complete!
echo ============================================================
pause
