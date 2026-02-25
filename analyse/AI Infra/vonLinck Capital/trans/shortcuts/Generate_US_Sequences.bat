@echo off
title Generate US Sequences
cd /d "%~dp0.."
echo ============================================================
echo               GENERATE US SEQUENCES
echo ============================================================
echo.
echo This will generate temporal sequences from US patterns.
echo.
python scripts/run_external.py --name generate_us "python pipeline/01_generate_sequences.py --input output/us/detected_patterns.parquet --output-dir output/us/sequences --skip-npy-export"
echo.
echo ============================================================
echo Generation complete! Check output/tasks/ for results.
echo ============================================================
pause
