@echo off
title Generate EU Sequences
cd /d "%~dp0.."
echo ============================================================
echo               GENERATE EU SEQUENCES
echo ============================================================
echo.
echo This will generate temporal sequences from EU patterns.
echo.
python scripts/run_external.py --name generate_eu "python pipeline/01_generate_sequences.py --input output/eu/detected_patterns.parquet --output-dir output/eu/sequences --skip-npy-export"
echo.
echo ============================================================
echo Generation complete! Check output/tasks/ for results.
echo ============================================================
pause
