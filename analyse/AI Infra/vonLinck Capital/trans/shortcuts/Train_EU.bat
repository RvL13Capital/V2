@echo off
title Train EU Model
cd /d "%~dp0.."
echo ============================================================
echo                    TRAIN EU MODEL
echo ============================================================
echo.
echo This will train the EU temporal model with 100 epochs.
echo.
set /p EPOCHS="Epochs (default 100): " || set EPOCHS=100
if "%EPOCHS%"=="" set EPOCHS=100
echo.
echo Starting training with %EPOCHS% epochs...
echo.
python scripts/run_external.py --name train_eu "python pipeline/02_train_temporal.py --sequences output/sequences/eu/sequences_20251228_184303.h5 --metadata output/sequences/eu/metadata_20251228_184303.parquet --epochs %EPOCHS%"
echo.
echo ============================================================
echo Training complete! Check output/tasks/ for results.
echo ============================================================
pause
