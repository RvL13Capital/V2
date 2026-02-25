@echo off
title Train US Model
cd /d "%~dp0.."
echo ============================================================
echo                    TRAIN US MODEL
echo ============================================================
echo.
echo This will train the US temporal model with 100 epochs.
echo.
set /p EPOCHS="Epochs (default 100): " || set EPOCHS=100
if "%EPOCHS%"=="" set EPOCHS=100
echo.
echo Starting training with %EPOCHS% epochs...
echo.
python scripts/run_external.py --name train_us "python pipeline/02_train_temporal.py --sequences output/sequences/us/sequences_20251225_105740.h5 --metadata output/sequences/us/metadata_20251225_105740.parquet --epochs %EPOCHS%"
echo.
echo ============================================================
echo Training complete! Check output/tasks/ for results.
echo ============================================================
pause
