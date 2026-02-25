@echo off
REM Quick fix for missing pyarrow dependency

echo Installing pyarrow for parquet file support...
call venv\Scripts\activate.bat
pip install pyarrow

echo.
echo Done! You can now run the analysis.
pause