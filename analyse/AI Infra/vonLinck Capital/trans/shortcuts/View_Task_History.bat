@echo off
title Task History
cd /d "%~dp0.."
echo ============================================================
echo                    TASK HISTORY
echo ============================================================
echo.
python scripts/check_task.py --list 10
echo.
echo ============================================================
echo.
set /p TASK="Enter task ID to view details (or press Enter to exit): "
if "%TASK%"=="" goto :end
python scripts/check_task.py %TASK% --tail 30
:end
pause
