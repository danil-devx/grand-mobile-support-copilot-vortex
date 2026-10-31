@echo off
chcp 65001 >nul
echo ========================================================
echo   Запуск Grand Mobile Support Copilot + Демо Web UI
echo ========================================================
echo.
echo Сервер запускается на http://localhost:8000 ...
echo Для остановки нажмите Ctrl+C
echo.
python_env\python.exe app.py
pause
