@echo off
set PYTHON=python_env\python.exe
if not exist %PYTHON% set PYTHON=python

if "%1"=="predict" goto PREDICT
if "%1"=="server" goto SERVER
if "%1"=="test" goto TEST

echo Использование:
echo   make.bat predict IN=tickets.jsonl OUT=pred.jsonl
echo   make.bat server
echo   make.bat test
goto END

:PREDICT
shift
%PYTHON% predict.py %*
goto END

:SERVER
%PYTHON% app.py
goto END

:TEST
%PYTHON% test_system.py
goto END

:END
