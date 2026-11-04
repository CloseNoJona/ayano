@echo off
rem Compila el motor. Uso: scripts\build.cmd [release^|portable^|debug]
setlocal
set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=release"
call "%~dp0vs.cmd" cmake --preset %PRESET%
if errorlevel 1 exit /b 1
call "%~dp0vs.cmd" cmake --build --preset %PRESET%
exit /b %ERRORLEVEL%
