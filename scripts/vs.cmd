@echo off
rem Ejecuta un comando dentro del entorno de Visual Studio (cl, cmake, ninja).
rem Uso: scripts\vs.cmd cmake --build --preset release
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo No se encontro vswhere. Instala Visual Studio Build Tools con C++. 1>&2
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
  echo No hay herramientas de C++ instaladas en Visual Studio. 1>&2
  exit /b 1
)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
%*
exit /b %ERRORLEVEL%
