@echo off
rem Compilacion guiada por perfil: compila, mide con bench y recompila optimizando
rem los caminos que mas se usan. Resultado: build\pgo\enroque.exe
setlocal
set "ROOT=%~dp0.."
set "BUILD=%ROOT%\build\pgo"
call "%~dp0vs.cmd" cmake -S "%ROOT%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DENROQUE_NATIVE=ON -DENROQUE_PGO=GENERATE
if errorlevel 1 exit /b 1
call "%~dp0vs.cmd" cmake --build "%BUILD%" --target enroque
if errorlevel 1 exit /b 1
call "%~dp0vs.cmd" "%BUILD%\enroque.exe" bench 14
if errorlevel 1 exit /b 1
call "%~dp0vs.cmd" cmake -S "%ROOT%" -B "%BUILD%" -DENROQUE_PGO=USE
if errorlevel 1 exit /b 1
call "%~dp0vs.cmd" cmake --build "%BUILD%" --target enroque
exit /b %ERRORLEVEL%
