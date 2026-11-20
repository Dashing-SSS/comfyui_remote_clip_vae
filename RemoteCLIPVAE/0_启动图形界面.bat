@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================================
echo   RemoteCLIPVAE  -  B side control panel  (start / stop / VRAM)
echo ================================================================
rem ---------- 0) 打包好的桌面程序优先 ----------
if exist "%~dp0RemoteCLIPVAE服务端.exe" (
  start "" "%~dp0RemoteCLIPVAE服务端.exe" %*
  exit /b 0
)

echo.
rem ---------- 1) any python, for the web panel (standard library only) ----------
set "ANYPY="
if exist "model_server\_python.txt" (
  set /p A=<model_server\_python.txt
  set "A=!A:"=!"
  call :checkpy "!A!"
)
if not defined ANYPY if exist "model_server\_comfy.txt" (
  set /p CF=<model_server\_comfy.txt
  set "CF=!CF:"=!"
  call :probe "!CF!"
)
if not defined ANYPY for /f "delims=" %%P in ('"%SystemRoot%\System32\where.exe" python 2^>nul') do call :checkpy "%%~P"
if defined ANYPY goto :havepy

echo   [x] No python found on PATH.
rem ---------- 0) 打包好的桌面程序优先 ----------
if exist "%~dp0RemoteCLIPVAE服务端.exe" (
  start "" "%~dp0RemoteCLIPVAE服务端.exe" %*
  exit /b 0
)

echo.echo   Drag the ComfyUI folder here ^(the one containing "comfy" and "main.py"^)
echo   and press Enter, I will use its python_embeded\python.exe ^(or python_embedded / .venv^):
rem ---------- 0) 打包好的桌面程序优先 ----------
if exist "%~dp0RemoteCLIPVAE服务端.exe" (
  start "" "%~dp0RemoteCLIPVAE服务端.exe" %*
  exit /b 0
)

echo.set "RAWCOM="
set /p RAWCOM=  ComfyUI folder:
if not defined RAWCOM goto :pyfallback
for /f "delims=" %%Q in ("!RAWCOM!") do set "RAWCOM=%%~Q"
call :probe "!RAWCOM!"
if not defined ANYPY goto :pyfallback
type nul >model_server\_comfy.txt
>>model_server\_comfy.txt echo !RAWCOM!
type nul >model_server\_python.txt
>>model_server\_python.txt echo !ANYPY!
goto :havepy

:pyfallback
set "ANYPY=python"

:havepy
rem ---------- 2) a python with tkinter + working Tcl, for the desktop window ----------
set "GUIPY="
if exist "model_server\_gui_python.txt" (
  set /p G=<model_server\_gui_python.txt
  set "G=!G:"=!"
  call :checktk "!G!"
)
if not defined GUIPY call :checktk "!ANYPY!"
if not defined GUIPY if exist "model_server\_comfy.txt" (
  set /p CF=<model_server\_comfy.txt
  call :probe "!CF!"
  if defined ANYPY call :checktk "!ANYPY!"
)
if not defined GUIPY for /f "delims=" %%P in ('"%SystemRoot%\System32\where.exe" python 2^>nul') do call :checktk "%%~P"
if not defined GUIPY for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python3*\python.exe"
  "C:\Python3*\python.exe"
  "C:\Program Files\Python3*\python.exe"
  "C:\Program Files (x86)\Python3*\python.exe"
) do call :checktk "%%~P"

rem ---------- 3) desktop window when tkinter works ----------
if defined GUIPY (
  echo   mode   : desktop window ^(tkinter^)
  echo   python : !GUIPY!
  rem ---------- 0) 打包好的桌面程序优先 ----------
if exist "%~dp0RemoteCLIPVAE服务端.exe" (
  start "" "%~dp0RemoteCLIPVAE服务端.exe" %*
  exit /b 0
)

echo.  type nul >model_server\_gui_python.txt
  >>model_server\_gui_python.txt echo !GUIPY!
  set "PYW="
  for %%D in ("!GUIPY!") do set "PYDIR=%%~dpD"
  if exist "!PYDIR!pythonw.exe" set "PYW=!PYDIR!pythonw.exe"
  if defined PYW if /i not "%~1"=="--console" if /i not "%~1"=="--smoke" (
    start "" "!PYW!" "model_server\gui.py" %*
    exit /b 0
  )
  "!GUIPY!" "model_server\gui.py" %*
  if errorlevel 1 pause
  exit /b 0
)

rem ---------- 4) fall back to the local web panel (no tkinter needed) ----------
echo   mode   : local web panel ^(a browser window will open^)
echo   python : !ANYPY!
rem ---------- 0) 打包好的桌面程序优先 ----------
if exist "%~dp0RemoteCLIPVAE服务端.exe" (
  start "" "%~dp0RemoteCLIPVAE服务端.exe" %*
  exit /b 0
)

echo."!ANYPY!" "model_server\panel.py" %*
if errorlevel 1 (
  rem ---------- 0) 打包好的桌面程序优先 ----------
if exist "%~dp0RemoteCLIPVAE服务端.exe" (
  start "" "%~dp0RemoteCLIPVAE服务端.exe" %*
  exit /b 0
)

echo.  echo [ERROR] panel exited with code %ERRORLEVEL%
  echo         If nothing opened, run 1_ first, or install Python 3.
  pause
)
exit /b 0

:checktk
if defined GUIPY goto :eof
if not exist %1 goto :eof
%1 -c "import tkinter;tkinter.Tcl()" >nul 2>nul
if not errorlevel 1 set "GUIPY=%~1"
goto :eof

:checkpy
if defined ANYPY goto :eof
if not exist %1 goto :eof
%1 -c "import sys" >nul 2>nul
if not errorlevel 1 set "ANYPY=%~1"
goto :eof

rem probe a ComfyUI folder for its python; also accepts a python.exe path
:probe
if defined ANYPY goto :eof
call :checkpy "%1"
call :checkpy "%~1\python_embeded\python.exe"
call :checkpy "%~1\python_embedded\python.exe"
call :checkpy "%~1\..\python_embeded\python.exe"
call :checkpy "%~1\..\..\python_embeded\python.exe"
call :checkpy "%~1\.venv\Scripts\python.exe"
goto :eof
