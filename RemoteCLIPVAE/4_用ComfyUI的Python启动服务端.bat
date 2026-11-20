@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0model_server"

echo ================================================================
echo   RemoteCLIPVAE  -  step 4 : start server with ComfyUI's python
echo ================================================================
echo.
echo Same server as step 2, but you point it at a ComfyUI folder and it
echo runs through run_headless.py --backend comfy --comfy-path ...
echo All logs stay in this console.
echo.

set "A1=%~1"
set "CF="
set "EXTRAS="
if "%A1%"=="" goto ask_cf
if not "%A1:~0,1%"=="-" goto take_path

rem first arg is a flag (--mock ...): use the folder remembered by step 1
if exist "_comfy.txt" set /p CF=<_comfy.txt
if defined CF goto grab_rest
goto ask_cf

:take_path
set "CF=%A1%"
shift

:grab_rest
if "%~1"=="" goto have_cf
set "EXTRAS=%EXTRAS% %~1"
shift
goto grab_rest

:ask_cf
echo Drag the ComfyUI folder here - the one CONTAINING the "comfy" folder.
echo.
set /p "CF=Path: "

:have_cf
set "CF=%CF:"=%"
if "%CF%"=="" goto abort
if "%CF:~-1%"=="\" set "CF=%CF:~0,-1%"
if not exist "%CF%\comfy\sd.py" (
  echo.
  echo [ERROR] this folder has no comfy\sd.py: %CF%
  echo         Point it at the folder that contains the "comfy" folder.
  pause
  exit /b 1
)

rem ---------- find ComfyUI's own python ----------
set "PY="
for %%P in (
  "%CF%\python_embeded\python.exe"
  "%CF%\python_embedded\python.exe"
  "%CF%\python\python.exe"
  "%CF%\.venv\Scripts\python.exe"
  "%CF%\venv\Scripts\python.exe"
  "%CF%\..\python_embeded\python.exe"
  "%CF%\..\.venv\Scripts\python.exe"
  "%CF%\..\..\python_embeded\python.exe"
  "%CF%\..\..\.venv\Scripts\python.exe"
) do if not defined PY if exist "%%~P" set "PY=%%~P"

if defined PY goto have_py
echo.
echo [WARN] no python_embeded / .venv found around that folder.
set /p "PY=Path to ComfyUI python.exe: "
set "PY=%PY:"=%"

:have_py
if not exist "%PY%" (
  echo.
  echo [ERROR] not found: %PY%
  pause
  exit /b 1
)

"%PY%" -c "import numpy, torch" >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] 这个 python 不能跑服务端（import numpy, torch 失败）：
  echo         %PY%
  echo.
  echo         多半是 PATH 上的 Microsoft Store 版 python 装了个残缺的 torch
  echo         ^(报 No module named 'torchgen' 就是这种^)。
  echo         请用 ComfyUI 自带的那份 python_embeded\python.exe / .venv\Scripts\python.exe。
  pause
  exit /b 1
)

echo.
echo   comfy  : %CF%
echo   python : %PY%
echo.

if defined CHECKONLY (
  echo [check] would run: "%PY%" run_headless.py --backend comfy --comfy-path "%CF%" %EXTRAS%
  exit /b 0
)

echo ------------------------------------------------------------------
"%PY%" run_headless.py --backend comfy --comfy-path "%CF%" %EXTRAS%

echo ------------------------------------------------------------------
echo Server stopped.
pause
exit /b 0

:abort
echo Cancelled.
pause
exit /b 1
