@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0model_server"

echo ================================================================
echo   RemoteCLIPVAE  -  step 2 : start model server (console)
echo ================================================================
echo.

rem ---------- 1) python remembered by 0_ / 1_ ----------
set "PY="
set "CAND="
if exist "_python.txt" set /p CAND=<_python.txt
if defined CAND set "CAND=!CAND:"=!"
if defined CAND call :probe "!CAND!"
set "CAND="

rem ---------- 2) comfyui folder remembered ----------
set "CF="
if not defined PY if exist "_comfy.txt" set /p CF=<_comfy.txt
if defined CF set "CF=!CF:"=!"
if not defined PY if defined CF call :trydir "!CF!"

rem ---------- 3) ask once ----------
if defined PY goto :havepy
echo   [x] No ComfyUI python recorded yet.
echo.
echo   Drag the ComfyUI folder here ^(the one containing "comfy" and "main.py"^)
echo   and press Enter - the server MUST run with ComfyUI's own python
echo   ^(python_embeded\python.exe / python_embedded / .venv\Scripts\python.exe^).
echo.
set "RAW="
set /p RAW=  ComfyUI folder:
if not defined RAW goto :trypath
for /f "delims=" %%Q in ("!RAW!") do set "RAW=%%~Q"
call :trydir "!RAW!"
if not defined PY goto :trypath
>_comfy.txt echo !RAW!
>_python.txt echo !PY!
goto :havepy

rem ---------- 4) last resort: python on PATH (often a broken store install) ----------
:trypath
for /f "delims=" %%P in ('"%SystemRoot%\System32\where.exe" python 2^>nul') do (
  if not defined PY call :probe "%%~P"
)
if not defined PY goto :nopy

:havepy
echo   python : !PY!
echo   config : config.json
echo.

if defined CHECKONLY (
  echo [check] would run: "!PY!" app.py --config config.json %*
  exit /b 0
)

echo   Tip: add  --mock  to test the network path without loading models.
echo ------------------------------------------------------------------
"!PY!" app.py --config config.json %*

echo ------------------------------------------------------------------
echo Server stopped.
pause
exit /b 0

:nopy
echo [ERROR] No usable ComfyUI python found.
echo.
echo   服务端必须用 ComfyUI 自带的 python，不能随便抓一个 PATH 里的 python：
echo     PATH 上的 Microsoft Store 版 python 经常装着一个残缺的 torch
echo     ^(报 ModuleNotFoundError: No module named 'torchgen' 就是这种情况^)。
echo.
echo   三种做法，任选一种：
echo     1^) 先双击  0_启动图形界面.bat ，在面板里选 ComfyUI 路径、点「自动找」；
echo     2^) 先双击  1_检查环境并生成配置.bat ，把 ComfyUI 目录拖进去；
echo     3^) 直接  4_用ComfyUI的Python启动服务端.bat "C:\你的\ComfyUI"
echo.
pause
exit /b 1

:probe
if defined PY goto :eof
if not exist %1 goto :eof
%1 -c "import numpy, torch" >nul 2>nul
if errorlevel 1 (
  echo   [skip] 这个 python 跑不了服务端（缺 torch / 装坏了）: %~1
  goto :eof
)
set "PY=%~1"
goto :eof

:trydir
if defined PY goto :eof
call :probe "%~1\python_embeded\python.exe"
call :probe "%~1\python_embedded\python.exe"
call :probe "%~1\python\python.exe"
call :probe "%~1\.venv\Scripts\python.exe"
call :probe "%~1\venv\Scripts\python.exe"
call :probe "%~1\..\python_embeded\python.exe"
call :probe "%~1\..\..\python_embeded\python.exe"
call :probe "%~1\..\.venv\Scripts\python.exe"
goto :eof
