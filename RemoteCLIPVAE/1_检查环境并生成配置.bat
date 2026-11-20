@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================================
echo   RemoteCLIPVAE  -  step 1 : check environment + write config
echo ================================================================
echo.
echo This kit does NOT create a venv on purpose.
echo The model server must run on ComfyUI's OWN python, because it
echo loads CLIP/VAE through ComfyUI's comfy.sd (same weights, same
echo math as the UNet on the other machine).
echo.

set "CF=%~1"
if not "%CF%"=="" goto have_cf

echo Drag your ComfyUI folder here - the one that CONTAINS the "comfy"
echo folder, for example:
echo   F:\ComfyUI_Mie_2026_V8.0_Base\ComfyUI_Mie_2026_V8.0_Base\ComfyUI
echo.
set /p "CF=Path: "

:have_cf
set "CF=%CF:"=%"
if "%CF%"=="" goto abort
if "%CF:~-1%"=="\" set "CF=%CF:~0,-1%"
if not exist "%CF%\comfy\sd.py" (
  echo.
  echo [ERROR] no comfy\sd.py in: %CF%
  echo         Point it at the folder that contains the "comfy" folder.
  pause
  exit /b 1
)

rem ---------- locate ComfyUI's own python ----------
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
) do if not defined PY if exist %%~P set "PY=%%~P"

if defined PY goto have_py
echo.
echo [WARN] no python_embeded / .venv found around that folder.
set /p "PY=Path to ComfyUI python.exe: "
set "PY=%PY:"=%"

:have_py
if not exist "%PY%" (
  echo.
  echo [ERROR] python not found: %PY%
  pause
  exit /b 1
)

echo.
echo   comfy  : %CF%
echo   python : %PY%
echo.

if defined CHECKONLY (
  echo [check] ok
  exit /b 0
)

echo ------------------------------------------------------------------
pushd "%~dp0model_server"
"%PY%" config.py --comfy-path "%CF%" --python "%PY%" --write --add-models
set "RC=%ERRORLEVEL%"
popd
echo ------------------------------------------------------------------

if not "%RC%"=="0" (
  echo.
  echo [ERROR] config failed. See messages above.
  pause
  exit /b 1
)

echo.
echo Done. Config written to: model_server\config.json
> "model_server\_comfy.txt" echo %CF%
echo Done. Remembered ComfyUI folder: model_server\_comfy.txt
echo Next:
echo   2_启动服务端.bat            start the server
echo   2_启动服务端.bat --mock     test the wiring without loading models
echo.
pause
exit /b 0

:abort
echo Cancelled.
pause
exit /b 1
