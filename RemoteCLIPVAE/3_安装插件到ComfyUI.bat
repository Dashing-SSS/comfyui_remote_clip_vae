@echo off
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0"

set "PLUGIN=comfyui_remote_clip_vae"
set "SRC=%~dp0%PLUGIN%"

echo ================================================================
echo   RemoteCLIPVAE  -  step 3 : install the ComfyUI custom node
echo ================================================================
echo.

if not exist "%SRC%\__init__.py" (
  echo [ERROR] plugin folder missing: %SRC%
  echo         Keep this .bat in the same folder as %PLUGIN%.
  pause
  exit /b 1
)

set "TARGET=%~1"
if not "%TARGET%"=="" goto have_target

echo Drag your ComfyUI "custom_nodes" folder onto this window and press
echo Enter, or type its full path, e.g.  D:\ComfyUI\custom_nodes
echo.
set /p "TARGET=Path: "

:have_target
set "TARGET=%TARGET:"=%"
if "%TARGET%"=="" goto abort
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET:~0,-1%"
if not exist "%TARGET%" (
  echo.
  echo [ERROR] folder not found: %TARGET%
  pause
  exit /b 1
)
if not exist "%TARGET%\..\main.py" (
  echo.
  echo [WARN] main.py not found in the parent folder - is this really custom_nodes?
  set /p "ANS=Copy anyway? [y/N]: "
  if /i not "%ANS%"=="y" goto abort
  if errorlevel 2 goto abort
)

echo.
echo Server address used by this plugin. The server window prints it too,
echo for example  http://192.168.3.2:8199
set "URL="
set /p "URL=URL [leave empty to keep default]: "
set "TK="
if not "%URL%"=="" set /p "TK=Token [empty = none]: "

if defined CHECKONLY (
  echo [check] copy %SRC%  to  %TARGET%\%PLUGIN%
  exit /b 0
)

echo.
set "XC=%SystemRoot%\System32\xcopy.exe"
if not exist "%XC%" set "XC=xcopy"
echo Copying to: %TARGET%\%PLUGIN%
"%XC%" "%SRC%" "%TARGET%\%PLUGIN%\" /E /I /Y /Q >nul
if errorlevel 1 (
  echo.
  echo [ERROR] copy failed - folder busy or no write permission.
  echo         Close ComfyUI and try again.
  pause
  exit /b 1
)

if not "%URL%"=="" (
  > "%TARGET%\%PLUGIN%\default_server.json" echo {"url": "%URL%", "token": "%TK%"}
  echo [OK] default server set to %URL%
)

echo.
echo [OK] installed: custom_nodes\%PLUGIN%
echo.
echo Next:
echo   1. Restart ComfyUI
echo   2. Node menu - category "comfyui_remote_clip_vae"  (5 nodes)
echo   3. Replace native CLIPTextEncode / VAELoader with the 远程 ones
echo      with the (远程) ones; KSampler stays untouched
echo.
pause
exit /b 0

:abort
echo Cancelled.
pause
exit /b 1
