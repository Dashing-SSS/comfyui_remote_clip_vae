@echo off
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0model_server"

echo ================================================================
echo   RemoteCLIPVAE  -  step 5 : end-to-end selftest
echo ================================================================
echo.
echo Runs the real node logic against the server: /health -> CLIP
echo encode -> VAE decode -> VAE encode. Start the server first.
echo.

set "PY="
if exist "_python.txt" set /p PY=<_python.txt
if defined PY if not exist "%PY%" set "PY="
if not defined PY (
  for %%P in ("..\..\python_embeded\python.exe" "..\..\.venv\Scripts\python.exe" "..\..\..\python_embeded\python.exe") do (
    if not defined PY if exist "%%~P" set "PY=%%~P"
  )
)
if not defined PY (
  "%SystemRoot%\System32\where.exe" python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo [ERROR] No python found. Run 1_ first.
  pause
  exit /b 1
)

set "URL="
set /p "URL=Server URL [http://127.0.0.1:8199]: "
if "%URL%"=="" set "URL=http://127.0.0.1:8199"

set "TK="
set /p "TK=Token [empty = none]: "
if not "%TK%"=="" set "TK=--token %TK%"

echo.
if defined CHECKONLY (
  echo [check] "%PY%" selftest.py --url "%URL%" %TK% %*
  exit /b 0
)

"%PY%" selftest.py --url "%URL%" %TK% %*
echo.
echo Exit code: %ERRORLEVEL%
pause
exit /b 0
