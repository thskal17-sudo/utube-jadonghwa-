@echo off
REM ============================================================
REM  Educational Shorts Maker - USB portable setup (Windows)
REM  This file is intentionally ASCII-only so that cmd.exe cannot
REM  garble it. All Korean messages come from setup_stage2.py,
REM  which runs under UTF-8.
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

set "ROOT=%~dp0.."
cd /d "%ROOT%"
set "PYDIR=%ROOT%\python"
set "PYEXE=%PYDIR%\python.exe"
set "LOG=%ROOT%\setup_log.txt"

echo ============================================ > "%LOG%"
echo  setup started %DATE% %TIME%                >> "%LOG%"
echo  root: %ROOT%                               >> "%LOG%"
echo ============================================ >> "%LOG%"

echo.
echo  [ Educational Shorts Maker - setup ]
echo.
echo  Installing into: %ROOT%
echo  Log file       : %LOG%
echo.
echo  This takes 10-30 minutes depending on your network.
echo  Do not close this window.
echo.

REM ---------- step 1: portable Python ----------
if exist "%PYEXE%" (
    echo  [1/5] Python already present, skipping download.
    echo [1] python exists >> "%LOG%"
    goto :haspython
)

echo  [1/5] Downloading portable Python...
echo [1] downloading python >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%ROOT%\python.zip'" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_download

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Expand-Archive -Path '%ROOT%\python.zip' -DestinationPath '%PYDIR%' -Force" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_extract
del "%ROOT%\python.zip" >nul 2>&1

REM Embeddable Python ignores site-packages until this file is fixed.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $p=Get-ChildItem '%PYDIR%\python*._pth' | Select-Object -First 1; $t=Get-Content $p.FullName; $t=$t -replace '^#\s*import site','import site'; if($t -notcontains 'import site'){$t+='import site'}; if($t -notcontains '..'){$t+='..'}; Set-Content -Path $p.FullName -Value $t -Encoding ASCII" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_pth

:haspython

REM ---------- step 2: pip ----------
"%PYEXE%" -m pip --version >nul 2>&1
if not errorlevel 1 (
    echo  [2/5] pip already present, skipping.
    echo [2] pip exists >> "%LOG%"
    goto :haspip
)
echo  [2/5] Installing pip...
echo [2] installing pip >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%ROOT%\get-pip.py'" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_download
"%PYEXE%" "%ROOT%\get-pip.py" --no-warn-script-location >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_pip
del "%ROOT%\get-pip.py" >nul 2>&1

:haspip

REM ---------- step 3: hand over to Python ----------
echo  [3/5] Installing libraries (this is the long part)...
echo [3] handing over to setup_stage2.py >> "%LOG%"
"%PYEXE%" "%ROOT%\usb\setup_stage2.py" --root "%ROOT%"
if errorlevel 1 goto :fail_stage2

echo.
echo  ============================================
echo   Setup finished.
echo   Next: double-click  watch_start.bat
echo  ============================================
echo.
pause
exit /b 0

:fail_download
echo. & echo  FAILED: could not download from the internet.
echo  Check your network / proxy, then run this file again.
echo  Details are in: %LOG%
echo [X] download failed >> "%LOG%"
pause & exit /b 1

:fail_extract
echo. & echo  FAILED: could not unzip Python.
echo  Your USB drive may be write-protected or full.
echo  Details are in: %LOG%
echo [X] extract failed >> "%LOG%"
pause & exit /b 1

:fail_pth
echo. & echo  FAILED: could not configure Python.
echo  Details are in: %LOG%
echo [X] pth patch failed >> "%LOG%"
pause & exit /b 1

:fail_pip
echo. & echo  FAILED: could not install pip.
echo  Details are in: %LOG%
echo [X] pip install failed >> "%LOG%"
pause & exit /b 1

:fail_stage2
echo. & echo  FAILED during library installation.
echo  Please send the log file to get help: %LOG%
echo [X] stage2 failed >> "%LOG%"
pause & exit /b 1
