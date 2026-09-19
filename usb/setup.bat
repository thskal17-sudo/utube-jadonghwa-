@echo off
REM ============================================================
REM  Educational Shorts Maker - USB portable setup (Windows)
REM
REM  ASCII only on purpose: cmd.exe garbles Korean text.
REM  All Korean messages come from setup_stage2.py under UTF-8.
REM
REM  Downloads go through %TEMP% first. A USB path may contain
REM  brackets or Korean characters that PowerShell treats as
REM  wildcards, which silently breaks -OutFile.
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

set "ROOT=%~dp0.."
pushd "%ROOT%" 2>nul || goto :fail_badpath
set "ROOT=%CD%"
popd
cd /d "%ROOT%" 2>nul || goto :fail_badpath

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

REM ---------- step 0: path sanity ----------
echo %ROOT%| findstr /C:"[" >nul && goto :fail_badpath
echo %ROOT%| findstr /C:"]" >nul && goto :fail_badpath
echo %ROOT%| findstr /C:"*" >nul && goto :fail_badpath
echo %ROOT%| findstr /C:"?" >nul && goto :fail_badpath
echo %ROOT%| findstr /C:"!" >nul && goto :fail_badpath

set "P=%ROOT%"
set /a PLEN=0
:countloop
if defined P (
    set "P=!P:~1!"
    set /a PLEN+=1
    goto :countloop
)
echo  path length: !PLEN! >> "%LOG%"
if !PLEN! GTR 90 goto :fail_longpath

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
set "TMPZIP=%TEMP%\edushorts_python.zip"
set "TMPDIR=%TEMP%\edushorts_python"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip','%TMPZIP%')" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_download
if not exist "%TMPZIP%" goto :fail_download

if exist "%TMPDIR%" rd /s /q "%TMPDIR%" >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::ExtractToDirectory('%TMPZIP%','%TMPDIR%')" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_extract

if not exist "%PYDIR%" mkdir "%PYDIR%" >nul 2>&1
xcopy "%TMPDIR%\*" "%PYDIR%\" /E /I /Y >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_extract
rd /s /q "%TMPDIR%" >nul 2>&1
del "%TMPZIP%" >nul 2>&1

:haspython

REM ---------- step 1b: search path ----------
REM This runs on EVERY launch, not only after a fresh download. A re-run must
REM be able to repair a broken ._pth - that is the whole point of re-running.
REM Embeddable Python ships a ._pth file that decides the entire search path.
REM Until it lists Lib\site-packages, pip installs succeed but nothing imports.
REM That exact failure happened in the field: "installed OK" then every single
REM ModuleNotFoundError. Patching the file line by line was too fragile, so the
REM whole file is rewritten with known-good contents. The first line must stay
REM the version-specific zip name, so it is read back out before rewriting.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $d=[IO.Path]::GetFullPath('%PYDIR%'); $p=[IO.Directory]::GetFiles($d,'python*._pth')[0]; $old=[IO.File]::ReadAllLines($p); $zip=@($old | Where-Object {$_ -like '*.zip'})[0]; if(-not $zip){$zip='python311.zip'}; [IO.File]::WriteAllLines($p,@($zip,'.','Lib\site-packages','..','import site'))" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_pth

REM Prove it worked before spending 20 minutes on downloads.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $d=[IO.Path]::GetFullPath('%PYDIR%'); $p=[IO.Directory]::GetFiles($d,'python*._pth')[0]; Write-Output ('--- ._pth now reads ---'); [IO.File]::ReadAllText($p)" >> "%LOG%" 2>&1


REM ---------- step 2: pip ----------
"%PYEXE%" -m pip --version >nul 2>&1
if not errorlevel 1 (
    echo  [2/5] pip already present, skipping.
    echo [2] pip exists >> "%LOG%"
    goto :haspip
)
echo  [2/5] Installing pip...
echo [2] installing pip >> "%LOG%"
set "TMPPIP=%TEMP%\edushorts_get-pip.py"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py','%TMPPIP%')" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_download
if not exist "%TMPPIP%" goto :fail_download
"%PYEXE%" "%TMPPIP%" --no-warn-script-location >> "%LOG%" 2>&1
if errorlevel 1 goto :fail_pip
del "%TMPPIP%" >nul 2>&1

:haspip

REM ---------- step 3-5: hand over to Python ----------
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


:fail_badpath
echo.
echo  FAILED: this folder path has characters Windows tools cannot handle.
echo.
echo   Brackets [ ] and some symbols break the installer.
echo.
echo   FIX: move this whole folder to a simple path, for example
echo.
echo        G:\shorts
echo.
echo   Then run usb\setup.bat again.
echo.
echo  Details are in: %LOG%
echo [X] bad path >> "%LOG%"
pause
exit /b 1

:fail_longpath
echo.
echo  FAILED: this folder path is too long (!PLEN! characters).
echo.
echo   Library folders nest deeply and Windows breaks past 260 characters.
echo.
echo   FIX: move this whole folder to a short path, for example
echo.
echo        G:\shorts
echo.
echo   Then run usb\setup.bat again.
echo.
echo  Details are in: %LOG%
echo [X] long path >> "%LOG%"
pause
exit /b 1

:fail_download
echo.
echo  FAILED: could not download from the internet.
echo.
echo   Your school network may block python.org or pypi.org.
echo   Try again on a different network (home or mobile hotspot).
echo.
echo  Details are in: %LOG%
echo [X] download failed >> "%LOG%"
pause
exit /b 1

:fail_extract
echo.
echo  FAILED: could not unzip Python.
echo.
echo   Your USB drive may be write-protected or full.
echo   Free up at least 2 GB and try again.
echo.
echo  Details are in: %LOG%
echo [X] extract failed >> "%LOG%"
pause
exit /b 1

:fail_pth
echo.
echo  FAILED: could not configure Python.
echo.
echo  Details are in: %LOG%
echo [X] pth patch failed >> "%LOG%"
pause
exit /b 1

:fail_pip
echo.
echo  FAILED: could not install pip.
echo.
echo   This usually means the download was blocked or incomplete.
echo.
echo  Details are in: %LOG%
echo [X] pip install failed >> "%LOG%"
pause
exit /b 1

:fail_stage2
echo.
echo  FAILED during library installation.
echo.
echo   Please send this log file to get help:
echo   %LOG%
echo.
echo [X] stage2 failed >> "%LOG%"
pause
exit /b 1
