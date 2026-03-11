@echo off
REM Runs start_all.bat (servers only, no Edge) then start_kiosk.ps1 (Edge in kiosk).
cd /d "%~dp0"

call "%~dp0start_all.bat" /startup

REM Launch Edge via PowerShell - the only method that works reliably on this system
echo.
echo Launching Edge in kiosk mode via start_kiosk.ps1...
powershell -ExecutionPolicy Bypass -File "%~dp0start_kiosk.ps1"

echo.
echo Done. Servers are running in separate windows; Edge should be in kiosk mode.
if "%~1"=="/startup" exit /b 0
pause
