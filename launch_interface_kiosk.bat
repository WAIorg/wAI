@echo off
echo Launching Interface in Edge Kiosk Mode...
echo.

REM Find Edge executable (common locations)
set EDGE_PATH=
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    set EDGE_PATH=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    set EDGE_PATH=C:\Program Files\Microsoft\Edge\Application\msedge.exe
)

if "%EDGE_PATH%"=="" (
    echo ERROR: Edge not found. Please install Microsoft Edge.
    pause
    exit /b 1
)

REM Launch Edge in kiosk mode pointing to the interface
echo Starting Edge in kiosk mode at http://localhost:5174
echo Press Alt+F4 to exit kiosk mode
echo Edge path: %EDGE_PATH%
echo.

REM Use the exact command format that works in command prompt
"%EDGE_PATH%" --kiosk http://localhost:5174

echo Edge launch command executed!
