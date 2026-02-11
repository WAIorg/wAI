@echo off
setlocal enabledelayedexpansion
echo Starting Interface in Edge Kiosk Mode...
echo.

REM Activate virtual environment
if exist "C:\Users\wai\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call C:\Users\wai\venv\Scripts\activate.bat
)

REM Get the directory where this script is located
cd /d "%~dp0"

REM Navigate to interface directory
cd interface

REM Check if node_modules exists (dependencies installed)
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

REM Start the Vite dev server in a new window
echo Starting Vite dev server...
start "Vite Dev Server" cmd /k "npm run dev"

REM Wait for the server to start
echo Waiting for server to start (5 seconds)...
timeout /t 5 /nobreak >nul
echo Server should be ready now.

REM Launch Edge in kiosk mode
echo Launching Edge in kiosk mode...
echo Press Alt+F4 or Ctrl+Alt+Delete to exit kiosk mode
echo.

REM Find Edge executable (common locations)
set EDGE_PATH=
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    set EDGE_PATH=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    set EDGE_PATH=C:\Program Files\Microsoft\Edge\Application\msedge.exe
)

if "%EDGE_PATH%"=="" (
    echo ERROR: Edge not found. Please install Microsoft Edge or update the path in this script.
    pause
    exit /b 1
)

REM Launch Edge in kiosk mode pointing to the interface
echo Launching Edge in kiosk mode...
echo Edge path: %EDGE_PATH%
echo URL: http://localhost:5174
echo.

REM Use the exact command format that works in command prompt
"%EDGE_PATH%" --kiosk http://localhost:5174

echo.
echo Edge launch command executed.
echo The interface should be visible at http://localhost:5174
echo.
echo If Edge didn't open, check:
echo   1. Is Edge installed at: %EDGE_PATH%
echo   2. Is the dev server running on port 5174?
echo   3. Try running: launch_interface_kiosk.bat manually
echo.
echo To exit kiosk mode, press Alt+F4 or close this window and stop the dev server.

pause
