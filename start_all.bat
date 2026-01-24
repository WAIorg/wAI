@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Starting Backend and Interface Servers
echo ========================================
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

REM Activate virtual environment
if exist "C:\Users\wai\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call C:\Users\wai\venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at C:\Users\wai\venv
    echo Using system Python. Make sure dependencies are installed.
)

REM ========================================
REM Step 1: Start Backend Server
REM ========================================
echo.
echo [1/4] Starting Backend Server...
cd backend
start "Backend Server" cmd /k "call C:\Users\wai\venv\Scripts\activate.bat && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
cd ..

REM Wait for backend to start
echo Waiting for backend server to start (3 seconds)...
timeout /t 3 /nobreak >nul
echo Backend server should be running on http://0.0.0.0:8000
echo.

REM ========================================
REM Step 2: Start Interface Dev Server
REM ========================================
echo [2/4] Starting Interface Dev Server...
cd interface

REM Check if node_modules exists (dependencies installed)
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

REM Start the Vite dev server in a new window
start "Vite Dev Server" cmd /k "npm run dev"
cd ..

REM Wait for the interface server to start
echo Waiting for interface server to start (5 seconds)...
timeout /t 5 /nobreak >nul
echo Interface server should be running on http://localhost:5174
echo.

REM ========================================
REM Step 3: Find Edge Executable
REM ========================================
echo [3/4] Finding Edge executable...
set EDGE_PATH=
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    set EDGE_PATH=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    set EDGE_PATH=C:\Program Files\Microsoft\Edge\Application\msedge.exe
)

if "%EDGE_PATH%"=="" (
    echo ERROR: Edge not found. Please install Microsoft Edge.
    echo.
    echo Backend and Interface servers are running, but Edge cannot be launched.
    echo You can manually open Edge and navigate to http://localhost:5174
    pause
    exit /b 1
)

echo Edge found at: %EDGE_PATH%
echo.

REM ========================================
REM Step 4: Launch Edge in Kiosk Mode
REM ========================================
echo [4/4] Launching Edge in kiosk mode...
echo.
echo ========================================
echo All servers are starting!
echo ========================================
echo Backend:  http://localhost:8000
echo Interface: http://localhost:5174
echo.
echo Press Alt+F4 to exit kiosk mode
echo Press Ctrl+C in server windows to stop servers
echo ========================================
echo.

REM Launch Edge in kiosk mode
echo Launching Edge...
echo Command: "%EDGE_PATH%" --kiosk http://localhost:5174
echo.

REM Use cmd /c start to ensure proper execution
cmd /c start "" "%EDGE_PATH%" --kiosk http://localhost:5174

echo.
echo Edge launch command executed!
echo If Edge didn't open, try running this command manually:
echo "%EDGE_PATH%" --kiosk http://localhost:5174
echo.
echo Servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
pause
