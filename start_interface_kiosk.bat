@echo off
echo Starting Interface in Chrome Kiosk Mode...
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

REM Start the Vite dev server in the background
echo Starting Vite dev server...
start /B npm run dev

REM Wait a few seconds for the server to start
timeout /t 3 /nobreak >nul

REM Launch Chrome in kiosk mode
echo Launching Chrome in kiosk mode...
echo Press Alt+F4 or Ctrl+Alt+Delete to exit kiosk mode
echo.

REM Find Chrome executable (common locations)
set CHROME_PATH=
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    set CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    set CHROME_PATH=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
) else if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    set CHROME_PATH=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe
)

if "%CHROME_PATH%"=="" (
    echo ERROR: Chrome not found. Please install Google Chrome or update the path in this script.
    pause
    exit /b 1
)

REM Launch Chrome in kiosk mode pointing to the interface
start "" "%CHROME_PATH%" --kiosk --app=http://localhost:5174

echo Chrome launched in kiosk mode.
echo The interface should be visible at http://localhost:5174
echo.
echo To exit kiosk mode, press Alt+F4 or close this window and stop the dev server.

pause
