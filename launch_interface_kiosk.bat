@echo off
echo Launching Interface in Chrome Kiosk Mode...
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
    echo ERROR: Chrome not found. Please install Google Chrome.
    pause
    exit /b 1
)

REM Launch Chrome in kiosk mode pointing to the interface
echo Starting Chrome in kiosk mode at http://localhost:5174
echo Press Alt+F4 to exit kiosk mode
echo.
start "" "%CHROME_PATH%" --kiosk --app=http://localhost:5174

echo Chrome launched successfully!
