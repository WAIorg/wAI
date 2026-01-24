@echo off
echo Starting Backend Server...
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

REM Navigate to backend directory
cd backend

REM Activate virtual environment
if exist "C:\Users\wai\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call C:\Users\wai\venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at C:\Users\wai\venv
    echo Using system Python. Make sure dependencies are installed: pip install -r requirements.txt
)

REM Start the FastAPI server
echo.
echo Starting FastAPI server on http://0.0.0.0:8000
echo Press Ctrl+C to stop the server
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
