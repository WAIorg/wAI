@echo off
REM Run this at Windows startup to start backend, interface, and Edge kiosk.
REM Use with Task Scheduler or a shortcut in the Startup folder (see STARTUP_README.txt).
cd /d "%~dp0"
call "%~dp0start_combined.bat" /startup
exit /b 0
