@echo off
echo Starting TF-Zombies AI Simulator...
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo Dependencies not found. Running installation first...
    call install.bat
    if %errorlevel% neq 0 (
        echo Installation failed. Cannot start application.
        pause
        exit /b 1
    )
)

REM Start the application
echo Launching application...
npm start

if %errorlevel% neq 0 (
    echo ERROR: Failed to start the application
    echo Try running: npm install
    echo.
    pause
    exit /b 1
)
