@echo off
echo Installing TF-Zombies AI Simulator...
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo Node.js and npm are installed. Proceeding with installation...
echo.

REM Install dependencies
echo Installing dependencies...
npm install

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo.
echo To run the application, use one of these commands:
echo   npm start
echo   npm run dev
echo.
echo Or simply double-click run.bat
echo.
pause
