@echo off
echo Running TF-Zombies Test Suite...
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

REM Check if dependencies are installed
if not exist "node_modules" (
    echo Dependencies not found. Installing...
    npm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        echo.
        pause
        exit /b 1
    )
)

REM Run tests
echo Running tests...
npm test

if %errorlevel% neq 0 (
    echo.
    echo Some tests failed. Check the output above for details.
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo All tests passed successfully!
    echo.
    echo To run tests with coverage: npm run test:coverage
    echo To run tests in watch mode: npm run test:watch
    echo.
    pause
)
