#!/bin/bash

echo "Running TF-Zombies Test Suite..."
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    echo
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Dependencies not found. Installing..."
    npm install
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        echo
        exit 1
    fi
fi

# Run tests
echo "Running tests..."
npm test

if [ $? -ne 0 ]; then
    echo
    echo "Some tests failed. Check the output above for details."
    echo
    exit 1
else
    echo
    echo "All tests passed successfully!"
    echo
    echo "To run tests with coverage: npm run test:coverage"
    echo "To run tests in watch mode: npm run test:watch"
    echo
fi
