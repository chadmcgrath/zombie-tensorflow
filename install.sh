#!/bin/bash

echo "Installing TF-Zombies AI Simulator..."
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    echo
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    echo
    exit 1
fi

echo "Node.js and npm are installed. Proceeding with installation..."
echo

# Install dependencies
echo "Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo
    exit 1
fi

echo
echo "Installation completed successfully!"
echo
echo "To run the application, use one of these commands:"
echo "  npm start"
echo "  npm run dev"
echo
echo "Or run: ./run.sh"
echo

# Make run.sh executable
chmod +x run.sh
