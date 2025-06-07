#!/bin/bash

echo "Starting TF-Zombies AI Simulator..."
echo

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Dependencies not found. Running installation first..."
    ./install.sh
    if [ $? -ne 0 ]; then
        echo "Installation failed. Cannot start application."
        exit 1
    fi
fi

# Start the application
echo "Launching application..."
npm start

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start the application"
    echo "Try running: npm install"
    echo
    exit 1
fi
