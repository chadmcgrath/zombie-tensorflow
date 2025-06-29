// Test setup file for Jest
// This file runs before each test file

// Use actual TensorFlow.js library
global.tf = require('@tensorflow/tfjs');

// Reduce console noise in tests
const originalWarn = console.warn;
const originalLog = console.log;

console.warn = jest.fn();
console.log = jest.fn();

// Restore console methods after tests if needed
global.restoreConsole = () => {
  console.warn = originalWarn;
  console.log = originalLog;
};
