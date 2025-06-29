// Test setup file for Vitest
// This file runs before each test file

import { vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import '@testing-library/jest-dom';

// Make TensorFlow.js available globally
global.tf = tf;

// Reduce console noise in tests
const originalWarn = console.warn;
const originalLog = console.log;

console.warn = vi.fn();
console.log = vi.fn();

// Restore console methods after tests if needed
global.restoreConsole = () => {
  console.warn = originalWarn;
  console.log = originalLog;
};
