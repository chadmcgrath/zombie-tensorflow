module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  testMatch: [
    '<rootDir>/tests/**/*.test.js',
    '<rootDir>/tests/**/*.spec.js'
  ],
  collectCoverageFrom: [
    'utilities.js',
    'config.js',
    'ppo.js',
    '!node_modules/**',
    '!main.js', // Electron main process
    '!scriptMain.js' // Contains DOM dependencies
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  moduleFileExtensions: ['js', 'json'],
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  globals: {
    'tf': {},
    'Chart': {},
    'window': {},
    'document': {}
  }
};
