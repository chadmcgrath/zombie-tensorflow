// Test setup file for Jest
// This file runs before each test file

// Mock TensorFlow.js for testing
global.tf = {
  ready: () => Promise.resolve(),
  memory: () => ({
    numTensors: 0,
    numBytes: 0
  }),
  tensor: (data) => ({
    shape: Array.isArray(data[0]) ? [data.length, data[0].length] : [data.length],
    dispose: () => {},
    dataSync: () => new Float32Array(10),
    data: () => Promise.resolve(new Float32Array(10))
  }),
  tidy: (fn) => fn(),
  dispose: () => {},
  train: {
    adam: (lr) => ({
      computeGradients: (fn) => ({ values: [], grads: [] }),
      applyGradients: () => {},
      dispose: () => {}
    })
  },
  layers: {
    input: (config) => ({ apply: (x) => x }),
    dense: (config) => ({ apply: (x) => x })
  },
  model: (config) => ({
    predict: () => global.tf.tensor([[0.5, 0.5]]),
    getWeights: () => [global.tf.tensor([1, 2, 3])],
    dispose: () => {}
  }),
  squeeze: (x) => x,
  multinomial: (x, n) => global.tf.tensor([0]),
  add: (a, b) => a,
  mul: (a, b) => a,
  randomStandardNormal: (shape) => global.tf.tensor(new Array(shape[0]).fill(0)),
  exp: (x) => x,
  logSoftmax: (x) => x,
  sum: (x) => x,
  oneHot: (x, n) => x,
  square: (x) => x,
  sub: (a, b) => a,
  div: (a, b) => a,
  scalar: (x) => x,
  log: (x) => x,
  abs: (x) => x,
  mean: (x) => ({ arraySync: () => 0 }),
  moments: (x) => ({ variance: { sqrt: () => ({ arraySync: () => 1 }) } }),
  where: (cond, a, b) => a,
  greater: (a, b) => a,
  minimum: (a, b) => a,
  neg: (x) => x,
  losses: {
    meanSquaredError: (a, b) => a
  },
  variable: (x, trainable, name) => x
};

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  warn: jest.fn(),
  log: jest.fn()
};
