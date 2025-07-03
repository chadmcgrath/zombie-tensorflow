# Zombie TensorFlow - PPO Reinforcement Learning

A JavaScript implementation of Proximal Policy Optimization (PPO) for reinforcement learning, built with TensorFlow.js and featuring a zombie survival game environment.

## Requirements

- **Node.js**: >= 16.0.0
- **npm**: >= 8.0.0
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for training)
- **GPU**: Optional - TensorFlow.js will use WebGL acceleration when available

## Installation

### Local Setup

1. **Clone the repository**
   ```sh
   git clone <repository-url>
   cd zombie-tensorflow
   ```

2. **Use the correct Node.js version** (if using nvm)
   ```sh
   nvm use
   ```

3. **Install dependencies**
   ```sh
   npm install
   ```

4. **Run the development server**
   ```sh
   npm run dev
   ```

5. **Run tests**
   ```sh
   npm test
   ```

### Docker Setup

1. **Build the Docker image**
   ```sh
   ./build_docker.sh zombie-tensorflow
   ```

2. **Run tests in Docker**
   ```sh
   docker run -t zombie-tensorflow ./run_tests.sh
   ```

3. **Run development server in Docker**
   ```sh
   docker run --network=host -v .:/app -it zombie-tensorflow npm exec vite dev --host
   ```

## Project Structure

- `src/ppo.js` - Core PPO algorithm implementation
- `src/ZombieGame.jsx` - React-based zombie survival game
- `src/gameLogic.js` - Game mechanics and environment
- `src/utilities.js` - Utility functions
- `src/config.js` - Configuration settings
- `test/ppo.test.js` - Comprehensive PPO test suite (all tests consolidated)
- `test/setup.js` - Test setup and configuration

## Running Tests

### Local Testing
```sh
npm test
```

### Docker Testing
Build and run tests in a containerized environment:

```sh
./build_docker.sh zombie-tensorflow
docker run -t zombie-tensorflow ./run_tests.sh
```

### Running Specific Tests
Run tests matching a specific pattern:

```sh
npm test -- ppo
# or with docker:
docker run -t zombie-tensorflow ./run_tests.sh ppo
```

## Development

### Local Development Server
```sh
npm run dev
```

### Docker Development with Hot Reloading
```sh
./build_docker.sh zombie-tensorflow
docker run --network=host -v .:/app -it zombie-tensorflow npm exec vite dev --host
```

## Test Status

Current test results:
- ✅ **10 tests passing** - Basic functionality works
- ❌ **20+ tests failing** - Comprehensive edge case coverage reveals missing robustness features

**Test Categories:**
- ✅ Basic PPO functionality (3 tests)
- ✅ Parameter validation (2 tests) 
- ✅ Input validation (2 tests)
- ✅ Callback handling (2 tests)
- ✅ Buffer management (1 test)
- ❌ **Advanced buffer management** (3 tests) - Empty buffer, zero variance, overflow protection
- ❌ **Environment interface validation** (3 tests) - Action/observation space validation, step format
- ❌ **Input data validation** (3 tests) - Shape consistency, action bounds, null handling
- ❌ **Training stability** (3 tests) - Gradient explosion, policy collapse, value divergence
- ❌ **Advanced edge cases** (4 tests) - Extreme parameters, architecture validation, state corruption

**Key Missing Features Exposed:**
- Parameter bounds validation (learning rates, clip ratios, gamma/lambda)
- Input sanitization (NaN/Infinity handling, shape validation)
- Buffer safety (overflow protection, empty buffer handling)
- Training stability (gradient clipping, entropy monitoring)
- Environment interface validation (action/observation space checks)

All failing tests represent pure PPO algorithm robustness issues that can be fixed with PPO knowledge alone - no TensorFlow expertise required.

## Features

- **PPO Algorithm**: Complete implementation with actor-critic networks
- **TensorFlow.js Integration**: GPU-accelerated training when available
- **Game Environment**: Interactive zombie survival game for testing
- **Comprehensive Testing**: Edge cases, memory management, and robustness tests
- **Docker Support**: Containerized development and testing environment

## Test Organization

All tests are consolidated into a single file (`test/ppo.test.js`) with clear categories:

**✅ Passing Tests (8):**
- Basic PPO functionality (3 tests)
- NaN/Infinity input validation (2 tests) 
- Callback error handling (2 tests)
- Buffer data consistency (1 test)

**❌ Pure PPO Algorithm Failures (8):**
- Parameter validation for learning rates, clip ratios, KL targets, steps, epochs, gamma, lambda

## Architecture

The PPO implementation supports both discrete and continuous action spaces, includes proper advantage estimation, and handles various edge cases for robust reinforcement learning training.
