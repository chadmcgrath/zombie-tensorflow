# TF-Zombies: AI Zombie Apocalypse Simulator

An interactive AI simulation where humans learn to survive against zombies using reinforcement learning (PPO) with TensorFlow.js. Watch as AI agents develop survival strategies in real-time!

## Features

- **Reinforcement Learning**: Humans use PPO (Proximal Policy Optimization) to learn survival strategies
- **Real-time Simulation**: Watch the learning process unfold in an interactive environment
- **Configurable Rewards**: Adjust reward parameters to influence learning behavior
- **Visual Sprites**: Toggle between sprite graphics and simple shapes
- **Performance Controls**: Adjust game speed and frame skipping for optimal performance
- **Save/Load Models**: Persist trained models for continued learning

## Prerequisites

Before installing, make sure you have:

- **Node.js** (version 16.0.0 or higher)
- **npm** (version 8.0.0 or higher)

You can check your versions with:
```bash
node --version
npm --version
```

If you need to install Node.js, download it from [nodejs.org](https://nodejs.org/).

## Installation

### Option 1: Automated Installation (Recommended)

**Windows:**
1. Double-click `install.bat` to automatically install dependencies
2. Double-click `run.bat` to start the application

**Mac/Linux:**
1. Run `./install.sh` to automatically install dependencies
2. Run `./run.sh` to start the application

### Option 2: Manual Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd zombie-tensorflow
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

   This will install:
   - Electron (for the desktop application)
   - ESLint (for code linting)

## Running the Application

### Quick Start
```bash
npm start
```

### Alternative Methods
```bash
# Development mode (same as start)
npm run dev

# Run with custom memory allocation
electron --max-old-space-size=4096 main.js

# Windows: Double-click run.bat
# Mac/Linux: ./run.sh
```

The application will open in an Electron window displaying the zombie simulation.

## How to Use

### Basic Controls

1. **Game Speed**: Use the slider at the top to control simulation speed (1-20x)
2. **Skip Frames**: Adjust frame skipping to improve performance on slower machines
3. **Add Entities**: 
   - Click "Add Human" to add more survivors
   - Click "Add Zombie" to increase the challenge
4. **Save/Load**: Preserve trained models for later use

### Advanced Features

- **Reward Configuration**: Click "Reward Configuration" to fine-tune learning parameters
- **Toggle Sprites**: Switch between animated sprites and simple geometric shapes
- **Show Eyes**: Visualize the agents' vision systems
- **Rush/Watch**: Toggle between fast training and observation modes

### Understanding the Simulation

- **Blue entities**: Humans learning to survive
- **Red entities**: Zombies pursuing humans
- **Stats panel**: Shows performance metrics including rewards, shots fired, and learning progress
- **Charts**: Real-time visualization of learning progress and rewards over time

## Project Structure

```
zombie-tensorflow/
├── main.js              # Electron main process
├── index.html           # Main application interface
├── scriptMain.js        # Core simulation logic
├── ppo.js              # PPO reinforcement learning implementation
├── utilities.js        # Helper functions
├── config.js           # Configuration settings
├── style.css           # Application styling
├── img/                # Sprite assets
│   ├── skeleton/       # Zombie sprites
│   └── survivor/       # Human sprites
└── package.json        # Project dependencies and scripts
```

## Troubleshooting

### Common Issues

1. **Application won't start**
   - Ensure Node.js and npm are properly installed
   - Try running `npm install` again
   - Check that all dependencies are installed

2. **Performance issues**
   - Increase the "Skip Frames" setting
   - Reduce the number of entities in the simulation
   - Lower the game speed

3. **Memory issues**
   - The application uses `--max-old-space-size=4096` to allocate more memory
   - Close other applications to free up system resources

### Additional Commands

```bash
# Reinstall all dependencies
npm run clean

# Run code linting
npm run lint

# Install dependencies explicitly
npm run install-deps
```

## Technologies Used

- **Electron**: Desktop application framework
- **TensorFlow.js**: Machine learning library
- **PPO**: Proximal Policy Optimization reinforcement learning
- **HTML5 Canvas**: Graphics rendering
- **Chart.js**: Data visualization
- **Bootstrap**: UI framework

## Credits

- Zombie code adapted from: Paul Robello (https://codepen.io/paulrobello/pen/LNvEvx)
- PPO reinforcement learning library from: https://github.com/zemlyansky/ppo-tfjs/
- Additional inspiration from: https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the simulation!
