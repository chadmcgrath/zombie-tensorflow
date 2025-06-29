import React, { useEffect, useRef } from 'react';
import './game.css';

const ZombieGame = () => {
  const canvasRef = useRef(null);
  const lossesChartRef = useRef(null);
  const rewardChartRef = useRef(null);

  useEffect(() => {
    // Load external dependencies
    const loadExternalScripts = async () => {
      // Load jQuery
      if (!window.$) {
        const jqueryScript = document.createElement('script');
        jqueryScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/jquery/2.2.2/jquery.min.js';
        document.head.appendChild(jqueryScript);
        await new Promise(resolve => jqueryScript.onload = resolve);
      }

      // Load Bootstrap
      if (!window.bootstrap) {
        const bootstrapScript = document.createElement('script');
        bootstrapScript.src = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js';
        document.head.appendChild(bootstrapScript);
        await new Promise(resolve => bootstrapScript.onload = resolve);
      }

      // Load Lodash
      if (!window._) {
        const lodashScript = document.createElement('script');
        lodashScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.6.1/lodash.min.js';
        document.head.appendChild(lodashScript);
        await new Promise(resolve => lodashScript.onload = resolve);
      }

      // Load Chart.js
      if (!window.Chart) {
        const chartScript = document.createElement('script');
        chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        document.head.appendChild(chartScript);
        await new Promise(resolve => chartScript.onload = resolve);
      }

      // Load Bootstrap CSS
      if (!document.querySelector('link[href*="bootstrap"]')) {
        const bootstrapCSS = document.createElement('link');
        bootstrapCSS.rel = 'stylesheet';
        bootstrapCSS.href = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css';
        document.head.appendChild(bootstrapCSS);
      }

      // Load Normalize CSS
      if (!document.querySelector('link[href*="normalize"]')) {
        const normalizeCSS = document.createElement('link');
        normalizeCSS.rel = 'stylesheet';
        normalizeCSS.href = 'https://cdnjs.cloudflare.com/ajax/libs/normalize/5.0.0/normalize.min.css';
        document.head.appendChild(normalizeCSS);
      }
    };

    loadExternalScripts().then(() => {
      // Initialize the game after all dependencies are loaded
      initializeGame();
    });

    return () => {
      // Cleanup if needed
    };
  }, []);

  const initializeGame = async () => {
    try {
      // Import game logic
      const { 
        initializeGame: startGame,
        toggleContinueLoop,
        toggleVampire,
        setGameSpeed,
        setShowEyes,
        setSkipFrames,
        toggleSprites,
        addZombie,
        addHuman,
        addVampire,
        updateRewardConfig,
        saveModels,
        loadModels
      } = await import('./gameLogic.js');
      
      // Get canvas element
      const canvas = canvasRef.current;
      if (!canvas) {
        console.error('Canvas not found');
        return;
      }
      
      // Initialize the game with the canvas
      console.log('Starting zombie game initialization...');
      await startGame(canvas);
      
      // Set up event handlers
      setupEventHandlers({
        toggleContinueLoop,
        toggleVampire,
        setGameSpeed,
        setShowEyes,
        setSkipFrames,
        toggleSprites,
        addZombie,
        addHuman,
        addVampire,
        updateRewardConfig,
        saveModels,
        loadModels
      });
      
      console.log('Game initialized successfully!');
    } catch (error) {
      console.error('Error initializing game:', error);
    }
  };

  const setupEventHandlers = (gameControls) => {
    // Game speed control
    const gameSpeedElement = document.getElementById('gameSpeed');
    if (gameSpeedElement) {
      gameSpeedElement.addEventListener('change', (e) => {
        gameControls.setGameSpeed(parseInt(e.target.value));
      });
    }

    // Show eyes control
    const showEyesElement = document.getElementById('show-eyes');
    if (showEyesElement) {
      showEyesElement.addEventListener('input', (e) => {
        gameControls.setShowEyes(parseInt(e.target.value));
      });
    }

    // Skip frames control
    const skipFramesElement = document.getElementById('skip-frames');
    if (skipFramesElement) {
      skipFramesElement.addEventListener('input', (e) => {
        gameControls.setSkipFrames(parseInt(e.target.value));
      });
    }

    // Button event handlers
    const buttons = {
      'toggle-sprites': gameControls.toggleSprites,
      'add-vampire-button': gameControls.addVampire,
      'vampire-button': gameControls.toggleVampire,
      'rush-watch-button': gameControls.toggleContinueLoop,
      'save-button': gameControls.saveModels,
      'load-button': gameControls.loadModels,
      'add-human-button': gameControls.addHuman,
      'add-zombie-button': gameControls.addZombie
    };

    Object.entries(buttons).forEach(([id, handler]) => {
      const element = document.getElementById(id);
      if (element) {
        element.addEventListener('click', handler);
      }
    });

    // Reward form handler
    const rewardForm = document.getElementById('rewardForm');
    if (rewardForm) {
      rewardForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const rewardConfig = {};
        for (let [key, value] of formData.entries()) {
          rewardConfig[key] = parseFloat(value);
        }
        gameControls.updateRewardConfig(rewardConfig);
        console.log('Updated reward config:', rewardConfig);
      });
    }

    // Modal handlers
    const rewardModal = document.getElementById('rewardModal');
    const openModalBtn = document.getElementById('openRewardModal');
    const closeModalBtns = document.querySelectorAll('.close-modal');

    if (openModalBtn && rewardModal) {
      openModalBtn.addEventListener('click', () => {
        rewardModal.style.display = 'block';
      });
    }

    closeModalBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        if (rewardModal) {
          rewardModal.style.display = 'none';
        }
      });
    });

    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
      if (e.target === rewardModal) {
        rewardModal.style.display = 'none';
      }
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      switch (e.code) {
        case 'Space':
          e.preventDefault();
          gameControls.toggleContinueLoop();
          break;
        case 'KeyV':
          gameControls.toggleVampire();
          break;
        case 'KeyL':
          gameControls.loadModels();
          break;
        case 'KeyS':
          gameControls.saveModels();
          break;
      }
    });
  };

  const showValue = (val, id) => {
    if (window.$) {
      window.$(id).text(val);
    }
  };

  const updateValue = (val, id) => {
    showValue(val, id);
    var minp = 0;
    var maxp = 1;

    var minv = Math.log(.00001);
    var maxv = Math.log(.1);

    var scale = (maxv - minv) / (maxp - minp);

    var expValue = Math.exp(minv + scale * (val - minp));
    console.log(expValue);
  };

  return (
    <div>
      <div>
        <input 
          id="gameSpeed" 
          type="range" 
          min="1" 
          max="20" 
          step="1" 
          defaultValue="4" 
          title="Game Speed"
          placeholder="Game Speed" 
        />
      </div>
      
      <div className="flex-container" style={{ display: 'flex', justifyContent: 'space-between' }}>
        <div id="stats-container" style={{ minWidth: '15em' }}>
          <div>
            <span>RewardTotal:</span>
            <span id='rewardTotal'></span>
          </div>
          <div>
            <span>missed shots:</span>
            <span id='missed-shots'>0</span>
          </div>
          <div>
            <span>hit shots:</span>
            <span id='hit-shots-baddy'>0</span>
          </div>
          <div>
            <span>hit shots human!:</span>
            <span id='hit-shots-human'>0</span>
          </div>
          <div>
            <span>Turns:</span>
            <span id='turns'></span>
          </div>
          <label htmlFor="skip-frames">Skip Frames</label>
          <input type="range" id="skip-frames" min="0" max="20" step="1" defaultValue="0" />
        </div>
        
        <div id="mid-container" className="flex-container">
          <div>
            <h1>AI Zombie Apocalypse Super Quest</h1>
            <h4>(you just kinda watch it for a while and see if the humans learn anything)</h4>
            <p>
              some zombie code from: Paul Robello (https://codepen.io/paulrobello/pen/LNvEvx) <br/>
              ppo reinforcement learning library from: https://github.com/zemlyansky/ppo-tfjs/<br/>
              little bit of help from : https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html<br/>
            </p>
          </div>
        </div>
        
        <div id="button-container" className="button-container">
          <div>
            <button id="openRewardModal" className="btn btn-primary">Reward Configuration</button>
            <button id="toggle-sprites" className="btn btn-primary">Toggle Sprites</button>
            <br/>
            <label htmlFor="show-eyes">Show Eyes</label>
            <input type="range" id="show-eyes" min="0" max="2" step="1" defaultValue="0" />
          </div>
          <div>
            <button id="add-vampire-button" className="btn btn-primary">Add Vampire</button>
            <button id="vampire-button" className="btn btn-primary">Vampire</button>
            <button id="rush-watch-button" className="btn btn-primary">Rush/Watch</button>
            <button id="smith-button" className="btn btn-primary">Move Blue</button>
          </div>
          <div>
            <button id="save-button" className="btn btn-primary">Save</button>
            <button id="load-button" className="btn btn-primary">Load</button>
            <button id="add-human-button" className="btn btn-primary">Add Human</button>
            <button id="add-zombie-button" className="btn btn-primary">Add Zombie</button>
          </div>
        </div>
      </div>
      
      <div id="canvas-container" className="flex-container">
        <div id="div1">
          <canvas 
            ref={canvasRef}
            id="canvas" 
            className="canvas-city" 
            width="1200" 
            height="500"
          ></canvas>
        </div>
      </div>
      
      <div>
        <canvas ref={lossesChartRef} id="losses-chart"></canvas>
      </div>
      
      <div>
        <canvas ref={rewardChartRef} id="rewardOverTimeChart"></canvas>
      </div>

      <div>
        neg rewards:
        <div id="neg-rewards"></div>
        <div id="bigNums">
          Samples:
          <div id="samples"></div>
          W
          <div id="weights"></div>
          CW
          <div id="criticWeights"></div>
          <div id="current-state">weights</div>
        </div>
      </div>

      <div id="rewardModal" className="modal">
        <div className="modal-content">
          <span className="close close-modal">&times;</span>
          <form id="rewardForm">
            <label htmlFor="hitShotReward">Hit Shot Reward</label>
            <input type="number" step="0.01" id="hitShotReward" name="hitShotReward" />
            <label htmlFor="biteReward">Bite Reward</label>
            <input type="number" step="0.01" id="biteReward" name="biteReward" />
            <label htmlFor="hitHumanReward">Hit Human Reward</label>
            <input type="number" step="0.01" id="hitHumanReward" name="hitHumanReward" />
            <label htmlFor="missedShotReward">Missed Shot Reward</label>
            <input type="number" step="0.01" id="missedShotReward" name="missedShotReward" />
            <label htmlFor="bumpWallReward">Bump Wall Reward</label>
            <input type="number" step="0.01" id="bumpWallReward" name="bumpWallReward" />
            <label htmlFor="bumpScreenReward">Bump Screen Reward</label>
            <input type="number" step="0.01" id="bumpScreenReward" name="bumpScreenReward" />
            <label htmlFor="bumpHumanReward">Bump Human Reward</label>
            <input type="number" step="0.01" id="bumpHumanReward" name="bumpHumanReward" />
            <label htmlFor="blockedVisionHuman">Blocked Vision Human</label>
            <input type="number" step="0.01" id="blockedVisionHuman" name="blockedVisionHuman" />
            <label htmlFor="blockedVisionWall">Blocked Vision Wall</label>
            <input type="number" step="0.01" id="blockedVisionWall" name="blockedVisionWall" />
            <label htmlFor="zombieProximityReward">Zombie Proximity Reward</label>
            <input type="number" step="0.01" id="zombieProximityReward" name="zombieProximityReward" />
            <button type="button" id="closeRewardModal" className="close-modal">Close</button>
            <button type="submit">Apply</button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ZombieGame;
