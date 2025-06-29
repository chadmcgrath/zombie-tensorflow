// Test script to check if imports are working
import('./src/config.js').then(config => {
    console.log('Config loaded:', Object.keys(config));
    return import('./src/utilities.js');
}).then(utilities => {
    console.log('Utilities loaded:', Object.keys(utilities));
    return import('./src/ppo.js');
}).then(ppo => {
    console.log('PPO loaded:', Object.keys(ppo));
    return import('./src/gameLogic.js');
}).then(gameLogic => {
    console.log('Game logic loaded:', Object.keys(gameLogic));
    console.log('All imports successful!');
}).catch(error => {
    console.error('Import error:', error);
});
