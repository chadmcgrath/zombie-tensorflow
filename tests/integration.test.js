/* eslint-env jest */

describe('Integration Tests', () => {
  test('should load utilities without errors', () => {
    expect(() => {
      require('../utilities.js');
    }).not.toThrow();
  });

  test('should load config without errors', () => {
    expect(() => {
      require('../config.js');
    }).not.toThrow();
  });

  test('utilities and config should work together', () => {
    require('../utilities.js');
    require('../config.js');
    
    // Test that we can create a square and use config
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    expect(square).toBeDefined();
    expect(rewardConfigs).toBeDefined();
    expect(rewardConfigs.default).toBeDefined();
  });

  test('vector math should work with game configurations', () => {
    require('../utilities.js');
    require('../config.js');
    
    const v1 = Vec(0, 0);
    const v2 = Vec(3, 4);
    const distance = v1.distFrom(v2);
    
    // Test that distance calculation works with reward ranges
    expect(distance).toBe(5);
    expect(distance).toBeGreaterThan(Math.abs(rewardConfigs.default.bumpWallReward));
  });

  test('eye vision system should integrate with collision detection', () => {
    require('../utilities.js');
    
    const eye = new Eye(0);
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    expect(eye.max_range).toBeGreaterThan(0);
    expect(square.bounds).toHaveLength(2);
    
    // Test ray intersection
    const origin = { x: -10, y: 50 };
    const direction = { x: 1, y: 0 };
    const intersection = square.rayIntersect(origin, direction);
    
    expect(intersection).toBeTruthy();
  });

  test('game state clock should work with reward timing', () => {
    require('../utilities.js');
    require('../config.js');
    
    const clock = GameState.createClock();
    const time1 = 1000;
    const time2 = 1100;
    
    GameState.updateClock(clock, time1);
    GameState.updateClock(clock, time2);
    
    // Delta should be reasonable for reward calculations
    expect(clock.delta).toBeGreaterThan(0);
    expect(clock.delta).toBeLessThanOrEqual(0.1);
    
    // Should work with reward multipliers
    const rewardMultiplier = clock.delta * rewardConfigs.default.baseReward;
    expect(rewardMultiplier).toBeGreaterThan(0);
  });
});
