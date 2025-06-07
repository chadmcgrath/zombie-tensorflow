/* eslint-env jest */
/* global rewardConfigs */

// Import the config file
require('../config.js');

describe('Reward Configurations', () => {
  test('should have rewardConfigs object', () => {
    expect(typeof rewardConfigs).toBe('object');
    expect(rewardConfigs).not.toBeNull();
  });

  test('should have default configuration', () => {
    expect(rewardConfigs).toHaveProperty('default');
    expect(typeof rewardConfigs.default).toBe('object');
  });

  test('should have explore configuration', () => {
    expect(rewardConfigs).toHaveProperty('explore');
    expect(typeof rewardConfigs.explore).toBe('object');
  });

  test('should have attack configuration', () => {
    expect(rewardConfigs).toHaveProperty('attack');
    expect(typeof rewardConfigs.attack).toBe('object');
  });

  describe('Default Configuration', () => {
    const config = rewardConfigs.default;

    test('should have all required reward properties', () => {
      expect(config).toHaveProperty('baseReward');
      expect(config).toHaveProperty('hitShotReward');
      expect(config).toHaveProperty('biteReward');
      expect(config).toHaveProperty('hitHumanReward');
      expect(config).toHaveProperty('missedShotReward');
      expect(config).toHaveProperty('bumpWallReward');
      expect(config).toHaveProperty('bumpScreenReward');
      expect(config).toHaveProperty('bumpHumanReward');
      expect(config).toHaveProperty('blockedVisionHuman');
      expect(config).toHaveProperty('blockedVisionWall');
      expect(config).toHaveProperty('farVisionReward');
      expect(config).toHaveProperty('zombieProximityReward');
    });

    test('should have correct reward values', () => {
      expect(config.baseReward).toBe(1);
      expect(config.hitShotReward).toBe(1);
      expect(config.biteReward).toBe(-1);
      expect(config.hitHumanReward).toBe(-1);
      expect(config.missedShotReward).toBe(-0.125);
      expect(config.bumpWallReward).toBe(-0.55);
      expect(config.bumpScreenReward).toBe(-0.75);
      expect(config.bumpHumanReward).toBe(-0.65);
      expect(config.blockedVisionHuman).toBe(0);
      expect(config.blockedVisionWall).toBe(-0.1);
      expect(config.farVisionReward).toBe(0);
      expect(config.zombieProximityReward).toBe(0);
    });

    test('should have positive rewards for good actions', () => {
      expect(config.baseReward).toBeGreaterThan(0);
      expect(config.hitShotReward).toBeGreaterThan(0);
    });

    test('should have negative rewards for bad actions', () => {
      expect(config.biteReward).toBeLessThan(0);
      expect(config.hitHumanReward).toBeLessThan(0);
      expect(config.missedShotReward).toBeLessThan(0);
      expect(config.bumpWallReward).toBeLessThan(0);
      expect(config.bumpScreenReward).toBeLessThan(0);
      expect(config.bumpHumanReward).toBeLessThan(0);
    });
  });

  describe('Explore Configuration', () => {
    const config = rewardConfigs.explore;

    test('should have all required reward properties', () => {
      expect(config).toHaveProperty('baseReward');
      expect(config).toHaveProperty('hitShotReward');
      expect(config).toHaveProperty('biteReward');
      expect(config).toHaveProperty('hitHumanReward');
      expect(config).toHaveProperty('missedShotReward');
      expect(config).toHaveProperty('bumpWallReward');
      expect(config).toHaveProperty('bumpScreenReward');
      expect(config).toHaveProperty('bumpHumanReward');
      expect(config).toHaveProperty('blockedVisionHuman');
      expect(config).toHaveProperty('blockedVisionWall');
      expect(config).toHaveProperty('farVisionReward');
      expect(config).toHaveProperty('zombieProximityReward');
    });

    test('should encourage exploration with different wall penalties', () => {
      expect(config.bumpWallReward).toBe(-0.5);
      expect(config.bumpScreenReward).toBe(-0.5);
      expect(config.blockedVisionWall).toBe(-0.5);
    });

    test('should discourage zombie proximity', () => {
      expect(config.zombieProximityReward).toBe(-0.1);
    });
  });

  describe('Attack Configuration', () => {
    const config = rewardConfigs.attack;

    test('should have all required reward properties', () => {
      expect(config).toHaveProperty('baseReward');
      expect(config).toHaveProperty('hitShotReward');
      expect(config).toHaveProperty('biteReward');
      expect(config).toHaveProperty('hitHumanReward');
      expect(config).toHaveProperty('missedShotReward');
      expect(config).toHaveProperty('bumpWallReward');
      expect(config).toHaveProperty('bumpScreenReward');
      expect(config).toHaveProperty('bumpHumanReward');
      expect(config).toHaveProperty('blockedVisionHuman');
      expect(config).toHaveProperty('blockedVisionWall');
      expect(config).toHaveProperty('farVisionReward');
      expect(config).toHaveProperty('zombieProximityReward');
    });

    test('should encourage aggressive behavior', () => {
      expect(config.hitShotReward).toBe(1.5);
      expect(config.farVisionReward).toBe(0.2);
    });

    test('should penalize blocked vision more heavily', () => {
      expect(config.blockedVisionHuman).toBe(-0.5);
      expect(config.blockedVisionWall).toBe(-0.2);
    });

    test('should discourage zombie proximity', () => {
      expect(config.zombieProximityReward).toBe(-0.1);
    });
  });

  describe('Configuration Consistency', () => {
    test('all configurations should have the same properties', () => {
      const defaultKeys = Object.keys(rewardConfigs.default).sort();
      const exploreKeys = Object.keys(rewardConfigs.explore).sort();
      const attackKeys = Object.keys(rewardConfigs.attack).sort();

      expect(exploreKeys).toEqual(defaultKeys);
      expect(attackKeys).toEqual(defaultKeys);
    });

    test('all configurations should have numeric values', () => {
      Object.values(rewardConfigs).forEach(config => {
        Object.values(config).forEach(value => {
          expect(typeof value).toBe('number');
          expect(isNaN(value)).toBe(false);
        });
      });
    });

    test('critical rewards should be consistent across configs', () => {
      expect(rewardConfigs.default.baseReward).toBe(rewardConfigs.explore.baseReward);
      expect(rewardConfigs.default.baseReward).toBe(rewardConfigs.attack.baseReward);
      
      expect(rewardConfigs.default.biteReward).toBe(rewardConfigs.explore.biteReward);
      expect(rewardConfigs.default.biteReward).toBe(rewardConfigs.attack.biteReward);
      
      expect(rewardConfigs.default.hitHumanReward).toBe(rewardConfigs.explore.hitHumanReward);
      expect(rewardConfigs.default.hitHumanReward).toBe(rewardConfigs.attack.hitHumanReward);
    });
  });

  describe('Reward Value Ranges', () => {
    test('should have reasonable reward ranges', () => {
      Object.values(rewardConfigs).forEach(config => {
        // Base reward should be positive
        expect(config.baseReward).toBeGreaterThan(0);
        expect(config.baseReward).toBeLessThanOrEqual(2);
        
        // Hit shot reward should be positive
        expect(config.hitShotReward).toBeGreaterThan(0);
        expect(config.hitShotReward).toBeLessThanOrEqual(2);
        
        // Negative rewards should not be too harsh
        expect(config.biteReward).toBeGreaterThanOrEqual(-2);
        expect(config.hitHumanReward).toBeGreaterThanOrEqual(-2);
        expect(config.missedShotReward).toBeGreaterThanOrEqual(-1);
        expect(config.bumpWallReward).toBeGreaterThanOrEqual(-1);
        expect(config.bumpScreenReward).toBeGreaterThanOrEqual(-1);
        expect(config.bumpHumanReward).toBeGreaterThanOrEqual(-1);
      });
    });
  });
});
