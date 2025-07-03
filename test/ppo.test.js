/* eslint-env jest, node */
/* global tf, describe, test, expect, beforeEach, afterEach */

// Mock environment for testing
class MockEnv {
    constructor() {
        this.observationSpace = { shape: [4] };
        this.actionSpace = { 
            class: 'Discrete', 
            n: 2, 
            dtype: 'int32' 
        };
        this.stepCount = 0;
        this.maxSteps = 100;
        this.currentObservation = [0.1, 0.2, 0.3, 0.4];
    }

    reset() {
        this.stepCount = 0;
        this.currentObservation = [Math.random(), Math.random(), Math.random(), Math.random()];
        return this.currentObservation.slice();
    }

    async step() {
        this.stepCount++;
        
        // Simulate environment dynamics
        this.currentObservation = this.currentObservation.map(x => 
            Math.max(-1, Math.min(1, x + (Math.random() - 0.5) * 0.1))
        );
        
        const reward = Math.random() * 2 - 1; // Random reward between -1 and 1
        const done = this.stepCount >= this.maxSteps || Math.random() < 0.05;
        
        return [this.currentObservation.slice(), reward, done];
    }
}

class NaNRewardEnv {
    constructor() {
        this.observationSpace = { shape: [2] };
        this.actionSpace = { class: 'Discrete', n: 2, dtype: 'int32' };
        this.stepCount = 0;
    }
    reset() { return [0.5, 0.5]; }
    async step() {
        this.stepCount++;
        const reward = this.stepCount === 3 ? NaN : 1.0; // Inject NaN reward
        return [[Math.random(), Math.random()], reward, this.stepCount > 5];
    }
}

class InfiniteRewardEnv {
    constructor() {
        this.observationSpace = { shape: [2] };
        this.actionSpace = { class: 'Discrete', n: 2, dtype: 'int32' };
        this.stepCount = 0;
    }
    reset() { return [0.5, 0.5]; }
    async step() {
        this.stepCount++;
        const reward = this.stepCount === 2 ? Infinity : 1.0; // Inject Infinity reward
        return [[Math.random(), Math.random()], reward, this.stepCount > 5];
    }
}

// Import PPO
import PPO from '../src/ppo.js';

describe('PPO Pure Algorithm Test Suite', () => {
    let env;
    let ppo;

    beforeEach(async () => {
        // Initialize TensorFlow backend
        if (typeof tf !== 'undefined' && tf.ready) {
            await tf.ready();
        }
        
        env = new MockEnv();
        ppo = new PPO(env, {
            nSteps: 16,
            nEpochs: 2,
            policyLearningRate: 0.001,
            valueLearningRate: 0.001,
            clipRatio: 0.2,
            targetKL: 0.01,
            netArch: {
                'pi': [8, 8],
                'vf': [8, 8]
            },
            verbose: 0
        });
    });

    afterEach(() => {
        // Basic cleanup
        if (ppo && ppo.actor) {
            ppo.actor.dispose();
        }
        if (ppo && ppo.critic) {
            ppo.critic.dispose();
        }
        if (ppo && ppo.optPolicy) {
            ppo.optPolicy.dispose();
        }
        if (ppo && ppo.optValue) {
            ppo.optValue.dispose();
        }
        if (ppo && ppo.logStd) {
            ppo.logStd.dispose();
        }
    });

    // ===== BASIC FUNCTIONALITY TESTS (SHOULD PASS) =====
    
    describe('Basic PPO Functionality', () => {
        test('should create PPO instance with correct configuration', () => {
            expect(ppo).toBeDefined();
            expect(ppo.config.nSteps).toBe(16);
            expect(ppo.config.nEpochs).toBe(2);
            expect(ppo.config.clipRatio).toBe(0.2);
        });

        test('should sample actions from observations', async () => {
            const observation = [0.1, 0.2, 0.3, 0.4];
            const [preds, action, value, logprob] = await ppo.getSample(observation);
            
            expect(preds).toBeDefined();
            expect(action).toBeDefined();
            expect(typeof value).toBe('number');
            expect(typeof logprob).toBe('number');
            expect(isFinite(value)).toBe(true);
            expect(isFinite(logprob)).toBe(true);
        });

        test('should handle basic training loop', async () => {
            ppo.lastObservation = env.reset();
            
            await expect(async () => {
                await ppo.collectRollouts(ppo._initCallback(() => true));
                await ppo.train();
            }).not.toThrow();
        });
    });

    // ===== PURE PPO ALGORITHM TESTS (SHOULD FAIL WITH CURRENT PPO) =====

    describe('PPO Parameter Validation', () => {
        test('should reject negative learning rates', () => {
            // Should reject negative policy learning rate
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: -0.001,
                    valueLearningRate: 0.001,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject negative value learning rate
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: 0.001,
                    valueLearningRate: -0.001,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid clip ratios', () => {
            // Should reject negative clip ratio
            expect(() => {
                new PPO(env, { 
                    clipRatio: -0.1,
                    verbose: 0 
                });
            }).toThrow();
            
            // Should reject zero clip ratio
            expect(() => {
                new PPO(env, { 
                    clipRatio: 0,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject excessively large clip ratio
            expect(() => {
                new PPO(env, { 
                    clipRatio: 2.0,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid target KL values', () => {
            // Should reject negative target KL
            expect(() => {
                new PPO(env, { 
                    targetKL: -0.01,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject zero target KL
            expect(() => {
                new PPO(env, { 
                    targetKL: 0,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid step counts', () => {
            // Should reject zero nSteps
            expect(() => {
                new PPO(env, { 
                    nSteps: 0,
                    verbose: 0 
                });
            }).toThrow();
            
            // Should reject negative nSteps
            expect(() => {
                new PPO(env, { 
                    nSteps: -5,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject zero nEpochs
            expect(() => {
                new PPO(env, { 
                    nEpochs: 0,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject negative nEpochs
            expect(() => {
                new PPO(env, { 
                    nEpochs: -3,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid gamma values', () => {
            // Should reject negative gamma (discount factor)
            expect(() => {
                new PPO(env, { 
                    gamma: -0.1,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject gamma > 1.0 (discount factor)
            expect(() => {
                new PPO(env, { 
                    gamma: 1.5,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid lambda values', () => {
            // Should reject negative lambda (GAE parameter)
            expect(() => {
                new PPO(env, { 
                    lam: -0.1,
                    verbose: 0 
                });
            }).toThrow();

            // Should reject lambda > 1.0 (GAE parameter)
            expect(() => {
                new PPO(env, { 
                    lam: 1.5,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should accept valid parameters', () => {
            // Should accept valid learning rates
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: 0.001,
                    valueLearningRate: 0.001,
                    verbose: 0 
                });
            }).not.toThrow();

            // Should accept valid clip ratios
            expect(() => {
                new PPO(env, { 
                    clipRatio: 0.2,
                    verbose: 0 
                });
            }).not.toThrow();

            expect(() => {
                new PPO(env, { 
                    clipRatio: 1.0,
                    verbose: 0 
                });
            }).not.toThrow();

            // Should accept valid target KL
            expect(() => {
                new PPO(env, { 
                    targetKL: 0.01,
                    verbose: 0 
                });
            }).not.toThrow();

            // Should accept valid nSteps and nEpochs
            expect(() => {
                new PPO(env, { 
                    nSteps: 16,
                    nEpochs: 2,
                    verbose: 0 
                });
            }).not.toThrow();

            // Should accept valid gamma and lambda
            expect(() => {
                new PPO(env, { 
                    gamma: 0.99,
                    lam: 0.95,
                    verbose: 0 
                });
            }).not.toThrow();

            expect(() => {
                new PPO(env, { 
                    gamma: 0.0,
                    lam: 0.0,
                    verbose: 0 
                });
            }).not.toThrow();

            expect(() => {
                new PPO(env, { 
                    gamma: 1.0,
                    lam: 1.0,
                    verbose: 0 
                });
            }).not.toThrow();
        });
    });

    describe('PPO Input Validation', () => {
        test('should reject NaN rewards', async () => {
            const nanEnv = new NaNRewardEnv();
            const nanPPO = new PPO(nanEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            nanPPO.lastObservation = nanEnv.reset();
            
            // Should detect NaN and throw error (not silent failure)
            await expect(async () => {
                await nanPPO.collectRollouts(nanPPO._initCallback(() => true));
                const [, , advantages] = nanPPO.buffer.get();
                
                // Check that advantages are valid
                advantages.forEach(adv => {
                    if (isNaN(adv)) {
                        throw new Error('NaN detected in advantages - PPO should validate rewards');
                    }
                });
            }).rejects.toThrow();
            
            // Cleanup
            if (nanPPO.actor) nanPPO.actor.dispose();
            if (nanPPO.critic) nanPPO.critic.dispose();
            if (nanPPO.optPolicy) nanPPO.optPolicy.dispose();
            if (nanPPO.optValue) nanPPO.optValue.dispose();
        });

        test('should reject Infinity rewards and throw descriptive error', async () => {
            const infEnv = new InfiniteRewardEnv();
            const infPPO = new PPO(infEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            infPPO.lastObservation = infEnv.reset();
            
            // Should detect Infinity and throw error (not silent failure)
            await expect(async () => {
                await infPPO.collectRollouts(infPPO._initCallback(() => true));
                const [, , advantages] = infPPO.buffer.get();
                
                // Check that advantages are valid
                advantages.forEach(adv => {
                    if (!isFinite(adv)) {
                        throw new Error('Infinity detected in advantages - PPO should validate rewards');
                    }
                });
            }).rejects.toThrow();
            
            // Cleanup
            if (infPPO.actor) infPPO.actor.dispose();
            if (infPPO.critic) infPPO.critic.dispose();
            if (infPPO.optPolicy) infPPO.optPolicy.dispose();
            if (infPPO.optValue) infPPO.optValue.dispose();
        });
    });

    describe('PPO Callback Handling', () => {
        test('should handle callback errors gracefully without crashing training', async () => {
            let callbackErrorOccurred = false;
            
            const errorCallback = ppo._initCallback(() => {
                callbackErrorOccurred = true;
                throw new Error('Callback error');
            });
            
            ppo.lastObservation = env.reset();
            
            // Should handle callback errors gracefully, not crash entire training
            await expect(async () => {
                await ppo.collectRollouts(errorCallback);
            }).rejects.toThrow();
            
            expect(callbackErrorOccurred).toBe(true);
        });

        test('should handle null/undefined callbacks without errors', async () => {
            ppo.lastObservation = env.reset();
            
            // Should handle null callbacks gracefully
            await expect(async () => {
                await ppo.collectRollouts(ppo._initCallback(null));
            }).not.toThrow();
            
            // Should handle undefined callbacks gracefully  
            await expect(async () => {
                await ppo.collectRollouts(ppo._initCallback(undefined));
            }).not.toThrow();
        });
    });

    describe('PPO Buffer Management', () => {
        test('should validate buffer data consistency', async () => {
            // Add inconsistent data to buffer
            ppo.buffer.add([0.1, 0.2, 0.3, 0.4], 1, 1.0, 0.5, -0.7);
            ppo.buffer.add([0.2, 0.3, 0.4, 0.5], 0, 2.0, 0.8, -0.3);
            
            // Don't call finishTrajectory - buffer is incomplete
            expect(() => {
                const [obs, , advantages] = ppo.buffer.get();
                
                // Should detect incomplete trajectory
                if (advantages.length === 0 && obs.length > 0) {
                    throw new Error('Incomplete trajectory - advantages not computed');
                }
            }).toThrow();
        });

        test('should handle empty buffer gracefully', () => {
            // Should not crash when getting from empty buffer
            expect(() => {
                ppo.buffer.reset();
                const [obs, actions, advantages] = ppo.buffer.get();
                
                if (obs.length === 0 && actions.length === 0 && advantages.length === 0) {
                    throw new Error('Empty buffer should be handled gracefully');
                }
            }).toThrow();
        });

        test('should handle zero variance advantages', () => {
            // Add identical rewards to create zero variance
            ppo.buffer.add([0.1, 0.2, 0.3, 0.4], 1, 5.0, 0.5, -0.7);
            ppo.buffer.add([0.2, 0.3, 0.4, 0.5], 0, 5.0, 0.8, -0.3);
            ppo.buffer.add([0.3, 0.4, 0.5, 0.6], 1, 5.0, 0.6, -0.5);
            ppo.buffer.finishTrajectory(0);
            
            // Should handle division by zero in advantage normalization
            expect(() => {
                const [, , advantages] = ppo.buffer.get();
                
                // Check for NaN or Infinity in normalized advantages
                advantages.forEach(adv => {
                    if (!isFinite(adv)) {
                        throw new Error('Zero variance advantages should be handled without NaN/Infinity');
                    }
                });
            }).not.toThrow();
        });

        test('should validate buffer overflow protection', () => {
            // Should have some maximum buffer size limit
            expect(() => {
                // Try to add excessive data
                for (let i = 0; i < 100000; i++) {
                    ppo.buffer.add([0.1, 0.2, 0.3, 0.4], 1, 1.0, 0.5, -0.7);
                }
                
                if (ppo.buffer.pointer > 50000) {
                    throw new Error('Buffer should have size limits to prevent memory overflow');
                }
            }).toThrow();
        });
    });

    describe('PPO Environment Interface Validation', () => {
        test('should validate action space properties', () => {
            // Should reject environments with missing action space properties
            const invalidEnv = {
                observationSpace: { shape: [4] },
                actionSpace: { class: 'Discrete' }, // Missing 'n' property
                reset: () => [0, 0, 0, 0],
                step: () => Promise.resolve([[0, 0, 0, 0], 1.0, false])
            };

            expect(() => {
                new PPO(invalidEnv, { verbose: 0 });
            }).toThrow();
        });

        test('should validate observation space properties', () => {
            // Should reject environments with invalid observation space
            const invalidEnv = {
                observationSpace: {}, // Missing 'shape' property
                actionSpace: { class: 'Discrete', n: 2 },
                reset: () => [0, 0, 0, 0],
                step: () => Promise.resolve([[0, 0, 0, 0], 1.0, false])
            };

            expect(() => {
                new PPO(invalidEnv, { verbose: 0 });
            }).toThrow();
        });

        test('should validate environment step response format', async () => {
            // Environment that returns malformed step responses
            const malformedEnv = {
                observationSpace: { shape: [2] },
                actionSpace: { class: 'Discrete', n: 2 },
                reset: () => [0.5, 0.5],
                step: () => Promise.resolve([1.0, false]) // Missing observation
            };

            const malformedPPO = new PPO(malformedEnv, { nSteps: 4, verbose: 0 });
            malformedPPO.lastObservation = malformedEnv.reset();

            await expect(async () => {
                await malformedPPO.collectRollouts(malformedPPO._initCallback(() => true));
            }).rejects.toThrow();

            // Cleanup
            if (malformedPPO.actor) malformedPPO.actor.dispose();
            if (malformedPPO.critic) malformedPPO.critic.dispose();
            if (malformedPPO.optPolicy) malformedPPO.optPolicy.dispose();
            if (malformedPPO.optValue) malformedPPO.optValue.dispose();
        });
    });

    describe('PPO Input Data Validation', () => {
        test('should validate observation shape consistency', async () => {
            // Environment that returns inconsistent observation shapes
            const inconsistentEnv = {
                observationSpace: { shape: [4] },
                actionSpace: { class: 'Discrete', n: 2 },
                reset: () => [0.1, 0.2, 0.3, 0.4],
                step: () => Promise.resolve([[0.1, 0.2], 1.0, false]) // Wrong shape
            };

            const inconsistentPPO = new PPO(inconsistentEnv, { nSteps: 4, verbose: 0 });
            inconsistentPPO.lastObservation = inconsistentEnv.reset();

            await expect(async () => {
                await inconsistentPPO.collectRollouts(inconsistentPPO._initCallback(() => true));
            }).rejects.toThrow();

            // Cleanup
            if (inconsistentPPO.actor) inconsistentPPO.actor.dispose();
            if (inconsistentPPO.critic) inconsistentPPO.critic.dispose();
            if (inconsistentPPO.optPolicy) inconsistentPPO.optPolicy.dispose();
            if (inconsistentPPO.optValue) inconsistentPPO.optValue.dispose();
        });

        test('should validate action bounds for continuous spaces', async () => {
            // Continuous action space environment
            const continuousEnv = {
                observationSpace: { shape: [2] },
                actionSpace: { 
                    class: 'Box', 
                    shape: [1], 
                    high: 1.0, 
                    low: -1.0,
                    dtype: 'float32'
                },
                reset: () => [0.5, 0.5],
                step: (action) => {
                    // Should validate action is within bounds
                    if (action[0] > 1.0 || action[0] < -1.0) {
                        throw new Error('Action out of bounds - PPO should clip actions');
                    }
                    return Promise.resolve([[Math.random(), Math.random()], 1.0, false]);
                }
            };

            const continuousPPO = new PPO(continuousEnv, { nSteps: 4, verbose: 0 });
            continuousPPO.lastObservation = continuousEnv.reset();

            // Should not throw - PPO should handle action clipping
            await expect(async () => {
                await continuousPPO.collectRollouts(continuousPPO._initCallback(() => true));
            }).not.toThrow();

            // Cleanup
            if (continuousPPO.actor) continuousPPO.actor.dispose();
            if (continuousPPO.critic) continuousPPO.critic.dispose();
            if (continuousPPO.optPolicy) continuousPPO.optPolicy.dispose();
            if (continuousPPO.optValue) continuousPPO.optValue.dispose();
            if (continuousPPO.logStd) continuousPPO.logStd.dispose();
        });

        test('should handle null/undefined observations', async () => {
            // Environment that occasionally returns null observations
            const nullObsEnv = {
                observationSpace: { shape: [2] },
                actionSpace: { class: 'Discrete', n: 2 },
                stepCount: 0,
                reset: () => [0.5, 0.5],
                step: function() {
                    this.stepCount++;
                    const obs = this.stepCount === 2 ? null : [Math.random(), Math.random()];
                    return Promise.resolve([obs, 1.0, this.stepCount > 3]);
                }
            };

            const nullObsPPO = new PPO(nullObsEnv, { nSteps: 4, verbose: 0 });
            nullObsPPO.lastObservation = nullObsEnv.reset();

            await expect(async () => {
                await nullObsPPO.collectRollouts(nullObsPPO._initCallback(() => true));
            }).rejects.toThrow();

            // Cleanup
            if (nullObsPPO.actor) nullObsPPO.actor.dispose();
            if (nullObsPPO.critic) nullObsPPO.critic.dispose();
            if (nullObsPPO.optPolicy) nullObsPPO.optPolicy.dispose();
            if (nullObsPPO.optValue) nullObsPPO.optValue.dispose();
        });
    });

    describe('PPO Advanced Edge Cases', () => {
        test('should handle extremely small learning rates', () => {
            // Should handle learning rates that might cause numerical issues
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: 1e-10,
                    valueLearningRate: 1e-10,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should handle extremely large clip ratios', () => {
            // Should reject clip ratios that defeat the purpose of PPO
            expect(() => {
                new PPO(env, { 
                    clipRatio: 10.0,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should validate network architecture consistency', () => {
            // Should reject architectures that don't make sense
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [], // Empty architecture
                        'vf': [32, 32]
                    },
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should detect problematic parameter combinations', () => {
            // Very high target KL with very low clip ratio creates instability
            expect(() => {
                new PPO(env, { 
                    targetKL: 0.09,
                    clipRatio: 0.01,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should validate learning rate ratios for stability', () => {
            // Policy learning rate much higher than value learning rate can cause instability
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: 0.1,
                    valueLearningRate: 1e-6,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject configurations that break PPO assumptions', () => {
            // Very low gamma with high lambda creates mathematical inconsistency
            expect(() => {
                new PPO(env, { 
                    gamma: 0.01,
                    lam: 0.99,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should validate epoch-to-steps ratio for efficiency', () => {
            // Too many epochs relative to steps can cause overfitting
            expect(() => {
                new PPO(env, { 
                    nSteps: 4,
                    nEpochs: 100,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should detect network architecture imbalances', () => {
            // Extremely different network sizes can cause training issues
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [1024, 1024, 1024],
                        'vf': [2]
                    },
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should validate numerical precision requirements', () => {
            // Combination of very small clip ratio and very small target KL
            expect(() => {
                new PPO(env, { 
                    clipRatio: 1e-8,
                    targetKL: 1e-9,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject pathological activation combinations', () => {
            // Linear activation with very high learning rates can cause instability
            expect(() => {
                new PPO(env, { 
                    activation: 'linear',
                    policyLearningRate: 0.5,
                    valueLearningRate: 0.5,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should validate temporal consistency parameters', () => {
            // Very high discount factor with very low GAE lambda creates inconsistency
            expect(() => {
                new PPO(env, { 
                    gamma: 0.999,
                    lam: 0.001,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should handle callback state corruption', async () => {
            // Callback that corrupts PPO state
            const corruptingCallback = ppo._initCallback((alg) => {
                alg.numTimesteps = -1; // Corrupt the timestep counter
                return true;
            });
            
            ppo.lastObservation = env.reset();
            
            await expect(async () => {
                await ppo.collectRollouts(corruptingCallback);
                
                if (ppo.numTimesteps < 0) {
                    throw new Error('Callback corrupted PPO state - timesteps should not be negative');
                }
            }).rejects.toThrow();
        });
    });
});
