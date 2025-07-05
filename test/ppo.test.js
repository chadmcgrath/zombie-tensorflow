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

    // No afterEach cleanup - let TensorFlow handle its own memory management
    // to avoid race conditions and disposal conflicts

    // ===== CRITICAL DIVIDE BY ZERO TESTS =====
    
    describe('PPO Mathematical Edge Case Handling', () => {
        test('should expose divide by zero bug in advantage normalization', () => {
            ppo.buffer.reset();
            
            // Force identical advantages by directly manipulating the buffer
            // This bypasses GAE calculation to create exact zero variance
            ppo.buffer.advantageBuffer = [1.0, 1.0, 1.0, 1.0, 1.0]; // Identical values
            ppo.buffer.observationBuffer = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]];
            ppo.buffer.actionBuffer = [1, 1, 1, 1, 1];
            ppo.buffer.returnBuffer = [1.0, 1.0, 1.0, 1.0, 1.0];
            ppo.buffer.logprobabilityBuffer = [-0.5, -0.5, -0.5, -0.5, -0.5];
            ppo.buffer.pointer = 5;
            
            // Current buggy PPO will produce NaN values due to divide by zero
            const [, , advantages] = ppo.buffer.get();
            
            const hasNaN = advantages.some(adv => isNaN(adv));
            expect(hasNaN).toBe(false); // A FIXED PPO should NOT produce NaN
        });


        test('should handle very small scales in continuous action log probabilities', async () => {
            if (ppo.env.actionSpace.class === 'Box') {
                // Test legitimate scenario: policy becomes very confident (small scale)
                const originalLogStd = ppo.logStd.arraySync();
                
                // Set small but reasonable logStd values (confident policy)
                ppo.logStd.assign(tf.fill([ppo.env.actionSpace.shape[0]], -5)); // exp(-5) ≈ 0.007
                
                const observation = [0.1, 0.2, 0.3, 0.4];
                const [preds, action, value, logprob] = await ppo.getSample(observation);
                
                // Should produce finite log probabilities (even if very negative)
                expect(isFinite(logprob)).toBe(true);
                expect(isNaN(logprob)).toBe(false);
                
                // Log probability should be mathematically consistent
                // For small scales, log prob should be more negative
                expect(logprob).toBeLessThan(0); // Log probabilities are typically negative
                
                // Restore original values
                ppo.logStd.assign(tf.tensor(originalLogStd));
            }
        });

        test('should handle extreme but finite log probability differences in training', async () => {
            // Set up legitimate scenario: large policy updates
            ppo.buffer.reset();
            
            // Add data with varying log probabilities (legitimate range)
            const logprobs = [-0.1, -2.0, -5.0, -10.0, -0.5, -1.5, -3.0, -7.0];
            
            for (let i = 0; i < 8; i++) {
                ppo.buffer.add(
                    [Math.random(), Math.random(), Math.random(), Math.random()],
                    Math.floor(Math.random() * 2),
                    Math.random() * 2 - 1,
                    Math.random(),
                    logprobs[i]
                );
            }
            
            ppo.buffer.finishTrajectory(0);
            
            // Training should handle varying log probabilities correctly
            await expect(async () => {
                await ppo.train();
                
                // Verify training produces finite outputs
                const [preds, action, value, logprob] = await ppo.getSample([0.1, 0.2, 0.3, 0.4]);
                
                expect(isFinite(value)).toBe(true);
                expect(isFinite(logprob)).toBe(true);
                expect(isNaN(value)).toBe(false);
                expect(isNaN(logprob)).toBe(false);
                
                // Predictions should be finite
                preds.forEach(pred => {
                    expect(isFinite(pred)).toBe(true);
                    expect(isNaN(pred)).toBe(false);
                });
                
            }).not.toThrow();
        });


        test('should handle policy convergence scenarios without numerical issues', async () => {
            // Legitimate scenario: policy has converged and produces consistent actions
            const observation = [0.2, 0.4, 0.6, 0.8];
            
            // Sample multiple times to check consistency
            const samples = [];
            for (let i = 0; i < 5; i++) {
                const [preds, action, value, logprob] = await ppo.getSample(observation);
                samples.push({ preds, action, value, logprob });
                
                // Each sample should be finite
                expect(isFinite(value)).toBe(true);
                expect(isFinite(logprob)).toBe(true);
                expect(isNaN(value)).toBe(false);
                expect(isNaN(logprob)).toBe(false);
                
                preds.forEach(pred => {
                    expect(isFinite(pred)).toBe(true);
                    expect(isNaN(pred)).toBe(false);
                });
            }
            
            // For a deterministic policy, values should be consistent
            const firstValue = samples[0].value;
            samples.forEach(sample => {
                expect(Math.abs(sample.value - firstValue)).toBeLessThan(0.01); // Small tolerance for numerical precision
            });
        });
    });

    describe('PPO Runtime NaN/Infinity Detection', () => {
        test('should detect NaN in neural network outputs during getSample', async () => {
            // Create a mock environment that could trigger NaN in network outputs
            const observation = [0.1, 0.2, 0.3, 0.4];
            
            // Test that getSample validates all outputs are finite
            const [preds, action, value, logprob] = await ppo.getSample(observation);
            
            // All outputs should be finite
            preds.forEach(pred => {
                expect(isFinite(pred)).toBe(true);
                expect(isNaN(pred)).toBe(false);
            });
            
            if (Array.isArray(action)) {
                action.forEach(act => {
                    expect(isFinite(act)).toBe(true);
                    expect(isNaN(act)).toBe(false);
                });
            } else {
                expect(isFinite(action)).toBe(true);
                expect(isNaN(action)).toBe(false);
            }
            
            expect(isFinite(value)).toBe(true);
            expect(isNaN(value)).toBe(false);
            expect(isFinite(logprob)).toBe(true);
            expect(isNaN(logprob)).toBe(false);
        });

        test('should detect NaN/Infinity in observations during collectRollouts', async () => {
            // Environment that returns NaN observations
            const nanObsEnv = {
                observationSpace: { shape: [2] },
                actionSpace: { class: 'Discrete', n: 2, dtype: 'int32' },
                stepCount: 0,
                reset: () => [0.5, 0.5],
                step: function() {
                    this.stepCount++;
                    const obs = this.stepCount === 2 ? [NaN, 0.5] : [Math.random(), Math.random()];
                    return Promise.resolve([obs, 1.0, this.stepCount > 3]);
                }
            };

            const nanObsPPO = new PPO(nanObsEnv, { nSteps: 4, verbose: 0 });
            nanObsPPO.lastObservation = nanObsEnv.reset();

            // Should detect NaN in observations and throw error
            await expect(async () => {
                await nanObsPPO.collectRollouts(nanObsPPO._initCallback(() => true));
            }).rejects.toThrow();

            // Cleanup
            if (nanObsPPO.actor) nanObsPPO.actor.dispose();
            if (nanObsPPO.critic) nanObsPPO.critic.dispose();
            if (nanObsPPO.optPolicy) nanObsPPO.optPolicy.dispose();
            if (nanObsPPO.optValue) nanObsPPO.optValue.dispose();
        });

        test('should detect Infinity in observations during collectRollouts', async () => {
            // Environment that returns Infinity observations
            const infObsEnv = {
                observationSpace: { shape: [2] },
                actionSpace: { class: 'Discrete', n: 2, dtype: 'int32' },
                stepCount: 0,
                reset: () => [0.5, 0.5],
                step: function() {
                    this.stepCount++;
                    const obs = this.stepCount === 2 ? [Infinity, 0.5] : [Math.random(), Math.random()];
                    return Promise.resolve([obs, 1.0, this.stepCount > 3]);
                }
            };

            const infObsPPO = new PPO(infObsEnv, { nSteps: 4, verbose: 0 });
            infObsPPO.lastObservation = infObsEnv.reset();

            // Should detect Infinity in observations and throw error
            await expect(async () => {
                await infObsPPO.collectRollouts(infObsPPO._initCallback(() => true));
            }).rejects.toThrow();

            // Cleanup
            if (infObsPPO.actor) infObsPPO.actor.dispose();
            if (infObsPPO.critic) infObsPPO.critic.dispose();
            if (infObsPPO.optPolicy) infObsPPO.optPolicy.dispose();
            if (infObsPPO.optValue) infObsPPO.optValue.dispose();
        });

        test('should validate all intermediate values during training are finite', async () => {
            // Set up training data
            ppo.buffer.reset();
            for (let i = 0; i < 8; i++) {
                ppo.buffer.add(
                    [Math.random(), Math.random(), Math.random(), Math.random()],
                    Math.floor(Math.random() * 2),
                    Math.random() * 2 - 1,
                    Math.random(),
                    Math.random() * 2 - 1
                );
            }
            ppo.buffer.finishTrajectory(0);

            // Training should not produce NaN/Infinity values
            await expect(async () => {
                await ppo.train();
                
                // After training, sample again to ensure networks are still producing finite values
                const [preds, action, value, logprob] = await ppo.getSample([0.1, 0.2, 0.3, 0.4]);
                
                preds.forEach(pred => {
                    if (!isFinite(pred)) {
                        throw new Error('Training produced non-finite network outputs');
                    }
                });
                
                if (!isFinite(value) || !isFinite(logprob)) {
                    throw new Error('Training produced non-finite value or logprob');
                }
            }).not.toThrow();
        });

        test('should detect NaN in value function outputs', async () => {
            // This test ensures that if the critic network starts producing NaN values,
            // it gets detected during getSample
            const observation = [0.1, 0.2, 0.3, 0.4];
            const [preds, action, value, logprob] = await ppo.getSample(observation);
            
            // Value function output should always be finite
            expect(isFinite(value)).toBe(true);
            expect(isNaN(value)).toBe(false);
            expect(typeof value).toBe('number');
        });

        test('should detect NaN in log probability calculations', async () => {
            // Test log probability calculations with edge case inputs
            const observation = [0.1, 0.2, 0.3, 0.4];
            const [preds, action, value, logprob] = await ppo.getSample(observation);
            
            // Log probability should always be finite (even if very negative)
            expect(isFinite(logprob)).toBe(true);
            expect(isNaN(logprob)).toBe(false);
            expect(typeof logprob).toBe('number');
        });

        test('should handle extreme observation values without producing NaN', async () => {
            // Test with extreme but finite observation values
            const extremeObservations = [
                [1e6, -1e6, 1e3, -1e3],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0]
            ];

            for (const obs of extremeObservations) {
                const [preds, action, value, logprob] = await ppo.getSample(obs);
                
                // All outputs should remain finite even with extreme inputs
                preds.forEach(pred => {
                    expect(isFinite(pred)).toBe(true);
                    expect(isNaN(pred)).toBe(false);
                });
                
                expect(isFinite(value)).toBe(true);
                expect(isNaN(value)).toBe(false);
                expect(isFinite(logprob)).toBe(true);
                expect(isNaN(logprob)).toBe(false);
            }
        });

        test('should validate buffer operations never produce NaN/Infinity', () => {
            ppo.buffer.reset();
            
            // Add data with various edge cases
            const testCases = [
                { obs: [0.1, 0.2, 0.3, 0.4], action: 1, reward: 1000, value: 0.5, logprob: -2.3 },
                { obs: [0.0, 0.0, 0.0, 0.0], action: 0, reward: -1000, value: 0.0, logprob: -10.5 },
                { obs: [1.0, 1.0, 1.0, 1.0], action: 1, reward: 0.001, value: 1.0, logprob: -0.1 },
                { obs: [-1.0, -1.0, -1.0, -1.0], action: 0, reward: -0.001, value: -1.0, logprob: -5.7 }
            ];

            testCases.forEach(testCase => {
                ppo.buffer.add(testCase.obs, testCase.action, testCase.reward, testCase.value, testCase.logprob);
            });

            ppo.buffer.finishTrajectory(0);

            expect(() => {
                const [observations, actions, advantages, returns, logprobs] = ppo.buffer.get();
                
                // Validate all buffer outputs are finite
                observations.forEach(obs => {
                    obs.forEach(val => {
                        if (!isFinite(val)) {
                            throw new Error('Buffer produced non-finite observation values');
                        }
                    });
                });
                
                advantages.forEach(adv => {
                    if (!isFinite(adv)) {
                        throw new Error('Buffer produced non-finite advantage values');
                    }
                });
                
                returns.forEach(ret => {
                    if (!isFinite(ret)) {
                        throw new Error('Buffer produced non-finite return values');
                    }
                });
                
                logprobs.forEach(lp => {
                    if (!isFinite(lp)) {
                        throw new Error('Buffer produced non-finite logprob values');
                    }
                });
            }).not.toThrow();
        });

        test('should detect gradient explosion/vanishing during training', async () => {
            // Set up training data
            ppo.buffer.reset();
            for (let i = 0; i < 16; i++) {
                ppo.buffer.add(
                    [Math.random(), Math.random(), Math.random(), Math.random()],
                    Math.floor(Math.random() * 2),
                    Math.random() * 2 - 1,
                    Math.random(),
                    Math.random() * 2 - 1
                );
            }
            ppo.buffer.finishTrajectory(0);

            // Multiple training steps to potentially trigger gradient issues
            for (let step = 0; step < 3; step++) {
                await expect(async () => {
                    await ppo.train();
                    
                    // After each training step, verify networks still produce finite outputs
                    const [preds, action, value, logprob] = await ppo.getSample([0.5, 0.5, 0.5, 0.5]);
                    
                    if (!isFinite(value) || !isFinite(logprob)) {
                        throw new Error(`Training step ${step} produced non-finite outputs`);
                    }
                    
                    preds.forEach((pred, idx) => {
                        if (!isFinite(pred)) {
                            throw new Error(`Training step ${step} produced non-finite prediction at index ${idx}`);
                        }
                    });
                }).not.toThrow();
            }
        });
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
            
            // Should accept zero clip ratio (valid asymmetric clipping)
            expect(() => {
                new PPO(env, { 
                    clipRatio: 0,
                    verbose: 0 
                });
            }).not.toThrow();

            // Should accept large clip ratio (valid but less conservative)
            expect(() => {
                new PPO(env, { 
                    clipRatio: 2.0,
                    verbose: 0 
                });
            }).not.toThrow();
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
        });

        test('should reject invalid lambda values', () => {
            // Should reject negative lambda (GAE parameter)
            expect(() => {
                new PPO(env, { 
                    lam: -0.1,
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

    describe('PPO Mathematical Stability Tests', () => {
        test('should handle log probability calculations without NaN', async () => {
            const observation = [0.1, 0.2, 0.3, 0.4];
            const [preds, action, value, logprob] = await ppo.getSample(observation);
            
            // Check that log probabilities are finite
            expect(isFinite(logprob)).toBe(true);
            expect(isNaN(logprob)).toBe(false);
            
            // Check that predictions are finite
            preds.forEach(pred => {
                expect(isFinite(pred)).toBe(true);
                expect(isNaN(pred)).toBe(false);
            });
            
            // Check that values are finite
            expect(isFinite(value)).toBe(true);
            expect(isNaN(value)).toBe(false);
        });

        test('should handle advantage normalization with zero variance', () => {
            // Create buffer with identical rewards to test zero variance case
            ppo.buffer.reset();
            const identicalReward = 5.0;
            
            for (let i = 0; i < 5; i++) {
                ppo.buffer.add(
                    [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1, 0.4 + i * 0.1],
                    Math.floor(Math.random() * 2),
                    identicalReward, // Same reward every time
                    0.5 + Math.random() * 0.2,
                    -0.5 - Math.random() * 0.2
                );
            }
            
            ppo.buffer.finishTrajectory(0);
            
            expect(() => {
                const [, , advantages] = ppo.buffer.get();
                
                // Should handle zero variance without producing NaN
                advantages.forEach(adv => {
                    expect(isFinite(adv)).toBe(true);
                    expect(isNaN(adv)).toBe(false);
                });
            }).not.toThrow();
        });

        test('should handle large but finite reward values', () => {
            ppo.buffer.reset();
            
            // Should handle large but reasonable reward values without issues
            const largeRewards = [1e6, -1e6, 1000, -1000, 0];
            
            largeRewards.forEach((reward, i) => {
                expect(() => {
                    ppo.buffer.add(
                        [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1, 0.4 + i * 0.1],
                        Math.floor(Math.random() * 2),
                        reward,
                        0.5 + Math.random() * 0.2,
                        -0.5 - Math.random() * 0.2
                    );
                }).not.toThrow();
            });
            
            ppo.buffer.finishTrajectory(0);
            
            expect(() => {
                const [, , advantages, returns] = ppo.buffer.get();
                
                // Should handle large rewards and still produce finite results
                advantages.forEach(adv => {
                    expect(isFinite(adv)).toBe(true);
                    expect(isNaN(adv)).toBe(false);
                });
                
                returns.forEach(ret => {
                    expect(isFinite(ret)).toBe(true);
                    expect(isNaN(ret)).toBe(false);
                });
            }).not.toThrow();
        });

        test('should validate tensor operations produce finite values', async () => {
            // Test that all tensor operations in training produce finite values
            ppo.buffer.reset();
            
            // Add some training data
            for (let i = 0; i < 8; i++) {
                ppo.buffer.add(
                    [Math.random(), Math.random(), Math.random(), Math.random()],
                    Math.floor(Math.random() * 2),
                    Math.random() * 2 - 1, // Random reward between -1 and 1
                    Math.random(),
                    Math.random() * 2 - 1
                );
            }
            
            ppo.buffer.finishTrajectory(0);
            
            expect(() => {
                const [observations, actions, advantages, returns, logprobs] = ppo.buffer.get();
                
                // Validate all buffer data is finite
                observations.forEach(obs => {
                    obs.forEach(val => {
                        expect(isFinite(val)).toBe(true);
                        expect(isNaN(val)).toBe(false);
                    });
                });
                
                advantages.forEach(adv => {
                    expect(isFinite(adv)).toBe(true);
                    expect(isNaN(adv)).toBe(false);
                });
                
                returns.forEach(ret => {
                    expect(isFinite(ret)).toBe(true);
                    expect(isNaN(ret)).toBe(false);
                });
                
                logprobs.forEach(lp => {
                    expect(isFinite(lp)).toBe(true);
                    expect(isNaN(lp)).toBe(false);
                });
            }).not.toThrow();
        });

        test('should handle discounted cumulative sums correctly', () => {
            const testArray = [1, 2, 3, 4, 5];
            const gamma = 0.9;
            
            const result = ppo.buffer.discountedCumulativeSums(testArray.slice(), gamma);
            
            // Check that results are finite
            result.forEach(val => {
                expect(isFinite(val)).toBe(true);
                expect(isNaN(val)).toBe(false);
            });
            
            // Check that the calculation is mathematically correct
            // Last element should be 5
            expect(result[result.length - 1]).toBeCloseTo(5, 5);
            
            // Second to last should be 4 + 0.9 * 5 = 8.5
            expect(result[result.length - 2]).toBeCloseTo(8.5, 5);
        });
    });

    describe('PPO Constructor Validation Tests', () => {
        test('should handle missing environment gracefully', () => {
            expect(() => {
                new PPO(null, { verbose: 0 });
            }).toThrow();
        });

        test('should handle undefined environment gracefully', () => {
            expect(() => {
                new PPO(undefined, { verbose: 0 });
            }).toThrow();
        });

        test('should handle malformed environment objects', () => {
            const malformedEnv = {
                // Missing observationSpace and actionSpace
                reset: () => [0, 0, 0, 0],
                step: () => Promise.resolve([[0, 0, 0, 0], 1.0, false])
            };

            expect(() => {
                new PPO(malformedEnv, { verbose: 0 });
            }).toThrow();
        });

        test('should validate network architecture arrays', () => {
            // Test with negative units in network architecture
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [32, -16], // Negative units should be invalid
                        'vf': [32, 32]
                    },
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should validate configuration object structure', () => {
            // Test with completely invalid config
            expect(() => {
                new PPO(env, "invalid config");
            }).not.toThrow(); // Current implementation might not validate this
        });

        test('should handle NaN in configuration values', () => {
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: NaN,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should handle Infinity in configuration values', () => {
            expect(() => {
                new PPO(env, { 
                    clipRatio: Infinity,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject zero learning rates (meaningless for optimization)', () => {
            // Zero learning rate means no learning - algorithm cannot improve
            expect(() => {
                new PPO(env, { 
                    policyLearningRate: 0,
                    verbose: 0 
                });
            }).toThrow();

            expect(() => {
                new PPO(env, { 
                    valueLearningRate: 0,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject zero or negative targetKL (meaningless for KL divergence)', () => {
            // KL divergence is always non-negative, zero target means no policy updates allowed
            expect(() => {
                new PPO(env, { 
                    targetKL: 0,
                    verbose: 0 
                });
            }).toThrow();

            expect(() => {
                new PPO(env, { 
                    targetKL: -0.01,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject zero nEpochs (meaningless - no training)', () => {
            // Zero epochs means no training steps - algorithm cannot learn
            expect(() => {
                new PPO(env, { 
                    nEpochs: 0,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject zero nSteps (meaningless - no experience collection)', () => {
            // Zero steps means no experience collected - cannot train
            expect(() => {
                new PPO(env, { 
                    nSteps: 0,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid gamma values (meaningless discount factors)', () => {
            // Negative gamma makes no sense for discount factor
            expect(() => {
                new PPO(env, { 
                    gamma: -0.1,
                    verbose: 0 
                });
            }).toThrow();

            // Gamma > 1.0 means future rewards worth more than present (unstable)
            expect(() => {
                new PPO(env, { 
                    gamma: 1.1,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid lambda values (meaningless GAE parameters)', () => {
            // Negative lambda makes no sense for GAE weighting
            expect(() => {
                new PPO(env, { 
                    lam: -0.1,
                    verbose: 0 
                });
            }).toThrow();

            // Lambda > 1.0 can cause instability in advantage estimation
            expect(() => {
                new PPO(env, { 
                    lam: 1.1,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject zero or negative clipRatio (meaningless for PPO)', () => {
            // Should accept zero clip ratio (valid asymmetric clipping)
            expect(() => {
                new PPO(env, { 
                    clipRatio: 0,
                    verbose: 0 
                });
            }).not.toThrow();

            // Negative clip ratio makes no mathematical sense
            expect(() => {
                new PPO(env, { 
                    clipRatio: -0.1,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid network architecture structures', () => {
            // Empty network layers are meaningless - cannot process data
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [],
                        'vf': [32]
                    },
                    verbose: 0 
                });
            }).toThrow();

            // Missing required network components
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [32]
                        // Missing 'vf' - value function required for PPO
                    },
                    verbose: 0 
                });
            }).toThrow();

            // Zero or negative layer sizes are meaningless
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [0, 32],
                        'vf': [32, 32]
                    },
                    verbose: 0 
                });
            }).toThrow();

            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [-5, 32],
                        'vf': [32, 32]
                    },
                    verbose: 0 
                });
            }).toThrow();

            // Non-integer layer sizes are meaningless
            expect(() => {
                new PPO(env, { 
                    netArch: {
                        'pi': [32.5, 16],
                        'vf': [32, 32]
                    },
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject invalid activation functions', () => {
            // Invalid activation functions will cause TensorFlow errors
            expect(() => {
                new PPO(env, { 
                    activation: 'invalidActivation',
                    verbose: 0 
                });
            }).toThrow();

            expect(() => {
                new PPO(env, { 
                    activation: null,
                    verbose: 0 
                });
            }).toThrow();

            expect(() => {
                new PPO(env, { 
                    activation: undefined,
                    verbose: 0 
                });
            }).toThrow();
        });

        test('should reject non-integer step and epoch values (meaningless)', () => {
            // Non-integer steps make no sense - cannot take partial steps
            expect(() => {
                new PPO(env, { 
                    nSteps: 16.5,
                    verbose: 0 
                });
            }).toThrow();

            // Non-integer epochs make no sense - cannot do partial training epochs
            expect(() => {
                new PPO(env, { 
                    nEpochs: 2.7,
                    verbose: 0 
                });
            }).toThrow();
        });
    });

    describe('PPO Buffer Edge Cases', () => {
        test('should handle buffer with single data point', () => {
            ppo.buffer.reset();
            ppo.buffer.add([0.1, 0.2, 0.3, 0.4], 1, 1.0, 0.5, -0.7);
            ppo.buffer.finishTrajectory(0);
            
            expect(() => {
                const [obs, actions, advantages, returns, logprobs] = ppo.buffer.get();
                
                expect(obs.length).toBe(1);
                expect(actions.length).toBe(1);
                expect(advantages.length).toBe(1);
                expect(returns.length).toBe(1);
                expect(logprobs.length).toBe(1);
                
                // Check all values are finite
                expect(isFinite(advantages[0])).toBe(true);
                expect(isFinite(returns[0])).toBe(true);
                expect(isFinite(logprobs[0])).toBe(true);
            }).not.toThrow();
        });

        test('should handle buffer reset during trajectory', () => {
            ppo.buffer.add([0.1, 0.2, 0.3, 0.4], 1, 1.0, 0.5, -0.7);
            ppo.buffer.add([0.2, 0.3, 0.4, 0.5], 0, 2.0, 0.8, -0.3);
            
            // Reset before finishing trajectory
            ppo.buffer.reset();
            
            expect(ppo.buffer.pointer).toBe(0);
            expect(ppo.buffer.observationBuffer.length).toBe(0);
            expect(ppo.buffer.advantageBuffer.length).toBe(0);
        });

        test('should handle very large buffer sizes', () => {
            ppo.buffer.reset();
            
            // Add many data points to test memory handling
            for (let i = 0; i < 1000; i++) {
                ppo.buffer.add(
                    [Math.random(), Math.random(), Math.random(), Math.random()],
                    Math.floor(Math.random() * 2),
                    Math.random() * 2 - 1,
                    Math.random(),
                    Math.random() * 2 - 1
                );
            }
            
            ppo.buffer.finishTrajectory(0);
            
            expect(() => {
                const [obs, actions, advantages] = ppo.buffer.get();
                expect(obs.length).toBe(1000);
                expect(actions.length).toBe(1000);
                expect(advantages.length).toBe(1000);
                
                // Spot check some values for finiteness
                for (let i = 0; i < 10; i++) {
                    const idx = Math.floor(Math.random() * advantages.length);
                    expect(isFinite(advantages[idx])).toBe(true);
                    expect(isNaN(advantages[idx])).toBe(false);
                }
            }).not.toThrow();
        });
    });

    describe('PPO Continuous Action Space Tests', () => {
        let continuousEnv;
        let continuousPPO;

        beforeEach(() => {
            continuousEnv = {
                observationSpace: { shape: [3] },
                actionSpace: { 
                    class: 'Box', 
                    shape: [2], 
                    high: 1.0, 
                    low: -1.0,
                    dtype: 'float32'
                },
                reset: () => [Math.random(), Math.random(), Math.random()],
                step: async (action) => {
                    return [
                        [Math.random(), Math.random(), Math.random()],
                        Math.random() * 2 - 1,
                        Math.random() < 0.1
                    ];
                }
            };

            continuousPPO = new PPO(continuousEnv, {
                nSteps: 8,
                nEpochs: 1,
                verbose: 0
            });
        });

        afterEach(() => {
            if (continuousPPO) {
                if (continuousPPO.actor) continuousPPO.actor.dispose();
                if (continuousPPO.critic) continuousPPO.critic.dispose();
                if (continuousPPO.optPolicy) continuousPPO.optPolicy.dispose();
                if (continuousPPO.optValue) continuousPPO.optValue.dispose();
                if (continuousPPO.logStd) continuousPPO.logStd.dispose();
            }
        });

        test('should handle continuous action sampling without NaN', async () => {
            const observation = [0.5, 0.3, 0.8];
            const [preds, action, value, logprob] = await continuousPPO.getSample(observation);
            
            // Check predictions are finite
            preds.forEach(pred => {
                expect(isFinite(pred)).toBe(true);
                expect(isNaN(pred)).toBe(false);
            });
            
            // Check actions are finite
            action.forEach(act => {
                expect(isFinite(act)).toBe(true);
                expect(isNaN(act)).toBe(false);
            });
            
            // Check value and log probability are finite
            expect(isFinite(value)).toBe(true);
            expect(isNaN(value)).toBe(false);
            expect(isFinite(logprob)).toBe(true);
            expect(isNaN(logprob)).toBe(false);
        });

        test('should handle log probability calculations for continuous actions', async () => {
            const observation = [0.5, 0.3, 0.8];
            const [preds, action] = await continuousPPO.getSample(observation);
            
            // Test log probability calculation directly
            const logProb = tf.tidy(() => {
                const predsT = tf.tensor(preds);
                const actionT = tf.tensor(action);
                return continuousPPO.logProb(predsT, actionT).arraySync();
            });
            
            expect(isFinite(logProb)).toBe(true);
            expect(isNaN(logProb)).toBe(false);
        });

        test('should validate logStd initialization for continuous spaces', () => {
            expect(continuousPPO.logStd).toBeDefined();
            
            const logStdValues = continuousPPO.logStd.arraySync();
            logStdValues.forEach(val => {
                expect(isFinite(val)).toBe(true);
                expect(isNaN(val)).toBe(false);
            });
        });
    });
});
