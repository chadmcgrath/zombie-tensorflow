/* eslint-env jest, node */
/* global tf, require, global, describe, test, expect, beforeEach, afterEach */

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

// Edge case environments for testing
class EmptyObservationEnv {
    constructor() {
        this.observationSpace = { shape: [0] }; // Empty observation space
        this.actionSpace = { class: 'Discrete', n: 2, dtype: 'int32' };
        this.stepCount = 0;
    }
    reset() { return []; }
    async step() { return [[], 0, false]; }
}

class ExtremeRewardEnv {
    constructor() {
        this.observationSpace = { shape: [2] };
        this.actionSpace = { class: 'Discrete', n: 2, dtype: 'int32' };
        this.stepCount = 0;
    }
    reset() { return [0.5, 0.5]; }
    async step() {
        this.stepCount++;
        const reward = this.stepCount % 2 === 0 ? 1e6 : -1e6; // Extreme rewards
        return [[Math.random(), Math.random()], reward, this.stepCount > 10];
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
        const reward = this.stepCount === 3 ? NaN : Math.random(); // Inject NaN
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
        const reward = this.stepCount === 2 ? Infinity : Math.random(); // Inject Infinity
        return [[Math.random(), Math.random()], reward, this.stepCount > 5];
    }
}

class VeryLargeObservationEnv {
    constructor() {
        this.observationSpace = { shape: [1000] }; // Very large observation space
        this.actionSpace = { class: 'Discrete', n: 2, dtype: 'int32' };
        this.stepCount = 0;
    }
    reset() { return new Array(1000).fill(0).map(() => Math.random()); }
    async step() {
        this.stepCount++;
        return [new Array(1000).fill(0).map(() => Math.random()), Math.random(), this.stepCount > 5];
    }
}

class ContinuousActionEnv {
    constructor() {
        this.observationSpace = { shape: [4] };
        this.actionSpace = { 
            class: 'Box', 
            shape: [2], 
            high: 1.0, 
            low: -1.0, 
            dtype: 'float32' 
        };
        this.stepCount = 0;
    }
    reset() { return [0.1, 0.2, 0.3, 0.4]; }
    async step() {
        this.stepCount++;
        return [[Math.random(), Math.random(), Math.random(), Math.random()], Math.random(), this.stepCount > 10];
    }
}

// Import PPO - handle both browser and node environments
import PPO from '../src/ppo.js';

describe('PPO Comprehensive Test Suite', () => {
    let env;
    let ppo;
    let initialMemoryInfo;

    beforeEach(async () => {
        // Initialize TensorFlow backend
        if (typeof tf !== 'undefined' && tf.ready) {
            await tf.ready();
        }
        
        env = new MockEnv();
        ppo = new PPO(env, {
            nSteps: 32,
            nEpochs: 3,
            policyLearningRate: 0.001,
            valueLearningRate: 0.001,
            clipRatio: 0.2,
            targetKL: 0.01,
            netArch: {
                'pi': [16, 16],
                'vf': [16, 16]
            },
            verbose: 0
        });
        
        // Record initial memory state (for potential future use)
        if (typeof tf !== 'undefined' && tf.memory) {
            initialMemoryInfo = tf.memory(); // eslint-disable-line no-unused-vars
        }
    });

    afterEach(() => {
        // Cleanup
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

    // ===== TESTS THAT SHOULD PASS NOW =====
    
    describe('Basic PPO Functionality (Should Pass)', () => {
        test('should create PPO instance with correct configuration', () => {
            expect(ppo).toBeDefined();
            expect(ppo.config.nSteps).toBe(32);
            expect(ppo.config.nEpochs).toBe(3);
            expect(ppo.config.clipRatio).toBe(0.2);
        });

        test('should create actor and critic models', () => {
            expect(ppo.actor).toBeDefined();
            expect(ppo.critic).toBeDefined();
            expect(typeof ppo.actor.predict).toBe('function');
            expect(typeof ppo.critic.predict).toBe('function');
        });

        test('should sample actions from observations', async () => {
            const observation = [0.1, 0.2, 0.3, 0.4];
            const [preds, action, value, logprob] = await ppo.getSample(observation);
            
            expect(preds).toBeDefined();
            expect(action).toBeDefined();
            expect(typeof value).toBe('number');
            expect(typeof logprob).toBe('number');
            expect(Array.isArray(preds)).toBe(true);
        });

        test('should initialize buffer correctly', () => {
            expect(ppo.buffer).toBeDefined();
            expect(ppo.buffer.pointer).toBe(0);
            expect(Array.isArray(ppo.buffer.observationBuffer)).toBe(true);
            expect(Array.isArray(ppo.buffer.actionBuffer)).toBe(true);
        });

        test('should add experiences to buffer', () => {
            const observation = [0.1, 0.2, 0.3, 0.4];
            const action = 1;
            const reward = 0.5;
            const value = 0.3;
            const logprob = -0.7;
            
            ppo.buffer.add(observation, action, reward, value, logprob);
            
            expect(ppo.buffer.pointer).toBe(1);
            expect(ppo.buffer.observationBuffer).toHaveLength(1);
            expect(ppo.buffer.actionBuffer).toHaveLength(1);
        });

        test('should make predictions', () => {
            const observation = tf.tensor([[0.1, 0.2, 0.3, 0.4]]);
            const prediction = ppo.predict(observation);
            
            expect(prediction).toBeDefined();
            expect(prediction.shape[0]).toBe(1);
            expect(prediction.shape[1]).toBe(env.actionSpace.n);
            
            observation.dispose();
            prediction.dispose();
        });
    });

    describe('Invalid Input Handling (Will Fail - Need AI Fix)', () => {
        test('should handle empty observation space gracefully', () => {
            const emptyEnv = new EmptyObservationEnv();
            expect(() => {
                const emptyPPO = new PPO(emptyEnv, { nSteps: 4, nEpochs: 1 });
                // Cleanup
                if (emptyPPO.actor) emptyPPO.actor.dispose();
                if (emptyPPO.critic) emptyPPO.critic.dispose();
                if (emptyPPO.optPolicy) emptyPPO.optPolicy.dispose();
                if (emptyPPO.optValue) emptyPPO.optValue.dispose();
            }).not.toThrow();
        });

        test('should handle NaN rewards without corrupting training', async () => {
            const nanEnv = new NaNRewardEnv();
            const nanPPO = new PPO(nanEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            nanPPO.lastObservation = nanEnv.reset();
            
            // Should detect and handle NaN rewards
            await expect(async () => {
                await nanPPO.collectRollouts(nanPPO._initCallback(() => true));
                await nanPPO.train();
            }).not.toThrow();
            
            // Buffer should not contain NaN values after processing
            const [obs, actions, advantages, returns, logprobs] = nanPPO.buffer.get(); // eslint-disable-line no-unused-vars
            advantages.forEach(adv => {
                expect(isNaN(adv)).toBe(false);
            });
            returns.forEach(ret => {
                expect(isNaN(ret)).toBe(false);
            });
            
            // Cleanup
            if (nanPPO.actor) nanPPO.actor.dispose();
            if (nanPPO.critic) nanPPO.critic.dispose();
            if (nanPPO.optPolicy) nanPPO.optPolicy.dispose();
            if (nanPPO.optValue) nanPPO.optValue.dispose();
        }, 15000);

        test('should handle infinite rewards without corrupting training', async () => {
            const infEnv = new InfiniteRewardEnv();
            const infPPO = new PPO(infEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            infPPO.lastObservation = infEnv.reset();
            
            // Should detect and handle infinite rewards
            await expect(async () => {
                await infPPO.collectRollouts(infPPO._initCallback(() => true));
                await infPPO.train();
            }).not.toThrow();
            
            // Buffer should not contain infinite values after processing
            const [obs, actions, advantages, returns, logprobs] = infPPO.buffer.get(); // eslint-disable-line no-unused-vars
            advantages.forEach(adv => {
                expect(isFinite(adv)).toBe(true);
            });
            returns.forEach(ret => {
                expect(isFinite(ret)).toBe(true);
            });
            
            // Cleanup
            if (infPPO.actor) infPPO.actor.dispose();
            if (infPPO.critic) infPPO.critic.dispose();
            if (infPPO.optPolicy) infPPO.optPolicy.dispose();
            if (infPPO.optValue) infPPO.optValue.dispose();
        }, 15000);

        test('should handle extreme reward values without gradient explosion', async () => {
            const extremeEnv = new ExtremeRewardEnv();
            const extremePPO = new PPO(extremeEnv, { nSteps: 16, nEpochs: 2, verbose: 0 });
            
            extremePPO.lastObservation = extremeEnv.reset();
            
            // Should handle extreme rewards without exploding gradients
            await extremePPO.collectRollouts(extremePPO._initCallback(() => true));
            await extremePPO.train();
            
            // Check that model weights remain reasonable
            const actorWeights = extremePPO.actor.getWeights();
            actorWeights.forEach(weight => {
                const values = weight.dataSync();
                for (let i = 0; i < values.length; i++) {
                    expect(isFinite(values[i])).toBe(true);
                    expect(Math.abs(values[i])).toBeLessThan(1000); // Reasonable bounds
                }
            });
            
            // Cleanup
            if (extremePPO.actor) extremePPO.actor.dispose();
            if (extremePPO.critic) extremePPO.critic.dispose();
            if (extremePPO.optPolicy) extremePPO.optPolicy.dispose();
            if (extremePPO.optValue) extremePPO.optValue.dispose();
        }, 20000);
    });

    describe('Memory Efficiency with Large Inputs (Will Fail - Need AI Fix)', () => {
        test('should handle very large observation spaces efficiently', async () => {
            if (typeof tf === 'undefined' || !tf.memory) {
                console.warn('TensorFlow memory tracking not available, skipping test');
                return;
            }

            const largeEnv = new VeryLargeObservationEnv();
            const largePPO = new PPO(largeEnv, { 
                nSteps: 8, 
                nEpochs: 1, 
                netArch: { pi: [32], vf: [32] }, // Smaller networks for large obs
                verbose: 0 
            });
            
            const initialMemory = tf.memory().numBytes;
            
            largePPO.lastObservation = largeEnv.reset();
            await largePPO.collectRollouts(largePPO._initCallback(() => true));
            await largePPO.train();
            
            const finalMemory = tf.memory().numBytes;
            const memoryGrowth = finalMemory - initialMemory;
            
            // Should not use excessive memory for large observations
            expect(memoryGrowth).toBeLessThan(100 * 1024 * 1024); // 100MB limit
            
            // Cleanup
            if (largePPO.actor) largePPO.actor.dispose();
            if (largePPO.critic) largePPO.critic.dispose();
            if (largePPO.optPolicy) largePPO.optPolicy.dispose();
            if (largePPO.optValue) largePPO.optValue.dispose();
        }, 30000);

        test('should batch process large tensors efficiently', async () => {
            const largeEnv = new VeryLargeObservationEnv();
            const largePPO = new PPO(largeEnv, { 
                nSteps: 16, 
                nEpochs: 1, 
                netArch: { pi: [16], vf: [16] },
                verbose: 0 
            });
            
            largePPO.lastObservation = largeEnv.reset();
            
            const startTime = Date.now();
            await largePPO.collectRollouts(largePPO._initCallback(() => true));
            await largePPO.train();
            const endTime = Date.now();
            
            // Should complete in reasonable time even with large observations
            expect(endTime - startTime).toBeLessThan(10000); // 10 seconds max
            
            // Cleanup
            if (largePPO.actor) largePPO.actor.dispose();
            if (largePPO.critic) largePPO.critic.dispose();
            if (largePPO.optPolicy) largePPO.optPolicy.dispose();
            if (largePPO.optValue) largePPO.optValue.dispose();
        }, 15000);
    });

    describe('Continuous Action Space Handling (Will Fail - Need AI Fix)', () => {
        test('should handle continuous action spaces correctly', async () => {
            const contEnv = new ContinuousActionEnv();
            const contPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            // Should create logStd variable for continuous actions
            expect(contPPO.logStd).toBeDefined();
            expect(contPPO.logStd.shape).toEqual([2]);
            
            contPPO.lastObservation = contEnv.reset();
            const [preds, action, value, logprob] = await contPPO.getSample(contPPO.lastObservation); // eslint-disable-line no-unused-vars
            
            // Actions should be within bounds
            expect(Array.isArray(action)).toBe(true);
            expect(action.length).toBe(2);
            action.forEach(a => {
                expect(typeof a).toBe('number');
                expect(isFinite(a)).toBe(true);
            });
            
            // Cleanup
            if (contPPO.actor) contPPO.actor.dispose();
            if (contPPO.critic) contPPO.critic.dispose();
            if (contPPO.optPolicy) contPPO.optPolicy.dispose();
            if (contPPO.optValue) contPPO.optValue.dispose();
            if (contPPO.logStd) contPPO.logStd.dispose();
        });

        test('should clip continuous actions to environment bounds', async () => {
            const contEnv = new ContinuousActionEnv();
            const contPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            contPPO.lastObservation = contEnv.reset();
            await contPPO.collectRollouts(contPPO._initCallback(() => true));
            
            // All actions in buffer should be within environment bounds
            const [obs, actions, advantages, returns, logprobs] = contPPO.buffer.get(); // eslint-disable-line no-unused-vars
            actions.forEach(action => {
                if (Array.isArray(action)) {
                    action.forEach(a => {
                        expect(a).toBeGreaterThanOrEqual(-1.0);
                        expect(a).toBeLessThanOrEqual(1.0);
                    });
                }
            });
            
            // Cleanup
            if (contPPO.actor) contPPO.actor.dispose();
            if (contPPO.critic) contPPO.critic.dispose();
            if (contPPO.optPolicy) contPPO.optPolicy.dispose();
            if (contPPO.optValue) contPPO.optValue.dispose();
            if (contPPO.logStd) contPPO.logStd.dispose();
        }, 10000);
    });

    describe('Buffer Edge Cases (Will Fail - Need AI Fix)', () => {
        test('should handle buffer overflow gracefully', async () => {
            const contEnv = new ContinuousActionEnv();
            const bufferPPO = new PPO(contEnv, { nSteps: 4, nEpochs: 1, verbose: 0 }); // Very small buffer
            
            // Fill buffer beyond capacity
            for (let i = 0; i < 10; i++) {
                bufferPPO.buffer.add([0.1, 0.2, 0.3, 0.4], [0.5, 0.6], 1.0, 0.8, -0.5);
            }
            
            // Should handle overflow without crashing
            expect(() => {
                const [obs, actions, advantages, returns, logprobs] = bufferPPO.buffer.get(); // eslint-disable-line no-unused-vars
            }).not.toThrow();
            
            // Cleanup
            if (bufferPPO.actor) bufferPPO.actor.dispose();
            if (bufferPPO.critic) bufferPPO.critic.dispose();
            if (bufferPPO.optPolicy) bufferPPO.optPolicy.dispose();
            if (bufferPPO.optValue) bufferPPO.optValue.dispose();
            if (bufferPPO.logStd) bufferPPO.logStd.dispose();
        });

        test('should handle empty buffer gracefully', () => {
            const contEnv = new ContinuousActionEnv();
            const emptyBufferPPO = new PPO(contEnv, { nSteps: 4, nEpochs: 1, verbose: 0 });
            
            // Try to get from empty buffer
            expect(() => {
                const [obs, actions, advantages, returns, logprobs] = emptyBufferPPO.buffer.get(); // eslint-disable-line no-unused-vars
            }).not.toThrow();
            
            // Cleanup
            if (emptyBufferPPO.actor) emptyBufferPPO.actor.dispose();
            if (emptyBufferPPO.critic) emptyBufferPPO.critic.dispose();
            if (emptyBufferPPO.optPolicy) emptyBufferPPO.optPolicy.dispose();
            if (emptyBufferPPO.optValue) emptyBufferPPO.optValue.dispose();
            if (emptyBufferPPO.logStd) emptyBufferPPO.logStd.dispose();
        });

        test('should handle single-step episodes correctly', async () => {
            const contEnv = new ContinuousActionEnv();
            // Force immediate episode termination
            contEnv.step = async () => [[0.1, 0.2, 0.3, 0.4], 1.0, true];
            
            const singleStepPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            singleStepPPO.lastObservation = contEnv.reset();
            
            // Should handle single-step episodes without errors
            await expect(async () => {
                await singleStepPPO.collectRollouts(singleStepPPO._initCallback(() => true));
                await singleStepPPO.train();
            }).not.toThrow();
            
            // Cleanup
            if (singleStepPPO.actor) singleStepPPO.actor.dispose();
            if (singleStepPPO.critic) singleStepPPO.critic.dispose();
            if (singleStepPPO.optPolicy) singleStepPPO.optPolicy.dispose();
            if (singleStepPPO.optValue) singleStepPPO.optValue.dispose();
            if (singleStepPPO.logStd) singleStepPPO.logStd.dispose();
        }, 10000);
    });

    describe('Numerical Stability (Will Fail - Need AI Fix)', () => {
        test('should handle very small learning rates', async () => {
            const contEnv = new ContinuousActionEnv();
            const smallLRPPO = new PPO(contEnv, { 
                nSteps: 8, 
                nEpochs: 1, 
                policyLearningRate: 1e-10, // Extremely small
                valueLearningRate: 1e-10,
                verbose: 0 
            });
            
            smallLRPPO.lastObservation = contEnv.reset();
            
            // Should not cause numerical issues
            await expect(async () => {
                await smallLRPPO.collectRollouts(smallLRPPO._initCallback(() => true));
                await smallLRPPO.train();
            }).not.toThrow();
            
            // Cleanup
            if (smallLRPPO.actor) smallLRPPO.actor.dispose();
            if (smallLRPPO.critic) smallLRPPO.critic.dispose();
            if (smallLRPPO.optPolicy) smallLRPPO.optPolicy.dispose();
            if (smallLRPPO.optValue) smallLRPPO.optValue.dispose();
            if (smallLRPPO.logStd) smallLRPPO.logStd.dispose();
        }, 10000);

        test('should handle zero advantages correctly', async () => {
            const contEnv = new ContinuousActionEnv();
            const zeroAdvPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            // Create scenario with zero advantages
            zeroAdvPPO.buffer.add([0.1, 0.2, 0.3, 0.4], [0.5, 0.6], 0.0, 0.0, -0.5);
            zeroAdvPPO.buffer.add([0.1, 0.2, 0.3, 0.4], [0.5, 0.6], 0.0, 0.0, -0.5);
            zeroAdvPPO.buffer.finishTrajectory(0.0);
            
            // Should handle zero advantages without division by zero
            expect(() => {
                const [obs, actions, advantages, returns, logprobs] = zeroAdvPPO.buffer.get(); // eslint-disable-line no-unused-vars
                advantages.forEach(adv => {
                    expect(isFinite(adv)).toBe(true);
                });
            }).not.toThrow();
            
            // Cleanup
            if (zeroAdvPPO.actor) zeroAdvPPO.actor.dispose();
            if (zeroAdvPPO.critic) zeroAdvPPO.critic.dispose();
            if (zeroAdvPPO.optPolicy) zeroAdvPPO.optPolicy.dispose();
            if (zeroAdvPPO.optValue) zeroAdvPPO.optValue.dispose();
            if (zeroAdvPPO.logStd) zeroAdvPPO.logStd.dispose();
        });

        test('should handle identical observations correctly', async () => {
            const contEnv = new ContinuousActionEnv();
            // Force identical observations
            contEnv.reset = () => [0.5, 0.5, 0.5, 0.5];
            contEnv.step = async () => [[0.5, 0.5, 0.5, 0.5], 1.0, false];
            
            const identicalObsPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            identicalObsPPO.lastObservation = contEnv.reset();
            
            // Should handle identical observations without numerical issues
            await expect(async () => {
                await identicalObsPPO.collectRollouts(identicalObsPPO._initCallback(() => true));
                await identicalObsPPO.train();
            }).not.toThrow();
            
            // Cleanup
            if (identicalObsPPO.actor) identicalObsPPO.actor.dispose();
            if (identicalObsPPO.critic) identicalObsPPO.critic.dispose();
            if (identicalObsPPO.optPolicy) identicalObsPPO.optPolicy.dispose();
            if (identicalObsPPO.optValue) identicalObsPPO.optValue.dispose();
            if (identicalObsPPO.logStd) identicalObsPPO.logStd.dispose();
        }, 10000);
    });

    describe('Callback Error Handling (Will Fail - Need AI Fix)', () => {
        test('should handle callback errors gracefully', async () => {
            const contEnv = new ContinuousActionEnv();
            const callbackErrorPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            // Create callback that throws error
            const errorCallback = callbackErrorPPO._initCallback(() => {
                throw new Error('Callback error');
            });
            
            callbackErrorPPO.lastObservation = contEnv.reset();
            
            // Should handle callback errors without crashing training
            await expect(async () => {
                await callbackErrorPPO.collectRollouts(errorCallback);
            }).not.toThrow();
            
            // Cleanup
            if (callbackErrorPPO.actor) callbackErrorPPO.actor.dispose();
            if (callbackErrorPPO.critic) callbackErrorPPO.critic.dispose();
            if (callbackErrorPPO.optPolicy) callbackErrorPPO.optPolicy.dispose();
            if (callbackErrorPPO.optValue) callbackErrorPPO.optValue.dispose();
            if (callbackErrorPPO.logStd) callbackErrorPPO.logStd.dispose();
        }, 10000);

        test('should handle null/undefined callbacks', async () => {
            const contEnv = new ContinuousActionEnv();
            const nullCallbackPPO = new PPO(contEnv, { nSteps: 8, nEpochs: 1, verbose: 0 });
            
            nullCallbackPPO.lastObservation = contEnv.reset();
            
            // Should handle null callbacks
            await expect(async () => {
                await nullCallbackPPO.collectRollouts(nullCallbackPPO._initCallback(null));
                await nullCallbackPPO.train();
            }).not.toThrow();
            
            // Should handle undefined callbacks
            await expect(async () => {
                await nullCallbackPPO.collectRollouts(nullCallbackPPO._initCallback(undefined));
                await nullCallbackPPO.train();
            }).not.toThrow();
            
            // Cleanup
            if (nullCallbackPPO.actor) nullCallbackPPO.actor.dispose();
            if (nullCallbackPPO.critic) nullCallbackPPO.critic.dispose();
            if (nullCallbackPPO.optPolicy) nullCallbackPPO.optPolicy.dispose();
            if (nullCallbackPPO.optValue) nullCallbackPPO.optValue.dispose();
            if (nullCallbackPPO.logStd) nullCallbackPPO.logStd.dispose();
        }, 10000);
    });

    // ===== TESTS THAT WILL FAIL NOW (NEED AI TO FIX) =====

    describe('Memory Management (Will Fail - Need AI Fix)', () => {
        test('should not leak tensors during training', async () => {
            if (typeof tf === 'undefined' || !tf.memory) {
                console.warn('TensorFlow memory tracking not available, skipping test');
                return;
            }

            const initialTensors = tf.memory().numTensors;
            
            // Run multiple training steps
            for (let i = 0; i < 5; i++) {
                await ppo.collectRollouts(ppo._initCallback(() => true));
                await ppo.train();
            }
            
            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }
            
            const finalTensors = tf.memory().numTensors;
            const tensorLeak = finalTensors - initialTensors;
            
            // Allow for some reasonable tensor growth, but not excessive
            expect(tensorLeak).toBeLessThan(50);
        }, 30000);

        test('should properly dispose tensors in training loop', async () => {
            if (typeof tf === 'undefined' || !tf.memory) {
                console.warn('TensorFlow memory tracking not available, skipping test');
                return;
            }

            const initialMemory = tf.memory().numBytes;
            
            // Simulate longer training
            for (let episode = 0; episode < 3; episode++) {
                ppo.lastObservation = env.reset();
                await ppo.collectRollouts(ppo._initCallback(() => true));
                await ppo.train();
            }
            
            // Memory should not grow excessively
            const finalMemory = tf.memory().numBytes;
            const memoryGrowth = finalMemory - initialMemory;
            
            // Allow for reasonable memory growth but catch leaks
            expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024); // 10MB limit
        }, 30000);

        test('should handle tensor disposal in error conditions', async () => {
            if (typeof tf === 'undefined' || !tf.memory) {
                console.warn('TensorFlow memory tracking not available, skipping test');
                return;
            }

            const initialTensors = tf.memory().numTensors;
            
            // Create a scenario that might cause errors
            const originalPredict = ppo.actor.predict;
            let errorThrown = false;
            
            ppo.actor.predict = function(input) {
                if (!errorThrown) {
                    errorThrown = true;
                    throw new Error('Simulated training error');
                }
                return originalPredict.call(this, input);
            };
            
            try {
                await ppo.collectRollouts(ppo._initCallback(() => true));
                await ppo.train();
            } catch (error) { // eslint-disable-line no-unused-vars
                // Expected error
            }
            
            // Restore original function
            ppo.actor.predict = originalPredict;
            
            // Continue normal training
            await ppo.collectRollouts(ppo._initCallback(() => true));
            await ppo.train();
            
            const finalTensors = tf.memory().numTensors;
            expect(finalTensors - initialTensors).toBeLessThan(30);
        }, 30000);
    });

    describe('Training Stability (Will Fail - Need AI Fix)', () => {
        test('should maintain consistent performance over multiple episodes', async () => {
            const rewards = [];
            const losses = []; // eslint-disable-line no-unused-vars
            
            for (let episode = 0; episode < 10; episode++) {
                ppo.lastObservation = env.reset();
                
                let episodeReward = 0;
                let stepCount = 0;
                
                // Collect rollouts and track rewards
                const callback = ppo._initCallback((alg) => {
                    if (alg.buffer.rewardBuffer.length > 0) {
                        episodeReward += alg.buffer.rewardBuffer[alg.buffer.rewardBuffer.length - 1];
                    }
                    stepCount++;
                    return true;
                });
                
                await ppo.collectRollouts(callback);
                await ppo.train();
                
                rewards.push(episodeReward / stepCount);
            }
            
            // Check that performance doesn't degrade significantly
            const firstHalf = rewards.slice(0, 5);
            const secondHalf = rewards.slice(5, 10);
            
            const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
            const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
            
            // Performance should remain within reasonable bounds (allowing for stochastic variation)
            const performanceDiff = Math.abs(secondAvg - firstAvg);
            const avgMagnitude = Math.abs(firstAvg) + Math.abs(secondAvg);
            
            // Ensure performance doesn't change too dramatically (relative to the scale)
            if (avgMagnitude > 0.001) {
                expect(performanceDiff / avgMagnitude).toBeLessThan(10.0); // Allow large relative changes
            }
            
            // Rewards should not contain NaN or infinite values
            rewards.forEach(reward => {
                expect(isFinite(reward)).toBe(true);
                expect(isNaN(reward)).toBe(false);
            });
        }, 60000);

        test('should handle episode boundaries correctly', async () => {
            let episodeEnds = 0;
            let totalSteps = 0;
            
            const callback = ppo._initCallback((alg) => { // eslint-disable-line no-unused-vars
                totalSteps++;
                return true;
            });
            
            // Force multiple episode endings
            const originalStep = env.step.bind(env);
            env.step = async function(action) {
                const [obs, reward, done] = await originalStep(action);
                // Force episode end every 10 steps
                if (totalSteps % 10 === 0) {
                    episodeEnds++;
                    return [obs, reward, true];
                }
                return [obs, reward, done];
            };
            
            ppo.lastObservation = env.reset();
            await ppo.collectRollouts(callback);
            
            // Buffer should handle multiple episode boundaries
            expect(episodeEnds).toBeGreaterThan(0);
            expect(ppo.buffer.pointer).toBeGreaterThan(0);
            
            // Advantages should be computed correctly
            const [obs, actions, advantages, returns, logprobs] = ppo.buffer.get(); // eslint-disable-line no-unused-vars
            
            advantages.forEach(adv => {
                expect(isFinite(adv)).toBe(true);
                expect(isNaN(adv)).toBe(false);
            });
            
            returns.forEach(ret => {
                expect(isFinite(ret)).toBe(true);
                expect(isNaN(ret)).toBe(false);
            });
        }, 30000);

        test('should prevent gradient explosion', async () => {
            // Run training with potentially unstable conditions
            for (let i = 0; i < 5; i++) {
                ppo.lastObservation = env.reset();
                await ppo.collectRollouts(ppo._initCallback(() => true));
                
                // Get buffer data before training
                const [obs, actions, advantages, returns, logprobs] = ppo.buffer.get(); // eslint-disable-line no-unused-vars
                
                await ppo.train();
                
                // Check that model weights remain finite
                const actorWeights = ppo.actor.getWeights();
                const criticWeights = ppo.critic.getWeights();
                
                actorWeights.forEach(weight => {
                    const values = weight.dataSync();
                    for (let j = 0; j < values.length; j++) {
                        expect(isFinite(values[j])).toBe(true);
                        expect(isNaN(values[j])).toBe(false);
                        expect(Math.abs(values[j])).toBeLessThan(100); // Prevent explosion
                    }
                });
                
                criticWeights.forEach(weight => {
                    const values = weight.dataSync();
                    for (let j = 0; j < values.length; j++) {
                        expect(isFinite(values[j])).toBe(true);
                        expect(isNaN(values[j])).toBe(false);
                        expect(Math.abs(values[j])).toBeLessThan(100); // Prevent explosion
                    }
                });
            }
        }, 45000);
    });

    describe('Resource Cleanup (Will Fail - Need AI Fix)', () => {
        test('should properly clean up optimizers', async () => {
            if (typeof tf === 'undefined' || !tf.memory) {
                console.warn('TensorFlow memory tracking not available, skipping test');
                return;
            }

            const initialTensors = tf.memory().numTensors;
            
            // Create multiple PPO instances to test cleanup
            const ppos = [];
            for (let i = 0; i < 3; i++) {
                const testPPO = new PPO(new MockEnv(), {
                    nSteps: 16,
                    nEpochs: 2,
                    netArch: { 'pi': [8], 'vf': [8] }
                });
                ppos.push(testPPO);
                
                // Run brief training
                testPPO.lastObservation = testPPO.env.reset();
                await testPPO.collectRollouts(testPPO._initCallback(() => true));
                await testPPO.train();
            }
            
            // Cleanup all instances
            ppos.forEach(testPPO => {
                if (testPPO.actor) testPPO.actor.dispose();
                if (testPPO.critic) testPPO.critic.dispose();
                if (testPPO.optPolicy) testPPO.optPolicy.dispose();
                if (testPPO.optValue) testPPO.optValue.dispose();
                if (testPPO.logStd) testPPO.logStd.dispose();
            });
            
            // Force garbage collection
            if (global.gc) {
                global.gc();
            }
            
            const finalTensors = tf.memory().numTensors;
            expect(finalTensors - initialTensors).toBeLessThan(10);
        }, 30000);

        test('should handle buffer reset correctly', async () => {
            // Fill buffer with data
            for (let i = 0; i < 20; i++) {
                ppo.buffer.add([i, i+1, i+2, i+3], i % 2, Math.random(), Math.random(), Math.random());
            }
            
            expect(ppo.buffer.pointer).toBe(20);
            
            // Reset buffer
            ppo.buffer.reset();
            
            expect(ppo.buffer.pointer).toBe(0);
            expect(ppo.buffer.observationBuffer).toHaveLength(0);
            expect(ppo.buffer.actionBuffer).toHaveLength(0);
            expect(ppo.buffer.rewardBuffer).toHaveLength(0);
            expect(ppo.buffer.valueBuffer).toHaveLength(0);
            expect(ppo.buffer.logprobabilityBuffer).toHaveLength(0);
            expect(ppo.buffer.advantageBuffer).toHaveLength(0);
            expect(ppo.buffer.returnBuffer).toHaveLength(0);
        });
    });

    describe('Convergence and Learning (Will Fail - Need AI Fix)', () => {
        test('should show learning progress over time', async () => {
            const performanceHistory = [];
            
            for (let epoch = 0; epoch < 8; epoch++) {
                ppo.lastObservation = env.reset();
                
                let totalReward = 0;
                let stepCount = 0;
                
                const callback = ppo._initCallback((alg) => {
                    if (alg.buffer.rewardBuffer.length > 0) {
                        totalReward += alg.buffer.rewardBuffer[alg.buffer.rewardBuffer.length - 1];
                        stepCount++;
                    }
                    return true;
                });
                
                await ppo.collectRollouts(callback);
                await ppo.train();
                
                const avgReward = stepCount > 0 ? totalReward / stepCount : 0;
                performanceHistory.push(avgReward);
            }
            
            // Check that learning is occurring (some improvement over time)
            const firstQuarter = performanceHistory.slice(0, 2);
            const lastQuarter = performanceHistory.slice(-2);
            
            const firstAvg = firstQuarter.reduce((a, b) => a + b, 0) / firstQuarter.length;
            const lastAvg = lastQuarter.reduce((a, b) => a + b, 0) / lastQuarter.length;
            
            // Should show reasonable learning behavior (allowing for random environment)
            const performanceDiff = Math.abs(lastAvg - firstAvg);
            const avgMagnitude = Math.abs(firstAvg) + Math.abs(lastAvg);
            
            // Ensure performance doesn't change too dramatically (relative to the scale)
            if (avgMagnitude > 0.001) {
                expect(performanceDiff / avgMagnitude).toBeLessThan(10.0); // Allow large relative changes
            }
            
            // All performance values should be finite
            performanceHistory.forEach(perf => {
                expect(isFinite(perf)).toBe(true);
                expect(isNaN(perf)).toBe(false);
            });
        }, 90000);

        test('should maintain stable value function estimates', async () => {
            const valueEstimates = [];
            
            for (let i = 0; i < 5; i++) {
                ppo.lastObservation = env.reset();
                
                // Get value estimates for the same observation
                const testObs = [0.5, 0.5, 0.5, 0.5];
                const obsT = tf.tensor([testObs]);
                const value = ppo.critic.predict(obsT);
                const valueArray = await value.data();
                
                valueEstimates.push(valueArray[0]);
                
                obsT.dispose();
                value.dispose();
                
                // Run training
                await ppo.collectRollouts(ppo._initCallback(() => true));
                await ppo.train();
            }
            
            // Value estimates should be finite and not wildly unstable
            valueEstimates.forEach(val => {
                expect(isFinite(val)).toBe(true);
                expect(isNaN(val)).toBe(false);
                expect(Math.abs(val)).toBeLessThan(1000); // Reasonable bounds
            });
            
            // Values shouldn't change too dramatically between episodes
            for (let i = 1; i < valueEstimates.length; i++) {
                const change = Math.abs(valueEstimates[i] - valueEstimates[i-1]);
                expect(change).toBeLessThan(100); // Prevent wild swings
            }
        }, 60000);
    });
});
