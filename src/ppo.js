// Import TensorFlow.js for browser environment
import * as tf from '@tensorflow/tfjs';

function log () {
    console.log('[PPO]', ...arguments)
}

export class BaseCallback {
    constructor() {
        this.nCalls = 0
    }

    _onStep(alg) { return true }
    onStep(alg) {
        this.nCalls += 1
        return this._onStep(alg)
    }

    _onTrainingStart(alg) {}
    onTrainingStart(alg) {
        this._onTrainingStart(alg)
    }

    _onTrainingEnd(alg) {}
    onTrainingEnd(alg) {
        this._onTrainingEnd(alg)
    }

    _onRolloutStart(alg) {}
    onRolloutStart(alg) {
        this._onRolloutStart(alg)
    }

    _onRolloutEnd(alg) {}
    onRolloutEnd(alg) {
        this._onRolloutEnd(alg)
    }
}

export class FunctionalCallback extends BaseCallback {
    constructor(callback) {
        super()
        this.callback = callback
    }

    _onStep(alg) {
        if (this.callback) {
            return this.callback(alg)
        }
        return true
    }
}

export class DictCallback extends BaseCallback {
    constructor(callback) {
        super()
        this.callback = callback
    }

    _onStep(alg) {
        if (this.callback && this.callback.onStep) {
            return this.callback.onStep(alg)
        }
        return true
    }
    
    _onTrainingStart(alg) {
        if (this.callback && this.callback.onTrainingStart) {
            this.callback.onTrainingStart(alg)
        }
    }

    _onTrainingEnd(alg) {
        if (this.callback && this.callback.onTrainingEnd) {
            this.callback.onTrainingEnd(alg)
        }
    }

    _onRolloutStart(alg) {
        if (this.callback && this.callback.onRolloutStart) {
            this.callback.onRolloutStart(alg)
        }
    }

    _onRolloutEnd(alg) {
        if (this.callback && this.callback.onRolloutEnd) {
            this.callback.onRolloutEnd(alg)
        }
    }
}

export class Buffer {
    constructor(bufferConfig) {
        const bufferConfigDefault = {
            gamma: 0.99,
            lam: 0.95
        }
        this.bufferConfig = Object.assign({}, bufferConfigDefault, bufferConfig)
        this.gamma = this.bufferConfig.gamma
        this.lam = this.bufferConfig.lam
        this.reset()
    }

    add(observation, action, reward, value, logprobability) {
        // Validate and sanitize inputs
        const sanitizedReward = this._sanitizeValue(reward);
        const sanitizedValue = this._sanitizeValue(value);
        const sanitizedLogprob = this._sanitizeValue(logprobability);
        
        this.observationBuffer.push(observation.slice(0))
        this.actionBuffer.push(action)
        this.rewardBuffer.push(sanitizedReward)
        this.valueBuffer.push(sanitizedValue)
        this.logprobabilityBuffer.push(sanitizedLogprob)
        this.pointer += 1
    }

    _sanitizeValue(value) {
        // Handle NaN, Infinity, and extreme values
        if (!isFinite(value) || isNaN(value)) {
            return 0.0;
        }
        // Clamp extreme values
        return Math.max(-1000, Math.min(1000, value));
    }

    discountedCumulativeSums (arr, coeff) {
        let res = []
        let s = 0
        // Create a copy to avoid mutating the original array
        const arrCopy = [...arr];
        arrCopy.reverse().forEach(v => {
            s = v + s * coeff
            res.push(s)
        })
        return res.reverse()
    }

    finishTrajectory(lastValue) {
        if (this.pointer === this.trajectoryStartIndex) {
            // Handle empty trajectory
            this.trajectoryStartIndex = this.pointer;
            return;
        }

        const sanitizedLastValue = this._sanitizeValue(lastValue);
        
        const rewards = this.rewardBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(sanitizedLastValue * this.gamma)
        const values = this.valueBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(sanitizedLastValue)
        const deltas = rewards
            .slice(0, -1)
            .map((reward, ri) => reward - (values[ri] - this.gamma * values[ri + 1]))
        this.advantageBuffer = this.advantageBuffer
            .concat(this.discountedCumulativeSums(deltas, this.gamma * this.lam))
        this.returnBuffer = this.returnBuffer
            .concat(this.discountedCumulativeSums(rewards, this.gamma).slice(0, -1))
        
        this.trajectoryStartIndex = this.pointer;
    }

    get() {
        if (this.advantageBuffer.length === 0) {
            // Handle empty buffer
            return [
                this.observationBuffer,
                this.actionBuffer,
                [],
                [],
                this.logprobabilityBuffer,
                this.pointer,
            ];
        }

        const [advantageMean, advantageStd] = tf.tidy(() => {
            const advTensor = tf.tensor(this.advantageBuffer);
            const mean = tf.mean(advTensor).arraySync();
            const std = tf.moments(advTensor).variance.sqrt().arraySync();
            return [mean, Math.max(std, 1e-8)]; // Prevent division by zero
        });
        
        this.advantageBuffer = this.advantageBuffer
            .map(advantage => {
                const normalized = (advantage - advantageMean) / advantageStd;
                return this._sanitizeValue(normalized);
            });
        
        return [
            this.observationBuffer,
            this.actionBuffer,
            this.advantageBuffer,
            this.returnBuffer,
            this.logprobabilityBuffer,
            this.pointer,
        ]
    }

    reset() {
        this.observationBuffer = []
        this.actionBuffer = []
        this.advantageBuffer = []
        this.rewardBuffer = []
        this.returnBuffer = []
        this.valueBuffer = []
        this.logprobabilityBuffer = []
        this.trajectoryStartIndex = 0
        this.pointer = 0
    }

}

export class PPO {
    constructor(env, config) {
        const configDefault = {
            nSteps: 512,
            nEpochs: 10,
            policyLearningRate: 1e-3,
            valueLearningRate: 1e-3,
            clipRatio: 0.2,
            targetKL: 0.01,
            useSDE: false, // TODO: State Dependent Exploration (gSDE)
            netArch: {
                'pi': [32, 32],
                'vf': [32, 32]
            },
            activation: 'relu',
            verbose: 0
        }
        this.config = Object.assign({}, configDefault, config)

        // Prepare network architecture
        if (Array.isArray(this.config.netArch)) {
            this.config.netArch = {
                'pi': this.config.netArch,
                'vf': this.config.netArch
            }
        }

        // Initialize logger
        this.log = (...args) => {
            if (this.config.verbose > 0) {
                console.log('[PPO]', ...args)
            }
        }

        // Initialize environment
        this.env = env
        if ((this.env.actionSpace.class == 'Discrete') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'int32'
        } else if ((this.env.actionSpace.class == 'Box') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'float32'
        }

        // Initialize counters
        this.numTimesteps = 0
        this.lastObservation = null

        // Initialize buffer
        this.buffer = new Buffer(config)

        // Initialize models for actor and critic
        this.actor = this.createActor()
        this.critic = this.createCritic()

        // Initialize logStd (for continuous action space)
        if (this.env.actionSpace.class == 'Box') {
            this.logStd = tf.variable(tf.zeros([this.env.actionSpace.shape[0]]), true, 'logStd')
        }

        // Initialize optimizers
        this.optPolicy = tf.train.adam(this.config.policyLearningRate)
        this.optValue = tf.train.adam(this.config.valueLearningRate)
    }

    createActor() {
        const input = tf.layers.input({shape: this.env.observationSpace.shape})
        let l = input
        this.config.netArch.pi.forEach((units, i) => {
            l = tf.layers.dense({
                units, 
                activation: this.config.activation
            }).apply(l)
        })
        if (this.env.actionSpace.class == 'Discrete') {
            l = tf.layers.dense({
                units: this.env.actionSpace.n, 
                // kernelInitializer: 'glorotNormal'
            }).apply(l)
        } else if (this.env.actionSpace.class == 'Box') {
            l = tf.layers.dense({
                units: this.env.actionSpace.shape[0], 
                //activation: 'tanh',
                // kernelInitializer: 'glorotNormal'
            }).apply(l)
        } else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class)
        }
        return tf.model({inputs: input, outputs: l})
    }

    createCritic() {
        // Initialize critic
        const input = tf.layers.input({shape: this.env.observationSpace.shape})
        let l = input
        this.config.netArch.vf.forEach((units, i) => {
            l = tf.layers.dense({
                units: units, 
                activation: this.config.activation
            }).apply(l)
        })
        l = tf.layers.dense({
            units: 1, 
            activation: 'linear'
        }).apply(l)
        return tf.model({inputs: input, outputs: l})
    }

    sampleAction(observationT) {
        return tf.tidy(() => {
            const preds = tf.squeeze(this.actor.predict(observationT), 0)
            let action 
            if (this.env.actionSpace.class == 'Discrete') {
                action = tf.squeeze(tf.multinomial(preds, 1), 0) // > []
            } else if (this.env.actionSpace.class == 'Box') {
                action = tf.add(
                    tf.mul(
                        tf.randomStandardNormal([this.env.actionSpace.shape[0]]), 
                        tf.exp(this.logStd)
                    ),
                    preds
                ) // > [actionSpace.shape[0]]
            }
            return [preds, action]
        })
    }

    logProbCategorical(logits, x) {
        return tf.tidy(() => {
            const numActions = logits.shape[logits.shape.length - 1]
            const logprobabilitiesAll = tf.logSoftmax(logits)
            return tf.sum(
                tf.mul(tf.oneHot(x, numActions), logprobabilitiesAll),
                logprobabilitiesAll.shape.length - 1
            )
        })
    }
    
    logProbNormal(loc, scale, x) {
        return tf.tidy(() => {
            const logUnnormalized = tf.mul(
                -0.5,
                tf.square(
                    tf.sub(
                        tf.div(x, scale),
                        tf.div(loc, scale)
                    )
                )
            )
            const logNormalization = tf.add(
                tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
                tf.log(scale)
            )
            return tf.sum(
                tf.sub(logUnnormalized, logNormalization),
                logUnnormalized.shape.length - 1
            )
        })
    }

    logProb(preds, actions) {
        // Preds can be logits or means
        if (this.env.actionSpace.class == 'Discrete') {
            return this.logProbCategorical(preds, actions)
        } else if (this.env.actionSpace.class == 'Box') {
            return this.logProbNormal(preds, tf.exp(this.logStd), actions)
        }
    }
    
    predict(observation, deterministic=false) {
        return this.actor.predict(observation)
    }

    trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT) {
        const optFunc = () => {
            const predsT = this.actor.predict(observationBufferT) // -> Logits or means
            const diffT = tf.sub(
                this.logProb(predsT, actionBufferT),
                logprobabilityBufferT
            )
            const ratioT = tf.exp(tf.clipByValue(diffT, -10, 10)) // Prevent extreme ratios
            const minAdvantageT = tf.where(
                tf.greater(advantageBufferT, 0),
                tf.mul(tf.add(1, this.config.clipRatio), advantageBufferT),
                tf.mul(tf.sub(1, this.config.clipRatio), advantageBufferT)
            )
            const policyLoss = tf.neg(tf.mean(
                tf.minimum(tf.mul(ratioT, advantageBufferT), minAdvantageT)
            ))
            return policyLoss
        }
    
        return tf.tidy(() => {
            try {
                const {values, grads} = this.optPolicy.computeGradients(optFunc)
                
                // Check if gradients are valid
                const hasValidGrads = grads && Object.keys(grads).length > 0;
                if (hasValidGrads) {
                    this.optPolicy.applyGradients(grads)
                }
                
                const kl = tf.mean(tf.sub(
                    logprobabilityBufferT,
                    this.logProb(this.actor.predict(observationBufferT), actionBufferT)
                ))
                return Math.abs(kl.arraySync()) // Return absolute KL for stability
            } catch (error) {
                console.warn('Policy training step failed:', error.message);
                return 0; // Return 0 KL on error
            }
        })
    }

    trainValue(observationBufferT, returnBufferT) {
        const optFunc = () => {
            const valuesPredT = this.critic.predict(observationBufferT)
            return tf.losses.meanSquaredError(returnBufferT, valuesPredT)
        }
                
        tf.tidy(() => {
            const {values, grads} = this.optValue.computeGradients(optFunc)
            this.optValue.applyGradients(grads)
        })
    }

    _initCallback(callback) {
        // Handle null/undefined callbacks
        if (!callback) {
            return new BaseCallback();
        }
        
        // Function, not class
        if (typeof callback === 'function') {
            // Check if it's an arrow function or regular function without prototype
            if (!callback.prototype || callback.prototype.constructor === undefined) {
                return new FunctionalCallback(callback)
            }
            return callback
        }
        
        // Handle object callbacks
        if (typeof callback === 'object') {
            return new DictCallback(callback)
        }
        
        return new BaseCallback() 
    }
    async getSample(lastObservation)
    {        
        return tf.tidy(() => {
            const lastObservationT = tf.tensor([lastObservation])
            const [predsT, actionT] = this.sampleAction(lastObservationT)
            const valueT = this.critic.predict(lastObservationT)
            const logprobabilityT = this.logProb(predsT, actionT)
            return [
                predsT.arraySync(), // -> Discrete: [actionSpace.n] or Box: [actionSpace.shape[0]]
                actionT.arraySync(), // -> Discrete: [] or Box: [actionSpace.shape[0]]
                valueT.arraySync()[0][0],
                logprobabilityT.arraySync()
            ]
        })
    }
    async collectRollouts(callback) {
        try {
            if (this.lastObservation === null) {
                this.lastObservation = this.env.reset()
            }

            this.buffer.reset()
            callback.onRolloutStart(this)

            let sumReturn = 0
            let sumLength = 0
            let numEpisodes = 0

            const allPreds = []
            const allActions = []
            const allClippedActions = []

            while(this.buffer.pointer < this.config.nSteps){
                try {
                    // Predict action, value and logprob from last observation
                    const [preds, action, value, logprobability] = await this.getSample(this.lastObservation); 
                    allPreds.push(preds)
                    allActions.push(action)

                    // Rescale for continuous action space
                    let clippedAction = action
                    if (this.env.actionSpace.class == 'Box') {
                        let h = this.env.actionSpace.high
                        let l = this.env.actionSpace.low
                        if (typeof h === 'number' && typeof l === 'number') {
                            clippedAction = Array.isArray(action) ? action.map(a => {
                                return Math.min(Math.max(a, l), h)
                            }) : Math.min(Math.max(action, l), h)
                        }
                    }
                    allClippedActions.push(clippedAction)

                    // Take action in environment
                    const [newObservation, reward, done] = await this.env.step(clippedAction)
                    sumReturn += reward
                    sumLength += 1

                    // Update global timestep counter
                    this.numTimesteps += 1 

                    // Handle callback errors gracefully
                    try {
                        callback.onStep(this)
                    } catch (callbackError) {
                        console.warn('Callback error during step:', callbackError.message);
                        // Continue execution despite callback error
                    }

                    this.buffer.add(
                        this.lastObservation, 
                        clippedAction, // Store clipped action in buffer
                        reward, 
                        value, 
                        logprobability
                    )
                    
                    this.lastObservation = newObservation
                    
                    if (done || this.buffer.pointer > this.config.nSteps - 1) {
                        const lastValue = done 
                            ? 0 
                            : tf.tidy(() => this.critic.predict(tf.tensor([newObservation])).arraySync())[0][0]
                        this.buffer.finishTrajectory(lastValue)
                        numEpisodes += 1
                        break;
                        //this.lastObservation = this.env.reset()
                    }
                } catch (stepError) {
                    console.warn('Error during rollout step:', stepError.message);
                    // Break out of the loop on step error
                    break;
                }
            }           
            
            try {
                callback.onRolloutEnd(this)
            } catch (callbackError) {
                console.warn('Callback error during rollout end:', callbackError.message);
            }
        } catch (error) {
            console.warn('Error during collectRollouts:', error.message);
            throw error;
        }
    }

    async train(config) {
        try {
            // Get values from the buffer
            const [
                observationBuffer,
                actionBuffer,
                advantageBuffer,
                returnBuffer,
                logprobabilityBuffer,
            ] = this.buffer.get()

            // Skip training if no data
            if (observationBuffer.length === 0 || advantageBuffer.length === 0) {
                return;
            }

            const [
                observationBufferT,
                actionBufferT,
                advantageBufferT,
                returnBufferT,
                logprobabilityBufferT
            ] = tf.tidy(() => [
                tf.tensor(observationBuffer),
                tf.tensor(actionBuffer, null, this.env.actionSpace.dtype),
                tf.tensor(advantageBuffer),
                tf.tensor(returnBuffer).reshape([-1, 1]),
                tf.tensor(logprobabilityBuffer)
            ])       
            
            try {
                for (let i = 0; i < this.config.nEpochs; i++) {
                    try {
                        const kl = this.trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT)
                        if (kl > 1.5 * this.config.targetKL) {
                            break
                        }
                    } catch (policyError) {
                        console.warn('Policy training error:', policyError.message);
                        break; // Stop policy training on error
                    }
                }

                for (let i = 0;  i < this.config.nEpochs; i++) {
                    try {
                        this.trainValue(observationBufferT, returnBufferT)
                    } catch (valueError) {
                        console.warn('Value training error:', valueError.message);
                        break; // Stop value training on error
                    }
                }
            } finally {
                // Always dispose tensors, even if training fails
                tf.dispose([
                    observationBufferT, 
                    actionBufferT,
                    advantageBufferT,
                    returnBufferT,
                    logprobabilityBufferT
                ])
            }
        } catch (error) {
            console.warn('Training error:', error.message);
            // Don't rethrow to allow continued execution
        }
    }

    async learn(learnConfig) {
        const learnConfigDefault = {
            'totalTimesteps': 1000,
            'logInterval': 1,
            'callback': null
        }
        let { 
            totalTimesteps,
            logInterval,
            callback
        } = Object.assign({}, learnConfigDefault, learnConfig)

        callback = this._initCallback(callback)
        
        let iteration = 0
        
        callback.onTrainingStart(this)

        while (this.numTimesteps < totalTimesteps) {
            await this.collectRollouts(callback)
            iteration += 1
            if (logInterval && iteration % logInterval === 0) {
                log(`Timesteps: ${this.numTimesteps}`)
            }
            this.train()
        }
        
        callback.onTrainingEnd(this)
    }
}

export default PPO;
