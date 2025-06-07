/* global $ */
/* global Chart */
/* global PPO */
/* global pi2, randomAngle, fixAngle, Vec, stuff_collide, rewardConfigs, createGrid */
/* global Eye, drawRotatedImage, requestAnimationFrameAsync, ModelPersistence, GameState */
/* global tf */

// TensorFlow configuration
tf.setBackend('cpu');

// Game Constants
const GAME_CONSTANTS = {
    NUM_EYES: 30,
    NUM_ACTIONS: 11,
    EYE_MAX_RANGE: 1000,
    MIN_HUMANS: 2,
    MIN_ZOMBIES: 3,
    MAX_ZOMBIE_SPAWNS: 40,
    ZOMBIE_GREEN: "#2f402f",
    HUMAN_SPEED: 3,
    BATCH_SIZE: 512,
    LEARNING_RATE: 0.001
};

// Game State
const gameState = {
    zCnt: 0,
    hCnt: 0,
    agents: [],
    blocks: [],
    canvas: document.getElementById('canvas'),
    ctx: null,
    clock: GameState.createClock(),
    actorLossValues: [],
    criticLossValues: [],
    continueLoop: false,
    loopCount: 0,
    negRewards: 0,
    totalRewards: 0,
    rewardOverTime: [],
    totalTurns: 0,
    missedShots: 0,
    hitShotsBaddy: 0,
    hitShotsHuman: 0,
    zombieSpeed: 1,
    maxZombieSpeed: GAME_CONSTANTS.HUMAN_SPEED * 2 / 3,
    isSprites: true,
    showEyes: 0,
    gameSpeed: 4,
    skipFrames: 0,
    maxId: 0,
    ppo: null,
    isVampire: false,
    chart: null,
    lossesChart: null
};

// Initialize context
gameState.ctx = gameState.canvas.getContext('2d');

// Sprite frame arrays
const spriteFrames = {
    zombieMove: [],
    zombieAttack: [],
    survivorMove: [],
    survivorShoot: []
};

// PPO Configuration
const configPpo = {
    nSteps: GAME_CONSTANTS.BATCH_SIZE,
    nEpochs: 10,
    policyLearningRate: GAME_CONSTANTS.LEARNING_RATE,
    valueLearningRate: GAME_CONSTANTS.LEARNING_RATE,
    clipRatio: 0.2,
    targetKL: 0.02,
    netArch: {
        'pi': [GAME_CONSTANTS.NUM_EYES * 2, GAME_CONSTANTS.NUM_EYES],
        'vf': [GAME_CONSTANTS.NUM_EYES * 2, GAME_CONSTANTS.NUM_EYES]
    },
    activation: 'elu',
    verbose: 0
};

// Reward configuration
let rewardConfig = rewardConfigs.explore;
for (let key in rewardConfig) {
    if (key !== 'baseReward') {
        rewardConfig[key] *= (rewardConfig.baseReward || 1);
    }
}

let {
    hitShotReward,
    biteReward,
    hitHumanReward,
    missedShotReward,
    bumpWallReward,
    bumpScreenReward,
    bumpHumanReward,
    blockedVisionHuman,
    blockedVisionWall,
    zombieProximityReward
} = rewardConfig;

// Sprite Loading
function initializeSprites() {
    // Load zombie move frames
    for (let i = 0; i < 17; i++) {
        spriteFrames.zombieMove[i] = new Image();
        spriteFrames.zombieMove[i].src = `img/skeleton/skeleton-move_${i}.png`;
    }
    
    // Load zombie attack frames
    for (let i = 0; i < 9; i++) {
        spriteFrames.zombieAttack[i] = new Image();
        spriteFrames.zombieAttack[i].src = `img/skeleton/skeleton-attack_${i}.png`;
    }
    
    // Load survivor move frames
    for (let i = 0; i < 20; i++) {
        spriteFrames.survivorMove[i] = new Image();
        spriteFrames.survivorMove[i].src = `img/survivor/survivor-move_rifle_${i}.png`;
    }
    
    // Load survivor shoot frames
    for (let i = 0; i < 3; i++) {
        spriteFrames.survivorShoot[i] = new Image();
        spriteFrames.survivorShoot[i].src = `img/survivor/survivor-shoot_rifle_${i}.png`;
    }
}

// Agent Class - Modernized
class Agent {
    constructor(config) {
        this.id = config.id;
        this.experiences = [];
        this.states = [];
        this.isLearning = false;
        this.eyes = [];
        this.rewardSignal = 0;
        
        // Agent properties
        this.maxHp = 50;
        this.currentHp = this.maxHp;
        this.items = [];
        this.type = config.type || 'human';
        this.pos = config.pos || new Vec(0, 0);
        this.minRad = config.rad || 8;
        this.rad = config.rad || 8;
        this.speed = config.speed || (this.isHuman ? GAME_CONSTANTS.HUMAN_SPEED : gameState.zombieSpeed);
        this.dir = this.isHuman ? new Vec(1, 0) : randomAngle();
        this.newDir = this.dir.getUnit();
        this.moveFactor = 1;
        this.v = new Vec(0, 0);
        this.state = config.state || 'idle';
        this.viewDist = config.viewDist || 1000;
        this.viewFov = config.viewFov || Math.PI / 2;
        this.viewFovD2 = this.viewFov / 2;
        this.nextTimer = Math.random() * 10;
        this.ring = config.ring || (this.type === 'human' ? 0 : 5);
        
        // Type flags
        this.isHuman = config.type === 'human';
        this.isZ = config.type === 'zombie' || config.type === 'vampire';
        this.isVampire = config.type === 'vampire';
        this.isShot = false;
        this.isBit = false;
        
        // Initialize eyes for humans
        if (this.isHuman) {
            const rads = 2 * Math.PI / GAME_CONSTANTS.NUM_EYES;
            for (let k = 0; k < GAME_CONSTANTS.NUM_EYES; k++) {
                this.eyes.push(new Eye(k * rads));
            }
        }
        
        // Set sprite frames
        this.moveFrames = this.isZ ? spriteFrames.zombieMove : spriteFrames.survivorMove;
        this.attackFrames = this.isZ ? spriteFrames.zombieAttack : spriteFrames.survivorShoot;
    }

    getVision() {
        const eyeStates = [];
        this.target = null;
        let closestZombieRange = GAME_CONSTANTS.EYE_MAX_RANGE;
        
        for (let ei = 0; ei < this.eyes.length; ei++) {
            const eye = this.eyes[ei];
            eye.sensed_proximity = eye.max_range;
            eye.sensed_type = 0;
            
            const currentEyeAnglePointing = this.dir.rotate(eye.angle).getUnit();
            const eyep = new Vec(
                this.pos.x + eye.max_range * currentEyeAnglePointing.x,
                this.pos.y + eye.max_range * currentEyeAnglePointing.y
            );
            
            if (isNaN(eyep.x)) {
                console.error('eyep.x is NaN');
            }
            
            const res = stuff_collide(this, eyep, gameState.blocks, true, true);
            if (res) {
                if (ei === 0) this.target = res.agent;
                eye.sensed_proximity = res.up.distFrom(this.pos);
                eye.sensed_type = res.type;
            }
            
            this.drawEyeVision(eye, currentEyeAnglePointing, ei);
            this.processEyeRewards(eye, ei, closestZombieRange);
            
            eyeStates.push(eye.sensed_proximity / eye.max_range, eye.sensed_type);
        }
        
        this.rewardSignal += zombieProximityReward * (1 - closestZombieRange / GAME_CONSTANTS.EYE_MAX_RANGE);
        return eyeStates;
    }

    drawEyeVision(eye, currentEyeAnglePointing) {
        gameState.ctx.strokeStyle = "rgb(0,0,0,0)";
        
        if (gameState.showEyes > 1) {
            gameState.ctx.strokeStyle = "rgb(255,150,150)";
            if (eye.sensed_type === -0.1) gameState.ctx.strokeStyle = "yellow";
            else if (eye.sensed_type === 0) gameState.ctx.strokeStyle = "rgb(200,200,200)";
        }
        
        if (gameState.showEyes > 0) {
            if (eye.sensed_type === 1) gameState.ctx.strokeStyle = "yellow";
            else if (eye.sensed_type === -1) gameState.ctx.strokeStyle = "green";
        }
        
        const sr = eye.sensed_proximity;
        gameState.ctx.beginPath();
        gameState.ctx.moveTo(this.pos.x, this.pos.y);
        const lineToX = this.pos.x + sr * currentEyeAnglePointing.x;
        const lineToY = this.pos.y + sr * currentEyeAnglePointing.y;
        gameState.ctx.lineTo(lineToX, lineToY);
        gameState.ctx.stroke();
    }

    processEyeRewards(eye, eyeIndex, closestZombieRange) {
        if (eyeIndex === 0) {
            if (eye.sensed_type === -0.1) {
                this.rewardSignal += blockedVisionWall * (1 - eye.sensed_proximity / (eye.max_range / 2));
            } else if (eye.sensed_type === 1) {
                this.rewardSignal += blockedVisionHuman;
            }
        } else if (eye.sensed_type === -1 && eye.sensed_proximity < closestZombieRange) {
            closestZombieRange = eye.sensed_proximity;
        }
        return closestZombieRange;
    }

    getColor() {
        if (this.isLearning) return 'blue';
        if (this.state === 'mouse') return '#FF00FF';
        if (this.state === 'attack') return 'red';
        if (this.isHuman) return 'purple';
        if (this.isZ) return GAME_CONSTANTS.ZOMBIE_GREEN;
        return '#AAAAAA';
    }

    see() {
        const seen = [];
        
        for (let i = 0; i < gameState.agents.length; i++) {
            const agent = gameState.agents[i];
            if (agent === this) continue;
            
            const distance = this.pos.distFrom(agent.pos);
            if (distance > this.viewDist) continue;
            
            const angleToAgent = Math.atan2(agent.pos.y - this.pos.y, agent.pos.x - this.pos.x);
            const fixedAngle = fixAngle(angleToAgent);
            const dir = Math.atan2(this.dir.y, this.dir.x);
            let a1 = fixAngle(dir - this.viewFovD2);
            let a2 = fixAngle(dir + this.viewFovD2);
            
            if (a1 > a2) [a1, a2] = [a2, a1];
            
            const inView = (a2 - a1 > Math.PI) ? 
                (fixedAngle <= a1 || fixedAngle > a2) : 
                (fixedAngle >= a1 && fixedAngle <= a2);
            
            if (inView && !this.isViewBlocked(agent, distance)) {
                seen.push({
                    agent: agent,
                    dist: distance,
                    angle: agent.pos.sub(this.pos).getUnit()
                });
            }
        }
        
        return seen.sort((a, b) => a.dist - b.dist);
    }

    isViewBlocked(targetAgent, distance) {
        const angle = targetAgent.pos.sub(this.pos).getUnit();
        
        for (const block of gameState.blocks) {
            const walls = block.rayIntersect(this.pos, angle);
            if (walls) {
                for (const wall of walls) {
                    if (wall.dist > 0 && wall.dist < distance) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    async getStates() {
        return [...this.getVision()];
    }

    async zombify(victim, zombie) {
        console.log(`Zombifying human: ${victim.id}`);
        const { pos, dir, id } = victim;
        Object.assign(victim, zombie);
        victim.id = id;
        victim.pos = pos;
        victim.dir = dir;
        victim.currentHp = this.maxHp;
        victim.ring = 1;
        victim.state = 'idle';
    }

    async logic(clock, action, agentExperienceResult) {
        this.moveFactor = this.isShot ? 0.1 : 1;

        if (this.isZ) {
            await this.processZombieLogic();
        } else {
            this.processHumanLogic(action);
        }

        this.updateRing(clock);
        this.nextTimer -= clock.delta;
        this.isShot = false;
        this.isBit = false;
        this.move(clock);
        this.handleCollisions();
        this.checkScreenBounds();

        if (this.isHuman) {
            return await this.processHumanRewards(agentExperienceResult);
        }
    }

    async processZombieLogic() {
        let inMelee = false;
        this.state = 'idle';

        const seen = this.see();
        const seeHuman = seen.find(s => s.agent.isHuman);
        
        if (seeHuman) {
            this.state = 'attack';
            if (this.nextTimer <= 0) {
                this.nextTimer = 1;
                this.newDir = seeHuman.angle;
                this.dir = seeHuman.angle;
            }
        }

        // Handle melee combat
        for (const seenAgent of seen) {
            if (seenAgent.dist <= this.rad * 2 && seenAgent.agent.isHuman) {
                const human = seenAgent.agent;
                this.moveFactor = 0;
                inMelee = true;
                human.currentHp--;
                human.isBit = true;

                if (gameState.isVampire || this.isVampire) {
                    this.currentHp += this.currentHp / 20;
                }

                human.rewardSignal += biteReward;
                gameState.negRewards += biteReward;
                
                if (human.currentHp < 1) {
                    await this.zombify(human, this);
                }
            }
        }

        this.updateZombieMovement(inMelee, seen);
    }

    updateZombieMovement(inMelee, seen) {
        this.rad = Math.max(this.minRad, this.minRad + this.currentHp - this.maxHp);
        
        if (!inMelee && this.moveFactor === 0) {
            if (this.nextTimer <= 0 || this.state === 'attack') {
                this.state = 'idle';
                this.nextTimer = 3 + Math.random() * 2;
                this.newDir = randomAngle();
                this.dir = this.newDir;
                this.moveFactor = 0.2;
            }
        } else if (this.state === 'idle' && Math.random() > 0.9 && seen[0] && this.moveFactor > 0) {
            if (this.nextTimer <= 0) {
                this.nextTimer = 5;
                this.newDir = seen[0].angle;
                this.dir = seen[0].angle;
            }
        } else if (this.state === 'idle' && this.nextTimer <= 0) {
            this.newDir = randomAngle();
            this.dir = this.newDir;
            this.nextTimer = 3 + Math.random() * 10;
        }
        
        if (this.isShot) this.moveFactor = 0.1;
    }

    processHumanLogic(action) {
        const numAngles = GAME_CONSTANTS.NUM_ACTIONS - 2;
        let newAngle = 0;
        
        if (action < numAngles) {
            newAngle = (action - Math.floor(numAngles / 2)) * (2 * Math.PI / this.eyes.length);
        } else if (action === GAME_CONSTANTS.NUM_ACTIONS - 2) {
            this.moveFactor = 0;
        } else {
            this.moveFactor = 0;
            this.shoot();
        }
        
        // Stop if collided with another human
        if (this.eyes[0].sensed_type === 1 && this.eyes[0].sensed_proximity < this.rad * 2) {
            this.rewardSignal += bumpHumanReward;
            gameState.negRewards += bumpHumanReward;
            this.moveFactor = 0;
        }
        
        const unitOldDir = new Vec(this.dir.x, this.dir.y).getUnit();
        const newVec = unitOldDir.rotate(newAngle);
        this.dir = newVec;
    }

    updateRing(clock) {
        if (this.ring) {
            this.ring += clock.delta * 20;
            if (this.ring > 100) this.ring = 0;
        }
    }

    move(clock) {
        const speed = this.moveFactor * this.speed * 10;
        const vx = this.dir.x * speed * clock.delta;
        const vy = this.dir.y * speed * clock.delta;
        this.pos.x += vx;
        this.pos.y += vy;
    }

    handleCollisions() {
        for (const block of gameState.blocks) {
            this.intersect = block.rayIntersect(this.pos, this.dir);
            if (this.intersect) {
                if (this.intersect[0].dist <= 0 && this.intersect[1].dist > 0) {
                    this.pos = this.intersect[0].pos;
                    this.rewardSignal += bumpWallReward;
                    if (this.isHuman) {
                        gameState.negRewards += bumpWallReward;
                    }
                    break;
                } else {
                    this.intersect = false;
                }
            }
        }
    }

    checkScreenBounds() {
        let bound = false;
        const { canvas } = gameState;
        
        if (this.pos.x < 0) {
            this.pos.x = 1;
            this.dir.x = 1;
            bound = true;
        }
        if (this.pos.y < 0) {
            this.pos.y = 1;
            this.dir.y = 1;
            bound = true;
        }
        if (this.pos.x > canvas.width) {
            this.pos.x = canvas.width - 1;
            this.dir.x = -1;
            bound = true;
        }
        if (this.pos.y > canvas.height) {
            this.pos.y = canvas.height - 1;
            this.dir.y = -1;
            bound = true;
        }
        
        if (bound) {
            this.rewardSignal += bumpScreenReward;
            gameState.negRewards += bumpScreenReward;
            this.dir.normalize();
        }
    }

    shoot() {
        const { ctx } = gameState;
        
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(this.pos.x, this.pos.y);
        
        const sr = this.eyes[0].sensed_proximity;
        const lineToX = this.pos.x + sr * this.dir.x;
        const lineToY = this.pos.y + sr * this.dir.y;
        const closestTarget = this.target;
        
        if (closestTarget) {
            if (closestTarget.isHuman) {
                ctx.strokeStyle = 'orange';
                this.rewardSignal += hitHumanReward;
                gameState.negRewards += hitHumanReward;
                gameState.hitShotsHuman += 1;
            } else {
                ctx.strokeStyle = 'red';
                if (!this.isBit) this.rewardSignal += hitShotReward;
                gameState.hitShotsBaddy += 1;
            }
            
            closestTarget.isShot = true;
            closestTarget.currentHp--;
            
            if (closestTarget.currentHp < 1) {
                closestTarget.ring = 5;
                console.log(`Killed target ${closestTarget.id} ${closestTarget.type}`);
                removeUnit(closestTarget);
            }
        } else {
            const missedReward = missedShotReward - Math.min(4, 1 * gameState.hitShotsBaddy / 10000);
            this.rewardSignal += missedReward;
            gameState.negRewards += missedReward;
            gameState.missedShots += 1;
            ctx.strokeStyle = 'purple';
        }
        
        ctx.lineTo(lineToX, lineToY);
        ctx.stroke();
    }

    async processHumanRewards(agentExperienceResult) {
        this.states = await this.getStates();
        const ret = {
            newObservation: this.states,
            reward: this.rewardSignal,
            done: false
        };
        
        if (agentExperienceResult) {
            agentExperienceResult.newObservation = this.states;
            agentExperienceResult.reward = this.rewardSignal;
        }
        
        gameState.totalRewards += this.rewardSignal;
        gameState.rewardOverTime.push(this.rewardSignal);
        $("#rewardTotal").text(gameState.totalRewards.toFixed(5));
        $("#neg-rewards").text(gameState.negRewards.toFixed(5));
        this.rewardSignal = 0;
        
        return ret;
    }

    draw(ctx) {
        if (this.ring) {
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.ring, 0, pi2, false);
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#FF0000';
            ctx.stroke();
        }
        
        let viewedRad = this.rad;
        let image = null;
        const dir = new Vec(this.dir.x, this.dir.y).getUnit();
        
        if (gameState.isSprites) {
            const images = this.moveFactor > 0 ? this.moveFrames : this.attackFrames;
            if (images.length > 0) {
                const imageChangeRate = this.moveFactor > 0 ? 1 / 4 : 1;
                const index = Math.floor((gameState.totalTurns + this.id) * imageChangeRate * this.moveFactor) % images.length;
                image = images[index];
                viewedRad = this.rad * 1.5;
                ctx.globalAlpha = 0.3;
            }
        }
        
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, viewedRad, 0, pi2, false);
        ctx.fillStyle = this.getColor();
        ctx.fill();
        ctx.globalAlpha = 1.0;
        
        if (image) {
            const width = this.rad * 4;
            const height = this.rad * 4;
            const centerX = this.pos.x - width / 2;
            const centerY = this.pos.y - height / 2;
            const angle = Math.atan2(dir.y, dir.x);
            drawRotatedImage(ctx, image, centerX, centerY, width, height, angle);
        } else {
            ctx.beginPath();
            ctx.moveTo(this.pos.x, this.pos.y);
            ctx.lineTo(this.pos.x + dir.x * this.rad, this.pos.y + dir.y * this.rad);
            ctx.strokeStyle = '#00FFFF';
            ctx.stroke();
        }
    }
}

// Environment Class
class Env {
    constructor() {
        this.actionSpace = {
            'class': 'Discrete',
            'n': GAME_CONSTANTS.NUM_ACTIONS,
        };
        this.observationSpace = {
            'class': 'Box',
            'shape': [GAME_CONSTANTS.NUM_EYES * 2],
            'dtype': 'float32',
        };
        this.resets = 0;
        this.i = 0;
    }

    async step(action) {
        if (Array.isArray(action)) {
            action = action[0];
        }
        
        const agentExperienceResult = {
            newObservation: null,
            reward: null
        };
        
        gameState.loopCount++;
        
        if ((gameState.loopCount > GAME_CONSTANTS.BATCH_SIZE || !gameState.continueLoop) && 
            (gameState.skipFrames === 0 || gameState.loopCount % gameState.skipFrames === 0)) {
            
            if (gameState.loopCount > GAME_CONSTANTS.BATCH_SIZE) {
                gameState.loopCount = 0;
            }
            
            if (gameState.loopCount < 1) {
                await this.updateUI();
            }
            
            gameState.ctx = gameState.canvas.getContext('2d');
            await requestAnimationFrameAsync(async (time) => await mainLoop(time, action, agentExperienceResult));
        } else {
            gameState.ctx = this.createDummyContext();
            if (gameState.loopCount >= GAME_CONSTANTS.BATCH_SIZE * 10) {
                gameState.loopCount = 0;
                gameState.continueLoop = false;
            }
            await mainLoop(0, action, agentExperienceResult);
        }
        
        this.i += 1;
        return [agentExperienceResult.newObservation, agentExperienceResult.reward, false];
    }

    createDummyContext() {
        return {
            isDummy: true,
            beginPath: () => {},
            arc: () => {},
            fill: () => {},
            stroke: () => {},
            clearRect: () => {},
            fillText: () => {},
            lineTo: () => {},
            moveTo: () => {}
        };
    }

    async updateUI() {
        const weights = gameState.ppo.actor.getWeights();
        const criticWeights = gameState.ppo.critic.getWeights();
        const weightsData = weights.map(weight => weight.dataSync());
        const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

        $("#weights").text(weightsData);
        $("#criticWeights").text(criticWeightsData);
        $("#current-state").text(gameState.agents.find(a => a.isHuman).states.join(', '));

        createOrUpdateRewardChart(gameState.rewardOverTime, GAME_CONSTANTS.BATCH_SIZE);
        await createOrUpdateLossesChart();
    }

    reset() {
        this.i = 0;
        const humanAgent = gameState.agents.find(a => a.isHuman);
        if (humanAgent && humanAgent.states.length > 0) {
            return humanAgent.states;
        }
        return new Array(this.observationSpace.shape[0]).fill(0.1);
    }
}

// Main Game Loop
async function mainLoop(time, action, agentExperienceResult) {
    if (!time) time = Date.now();
    
    GameState.updateClock(gameState.clock, time, gameState.gameSpeed);
    if (gameState.hCnt) gameState.clock.total += gameState.clock.delta;

    gameState.ctx.clearRect(0, 0, gameState.canvas.width, gameState.canvas.height);
    gameState.hCnt = 0;
    gameState.zCnt = 0;

    // Draw blocks
    for (const block of gameState.blocks) {
        block.draw(gameState.ctx);
    }

    // Spawn zombies based on game progression
    if (gameState.totalTurns > 10000 && gameState.totalTurns % 1000 === 0) {
        const numZombs = Math.min((gameState.totalTurns - 10000) / 4000, GAME_CONSTANTS.MAX_ZOMBIE_SPAWNS);
        for (let i = 0; i < numZombs; i++) {
            const speed = Math.min(gameState.zombieSpeed + gameState.totalTurns / 40000, gameState.maxZombieSpeed);
            addUnit({ type: 'zombie', speed });
        }
    }

    // Process zombies
    await processZombies();
    
    // Process humans
    await processHumans(action, agentExperienceResult);

    // Draw UI
    drawUI();
    
    gameState.totalTurns++;
    updateStats();
    
    return agentExperienceResult;
}

async function processZombies() {
    let zombies = gameState.agents.filter(agent => agent.isZ);
    
    if (zombies.length < GAME_CONSTANTS.MIN_ZOMBIES) {
        const zombieHousePos = new Vec(gameState.canvas.width / 2, gameState.canvas.height / 2);
        addUnit({ type: 'zombie', pos: zombieHousePos, speed: gameState.zombieSpeed });
        zombies = gameState.agents.filter(agent => agent.isZ);
    }
    
    for (const zombie of zombies) {
        gameState.zCnt++;
        await zombie.logic(gameState.clock);
        zombie.draw(gameState.ctx);
    }
}

async function processHumans(action, agentExperienceResult) {
    let humans = gameState.agents.filter(agent => agent.isHuman);
    
    if (humans.length < GAME_CONSTANTS.MIN_HUMANS) {
        const hasLearning = humans.some(h => h.isLearning);
        const block = gameState.blocks[Math.floor(Math.random() * gameState.blocks.length)];
        addUnit({ type: 'human', pos: new Vec(block.pos.x, block.pos.y) }, !hasLearning);
        humans = gameState.agents.filter(agent => agent.isHuman);
    }
    
    gameState.hCnt = humans.length;

    for (let i = 0; i < humans.length; i++) {
        const human = humans[i];
        
        if (i === 0) {
            human.isLearning = true;
            await human.logic(gameState.clock, action, agentExperienceResult);
        } else {
            const states = [...(human.states && human.states.length > 0 ? human.states : await human.getStates())];
            const [preds, , value, logprobability] = await gameState.ppo.getSample(states);
            human.isLearning = false;
            const bestAction = tf.argMax(preds).dataSync()[0];
            const rets = await human.logic(gameState.clock, bestAction);

            if (i < 1) {
                human.experiences.push([states, bestAction, rets.reward, value, logprobability]);

                if (gameState.ppo.buffer.pointer === 0 && gameState.totalTurns > 0) {
                    for (const [states, action, reward, value, logprobability] of human.experiences) {
                        gameState.ppo.buffer.add(states, action, reward, value, logprobability);
                    }
                    human.experiences = [];
                }
            }
        }
        human.draw(gameState.ctx);
    }
}

function drawUI() {
    const { ctx, canvas } = gameState;
    
    ctx.font = '20pt Calibri';
    ctx.lineWidth = 1;
    ctx.fillStyle = 'black';
    const msg = `Zed:${gameState.zCnt}     Hum:${gameState.hCnt}    Time:${Math.floor(gameState.clock.total)}`;
    ctx.fillText(msg, canvas.width / 3 + 1, 21);
    ctx.fillStyle = 'white';
    ctx.fillText(msg, canvas.width / 3, 20);
    ctx.beginPath();
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#FFFFFF';
    ctx.stroke();
}

function updateStats() {
    $("#hit-shots-human").text(gameState.hitShotsHuman);
    $("#hit-shots-baddy").text(gameState.hitShotsBaddy);
    $("#missed-shots").text(gameState.missedShots);
    $("#turns").text(gameState.totalTurns);
}

// Chart Functions
async function createOrUpdateLossesChart() {
    const ctx = document.getElementById('losses-chart').getContext('2d');
    
    if (!gameState.lossesChart) {
        gameState.lossesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: gameState.actorLossValues.map((_, i) => i + 1),
                datasets: [{
                    label: 'Actor Loss',
                    data: gameState.actorLossValues,
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }, {
                    label: 'Critic Loss',
                    data: gameState.criticLossValues,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } else {
        gameState.lossesChart.data.labels = gameState.actorLossValues.map((_, i) => i + 1);
        gameState.lossesChart.data.datasets[0].data = gameState.actorLossValues;
        gameState.lossesChart.data.datasets[1].data = gameState.criticLossValues;
        gameState.lossesChart.update();
    }
}

function createOrUpdateRewardChart(rewardOverTime, batchSize) {
    const ctx = document.getElementById('rewardOverTimeChart').getContext('2d');

    // Calculate average reward for each batch
    const avgRewards = [];
    for (let i = 0; i < rewardOverTime.length; i += batchSize) {
        const batch = rewardOverTime.slice(i, i + batchSize);
        const batchAvg = batch.reduce((a, b) => a + b, 0) / batch.length;
        avgRewards.push(batchAvg);
    }

    if (!gameState.chart) {
        gameState.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: avgRewards.map((_, i) => i),
                datasets: [{
                    label: 'Average reward over time',
                    data: avgRewards,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } else {
        gameState.chart.data.labels = avgRewards.map((_, i) => i);
        gameState.chart.data.datasets[0].data = avgRewards;
        gameState.chart.update();
    }
}

// Unit Management
async function removeUnit(unit) {
    for (const agent of gameState.agents) {
        agent.items = agent.items.filter(item => item !== unit);
    }
    gameState.agents = gameState.agents.filter(agent => agent !== unit);
}

async function addUnit(config, isLearning = false) {
    gameState.maxId++;
    const agent = new Agent({
        id: gameState.maxId,
        type: config.type || 'zombie',
        viewDist: 1000,
        pos: config?.pos ? new Vec(config.pos.x, config.pos.y) : 
             new Vec(gameState.canvas.width * Math.random(), gameState.canvas.height * Math.random()),
        speed: config.speed || null,
    });
    
    if (isLearning) {
        gameState.agents.unshift(agent);
    } else {
        gameState.agents.push(agent);
    }
    
    const humans = gameState.agents.filter(a => a.isHuman);
    humans.forEach(h => {
        h.items = gameState.agents.filter(a => a.id !== h.id);
        h.viewDist = 1000;
    });
}

// Model Management
const loadModels = async () => {
    const name = 'zed-tf';
    const models = await ModelPersistence.load(name);
    gameState.ppo.actor = models.actor;
    gameState.ppo.critic = models.critic;
};

const saveModels = async () => {
    const name = 'zed-tf';
    await ModelPersistence.save(gameState.ppo.actor, gameState.ppo.critic, name);
};

// Event Listeners
function setupEventListeners() {
    // Keyboard events
    window.addEventListener('keydown', (event) => {
        switch (event.code) {
            case 'Space':
                gameState.continueLoop = !gameState.continueLoop;
                break;
            case 'KeyV':
                gameState.isVampire = !gameState.isVampire;
                break;
            case 'KeyL':
                loadModels();
                break;
            case 'KeyS':
                saveModels();
                break;
        }
    });

    // Button events
    $('#vampire-button').click(() => {
        gameState.isVampire = !gameState.isVampire;
    });

    $('#add-vampire-button').click(() => {
        addUnit({ type: 'vampire' });
    });

    $('#rush-watch-button').click(() => {
        gameState.continueLoop = !gameState.continueLoop;
    });

    $('#save-button').click(async () => {
        await saveModels();
    });

    $('#smith-button').click(async () => {
        const learningIndex = gameState.agents.findIndex(a => a.isLearning);
        const nonLearningIndex = gameState.agents.findIndex(a => !a.isLearning && a.isHuman);
        
        if (learningIndex !== -1 && nonLearningIndex !== -1) {
            const learningAgent = gameState.agents[learningIndex];
            const nonLearningAgent = gameState.agents[nonLearningIndex];
            
            learningAgent.isLearning = false;
            nonLearningAgent.isLearning = true;
            
            gameState.agents[learningIndex] = nonLearningAgent;
            gameState.agents[nonLearningIndex] = learningAgent;
        }
    });

    $('#load-button').click(async () => {
        await loadModels();
    });

    $('#add-zombie-button').click(async () => {
        await addUnit({ type: 'zombie' });
    });

    $('#add-human-button').click(async () => {
        await addUnit({ type: 'human' });
    });

    $('#toggle-sprites').click(async () => {
        gameState.isSprites = !gameState.isSprites;
    });

    // Slider events
    $('#show-eyes').on('input', function () {
        gameState.showEyes = +$(this).val();
    });

    $('#skip-frames').on('input', function () {
        gameState.skipFrames = +$(this).val();
    });

    $('#gameSpeed').change(function () {
        gameState.gameSpeed = $(this).val();
    }).val(gameState.gameSpeed);

    $('#maxZombieSpeed').change(function () {
        gameState.zombieSpeed = $(this).val();
    });

    // Modal events
    const rewardModal = $('#rewardModal');
    $('#openRewardModal').click(() => {
        rewardModal.show();
    });

    $('.close-modal').click(() => {
        rewardModal.hide();
    });

    $(window).click((event) => {
        if (event.target === rewardModal[0]) {
            rewardModal.hide();
        }
    });

    $('#rewardForm').submit((event) => {
        event.preventDefault();
        const newRewardConfig = $(event.target).serializeArray().reduce((obj, item) => {
            obj[item.name] = +item.value;
            return obj;
        }, {});
        
        console.log(newRewardConfig);
        
        ({
            hitShotReward,
            biteReward,
            hitHumanReward,
            missedShotReward,
            bumpWallReward,
            bumpScreenReward,
            bumpHumanReward,
            blockedVisionHuman,
            blockedVisionWall,
            zombieProximityReward
        } = newRewardConfig);
    });
}

// Initialization
(async function initialize() {
    // Initialize sprites
    initializeSprites();
    
    // Fill in the form with current values
    for (const key in rewardConfig) {
        $(`#${key}`).val(rewardConfig[key]);
    }

    $("#help").click(function () {
        $(this).hide();
    });

    // Create game world
    const squares = createGrid(gameState.ctx, 20, 100, 50, 100, 70);
    for (const square of squares) {
        gameState.blocks.push(square);
    }

    // Create initial agents
    const maxAgents = 53;
    const numHumans = 50;
    
    for (let i = 0; i < maxAgents; i++) {
        const agent = new Agent({
            id: i + 1,
            type: i < numHumans ? 'human' : 'zombie',
            viewDist: 1000,
            pos: i === 0 ? new Vec(gameState.canvas.width / 2, gameState.canvas.height / 2) :
                 new Vec(gameState.canvas.width * Math.random(), gameState.canvas.height * Math.random()),
        });
        gameState.agents.push(agent);
        gameState.maxId = i + 1;
    }

    const humans = gameState.agents.filter(a => a.isHuman);
    humans.forEach(h => {
        h.items = gameState.agents.filter(a => a.id !== h.id);
        h.viewDist = 1000;
    });

    // Setup event listeners
    setupEventListeners();

    // Initialize environment and PPO
    const env = new Env();
    gameState.ppo = new PPO(env, configPpo);
    
    await gameState.ppo.learn({
        'totalTimesteps': Infinity,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config);
            }
        }
    });
})();

// Assets credit: zombies and survivors from https://opengameart.org/content/animated-top-down-zombie
