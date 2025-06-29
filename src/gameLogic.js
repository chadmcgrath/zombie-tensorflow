import * as tf from '@tensorflow/tfjs';
import { PPO } from './ppo1.js';
import { rewardConfigs } from './config.js';
import { 
    Vec, 
    randomAngle, 
    fixAngle, 
    stuff_collide, 
    createGrid, 
    Square 
} from './utilities.js';

// Set TensorFlow backend
tf.setBackend('cpu');

// Game state variables
let zCnt = 0;
let hCnt = 0;
let agents = [];
const blocks = [];
let canvas = null;
let ctx = null;
const clock = {
    total: 0,
    start: 0,
    time: 0,
    delta: 0
};

const numEyes = 30;
const numInputs = numEyes * 2;
const actorLossValues = [];
const criticLossValues = [];
let continueLoop = false;
let loopCount = 0;
let negRewards = 0;
let totalRewards = 0;
let rewardOverTime = [];
let totalTurns = 0;
let missedShots = 0;
let hitShotsBaddy = 0;
let hitShotsHuman = 0;
const minHumans = 2;
const minZombies = 3;
const maxZombieSpawns = 40;

const zombieGreen = "#2f402f";
let zombieHousePos = null;
let zombieSpeed = 1;
const humanSpeed = 3;
let maxZombieSpeed = humanSpeed * 2 / 3;
const numActions = 11;

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

let isSprites = true;
let showEyes = 0;
const eyeMaxRange = 1000;
let gameSpeed = 4;
let skipFrames = 0;

let batchSize = 512;
const learningRate = .001;
const configPpo = {
    nSteps: batchSize,
    nEpochs: 10,
    policyLearningRate: learningRate,
    valueLearningRate: learningRate,
    clipRatio: 0.2,
    targetKL: 0.02,
    netArch: {
        'pi': [numInputs, numInputs/2],
        'vf': [numInputs, numInputs/2]
    },
    activation: 'elu',
    verbose: 0
};

// Sprite frames
const zombieMoveFrames = [];
const zombieAttackFrames = [];
const survivorMoveFrames = [];
const survivorShootFrames = [];

let maxId = 0;
let ppo = null;
let isVampire = false;

// Eye class
class Eye {
    constructor(angle) {
        this.angle = angle;
        this.max_range = eyeMaxRange;
        this.sensed_proximity = eyeMaxRange;
        this.sensed_type = 0;
    }
}

// Agent class
class Agent {
    constructor(config) {
        this.id = config.id;
        this.experiences = [];
        this.states = [];
        this.isLearning = false;
        const maxHp = 50;
        const eyeCount = numEyes;
        this.eyes = [];
        this.rewardSignal = 0;

        const rads = 2 * Math.PI / eyeCount;
        this.isHuman = config.type === 'human';
        this.isZ = config.type === 'zombie' || config.type === 'vampire';
        this.isVampire = config.type === 'vampire';
        this.isShot = false;
        this.isBit = false;
        
        if (this.isHuman) {
            for (var k = 0; k < eyeCount; k++) { 
                this.eyes.push(new Eye(k * rads)); 
            }
        }

        this.maxHp = maxHp;
        this.currentHp = maxHp;
        this.items = [];

        this.type = config.type || 'human';
        this.pos = config.pos || new Vec(0, 0);
        this.minRad = config.rad || 8;
        this.rad = config.rad || 8;
        this.speed = config.speed || (this.isHuman ? humanSpeed : zombieSpeed);
        this.dir = this.isHuman ? new Vec(1, 0) : randomAngle();
        this.newDir = this.dir.getUnit();

        this.moveFactor = 1;
        this.moveFrames = this.isZ ? zombieMoveFrames : survivorMoveFrames;
        this.attackFrames = this.isZ ? zombieAttackFrames : survivorShootFrames;
        this.v = new Vec(0, 0);
        this.state = config.state || 'idle';
        this.viewDist = config.viewDist || 1000;
        this.viewFov = (config.viewFov || Math.PI / 2);
        this.viewFovD2 = this.viewFov / 2;
        this.nextTimer = Math.random() * 10;
        this.ring = config.ring || this.type === 'human' ? 0 : 5;
    }

    getVision() {
        let eyeStates = [];
        var a = this;
        a.target = null;
        let closestZombieRange = eyeMaxRange;
        
        for (var ei = 0, ne = a.eyes.length; ei < ne; ei++) {
            var e = a.eyes[ei];
            e.sensed_proximity = e.max_range;
            e.sensed_type = 0;
            const currentEyeAnglePointing = a.dir.rotate(e.angle).getUnit();
            var eyep = new Vec(a.pos.x + e.max_range * currentEyeAnglePointing.x,
                a.pos.y + e.max_range * currentEyeAnglePointing.y);
            
            if (isNaN(eyep.x)) {
                console.error('eyep.x is NaN');
            }
            
            var res = stuff_collide(a, eyep, blocks, true, true);
            if (res) {
                if (ei === 0)
                    a.target = res.agent;
                e.sensed_proximity = res.up.distFrom(a.pos);
                e.sensed_type = res.type;
            }

            if (ctx && !ctx.isDummy) {
                ctx.strokeStyle = "rgb(0,0,0,0)";
                if (showEyes > 1) {
                    ctx.strokeStyle = "rgb(255,150,150)";
                    if (e.sensed_type === -.1) { ctx.strokeStyle = "yellow"; }
                    else if (e.sensed_type === 0) { ctx.strokeStyle = "rgb(200,200,200)"; }
                }
                if (showEyes > 0) {
                    if (e.sensed_type === 1) { ctx.strokeStyle = "yellow"; }
                    else if (e.sensed_type === -1) { ctx.strokeStyle = "rgb(150,255,150)"; }
                    if (e.sensed_type === -1) { ctx.strokeStyle = "green"; }
                }

                const sr = e.sensed_proximity;
                ctx.beginPath();
                ctx.moveTo(a.pos.x, a.pos.y);
                const lineToX = a.pos.x + sr * currentEyeAnglePointing.x;
                const lineToY = a.pos.y + sr * currentEyeAnglePointing.y;
                ctx.lineTo(lineToX, lineToY);
                ctx.stroke();
            }

            let type = e.sensed_type;
            
            if (ei === 0) {           
                if (e.sensed_type === -.1)
                    this.rewardSignal = this.rewardSignal + (blockedVisionWall * (1 - e.sensed_proximity / (e.max_range / 2)));
                else if (e.sensed_type === 1)
                    this.rewardSignal = this.rewardSignal + blockedVisionHuman;
            }
            else if (e.sensed_type === -1 && e.sensed_proximity < closestZombieRange) {
                closestZombieRange = e.sensed_proximity;
            }
            
            eyeStates.push(e.sensed_proximity/e.max_range, type);
        }
        
        this.rewardSignal = this.rewardSignal + zombieProximityReward * (1 - closestZombieRange / eyeMaxRange);
        return eyeStates;
    }

    getColor() {
        if (this.isLearning) return 'blue';
        if (this.state === 'mouse') return '#FF00FF';
        if (this.state === 'attack') return 'red';
        if (this.isHuman) return 'purple';
        if (this.isZ) return zombieGreen;
        return '#AAAAAA';
    }

    see() {
        var seen = [];
        var a, d, ato;
        for (var i = 0, l = agents.length; i < l; i++) {
            a = agents[i];
            if (a === this) continue;
            d = this.pos.distFrom(a.pos);
            if (d > this.viewDist) continue;
            ato = Math.atan2(a.pos.y - this.pos.y, a.pos.x - this.pos.x);
            ato = fixAngle(ato);
            var dir = Math.atan2(this.dir.y, this.dir.x);
            var a1 = fixAngle(dir - this.viewFovD2);
            var a2 = fixAngle(dir + this.viewFovD2);
            if (a1 > a2) {
                var t = a1;
                a1 = a2;
                a2 = t;
            }
            var good = false;
            if (a2 - a1 > Math.PI) {
                if (ato <= a1 || ato > a2) good = true;
            } else {
                if (ato >= a1 && ato <= a2) good = true;
            }
            if (good) {
                const angle = a.pos.sub(this.pos).getUnit();
                let viewBlocked = false;
                for (var wi = 0, wl = blocks.length; wi < wl; wi++) {
                    let walls = blocks[wi].rayIntersect(this.pos, angle);
                    if (walls) {
                        for (let wall of walls) {
                            if (wall.dist > 0 && wall.dist < d) {
                                viewBlocked = true;
                                break;
                            }
                        }
                    }
                    if (viewBlocked) break;
                }
                if (!viewBlocked)
                    seen.push({
                        agent: a,
                        dist: d,
                        angle: angle
                    });
            }
        }

        if (seen.length > 1) seen.sort(function (a, b) {
            if (a.dist === b.dist) return 0;
            return a.dist < b.dist ? -1 : 1;
        });
        return seen;
    }

    async getStates() {
        const vision = [...this.getVision()];
        return vision;
    }

    async zombify(victim, zombie) {
        console.log('zombifying human: ' + victim.id);
        const p = victim.pos;
        const d = victim.dir;
        const id = victim.id;
        Object.assign(victim, zombie);
        victim.id = id;
        victim.pos = p;
        victim.dir = d;
        victim.currentHp = this.maxHp;
        victim.ring = 1;
        victim.state = 'idle';
    }

    async logic(clock, action, agentExperienceResult) {
        this.moveFactor = this.isShot ? .1 : 1;

        if (this.isZ) {
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

            for (let i = 0, l = seen.length; i < l; i++) {
                if (seen[i].dist <= this.rad * 2) {
                    this.moveFactor = 0;
                    if (seen[i].agent.isHuman) {
                        const human = seen[i].agent;
                        this.moveFactor = 0;
                        inMelee = true;
                        --human.currentHp;
                        human.isBit = true;

                        if (isVampire || this.isVampire)
                            ++this.currentHp / 20;

                        human.rewardSignal = human.rewardSignal + biteReward;
                        negRewards = negRewards + biteReward;
                        if (human.currentHp < 1)
                            await this.zombify(human, this);
                    }
                }
            }

            this.rad = Math.max(this.minRad, this.minRad + this.currentHp - this.maxHp);
            
            if (!inMelee && this.moveFactor === 0) {
                if (this.nextTimer <= 0 || this.state === 'attack') {
                    this.state = 'idle';
                    this.nextTimer = 3 + Math.random() * 2;
                    this.newDir = randomAngle();
                    this.dir = this.newDir;
                    this.moveFactor = .2;
                }
            }
            else if (this.state === 'idle' && Math.random() > 0.9 && seen[0] && this.moveFactor > 0) {
                if (this.nextTimer <= 0) {
                    this.nextTimer = 5;
                    this.newDir = seen[0].angle;
                    this.dir = seen[0].angle;
                }
            }
            else if (this.state === 'idle') {
                if (this.nextTimer <= 0) {
                    this.newDir = randomAngle();
                    this.dir = this.newDir;
                    this.nextTimer = 3 + Math.random() * 10;
                }
            }
            if (this.isShot)
                this.moveFactor = .1;
        }
        else {
            const actionSelected = action;
            let newAngle = 0;
            const numAngles = numActions - 2;
            if (actionSelected < numAngles) {
                newAngle = (actionSelected - (Math.floor(numAngles / 2))) * (2 * Math.PI / this.eyes.length);
            } else if (actionSelected === (numActions - 2)) {
                this.moveFactor = 0;
            }
            else {
                this.moveFactor = 0;
                this.shoot(this);
            }
            
            if (this.eyes[0].sensed_type === 1 && this.eyes[0].sensed_proximity < this.rad * 2) {
                this.rewardSignal = this.rewardSignal + bumpHumanReward;
                negRewards = negRewards + bumpHumanReward;
                this.moveFactor = 0;
            }

            const unitOldDir = new Vec(this.dir.x, this.dir.y).getUnit();
            const newVec = unitOldDir.rotate(newAngle);
            this.dir = newVec;
        }

        if (this.ring) {
            this.ring += clock.delta * 20;
            if (this.ring > 100) this.ring = 0;
        }
        this.nextTimer -= clock.delta;

        var speed = this.moveFactor * (this.speed) * 10;
        this.isShot = false;
        this.isBit = false;
        
        var vx = this.dir.x * speed * clock.delta;
        var vy = this.dir.y * speed * clock.delta;
        
        this.pos.x += vx;
        this.pos.y += vy;
        
        for (let i = 0, l = blocks.length; i < l; i++) {
            this.intersect = blocks[i].rayIntersect(this.pos, this.dir)
            if (this.intersect) {
                if (this.intersect[0].dist <= 0 && this.intersect[1].dist > 0) {
                    this.pos = this.intersect[0].pos;
                    this.rewardSignal = this.rewardSignal + bumpWallReward
                    if (this.isHuman)
                        negRewards = negRewards + bumpWallReward;
                    break;
                }
                else {
                    this.intersect = false;
                }
            }
        }

        this.CheckScreenBounds();

        if (this.isHuman === true) {
            this.states = await this.getStates();
            const ret = {
                newObservation: this.states,
                reward: this.rewardSignal,
                done: false
            }
            
            if (agentExperienceResult) {
                agentExperienceResult.newObservation = this.states;
                agentExperienceResult.reward = this.rewardSignal;
            }

            totalRewards += this.rewardSignal;
            rewardOverTime.push(this.rewardSignal);
            
            // Update UI elements if they exist
            if (typeof window !== 'undefined' && window.$) {
                window.$("#rewardTotal").text(totalRewards.toFixed(5));
                window.$("#neg-rewards").text(negRewards.toFixed(5));
            }
            
            this.rewardSignal = 0;
            return ret;
        }
    }

    CheckScreenBounds() {
        var bound = false;
        if (this.pos.x < 0) {
            this.pos.x = 1;
            this.dir.x = 1;
            bound = true
        }
        if (this.pos.y < 0) {
            this.pos.y = 1;
            this.dir.y = 1;
            bound = true;
        }
        if (this.pos.x > ctx.canvas.width) {
            this.pos.x = ctx.canvas.width - 1;
            this.dir.x = -1;
            bound = true;
        }
        if (this.pos.y > ctx.canvas.height) {
            this.pos.y = ctx.canvas.height - 1;
            this.dir.y = -1;
            bound = true;
        }
        if (bound) {
            this.rewardSignal = this.rewardSignal + bumpScreenReward;
            negRewards = negRewards + bumpScreenReward;
            this.dir.normalize();
        }
    }

    shoot(agent) {
        if (!ctx || ctx.isDummy) return;
        
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(agent.pos.x, agent.pos.y);
        const sr = agent.eyes[0].sensed_proximity;
        const lineToX = agent.pos.x + sr * agent.dir.x;
        const lineToY = agent.pos.y + sr * agent.dir.y;
        const closestTarget = agent.target;
        
        if (closestTarget) {
            if (closestTarget.isHuman) {
                ctx.strokeStyle = 'orange';
                agent.rewardSignal += hitHumanReward;
                negRewards += hitHumanReward;
                hitShotsHuman += 1;
            }
            else {
                ctx.strokeStyle = 'red';
                !agent.isBit && (agent.rewardSignal += hitShotReward);
                hitShotsBaddy += 1;
            }
            closestTarget.isShot = true;
            closestTarget.currentHp--;
            if (closestTarget.currentHp < 1) {
                closestTarget.ring = 5;
                console.log('killed target' + closestTarget.id + closestTarget.type);
                removeUnit(closestTarget);
            }
        }
        else {
            const r = missedShotReward - Math.min(4,1 * (hitShotsBaddy)/10000);
            agent.rewardSignal += r;
            negRewards +=r;
            missedShots += 1;
            ctx.strokeStyle = 'purple';
        }
        ctx.lineTo(lineToX, lineToY);
        ctx.stroke();
    }

    draw(ctx) {
        if (!ctx || ctx.isDummy) return;
        
        if (this.ring) {
            ctx.beginPath();
            ctx.arc(this.pos.x, this.pos.y, this.ring, 0, Math.PI * 2, false);
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#FF0000';
            ctx.stroke();
        }
        
        let viewedRad = this.rad;
        let image = null;
        var dir = new Vec(this.dir.x, this.dir.y).getUnit();
        
        if (isSprites) {
            const images = this.moveFactor > 0 ? this.moveFrames : this.attackFrames;
            if (images.length > 0) {
                const imageChangeRate = this.moveFactor > 0 ? 1 / 4 : 1;
                const index = Math.floor(((totalTurns + this.id) * imageChangeRate * this.moveFactor)) % images.length;
                image = images[index];
                viewedRad = this.rad * 1.5;
                ctx.globalAlpha = 0.3;
            }
        }
        
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, viewedRad, 0, Math.PI * 2, false);
        ctx.fillStyle = this.getColor();
        ctx.fill();
        ctx.globalAlpha = 1.0;
        
        if (image) {
            const width = this.rad * 4;
            const height = this.rad * 4;
            let centerX = this.pos.x - width / 2;
            let centerY = this.pos.y - height / 2;
            let angle = Math.atan2(dir.y, dir.x);
            drawRotatedImage(ctx, image, centerX, centerY, width, height, angle);
        }
        else {
            ctx.beginPath();
            ctx.moveTo(this.pos.x, this.pos.y);
            ctx.lineTo(this.pos.x + dir.x * this.rad, this.pos.y + dir.y * this.rad);
            ctx.strokeStyle = '#00FFFF';
            ctx.stroke();
        }
    }
}

function drawRotatedImage(ctx, image, x, y, width, height, angle) {
    ctx.save();
    ctx.translate(x + width / 2, y + height / 2);
    ctx.rotate(angle);
    ctx.drawImage(image, -width / 2, -height / 2, width, height);
    ctx.restore();
}

// Environment class
class Env {
    constructor() {
        this.actionSpace = {
            'class': 'Discrete',
            'n': numActions,
        }
        this.observationSpace = {
            'class': 'Box',
            'shape': [numInputs],
            'dtype': 'float32',
        }
        this.resets = 0
        this.i = 0
    }

    async step(action) {
        if (Array.isArray(action)) {
            action = action[0]
        }
        const agentExperienceResult = {
            newObservation: null,
            reward: null
        };
        loopCount++;
        if ((loopCount > batchSize || !continueLoop) && (skipFrames===0 || loopCount % skipFrames === 0)) {
            if (loopCount > batchSize)
                loopCount = 0;

            if (loopCount < 1) {
                updateUI();
            }

            await requestAnimationFrameAsync(async (time) => await mainLoop(time, action, agentExperienceResult));
        } else{
            ctx = {
                isDummy: true,
                beginPath: function () { },
                arc: function () { },
                fill: function () { },
                stroke: function () { },
                canvas: canvas
            };
            if (loopCount >= batchSize * 10) {
                loopCount = 0;
                continueLoop = false;
            }
            await mainLoop(0, action, agentExperienceResult);
        }
        this.i += 1
        return [agentExperienceResult.newObservation, agentExperienceResult.reward, false];
    }

    reset() {
        this.i = 0;
        const states = agents.find(a => a.isHuman)?.states;
        if (states && states.length > 0)
            return states;
        const array = new Array(this.observationSpace.shape[0]).fill(.1);
        return array;
    }
}

function updateUI() {
    if (typeof window !== 'undefined' && window.$) {
        const weights = ppo.actor.getWeights();
        const criticWeights = ppo.critic.getWeights();
        const weightsData = weights.map(weight => weight.dataSync());
        const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

        window.$("#weights").text(weightsData);
        window.$("#criticWeights").text(criticWeightsData);
        window.$("#current-state").text(agents.find(a => a.isHuman)?.states?.join(', ') || '');
        window.$("#hit-shots-human").text(hitShotsHuman);
        window.$("#hit-shots-baddy").text(hitShotsBaddy);
        window.$("#missed-shots").text(missedShots);
        window.$("#turns").text(totalTurns);
    }
}

async function mainLoop(time, action, agentExperienceResult) {
    if (!time) {
        time = Date.now();
    }
    if (!clock.start) clock.start = time;
    if (clock.time) clock.delta = (time - clock.time) / 1000.0;
    clock.time = time;
    if (clock.delta > 0.1) clock.delta = 0.1;
    if (clock.delta < 0.01) clock.delta = 0.01;
    clock.delta *= gameSpeed;
    if (hCnt) clock.total += (clock.delta);

    if (ctx && !ctx.isDummy) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    hCnt = 0;
    zCnt = 0;

    for (let i = 0, l = blocks.length; i < l; i++)
        blocks[i].draw(ctx);
        
    if (totalTurns > 10000 && totalTurns % 1000 === 0) {
        const numZombs = Math.min((totalTurns - 10000) / 4000, maxZombieSpawns);
        for (let i = 0; i < numZombs; i++) {
            addUnit({ type: 'zombie', speed: Math.min(zombieSpeed + totalTurns / 40000, maxZombieSpeed) });
        }
    }
    
    let zombies = agents.filter(agent => agent.isZ);

    if (zombies.length < minZombies) {
        addUnit({ type: 'zombie', pos: zombieHousePos, speed: zombieSpeed });
        zombies = agents.filter(agent => agent.isZ);
    }
    
    for (let i = 0, l = zombies.length; i < l; i++) {
        zCnt++;
        await zombies[i].logic(clock);
        zombies[i].draw(ctx);
    }

    let humans = agents.filter(agent => agent.isHuman);
    if (humans.length < minHumans) {
        const hasLearning = humans.some(h => h.isLearning);
        const block = blocks[Math.floor(Math.random() * blocks.length)];
        addUnit({ type: 'human', pos: new Vec(block.pos.x, block.pos.y)}, !hasLearning );
        humans = agents.filter(agent => agent.isHuman);
    }
    hCnt = humans.length;

    for (let i = 0, l = humans.length; i < l; i++) {
        if (i === 0) {
            humans[i].isLearning = true;
            await humans[i].logic(clock, action, agentExperienceResult);
        }
        else {
            const states = [...(humans[i].states && humans[i].states.length > 0) ? humans[i].states : await humans[i].getStates()];
            const [preds, actionProximal, value, logprobability] = await ppo.getSample(states);
            humans[i].isLearning = false;
            const action = tf.argMax(preds).dataSync()[0];
            const rets = await humans[i].logic(clock, action);

            if (i < 1) {
                humans[i].experiences.push([states, action, rets.reward, value, logprobability]);

                if (ppo.buffer.pointer === 0 && totalTurns > 0) {
                    for (const [states, action, reward, value, logprobability] of humans[i].experiences) {
                        ppo.buffer.add(states, action, reward, value, logprobability);
                    };
                    humans[i].experiences = [];
                }
            }
        }
        humans[i].draw(ctx);
    }

    if (ctx && !ctx.isDummy) {
        ctx.font = '20pt Calibri';
        ctx.lineWidth = 1;
        ctx.fillStyle = 'black';
        var msg = 'Zed:' + zCnt + '     Hum:' + hCnt + '    ' + 'Time:' + Math.floor(clock.total);
        ctx.fillText(msg, ctx.canvas.width / 3 + 1, 21);
        ctx.fillStyle = 'white';
        ctx.fillText(msg, ctx.canvas.width / 3, 20);
        ctx.beginPath();
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#FFFFFF';
        ctx.stroke();
    }
    
    totalTurns++;
    return agentExperienceResult;
}

function requestAnimationFrameAsync(func) {
    return new Promise((resolve) => {
        requestAnimationFrame((time) => {
            resolve(func(time));
        });
    });
}

async function removeUnit(unit) {
    for (const a of agents) {
        a.items = a.items.filter(a => a !== unit);
    }
    agents = agents.filter(a => a !== unit);
}

async function addUnit(config, isLearning = false) {
    ++maxId;
    const a = new Agent({
        id: maxId,
        type: config.type || 'zombie',
        viewDist: 1000,
        pos: config?.pos ? new Vec(config.pos.x, config.pos.y) : new Vec(canvas.width * Math.random(), canvas.height * Math.random()),
        speed: config.speed || null,
    });
    
    if (isLearning)
        agents.unshift(a);
    else
        agents.push(a)
        
    const humans = agents.filter(a => a.isHuman === true);
    humans.forEach(h => {
        h.items = agents.filter(a => a.id !== h.id);
        h.viewDist = 1000;
    });
}

// Load sprite frames
function loadSprites() {
    for (let i = 0; i < 17; i++) {
        zombieMoveFrames[i] = new Image();
        zombieMoveFrames[i].src = `img/skeleton/skeleton-move_${i}.png`;
    }
    for (let i = 0; i < 9; i++) {
        zombieAttackFrames[i] = new Image();
        zombieAttackFrames[i].src = `img/skeleton/skeleton-attack_${i}.png`;
    }
    for (let i = 0; i < 20; i++) {
        survivorMoveFrames[i] = new Image();
        survivorMoveFrames[i].src = `img/survivor/survivor-move_rifle_${i}.png`;
    }
    for (let i = 0; i < 3; i++) {
        survivorShootFrames[i] = new Image();
        survivorShootFrames[i].src = `img/survivor/survivor-shoot_rifle_${i}.png`;
    }
}

// Initialize game
export async function initializeGame(canvasElement) {
    canvas = canvasElement;
    ctx = canvas.getContext('2d');
    zombieHousePos = new Vec(ctx.canvas.width / 2, ctx.canvas.height / 2);
    
    // Load sprites
    loadSprites();
    
    // Create blocks/obstacles
    const squares = createGrid(ctx, 20, 100, 50, 100, 70);
    for (const s of squares) {
        blocks.push(s);
    }
    
    // Create initial agents
    const maxAgents = 53;
    const numHumans = 50;
    for (let i = 0, l = maxAgents; i < l; i++) {
        agents.push(new Agent({
            id: i + 1,
            type: i < numHumans ? 'human' : 'zombie',
            viewDist: 1000,
            pos: i === 0 ? new Vec(canvas.width / 2, canvas.height / 2)
                : new Vec(canvas.width * Math.random(), canvas.height * Math.random()),
        }));
        maxId = i + 1;
    }
    
    const humans = agents.filter(a => a.isHuman === true);
    humans.forEach(h => {
        h.items = agents.filter(a => a.id !== h.id);
        h.viewDist = 1000;
    });
    
    // Initialize PPO
    const env = new Env();
    ppo = new PPO(env, configPpo);
    
    // Start learning
    await ppo.learn({
        'totalTimesteps': Infinity,
        'callback': {
            'onTrainingStart': function (p) {
                console.log('PPO Training Started:', p.config)
            }
        }
    });
}

// Export game control functions
export function toggleContinueLoop() {
    continueLoop = !continueLoop;
}

export function toggleVampire() {
    isVampire = !isVampire;
}

export function setGameSpeed(speed) {
    gameSpeed = speed;
}

export function setShowEyes(value) {
    showEyes = value;
}

export function setSkipFrames(value) {
    skipFrames = value;
}

export function toggleSprites() {
    isSprites = !isSprites;
}

export async function addZombie() {
    await addUnit({ type: 'zombie' });
}

export async function addHuman() {
    await addUnit({ type: 'human' });
}

export async function addVampire() {
    await addUnit({ type: 'vampire' });
}

export function updateRewardConfig(newConfig) {
    Object.assign(rewardConfig, newConfig);
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
    } = rewardConfig);
}

export async function saveModels() {
    if (!ppo) return;
    const name = 'zed-tf'
    await ppo.actor.save('indexeddb://actor-' + name);
    await ppo.critic.save('indexeddb://critic-' + name);
    console.log('Models saved to IndexedDB: ' + name);
}

export async function loadModels() {
    if (!ppo) return;
    try {
        const name = 'zed-tf'
        ppo.actor = await tf.loadLayersModel('indexeddb://actor-' + name);
        ppo.critic = await tf.loadLayersModel('indexeddb://critic-' + name);
        console.log('Models loaded from IndexedDB: ' + name);
    } catch (error) {
        console.log('No saved models found or error loading:', error);
    }
}

// Export game state getters
export function getGameStats() {
    return {
        totalRewards,
        negRewards,
        totalTurns,
        missedShots,
        hitShotsBaddy,
        hitShotsHuman,
        zCnt,
        hCnt,
        rewardOverTime,
        actorLossValues,
        criticLossValues
    };
}
