
/* global $ */
/* global Chart */
/* global PPO */
/* global pi2, randomAngle, fixAngle, Square, Vec, stuff_collide */

/* global tf */
//tf.enableDebugMode();//
tf.setBackend('cpu');

let zCnt = 0;
let hCnt = 0;
let agents = [];
const blocks = [];
const canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
const clock = {
    total: 0,
    start: 0,
    time: 0,
    delta: 0
};
const numEyes = 30;
const numInputs = numEyes * 3;
const actorLossValues = [];
const criticLossValues = [];
let continueLoop = false;
let loopCount = 0;
let states = [];
let negRewards = 0;
let totalRewards = 0;
let rewardOverTime = [];
let totalTurns = 0;
let missedShots = 0;
let hitShotsBaddy = 0;
let hitShotsHuman = 0;
const buildingHumans = 10;
const minHumans = 2;
const minZombies = 3;
const maxZombieSpawns = 50;
const zombieGreen = "#2f402f";
const zombieHousePos = new Vec(ctx.canvas.width / 2, ctx.canvas.height / 2);
const zombieSpeed = 1;
const humanSpeed = 3;
// Hyperparameters
const numActions = 11;
let batchSize = +$('#slider-batch').val();
let epsilonGreedy = 0;
const learningRate = .001;

const baseReward = 1;// bites and hit shots
const biteReward = baseReward * -1;
const hitHumanReward = baseReward * -1;
const missedShotReward = baseReward * -.125;//.125;
const bumpWallReward = baseReward * -.55;
const bumpScreenReward = baseReward * -.75;
const bumpHumanReward = baseReward * -.5;

// these rewards don't seem to do much even at higher numbers. i think they're too confusing
const blockedVisionHuman = baseReward * -.5;
const blockedVisionWall = baseReward * -.25;
// these spawn unpredictably so the game would have to be tweaked for this to be of much help
const zombieProximityReward = 0;//baseReward * -.25;


let isSprites = true;
let showEyes = 0;
const eyeMaxRange = 1000;

let gameSpeed = 4;


const configPpo = {
    nSteps: batchSize,                 // Number of steps to collect rollouts
    nEpochs: 10,                 // Number of epochs for training the policy and value networks
    policyLearningRate: learningRate,    // Learning rate for the policy network
    valueLearningRate: learningRate,     // Learning rate for the value network
    clipRatio: 0.2,              //.2- PPO clipping ratio for the objective function
    targetKL: 0.02,            // .01-Target KL divergence for early stopping during policy optimization
    netArch: {
        'pi': [numInputs, numInputs],           // Network architecture for the policy network
        'vf': [numInputs, numInputs]           // Network architecture for the value network
    },
    activation: 'tanh',          //relu, elu Activation function to be used in both policy and value networks
    verbose: 0                 // cm-does this do anything? - Verbosity level (0 for no logging, 1 for logging)
}
const zombieMoveFrames = [];
const zombieAttackFrames = [];
const survivorMoveFrames = [];
const survivorShootFrames = [];
for (let i = 0; i < 17; i++) {  // Replace 10 with the number of images you have
    zombieMoveFrames[i] = new Image();
    zombieMoveFrames[i].src = `img/skeleton/skeleton-move_${i}.png`;  // Adjust the path and filename as needed
}
for (let i = 0; i < 9; i++) {  // Replace 10 with the number of images you have
    zombieAttackFrames[i] = new Image();
    zombieAttackFrames[i].src = `img/skeleton/skeleton-attack_${i}.png`;  // Adjust the path and filename as needed
}
for (let i = 0; i < 20; i++) {  // Replace 10 with the number of images you have
    survivorMoveFrames[i] = new Image();
    survivorMoveFrames[i].src = `img/survivor/survivor-move_rifle_${i}.png`;  // Adjust the path and filename as needed
}
for (let i = 0; i < 3; i++) {  // Replace 10 with the number of images you have
    survivorShootFrames[i] = new Image();
    survivorShootFrames[i].src = `img/survivor/survivor-shoot_rifle_${i}.png`;  // Adjust the path and filename as needed
}


//karpathy's eye
var Eye = function (angle) {
    this.angle = angle; // angle relative to agent its on
    this.max_range = eyeMaxRange;
    this.sensed_proximity = eyeMaxRange;
    this.sensed_type = 0; // what does the eye see?

}
function Agent(config) {
    this.id = config.id;
    this.experiences = [];

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
    if (this.isHuman)
        for (var k = 0; k < eyeCount; k++) { this.eyes.push(new Eye(k * rads)); }

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
    //todo: remove duplicate position
    Object.defineProperty(this, 'angle', {
        get: function () {
            return new Vec(this.dir.x, this.dir.y);
        }
    });
    Object.defineProperty(this, 'p', {
        get: function () {
            return new Vec(this.pos.x, this.pos.y);
        }
    });
    this.v = new Vec(0, 0);
    //
    this.state = config.state || 'idle';
    this.viewDist = config.viewDist || 1000;
    this.viewFov = (config.viewFov || Math.PI / 2);
    this.viewFovD2 = this.viewFov / 2;
    this.nextTimer = Math.random() * 10;
    this.ring = config.ring || this.type === 'human' ? 0 : 5;
}

Agent.prototype.getVision = function () {

    let eyeStates = [];
    var a = this;
    a.target = null;
    const pos = a.p;
    const angle = a.angle;
    let closestZombieRange = eyeMaxRange;
    for (var ei = 0, ne = a.eyes.length; ei < ne; ei++) {
        var e = a.eyes[ei];
        e.sensed_proximity = e.max_range;
        e.sensed_type = 0;
        const currentEyeAnglePointing = angle.rotate(e.angle).getUnit();
        // we have a line from p to p->eyep
        var eyep = new Vec(pos.x + e.max_range * currentEyeAnglePointing.x,
            pos.y + e.max_range * currentEyeAnglePointing.y);

        if (isNaN(eyep.x)) {
            console.error('eyep.x is NaN');
        }

        var res = stuff_collide(a, eyep, blocks, true, true);
        if (res) {
            // eye collided with anything
            if (ei === 0)
                a.target = res.agent;
            e.sensed_proximity = res.up.distFrom(a.p);
            e.sensed_type = res.type;
        }

        ctx.strokeStyle = "rgb(0,0,0,0)";
        if (showEyes > 1) {
            ctx.strokeStyle = "rgb(255,150,150)";
            if (e.sensed_type === -.1) { ctx.strokeStyle = "yellow"; }// wall
            else if (e.sensed_type === 0) { ctx.strokeStyle = "rgb(200,200,200)"; } //nothing
        }
        if (showEyes > 0) {
            if (e.sensed_type === 1) { ctx.strokeStyle = "yellow"; } // human
            else if (e.sensed_type === -1) { ctx.strokeStyle = "rgb(150,255,150)"; } // z
            if (e.sensed_type === -1) { ctx.strokeStyle = "green"; } // z
        }

        const sr = e.sensed_proximity;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        const lineToX = pos.x + sr * currentEyeAnglePointing.x;
        const lineToY = pos.y + sr * currentEyeAnglePointing.y;
        ctx.lineTo(lineToX, lineToY);
        ctx.stroke();

        let type = e.sensed_type;
        // clip whether it cares about walls that are far away
        if (type === -.1 && sr < 100)
            type = 0;
        if (type === 'zombie')
            type = -1;
        if (type === 'human')
            type = 1;

        if (ei === 0) {
            if (e.sensed_type === 1)
                this.rewardSignal = this.rewardSignal + blockedVisionHuman;
            if (e.sensed_type === -.1)
                this.rewardSignal = this.rewardSignal + (blockedVisionWall * (1 - e.sensed_proximity / e.max_range));
        }
        if (e.sensed_type === -1 && e.sensed_proximity < closestZombieRange) {
            closestZombieRange = e.sensed_proximity;

        }
        // add to state for ML
        // tensorflow inputs
        // we may only need distance here, and let the nn figure out the rest
        // based on which eye sense the object
        eyeStates.push(sr * currentEyeAnglePointing.x / e.max_range, sr * currentEyeAnglePointing.y / e.max_range, type);
    }
    this.rewardSignal = this.rewardSignal + zombieProximityReward * (1 - e.sensed_proximity / e.max_range);

    // tensorflow inputs
    return eyeStates;
}

Agent.prototype.getColor = function () {
    if (this.isLearning) return 'blue';
    if (this.state === 'mouse') return '#FF00FF';
    if (this.state === 'attack') return 'red';
    if (this.isHuman) return 'purple';
    if (this.isZ) return 'green';
    return '#AAAAAA';
};

Agent.prototype.see = function () {
    var seen = [];
    var a, d, ato;
    for (var i = 0, l = agents.length; i < l; i++) {
        // check if what we see is blocked by a wall   
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
Agent.prototype.getStates = async function () {
    const vision = [...this.getVision()];
    return vision;
}
Agent.prototype.zombify = async function (victim, zombie) {
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

Agent.prototype.logic = async function (clock, action, states, agentExperienceResult) {
    this.moveFactor = this.isShot ? .1 : 1;
    var batchValue = $('#slider-batch').val();
    var epsilonValue = 0;//$('#slider-epsilon').val();

    epsilonGreedy = +epsilonValue;
    batchSize = +batchValue;
    // // Setters
    // $('#slider-batch').val(8); 
    // $('#slider-lr').val(0.01); 
    // $('#slider-epsilon').val(0.2); 


    if (this.isHuman === false) {
        let inMelee = false;
        this.state = 'idle';

        const seen = this.see();
        const seeHuman = seen.find(s => s.agent.isHuman);
        if (seeHuman) {
            this.state = 'attack';
            this.nextTimer = 5;
            this.newDir = seeHuman.angle;
            this.dir = seeHuman.angle;
        }

        for (let i = 0, l = seen.length; i < l; i++) {
            if (seen[i].dist <= this.rad * 2) {
                // bite and maybe convert human to zombie
                this.moveFactor = 0;
                if (seen[i].agent.isHuman) {
                    const human = seen[i].agent;
                    this.moveFactor = 0;
                    inMelee = true;
                    --human.currentHp;
                    human.isBit = true;

                    if (isVampire || this.isVampire)
                        ++this.currentHp / 10;

                    //tf ml reward
                    human.rewardSignal = human.rewardSignal + biteReward;
                    negRewards = negRewards + biteReward;
                    if (human.currentHp < 1)
                        await this.zombify(human, this);

                }
            }
        }

        this.rad = Math.max(this.minRad, this.minRad + this.currentHp - this.maxHp);
        // try wandering if its stuck
        if (!inMelee && this.moveFactor === 0) {
            if (this.nextTimer <= 0) {
                this.state = 'idle';
                this.nextTimer = 3 + Math.random() * 10;
                this.newDir = randomAngle();
                this.dir = this.newDir;
                this.moveFactor = .2;
            }
        }
        // follow other zombie
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

        let actionSelected;
        if (Math.random() < epsilonGreedy) {
            // Take a random action
            actionSelected = Math.floor(Math.random() * numActions);
            const actionsOneHot = Array(numActions).fill(.1);
            actionsOneHot[actionSelected] = .8;
        } else {
            actionSelected = action;
        }

        let newAngle = 0;
        const numAngles = numActions - 2;
        if (actionSelected < numAngles) {
            newAngle = (actionSelected - (Math.floor(numAngles / 2))) * (2 * Math.PI / this.eyes.length);
        } else if (actionSelected === (numActions - 2)) {
            newAngle = Math.PI;
        }
        else {
            this.moveFactor = 0;
            this.shoot(this);
        }
        // stop if collided with another human
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
    // get velociy
    var vx = this.dir.x * speed * clock.delta;
    var vy = this.dir.y * speed * clock.delta;
    // move
    this.pos.x += vx;
    this.pos.y += vy;
    // prevent walking through blocks
    for (let i = 0, l = blocks.length; i < l; i++) {
        if (this.intersect = blocks[i].rayIntersect(this.pos, this.dir)) {
            if (this.intersect[0].dist <= 0 && this.intersect[1].dist > 0) {
                this.pos = this.intersect[0].pos;
                this.rewardSignal = this.rewardSignal + bumpWallReward
                negRewards = negRewards + bumpWallReward;
                //this.newDir = this.intersect[0].n;
                this.dir = randomAngle();
                break;
            }
            else {
                this.intersect = false;
            }
        }
    }

    // if we hit a wall turn arround
    this.CheckScreenBounds();

    if (this.isHuman === true && states.length > 0) {
        states = await this.getStates();
        const ret = {
            newObservation: states,
            reward: this.rewardSignal,
            done: false
        }
        if (agentExperienceResult) {
            agentExperienceResult.newObservation = states;
            agentExperienceResult.reward = this.rewardSignal;
        }

        totalRewards += this.rewardSignal;
        rewardOverTime.push(this.rewardSignal);
        $("#rewardTotal").text(totalRewards);
        $("#neg-rewards").text(negRewards);
        this.rewardSignal = 0;
        return ret;
    }
}
Agent.prototype.CheckScreenBounds = function () {
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
Agent.prototype.shoot = (agent) => {

    // Draw red line to the closest baddy
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
            !agent.isBit && (agent.rewardSignal += baseReward);
            hitShotsBaddy += 1;
        }


        closestTarget.isShot = true;
        closestTarget.currentHp--;
        closestTarget.state = 'idle';
        if (closestTarget.currentHp < 1) {
            closestTarget.ring = 5;
            console.log('killed target' + closestTarget.id + closestTarget.type);
            //todo: remove or create some kinda goodie
            // this will remove humans too, but w
            removeUnit(closestTarget);
        }
    }
    else {

        // missed! purple line is missed shot. the agent did not move but shot nothing.
        // to do: for now, we disgourage it from stopping and missing.
        agent.rewardSignal = agent.rewardSignal + missedShotReward;
        negRewards = negRewards + missedShotReward;
        missedShots += 1;
        ctx.strokeStyle = 'purple';
    }
    ctx.lineTo(lineToX, lineToY);
    ctx.stroke();

}
Agent.prototype.draw = function (ctx) {

    if (this.ring) {
        ctx.beginPath();
        ctx.arc(this.pos.x, this.pos.y, this.ring, 0, pi2, false);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#FF0000';
        ctx.stroke();
    }
    let viewedRad = this.rad;
    let image = null;
    var dir = new Vec(this.dir.x, this.dir.y).getUnit();
    if (isSprites) {    
        const images = this.moveFactor > 0 ? this.moveFrames : this.attackFrames;
        const imageChangeRate = this.moveFactor > 0 ? 1 / 4 : 1;
        if (images.length > 0) {
            const index = Math.floor(((totalTurns + this.id) * imageChangeRate * this.moveFactor)) % images.length;
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
    if(image){
        const width = this.rad * 4;
        const height = this.rad * 4;
        let centerX = this.pos.x - width / 2;
        let centerY = this.pos.y - height / 2;
        let angle = Math.atan2(dir.y, dir.x);             
        drawRotatedImage(ctx, image, centerX, centerY, width, height, angle);
        
    }
    else {
        // ctx.lineWidth = 2;
        // ctx.strokeStyle = '#FFFFFF';
        // ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(this.pos.x, this.pos.y);
        ctx.lineTo(this.pos.x + dir.x * this.rad, this.pos.y + dir.y * this.rad);
        ctx.strokeStyle = '#00FFFF';
        ctx.stroke();
    }
};
function drawRotatedImage(ctx, image, x, y, width, height, angle) {
    ctx.save();
    ctx.translate(x + width / 2, y + height / 2);
    ctx.rotate(angle);
    ctx.drawImage(image, -width / 2, -height / 2, width, height);
    ctx.restore();
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

    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hCnt = 0;
    zCnt = 0;

    for (let i = 0, l = blocks.length; i < l; i++)
        blocks[i].draw(ctx);
    if (totalTurns > 10000 && totalTurns % 1000 === 0) {
        const numZombs = Math.min((totalTurns - 10000) / 2000, maxZombieSpawns);
        for (let i = 0; i < numZombs; i++) {
            addUnit({ type: 'zombie', pos: zombieHousePos, speed: Math.min(zombieSpeed + totalTurns / 40000, 3) });
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
        // get random building, human comes out of it
        const block = blocks[Math.floor(Math.random() * blocks.length)];
        addUnit({ type: 'human', pos: new Vec(block.pos.x, block.pos.y) });
        humans = agents.filter(agent => agent.isHuman);
    }
    hCnt = humans.length;

    for (let i = 0, l = humans.length; i < l; i++) {
        if (i === 0) {
            humans[i].isLearning = true;
            const states = await humans[0].getStates();
            await humans[i].logic(clock, action, states, agentExperienceResult);
        }
        else {
            // these are the other humans. they use the best action, rather than the proximal action
            const states = await humans[i].getStates();
            const [preds, actionProximal, value, logprobability] = await ppo.getSample(states);
            humans[i].isLearning = false;
            const action = tf.argMax(preds).dataSync()[0];
            const rets = await humans[i].logic(clock, action, states);
            if (i < 0) {
                // hack to add argMax agent to add to buffer in seequence
                humans[i].experiences.push([states,
                    action,
                    rets.reward,
                    value,
                    logprobability]);

                if (ppo.buffer.pointer === 0 && totalTurns > 0) {
                    for (const [states,
                        action,
                        reward,
                        value,
                        logprobability] of humans[i].experiences) {
                        ppo.buffer.add(
                            states,
                            action,
                            reward,
                            value,
                            logprobability);
                    };
                    humans[i].experiences = [];

                }
            }
        }
        humans[i].draw(ctx);
    }

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
    totalTurns++;
    $("#hit-shots-human").text(hitShotsHuman);
    $("#hit-shots-baddy").text(hitShotsBaddy);
    $("#missed-shots").text(missedShots);
    $("#turns").text(totalTurns);
    return agentExperienceResult;
}
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
        if (loopCount > batchSize || !continueLoop) {
            if (loopCount > batchSize)
                loopCount = 0;

            if (loopCount < 1) {
                const weights = ppo.actor.getWeights();
                const criticWeights = ppo.critic.getWeights();
                const weightsData = weights.map(weight => weight.dataSync());
                const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

                $("#weights").text(weightsData);
                $("#criticWeights").text(criticWeightsData);
                $("#current-state").text(states.join(', '));

                createOrUpdateRewardChart(rewardOverTime, batchSize)
                //await createOrUpdateLossesChart();

            }

            ctx = canvas.getContext('2d');
            await requestAnimationFrameAsync(async (time) => await mainLoop(time, action, agentExperienceResult));
        } else if (continueLoop) {
            // don't draw. just keep going and train the model
            ctx = {
                isDummy: true,
                beginPath: function () { },
                arc: function () { },
                fill: function () { },
                stroke: function () { },
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
        const states = agents.find(a => a.isHuman).getStates();
        if (states.length > 0)
            return states;
        const array = new Array(this.observationSpace.shape[0]).fill(.1);
        return array;
    }
}
function requestAnimationFrameAsync(func) {
    return new Promise((resolve) => {
        requestAnimationFrame((time) => {
            resolve(func(time));
        });
    });
}

let lossesChart;
async function createOrUpdateLossesChart() {
    const ctx = document.getElementById('losses-chart').getContext('2d');
    if (!lossesChart) {
        lossesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: actorLossValues.map((_, i) => i + 1),
                datasets: [{
                    label: 'Actor Loss 1',
                    data: actorLossValues,
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }, {
                    label: 'Critic Loss 1',
                    data: criticLossValues,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },]
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
        lossesChart.data.labels = actorLossValues.map((_, i) => i + 1);
        lossesChart.data.datasets[0].data = actorLossValues;
        lossesChart.data.datasets[1].data = criticLossValues;
        lossesChart.update();
    }
}

let chart; // Declare chart variable outside the function

function createOrUpdateRewardChart(rewardOverTime, batchSize) {
    let ctx = document.getElementById('rewardOverTimeChart').getContext('2d');

    // Calculate average reward for each batch
    let avgRewards = [];
    for (let i = 0; i < rewardOverTime.length; i += batchSize) {
        let batch = rewardOverTime.slice(i, i + batchSize);
        let batchAvg = batch.reduce((a, b) => a + b, 0) / batch.length;
        avgRewards.push(batchAvg);
    }

    if (!chart) {
        // If the chart does not exist, create it
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: avgRewards.map((_, i) => i), // X-axis labels are just indices
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
        // If the chart already exists, update its data
        chart.data.labels = avgRewards.map((_, i) => i);
        chart.data.datasets[0].data = avgRewards;
        chart.update();
    }
}
async function removeUnit(unit) {
    for (const a of agents) {
        a.items = a.items.filter(a => a !== unit);
    }
    agents = agents.filter(a => a !== unit);
}
async function addUnit(config) {
    ++maxId;
    const a = new Agent({
        id: maxId,
        type: config.type || 'zombie',
        viewDist: 1000,
        pos: config.pos || new Vec(canvas.width * Math.random(), canvas.height * Math.random()),
        speed: config.speed || null,

    });
    agents.push(a);
    const humans = agents.filter(a => a.isHuman === true);
    humans.forEach(h => {
        h.items = agents.filter(a => a.id !== h.id);
        h.viewDist = 1000;
    });
}
let maxId = 0;
let ppo = null;
let isVampire = false;
window.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
        continueLoop = !continueLoop;
    }
});
window.addEventListener('keydown', (event) => {
    if (event.code === 'KeyV') {
        isVampire = !isVampire;
    }
});
window.addEventListener('keydown', (event) => {
    if (event.code === 'KeyL') {
        loadModels();
    }
});
window.addEventListener('keydown', (event) => {
    if (event.code === 'KeyS') {
        saveModels();
    }
});
$('#vampire-button').click(function () {
    isVampire = !isVampire;
});
$('#add-vampire-button').click(function () {
    addUnit({ type: 'vampire' });
});
$('#rush-watch-button').click(function () {
    continueLoop = !continueLoop;
});

$('#save-button').click(async function () {
    await saveModels();
});

$('#smith-button').click(async function () {
    const li = agents.findIndex(a => a.isLearning);
    const yi = agents.findIndex(a => !a.isLearning && a.isHuman);
    const l = agents[li];
    const y = agents[yi];
    l.isLearning = false;
    y.isLearning = true;
    agents[li] = y;
    agents[yi] = l;
});
$('#load-button').click(async function () {
    await loadModels();
});
$('#add-zombie-button').click(async function () {
    await addUnit({ type: 'zombie' });
});
$('#add-human-button').click(async function () {
    await addUnit({ type: 'human' });
});
$('#toggle-sprites').click(async function () {
    isSprites = !isSprites;
});
const loadModels = async () => {
    const name = 'zed-tf'
    ppo.actor = await tf.loadLayersModel('indexeddb://actor-' + name);
    ppo.critic = await tf.loadLayersModel('indexeddb://critic-' + name);
    //const name = 'zed-tf';
    // ppo.actor = await tf.loadLayersModel('indexeddb://zed-tf-actor-current');
    // ppo.critic = await tf.loadLayersModel('indexeddb://zed-tf-critic-current');
}
const saveModels = async () => {
    const name = 'zed-tf'
    await ppo.actor.save('indexeddb://actor-' + name);
    await ppo.critic.save('indexeddb://critic-' + name);
    console.log('saved to indexxed db: ' + name);
    await saveToFiles(name);

}
const saveToFiles = async () => {

    await ppo.actor.save('downloads://actor-' + name);
    await ppo.critic.save('downloads://critic-' + name);
}
const loadModelFiles = async () => {
    const actorInput = document.getElementById('actorInput');
    const criticInput = document.getElementById('criticInput');

    const actorFile = actorInput.files[0];
    const criticFile = criticInput.files[0];

    const actorUrl = URL.createObjectURL(actorFile);
    const criticUrl = URL.createObjectURL(criticFile);

    ppo.actor = await tf.loadLayersModel(actorUrl);
    ppo.critic = await tf.loadLayersModel(criticUrl);
}
//const zombieSpeedElement = $('#zombieSpeed');
//zombieSpeedElement.addEventListener('change', updateZombieSpeed);
//let zombieSpeed
//$('#maxZombieSpeed').addEventListener('change', updateMaxZombieSpeed);
//$('#radius').addEventListener('change', updateRadius);
// document.getElementById('actorInput').addEventListener('change', loadActorModel);
// document.getElementById('criticInput').addEventListener('change', loadCriticModel);

function updateZombieSpeed(event) {
    zombieSpeed = event.target.value;
    // Update minZombieSpeed in your application
}

function updateMaxZombieSpeed(event) {
    maxZombieSpeed = event.target.value;
    // Update maxZombieSpeed in your application
}

function updateRadius(event) {
    radius = event.target.value;
    // Update radius in your application
}
$('#show-eyes').on('input', function () {
    showEyes = +$(this).val();
});
(async function () {

    $('#gameSpeed').change(function () {
        gameSpeed = $(this).val();
    }).val(gameSpeed);

    $("#help").click(function () {
        $(this).hide();
    });


    // eslint-disable-next-line no-unused-vars
    const windowArea = $(window).width() * $(window).height();

    const blockNum = windowArea / 50000;
    const squares = createGrid(ctx, 20, 100, 50, 100, 70);
    for (const s of squares) {
        blocks.push(s);
    }
    // for (let i = 0, l = blockNum; i < l; i++) {
    //     blocks.push(new Square({}, ctx));
    // }
    // const zombieHouseConfig = {
    //     fill: zombieGreen,
    //     stroke: 'olive',

    //     pos: zombieHousePos,
    //     width: ctx.canvas.height / 10,
    //     height: ctx.canvas.height / 10,
    // }
    // blocks.push(new Square(zombieHouseConfig, ctx));
    // not sure how the height works for drawing the square but I'll wing it
    const riverConfig1 = {
        fill: "blue",
        stroke: 'navy',

        pos: {
            x: (ctx.canvas.width / 2),
            y: 0,
        },
        width: ctx.canvas.width / 50,
        height: ctx.canvas.height - 10,
    }
    const riverConfig2 = {
        fill: "blue",
        stroke: 'navy',

        pos: {
            x: (ctx.canvas.width / 2),
            y: ctx.canvas.height + 10,
        },
        width: ctx.canvas.width / 50,
        height: ctx.canvas.height,
    }
    //blocks.push(new Square(riverConfig1));
    //blocks.push(new Square(riverConfig2));
    const maxAgents = 53;//windowArea / 10000;
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
    const env = new Env();
    ppo = new PPO(env, configPpo);
    await ppo.learn({
        'totalTimesteps': 10000000,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config)
            }
        }
    })
})();

//assets - zombies and survivors : https://opengameart.org/content/animated-top-down-zombie

