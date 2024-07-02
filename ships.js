
/* global $ */
/* global Chart */
/* global PPO */
/* global line_point_intersect, Vec */

/* global tf */
//tf.enableDebugMode();//
tf.setBackend('cpu');
const shipsConfigBots = {
    numShips: 20,
    numTeams: 4,
    isBot: true,
    isLearner: false
}
let totalTurns = 0;
let skipFrames = 0;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let rewardSignals = [];
const negrewards = [];
let beams = [];

const draw = (ships) => {
    const now = Date.now();
    
    // Draw and filter beams in one pass
    beams = beams.filter(beam => {
        const { start, end, color, width, time, lifeTime } = beam;
        const age = now - time;
        const opacity = 1 - age / lifeTime;
        
        if (opacity <= 0) return false;
        
        ctx.lineWidth = width * opacity;
        ctx.strokeStyle = color;
        ctx.globalAlpha = opacity;
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
        
        return true;
    });
    
    ctx.globalAlpha = 1;

    // Reusable objects for ship drawing
    const p1 = {}, p2 = {}, p3 = {};

    ships.forEach(ship => {
        // Draw the shield
        ctx.beginPath();
        ctx.arc(ship.position.x, ship.position.y, ship.shield.radius, 0, Math.PI * 2);
        ctx.fillStyle = getColor(ship.hp, 0, ship.maxHp, 0.3);
        ctx.fill();

        // Calculate ship triangle points
        const height = ship.shield.radius / 2;
        const base = 4 * height;
        p1.x = -height / 2; p1.y = -base / 2;
        p2.x = -height / 2; p2.y = base / 2;
        p3.x = height / 2;  p3.y = 0;

        rotatePoint(p1, ship.direction, ship.position);
        rotatePoint(p2, ship.direction, ship.position);
        rotatePoint(p3, ship.direction, ship.position);

        // Draw the ship
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.closePath();
        ctx.fillStyle = ship.color;
        ctx.fill();
    });

    // const learner = ships.find(a => a.isLearner);
    // if (learner) {
    //     // Draw learner direction
    //     ctx.strokeStyle = 'purple';
    //     ctx.beginPath();
    //     ctx.moveTo(learner.position.x, learner.position.y);
    //     ctx.lineTo(
    //         learner.position.x + learner.direction.x * 100,
    //         learner.position.y + learner.direction.y * 100
    //     );
    //     ctx.stroke();

    //     // Draw lines to other ships
    //     ctx.strokeStyle = 'green';
    //     const otherShips = learner.getOrderedShips();
    //     otherShips.forEach(ship => {
    //         ctx.beginPath();
    //         ctx.moveTo(learner.position.x, learner.position.y);
    //         ctx.lineTo(ship.position.x, ship.position.y);
    //         ctx.stroke();
    //     });
    // }
}

// Assume this helper function is defined elsewhere
function rotatePoint(point, direction, center) {
    const cos = direction.x;
    const sin = direction.y;
    const x = point.x;
    const y = point.y;
    point.x = x * cos - y * sin + center.x;
    point.y = x * sin + y * cos + center.y;
}
const drawOld = (ships) => {
    // Draw the beams   
    beams.forEach(beam => {
        const { start, end, color, width, time, lifeTime } = beam;
        const age = Date.now() - time;
        const opacity = 1 - age / lifeTime;
        if (opacity <= 0) return;
        ctx.lineWidth = width * (1 - age / lifeTime);
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.globalAlpha = opacity;
        ctx.stroke();
        ctx.globalAlpha = 1;
    });
    // remove beam if opacity is 0
    beams.forEach((beam, i) => {
        if (beam.opacity === 0) {
            beams.splice(i, 1);
        }
    });
    ships.forEach(ship => {
        // Draw the shield
        ctx.beginPath();
        ctx.arc(ship.position.x, ship.position.y, ship.shield.radius, 0, Math.PI * 2, false);

        // Color the shield
        ctx.fillStyle = getColor(ship.hp, 0, ship.maxHp, .3);
        ctx.fill();
        let height = ship.shield.radius / 2;
        let base = 4 * height;

        let p1 = { x: ship.position.x - height / 2, y: ship.position.y - base / 2 };
        let p2 = { x: ship.position.x - height / 2, y: ship.position.y + base / 2 };
        let p3 = { x: ship.position.x + height / 2, y: ship.position.y };

        p1 = rotatePoint(p1, ship.direction, ship.position);
        p2 = rotatePoint(p2, ship.direction, ship.position);
        p3 = rotatePoint(p3, ship.direction, ship.position);

        // Draw the triangle
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.closePath();

        ctx.fillStyle = ship.color;
        ctx.fill();

    });
    const learner = ships.find(a => a.isLearner);
    const otherShips = learner.getOrderedShips();
    // draw purple line in learner.direction
    ctx.beginPath();
    ctx.strokeStyle = 'purple';
    ctx.moveTo(learner.position.x, learner.position.y);
    ctx.lineTo(learner.position.x + learner.direction.x * 100, learner.position.y + learner.direction.y * 100);
    ctx.stroke();
    otherShips.forEach(ship => {
        ctx.beginPath();
        ctx.strokeStyle = 'green';
        ctx.moveTo(learner.position.x, learner.position.y);
        ctx.lineTo(ship.position.x, ship.position.y);
        ctx.stroke();
    });

}

function createShip(id, team, maxHp, acceleration, isBot, isLearner, color, baseDiameter, direction, position, velocity) {
    return {
        id: id,
        states: [],
        rewardSignal: 0,
        radius: baseDiameter / 2,
        team: team || 0,
        maxHp: maxHp || 500,
        hp: maxHp || 500,
        shape: 'triangle',
        color: color,
        shield: {
            shape: 'circle',
            radius: baseDiameter / 2,
            position: 'center'
        },
        direction: direction,
        position: position,
        velocity: velocity,
        acceleration: acceleration || .001,
        isBot: isBot,
        isLearner: isLearner,
        getObservation: function () {
            this.states = this.getStates();
            return {
                newObservation: this.states,
                reward: this.rewardSignal,
                done: false
            }
        },
        getOrderedShips: function(){
            return ships.filter(x => x.id !== this.id).sort((a, b) => {
                // Sort by team
                if (a.team < b.team) return -1;
                if (a.team > b.team) return 1;

                // If team is the same, sort by id
                if (a.id < b.id) return -1;
                if (a.id > b.id) return 1;

                return 0; // equal
            });
        },
        getStates: function () {
            const velocityDenominator = 100;
            const positionDenominator = Math.max(canvas.width, canvas.height);
            const selfShip = [
                this.direction.x,
                this.direction.y,
                this.velocity.x / velocityDenominator,
                this.velocity.y / velocityDenominator,
                this.position.x / positionDenominator,
                this.position.y / positionDenominator,              
                this.hp / this.maxHp,

            ];
            const otherShips = this.getOrderedShips();

            for (const ship of otherShips) {
                const otherShip = [
                    ship.direction.x,
                    ship.direction.y,
                    (ship.velocity.x - this.velocity.x) / velocityDenominator,
                    (ship.velocity.y - this.velocity.y) / velocityDenominator,
                    (ship.position.x - this.position.x) / positionDenominator,
                    (ship.position.y - this.position.y) / positionDenominator,              
                    ship.hp / ship.maxHp,
                    //ship.team===this.team?1:-1,
                ];
                selfShip.push(...otherShip);
            }
            return selfShip;

        },
        reflectOffScreen: function () {
            const radius = this.radius;
            const maxSpeed = 10;
            this.velocity = new Vec(this.velocity);
            const speed = this.velocity.length();
            if (speed >= maxSpeed)
                this.velocity.scale(maxSpeed / speed);
        
            if (this.position.x - radius < 0) {
                this.velocity.x = -this.velocity.x; // Invert velocity
                this.position.x = radius; // Correct position to just inside the boundary
            } else if (this.position.x + radius > canvas.width) {
                this.velocity.x = -this.velocity.x; // Invert velocity
                this.position.x = canvas.width - radius; // Correct position to just inside the boundary
            }
            if (this.position.y - radius < 0) {
                this.velocity.y = -this.velocity.y; // Invert velocity
                this.position.y = radius; // Correct position to just inside the boundary
            } else if (this.position.y + radius > canvas.height) {
                this.velocity.y = -this.velocity.y; // Invert velocity
                this.position.y = canvas.height - radius; // Correct position to just inside the boundary
            }
        },
        move: function () {
            this.reflectOffScreen();
            let dir = this.direction;
            this.velocity.x += dir.x * this.acceleration;
            this.velocity.y += dir.y * this.acceleration;
            this.position.x += this.velocity.x;
            this.position.y += this.velocity.y;
        },
        getClosestEnemy: function (ships) {
            let closestDistance = Infinity;
            let closestShip = null;
            for (let ship of ships) {
                if (ship.team !== this.team) {
                    let dx = this.position.x - ship.position.x;
                    let dy = this.position.y - ship.position.y;
                    let distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestShip = ship;
                    }
                }
            }
            return closestShip;
        },
        turnTowardsEnemy: function (ships) {
            // sometimes we already have picked a target
            let closestEnemy = ships.length < 2 ? ships?.[0] : this.getClosestEnemy(ships);
            if (closestEnemy) {
                const maxTurnRate = Math.PI / 16;
                let dx = closestEnemy.position.x - this.position.x;
                let dy = closestEnemy.position.y - this.position.y;
                let targetAngle = Math.atan2(dy, dx);
                let currentAngle = Math.atan2(this.direction.y, this.direction.x);
                let deltaAngle = targetAngle - currentAngle;
                deltaAngle = ((deltaAngle + Math.PI) % (2 * Math.PI)) - Math.PI;
                deltaAngle = Math.max(-maxTurnRate, Math.min(maxTurnRate, deltaAngle));

                // Add the deltaAngle to the current angle to get the new angle
                let newAngle = currentAngle + deltaAngle;
                this.direction.x = Math.cos(newAngle);
                this.direction.y = Math.sin(newAngle);
            }
        },
        shoot: function (d, enemies) {
            const endPoint = new Vec(
                this.position.x + this.direction.x * d,
                this.position.y + this.direction.y * d);

            for (let enemy of enemies) {
                const { up } = line_point_intersect(new Vec(this.position.x, this.position.y), endPoint,
                    new Vec(enemy.position.x, enemy.position.y), enemy.radius);
                if (up) {

                    enemy.hp -= 1;
                    this.handleDingedReward(enemy);
                    if (!this.isBot) {
                        // the learner gets a reward for hitting the enemy
                        // learners team adds the reward to the array as well
                        this.handleHitReward(enemy);
                    }
                    beams.push({
                        start: { ...this.position },
                        end: up,
                        color: this.color,
                        time: Date.now(),
                        width: 3,
                        lifeTime: 1500, //seconds

                    });
                    break;
                }
            }
        },
        handleHitReward: function (ship) {
            const r = 2 + (1 - ship.hp / ship.maxHp);
            this.rewardSignal += r;
            //rewardSignals.push(r);
            //rewardSignals.push(1);
        },
        handleDingedReward: function (ship) {
            //const r = 1 - ship.hp / ship.maxHp;
            //ship.rewardSignal -= r;

        },
    }
}

const getColor = (value, min, max, opacity = .5) => {
    let normalizedValue = (value - min) / (max - min);
    let hue = normalizedValue * 240;
    return `hsla(${hue}, 100%, 50%, ${opacity})`;
}
function createShips(shipConfig) {
    const { numShips, numTeams } = shipConfig;
    const teams = Array.from({ length: numTeams }, (_, i) => getColor(i, 0, numTeams, 1));
    const sectionWidth = canvas.width / numTeams;
    const ships = Array.from({ length: numShips }, (_, i) => {
        const teamIndex = (i % numTeams);
        
        const acceleration = .01;
        const isBot = teamIndex != 0;
        const maxHp = isBot ? 500 : 100000;
        const isLearner = i === 0;
        const color = isLearner ? "blue" : teams[teamIndex];
        const baseDiameter = 50;
        const id = i + 1;
        const position = {
            x: (sectionWidth * teamIndex) + (Math.random() * sectionWidth),
            y: Math.random() * canvas.height
        };
        const direction = new Vec(canvas.height / 2 - position.y, canvas.width / 2 - position.x);
        const velocity = { x: 0, y: 0 };
        return {
            ...createShip(id, teamIndex + 1, maxHp, acceleration, isBot, isLearner, color, baseDiameter, direction, position, velocity),
        }
    });
    return ships;
}
const learningRate = .001;
const numShips = 6;
const numTeams = 2;
const batchSize = 512;
const numInputs = numShips * (2 + 2 + 2 + 1);// + (numShips-1);//pos, vel, acc, shield + REMOVED team

const actions = [-Math.PI / 8, -Math.PI / 16, -Math.PI / 32, -Math.PI / 64, 0, Math.PI / 64, Math.PI / 32, Math.PI / 16, Math.PI / 8];
const configPpo = {
    nSteps: batchSize,                 // Number of steps to collect rollouts
    nEpochs: 20,                 // Number of epochs for training the policy and value networks
    policyLearningRate: learningRate,    // Learning rate for the policy network
    valueLearningRate: learningRate,     // Learning rate for the value network
    clipRatio: 0.2,              //.2- PPO clipping ratio for the objective function
    targetKL: 0.02,            // .01-Target KL divergence for early stopping during policy optimization
    netArch: {
        'pi': [numInputs*2, numInputs],           // Network architecture for the policy network
        'vf': [numInputs*2, numInputs]           // Network architecture for the value network
    },
    activation: 'elu',          //relu, elu Activation function to be used in both policy and value networks
    verbose: 0                // cm-does this do anything? - Verbosity level (0 for no logging, 1 for logging)
}
let ppo;
let ships;

async function shipTurns(action) {
    const observations = [];
    for (const ship of ships) {
        if (ship.hp <= 0) {
            ship.hp = ship.maxHp;
            ship.position = {
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height
            };
            ship.velocity = { x: 0, y: 0 };
        }
        const enemies = ships.filter(s => s.team !== ship.team)
            .sort((a, b) => {
                const distanceA = Math.sqrt(Math.pow(a.position.x - ship.position.x, 2) + Math.pow(a.position.y - ship.position.y, 2));
                const distanceB = Math.sqrt(Math.pow(b.position.x - ship.position.x, 2) + Math.pow(b.position.y - ship.position.y, 2));
                return distanceA - distanceB;
            });
        if (ship.isBot) {
            ship.turnTowardsEnemy([enemies?.[0]]);
        }
        else if (ship.isLearner) {
            const angle = actions[action];//action * Math.PI/8;//
            ship.direction = new Vec(ship.direction).rotate(angle).getUnit();
        }
        else {
            // not a bot or learner
            const states = ship.getStates();
            // eslint-disable-next-line no-unused-vars
            const [preds, actionProximal, value, logprobability] = await ppo.getSample(states);
            const action = tf.argMax(preds).dataSync()[0];
            const angle = actions[action];//preds * Math.PI/8;
            ship.direction = new Vec(ship.direction).rotate(angle).getUnit();
        }
        ship.shoot(1200, enemies);
        ship.move();       
    }
    for (const ship of ships) {
        if (ship.isLearner) {
            const observation = ship.getObservation();
            observations.push(observation);
        }       
        if(!ship.isBot)
            rewardSignals.push(ship.rewardSignal);
        ship.rewardSignal = 0;
    }
    return observations;
}
async function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    draw(ships);
    $("#turns").text(totalTurns);
    if (totalTurns % batchSize === 0) {
        const weights = ppo.actor.getWeights();
        const criticWeights = ppo.critic.getWeights();
        const weightsData = weights.map(weight => weight.dataSync());
        const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

        $("#weights").text(weightsData);
        $("#criticWeights").text(criticWeightsData);
        $("#current-state").text(ships.find(a => a.isLearner).states.join(', '));
        createOrUpdateRewardChart();
    }
}
class Env {
    constructor() {
        this.actionSpace = {          
                // 'class': 'Box',
                // 'shape': [1],
                // 'low': -1,
                // 'high': 1,
            'class': 'Discrete',
            'n': actions.length,
        }
        this.observationSpace = {
            'class': 'Box',
            'shape': [numInputs],
            'dtype': 'float32',
        }
        this.resets = 0
    }
    async step(action) {
        const observations = await shipTurns(action);
        if (skipFrames == 0 || totalTurns % skipFrames === 0) {
            // dont know why i ended up having to do this:
            await requestAnimationFrameAsync(async () => await animate());
        }
        if (observations.length != 1) {
            console.error('observations != 1: ', observations);
        }
        ++totalTurns;
        return [observations[0].newObservation, observations[0].reward, false];
    }
    reset() {
        this.i = 0;
        const states = ships.find(a => a.isLearner).states;
        if (states.length > 0)
            return states;
        const array = new Array(this.observationSpace.shape[0]).fill(.1);
        return array;
    }
}
function requestAnimationFrameAsync(func) {
    return new Promise((resolve) => {
        requestAnimationFrame(() => {
            resolve(func());
        });
    });
}
(async function () {
    ships = createShips({ numShips, numTeams });
    const env = new Env();
    ppo = new PPO(env, configPpo);
    await ppo.learn({
        'totalTimesteps': 10000000,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config)
            }
        }
    });
})();
$('#skip-frames').on('input', function () {
    skipFrames = +$(this).val();
});

let avgRewardsGlobal = []; 
let chart;

function createOrUpdateRewardChart() {
    let canvas = document.getElementById('rewardOverTimeChart');
    if (!canvas) {
        console.error('Canvas element #rewardOverTimeChart not found.');
        return;
    }
    let ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Unable to get canvas context.');
        return;
    }

    // Calculate new average rewards
    if (rewardSignals.length > 0) {
        for (let i = 0; i < rewardSignals.length; i += batchSize) {
            let batch = rewardSignals.slice(i, i + batchSize);
            let batchAvg = batch.reduce((a, b) => a + b, 0) / batch.length;
            avgRewardsGlobal.push(batchAvg);
        }
        rewardSignals = []; // Reset rewardSignals after updating avgRewards
    }

    // Calculate moving average
    const movingAverageWindow = 10; // Adjust as needed
    const movingAverage = avgRewardsGlobal.map((_, index, array) => {
        const start = Math.max(0, index - movingAverageWindow + 1);
        const end = index + 1;
        return array.slice(start, end).reduce((sum, value) => sum + value, 0) / (end - start);
    });

    if (!chart) {
        // If the chart does not exist, create it
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: avgRewardsGlobal.map((_, i) => i),
                datasets: [{
                    label: 'Average reward',
                    data: avgRewardsGlobal,
                    borderColor: 'rgba(75, 192, 192, 0.5)',
                    fill: false
                }, {
                    label: 'Moving average',
                    data: movingAverage,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    fill: false
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: false // Changed to false to better show trends
                    }
                },
                animation: {
                    duration: 0 // Disable animation for performance
                }
            }
        });
    } else {
        // If the chart already exists, update its data
        chart.data.labels = avgRewardsGlobal.map((_, i) => i);
        chart.data.datasets[0].data = avgRewardsGlobal;
        chart.data.datasets[1].data = movingAverage;
        chart.update();
    }

    // Log some statistics
    console.log('Total data points:', avgRewardsGlobal.length);
    console.log('Latest average reward:', avgRewardsGlobal[avgRewardsGlobal.length - 1]);
    console.log('Latest moving average:', movingAverage[movingAverage.length - 1]);
}