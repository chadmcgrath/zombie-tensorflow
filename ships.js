
/* global $ */
/* global Chart */
/* global PPO */
/* global line_point_intersect, Vec */

/* global tf */
//tf.enableDebugMode();//
const isDiscrete = false;
const isSpace = false;
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
        p3.x = height / 2; p3.y = 0;

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
    // // show white lines from pi/64 to -pi/64 from current direction
    // ctx.strokeStyle = 'white';
    // ctx.beginPath();
    // ctx.moveTo(learner.position.x, learner.position.y);
    // ctx.lineTo(
    //     learner.position.x + (new Vec(learner.direction).rotate(Math.PI/64).x *1000),
    //     learner.position.y + (new Vec(learner.direction).rotate(Math.PI/64).y * 1000),
    // );
    // ctx.moveTo(learner.position.x, learner.position.y);
    // ctx.lineTo(
    //     learner.position.x + (new Vec(learner.direction).rotate(-Math.PI/64).x * 1000),
    //     learner.position.y + (new Vec(learner.direction).rotate(-Math.PI/64).y * 1000),
    // );
    // ctx.stroke();

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

function rotatePoint(point, direction, center) {
    const cos = direction.x;
    const sin = direction.y;
    const x = point.x;
    const y = point.y;
    point.x = x * cos - y * sin + center.x;
    point.y = x * sin + y * cos + center.y;
}
function calculateTurnDirectionVector(basePosition, baseDirection, targetPosition) {
    // Calculate the vector from base to target
    const vectorToTarget = {
        x: targetPosition.x - basePosition.x,
        y: targetPosition.y - basePosition.y
    };

    // Normalize the vector to target
    const magnitudeToTarget = Math.sqrt(vectorToTarget.x ** 2 + vectorToTarget.y ** 2);
    const normalizedVectorToTarget = {
        x: vectorToTarget.x / magnitudeToTarget,
        y: vectorToTarget.y / magnitudeToTarget
    };

    // Normalize the base direction
    const magnitudeBaseDirection = Math.sqrt(baseDirection.x ** 2 + baseDirection.y ** 2);
    const normalizedBaseDirection = {
        x: baseDirection.x / magnitudeBaseDirection,
        y: baseDirection.y / magnitudeBaseDirection
    };

    // The direction to turn to face the target can be represented as the difference between the normalized target vector and the base direction
    const turnDirectionVector = {
        x: normalizedVectorToTarget.x - normalizedBaseDirection.x,
        y: normalizedVectorToTarget.y - normalizedBaseDirection.y
    };

    // Optionally, normalize the turn direction vector if a unit vector is desired
    const magnitudeTurnDirection = Math.sqrt(turnDirectionVector.x ** 2 + turnDirectionVector.y ** 2);
    const normalizedTurnDirectionVector = {
        x: turnDirectionVector.x / magnitudeTurnDirection,
        y: turnDirectionVector.y / magnitudeTurnDirection
    };

    return normalizedTurnDirectionVector;
}
function calculateRelativeDirection(basePosition, baseDirection, targetPosition) {
    // Calculate the vector from base to target
    const vectorToTarget = {
        x: targetPosition.x - basePosition.x,
        y: targetPosition.y - basePosition.y
    };
    // Calculate the angle of this vector
    const angleToTarget = Math.atan2(vectorToTarget.y, vectorToTarget.x);
    // Calculate the angle of the base direction
    const baseAngle = Math.atan2(baseDirection.y, baseDirection.x);
    // Calculate the relative angle
    let relativeAngle = angleToTarget - baseAngle;
    // Normalize the angle to be between -π and π
    relativeAngle = (relativeAngle + Math.PI * 3) % (Math.PI * 2) - Math.PI;
    return relativeAngle / Math.PI;
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
        acceleration: acceleration || .0001,
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
        getOrderedShips: function () {
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
                //Math.atan2(this.direction.y, this.direction.x)/Math.PI,
                ...(this.isSpace ? [this.direction.x, this.direction.y] : []),
                this.velocity.x / velocityDenominator,
                this.velocity.y / velocityDenominator,
                this.position.x / positionDenominator,
                this.position.y / positionDenominator,
                this.hp / this.maxHp,

            ];
            const otherShips = this.getOrderedShips();

            for (const ship of otherShips) {
                const rel = calculateRelativeDirection(this.position, this.direction, ship.position);
                const otherShip = [
                    // rel,
                    // Math.atan2(ship.direction.y, ship.direction.x)/Math.PI,
                    Math.cos(rel),
                    Math.sin(rel),
                    ...(this.isSpace ? [this.direction.x, this.direction.y] : []),
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
            let reflected = false;
            const radius = this.radius;
            const buffer = 1; // Small buffer to ensure the ship is fully off the edge
        
            if (this.position.x - radius < 0) {
                this.velocity.x = Math.abs(this.velocity.x); // Ensure positive x velocity
                this.position.x = radius + buffer;
                reflected = true;
            } else if (this.position.x + radius > canvas.width) {
                this.velocity.x = -Math.abs(this.velocity.x); // Ensure negative x velocity
                this.position.x = canvas.width - radius - buffer;
                reflected = true;
            }
            if (this.position.y - radius < 0) {
                this.velocity.y = Math.abs(this.velocity.y); // Ensure positive y velocity
                this.position.y = radius + buffer;
                reflected = true;
            } else if (this.position.y + radius > canvas.height) {
                this.velocity.y = -Math.abs(this.velocity.y); // Ensure negative y velocity
                this.position.y = canvas.height - radius - buffer;
                reflected = true;
            }
            if (reflected) {
                const maxSpeed = 1;
                this.velocity = new Vec(this.velocity);
                const speed = this.velocity.length();
                if (speed > maxSpeed)
                    this.velocity.scale(maxSpeed / speed);
                this.rewardSignal -= .01;
                
                // Add a small random component to the velocity to help unstick the ship
                this.velocity.x += (Math.random() - 0.5) * 0.1;
                this.velocity.y += (Math.random() - 0.5) * 0.1;
            }
        },
        move: function (reductionFactor) {
            this.reflectOffScreen();
            let dir = this.direction;
            this.velocity.x += dir.x * this.acceleration * reductionFactor;
            this.velocity.y += dir.y * this.acceleration * reductionFactor;
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
        turnTowardsEnemy: function (ships, maxTurnRate) {
            // sometimes we already have picked a target
            let deltaAngle = 0;
            let closestEnemy = ships.length < 2 ? ships?.[0] : this.getClosestEnemy(ships);
            if (closestEnemy) {
                let dx = closestEnemy.position.x - this.position.x;
                let dy = closestEnemy.position.y - this.position.y;
                let targetAngle = Math.atan2(dy, dx);
                let currentAngle = Math.atan2(this.direction.y, this.direction.x);
                deltaAngle = targetAngle - currentAngle;
                deltaAngle = ((deltaAngle + Math.PI) % (2 * Math.PI)) - Math.PI;
                deltaAngle = Math.max(-maxTurnRate, Math.min(maxTurnRate, deltaAngle));

                // Add the deltaAngle to the current angle to get the new angle
                let newAngle = currentAngle + deltaAngle;
                this.direction.x = Math.cos(newAngle);
                this.direction.y = Math.sin(newAngle);              
            } 
            return deltaAngle;         
        },
        shoot: function (d, enemies) {
            const endPoint = new Vec(this.position.x + this.direction.x * d, this.position.y + this.direction.y * d);

            for (let enemy of enemies) {
                const { up, ua } = line_point_intersect(new Vec(this.position), endPoint,
                    new Vec(enemy.position), enemy.radius);
                if (up && ua >= 0 && ua <= 1) {
                    enemy.hp -= 1;
                    this.handleDingedReward(enemy);
                    if (!this.isBot) {
                        this.handleHitReward(enemy);
                    }
                    beams.push({
                        start: { ...this.position },
                        end: up,
                        color: this.color,
                        time: Date.now(),
                        width: 3,
                        lifeTime: 1500,
                    });
                }
                else if (ua > 1 && !this.isBot) {
                    // curiculum reward for seeing pointing at baddie
                    this.rewardSignal += Math.min(1 / (ua), .4);
                    beams.push({
                        start: { ...this.position },
                        end: up,
                        color: this.isLearner ? 'purple' : 'orange',
                        time: Date.now(),
                        width: 1,
                        lifeTime: 500,
                    });
                }
                else {
                    this.rewardSignal -= .01;
                }
            }
        },
        handleHitReward: function (ship) {
            const r = .5 + .5 * (1 - ship.hp / ship.maxHp);
            this.rewardSignal += r;
            //rewardSignals.push(r);
            //rewardSignals.push(1);
        },
        handleDingedReward: function (ship) {
            const r = .5 * (.5 + .5 * ship.hp / ship.maxHp);
            ship.rewardSignal -= r;

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
const batchSize = 1024;
const numInputs = numShips * (2 + 2 + 1) + (numShips - 1) * 2 + (isSpace ? numShips * 2 : 0);//pos, vel, acc, shield + relatviedirection. REMOVED team

const actions = [-Math.PI / 16, -Math.PI * 3 / 64, -Math.PI / 32, -Math.PI / 64, -Math.PI / 128, 0, Math.PI / 128, Math.PI / 64, Math.PI / 32, Math.PI * 3 / 64, Math.PI / 16];
const configPpo = {
    nSteps: batchSize,                 // Number of steps to collect rollouts
    nEpochs: 20,                 // Number of epochs for training the policy and value networks
    policyLearningRate: learningRate,    // Learning rate for the policy network
    valueLearningRate: learningRate,     // Learning rate for the value network
    clipRatio: 0.2,              //.2- PPO clipping ratio for the objective function
    targetKL: 0.015,            // .01-Target KL divergence for early stopping during policy optimization
    netArch: {
        'pi': [128, 64],           // Network architecture for the policy network
        'vf': [128, 64]           // Network architecture for the value network
    },
    activation: 'elu',          //relu, elu Activation function to be used in both policy and value networks
    verbose: 0                // cm-does this do anything? - Verbosity level (0 for no logging, 1 for logging)
}
let ppo;
let ships;
const maxAngle = Math.PI / 256;
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

        let angle = 0;
        if (ship.isBot) {
            angle = ship.turnTowardsEnemy([enemies?.[0]], maxAngle);
        }
        else if (ship.isLearner) {
            angle = isDiscrete ? actions[action] : action * maxAngle;
            ship.direction = new Vec(ship.direction).rotate(angle).getUnit();
        }
        else {
            // not a bot or learner
            const states = ship.getStates();
            // eslint-disable-next-line no-unused-vars
            const [preds, actionProximal, value, logprobability] = await ppo.getSample(states);
            if (isDiscrete) {
                const action = tf.argMax(preds).dataSync()[0];
                angle = actions[action];
            } else {
                angle = preds * maxAngle;
            }
            ship.direction = new Vec(ship.direction).rotate(angle).getUnit();
        }
        ship.shoot(400, enemies);
        let reductionFactor = 1;
        if (!isSpace) {
            reductionFactor = 1 - (Math.abs(angle) * .9 / maxAngle);
            ship.velocity = new Vec(ship.velocity);
            const speed = ship.velocity.length();
            ship.velocity = new Vec(ship.direction).getUnit();
            ship.velocity.scale(speed);
        }

        ship.move(reductionFactor);
    }
    for (const ship of ships) {
        if (ship.isLearner) {
            const observation = ship.getObservation();
            observations.push(observation);
        }
        if (!ship.isBot)
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
        this.actionSpace = isDiscrete ? {
            'class': 'Discrete',
            'n': actions.length,
        } : {
            'class': 'Box',
            'shape': [1],
            'low': -1,
            'high': 1,
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
}