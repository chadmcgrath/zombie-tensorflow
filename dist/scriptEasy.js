/* global tf */
/* global $ */
//tf.enableDebugMode();
tf.setBackend('cpu');


window.addEventListener('beforeunload', () => {
  model.dispose();
});
class ActorCriticModel {

  constructor(learningRate, numLayers, numInputs, numActions, hiddenUnits, gamma, batchSize) {
    this.maxMemorySize = 5000;
    this.numInputs = numInputs;
    this.numActions = numActions;
    this.hiddenUnits = hiddenUnits;

    this.actor = this.createActorModel();
    this.critic = this.createCriticModel();
    this.memory = [];
    this.alpha = 0.6;  // Controls how much prioritization is used
    this.priorities = [];

    // Decay rate (not currently used - i'm hoping adam will handle it)
    this.decayRate = 0.01;
    this.steps = 1000000;
    this.globalStep = 0;

    this.gamma = gamma;
    this.batchSize = batchSize;
    this.kernelInitializer = tf.initializers.heNormal();
    this.initialLearningRate = learningRate;
    this.numHiddenLayers = numLayers;

    this.hiddenActivation = 'relu';
    this.regularizers = null;//tf.regularizers.l2({ l2: 0.001 });
  }
  dispose() {
    this.actor.dispose();
    this.critic.dispose();
  }
  async save(location) {
    await this.actor.save(location + '/actor');
    await this.critic.save(location + '/critic');
  }
  createActorModel() {
    const stateInput = tf.input({ shape: [this.numInputs] });

    let hidden = tf.layers.dense({
      units: this.hiddenUnits,
      activation: this.hiddenActivation,
      kernelInitializer: this.kernelInitializer,
      kernelRegularizer: this.regularizers
    }).apply(stateInput);
    for (let i = 1; i < this.numHiddenLayers; i++) {
      hidden = tf.layers.dense({
        units: this.hiddenUnits,
        activation: this.hiddenActivation,
        kernelInitializer: this.kernelInitializer,
        kernelRegularizer: this.regularizers
      }).apply(hidden);
    }

    const actions = tf.layers.dense({
      units: numActions,
      activation: 'softmax',
      kernelInitializer: this.kernelInitializer,
      kernelRegularizer: this.regularizers
    }).apply(hidden);

    const model = tf.model({ inputs: stateInput, outputs: actions });

    const optimizer = tf.train.adam(this.initialLearningRate);

    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

    return model;
  }

  createCriticModel() {
    const stateInput = tf.input({ shape: [this.numInputs] });
    const actionInput = tf.input({ shape: [this.numActions] });

    let stateHidden = tf.layers.dense({
      units: this.hiddenUnits,
      activation: this.hiddenActivation,
      kernelInitializer: this.kernelInitializer,
      kernelRegularizer: this.regularizers
    }).apply(stateInput);
    for (let i = 1; i < this.numHiddenLayers; i++) {
      stateHidden = tf.layers.dense({
        units: this.hiddenUnits,
        activation: this.hiddenActivation,
        kernelInitializer: this.kernelInitializer,
        kernelRegularizer: this.regularizers
      }).apply(stateHidden);
    }

    let actionHidden = tf.layers.dense({
      units: this.hiddenUnits,
      activation: this.hiddenActivation,
      kernelInitializer: this.kernelInitializer,
      kernelRegularizer: this.regularizers
    }).apply(actionInput);
    for (let i = 1; i < this.numHiddenLayers; i++) {
      actionHidden = tf.layers.dense({
        units: this.hiddenUnits,
        activation: this.hiddenActivation,
        kernelInitializer: this.kernelInitializer,
        kernelRegularizer: this.regularizers
      }).apply(actionHidden);
    }

    const merged = tf.layers.concatenate().apply([stateHidden, actionHidden]);
    const output = tf.layers.dense({
      units: 1,
      activation: 'linear',
      kernelInitializer: this.kernelInitializer,
      kernelRegularizer: this.regularizers
    }).apply(merged);

    const model = tf.model({ inputs: [stateInput, actionInput], outputs: output });
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return model;
  }

  async fitActor(states, advantages, actions) {
    const statesTensor = states instanceof tf.Tensor ? states : tf.tensor2d([states]);
    let advantagesTensor = advantages instanceof tf.Tensor ? advantages : tf.tensor2d([advantages]);

    // Ensure advantagesTensor is a rank 2 tensor
    if (advantagesTensor.rank === 1) {
      advantagesTensor = advantagesTensor.expandDims(1);
    }

    const actionsMax = actions.argMax(1);
    // Create a 2D tensor where the advantage is applied to the action with the maximum probability
    const actionsOneHot = tf.oneHot(actionsMax, this.numActions);
    advantagesTensor = advantagesTensor.mul(actionsOneHot);

    return await this.actor.trainOnBatch(statesTensor, advantagesTensor);
  }

  async fitCritic(states, actions, qValues) {
    const statesTensor = states instanceof tf.Tensor ? states : tf.tensor2d([states]);
    const actionsTensor = actions instanceof tf.Tensor ? actions : tf.tensor2d([actions]);
    const qValuesTensor = qValues instanceof tf.Tensor ? qValues : tf.tensor2d([qValues]);
    return await this.critic.trainOnBatch([statesTensor, actionsTensor], qValuesTensor);
  }

  predictActor(states) {
    return tf.tidy(() => {
      let statesTensor = states instanceof tf.Tensor ? states : tf.tensor2d([states]);
      return this.actor.predict(statesTensor);
    });
  }

  predictCritic(states, actions) {
    return tf.tidy(() => {
      const statesTensor = states instanceof tf.Tensor ? states : tf.tensor2d([states]);
      const actionsTensor = actions instanceof tf.Tensor ? actions : tf.tensor2d([actions]);
      return this.critic.predict([statesTensor, actionsTensor]).squeeze();
    });
  }
  getNumSamples(data) {
    if (data instanceof tf.Tensor) {
      return data.shape[0];
    } else if (Array.isArray(data)) {
      return data.length;
    }
  }

  remember([state, oldActions, reward, nextState, actions, tdError]) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift();
      this.priorities.shift();
    }
    const tdErrorValue = isNaN(tdError) ? tdError.dataSync()[0] : tdError;
    if (actions.some(a => a > 15) || actions.some(a => a < 0) || actions.length !== this.numActions)
      console.error('actions contain values outside of [-1S,1] or is 0 length');
    if (oldActions.some(a => a > 15) || oldActions.some(a => a < 0) || oldActions.length !== this.numActions)
      console.error('oldactions contain values outside of [-1S,1] or is 0 length');
    this.memory.push([state, oldActions, reward, nextState, actions, tdErrorValue]);
    const priority = Math.pow(Math.abs(tdErrorValue) + 1e-6, this.alpha);  // Prioritize experiences with higher error
    if (isNaN(tdErrorValue) || isNaN(priority))
      console.error('Error or priority is NaN');
    else
      this.priorities.push(priority);  // Prioritize experiences with higher error
  }
  sample(batchSize) {
    // Calculate the sum of the priorities
    const prioritiesSum = this.priorities.reduce((a, b) => a + b, 0);

    // Check if the priorities sum is zero
    if (prioritiesSum === 0) {
      // All priorities are zero, return a random sample
      return this.memory.slice(0, batchSize);
    }

    const batch = [];
    const indices = [];
    for (let i = 0; i < batchSize; i++) {
      const rand = Math.random() * prioritiesSum;
      let cumulativeSum = 0;
      for (let j = 0; j < this.memory.length; j++) {
        cumulativeSum += this.priorities[j];
        if (cumulativeSum > rand) {
          batch.push(this.memory[j]);
          indices.push(j);
          break;
        }
      }
    }
    return [batch, indices];
  }

  async train() {
    const model = this;
    const [sample, sampleIndices] = model.sample(this.batchSize);

    // Prepare batch data
    let batchOldStates = [];
    let batchActions = [];
    let batchRewardSignals = [];
    let batchNewStates = [];

    for (let [oldStates, actions, rewardSignal, states] of sample) {
      if (actions.length !== this.numActions)
        console.error('actions is 0 length: sample' + sample);
      batchOldStates.push(oldStates);
      batchActions.push(actions);
      batchRewardSignals.push(rewardSignal);
      batchNewStates.push(states);
    }

    // Convert to tensors
    const oldStatesTensor = tf.tensor(batchOldStates);
    const actionsTensor = tf.tensor(batchActions);
    const rewardSignalsTensor = tf.tensor(batchRewardSignals);
    const newStatesTensor = tf.tensor(batchNewStates);
    if (tf.any(tf.isNaN(oldStatesTensor)).dataSync()[0]) {
      console.error('oldStatesTensor contains NaN');
    }
    if (tf.any(tf.isNaN(actionsTensor)).dataSync()[0]) {
      console.error('actionsTensor contains NaN');
    }
    if (tf.any(tf.isNaN(rewardSignalsTensor)).dataSync()[0]) {
      console.error('rewardSignalsTensor contains NaN');
    }
    if (tf.any(tf.isNaN(newStatesTensor)).dataSync()[0]) {
      console.error('newStatesTensor  contains NaN');
    }
    const oldQValue = model.predictCritic(oldStatesTensor, actionsTensor);

    if (tf.any(tf.isNaN(oldQValue)).dataSync()[0]) {
      console.error('oldQValue contains NaN ' + oldQValue.dataSync()[0]);
      const weights = model.actor.getWeights();
      const criticWeights = model.critic.getWeights();
      const weightsData = weights.map(weight => weight.dataSync());
      const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

      $("#weights").text(weightsData);
      $("#criticWeights").text(criticWeightsData);
      console.error(criticWeightsData);
      $("#current-state").text(states.join(', '));
    }

    const newQValue = model.predictCritic(newStatesTensor, actionsTensor);

    if (tf.any(tf.isNaN(newQValue)).dataSync()[0]) {
      console.error('newQValue  contains NaN ' + oldQValue.dataSync()[0]);
    }
    // Compute target Q-values
    const targetQValue = rewardSignalsTensor.add(newQValue.mul(this.gamma));
    if (tf.any(tf.isNaN(targetQValue)).dataSync()[0]) {
      console.error('targetQValue  contains NaN');
    }
    // Update critic model
    const criticHistory = await model.fitCritic(oldStatesTensor, actionsTensor, targetQValue);
    if (isNaN(criticHistory)) {
      const weights = model.actor.getWeights();
      const criticWeights = model.critic.getWeights();
      const weightsData = weights.map(weight => weight.dataSync());
      const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

      $("#weights").text(weightsData);
      $("#criticWeights").text(criticWeightsData);
      $("#current-state").text(states.join(', '));
      console.error('criticHistory NaN');
    }
    // Update actor model
    const advantage = targetQValue.sub(newQValue);
    const actorHistory = await model.fitActor(oldStatesTensor, advantage, actionsTensor);

    // Compute TD errors for each experience in the batch
    const qValueNextState = newQValue.reshape([this.batchSize]);
    const qValueCurrent = oldQValue.reshape([this.batchSize]);
    const tdErrors = rewardSignalsTensor.add(qValueNextState.mul(this.gamma)).sub(qValueCurrent);

    const tdErrorsArray = tdErrors.dataSync();
    if (tdErrorsArray.some(t => isNaN(t))) {
      console.error('tdErrors contains NaN');
    }
    for (let i = 0; i < sample.length; i++) {
      const tdError = tdErrorsArray[i];

      // Update tdError and priority for existing experience
      if (this.memory[sampleIndices[i]]) {
        this.memory[sampleIndices[i]][5] = tdError;
        const priority = Math.abs(tdError);  // Recalculate priority
        this.priorities[sampleIndices[i]] = priority;  // Update priority
      } else {
        // Add new experience
        console.log('Adding new experience, this probably shouldnt ever happen');
        this.remember([...sample[i], tdError]);
      }
    }
    return [actorHistory, criticHistory, sample]
  }
}

const actorLossValues = [];
const criticLossValues = [];
let continueLoop = false;
let loopCount = 0;
let states = [];
let oldStates = []
let actions = [];
let negRewards = 0;
let totalRewards = 0;
let rewardOverTime = [];
let totalTurns = 0;
let missedShots = 0;
let hitShotsBaddy = 0;
// Hyperparameters

const gamma = 0.1;  // Discount factor
const numActions = 16;
const numInputs = 91;
const learningRate = .0003;//+$('#slider-lr').val();
const numHidden = 90;
let batchSize = +$('#slider-batch').val();
const numLayers = 1;
let epsilon = .15;


const model = new ActorCriticModel(learningRate, numLayers, numInputs, numActions, numHidden, gamma, batchSize);
const eyeMaxRange = 1000;
var oneRad = Math.PI / 180;
var pi2 = Math.PI * 2;
var gameSpeed = 3;
var EPS = 0.01;



function clone(v) {
  return {
    x: v.x,
    y: v.y
  };
}

function normalize(v) {
  var d = Math.sqrt(v.x * v.x + v.y * v.y);
  if (d === 0) return v;
  v.x /= d;
  v.y /= d;
  return v;
}

function sub(v1, v2) {
  return {
    x: v1.x - v2.x,
    y: v1.y - v2.y
  };
}

function project(v, d, t) {
  return {
    x: v.x + d.x * t,
    y: v.y + d.y * t
  };
}

function randomAngle(s) {
  var r = Math.random() * pi2;
  if (!s) s = 1;
  return {
    x: Math.cos(r) * s,
    y: Math.sin(r) * s
  };
}

function fixAngle(d) {
  while (d < 0) d += pi2;
  while (d >= pi2) d -= pi2;
  return d;
}

// these act as obstacles
function Square(config) {
  this.fill = config.fill || '#CCC';
  this.stroke = config.stroke || '#000';
  this.pos = config.pos || {
    x: ctx.canvas.width * Math.random(),
    y: ctx.canvas.height * Math.random()
  };
  this.width = config.width || 20 + Math.random() * 100;
  this.height = config.height || 20 + Math.random() * 100;
  // extents used for AABB testing
  this.bounds = [{
    x: this.pos.x - this.width / 2,
    y: this.pos.y - this.height / 2
  }, {
    x: this.pos.x + this.width / 2,
    y: this.pos.y + this.height / 2
  }]
}
Square.prototype.draw = function (ctx) {
  ctx.beginPath();
  ctx.rect(this.bounds[0].x, this.bounds[0].y, this.width, this.height);
  ctx.fillStyle = this.fill;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = this.stroke;
  ctx.stroke();
};
Square.prototype.pointNormal = function (p) {
  if (Math.abs(p.x - this.bounds[0].x) < EPS) {
    return {
      x: -1,
      y: 0
    };
  } else if (Math.abs(p.x - this.bounds[1].x) < EPS) {
    return {
      x: 1,
      y: 0
    };
  } else if (Math.abs(p.y - this.bounds[0].y) < EPS) {
    return {
      x: 0,
      y: -1
    };
  } else if (Math.abs(p.y - this.bounds[1].y) < EPS) {
    return {
      x: 0,
      y: 1
    };
  }
  return {
    x: 0,
    y: 1
  };
};
Square.prototype.rayIntersect = function (o, d) {
  var tmin, tmax, tymin, tymax;
  if (d.x >= 0) {
    tmin = (this.bounds[0].x - o.x) / d.x;
    tmax = (this.bounds[1].x - o.x) / d.x;
  } else {
    tmin = (this.bounds[1].x - o.x) / d.x;
    tmax = (this.bounds[0].x - o.x) / d.x;
  }
  if (d.y >= 0) {
    tymin = (this.bounds[0].y - o.y) / d.y;
    tymax = (this.bounds[1].y - o.y) / d.y;
  } else {
    tymin = (this.bounds[1].y - o.y) / d.y;
    tymax = (this.bounds[0].y - o.y) / d.y;
  }
  if (tmin > tymax || tymin > tmax) return false;
  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;
  var p1 = {
    dist: tmin,
    pos: project(o, d, tmin)
  };
  var p2 = {
    dist: tmax,
    pos: project(o, d, tmax)
  };
  p1.n = this.pointNormal(p1.pos);
  p2.n = this.pointNormal(p2.pos);

  //p1.n = normalize(sub(p1.pos,this.pos));
  //p2.n = normalize(sub(p2.pos,this.pos));
  return [p1, p2];
}

// A 2D vector utility (Karpathy)
var Vec = function (x, y) {
  this.x = x;
  this.y = y;
}
Vec.prototype = {

  // utilities
  dist_from: function (v) { return Math.sqrt(Math.pow(this.x - v.x, 2) + Math.pow(this.y - v.y, 2)); },
  length: function () { return Math.sqrt(Math.pow(this.x, 2) + Math.pow(this.y, 2)); },

  // new vector returning operations
  add: function (v) { return new Vec(this.x + v.x, this.y + v.y); },
  sub: function (v) { return new Vec(this.x - v.x, this.y - v.y); },
  rotate: function (a) {  // CLOCKWISE
    return new Vec(this.x * Math.cos(a) + this.y * Math.sin(a),
      -this.x * Math.sin(a) + this.y * Math.cos(a));
  },
  getAngle: function () { return Math.atan2(this.y, this.x); },
  getUnit: function () { var d = this.length(); return new Vec(this.x / d, this.y / d); },
  // in place operations
  scale: function (s) { this.x *= s; this.y *= s; },
  normalize: function () { var d = this.length(); this.scale(1.0 / d); }
}

function line_intersect(line1Start, line1End, line2Start, line2End) {
  let denominator = ((line2End.y - line2Start.y) * (line1End.x - line1Start.x)) - ((line2End.x - line2Start.x) * (line1End.y - line1Start.y));

  if (denominator === 0) {
    return false; // lines are parallel
  }

  let ua = (((line2End.x - line2Start.x) * (line1Start.y - line2Start.y)) - ((line2End.y - line2Start.y) * (line1Start.x - line2Start.x))) / denominator;
  let ub = (((line1End.x - line1Start.x) * (line1Start.y - line2Start.y)) - ((line1End.y - line1Start.y) * (line1Start.x - line2Start.x))) / denominator;

  if (ua < 0 || ua > 1 || ub < 0 || ub > 1) {
    return false; // intersection is outside the line segments
  }

  let x = line1Start.x + ua * (line1End.x - line1Start.x);
  let y = line1Start.y + ua * (line1End.y - line1Start.y);

  return { ua, ub, up: new Vec(x, y) };
}
var line_point_intersect = function (p1, p2, p0, rad) {
  p0 = new Vec(p0.x, p0.y);
  var v = new Vec(p2.y - p1.y, -(p2.x - p1.x)); // perpendicular vector
  var d = Math.abs((p2.x - p1.x) * (p1.y - p0.y) - (p1.x - p0.x) * (p2.y - p1.y));
  d = d / v.length();
  if (d > rad) { return false; }

  v.normalize();
  v.scale(d);
  var up = p0.add(v);
  if (!up) {
    console.log("up empty")
  }
  if (Math.abs(p2.x - p1.x) > Math.abs(p2.y - p1.y)) {
    var ua = (up.x - p1.x) / (p2.x - p1.x);
  } else {
    // eslint-disable-next-line no-redeclare
    var ua = (up.y - p1.y) / (p2.y - p1.y);
  }
  if (ua > 0.0 && ua < 1.0) {
    return { ua: ua, up: up };
  }
  return false;
}

function lineIntersectsSquare(lineStart, lineEnd, square) {
  let topLeft = { x: square.bounds[0].x, y: square.bounds[0].y };
  let topRight = { x: square.bounds[1].x, y: square.bounds[0].y };
  let bottomLeft = { x: square.bounds[0].x, y: square.bounds[1].y };
  let bottomRight = { x: square.bounds[1].x, y: square.bounds[1].y };

  let sides = [[topLeft, topRight], [topRight, bottomRight], [bottomRight, bottomLeft], [bottomLeft, topLeft]];

  let closestIntersection = null;
  let minDistance = Infinity;

  for (let side of sides) {
    let intersection = line_intersect(lineStart, lineEnd, side[0], side[1]);
    if (intersection) {
      let dx = lineStart.x - intersection.up.x;
      let dy = lineStart.y - intersection.up.y;
      let distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < minDistance) {
        minDistance = distance;
        closestIntersection = intersection;

      }
    }
  }
  return closestIntersection ? { ...closestIntersection, distance: minDistance } : false;
}
//karpathy collision code- psoibly replace with rayIntersect
// this doesn't handle buildings properly cuz the origina code din't need to.
const stuff_collide = (agent, p2, check_walls, check_items) => {
  var minres = false;
  p2 = new Vec(p2.x, p2.y);
  const p1 = agent.p;
  // collide with walls
  if (check_walls) {
    const windowWalls = {
      bounds: [{ x: 0, y: 0, width: window.innerWidth, height: 0 }, // top    
      { x: window.innerWidth, y: window.innerHeight, width: 0, height: window.innerHeight }]
    };
    const allWalls = blocks.concat(windowWalls);

    for (var i = 0, n = allWalls.length; i < n; i++) {
      var wall = allWalls[i];

      let res = lineIntersectsSquare(p1, p2, wall);
      if (res) {
        res.type = -.1; // is wall
        // Calculate the distance from p1 to the intersection point
        let dx = p1.x - res.up.x;
        let dy = p1.y - res.up.y;
        let distance = Math.sqrt(dx * dx + dy * dy);
        res.distance = distance;

        if (!minres) { minres = res; }
        else {
          // Check if it's closer
          if (res.distance < minres.distance) {
            // If yes, replace it
            minres = res;
          }
        }
      }
    }
  }

  // collide with items
  if (check_items) {
    // eslint-disable-next-line no-redeclare
    for (var i = 0, n = agent.items.length; i < n; i++) {
      var it = agent.items[i];
      var res = line_point_intersect(p1, p2, it.p, it.rad);
      if (res) {
        res.vx = it.v.x; // velocty information
        res.vy = it.v.y;

        res.type = it.type.isHuman ? 1 : -1; // store type of item
        res.agent = it;
        if (!minres) { minres = res; }
        else {
          if (res.ua < minres.ua) { minres = res; }
        }
      }
    }
  }

  return minres;
}
//karpathy
var Eye = function (angle) {
  this.angle = angle; // angle relative to agent its on
  this.max_range = eyeMaxRange;
  this.sensed_proximity = eyeMaxRange; // what the eye is seeing. will be set in world.tick()
  this.sensed_type = 0; // what does the eye see?

}
function Agent(config) {
  this.id = config.id;

  const maxHp = 1000000;
  const eyeCount = 30;
  this.eyes = [];
  this.rewardSignal = 0;

  const rads = 2 * Math.PI / eyeCount;
  this.isHuman = config.type === 'human';
  this.isZ = config.type === 'zombie';
  if (this.isHuman)
    for (var k = 0; k < eyeCount; k++) { this.eyes.push(new Eye(k * rads)); }

  this.maxHp = maxHp;
  this.currentHp = maxHp;
  this.items = [];

  this.type = config.type || 'human';
  this.pos = config.pos || {
    x: 0,
    y: 0
  };

  this.rad = 20;
  this.speed = config.speed || this.type === 'human' ? 3 + Math.random() : 1 + Math.random();
  this.turnSpeed = config.turnSpeed || this.type === 'human' ? oneRad * 2 : oneRad;
  this.dir = randomAngle();
  this.newDir = clone(this.dir);

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
  this.oangle = 0;
  this.op = this.pos;// old position
  this.v = new Vec(0, 0);
  //
  this.state = config.state || 'idle';
  this.viewDist = config.viewDist || 1000;
  this.viewFov = (config.viewFov || this.type === 'human' ? 90 : 45) * oneRad;
  this.viewFovD2 = this.viewFov / 2;
  this.nextTimer = Math.random() * 10;
  this.ring = config.ring || this.type === 'human' ? 0 : 5;
}

Agent.prototype.getVision = () => {

  let eyeStates = [];
  for (let i = 0, n = this.agents.length; i < n; i++) {
    var a = this.agents[i];
    a.target = null;
    // zombies do not have eyes
    if (a.eyes.length === 0) continue;
    const pos = a.p;
    const angle = a.angle;
    


    for (var ei = 0, ne = a.eyes.length; ei < ne; ei++) {
      var e = a.eyes[ei];
      const eangle = e.angle;
      const currentEyeAnglePointing = angle.rotate(eangle).getUnit();
      // we have a line from p to p->eyep
      var eyep = new Vec(pos.x + e.max_range * currentEyeAnglePointing.x,
        pos.y + e.max_range * currentEyeAnglePointing.y);

      if (isNaN(eyep.x)) {
        console.error('eyep.x is NaN');
      }

      var res = stuff_collide(a, eyep, true, true);
      if (res) {
        // eye collided with anything
        if (ei === 0) {
          let isCollisonMain = false;
          const p1 = new Vec(a.pos.x, a.pos.y);
            const p2 = new Vec(a.pos.x + a.dir.x * eyeMaxRange, a.pos.y + a.dir.y * eyeMaxRange);
          for (let i = 0, n = a.items.length; i < n; i++) {
            var it = a.items[i];
            
            var res2 = line_point_intersect(p1, p2, it.p, it.rad);
            isCollisonMain = res2;
          }
          a.target = res.agent;
          if (isCollisonMain !== false && !res.agent) {
            console.error("what up");
            var res3 = stuff_collide(a, eyep, true, true);
          }
        }

        e.sensed_proximity = res.up.dist_from(a.p);
        e.sensed_type = res.type;
        if ('vx' in res) {
          e.vx = res.vx;
          e.vy = res.vy;
        } else {
          e.vx = 0;
          e.vy = 0;
        }
      } else {
        e.sensed_proximity = e.max_range;
        e.sensed_type = 0;
        e.vx = 0;
        e.vy = 0;
      }
      ctx.strokeStyle = "rgb(255,150,150)";
      if (e.sensed_type === -.1) {
        ctx.strokeStyle = "yellow"; // wall
      }
      if (e.sensed_type === 0) {
        ctx.strokeStyle = "rgb(200,200,200)"; //nothing
      }
      if (e.sensed_type === 1) { ctx.strokeStyle = "rgb(255,150,150)"; } // human
      //if (e.sensed_type === -1) { ctx.strokeStyle = "rgb(150,255,150)"; } // z
      if (e.sensed_type === -1) { ctx.strokeStyle = "green"; } // z
      if (ei === 0) {
        ctx.strokeStyle = "blue";
        $('#blue-eye').text(currentEyeAnglePointing);
      }

      const sr = e.sensed_proximity;
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
      const lineToX = pos.x + sr * currentEyeAnglePointing.x;
      const lineToY = pos.y + sr * currentEyeAnglePointing.y;
      ctx.lineTo(lineToX, lineToY);
      ctx.stroke();

      let type = e.sensed_type;
      if (type === 'zombie')
        type = -1;
      if (type === 'human')
        type = 1;

      // add to state for ML
      // tensorflow inputs
      eyeStates.push(sr *currentEyeAnglePointing.x/e.max_range, sr *currentEyeAnglePointing.y/e.max_range, type);
    }
  }
  // tensorflow inputs
  return eyeStates;
}

Agent.prototype.getColor = function () {
  if (this.state === 'mouse') return '#FF00FF';
  if (this.state === 'panic') return 'yellow';
  if (this.state === 'attack') return 'red';
  if (this.type === 'human') return 'blue';
  if (this.type === 'zombie') return 'green';
  return '#AAAAAA';
};
Agent.prototype.distTo = function (o) {
  var dx = o.x - this.pos.x;
  var dy = o.y - this.pos.y;
  return Math.sqrt(dx * dx + dy * dy);
};


Agent.prototype.see = function () {
  var seen = [];
  var a, d, ato;
  for (var i = 0, l = agents.length; i < l; i++) {
    // check if what we see is blocked by a wall   
    a = agents[i];
    if (a === this) continue;
    d = this.distTo(a.pos);
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
      const angle = normalize(sub(a.pos, this.pos))
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

Agent.prototype.logic = async function (ctx, clock) {

  var batchValue = $('#slider-batch').val();
  var epsilonValue = $('#slider-epsilon').val();

  this.epsilon = +epsilonValue;
  batchSize = +batchValue;
  model.batchSize = +batchValue;
  // // Setters
  // $('#slider-batch').val(8); // replace 8 with the value you want to set
  // $('#slider-lr').val(0.01); // replace 0.01 with the value you want to set
  // $('#slider-epsilon').val(0.2); // replace 0.2 with the value you want to set
  this.op = this.pos;
  var seen = this.see();

  // convert humans to zombie
  if (this.type === 'zombie' && seen.length) {
    for (var i = 0, l = seen.length; i < l; i++) {
      if (seen[i].agent.isHuman === true && seen[i].dist < 10) {
        // change to remove a hitpoint
        --seen[i].agent.currentHp;

        //tf ml reward
        seen[i].agent.rewardSignal = seen[i].agent.rewardSignal - 1;
        negRewards = negRewards - 1;

        if (seen[i].agent.currentHp < 1) {
          seen[i].agent.isHuman = false;
          seen[i].agent.eyes = [];
          seen[i].agent.type = this.type;
          seen[i].agent.viewFov = this.viewFov;
          seen[i].agent.viewFovD2 = this.viewFovD2;
          seen[i].agent.speed = this.speed;
          seen[i].agent.turnSpeed = this.turnSpeed;
          seen[i].agent.state = 'idle';
          seen[i].agent.ring = 1;
        }
      }
    }
  }

  let isMoving = true;
  if (this.isHuman === false && seen.some(s => s.agent.isHuman) === false) {
    this.nextTimer = 3 + Math.random() * 10;
    this.newDir = randomAngle();
    this.state = 'idle';

  }
  for (let i = 0, l = seen.length; i < l; i++) {
    const agentType = seen[i].agent.type;

    if (this.isHuman === false) {

      // attack human
      if (agentType === 'human') {
        this.state = 'attack';
        this.nextTimer = 5;
        this.newDir = seen[i].angle;
        this.dir = seen[i].angle;
        break;
      }
      // follow other zombie
      if (this.state === 'idle' && agentType === 'zombie' && Math.random() > 0.9) { // && seen[i].agent.state==='attack') {
        this.nextTimer = 5;
        this.newDir = seen[i].angle;
        this.dir = seen[i].angle;
      }
    }
  }   // for i in seen
  let oldActions;
  if (this.isHuman) {
    this.state = 'panic';
    this.nextTimer = 5;


    if (states.length < 1) {
      states = [];
      states.push(...this.getVision());
      const target = !this.target ? 0 : this.target.isHuman ? 1 : -1;

      states.push(target);
      console.log('got vision 1st time');
    }
    const predictedActionTensor = model.predictActor(states);

    if (actions.length < 1) {
      const actionsOneHot = Array(numActions).fill(0);
      actionsOneHot[numActions/2] = 1;
      actions = actionsOneHot;
      console.log('made fake actions shoot');
    }

    // Perform action and get new state and reward
    oldActions = [...actions];
    oldStates = [...states];
    states = [];
    states.push(...this.getVision());
    const target = !this.target ? 0 : this.target.isHuman ? 1 : -1;
    states.push(target);

    actions = [];
    let actionSelected;
    if (Math.random() < epsilon) {
      // Take a random action
      actionSelected = Math.floor(Math.random() * 16);
      const actionsOneHot = Array(numActions).fill(0);
      actionsOneHot[actionSelected] = 1;
      actions = actionsOneHot;
    } else {
      actionSelected = tf.argMax(predictedActionTensor, 1).dataSync()[0];
      actions = predictedActionTensor.arraySync()[0];
    }
    let newAngle = 0;
    if (actionSelected < (numActions - 1)) {
      const predictedActionVal = (actionSelected - ((numActions / 2) - 1)) * (360 / 30) * oneRad;
      newAngle = predictedActionVal;
    } else {
      isMoving = false;
      this.shoot(this);
    }
    
    const unitOldDir = new Vec(this.dir.x, this.dir.y).getUnit();
    const newVec = unitOldDir.rotate(newAngle);
    this.dir = newVec;
    if (actions.length !== 16)
      console.error('actions length is not 16');
  }


  if (this.ring) {
    this.ring += clock.delta * 20;
    if (this.ring > 100) this.ring = 0;
  }
  this.nextTimer -= clock.delta;
  // //zombies when timer runs out go back to random wandering
  // if (this.isHuman === false && this.nextTimer <= 0) {
  //   this.nextTimer = 3 + Math.random() * 3;
  //   this.dir = randomAngle();
  //   this.state = 'idle';
  // }

  var speed = isMoving && this.isHuman ? (this.speed) * 10 : 0;

  // get velociy
  var vx = this.dir.x * speed * clock.delta;
  var vy = this.dir.y * speed * clock.delta;
  // move
  this.pos.x += vx;
  this.pos.y += vy;
  // prevent walking through blocks
  // eslint-disable-next-line no-redeclare
  for (var i = 0, l = blocks.length; i < l; i++) {
    if (this.intersect = blocks[i].rayIntersect(this.pos, this.dir)) {
      if (this.intersect[0].dist <= 0 && this.intersect[1].dist > 0) {
        this.pos = this.intersect[0].pos;
        this.rewardSignal = this.rewardSignal - .1;
        negRewards = negRewards - .1;
        //this.newDir = this.intersect[0].n;
        this.dir = randomAngle();
        break;
      } else if (this.intersect[0].dist <= this.viewDist) {

      } else {
        this.intersect = false;
      }
    }
  }

  // if we hit a wall turn arround
  var bound = false;
  if (this.pos.x < 0) {
    this.pos.x = 0;
    this.dir.x = 1;
    bound = true
  }
  if (this.pos.y < 0) {
    this.pos.y = 0;
    this.dir.y = 1;
    bound = true;
  }
  if (this.pos.x > ctx.canvas.width) {
    this.pos.x = ctx.canvas.width;
    this.dir.x = -1;
    bound = true;
  }
  if (this.pos.y > ctx.canvas.height) {
    this.pos.y = ctx.canvas.height;
    this.dir.y = -1;
    bound = true;
  }
  if (bound) {
    this.rewardSignal = this.rewardSignal - .1;
    negRewards = negRewards - .1;
    normalize(this.dir);
  }


  if (this.isHuman === true && states.length > 0) {

    model.remember([oldStates, oldActions, this.rewardSignal, states, actions, 1]);
    if (totalTurns % batchSize === 0) {
      const [actorHistory, criticHistory, samples] = await model.train();
      actorLossValues.push(actorHistory);
      criticLossValues.push(criticHistory);
      if (totalTurns % (batchSize * 100) === 0) {
        const saveResult = await model.save('indexeddb://zombie-ac-1layer-2-reward-1-rotate');
      }
      $("#samples").text(samples.length);
    }

    totalRewards += this.rewardSignal;
    rewardOverTime.push(this.rewardSignal);
    $("#rewardTotal").text(totalRewards);
    $("#neg-rewards").text(negRewards);

    this.rewardSignal = 0;
  }
}
Agent.prototype.shoot = (agent) => {

  // Draw red line to the closest baddy

  ctx.beginPath();
  ctx.lineWidth = 6;

  ctx.moveTo(agent.pos.x, agent.pos.y);

  const closestBaddy = agent.target;
  if (closestBaddy) {
    ctx.strokeStyle = 'red';
    agent.rewardSignal += 10;
    hitShotsBaddy += 1;
    ctx.lineTo(closestBaddy.pos.x, closestBaddy.pos.y);
    closestBaddy.currentHp--;
    closestBaddy.state = 'idle';
    if (closestBaddy.currentHp < 1)

      closestBaddy.ring = 10;
    if (closestBaddy.hp < 1) {
      //todo: remove or create some kinda goodie
      //closestBaddy.agent.viewFov = this.viewFov;
      //closestBaddy.agent.viewFovD2 = this.viewFovD2;
      closestBaddy.speed = 0;
      //closestBaddy.agent.turnSpeed = this.turnSpeed;

      closestBaddy.ring = 20;
      agents = agents.filter(agent => agent !== closestBaddy);
      agent.items - agent.items.filter(agent => agent !== closestBaddy);
    }
  }
  else {

    // missed! purple line is missed shot. the agent did not move but shot nothing.
    // to do: for now, we disgourage it from stopping and missing.
    //agent.rewardSignal = agent.rewardSignal - .1;
    //negRewards = negRewards - .1;
    missedShots += 1;
    ctx.strokeStyle = 'purple';
    ctx.lineTo(agent.pos.x + agent.dir.x * eyeMaxRange, agent.pos.y + agent.dir.y * eyeMaxRange);
    // check if hits any agent items
  }
  ctx.stroke();

}
Agent.prototype.draw = function (ctx) {

  ctx.beginPath();
  ctx.arc(this.pos.x, this.pos.y, this.rad, 0, pi2, false);
  ctx.fillStyle = this.getColor();
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#FFFFFF';
  ctx.stroke();
  if (this.ring) {
    ctx.beginPath();
    ctx.arc(this.pos.x, this.pos.y, this.ring, 0, pi2, false);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#FF0000';
    ctx.stroke();
  }
  // draw view arc
  // var dir = Math.atan2(this.dir.y, this.dir.x);
  // ctx.beginPath();
  // ctx.arc(this.pos.x, this.pos.y, this.viewDist, dir - this.viewFovD2, dir + this.viewFovD2, false);
  // ctx.lineTo(this.pos.x, this.pos.y);
  // ctx.closePath();
  // ctx.lineWidth = 1;
  // ctx.stroke();
  var dir = new Vec(this.dir.x, this.dir.y);

  ctx.beginPath();
  ctx.moveTo(this.pos.x, this.pos.y);
  ctx.lineTo(this.pos.x + dir.x * this.rad, this.pos.y + dir.y * this.rad);
  ctx.strokeStyle = '#00FFFF';
  ctx.stroke();
  if (this.intersect) {
    // ctx.beginPath();
    // ctx.arc(this.intersect[0].pos.x, this.intersect[0].pos.y, 2, 0, pi2, false);
    // ctx.fillStyle = '#F00';
    // ctx.fill();
    // ctx.beginPath();
    // ctx.lineWidth = 1;
    // ctx.moveTo(this.pos.x,this.pos.y);
    // var n=project(this.pos,this.intersect[0].n,20);
    // ctx.lineTo(n.x,n.y);
    // ctx.strokeStyle = '#00FFFF';
    // ctx.stroke();
  }
};
var zCnt = 0;
let hCnt = 0;
var agents = [];
var blocks = [];
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var fps = 0;
var clock = {
  total: 0,
  start: 0,
  time: 0,
  delta: 0
};

async function mainLoop(time) {
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

  for (let i = 0, l = blocks.length; i < l; i++) {
    blocks[i].draw(ctx);
  }

  for (let i = 0, l = agents.length; i < l; i++) {
    if (agents[i].type === 'human') hCnt++;
    if (agents[i].type === 'zombie') zCnt++;
    await agents[i].logic(ctx, clock);
    agents[i].draw(ctx);
  }

  ctx.font = '20pt Calibri';
  ctx.lineWidth = 1;
  ctx.fillStyle = 'black';
  var msg = 'Zed:' + zCnt + '     Hum:' + hCnt + '    ' + 'Time:' + Math.floor(clock.total) + ' FPS ' + fps;
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
  $("#hit-shots-baddy").text(hitShotsBaddy);
  $("#missed-shots").text(missedShots);
  $("#turns").text(totalTurns);
}
let isRunning = false;
async function goTooFast() {
  if (isRunning) {
    return;
  }
  isRunning = true;
  await mainLoop();
  loopCount++;

  if (loopCount <= 1) {
    const weights = model.actor.getWeights();
    const criticWeights = model.critic.getWeights();
    const weightsData = weights.map(weight => weight.dataSync());
    const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

    $("#weights").text(weightsData);
    $("#criticWeights").text(criticWeightsData);
    $("#current-state").text(states.join(', '));

    await createOrUpdateRewardChart(rewardOverTime, batchSize)
    await createOrUpdateLossesChart();

  }
  isRunning = false;
  if (loopCount >= batchSize * 10 || !continueLoop) {
    if (loopCount >= batchSize * 10)
      loopCount = 0;
    //if(!ctx.isDummy)
    ctx = canvas.getContext('2d');
    requestAnimationFrame(goTooFast);
  } else if (continueLoop) {
    // don't draw. just keep going and trian the model
    ctx = {
      isDummy: true,
      beginPath: function () { },
      arc: function () { },
      fill: function () { },
      stroke: function () { },
      // Add any other methods you use
    };

    await goTooFast();
  }
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

async function createOrUpdateRewardChart(rewardOverTime, batchSize) {
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


goTooFast();
//////////////////////////////////////

window.addEventListener('keydown', (event) => {
  if (event.code === 'Space') {
    continueLoop = !continueLoop;
  }
});
$(function () {

  $('#gameSpeed').change(function () {
    gameSpeed = $(this).val();
  }).val(gameSpeed);


  $("#help").click(function () {
    $(this).hide();
  });
  const windowArea = $(window).width() * $(window).height();
  const blockNum = 0;// windowArea / 50000;
  for (let i = 0, l = blockNum; i < l; i++) {
    blocks.push(new Square({}));
  }
  const maxAgents = 15;//windowArea / 10000;
  const numHumans = 1;
  for (let i = 0, l = maxAgents; i < l; i++) {
    agents.push(new Agent({
      id: i + 1,
      type: i < numHumans ? 'human' : 'zombie',
      viewDist: 1000,
      pos: {
        x: canvas.width * Math.random(),
        y: canvas.height * Math.random()
      }
    }));

  }
  const humans = agents.filter(a => a.isHuman === true);
  humans.forEach(h => {
    h.items = agents.filter(a => a.id !== h.id);
    h.viewDist = 1000;
  });

})
