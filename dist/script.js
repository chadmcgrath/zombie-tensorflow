tf.setBackend('cpu');
function circularLoss(yTrue, yPred) {

  // Convert angles to 2D vectors
  //const yTrueVector = [tf.cos(yTrue), tf.sin(yTrue)];
  //const yPredVector = [tf.cos(yPred), tf.sin(yPred)];

  // Use Mean Squared Error loss on the vectors
  return tf.losses.meanSquaredError(yTrue, yPred);
}
class ActorCriticModel {

  constructor(numInputs, numActions, hiddenUnits, gamma, batchSize) {
    this.maxMemorySize = 5000;
    this.numInputs = numInputs;
    this.numActions = numActions;
    this.hiddenUnits = hiddenUnits;

    this.actor = this.createActorModel();
    this.critic = this.createCriticModel();
    this.memory = [];
    this.alpha = 0.6;  // Controls how much prioritization is used
    this.priorities = [];

    
    // Decay rate
    this.decayRate = 0.01;
    this.steps = 1000000;
    this.globalStep = 0;


    this.gamma = gamma;
    this.batchSize = batchSize;
    this.numOutputs = 2;
    this.kernelInitializer = tf.initializers.glorotNormal();
    this.initialLearningRate = 0.005;
  }
  setBatchSize(n) {
    this.batchSize = n;
  }
  // todo: this is broken
  calculateLearningRate() {
    return this.initialLearningRate * Math.pow(this.decayRate, Math.floor(this.globalStep / this.steps));
  }
  // todo: this is broken
  policyGradientLoss = (actions, advantages) => {
    const logProbs = tf.log(actions.add(1));
    return tf.losses.computeWeightedLoss(logProbs, advantages);
  };

  createActorModel() {
    const stateInput = tf.input({ shape: [this.numInputs] });

    const hidden = tf.layers.dense({ units: this.hiddenUnits, activation: 'relu', kernelInitializer: this.kernelInitializer }).apply(stateInput);
    const hidden2 = tf.layers.dense({ units: this.hiddenUnits / 2, activation: 'relu', kernelInitializer: this.kernelInitializer }).apply(hidden);
    // Separate output layers for each action
    const continuousAction = tf.layers.dense({ units: 1, activation: 'tanh', kernelInitializer: this.kernelInitializer }).apply(hidden2);  // Q-value for continuous action
    const binaryAction = tf.layers.dense({ units: 1, activation: 'sigmoid', kernelInitializer: this.kernelInitializer }).apply(hidden2);  // Q-value for binary action

    const model = tf.model({ inputs: stateInput, outputs: [continuousAction, binaryAction] });

    const optimizer = tf.train.adam(this.initialLearningRate);

    model.compile({ optimizer: optimizer, loss: ['meanSquaredError', 'binaryCrossentropy'] });

    return model;
  }
  async fitActor(states, actions, advantagesContinuous, advantagesBinary) {
    ++this.globalStep;
    const statesTensor = states instanceof tf.Tensor ? states : tf.tensor2d(states);
    const advantagesContinuousTensor = advantagesContinuous instanceof tf.Tensor ? advantagesContinuous : tf.tensor2d(advantagesContinuous.map(x => [x]));
    const advantagesBinaryTensor = advantagesBinary instanceof tf.Tensor ? advantagesBinary : tf.tensor2d(advantagesBinary.map(x => [x]));
    const numSamples = Math.min(this.batchSize, this.getNumSamples(states), this.getNumSamples(actions), this.getNumSamples(advantagesContinuous), this.getNumSamples(advantagesBinary));
    const statesBatch = statesTensor.slice([0, 0], [numSamples, statesTensor.shape[1]]);
    const advantagesContinuousBatch = advantagesContinuousTensor.slice([0, 0], [numSamples, 1]);
    const advantagesBinaryBatch = advantagesBinaryTensor.slice([0, 0], [numSamples, 1]);

    await this.actor.trainOnBatch([statesBatch], [advantagesContinuousBatch, advantagesBinaryBatch]);
  }
  createCriticModel() {
    const stateInput = tf.input({ shape: [this.numInputs] });
    const actionInput = tf.input({ shape: [this.numActions] });

    const stateHidden = tf.layers.dense({ units: this.hiddenUnits, activation: 'relu', kernelInitializer: this.kernelInitializer }).apply(stateInput);
    const stateHidden2 = tf.layers.dense({ units: this.hiddenUnits / 2, activation: 'relu', kernelInitializer: this.kernelInitializer }).apply(stateHidden);
    const actionHidden = tf.layers.dense({ units: this.hiddenUnits, activation: 'relu', kernelInitializer: this.kernelInitializer }).apply(actionInput);
    const actionHidden2 = tf.layers.dense({ units: this.hiddenUnits / 2, activation: 'relu', kernelInitializer: this.kernelInitializer }).apply(actionHidden);

    const merged = tf.layers.concatenate().apply([stateHidden2, actionHidden2]);
    const output1 = tf.layers.dense({ units: 1, activation: 'tanh', kernelInitializer: this.kernelInitializer }).apply(merged);
    const output2 = tf.layers.dense({ units: 1, activation: 'sigmoid', kernelInitializer: this.kernelInitializer }).apply(merged);
    const model = tf.model({ inputs: [stateInput, actionInput], outputs: [output1, output2] });
    model.compile({ optimizer: 'adam', loss: ['meanSquaredError', 'binaryCrossentropy'] });

    return model;
} 
async fitCritic([states, actions], [q1, q2]) {
  const statesTensor = states instanceof tf.Tensor ? states : tf.tensor2d(states);
  const actionsTensor = actions instanceof tf.Tensor ? actions : tf.tensor2d(actions);
  const q1Tensor = q1 instanceof tf.Tensor ? q1 : tf.tensor2d(q1.map(x => [x]));
  const q2Tensor = q2 instanceof tf.Tensor ? q2 : tf.tensor2d(q2.map(x => [x]));

  const numSamples = Math.min(this.batchSize, this.getNumSamples(states), this.getNumSamples(actions), this.getNumSamples(q1), this.getNumSamples(q2));
  const statesBatch = statesTensor.slice([0, 0], [numSamples, statesTensor.shape[1]]);
  const actionsBatch = actionsTensor.slice([0, 0], [numSamples, actionsTensor.shape[1]]);
  const q1Batch = q1Tensor.slice([0, 0], [numSamples, 1]);
  const q2Batch = q2Tensor.slice([0, 0], [numSamples, 1]);

  await this.critic.trainOnBatch([statesBatch, actionsBatch], [q1Batch, q2Batch]);
}
  predictCritic(states, actions) {
    return tf.tidy(() => {
      const statesTensor = Array.isArray(states) ? tf.tensor2d(states).slice([0, 0], [Math.min(batchSize, states.length), states[0].length]) : states;
      const actionsTensor = Array.isArray(actions) ? tf.tensor2d(actions).slice([0, 0], [Math.min(batchSize, states.length), actions[0].length]) : actions;
      const criticOutput = this.critic.predict([statesTensor, actionsTensor]);
      if (Array.isArray(criticOutput) && criticOutput.length === 2) {
        const [criticOutput1, criticOutput2] = criticOutput;
        return [criticOutput1, criticOutput2];
      } else {
        console.error('critic model output is not an array of length 2');
      }
    });
  }
  
  getNumSamples(data) {
    if (data instanceof tf.Tensor) {
      return data.shape[0];
    } else if (Array.isArray(data)) {
      return data.length;
    }
  }

  remember([state, oldActions, reward, nextState, action, tdError]) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift();
      this.priorities.shift();
    }
    const tdErrorValue = isNaN(tdError) ? tdError.dataSync()[0] : tdError;
    this.memory.push([state, oldActions, reward, nextState, action, tdErrorValue]);
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
    let batchOldActions = [];

    for (let [oldStates, oldActions, rewardSignal, states, actions] of sample) {
      batchOldStates.push(oldStates);
      batchActions.push(actions);
      batchRewardSignals.push(rewardSignal);
      batchNewStates.push(states);
      batchOldActions.push(oldActions);
    }

    // Convert to tensors
    const oldStatesTensor = tf.tensor(batchOldStates);
    const oldActionsTensor = tf.tensor(batchOldActions);
    const actionsTensor = tf.tensor(batchActions);
    const rewardSignalsTensor = tf.tensor(batchRewardSignals);
    const newStatesTensor = tf.tensor(batchNewStates);
    const [oldQValueContinuous, oldQValueBinary] = model.predictCritic(oldStatesTensor, oldActionsTensor);
    // Predict Q-values for new states
    const [newQValue1, newQValue2] = model.predictCritic(newStatesTensor, actionsTensor);
    // if (newQValue1.some(isNaN) || newQValue2.some(isNaN)) {
    //   console.error('newQValue1 or newQValue2 contains NaN');
    // }

    // Compute target Q-values
    const targetQValue1 = rewardSignalsTensor.add(newQValue1.mul(this.gamma));
    const targetQValue2 = rewardSignalsTensor.add(newQValue2.mul(this.gamma));

    // Update critic model
    await model.fitCritic([newStatesTensor, actionsTensor], [targetQValue1, targetQValue2]);

    // Update actor model
    const advantageContinuous = targetQValue1.sub(newQValue1);
    const advantageBinary = targetQValue2.sub(newQValue2);
    await model.fitActor(oldStatesTensor, actionsTensor, advantageContinuous, advantageBinary);

    // Compute TD errors for each experience in the batch
    const qValueNextState = newQValue1.add(newQValue2).div(tf.scalar(2)).reshape([this.batchSize]);
    // why are we adding these?
    const qValueCurrent = oldQValueContinuous.add(oldQValueBinary).div(tf.scalar(2)).reshape([this.batchSize]);
    const tdErrors = rewardSignalsTensor.add(qValueNextState.mul(this.gamma)).sub(qValueCurrent);
    const tdErrorsArray = tdErrors.dataSync();
    for (let i = 0; i < sample.length; i++) {
      const tdError = tdErrorsArray[i];

      // Update tdError and priority for existing experience
      if (this.memory[sampleIndices[i]]) {
        this.memory[sampleIndices[i]][5] = tdError;
        const priority = Math.pow(Math.abs(tdError) + 1e-6, this.alpha);  // Recalculate priority
        this.priorities[sampleIndices[i]] = priority;  // Update priority
      } else {
        // Add new experience
        // should no longer happen anymore
        console.log('Adding new experience, this probably shouldnt ever happen');
        this.remember([...sample[i], tdError]);
      }
    }
  }
  getWeightCollection(w) {
    const ret = [];
    weights.forEach((weight, i) => {
      ret.push(weight);
    });
    return ret;
  }

  predictActor(states, actions) {
    let statesTensor = states;
    let actionsTensor = actions;
    return tf.tidy(() => {
      if (!(states instanceof tf.Tensor))
        statesTensor = tf.tensor(states).reshape([-1, 93]);
      if (!(actions instanceof tf.Tensor))
        actionsTensor = tf.tensor(actions).reshape([-1, 2]);

      const actorOutput = this.actor.predict([statesTensor]);

      if (Array.isArray(actorOutput) && actorOutput.length === 2) {
        const [continuousActionTensor, binaryActionTensor] = actorOutput;
        return [continuousActionTensor, binaryActionTensor];
      } else {
        console.error('Actor model output is not an array of length 2');
      }
    });
  }
}


let continueLoop = false;
let loopCount = 0;
let states = [];
let oldStates = []
let actions = [0, 0];
let rewards = [];
let totalRewards = 0;
let rewardOverTime = [];
let wallHitReward = 0;
let baddyHitReward = 0;
let totalTurns = 0;

// Hyperparameters
const numEpisodes = 5000;
const gamma = 0.95;  // Discount factor
const numActions = 2;
const numInputs = 93;
//todo:// hidden to low?

const numHidden = 64;
const batchSize = 512;
const model = new ActorCriticModel(numInputs, 2, 128, gamma, batchSize);
const winWidth = +$(window).width();
const winHeight = +$(window).height();
const maxWinSide = Math.max(winWidth, winHeight);
const eyeMaxRange = 1000;
var oneRad = Math.PI / 180;
var pi2 = Math.PI * 2;
var gameSpeed = 5;
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

function add(v1, v2) {
  return {
    x: v1.x + v2.x,
    y: v1.y + v2.y
  };
}

function scale(v, s) {
  return {
    x: v.x * s,
    y: v.y * s
  };
}

function negate(v) {
  return {
    x: v.x * -1,
    y: v.y * -1
  };
}

function project(v, d, t) {
  return {
    x: v.x + d.x * t,
    y: v.y + d.y * t
  };
}

function combine(v1, v2, f1, f2) {
  return {
    x: v1.x * f1 + v2.x * f2,
    y: v1.y * f1 + v2.y * f2
  };
}

function lerp(v1, v2, t) {
  return {
    x: v1.x * (1 - t) + v2.x * t,
    y: v1.y * (1 - t) + v2.y * t
  };
}

function dot(v1, v2) {
  return (v1.x + v2.x) * (v1.y + v2.y);
}

function cross(v) {
  return {
    x: -v.y,
    y: v.x
  };
}

function reflect(v, n) {
  var vdotn = -2 * dot(v, n);
  return normalize(project(v, n, vdotn));
}

function equals(v1, v2) {
  return v1.x === v2.x && v1.y === v2.y;
}

function rotate(v, r) {
  return {
    x: v.x * Math.cos(r) - v.y * Math.sin(r),
    y: v.x * Math.sin(r) + v.y * Math.cos(r)
  };
}

function pointFromRad(r, s) {
  if (!s) s = 1;
  return {
    x: Math.cos(r) * s,
    y: Math.sin(r) * s
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

function turnTo(dir1, dir2) {
  if (equals(dir1, dir2)) return clone(dir1);
  return lerp(dir1, dir2, 1);
}

var mouse = {
  x: 0,
  y: 0,
  down: false,
  msg: ''
};
// Wall is made up of two points
var Wall = function (p1, p2) {
  this.p1 = p1;
  this.p2 = p2;
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

// line intersection helper function: does line segment (p1,p2) intersect segment (p3,p4) ?
var line_intersectold = function (p1, p2, p3, p4) {
  var denom = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
  if (denom === 0.0) { return false; } // parallel lines
  var ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denom;
  var ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denom;
  if (ua > 0.0 && ua < 1.0 && ub > 0.0 && ub < 1.0) {
    var up = new Vec(p1.x + ua * (p2.x - p1.x), p1.y + ua * (p2.y - p1.y));
    return { ua: ua, ub: ub, up: up }; // up is intersection point
  }
  return false;
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
  var v = new Vec(p2.y - p1.y, -(p2.x - p1.x)); // perpendicular vector
  var d = Math.abs((p2.x - p1.x) * (p1.y - p0.y) - (p1.x - p0.x) * (p2.y - p1.y));
  d = d / v.length();
  if (d > rad) { return false; }

  v.normalize();
  v.scale(d);
  var up = p0.add(v);
  if (Math.abs(p2.x - p1.x) > Math.abs(p2.y - p1.y)) {
    var ua = (up.x - p1.x) / (p2.x - p1.x);
  } else {
    var ua = (up.y - p1.y) / (p2.y - p1.y);
  }
  if (ua > 0.0 && ua < 1.0) {
    return { ua: ua, up: up };
  }
  return false;
}

function lineIntersectsLine(a, b, c, d) {
  let cross = (d.y - c.y) * (b.x - a.x) - (d.x - c.x) * (b.y - a.y);
  if (cross === 0) return false;

  let t1 = ((d.x - c.x) * (a.y - c.y) - (d.y - c.y) * (a.x - c.x)) / cross;
  let t2 = ((b.x - a.x) * (a.y - c.y) - (b.y - a.y) * (a.x - c.x)) / cross;

  return t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1;
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
  const p1 = agent.p;
  // collide with walls
  if (check_walls) {
    const windowWalls = {
      bounds: [{ x: 0, y: 0, width: window.innerWidth, height: 0 }, // top
      // { bounds: { x: 0, y: window.innerHeight, width: window.innerWidth, height: 0 } }, // bottom
      // { bounds: { x: 0, y: 0, width: 0, height: window.innerHeight }}, // left
      { x: window.innerWidth, y: window.innerHeight, width: 0, height: window.innerHeight }]
    }; // right

    //const windowWallsXY = { bounds: windowWalls.map(wall => {wall.bounds.x, wall.bounds.y}) }
    // const windowWalls = [
    //   { bounds: { x: 0, y: 0, width: window.innerWidth, height: 0 }} , // top
    //   { bounds: { x: 0, y: window.innerHeight, width: window.innerWidth, height: 0 }} , // bottom
    //   { bounds: { x: 0, y: 0, width: 0, height: window.innerHeight }} , // left
    //   { bounds: { x: window.innerWidth, y: 0, width: 0, height: window.innerHeight }} // right
    // ];
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
      // if (res) {
      //   res.type = -.1; // is wall
      //   if (!minres) { minres = res; }
      //   else {
      //     // check if its closer
      //     if (res.ua < minres.ua) {
      //       // if yes replace it
      //       minres = res;
      //     }
      //   }
      // }
    }
  }

  // collide with items
  if (check_items) {
    for (var i = 0, n = agent.items.length; i < n; i++) {
      var it = agent.items[i];
      var res = line_point_intersect(p1, p2, it.p, it.rad);
      if (res) {
        res.type = it.type.isHuman ? 1 : -1; // store type of item
        res.vx = it.v.x; // velocty information
        res.vy = it.v.y;
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

  const maxHp = 50000000;
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

  this.rad = 10;
  this.speed = config.speed || this.type === 'human' ? 2 + Math.random() : 1 + Math.random();
  this.turnSpeed = config.turnSpeed || this.type === 'human' ? oneRad * 2 : oneRad;
  this.dir = randomAngle();
  this.newDir = clone(this.dir);

  //todo: remove duplicate position
  Object.defineProperty(this, 'angle', {
    get: function () {
      return this.dir;
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
  for (var i = 0, n = this.agents.length; i < n; i++) {
    var a = this.agents[i];
    const pos = a.p;
    const oPointAngle = new Vec(Math.cos(a.oangle), Math.sin(a.oangle));
    const angle = a.angle;
    const agentRads = Math.atan2(angle.x, angle.y);
    if (isNaN(agentRads)) {
      console.error('agentRads is NaN');
    }

    for (var ei = 0, ne = a.eyes.length; ei < ne; ei++) {
      var e = a.eyes[ei];
      const eangle = e.angle;
      // we have a line from p to p->eyep
      var eyep = new Vec(pos.x + e.max_range * Math.sin(agentRads + eangle),
        pos.y + e.max_range * Math.cos(agentRads + eangle));
      if (isNaN(eyep.x)) {
        console.error('eyep.x is NaN');
      }
      var res = stuff_collide(a, eyep, true, true);
      if (res) {
        // eye collided with wall
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
      if (ei === 0) ctx.strokeStyle = "blue";
      ctx.beginPath();

      ctx.moveTo(pos.x, pos.y);
      const sr = e.sensed_proximity;
      const eyeAngle = rightLeftAngle(agentRads + eangle);
      const lineToX = pos.x + sr * Math.sin(eyeAngle);
      const lineToY = pos.y + sr * Math.cos(eyeAngle);
      ctx.lineTo(lineToX, lineToY);
      ctx.stroke();
      
      let type = e.sensed_type;
      if (type === 'zombie')
        type = -1;
      if (type === 'human')
        type = 1;

      // add to state for ML
      // closeness of proximity is probably easier to process than distance
      const closeness = 1 - e.sensed_proximity / e.max_range
      // tensorflow inputs
      eyeStates.push(eyeAngle/Math.PI, closeness, type);
    }
  }
  // tensorflow inputs
  return eyeStates;
}
// converts to bewett - Pi and Pi
function rightLeftAngle(d) {
  while (d < -Math.PI) d += 2 * Math.PI;
  while (d >= Math.PI) d -= 2 * Math.PI;
  return d;
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
Agent.prototype.distTo2 = function (o) {
  var dx = o.x - this.pos.x;
  var dy = o.y - this.pos.y;
  return dx * dx + dy * dy;
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
    // if (good) {
    //   const angle = normalize(sub(a.pos, this.pos))
    //   viewBlocked = false;
    //   for (var wi = 0, wl = blocks.length; wi < wl; wi++) {
    //     // todo: why was this only = not ==
    //     if (walls = blocks[wi].rayIntersect(a.pos, angle)) {
    //       if (walls && walls[0].dist > 0 && walls[0].dist < d) {
    //         viewBlocked = true;
    //         break;
    //       }
    //     }
    //   }
    //   if (!viewBlocked)
    //     seen.push({
    //       agent: a,
    //       dist: d,
    //       angle: angle
    //     });
    // }
  }

  if (seen.length > 1) seen.sort(function (a, b) {
    if (a.dist === b.dist) return 0;
    return a.dist < b.dist ? -1 : 1;
  });
  if (mouse.down) {
    a = {
      type: 'mouse',
      pos: mouse
    };
    var d = this.distTo(a.pos);
    if (d <= this.viewDist) {
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
      //mouse.msg = a1.toFixed(1) + " " + ato.toFixed(1) + " " + a2.toFixed(1);
      var good = false;
      if (a2 - a1 > Math.PI) {
        if (ato <= a1 || ato > a2) good = true;
      } else {
        if (ato >= a1 && ato <= a2) good = true;
      }
      if (good)
        seen.unshift({
          agent: a,
          dist: d,
          angle: normalize(sub(a.pos, this.pos))
        });
    } // mouse.down
  }

  return seen;
}

Agent.prototype.logic = async function (ctx, clock) {

  this.op = this.pos;
  var seen = this.see();

  // convert humans to zombie
  if (this.type === 'zombie' && seen.length) {
    for (var i = 0, l = seen.length; i < l; i++) {
      if (seen[i].agent.isHuman === true && seen[i].dist < 10) {
        // change to remove a hitpoint
        --seen[i].agent.currentHp;

        //tf ml reward
        --seen[i].agent.rewardSignal;
        --agents.filter(a => a.id === seen[i].agent.id)[0].rewardSignal;

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
  for (var i = 0, l = seen.length; i < l; i++) {
    const agentType = seen[i].agent.type;
    // follow mouse
    if (agentType === 'mouse') {
      this.state = 'mouse';
      this.nextTimer = 3;
      this.newDir = seen[i].angle;
      break;
    }


    if (this.type === 'zombie') {
      if (!seen) {

        this.nextTimer = 3 + Math.random() * 10;
        this.newDir = randomAngle();
        this.state = 'idle';
        //continue;

      }
      // attack human
      if (agentType === 'human') {
        this.state = 'attack';
        this.nextTimer = 5;
        this.newDir = seen[i].angle;
        break;
      }
      // follow other zombie
      if (this.state === 'idle' && agentType === 'zombie' && Math.random() > 0.9) { // && seen[i].agent.state==='attack') {
        this.nextTimer = 5;
        this.newDir = seen[i].angle;
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
      states.push(Math.atan2(this.dir.x, this.dir.y) / Math.PI, this.pos.x / maxWinSide, this.pos.y / maxWinSide);
    }
    const [continuousAction, binaryAction] = model.predictActor([states], [actions]);


    // Perform action and get new state and reward
    oldActions = [...actions];
    oldStates = [...states];
    states = [];
    states.push(...this.getVision());
    states.push(Math.atan2(this.dir.x, this.dir.y) / Math.PI, this.pos.x / maxWinSide, this.pos.y / maxWinSide);

    let epsilon = .15;

    actions = [];
    if (Math.random() < epsilon) {
      // Take a random action
      this.newDir = randomAngle();
      const rndMove = Math.random();
      isMoving = rndMove < .5;
      actions.push(Math.atan2(this.newDir.x, this.newDir.y), rndMove);
    } else {

      const continuousActionVal = continuousAction.dataSync()[0];
      const binarayActionVal = binaryAction.dataSync()[0];
      actions.push(continuousActionVal, binarayActionVal);
      //choose current action, [rotate, move or shoot]        
      // -pi to pi
      const newAngle = (continuousActionVal) * Math.PI;//continuousActionVal * Math.PI;
      isMoving = binarayActionVal < .5;
      this.newDir = new Vec(Math.cos(newAngle), Math.sin(newAngle)).getUnit();

      const unitOldDir = new Vec(this.dir.x, this.dir.y).getUnit();
      const newVec = unitOldDir.rotate(newAngle);
      this.dir = newVec;
      
    }
    if(!isMoving)
      this.shoot(this, seen);
  }
  

  if (this.ring) {
    this.ring += clock.delta * 20;
    if (this.ring > 100) this.ring = 0;
  }
  this.nextTimer -= clock.delta;
  //zombies when timer runs out go back to random wandering
  if (this.isHuman === false && this.nextTimer <= 0) {
    this.nextTimer = 3 + Math.random() * 3;
    this.dir=randomAngle();
    this.state = 'idle';
  }
  if (!this.da) this.da = 0;
  //this.da+=0.01;
  //this.newDir=pointFromRad(this.da);
  // turn twards desired direction
  // todo. find out max turn rate and if it applies here for humans (isMoving)
  
  if (isNaN(this.dir.x) || isNaN(this.dir.y)) {
    console.error('this.dir contains NaN');
  }
  var speed = isMoving ? (this.speed) * 10 : 0;

  // get velociy
  var vx = this.dir.x * speed * clock.delta;
  var vy = this.dir.y * speed * clock.delta;
  // move
  this.pos.x += vx;
  this.pos.y += vy;
  // prevent walking through blocks
  for (var i = 0, l = blocks.length; i < l; i++) {
    if (this.intersect = blocks[i].rayIntersect(this.pos, this.dir)) {
      if (this.intersect[0].dist <= 0 && this.intersect[1].dist > 0) {
        this.pos = this.intersect[0].pos;
        this.rewardSignal = this.rewardSignal - .1;
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
    normalize(this.dir);
  }


  if (this.isHuman === true && states.length > 0) {

    model.remember([oldStates, oldActions, this.rewardSignal, states, actions, .1]);
    if(totalTurns % batchSize === 0){
      await model.train(oldStates, oldActions, this.rewardSignal, states, actions, batchSize);
      const saveResult = await model.save('localstorage://zombie-sim-model-ac-1');
    }

    totalRewards += this.rewardSignal;
    rewardOverTime.push(this.rewardSignal);
    $("#rewardTotal").text(totalRewards);
    this.rewardSignal = 0;
  }
}
Agent.prototype.shoot = (agent, seen) => {
  const shootAngle = .5;//rads

  const baddies = agent.items.filter(item => item.isHuman === false && seen.some(seenItem => seenItem.agent.id === item.id));
  const newDirNorm = normalize(agent.dir);

  const baddiesInRange = baddies.filter(baddy => {
    const baddyDir = { x: baddy.pos.x - agent.pos.x, y: baddy.pos.y - agent.pos.y };
    const baddyDirNorm = normalize(baddyDir);

    //  Calculate the dot product of the normalized vectors
    const dotProduct = newDirNorm.x * baddyDirNorm.x + newDirNorm.y * baddyDirNorm.y;

    // Check if the cosine of the angle is greater than the cosine of n radians
    return dotProduct > Math.cos(shootAngle);
  });

  ctx.beginPath();
  ctx.lineWidth = 3;
  ctx.arc(agent.pos.x, agent.pos.y, 5, 0, 2 * Math.PI, false);
  ctx.fillStyle = 'red';
  ctx.fill();

  // Find the closest baddy
  let closestBaddy = null;
  let minDistance = Infinity;
  baddiesInRange.forEach(baddy => {
    const dx = baddy.pos.x - agent.pos.x;
    const dy = baddy.pos.y - agent.pos.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < minDistance) {
      minDistance = distance;
      closestBaddy = baddy;
    }
  });

  // Draw red line to the closest baddy
  // todo: keep it alive on canvas
  ctx.beginPath();
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'red';
  ctx.moveTo(agent.pos.x, agent.pos.y);
  if (closestBaddy) {
    agent.rewardSignal += .6;
    ctx.lineTo(closestBaddy.pos.x, closestBaddy.pos.y);
    closestBaddy.currentHp--;
    closestBaddy.state = 'idle';
    if (closestBaddy.currentHp < 1)

      closestBaddy.ring = 10;
    if (closestBaddy.hp < 1) {
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
    // missed!
    // to do: for now, we disgourage it from stopping and missing.
    //--agent.rewardSignal;
    ctx.strokeStyle = 'purple';
    ctx.lineTo(agent.pos.x + agent.dir.x * eyeMaxRange, agent.pos.y + agent.dir.y * eyeMaxRange);
  }
  ctx.stroke();

}
Agent.prototype.draw = function (ctx) {

  ctx.beginPath();
  ctx.arc(this.pos.x, this.pos.y, 5, 0, pi2, false);
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
  var dir = project(this.pos, this.dir, 10);
  ctx.beginPath();
  ctx.moveTo(this.pos.x, this.pos.y);
  ctx.lineTo(dir.x, dir.y);
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
var zCnt = hCnt = 0;
var agents = [];
var blocks = [];
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var fps = 0;
var fpsc = 0;
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
  mouse.msg = '';

  for (var i = 0, l = blocks.length; i < l; i++) {
    blocks[i].draw(ctx);
  }

  for (var i = 0, l = agents.length; i < l; i++) {
    if (agents[i].type === 'human') hCnt++;
    if (agents[i].type === 'zombie') zCnt++;
    await agents[i].logic(ctx, clock);
    agents[i].draw(ctx);
  }

  ctx.font = '20pt Calibri';
  ctx.lineWidth = 1;
  ctx.fillStyle = 'black';
  var msg = 'Zed:' + zCnt + '     Hum:' + hCnt + '    ' + 'Time:' + Math.floor(clock.total) + ' FPS ' + fps + ' ' + mouse.msg;
  ctx.fillText(msg, ctx.canvas.width / 3 + 1, 21);
  ctx.fillStyle = 'white';
  ctx.fillText(msg, ctx.canvas.width / 3, 20);

  ctx.beginPath();
  ctx.arc(mouse.x, mouse.y, 3, 0, 2 * Math.PI, false);
  ctx.fillStyle = 'white';
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#FFFFFF';
  ctx.stroke();
  fpsc++;
  totalTurns++;

  $("#turns").text(totalTurns);
}
let oldWeightsData = null;
let oldCriticWeights = null;
function goTooFast() {

  mainLoop().then(() => {
    loopCount++;

    if (loopCount == batchSize) {
      const weights = model.actor.getWeights();
      const criticWeights = model.critic.getWeights();
      const weightsData = weights.map(weight => weight.dataSync());
      const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());
      // let diff = '';
      // if (oldWeightsData !== null) {
      //   for (let i = 0; i < Math.max(oldWeightsData.length, weightsData.length); i++) {
      //     if (oldWeightsData[i] !== weightsData[i]) {
      //       diff += '^';
      //     } else {
      //       diff += ' ';
      //     }
      //   }
      // }
      // console.log('Old: ' + oldWeightsData);
      // console.log('New: ' + weights);
      // console.log('Diff:' + diff);
      //console.log('OldCrit: ' + oldWeights);
      //console.log('NewCrit: ' + eights);
      //console.log('DiffCrit:' + diff);

      $("#weights").text(weightsData);
      $("#criticWeights").text(criticWeightsData);
      $("#current-state").text(states.join(', '));
      oldWeights = [...weightsData];
      oldCriticWeights = [...criticWeightsData];
      oldWeightsData = [...weightsData];
      createOrUpdateRewardChart(rewardOverTime)

    }
    if (loopCount >= batchSize + 1 || !continueLoop) {
      if (loopCount >= batchSize + 1)
        loopCount = 0;
      requestAnimationFrame(goTooFast);
    } else if (continueLoop) {
      goTooFast();
    }

  });
}

let chart; // Declare chart variable outside the function

function createOrUpdateRewardChart(rewardOverTime) {
  let ctx = document.getElementById('rewardOverTimeChart').getContext('2d');

  if (!chart) {
    // If the chart does not exist, create it
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: rewardOverTime.map((_, i) => i), // X-axis labels are just indices
        datasets: [{
          label: 'Reward over time',
          data: rewardOverTime,
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
    chart.data.labels = rewardOverTime.map((_, i) => i);
    chart.data.datasets[0].data = rewardOverTime;
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
// setInterval(function () {
//   fps = fpsc;
//   fpsc = 0;
// }, 1000);
$(function () {

  $('#gameSpeed').change(function () {
    gameSpeed = $(this).val();
  }).val(gameSpeed);
  $(canvas).mousemove(function (event) {
    mouse.x = event.clientX;
    mouse.y = event.clientY;
  });
  $(canvas).mousedown(function (event) {
    mouse.down = true;
  });
  $(canvas).mouseup(function (event) {
    mouse.down = false;
  });


  $("#help").click(function () {
    $(this).hide();
  });
  const windowArea = $(window).width() * $(window).height();
  const blockNum = windowArea / 50000;
  for (var i = 0, l = blockNum; i < l; i++) {
    blocks.push(new Square({}));
  }
  const maxAgents = 10;//windowArea / 10000;
  const numHumans = 1;
  for (var i = 0, l = maxAgents; i < l; i++) {
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
