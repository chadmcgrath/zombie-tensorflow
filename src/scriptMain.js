
/* global $ */
/* global Chart */
/* global PPO */
//tf.enableDebugMode();

/* global tf */
tf.setBackend('cpu');


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
// Hyperparameters


const numActions = 16;

let batchSize = +$('#slider-batch').val();
let epsilonGreedy = 0;
const baseReward = 1;
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
// the original didn't handle buildings properly cuz the original code didn't need to.
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

    this.rad = 10;
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
    this.v = new Vec(0, 0);
    //
    this.state = config.state || 'idle';
    this.viewDist = config.viewDist || 1000;
    this.viewFov = (config.viewFov || this.type === 'human' ? 90 : 90) * oneRad;
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
                if(ei===0)
                    a.target = res.agent;
                // this is to check for an intermittent bug
                if (ei === 0 && !res.agent) {
                    let isCollisonMain = false;
                    const p1 = new Vec(a.pos.x, a.pos.y);
                    const p2 = new Vec(a.pos.x + a.dir.x * eyeMaxRange, a.pos.y + a.dir.y * eyeMaxRange);
                    for (let i = 0, n = a.items.length; i < n; i++) {
                        var it = a.items[i];

                        var res2 = line_point_intersect(p1, p2, it.p, it.rad);
                        isCollisonMain = res2;
                    }

                    if (isCollisonMain !== false && !res.agent) {
                        console.error("it seems a wall got in the way of the agent's vision when it should not have.");
                        //// eslint-disable-next-line no-unused-vars
                        //var res3 = stuff_collide(a, eyep, true, true);
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
            eyeStates.push(sr * currentEyeAnglePointing.x / e.max_range, sr * currentEyeAnglePointing.y / e.max_range, type);
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

Agent.prototype.logic = async function (ctx, clock, action, agentExperienceResult) {
    let isMoving = true;
    var batchValue = $('#slider-batch').val();
    var epsilonValue = 0;//$('#slider-epsilon').val();

    epsilonGreedy = +epsilonValue;
    batchSize = +batchValue;
    // // Setters
    // $('#slider-batch').val(8); // replace 8 with the value you want to set
    // $('#slider-lr').val(0.01); // replace 0.01 with the value you want to set
    // $('#slider-epsilon').val(0.2); // replace 0.2 with the value you want to set
    const seen = this.see();

    // bite and maybe convert humans to zombie
    if (this.isHuman === false) {
        if (seen.some(s => s.agent.isHuman) === false) {
            this.state = 'idle';
            this.nextTimer = 3 + Math.random() * 10;
            this.newDir = randomAngle();
            this.state = 'idle';
        }
        for (let i = 0, l = seen.length; i < l; i++) {
            if (seen[i].agent.isHuman === true && seen[i].dist < this.rad) {
                // change to remove a hitpoint
                --seen[i].agent.currentHp;

                //tf ml reward
                seen[i].agent.rewardSignal = seen[i].agent.rewardSignal - baseReward *4;
                negRewards = negRewards - baseReward * 4;

                if (seen[i].agent.currentHp < 1) {
                    console.log('zombifying human' + seen[i].agent.id);
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
            const agentType = seen[i].agent.type;

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
    }

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

        // Perform action and get new state and reward
        states = [];
        states.push(...this.getVision());
        const target = !this.target ? 0 : this.target.isHuman ? 1 : -1;
        states.push(target);

        let actionSelected;
        if (Math.random() < epsilonGreedy) {
            // Take a random action
            actionSelected = Math.floor(Math.random() * 16);
            const actionsOneHot = Array(numActions).fill(.1);
            actionsOneHot[actionSelected] = .85;
        } else {
            actionSelected = action;
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
    }


    if (this.ring) {
        this.ring += clock.delta * 20;
        if (this.ring > 100) this.ring = 0;
    }
    this.nextTimer -= clock.delta;

    var speed = isMoving ? (this.speed) * 10 : 0;

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
                this.rewardSignal = this.rewardSignal - baseReward / 10;
                negRewards = negRewards - baseReward / 10;
                //this.newDir = this.intersect[0].n;
                this.dir = randomAngle();
                break;
            }
            else if (this.intersect[0].dist <= this.viewDist) {
            }
            else {
                this.intersect = false;
            }
        }
    }

    // if we hit a wall turn arround
    // bouncing it by 50 might be kinda high but i don't want 0 magnitude inputs for the eyes x, y
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
        this.rewardSignal = this.rewardSignal - baseReward / 10;
        negRewards = negRewards - baseReward / 10;
        normalize(this.dir);
    }


    if (this.isHuman === true && states.length > 0) {
        agentExperienceResult.newObservation = states;
        totalRewards += this.rewardSignal;
        rewardOverTime.push(this.rewardSignal);
        $("#rewardTotal").text(totalRewards);
        $("#neg-rewards").text(negRewards);
        agentExperienceResult.reward = this.rewardSignal;
        this.rewardSignal = 0;
    }
}
Agent.prototype.shoot = (agent) => {

    // Draw red line to the closest baddy

    ctx.beginPath();
    ctx.lineWidth = 6;

    ctx.moveTo(agent.pos.x, agent.pos.y);

    const closestTarget = agent.target;
    if (closestTarget) {
        if (closestTarget.isHuman) {
            ctx.strokeStyle = 'orange';
            agent.rewardSignal -= baseReward;
            negRewards -= baseReward;
            hitShotsHuman += 1;

        }
        else {
            ctx.strokeStyle = 'red';
            agent.rewardSignal += baseReward;
            hitShotsBaddy += 1;
        }
        ctx.lineTo(closestTarget.pos.x, closestTarget.pos.y);
        closestTarget.currentHp--;
        closestTarget.state = 'idle';
        if (closestTarget.currentHp < 1)

            closestTarget.ring = 5;
        if (closestTarget.hp < 1) {
            console.log('killed target' + closestTarget.id +closestTarget.type);
            //todo: remove or create some kinda goodie
            //closestBaddy.agent.viewFov = this.viewFov;
            //closestBaddy.agent.viewFovD2 = this.viewFovD2;
            closestTarget.speed = 0;
            //closestBaddy.agent.turnSpeed = this.turnSpeed;

            closestTarget.ring = 5;
            agents = agents.filter(agent => agent !== closestTarget);
            agent.items - agent.items.filter(agent => agent !== closestTarget);
        }
    }
    else {

        // missed! purple line is missed shot. the agent did not move but shot nothing.
        // to do: for now, we disgourage it from stopping and missing.
        agent.rewardSignal = agent.rewardSignal - baseReward / 10;
        negRewards = negRewards - baseReward / 10;
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

    for (let i = 0, l = blocks.length; i < l; i++) {
        blocks[i].draw(ctx);
    }

    for (let i = 0, l = agents.length; i < l; i++) {
        if (agents[i].type === 'human') hCnt++;
        if (agents[i].type === 'zombie') zCnt++;
        // this will end up  taking the last human's observation for the NN
        await agents[i].logic(ctx, clock, action, agentExperienceResult);
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
    $("#hit-shots-human").text(hitShotsHuman);
    $("#hit-shots-baddy").text(hitShotsBaddy);
    $("#missed-shots").text(missedShots);
    $("#turns").text(totalTurns);
}
class Env {
    constructor() {
        this.actionSpace = {
            'class': 'Discrete',
            'n': 16,
        }
        this.observationSpace = {
            'class': 'Discrete',
            'shape': 91,
            'dtype': 'float32'
        }
        this.resets = 0
    }
    async step(action) {
        if (Array.isArray(action)) {
            action = action[0]
        }
        if (isRunning) {
            return;
        }
        isRunning = true;
        //hack move zombies  on first step, then move 1 human perstep
        const agentExperienceResult = {
            newObservation: null,
            reward: null
        };
        loopCount++;
        if (loopCount >= batchSize * 10 || !continueLoop) {
            if (loopCount >= batchSize * 100) {
                continueLoop = false;
            }
            if (loopCount >= batchSize * 10) {
                loopCount = 0;

            }

            if (loopCount <= 1) {
                // const weights = model.actor.getWeights();
                // const criticWeights = model.critic.getWeights();
                // const weightsData = weights.map(weight => weight.dataSync());
                // const criticWeightsData = criticWeights.map(criticWeight => criticWeight.dataSync());

                // $("#weights").text(weightsData);
                // $("#criticWeights").text(criticWeightsData);
                // $("#current-state").text(states.join(', '));

                createOrUpdateRewardChart(rewardOverTime, batchSize)
                //await createOrUpdateLossesChart();

            }

            ctx = canvas.getContext('2d');
            isRunning = false;
            await requestAnimationFrameAsync(async (time) => await mainLoop(time, action, agentExperienceResult));
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
            isRunning = false;
            // this isn't detecing the spacebar while the thrrad is blocked by the loop
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
        this.i = 0
        return new Array(this.observationSpace.shape).fill(.1);
    }
}
function requestAnimationFrameAsync(func) {
    return new Promise((resolve) => {
        requestAnimationFrame((time) => {
            resolve(func(time));
        });
    });
}

let isRunning = false;
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


//////////////////////////////////////

window.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
        continueLoop = !continueLoop;
    }
});
let ppo = null;

(async function () {

    $('#gameSpeed').change(function () {
        gameSpeed = $(this).val();
    }).val(gameSpeed);


    $("#help").click(function () {
        $(this).hide();
    });
    // eslint-disable-next-line no-unused-vars
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
    const env = new Env();
    const config = {
        nSteps: 512,                 // Number of steps to collect rollouts
        nEpochs: 10,                 // Number of epochs for training the policy and value networks
        policyLearningRate: .0005,    // Learning rate for the policy network
        valueLearningRate: .0005,     // Learning rate for the value network
        clipRatio: 0.2,              // PPO clipping ratio for the objective function
        targetKL: 0.01,              // Target KL divergence for early stopping during policy optimization
        netArch: {
            'pi': [90],          // Network architecture for the policy network
            'vf': [90]           // Network architecture for the value network
        },
        activation: 'tanh',          // Activation function to be used in both policy and value networks
        verbose: 0                   // Verbosity level (0 for no logging, 1 for logging)
    }
    ppo = new PPO(env, config);
    //ppo.actor = await tf.loadLayersModel('indexeddb://zombie-actor-may20');
    //ppo.critic = await tf.loadLayersModel('indexeddb://zombie-critic-may20');

    await ppo.learn({
        'totalTimesteps': 1000000,
        'callback': {
            'onTrainingStart': function (p) {
                console.log(p.config)
            }
        }
    })


})();