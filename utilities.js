/* eslint-disable no-unused-vars */
/* global Chart */
/* global tf */
const pi2 = Math.PI * 2;
var EPS = 0.01;

function project(v, d, t) {
    return new Vec(v.x + d.x * t, v.y + d.y * t);
}

function randomAngle(s) {
    var r = Math.random() * pi2;
    if (!s) s = 1;
    return new Vec(Math.cos(r) * s, Math.sin(r) * s);
}

function fixAngle(d) {
    while (d < 0) d += pi2;
    while (d >= pi2) d -= pi2;
    return d;
}
function createGrid(ctx, minX, maxX, minY, maxY, z) {
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    const squares = [];

    for (let i = 20; i < canvasWidth; i += maxX + z) {
        for (let j = 20; j < canvasHeight; j += maxY + z) {
            const width = Math.random() * (maxX - minX) + minX;
            const height = Math.random() * (maxY - minY) + minY;
            const square = new Square({
                pos: { x: i + width / 2, y: j + height / 2 },
                width: width,
                height: height
            });
            squares.push(square);
        }
    }

    return squares;
}
function Square(config) {
    this.fill = config.fill || '#CCC';
    this.stroke = config.stroke || '#000';

    const randomHeight = Math.random() * 100;
    const randomWidth = Math.random() * 100;
    
    this.pos = config.pos;
    this.width = config.width || randomWidth+ 20;
    this.height = config.height || randomHeight+ 20;
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
    ctx.rect(this.pos.x - this.width / 2, this.pos.y - this.height / 2, this.width, this.height);
    ctx.fillStyle = this.fill;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = this.stroke;
    ctx.stroke();
};
//these act as obstacles
// function Square(config, ctx) {
//     this.fill = config.fill || '#CCC';
//     this.stroke = config.stroke || '#000';
//     const randomHeight = Math.random() * 100;
//     const randomWidth = Math.random() * 100;
//     const randomX = Math.random() * ctx.canvas.width;
//     const randomY = Math.random() * ctx.canvas.height;
    
//     this.pos = config.pos || {
//         x: randomX,
//         y: randomY
//     };
//     this.width = config.width || randomWidth+ 20;
//     this.height = config.height || randomHeight+ 20;
//     this.bounds = [{
//         x: this.pos.x - this.width / 2,
//         y: this.pos.y - this.height / 2
//     }, {
//         x: this.pos.x + this.width / 2,
//         y: this.pos.y + this.height / 2
//     }]
// }
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
    return [p1, p2];
}

// A 2D vector utility (originally from Karpathy)

const Vec = function (x, y) {
    const vec = {
        x: x?.x !== undefined && x?.y !== undefined ? x.x : x,
        y: x?.x !== undefined && x?.y !== undefined ? x.y : y,
        distFrom: function (v) { return Math.sqrt(Math.pow(this.x - v.x, 2) + Math.pow(this.y - v.y, 2)); },
        length: function () { return Math.sqrt(Math.pow(this.x, 2) + Math.pow(this.y, 2)); },
        add: function (v) { return Vec(this.x + v.x, this.y + v.y); },
        sub: function (v) { return Vec(this.x - v.x, this.y - v.y); },
        rotate: function (a) {  // CLOCKWISE
            return Vec(this.x * Math.cos(a) + this.y * Math.sin(a),
                -this.x * Math.sin(a) + this.y * Math.cos(a));
        },
        getAngle: function () { return Math.atan2(this.y, this.x); },
        getUnit: function () { var d = this.length(); return Vec(this.x / d, this.y / d); },
        scale: function (s) { this.x *= s; this.y *= s; },
        normalize: function () { var d = this.length(); this.scale(1.0 / d); }
    };
    return vec;
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
const line_point_intersect = function (p1, p2, p0, rad) {
    var dx = p2.x - p1.x;
    var dy = p2.y - p1.y;
    var perp_dx = dy; // Perpendicular vector x-component
    var perp_dy = -dx; // Perpendicular vector y-component

    // Direct calculation of perpendicular distance from point to line
    var d = Math.abs(dx * (p1.y - p0.y) - (p1.x - p0.x) * dy) / Math.sqrt(perp_dx * perp_dx + perp_dy * perp_dy);
    if (d > rad) return false;

    // Normalize the perpendicular vector and scale by d
    var len = Math.sqrt(perp_dx * perp_dx + perp_dy * perp_dy);
    perp_dx = (perp_dx / len) * d;
    perp_dy = (perp_dy / len) * d;

    // Calculate the intersection point
    var up = new Vec(p0.x + perp_dx, p0.y + perp_dy);

    // Use the axis with the greater delta to calculate ua
    var ua;
    if (Math.abs(dx) > Math.abs(dy)) {
        ua = (up.x - p1.x) / dx;
    } else {
        ua = (up.y - p1.y) / dy;
    }
    return { ua: ua, up: up };
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
const stuff_collide = (agent, p2, blocks, check_walls, check_items) => {
    var minres = false;
    p2 = new Vec(p2.x, p2.y);
    const p1 = agent.pos;
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
            var res = line_point_intersect(p1, p2, it.pos, it.rad);
            if (res && res.ua > 0 && res.ua < 1) {
                res.vx = it.v.x; // velocty information
                res.vy = it.v.y;

                res.type = it.isHuman ? 1 : -1; // store type of item
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

// Image loading utilities
function loadImageFrames(basePath, count, prefix = '') {
    const frames = [];
    for (let i = 0; i < count; i++) {
        frames[i] = new Image();
        frames[i].src = `${basePath}/${prefix}${i}.png`;
    }
    return frames;
}

// Animation utilities
function drawRotatedImage(ctx, image, x, y, width, height, angle) {
    ctx.save();
    ctx.translate(x + width / 2, y + height / 2);
    ctx.rotate(angle);
    ctx.drawImage(image, -width / 2, -height / 2, width, height);
    ctx.restore();
}

// Async utilities
function requestAnimationFrameAsync(func) {
    return new Promise((resolve) => {
        requestAnimationFrame((time) => {
            resolve(func(time));
        });
    });
}

// Chart utilities
function createOrUpdateChart(chartInstance, ctx, data, labels, chartConfig) {
    if (!chartInstance) {
        return new Chart(ctx, {
            type: chartConfig.type || 'line',
            data: {
                labels: labels,
                datasets: chartConfig.datasets
            },
            options: chartConfig.options || {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } else {
        chartInstance.data.labels = labels;
        chartInstance.data.datasets.forEach((dataset, index) => {
            if (chartConfig.datasets[index]) {
                dataset.data = chartConfig.datasets[index].data;
            }
        });
        chartInstance.update();
        return chartInstance;
    }
}

// Model persistence utilities
const ModelPersistence = {
    async save(actor, critic, name = 'zed-tf') {
        await actor.save(`indexeddb://actor-${name}`);
        await critic.save(`indexeddb://critic-${name}`);
        console.log(`Models saved to IndexedDB: ${name}`);
        return this.saveToFiles(actor, critic, name);
    },

    async load(name = 'zed-tf') {
        const actor = await tf.loadLayersModel(`indexeddb://actor-${name}`);
        const critic = await tf.loadLayersModel(`indexeddb://critic-${name}`);
        return { actor, critic };
    },

    async saveToFiles(actor, critic, name = 'zed-tf') {
        await actor.save(`downloads://actor-${name}`);
        await critic.save(`downloads://critic-${name}`);
    }
};

// Game state utilities
const GameState = {
    createClock() {
        return {
            total: 0,
            start: 0,
            time: 0,
            delta: 0
        };
    },

    updateClock(clock, time, gameSpeed = 1) {
        if (!time) time = Date.now();
        if (!clock.start) clock.start = time;
        if (clock.time) clock.delta = (time - clock.time) / 1000.0;
        clock.time = time;
        if (clock.delta > 0.1) clock.delta = 0.1;
        if (clock.delta < 0.01) clock.delta = 0.01;
        clock.delta *= gameSpeed;
        return clock;
    }
};

// Eye class for vision system
class Eye {
    constructor(angle) {
        this.angle = angle;
        this.max_range = 1000; // eyeMaxRange constant
        this.sensed_proximity = this.max_range;
        this.sensed_type = 0;
    }
}
