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
Square.prototype.draw = function(ctx) {
  ctx.beginPath();
  ctx.rect(this.bounds[0].x, this.bounds[0].y, this.width, this.height);
  ctx.fillStyle = this.fill;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = this.stroke;
  ctx.stroke();
};
Square.prototype.pointNormal = function(p) {
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
Square.prototype.rayIntersect = function(o, d) {
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
  return lerp(dir1, dir2, 0.1);
}

var mouse = {
  x: 0,
  y: 0,
  down: false,
  msg: ''
};

function Agent(config) {
  this.type = config.type || 'human';
  this.pos = config.pos || {
    x: 0,
    y: 0
  };
  this.speed = config.speed || this.type === 'human' ? 2 + Math.random() : 1 + Math.random();
  this.turnSpeed = config.turnSpeed || this.type === 'human' ? oneRad * 2 : oneRad;
  this.dir = randomAngle();
  this.newDir = clone(this.dir);
  this.state = config.state || 'idle';
  this.viewDist = config.viewDist || 100;
  this.viewFov = (config.viewFov || this.type === 'human' ? 90 : 45) * oneRad;
  this.viewFovD2 = this.viewFov / 2;
  this.nextTimer = Math.random() * 10;
  this.ring = config.ring || this.type === 'human' ? 0 : 5;
}
Agent.prototype.getColor = function() {
  if (this.state === 'mouse') return '#FF00FF';
  if (this.state === 'panic') return 'yellow';
  if (this.state === 'attack') return 'red';
  if (this.type === 'human') return 'blue';
  if (this.type === 'zombie') return 'green';
  return '#AAAAAA';
};
Agent.prototype.distTo = function(o) {
  var dx = o.x - this.pos.x;
  var dy = o.y - this.pos.y;
  return Math.sqrt(dx * dx + dy * dy);
};
Agent.prototype.distTo2 = function(o) {
  var dx = o.x - this.pos.x;
  var dy = o.y - this.pos.y;
  return dx * dx + dy * dy;
};

Agent.prototype.see = function() {
  var seen = [];
  var a, d, ato;
  for (var i = 0, l = agents.length; i < l; i++) {
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
    if (good)
      seen.push({
        agent: a,
        dist: d,
        angle: normalize(sub(a.pos, this.pos))
      });
  }

  if (seen.length > 1) seen.sort(function(a, b) {
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

Agent.prototype.logic = function(ctx, clock) {
  var seen = this.see();
  var walls, viewBlocked;
  // convert humans to zombie
  if (this.type === 'zombie' && seen.length) {
    for (var i = 0, l = seen.length; i < l; i++) {
      if (seen[i].agent.type === 'human' && seen[i].dist < 10) {
        seen[i].agent.type = this.type;
        seen[i].agent.viewFov = this.viewFov;
        seen[i].agent.viewFovD2 = this.viewFovD2;
        seen[i].agent.speed = this.speed;
        seen[i].agent.turnSpeed = this.turnSpeed;
        seen[i].agent.state = 'idle';
        seen[i].agent.ring=1;
      }
    }
  } // convert to zombie

  for (var i = 0, l = seen.length; i < l; i++) {
    // follow mouse
    if (seen[i].agent.type === 'mouse') {
      this.state = 'mouse';
      this.nextTimer = 3;
      this.newDir = seen[i].angle;
      break;
    }
    // check if what we see is blocked by a wall
    viewBlocked = false;
    for (var wi = 0, wl = blocks.length; wi < wl; wi++) {
      if (walls = blocks[wi].rayIntersect(this.pos, seen[i].angle)) {
        if (walls[0].dist > 0 && walls[0].dist < seen[i].dist) {
          viewBlocked = true;
          break;
        }
      }
    }
    if (viewBlocked) continue;
    if (this.type === 'zombie') {
      // attack human
      if (seen[i].agent.type === 'human') {
        this.state = 'attack';
        this.nextTimer = 5;
        this.newDir = seen[i].angle;
        break;
      }
      // follow other zombie
      if (this.state === 'idle' && seen[i].agent.type === 'zombie' && Math.random() > 0.9) { // && seen[i].agent.state==='attack') {
        this.nextTimer = 5;
        this.newDir = seen[i].angle;
      }
    } else { // we are human
      // panic if we see a zombie and run the other way
      if (seen[i].agent.type === 'zombie') {
        this.state = 'panic';
        this.nextTimer = 5;
        this.newDir = scale(seen[i].angle, -1);
        break;
      }
      // panic if we see another panicked human and run in same dir as other human
      if (seen[i].agent.type === 'human' && seen[i].agent.state === 'panic') {
        this.state = 'panic';
        this.nextTimer = 5;
        //this.newDir = seen[i].angle
      }
    }
  } // for i in seen
  if (this.ring){
    this.ring+=clock.delta*20;
    if (this.ring>100) this.ring=0;
  }
  this.nextTimer -= clock.delta;
  // when timer runs out go back to random wandering
  if (this.nextTimer <= 0) {
    this.nextTimer = 3 + Math.random() * 10;
    this.newDir = randomAngle();
    this.state = 'idle';
  }
  if (!this.da) this.da = 0;
  //this.da+=0.01;
  //this.newDir=pointFromRad(this.da);
  // turn twards desired direction
  this.dir = turnTo(this.dir, this.newDir);

  var speed = this.speed * 10;
  // pannic makes humans move faster
  if (this.state === 'panic') speed += 1;
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
        //this.newDir = this.intersect[0].n;
        this.newDir = randomAngle();
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
    this.newDir.x = 1;
    bound = true
  }
  if (this.pos.y < 0) {
    this.pos.y = 0;
    this.newDir.y = 1;
    bound = true;
  }
  if (this.pos.x > ctx.canvas.width) {
    this.pos.x = ctx.canvas.width;
    this.newDir.x = -1;
    bound = true;
  }
  if (this.pos.y > ctx.canvas.height) {
    this.pos.y = ctx.canvas.height;
    this.newDir.y = -1;
    bound = true;
  }
  if (bound) {
    normalize(this.newDir);
  }
};
Agent.prototype.draw = function(ctx) {
  // draw body
  ctx.beginPath();
  ctx.arc(this.pos.x, this.pos.y, 5, 0, pi2, false);
  ctx.fillStyle = this.getColor();
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#FFFFFF';
  ctx.stroke();
  if (this.ring){
    ctx.beginPath();
    ctx.arc(this.pos.x, this.pos.y, this.ring, 0, pi2, false);
    ctx.lineWidth = 1;
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
var fps=0;
var fpsc=0;
var clock = {
  total: 0,
  start: 0,
  time: 0,
  delta: 0
};

function mainLoop(time) {
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
    agents[i].logic(ctx, clock);
    agents[i].draw(ctx, clock);
  }

  ctx.font = '20pt Calibri';
  ctx.lineWidth = 1;
  ctx.fillStyle = 'black';
  var msg='Zed:' + zCnt + '     Hum:' + hCnt + '    ' + 'Time:' + Math.floor(clock.total) +' FPS '+fps+' '+ mouse.msg;
  ctx.fillText(msg, ctx.canvas.width / 3+1, 21);
  ctx.fillStyle = 'white';
  ctx.fillText(msg, ctx.canvas.width / 3, 20);

  ctx.beginPath();
  ctx.arc(mouse.x, mouse.y, 3, 0, 2 * Math.PI, false);
  ctx.fillStyle = 'white';
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#FFFFFF';
  ctx.stroke();
  requestAnimationFrame(mainLoop);
  fpsc++;
}
requestAnimationFrame(mainLoop);
setInterval(function(){
  fps=fpsc;
  fpsc=0;
},1000);
$(function() {
  $('#gameSpeed').change(function() {
    gameSpeed = $(this).val();
  }).val(gameSpeed);
  $(canvas).mousemove(function(event) {
    mouse.x = event.clientX;
    mouse.y = event.clientY;
  });
  $(canvas).mousedown(function(event) {
    mouse.down = true;
  });
  $(canvas).mouseup(function(event) {
    mouse.down = false;
  });

  $(window).resize(function() {
    $(canvas).attr({
      width: $(window).width(),
      height: $(window).height()
    });
  }).resize();
  $("#help").click(function(){
    $(this).hide();
  });
  for (var i = 0, l=$(window).width()*$(window).height()/20000; i < l; i++) {
    blocks.push(new Square({}));
  }
  
  for (var i = 0, l=$(window).width()*$(window).height()/10000; i < l; i++) {
    agents.push(new Agent({
      type: i < 1 ? 'zombie' : 'human',
      pos: {
        x: canvas.width * Math.random(),
        y: canvas.height * Math.random()
      }
    }));
  }
  
})