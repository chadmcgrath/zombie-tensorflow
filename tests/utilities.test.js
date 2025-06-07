/* eslint-env jest */
/* global Vec, Square, project, randomAngle, fixAngle, line_intersect, line_point_intersect, lineIntersectsSquare, stuff_collide, loadImageFrames, drawRotatedImage, requestAnimationFrameAsync, createOrUpdateChart, ModelPersistence, GameState, Eye, createGrid */

// Import the utilities file
require('../utilities.js');

describe('Vector Utilities', () => {
  describe('Vec class', () => {
    test('should create a vector with x and y coordinates', () => {
      const v = Vec(3, 4);
      expect(v.x).toBe(3);
      expect(v.y).toBe(4);
    });

    test('should create a vector from another vector', () => {
      const v1 = Vec(3, 4);
      const v2 = Vec(v1);
      expect(v2.x).toBe(3);
      expect(v2.y).toBe(4);
    });

    test('should calculate distance between vectors', () => {
      const v1 = Vec(0, 0);
      const v2 = Vec(3, 4);
      expect(v1.distFrom(v2)).toBe(5);
    });

    test('should calculate vector length', () => {
      const v = Vec(3, 4);
      expect(v.length()).toBe(5);
    });

    test('should add vectors', () => {
      const v1 = Vec(1, 2);
      const v2 = Vec(3, 4);
      const result = v1.add(v2);
      expect(result.x).toBe(4);
      expect(result.y).toBe(6);
    });

    test('should subtract vectors', () => {
      const v1 = Vec(5, 7);
      const v2 = Vec(2, 3);
      const result = v1.sub(v2);
      expect(result.x).toBe(3);
      expect(result.y).toBe(4);
    });

    test('should rotate vector', () => {
      const v = Vec(1, 0);
      const rotated = v.rotate(Math.PI / 2);
      expect(Math.abs(rotated.x)).toBeCloseTo(0, 5);
      expect(rotated.y).toBeCloseTo(1, 5);
    });

    test('should get angle of vector', () => {
      const v = Vec(1, 1);
      expect(v.getAngle()).toBeCloseTo(Math.PI / 4, 5);
    });

    test('should get unit vector', () => {
      const v = Vec(3, 4);
      const unit = v.getUnit();
      expect(unit.length()).toBeCloseTo(1, 5);
      expect(unit.x).toBeCloseTo(0.6, 5);
      expect(unit.y).toBeCloseTo(0.8, 5);
    });

    test('should scale vector', () => {
      const v = Vec(2, 3);
      v.scale(2);
      expect(v.x).toBe(4);
      expect(v.y).toBe(6);
    });

    test('should normalize vector', () => {
      const v = Vec(3, 4);
      v.normalize();
      expect(v.length()).toBeCloseTo(1, 5);
    });
  });

  describe('project function', () => {
    test('should project vector along direction', () => {
      const v = Vec(0, 0);
      const d = Vec(1, 0);
      const result = project(v, d, 5);
      expect(result.x).toBe(5);
      expect(result.y).toBe(0);
    });
  });

  describe('randomAngle function', () => {
    test('should generate random angle vector', () => {
      const v = randomAngle();
      expect(v.length()).toBeCloseTo(1, 5);
    });

    test('should generate random angle vector with scale', () => {
      const v = randomAngle(2);
      expect(v.length()).toBeCloseTo(2, 5);
    });
  });

  describe('fixAngle function', () => {
    test('should fix negative angles', () => {
      const result = fixAngle(-Math.PI);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(Math.PI * 2);
    });

    test('should fix angles greater than 2π', () => {
      const result = fixAngle(Math.PI * 3);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(Math.PI * 2);
    });

    test('should leave valid angles unchanged', () => {
      const angle = Math.PI;
      const result = fixAngle(angle);
      expect(result).toBe(angle);
    });
  });
});

describe('Square class', () => {
  test('should create a square with default properties', () => {
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    expect(square.pos.x).toBe(50);
    expect(square.pos.y).toBe(50);
    expect(square.width).toBe(100);
    expect(square.height).toBe(100);
    expect(square.bounds).toHaveLength(2);
  });

  test('should calculate bounds correctly', () => {
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    expect(square.bounds[0].x).toBe(0);
    expect(square.bounds[0].y).toBe(0);
    expect(square.bounds[1].x).toBe(100);
    expect(square.bounds[1].y).toBe(100);
  });

  test('should have draw method', () => {
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    expect(typeof square.draw).toBe('function');
  });

  test('should calculate point normal', () => {
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    const normal = square.pointNormal({ x: 0, y: 50 });
    expect(normal.x).toBe(-1);
    expect(normal.y).toBe(0);
  });

  test('should perform ray intersection', () => {
    const square = new Square({
      pos: { x: 50, y: 50 },
      width: 100,
      height: 100
    });
    
    const origin = { x: -10, y: 50 };
    const direction = { x: 1, y: 0 };
    const result = square.rayIntersect(origin, direction);
    
    expect(result).toBeTruthy();
    expect(Array.isArray(result)).toBe(true);
  });
});

describe('Line Intersection Functions', () => {
  describe('line_intersect', () => {
    test('should find intersection of two lines', () => {
      const line1Start = { x: 0, y: 0 };
      const line1End = { x: 10, y: 0 };
      const line2Start = { x: 5, y: -5 };
      const line2End = { x: 5, y: 5 };
      
      const result = line_intersect(line1Start, line1End, line2Start, line2End);
      
      expect(result).toBeTruthy();
      expect(result.up.x).toBe(5);
      expect(result.up.y).toBe(0);
    });

    test('should return false for parallel lines', () => {
      const line1Start = { x: 0, y: 0 };
      const line1End = { x: 10, y: 0 };
      const line2Start = { x: 0, y: 5 };
      const line2End = { x: 10, y: 5 };
      
      const result = line_intersect(line1Start, line1End, line2Start, line2End);
      
      expect(result).toBe(false);
    });

    test('should return false for non-intersecting line segments', () => {
      const line1Start = { x: 0, y: 0 };
      const line1End = { x: 5, y: 0 };
      const line2Start = { x: 10, y: -5 };
      const line2End = { x: 10, y: 5 };
      
      const result = line_intersect(line1Start, line1End, line2Start, line2End);
      
      expect(result).toBe(false);
    });
  });

  describe('line_point_intersect', () => {
    test('should find intersection of line with point radius', () => {
      const p1 = { x: 0, y: 0 };
      const p2 = { x: 10, y: 0 };
      const p0 = { x: 5, y: 2 };
      const rad = 3;
      
      const result = line_point_intersect(p1, p2, p0, rad);
      
      expect(result).toBeTruthy();
      expect(typeof result.ua).toBe('number');
      expect(result.up).toBeTruthy();
    });

    test('should return false when point is too far from line', () => {
      const p1 = { x: 0, y: 0 };
      const p2 = { x: 10, y: 0 };
      const p0 = { x: 5, y: 10 };
      const rad = 3;
      
      const result = line_point_intersect(p1, p2, p0, rad);
      
      expect(result).toBe(false);
    });
  });

  describe('lineIntersectsSquare', () => {
    test('should find intersection of line with square', () => {
      const lineStart = { x: -5, y: 50 };
      const lineEnd = { x: 15, y: 50 };
      const square = new Square({
        pos: { x: 50, y: 50 },
        width: 100,
        height: 100
      });
      
      const result = lineIntersectsSquare(lineStart, lineEnd, square);
      
      expect(result).toBeTruthy();
      expect(typeof result.distance).toBe('number');
    });

    test('should return false when line does not intersect square', () => {
      const lineStart = { x: -5, y: -5 };
      const lineEnd = { x: -5, y: -10 };
      const square = new Square({
        pos: { x: 50, y: 50 },
        width: 100,
        height: 100
      });
      
      const result = lineIntersectsSquare(lineStart, lineEnd, square);
      
      expect(result).toBe(false);
    });
  });
});

describe('Grid Creation', () => {
  test('should create grid of squares', () => {
    const mockCtx = {
      canvas: { width: 200, height: 200 }
    };
    
    const grid = createGrid(mockCtx, 10, 50, 10, 50, 5);
    
    expect(Array.isArray(grid)).toBe(true);
    expect(grid.length).toBeGreaterThan(0);
    expect(grid[0]).toBeInstanceOf(Square);
  });
});

describe('Game State Utilities', () => {
  describe('GameState.createClock', () => {
    test('should create a clock object', () => {
      const clock = GameState.createClock();
      
      expect(clock).toHaveProperty('total');
      expect(clock).toHaveProperty('start');
      expect(clock).toHaveProperty('time');
      expect(clock).toHaveProperty('delta');
      expect(clock.total).toBe(0);
    });
  });

  describe('GameState.updateClock', () => {
    test('should update clock with time', () => {
      const clock = GameState.createClock();
      const time = Date.now();
      
      const updatedClock = GameState.updateClock(clock, time);
      
      expect(updatedClock.start).toBe(time);
      expect(updatedClock.time).toBe(time);
    });

    test('should calculate delta time', () => {
      const clock = GameState.createClock();
      const time1 = 1000;
      const time2 = 1100;
      
      GameState.updateClock(clock, time1);
      const updatedClock = GameState.updateClock(clock, time2);
      
      expect(updatedClock.delta).toBeCloseTo(0.1, 2);
    });

    test('should apply game speed multiplier', () => {
      const clock = GameState.createClock();
      const time1 = 1000;
      const time2 = 1100;
      const gameSpeed = 2;
      
      GameState.updateClock(clock, time1);
      const updatedClock = GameState.updateClock(clock, time2, gameSpeed);
      
      expect(updatedClock.delta).toBeCloseTo(0.2, 2);
    });

    test('should cap delta time', () => {
      const clock = GameState.createClock();
      const time1 = 1000;
      const time2 = 1200; // 200ms difference
      
      GameState.updateClock(clock, time1);
      const updatedClock = GameState.updateClock(clock, time2);
      
      expect(updatedClock.delta).toBeLessThanOrEqual(0.1);
    });
  });
});

describe('Eye class', () => {
  test('should create an eye with angle', () => {
    const eye = new Eye(Math.PI / 4);
    
    expect(eye.angle).toBe(Math.PI / 4);
    expect(eye.max_range).toBe(1000);
    expect(eye.sensed_proximity).toBe(1000);
    expect(eye.sensed_type).toBe(0);
  });
});

describe('Image Loading Utilities', () => {
  describe('loadImageFrames', () => {
    test('should create array of images', () => {
      const frames = loadImageFrames('test/path', 3, 'frame_');
      
      expect(Array.isArray(frames)).toBe(true);
      expect(frames).toHaveLength(3);
      expect(frames[0]).toBeInstanceOf(Image);
      expect(frames[0].src).toBe('test/path/frame_0.png');
    });
  });

  describe('drawRotatedImage', () => {
    test('should be a function', () => {
      expect(typeof drawRotatedImage).toBe('function');
    });
  });
});

describe('Async Utilities', () => {
  describe('requestAnimationFrameAsync', () => {
    test('should return a promise', () => {
      const result = requestAnimationFrameAsync(() => 'test');
      expect(result).toBeInstanceOf(Promise);
    });

    test('should resolve with function result', async () => {
      const result = await requestAnimationFrameAsync(() => 'test');
      expect(result).toBe('test');
    });
  });
});

describe('Chart Utilities', () => {
  describe('createOrUpdateChart', () => {
    test('should create new chart when chartInstance is null', () => {
      const mockCtx = {};
      const data = [1, 2, 3];
      const labels = ['A', 'B', 'C'];
      const chartConfig = {
        type: 'line',
        datasets: [{ data: data }]
      };
      
      const result = createOrUpdateChart(null, mockCtx, data, labels, chartConfig);
      
      expect(result).toBeTruthy();
    });

    test('should update existing chart', () => {
      const mockChart = {
        data: {
          labels: [],
          datasets: [{ data: [] }]
        },
        update: jest.fn()
      };
      
      const data = [1, 2, 3];
      const labels = ['A', 'B', 'C'];
      const chartConfig = {
        datasets: [{ data: data }]
      };
      
      const result = createOrUpdateChart(mockChart, {}, data, labels, chartConfig);
      
      expect(result).toBe(mockChart);
      expect(mockChart.update).toHaveBeenCalled();
      expect(mockChart.data.labels).toEqual(labels);
    });
  });
});

describe('Model Persistence', () => {
  describe('ModelPersistence.save', () => {
    test('should have save method', () => {
      expect(typeof ModelPersistence.save).toBe('function');
    });
  });

  describe('ModelPersistence.load', () => {
    test('should have load method', () => {
      expect(typeof ModelPersistence.load).toBe('function');
    });
  });

  describe('ModelPersistence.saveToFiles', () => {
    test('should have saveToFiles method', () => {
      expect(typeof ModelPersistence.saveToFiles).toBe('function');
    });
  });
});
