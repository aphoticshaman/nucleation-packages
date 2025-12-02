/**
 * Proprietary Formula Tests
 *
 * Tests for Phase Transition, Signal Fusion, Cascade Predictor, etc.
 * Note: Many algorithms are stochastic, so tests verify behavior rather than exact values.
 */

import { describe, it, expect } from 'vitest';
import { PhaseTransitionModel, SystemPhase } from '../formulas/phase-transition';
import { SignalFusionRecipe } from '../formulas/signal-fusion';
import { CascadePredictor } from '../formulas/cascade-predictor';
import { SentimentHarmonic } from '../formulas/sentiment-harmonic';
import { AnomalyFingerprintDetector } from '../formulas/anomaly-fingerprint';
import { QuantumInspiredOptimizer } from '../formulas/quantum-optimize';

describe('PhaseTransitionModel', () => {
  it('should analyze phase from signals', () => {
    const model = new PhaseTransitionModel();

    const signals = new Map<string, number[]>();
    signals.set(
      'market',
      Array(50)
        .fill(0)
        .map(() => Math.random() * 0.3 + 0.5)
    );
    signals.set(
      'sentiment',
      Array(50)
        .fill(0)
        .map(() => Math.random() * 0.3 + 0.5)
    );

    const state = model.analyzePhase(signals);

    expect(state.phase).toBeDefined();
    expect(Object.values(SystemPhase)).toContain(state.phase);
    expect(state.temperature).toBeGreaterThanOrEqual(0);
    expect(state.temperature).toBeLessThanOrEqual(1);
    expect(state.orderParameter).toBeGreaterThanOrEqual(0);
    expect(state.orderParameter).toBeLessThanOrEqual(1);
  });

  it('should produce consistent output format', () => {
    const model = new PhaseTransitionModel();

    const signals = new Map<string, number[]>();
    signals.set(
      'volatile',
      Array(50)
        .fill(0)
        .map(() => Math.random())
    );

    const state = model.analyzePhase(signals);

    // Verify all expected fields exist
    expect(typeof state.temperature).toBe('number');
    expect(typeof state.orderParameter).toBe('number');
    expect(typeof state.criticalExponent).toBe('number');
    expect(typeof state.nucleationSites).toBe('number');
    expect(typeof state.confidence).toBe('number');
  });

  it('should build history for forecasting', () => {
    const model = new PhaseTransitionModel();

    // Build up history
    for (let i = 0; i < 15; i++) {
      const signals = new Map<string, number[]>();
      signals.set(
        'test',
        Array(30)
          .fill(0)
          .map(() => 0.5 + Math.random() * 0.3)
      );
      model.analyzePhase(signals);
    }

    // Forecasting should work (may or may not predict based on state)
    const signals = new Map<string, number[]>();
    signals.set(
      'test',
      Array(30)
        .fill(0)
        .map(() => 0.8 + Math.random() * 0.1)
    );
    const forecast = model.forecastTransition(signals);

    // Should return either null or valid forecast
    if (forecast) {
      expect(forecast.probability).toBeGreaterThanOrEqual(0);
      expect(forecast.probability).toBeLessThanOrEqual(1);
    }
  });
});

describe('SignalFusionRecipe', () => {
  it('should fuse multiple signals', () => {
    const fusion = new SignalFusionRecipe();

    fusion.registerSignal({
      name: 'SEC',
      category: 'official',
      reliability: 0.95,
      latency: 0,
      noiseLevel: 0.1,
    });

    fusion.registerSignal({
      name: 'news',
      category: 'news',
      reliability: 0.7,
      latency: 2,
      noiseLevel: 0.3,
    });

    const signals = new Map<string, number[]>();
    signals.set('SEC', [0.6, 0.65, 0.7, 0.68, 0.72]);
    signals.set('news', [0.5, 0.55, 0.6, 0.58, 0.62]);

    const result = fusion.fuse(signals);

    // Should return a valid fused signal object
    expect(result).toBeDefined();
    expect(result).toHaveProperty('value');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('contributors');
    expect(result).toHaveProperty('regime');
    expect(['risk-on', 'risk-off', 'transitional', 'uncertain']).toContain(result.regime);
  });

  it('should handle single signal', () => {
    const fusion = new SignalFusionRecipe();

    fusion.registerSignal({
      name: 'test',
      category: 'official',
      reliability: 0.9,
      latency: 0,
      noiseLevel: 0.1,
    });

    const signals = new Map<string, number[]>();
    signals.set('test', [0.8]);

    const result = fusion.fuse(signals);

    expect(result).toBeDefined();
    expect(result).toHaveProperty('contributors');
  });
});

describe('CascadePredictor', () => {
  it('should analyze cascade state', () => {
    const predictor = new CascadePredictor();

    const signals = new Map<string, number[]>();
    signals.set(
      'social',
      Array(30)
        .fill(0)
        .map(() => Math.random() * 0.5)
    );
    signals.set(
      'market',
      Array(30)
        .fill(0)
        .map(() => Math.random() * 0.5)
    );

    const state = predictor.analyzeState(signals);

    expect(['dormant', 'seeding', 'spreading', 'peak', 'declining', 'exhausted']).toContain(
      state.phase
    );
    expect(state.currentIntensity).toBeGreaterThanOrEqual(0);
    expect(state.currentIntensity).toBeLessThanOrEqual(1);
  });

  it('should run SIR simulation', () => {
    const predictor = new CascadePredictor();

    const trajectory = predictor.simulateSir(0.01, 100);

    expect(trajectory).toHaveLength(101);
    expect(trajectory[0]).toBe(0.01);
    // SIR should have some dynamics
    expect(Math.max(...trajectory)).toBeGreaterThan(trajectory[0]);
  });

  it('should predict cascades', () => {
    const predictor = new CascadePredictor();

    const signals = new Map<string, number[]>();
    signals.set(
      'test',
      Array(30)
        .fill(0)
        .map((_, i) => 0.1 + i * 0.02)
    );

    const prediction = predictor.predict(signals);

    expect(prediction.probability).toBeGreaterThanOrEqual(0);
    expect(prediction.probability).toBeLessThanOrEqual(1);
    expect(prediction.confidence).toBeGreaterThanOrEqual(0);
  });
});

describe('SentimentHarmonic', () => {
  it('should analyze sentiment harmonics', () => {
    const analyzer = new SentimentHarmonic();

    const signals = new Map<string, number[]>();
    signals.set(
      'news',
      Array(50)
        .fill(0)
        .map((_, i) => 0.5 + 0.3 * Math.sin((2 * Math.PI * i) / 10) + Math.random() * 0.1)
    );

    const result = analyzer.analyze(signals);

    expect(result.composite).toBeGreaterThanOrEqual(-1);
    expect(result.composite).toBeLessThanOrEqual(1);
    expect(result.harmonics.baseline).toBeDefined();
    expect(['euphoric', 'optimistic', 'neutral', 'fearful', 'panic']).toContain(result.regime);
  });

  it('should handle contrarian detection', () => {
    const analyzer = new SentimentHarmonic();

    const sentiment = new Map<string, number[]>();
    sentiment.set(
      'social',
      Array(20)
        .fill(0)
        .map(() => 0.9 + Math.random() * 0.1)
    );

    const fundamentals = new Map<string, number[]>();
    fundamentals.set(
      'earnings',
      Array(20)
        .fill(0)
        .map(() => -0.5 - Math.random() * 0.3)
    );

    const signal = analyzer.detectContrarianSignal(sentiment, fundamentals);

    expect(signal.direction).toBeDefined();
    expect(['bullish', 'bearish', 'neutral']).toContain(signal.direction);
    expect(typeof signal.strength).toBe('number');
  });
});

describe('AnomalyFingerprintDetector', () => {
  it('should run detection without error', () => {
    const detector = new AnomalyFingerprintDetector();

    // Build some baseline
    for (let i = 0; i < 40; i++) {
      const signals = new Map<string, number[]>();
      signals.set(
        'normal',
        Array(30)
          .fill(0)
          .map(() => 50 + Math.random() * 5)
      );
      detector.detect(signals);
    }

    // Run detection
    const anomalySignals = new Map<string, number[]>();
    anomalySignals.set(
      'normal',
      Array(30)
        .fill(0)
        .map(() => 100 + Math.random() * 10)
    );

    const result = detector.detect(anomalySignals);

    expect(result).toBeDefined();
    expect(typeof result.isAnomaly).toBe('boolean');
    expect(typeof result.severity).toBe('number');
    expect(result.severity).toBeGreaterThanOrEqual(0);
  });

  it('should calculate phi (integrated information)', () => {
    const detector = new AnomalyFingerprintDetector();

    const signals = new Map<string, number[]>();
    signals.set('a', [1, 2, 3, 4, 5]);
    signals.set('b', [2, 4, 6, 8, 10]);
    signals.set('c', [5, 3, 4, 2, 1]);

    const phi = detector.calculatePhi(signals);

    expect(phi.phi).toBeGreaterThanOrEqual(0);
    expect(phi.partitions.length).toBeGreaterThan(0);
  });
});

describe('QuantumInspiredOptimizer', () => {
  it('should run optimization', () => {
    const optimizer = new QuantumInspiredOptimizer();

    const result = optimizer.solve(
      {
        dimensions: 2,
        objective: (x) => (x[0] - 3) ** 2 + (x[1] - 4) ** 2,
        bounds: [
          { min: 0, max: 10 },
          { min: 0, max: 10 },
        ],
      },
      { maxIterations: 1000 }
    );

    // Should find something better than worst case (100)
    expect(result.value).toBeLessThan(50);
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.solution).toHaveLength(2);
  });

  it('should optimize portfolio', () => {
    const optimizer = new QuantumInspiredOptimizer();

    const assets = ['A', 'B', 'C'];
    const returns = new Map<string, number[]>();
    returns.set('A', [0.1, 0.05, 0.08, 0.12, 0.03]);
    returns.set('B', [0.05, 0.08, 0.02, 0.06, 0.09]);
    returns.set('C', [0.03, 0.04, 0.05, 0.02, 0.06]);

    const result = optimizer.optimizePortfolio(assets, returns);

    expect(result.weights.size).toBe(3);

    // Weights should sum to approximately 1
    let sum = 0;
    for (const w of result.weights.values()) {
      expect(w).toBeGreaterThanOrEqual(0);
      sum += w;
    }
    expect(sum).toBeCloseTo(1, 1);
  });

  it('should select signals', () => {
    const optimizer = new QuantumInspiredOptimizer();

    const signals = ['A', 'B', 'C', 'D', 'E'];
    const evaluator = (selected: string[]) => {
      let score = 0;
      if (selected.includes('A')) score += 1;
      if (selected.includes('B')) score += 0.8;
      if (selected.includes('C')) score += 0.3;
      return score;
    };

    const result = optimizer.selectSignals(signals, evaluator, 2);

    expect(result.selected).toHaveLength(2);
    expect(result.score).toBeGreaterThan(0);
  });
});
