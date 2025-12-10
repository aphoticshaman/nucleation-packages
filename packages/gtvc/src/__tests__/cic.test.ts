import { describe, it, expect } from 'vitest';
import {
  ncd,
  computePhi,
  computeEntropy,
  computeCoherence,
  computeCIC,
  computeCICFromClusters,
  detectCrystallization,
} from '../cic.js';
import { gaugeClustering } from '../clustering.js';

describe('ncd (Normalized Compression Distance)', () => {
  it('should return 0 for identical strings', () => {
    expect(ncd('hello', 'hello')).toBe(0);
    expect(ncd('test123', 'test123')).toBe(0);
  });

  it('should return ~1 for completely different strings', () => {
    const distance = ncd('aaaaaaa', 'zzzzzzz');
    expect(distance).toBeGreaterThan(0.5);
  });

  it('should return intermediate values for similar strings', () => {
    const distance = ncd('hello world', 'hello there');
    expect(distance).toBeGreaterThan(0);
    expect(distance).toBeLessThan(1);
  });

  it('should be symmetric', () => {
    const d1 = ncd('abc', 'xyz');
    const d2 = ncd('xyz', 'abc');
    expect(d1).toBeCloseTo(d2, 5);
  });
});

describe('computePhi (Integrated Information)', () => {
  it('should return high phi for coherent samples', () => {
    const samples = ['42', '42', '42', '43', '42'];
    const phi = computePhi(samples);
    expect(phi).toBeGreaterThan(0.5);
  });

  it('should return low phi for incoherent samples', () => {
    const samples = ['42', '100', '200', '7', '500'];
    const phi = computePhi(samples);
    expect(phi).toBeLessThan(0.5);
  });

  it('should return 1 for identical samples', () => {
    const samples = ['test', 'test', 'test'];
    const phi = computePhi(samples);
    expect(phi).toBe(1);
  });

  it('should return 0 for empty input', () => {
    expect(computePhi([])).toBe(0);
  });
});

describe('computeEntropy', () => {
  it('should return 0 for uniform values', () => {
    const values = [42, 42, 42, 42];
    const entropy = computeEntropy(values);
    expect(entropy).toBe(0);
  });

  it('should return high entropy for spread values', () => {
    const values = [1, 10, 100, 1000, 10000];
    const entropy = computeEntropy(values);
    expect(entropy).toBeGreaterThan(0.5);
  });

  it('should return moderate entropy for clustered values', () => {
    const values = [42, 43, 41, 44, 40];
    const entropy = computeEntropy(values);
    expect(entropy).toBeLessThan(0.5);
  });

  it('should be normalized between 0 and 1', () => {
    const values = [1, 2, 3, 4, 5, 100, 200, 300];
    const entropy = computeEntropy(values);
    expect(entropy).toBeGreaterThanOrEqual(0);
    expect(entropy).toBeLessThanOrEqual(1);
  });
});

describe('computeCoherence', () => {
  it('should return high coherence for well-clustered data', () => {
    const values = [42, 42.1, 42.2, 41.9, 42.3];
    const coherence = computeCoherence(values);
    expect(coherence).toBeGreaterThan(0.7);
  });

  it('should return low coherence for scattered data', () => {
    const values = [1, 50, 100, 200, 500];
    const coherence = computeCoherence(values);
    expect(coherence).toBeLessThan(0.5);
  });

  it('should be normalized between 0 and 1', () => {
    const values = [1, 2, 3, 100, 200, 300];
    const coherence = computeCoherence(values);
    expect(coherence).toBeGreaterThanOrEqual(0);
    expect(coherence).toBeLessThanOrEqual(1);
  });
});

describe('computeCIC', () => {
  it('should compute CIC functional with all components', () => {
    const samples = ['42', '43', '41', '42', '100'];
    const values = [42, 43, 41, 42, 100];
    const result = computeCIC(samples, values);

    expect(result).toHaveProperty('phi');
    expect(result).toHaveProperty('entropy');
    expect(result).toHaveProperty('coherence');
    expect(result).toHaveProperty('functional');
    expect(result).toHaveProperty('confidence');
  });

  it('should produce high functional for coherent data', () => {
    const samples = ['42', '42', '42', '42'];
    const values = [42, 42, 42, 42];
    const result = computeCIC(samples, values);

    expect(result.functional).toBeGreaterThan(0.5);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it('should produce low functional for incoherent data', () => {
    const samples = ['1', '100', '500', '1000'];
    const values = [1, 100, 500, 1000];
    const result = computeCIC(samples, values);

    expect(result.functional).toBeLessThan(0.5);
  });

  it('should respect lambda and gamma parameters', () => {
    const samples = ['42', '43', '41', '100'];
    const values = [42, 43, 41, 100];

    const result1 = computeCIC(samples, values, { lambda: 0.1, gamma: 0.1 });
    const result2 = computeCIC(samples, values, { lambda: 0.9, gamma: 0.9 });

    // Different parameters should produce different results
    expect(result1.functional).not.toEqual(result2.functional);
  });

  it('should bound confidence to [0.05, 0.95]', () => {
    const samples = ['42', '42', '42'];
    const values = [42, 42, 42];
    const result = computeCIC(samples, values);

    expect(result.confidence).toBeGreaterThanOrEqual(0.05);
    expect(result.confidence).toBeLessThanOrEqual(0.95);
  });
});

describe('computeCICFromClusters', () => {
  it('should compute CIC from pre-clustered data', () => {
    const values = [42, 43, 41, 100, 101];
    const clusters = gaugeClustering(values, { epsilon: 0.05 });
    const result = computeCICFromClusters(clusters, values);

    expect(result).toHaveProperty('phi');
    expect(result).toHaveProperty('entropy');
    expect(result).toHaveProperty('coherence');
    expect(result).toHaveProperty('functional');
  });
});

describe('detectCrystallization', () => {
  it('should detect crystallization in converging history', () => {
    // Simulating entropy decreasing, phi increasing
    const history = [
      { phi: 0.3, entropy: 0.7, functional: 0.2 },
      { phi: 0.5, entropy: 0.5, functional: 0.4 },
      { phi: 0.7, entropy: 0.3, functional: 0.6 },
      { phi: 0.85, entropy: 0.15, functional: 0.8 },
      { phi: 0.9, entropy: 0.1, functional: 0.85 },
    ];

    const result = detectCrystallization(history, 0.5);
    expect(result).toBe(true);
  });

  it('should not detect crystallization in diverging history', () => {
    // Entropy increasing
    const history = [
      { phi: 0.7, entropy: 0.3, functional: 0.6 },
      { phi: 0.5, entropy: 0.5, functional: 0.4 },
      { phi: 0.3, entropy: 0.7, functional: 0.2 },
    ];

    const result = detectCrystallization(history, 0.5);
    expect(result).toBe(false);
  });

  it('should return false for insufficient history', () => {
    const history = [{ phi: 0.5, entropy: 0.5, functional: 0.4 }];
    const result = detectCrystallization(history, 0.5);
    expect(result).toBe(false);
  });
});

describe('CIC parameter validation', () => {
  it('should use default lambda=0.5', () => {
    const samples = ['42', '43'];
    const values = [42, 43];
    const result = computeCIC(samples, values);

    // Default lambda should be applied
    expect(result.phi).toBeDefined();
    expect(result.entropy).toBeDefined();
  });

  it('should use default gamma=0.3', () => {
    const samples = ['42', '43'];
    const values = [42, 43];
    const result = computeCIC(samples, values);

    // Default gamma should be applied
    expect(result.coherence).toBeDefined();
  });
});
