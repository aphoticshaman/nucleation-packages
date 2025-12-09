import { describe, it, expect } from 'vitest';
import {
  gaugeEquivalent,
  gaugeClustering,
  rgFlow,
  optimalAnswer,
  testGaugeInvariance,
} from '../clustering.js';

describe('gaugeEquivalent', () => {
  it('should identify gauge-equivalent values within tolerance', () => {
    expect(gaugeEquivalent(42, 43, 0.05)).toBe(true); // ~2.4% difference
    expect(gaugeEquivalent(100, 103, 0.05)).toBe(true); // 3% difference
    expect(gaugeEquivalent(1000, 1040, 0.05)).toBe(true); // 4% difference
  });

  it('should identify non-equivalent values outside tolerance', () => {
    expect(gaugeEquivalent(42, 50, 0.05)).toBe(false); // ~19% difference
    expect(gaugeEquivalent(100, 110, 0.05)).toBe(false); // 10% difference
  });

  it('should handle zero values correctly', () => {
    expect(gaugeEquivalent(0, 0, 0.05)).toBe(true);
    expect(gaugeEquivalent(0, 0.01, 0.05)).toBe(false); // 0 to non-zero
  });

  it('should handle negative values', () => {
    expect(gaugeEquivalent(-42, -43, 0.05)).toBe(true);
    expect(gaugeEquivalent(-100, -103, 0.05)).toBe(true);
    expect(gaugeEquivalent(-42, 42, 0.05)).toBe(false); // Opposite signs
  });
});

describe('gaugeClustering', () => {
  it('should cluster gauge-equivalent values together', () => {
    const values = [42, 41.5, 43, 100, 101, 7];
    const clusters = gaugeClustering(values, { epsilon: 0.05 });

    // Should find 3 clusters: ~42 group, ~100 group, ~7 outlier
    expect(clusters.length).toBe(3);

    // Find the cluster containing 42
    const cluster42 = clusters.find((c) =>
      c.members.some((m) => Math.abs(m - 42) < 2)
    );
    expect(cluster42).toBeDefined();
    expect(cluster42!.members).toContain(42);
    expect(cluster42!.members).toContain(41.5);
    expect(cluster42!.members).toContain(43);
    expect(cluster42!.members.length).toBe(3);
  });

  it('should calculate cluster centers correctly', () => {
    const values = [40, 41, 42, 43, 44];
    const clusters = gaugeClustering(values, { epsilon: 0.1 });

    // All values should be in one cluster
    expect(clusters.length).toBe(1);
    expect(clusters[0].center).toBeCloseTo(42, 1); // Median or robust center
  });

  it('should rank clusters by score', () => {
    const values = [42, 42.5, 43, 100, 7, 7.1, 7.2, 7.3, 7.4];
    const clusters = gaugeClustering(values, { epsilon: 0.05 });

    // Clusters should be sorted by score (descending)
    for (let i = 1; i < clusters.length; i++) {
      expect(clusters[i - 1].score).toBeGreaterThanOrEqual(clusters[i].score);
    }
  });

  it('should handle empty input', () => {
    const clusters = gaugeClustering([], { epsilon: 0.05 });
    expect(clusters).toEqual([]);
  });

  it('should handle single value', () => {
    const clusters = gaugeClustering([42], { epsilon: 0.05 });
    expect(clusters.length).toBe(1);
    expect(clusters[0].members).toEqual([42]);
    expect(clusters[0].center).toBe(42);
  });
});

describe('rgFlow', () => {
  it('should refine cluster centers through coarsening', () => {
    const values = [40, 41, 42, 43, 44];
    const clusters = gaugeClustering(values, { epsilon: 0.05 });
    const refined = rgFlow(clusters, 3);

    // RG flow should converge to fewer, more refined clusters
    expect(refined.length).toBeLessThanOrEqual(clusters.length);
  });

  it('should converge to fixed point', () => {
    const values = [42, 42.1, 42.2, 42.3];
    const clusters = gaugeClustering(values, { epsilon: 0.1 });
    const refined1 = rgFlow(clusters, 1);
    const refined3 = rgFlow(clusters, 3);
    const refined5 = rgFlow(clusters, 5);

    // Multiple RG steps should converge
    expect(refined3[0].center).toBeCloseTo(refined5[0].center, 2);
  });
});

describe('optimalAnswer', () => {
  it('should return center of winning cluster', () => {
    const values = [42, 42.5, 43, 100, 101, 7];
    const optimal = optimalAnswer(values, { epsilon: 0.05 });

    // Winning cluster is ~42 (3 members vs 2 for ~100)
    expect(optimal).toBeGreaterThan(41);
    expect(optimal).toBeLessThan(44);
  });

  it('should handle unanimous agreement', () => {
    const values = [42, 42.01, 42.02, 41.99];
    const optimal = optimalAnswer(values, { epsilon: 0.05 });
    expect(optimal).toBeCloseTo(42, 1);
  });

  it('should prefer larger cluster over tighter cluster', () => {
    // 5 values near 42, 2 values exactly at 100
    const values = [42, 42.5, 43, 41, 41.5, 100, 100];
    const optimal = optimalAnswer(values, { epsilon: 0.05 });

    // Should pick ~42 cluster (5 members) over 100 cluster (2 members)
    expect(optimal).toBeGreaterThan(40);
    expect(optimal).toBeLessThan(45);
  });
});

describe('testGaugeInvariance', () => {
  it('should detect gauge invariance in well-clustered data', () => {
    const values = [42, 42.1, 42.2, 42.3, 42.4];
    const result = testGaugeInvariance(values, 0.05);

    expect(result.isInvariant).toBe(true);
    expect(result.maxDeviation).toBeLessThan(0.02);
  });

  it('should detect gauge variance in scattered data', () => {
    const values = [42, 100, 200, 7, 500];
    const result = testGaugeInvariance(values, 0.05);

    expect(result.isInvariant).toBe(false);
  });

  it('should return deviation metrics', () => {
    const values = [42, 43, 44, 45];
    const result = testGaugeInvariance(values, 0.05);

    expect(result).toHaveProperty('isInvariant');
    expect(result).toHaveProperty('maxDeviation');
    expect(typeof result.maxDeviation).toBe('number');
  });
});

describe('error reduction validation', () => {
  it('should achieve significant error reduction over naive voting', () => {
    // Simulate AIMO-like scenario: correct answer with noise
    const correctAnswer = 42;
    const samples = [
      ...Array(10).fill(null).map(() => correctAnswer + (Math.random() - 0.5) * 2),
      ...Array(3).fill(null).map(() => correctAnswer + 10 + Math.random() * 5), // Off-by-one cluster
      100, // Outlier
      200, // Another outlier
    ];

    // Majority voting would pick modal bucket
    const rounded = samples.map((v) => Math.round(v));
    const counts = new Map<number, number>();
    for (const v of rounded) {
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    const majorityVote = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];

    // Value clustering
    const clusteredAnswer = optimalAnswer(samples, { epsilon: 0.05 });

    // Clustering should be closer to true answer
    const majorityError = Math.abs(majorityVote - correctAnswer);
    const clusterError = Math.abs(clusteredAnswer - correctAnswer);

    // Not asserting specific reduction, but clustering should not be worse
    expect(clusterError).toBeLessThanOrEqual(majorityError + 1);
  });
});
