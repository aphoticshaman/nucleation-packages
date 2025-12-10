/**
 * CIC Functional Computation
 *
 * Implements the Compression-Integration-Causality functional:
 * F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
 *
 * Where:
 * - Φ (phi) = integrated information via compression cohesion
 * - H = representation entropy
 * - C_multi = multi-scale coherence
 */

import {
  CICState,
  CICConfig,
  DEFAULT_CIC_CONFIG,
  Cluster,
} from './types.js';
import { gaugeClustering } from './clustering.js';

/**
 * Compute Normalized Compression Distance between two strings
 *
 * NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
 *
 * We approximate compression using string length after JSON stringify
 * (a proxy for Kolmogorov complexity).
 *
 * @param a - First string
 * @param b - Second string
 * @returns NCD value in [0, 1]
 */
export function ncd(a: string, b: string): number {
  // Simple compression proxy: unique character count ratio
  // In production, use actual compression (e.g., pako/gzip)
  const setA = new Set(a);
  const setB = new Set(b);
  const setAB = new Set(a + b);

  const cA = setA.size + a.length * 0.1; // Penalize length
  const cB = setB.size + b.length * 0.1;
  const cAB = setAB.size + (a.length + b.length) * 0.1;

  const minC = Math.min(cA, cB);
  const maxC = Math.max(cA, cB);

  if (maxC === 0) return 0;

  return (cAB - minC) / maxC;
}

/**
 * Compute integrated information Φ via compression cohesion
 *
 * High Φ = samples share algorithmic structure
 * Φ = 1 - mean(pairwise NCD)
 *
 * @param samples - Array of string samples (reasoning traces, etc.)
 * @returns Φ value in [0, 1]
 */
export function computePhi(samples: string[]): number {
  const n = samples.length;
  if (n < 2) return 0;

  let totalNCD = 0;
  let count = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      totalNCD += ncd(samples[i], samples[j]);
      count++;
    }
  }

  const meanNCD = totalNCD / count;
  return Math.max(0, 1 - meanNCD);
}

/**
 * Compute representation entropy H
 *
 * Low H = answers are concentrated
 * Uses histogram-based entropy estimation
 *
 * @param values - Array of numeric values
 * @param bins - Number of histogram bins
 * @returns Entropy value (normalized to [0, 1])
 */
export function computeEntropy(values: number[], bins: number = 20): number {
  if (values.length === 0) return 0;
  if (values.length === 1) return 0;

  // Find range
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;

  if (range === 0) return 0; // All values identical

  // Build histogram
  const histogram = new Array(bins).fill(0);
  const binWidth = range / bins;

  for (const v of values) {
    const binIndex = Math.min(
      Math.floor((v - min) / binWidth),
      bins - 1
    );
    histogram[binIndex]++;
  }

  // Compute entropy
  const n = values.length;
  let entropy = 0;

  for (const count of histogram) {
    if (count > 0) {
      const p = count / n;
      entropy -= p * Math.log2(p);
    }
  }

  // Normalize by max entropy (uniform distribution)
  const maxEntropy = Math.log2(bins);
  return entropy / maxEntropy;
}

/**
 * Compute multi-scale coherence C_multi
 *
 * C_multi = w1·C1 + w2·C2 + w3·C3
 *
 * Where:
 * - C1 = exact repetition rate
 * - C2 = proximity rate (within 5%)
 * - C3 = inverse coefficient of variation
 *
 * @param values - Array of numeric values
 * @param weights - Weights for [C1, C2, C3] (default [0.5, 0.3, 0.2])
 * @returns C_multi value in [0, 1]
 */
export function computeCoherence(
  values: number[],
  weights: [number, number, number] = [0.5, 0.3, 0.2]
): number {
  const n = values.length;
  if (n < 2) return 0;

  // C1: Exact repetition rate
  const counts = new Map<number, number>();
  for (const v of values) {
    counts.set(v, (counts.get(v) || 0) + 1);
  }
  let exactPairs = 0;
  for (const count of counts.values()) {
    exactPairs += count * (count - 1);
  }
  const c1 = exactPairs / (n * (n - 1));

  // C2: Proximity rate (within 5%)
  let proximityPairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const maxAbs = Math.max(Math.abs(values[i]), Math.abs(values[j]));
      if (maxAbs > 0) {
        const relDiff = Math.abs(values[i] - values[j]) / maxAbs;
        if (relDiff < 0.05) {
          proximityPairs++;
        }
      }
    }
  }
  const c2 = (2 * proximityPairs) / (n * (n - 1));

  // C3: Inverse coefficient of variation
  const mean = values.reduce((sum, v) => sum + v, 0) / n;
  const variance =
    values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);
  const c3 = Math.abs(mean) > 1e-10 ? 1 / (1 + std / Math.abs(mean)) : 0;

  return weights[0] * c1 + weights[1] * c2 + weights[2] * c3;
}

/**
 * Compute the full CIC functional
 *
 * F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
 *
 * @param samples - String samples (for Φ computation)
 * @param values - Numeric values (for H and C computation)
 * @param config - CIC configuration
 * @returns Full CIC state including functional value and confidence
 */
export function computeCIC(
  samples: string[],
  values: number[],
  config: Partial<CICConfig> = {}
): CICState {
  const { lambda, gamma, confidenceBounds } = {
    ...DEFAULT_CIC_CONFIG,
    ...config,
  };

  // Compute components
  const phi = computePhi(samples);
  const entropy = computeEntropy(values);
  const coherence = computeCoherence(values);

  // Compute functional
  const functional = phi - lambda * entropy + gamma * coherence;

  // Derive confidence (sigmoid-like mapping to [min, max])
  const rawConfidence = (functional + 1) / 2; // Map [-1, 1] to [0, 1]
  const confidence = Math.min(
    confidenceBounds[1],
    Math.max(confidenceBounds[0], rawConfidence)
  );

  return {
    phi,
    entropy,
    coherence,
    functional,
    confidence,
  };
}

/**
 * Detect phase transition in CIC time series
 *
 * Crystallization occurs when dΦ/dt ≈ λ·dH/dt
 * (the system has converged to a stable state)
 *
 * @param cicHistory - Array of CIC states over time
 * @param lambda - Lambda parameter
 * @returns Whether crystallization is detected
 */
export function detectCrystallization(
  cicHistory: CICState[],
  lambda: number = 0.5
): boolean {
  if (cicHistory.length < 3) return false;

  const n = cicHistory.length;

  // Compute derivatives (finite differences)
  const dPhi = cicHistory[n - 1].phi - cicHistory[n - 2].phi;
  const dH = cicHistory[n - 1].entropy - cicHistory[n - 2].entropy;

  // Check crystallization condition
  const tolerance = 0.05;
  return Math.abs(dPhi - lambda * dH) < tolerance;
}

/**
 * Compute CIC functional from clusters
 *
 * Alternative computation using pre-clustered data
 *
 * @param clusters - Array of clusters
 * @param totalValues - Total number of values
 * @param config - CIC configuration
 * @returns CIC state
 */
export function computeCICFromClusters(
  clusters: Cluster[],
  totalValues: number,
  config: Partial<CICConfig> = {}
): CICState {
  const { lambda, gamma, confidenceBounds } = {
    ...DEFAULT_CIC_CONFIG,
    ...config,
  };

  if (clusters.length === 0 || totalValues === 0) {
    return {
      phi: 0,
      entropy: 1,
      coherence: 0,
      functional: -lambda,
      confidence: confidenceBounds[0],
    };
  }

  // Φ: Proportion of values in top cluster (cohesion)
  const topCluster = clusters[0];
  const phi = topCluster.size / totalValues;

  // H: Entropy across clusters
  let entropy = 0;
  for (const cluster of clusters) {
    const p = cluster.size / totalValues;
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }
  // Normalize
  const maxEntropy = Math.log2(Math.max(clusters.length, 2));
  entropy = entropy / maxEntropy;

  // C: Average tightness weighted by size
  let coherence = 0;
  for (const cluster of clusters) {
    coherence += (cluster.size / totalValues) * cluster.tightness;
  }

  const functional = phi - lambda * entropy + gamma * coherence;

  const rawConfidence = (functional + 1) / 2;
  const confidence = Math.min(
    confidenceBounds[1],
    Math.max(confidenceBounds[0], rawConfidence)
  );

  return {
    phi,
    entropy,
    coherence,
    functional,
    confidence,
  };
}
