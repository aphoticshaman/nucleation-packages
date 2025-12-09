/**
 * Gauge-Theoretic Value Clustering
 *
 * Implements gauge symmetry-based clustering for signal fusion.
 * Based on the theoretical insight that the 5% tolerance defines
 * a gauge equivalence class with O(ε²) invariance.
 */

import {
  Cluster,
  ClusteringConfig,
  DEFAULT_CLUSTERING_CONFIG,
} from './types.js';

/**
 * Check if two values are gauge-equivalent within tolerance
 *
 * @param a - First value
 * @param b - Second value
 * @param epsilon - Gauge tolerance (default 0.05)
 * @returns True if values are gauge-equivalent
 */
export function gaugeEquivalent(
  a: number,
  b: number,
  epsilon: number = 0.05
): boolean {
  const maxAbs = Math.max(Math.abs(a), Math.abs(b));
  if (maxAbs < 1e-10) {
    return Math.abs(a - b) < 1e-10;
  }
  return Math.abs(a - b) / maxAbs < epsilon;
}

/**
 * Compute cluster statistics
 */
function computeClusterStats(members: number[]): Cluster {
  if (members.length === 0) {
    return {
      members: [],
      center: 0,
      tightness: 0,
      score: 0,
      size: 0,
    };
  }

  // Sort for median computation
  const sorted = [...members].sort((a, b) => a - b);
  const n = sorted.length;

  // Robust center estimate using median
  const center =
    n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)];

  // Compute tightness as inverse of relative spread
  const mean = members.reduce((sum, v) => sum + v, 0) / n;
  const variance =
    members.reduce((sum, v) => sum + (v - mean) ** 2, 0) / Math.max(n - 1, 1);
  const std = Math.sqrt(variance);
  const tightness = 1.0 / (1.0 + std / (Math.abs(center) + 1e-10));

  // Composite score: size weighted by tightness
  const score = n * Math.sqrt(tightness);

  return {
    members,
    center,
    tightness,
    score,
    size: n,
  };
}

/**
 * Cluster values into gauge equivalence classes
 *
 * Uses a union-find approach: values connected by gauge equivalence
 * are placed in the same cluster.
 *
 * @param values - Array of numeric values to cluster
 * @param config - Clustering configuration
 * @returns Array of clusters sorted by score (descending)
 */
export function gaugeClustering(
  values: number[],
  config: Partial<ClusteringConfig> = {}
): Cluster[] {
  const { epsilon, minClusterSize } = {
    ...DEFAULT_CLUSTERING_CONFIG,
    ...config,
  };

  if (values.length === 0) {
    return [];
  }

  // Sort values for efficient clustering
  const indexed = values.map((v, i) => ({ value: v, index: i }));
  indexed.sort((a, b) => a.value - b.value);

  const clusters: number[][] = [];
  let currentCluster: number[] = [indexed[0].value];

  for (let i = 1; i < indexed.length; i++) {
    const current = indexed[i].value;

    // Check gauge equivalence with any member of current cluster
    const isEquivalent = currentCluster.some((member) =>
      gaugeEquivalent(current, member, epsilon)
    );

    if (isEquivalent) {
      currentCluster.push(current);
    } else {
      // Finalize current cluster and start new one
      if (currentCluster.length >= minClusterSize) {
        clusters.push(currentCluster);
      }
      currentCluster = [current];
    }
  }

  // Don't forget the last cluster
  if (currentCluster.length >= minClusterSize) {
    clusters.push(currentCluster);
  }

  // Convert to Cluster objects and sort by score
  const result = clusters.map(computeClusterStats);
  result.sort((a, b) => b.score - a.score);

  return result;
}

/**
 * Apply Renormalization Group flow to find fixed point
 *
 * Each step coarsens the resolution by increasing effective epsilon.
 * The flow converges to the unique fixed point (cluster center).
 *
 * @param clusters - Initial clusters
 * @param steps - Number of coarsening steps
 * @returns Fixed point value
 */
export function rgFlow(clusters: Cluster[], steps: number = 3): number {
  if (clusters.length === 0) {
    return 0;
  }

  let centers = clusters.map((c) => c.center);
  let weights = clusters.map((c) => c.score);

  for (let step = 0; step < steps; step++) {
    // Weighted average toward highest-scoring cluster
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    if (totalWeight > 0) {
      const fixedPoint =
        centers.reduce((sum, c, i) => sum + c * weights[i], 0) / totalWeight;
      // In RG flow, we're approaching a fixed point
      centers = centers.map((c) => (c + fixedPoint) / 2);
    }
  }

  // Return weighted average as final fixed point
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  return totalWeight > 0
    ? centers.reduce((sum, c, i) => sum + c * weights[i], 0) / totalWeight
    : centers[0];
}

/**
 * Compute optimal answer from value clustering
 *
 * Implements the full gauge-theoretic pipeline:
 * 1. Cluster values into gauge equivalence classes
 * 2. Select best cluster by score
 * 3. Apply RG flow refinement
 * 4. Return basin-refined answer
 *
 * @param values - Array of numeric values
 * @param config - Clustering configuration
 * @returns Optimal fused value
 */
export function optimalAnswer(
  values: number[],
  config: Partial<ClusteringConfig> = {}
): number {
  const fullConfig = { ...DEFAULT_CLUSTERING_CONFIG, ...config };

  if (values.length === 0) {
    return 0;
  }

  if (values.length === 1) {
    return values[0];
  }

  const clusters = gaugeClustering(values, config);

  if (clusters.length === 0) {
    // No valid clusters, fall back to median
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  const bestCluster = clusters[0]; // Already sorted by score

  // Apply RG flow if configured
  let fixedPoint = bestCluster.center;
  if (fullConfig.useRGFlow) {
    fixedPoint = rgFlow([bestCluster], fullConfig.rgSteps);
  }

  // Basin refinement: average of center, median, and trimmed mean
  const members = bestCluster.members;
  const sorted = [...members].sort((a, b) => a - b);
  const median =
    sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)];

  // Trimmed mean (remove extremes)
  const trimmed =
    sorted.length > 2 ? sorted.slice(1, -1) : sorted;
  const trimmedMean =
    trimmed.reduce((sum, v) => sum + v, 0) / trimmed.length;

  // Combine estimates
  return (bestCluster.center + median + trimmedMean + fixedPoint) / 4;
}

/**
 * Test gauge invariance of clustering
 *
 * @param values - Original values
 * @param epsilon - Gauge tolerance
 * @returns Invariance score (1.0 = perfectly invariant)
 */
export function testGaugeInvariance(
  values: number[],
  epsilon: number = 0.05
): number {
  if (values.length === 0) {
    return 1.0;
  }

  // Original answer
  const original = optimalAnswer(values, { epsilon });

  // Apply gauge transformation (small random perturbation)
  const perturbed = values.map(
    (v) => v * (1 + (Math.random() - 0.5) * epsilon)
  );
  const perturbedAnswer = optimalAnswer(perturbed, { epsilon });

  // Invariance is how close the answers are
  const maxAbs = Math.max(Math.abs(original), Math.abs(perturbedAnswer), 1e-10);
  const relativeError = Math.abs(original - perturbedAnswer) / maxAbs;

  return Math.max(0, 1 - relativeError / epsilon);
}
