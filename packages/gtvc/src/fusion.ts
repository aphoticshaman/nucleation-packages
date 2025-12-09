/**
 * Signal Fusion Module
 *
 * Combines multiple signal sources using Gauge-Theoretic Value Clustering
 * and CIC functional optimization.
 */

import {
  FusionResult,
  ClusteringConfig,
  CICConfig,
  PhaseIndicator,
  DEFAULT_CLUSTERING_CONFIG,
  DEFAULT_CIC_CONFIG,
} from './types.js';
import {
  gaugeClustering,
  optimalAnswer,
  testGaugeInvariance,
} from './clustering.js';
import { computeCICFromClusters, detectCrystallization } from './cic.js';

/**
 * Signal fusion configuration
 */
export interface FusionConfig extends ClusteringConfig, CICConfig {
  /** Whether to track convergence history */
  trackHistory?: boolean;
  /** Maximum history length */
  maxHistory?: number;
}

const DEFAULT_FUSION_CONFIG: FusionConfig = {
  ...DEFAULT_CLUSTERING_CONFIG,
  ...DEFAULT_CIC_CONFIG,
  trackHistory: true,
  maxHistory: 100,
};

/**
 * Signal Fuser class
 *
 * Maintains state for streaming signal fusion with convergence detection.
 */
export class SignalFuser {
  private config: Required<FusionConfig>;
  private history: FusionResult[] = [];

  constructor(config: Partial<FusionConfig> = {}) {
    this.config = {
      ...DEFAULT_FUSION_CONFIG,
      ...config,
    } as Required<FusionConfig>;
  }

  /**
   * Fuse multiple signal values into optimal output
   *
   * @param values - Array of numeric signal values
   * @param traces - Optional reasoning traces for Φ computation
   * @returns Fusion result with confidence and diagnostics
   */
  fuse(values: number[], traces?: string[]): FusionResult {
    if (values.length === 0) {
      throw new Error('Cannot fuse empty signal array');
    }

    // Perform gauge clustering
    const clusters = gaugeClustering(values, this.config);

    // Compute CIC state
    const cicState = computeCICFromClusters(clusters, values.length, this.config);

    // Get optimal answer
    const fusedValue = optimalAnswer(values, this.config);

    // Test gauge invariance
    const gaugeInvariance = testGaugeInvariance(values, this.config.epsilon);

    // Build result
    const result: FusionResult = {
      value: fusedValue,
      confidence: cicState.confidence,
      cicState,
      winningCluster: clusters[0] || {
        members: values,
        center: fusedValue,
        tightness: 1,
        score: values.length,
        size: values.length,
      },
      allClusters: clusters,
      gaugeInvariance,
    };

    // Track history if configured
    if (this.config.trackHistory) {
      this.history.push(result);
      if (this.history.length > this.config.maxHistory) {
        this.history.shift();
      }
    }

    return result;
  }

  /**
   * Check if the signal has crystallized (converged)
   *
   * @returns True if crystallization detected
   */
  hasCrystallized(): boolean {
    if (this.history.length < 3) return false;
    const cicHistory = this.history.map((r) => r.cicState);
    return detectCrystallization(cicHistory, this.config.lambda);
  }

  /**
   * Get current phase indicator
   *
   * @returns Phase indicator with confidence
   */
  getPhaseIndicator(): PhaseIndicator {
    if (this.history.length < 3) {
      return {
        phase: 'stable',
        confidence: 0.5,
        varianceTrend: 'stable',
      };
    }

    const recent = this.history.slice(-10);
    const variances = recent.map(
      (r) => r.winningCluster.members.reduce(
        (sum, v) => sum + (v - r.value) ** 2,
        0
      ) / r.winningCluster.size
    );

    // Compute variance trend
    const n = variances.length;
    const halfN = Math.floor(n / 2);
    const firstHalf = variances.slice(0, halfN);
    const secondHalf = variances.slice(halfN);

    const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    let varianceTrend: 'increasing' | 'stable' | 'decreasing';
    if (avgSecond > avgFirst * 1.1) {
      varianceTrend = 'increasing';
    } else if (avgSecond < avgFirst * 0.9) {
      varianceTrend = 'decreasing';
    } else {
      varianceTrend = 'stable';
    }

    // Determine phase
    const crystallized = this.hasCrystallized();
    const latestConfidence = recent[recent.length - 1].confidence;

    let phase: PhaseIndicator['phase'];
    if (crystallized && latestConfidence > 0.8) {
      phase = 'stable';
    } else if (varianceTrend === 'decreasing' && !crystallized) {
      phase = 'pre_transition';
    } else if (varianceTrend === 'increasing') {
      phase = 'transitioning';
    } else {
      phase = 'stable';
    }

    return {
      phase,
      confidence: latestConfidence,
      varianceTrend,
    };
  }

  /**
   * Reset the fuser state
   */
  reset(): void {
    this.history = [];
  }

  /**
   * Get fusion history
   */
  getHistory(): FusionResult[] {
    return [...this.history];
  }
}

/**
 * One-shot signal fusion
 *
 * Convenience function for single fusion without state tracking.
 *
 * @param values - Array of numeric values
 * @param config - Fusion configuration
 * @returns Fused value and confidence
 */
export function fuseSignals(
  values: number[],
  config: Partial<FusionConfig> = {}
): { value: number; confidence: number } {
  const fuser = new SignalFuser(config);
  const result = fuser.fuse(values);
  return {
    value: result.value,
    confidence: result.confidence,
  };
}

/**
 * Multi-source signal fusion
 *
 * Fuses signals from multiple sources with source-specific weights.
 *
 * @param sources - Map of source name to values
 * @param weights - Map of source name to weight (default 1.0)
 * @param config - Fusion configuration
 * @returns Fused result
 */
export function fuseMultiSource(
  sources: Map<string, number[]>,
  weights?: Map<string, number>,
  config: Partial<FusionConfig> = {}
): FusionResult {
  // Flatten all values with weights
  const allValues: number[] = [];
  const defaultWeight = 1.0;

  for (const [source, values] of sources) {
    const weight = weights?.get(source) ?? defaultWeight;
    // Add values multiple times based on weight (integer approximation)
    const count = Math.max(1, Math.round(weight * values.length));
    for (let i = 0; i < count; i++) {
      allValues.push(values[i % values.length]);
    }
  }

  const fuser = new SignalFuser(config);
  return fuser.fuse(allValues);
}

/**
 * Streaming signal aggregator
 *
 * For real-time signal fusion with sliding window.
 */
export class StreamingAggregator {
  private window: number[] = [];
  private windowSize: number;
  private fuser: SignalFuser;

  constructor(windowSize: number = 100, config: Partial<FusionConfig> = {}) {
    this.windowSize = windowSize;
    this.fuser = new SignalFuser(config);
  }

  /**
   * Add a new value to the stream
   *
   * @param value - New signal value
   * @returns Current fused state (or null if window not full)
   */
  add(value: number): FusionResult | null {
    this.window.push(value);

    if (this.window.length > this.windowSize) {
      this.window.shift();
    }

    if (this.window.length < Math.min(10, this.windowSize)) {
      return null; // Not enough data yet
    }

    return this.fuser.fuse(this.window);
  }

  /**
   * Get current aggregated value
   */
  getCurrentValue(): number | null {
    if (this.window.length < 2) return null;
    return optimalAnswer(this.window);
  }

  /**
   * Get current phase
   */
  getPhase(): PhaseIndicator {
    return this.fuser.getPhaseIndicator();
  }

  /**
   * Reset the aggregator
   */
  reset(): void {
    this.window = [];
    this.fuser.reset();
  }
}
