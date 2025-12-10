/**
 * @nucleation/gtvc
 *
 * Gauge-Theoretic Value Clustering for Signal Fusion
 *
 * This module implements the theoretical insights from Chapter 26:
 * - Gauge symmetry structure of value clustering
 * - CIC functional for confidence calibration
 * - Renormalization group flow for fixed-point finding
 * - Phase transition detection for convergence
 *
 * @packageDocumentation
 */

// Types
export type {
  Cluster,
  CICState,
  ClusteringConfig,
  CICConfig,
  FusionResult,
  PhaseIndicator,
} from './types.js';

export {
  DEFAULT_CLUSTERING_CONFIG,
  DEFAULT_CIC_CONFIG,
} from './types.js';

// Clustering
export {
  gaugeEquivalent,
  gaugeClustering,
  rgFlow,
  optimalAnswer,
  testGaugeInvariance,
} from './clustering.js';

// CIC
export {
  ncd,
  computePhi,
  computeEntropy,
  computeCoherence,
  computeCIC,
  computeCICFromClusters,
  detectCrystallization,
} from './cic.js';

// Fusion
export type { FusionConfig } from './fusion.js';
export {
  SignalFuser,
  fuseSignals,
  fuseMultiSource,
  StreamingAggregator,
} from './fusion.js';
