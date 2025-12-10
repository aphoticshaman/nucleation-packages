/**
 * Gauge-Theoretic Value Clustering Types
 *
 * Core types for the GTVC signal fusion module.
 */

/**
 * A cluster of gauge-equivalent values
 */
export interface Cluster {
  /** Member values in the cluster */
  members: number[];
  /** Cluster center (robust estimate) */
  center: number;
  /** Tightness metric (inverse of spread) */
  tightness: number;
  /** Composite score for ranking */
  score: number;
  /** Number of members */
  size: number;
}

/**
 * CIC Functional state
 */
export interface CICState {
  /** Integrated information (compression cohesion) */
  phi: number;
  /** Representation entropy */
  entropy: number;
  /** Multi-scale coherence */
  coherence: number;
  /** Combined CIC functional value */
  functional: number;
  /** Confidence derived from CIC */
  confidence: number;
}

/**
 * Configuration for value clustering
 */
export interface ClusteringConfig {
  /** Gauge tolerance (default 0.05 = 5%) */
  epsilon?: number;
  /** Minimum cluster size to consider */
  minClusterSize?: number;
  /** Whether to apply RG flow refinement */
  useRGFlow?: boolean;
  /** Number of RG coarsening steps */
  rgSteps?: number;
}

/**
 * Configuration for CIC computation
 */
export interface CICConfig {
  /** Lambda parameter for entropy weight */
  lambda?: number;
  /** Gamma parameter for coherence weight */
  gamma?: number;
  /** Confidence bounds [min, max] */
  confidenceBounds?: [number, number];
}

/**
 * Signal fusion result
 */
export interface FusionResult {
  /** Fused output value */
  value: number;
  /** Confidence in the fusion */
  confidence: number;
  /** CIC state at fusion */
  cicState: CICState;
  /** Winning cluster details */
  winningCluster: Cluster;
  /** All clusters found */
  allClusters: Cluster[];
  /** Gauge invariance score */
  gaugeInvariance: number;
}

/**
 * Phase transition indicator for signal streams
 */
export interface PhaseIndicator {
  /** Current phase state */
  phase: 'stable' | 'pre_transition' | 'transitioning' | 'post_transition';
  /** Phase confidence */
  confidence: number;
  /** Time to predicted transition (if applicable) */
  timeToTransition?: number;
  /** Variance trend (increasing/decreasing) */
  varianceTrend: 'increasing' | 'stable' | 'decreasing';
}

/**
 * Default configurations
 */
export const DEFAULT_CLUSTERING_CONFIG: Required<ClusteringConfig> = {
  epsilon: 0.05,
  minClusterSize: 2,
  useRGFlow: true,
  rgSteps: 3,
};

export const DEFAULT_CIC_CONFIG: Required<CICConfig> = {
  lambda: 0.5,
  gamma: 0.3,
  confidenceBounds: [0.05, 0.95],
};
