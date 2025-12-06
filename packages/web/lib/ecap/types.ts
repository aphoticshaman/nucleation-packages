/**
 * ECAP - Entangled Co-Adaptive Protocol
 * TypeScript type definitions and interfaces
 */

/**
 * Representation of human cognitive state.
 * In practice, inferred from user interaction proxies.
 */
export interface CognitiveState {
  /** Extracted feature vector from interaction patterns */
  featureVector: number[];
  /** Attention distribution across UI elements */
  attentionPattern: number[];
  /** Cognitive load estimate 0-1 (0=relaxed, 1=overwhelmed) */
  cognitiveLoad: number;
  /** Information entropy of current thought stream */
  entropy: number;
  /** Unix timestamp */
  timestamp: number;
}

/**
 * Representation of AI model state
 */
export interface AIState {
  /** Hash of current parameters for change detection */
  paramsHash: string;
  /** Summary statistics of embedding space */
  embeddingMean: number;
  embeddingStd: number;
  /** Current adaptation rate multiplier */
  adaptationRate: number;
  /** Adaptation generation counter */
  generation: number;
}

/**
 * Stored interaction pattern from immune memory
 */
export interface InteractionPattern {
  patternId: string;
  humanFeatures: number[];
  aiEmbeddingSummary: number[];
  reward: number;
  timestamp: number;
  accessCount: number;
}

/**
 * Result of pattern recognition check
 */
export interface PatternRecognitionResult {
  isSelf: boolean;
  confidence: number;
  matchedPatternId?: string;
}

/**
 * Adaptation parameters computed by ECAP
 */
export interface AdaptationParameters {
  /** Scalar learning rate [0, 1] */
  learningRate: number;
  /** Per-dimension gradient scaling */
  gradientScale: number[];
  /** Which dimensions to focus on */
  attentionMask: number[];
  /** Immune regulation factor */
  regulation: number;
}

/**
 * Full interaction processing result
 */
export interface InteractionResult extends AdaptationParameters {
  /** Current human-AI correlation */
  correlation: number;
  /** Homeostasis adaptation rate */
  adaptationRate: number;
  /** Computed or external reward */
  reward: number;
  /** Whether pattern matches immune memory */
  isSelfPattern: boolean;
  /** Immune memory confidence */
  immuneConfidence: number;
}

/**
 * ECAP engine metrics
 */
export interface ECAPMetrics {
  avgCorrelation: number;
  avgLoad: number;
  avgReward: number;
  adaptationCount: number;
  immuneHits: number;
  selfPatternRatio: number;
  generation: number;
  cumulativeReward: number;
  interactionCount: number;
}

/**
 * Homeostasis controller state
 */
export interface HomeostasisState {
  currentRate: number;
  targetCorrelation: number;
  targetLoad: number;
  integralError: number;
}

/**
 * Entanglement monitor statistics
 */
export interface EntanglementStats {
  avgCorrelation: number;
  stdCorrelation: number;
  trend: number;
  historyLength: number;
}

/**
 * Immune memory statistics
 */
export interface ImmuneStats {
  totalInteractions: number;
  selfClassifications: number;
  nonSelfClassifications: number;
  patternRetrievals: number;
  selfPatternCount: number;
  nonSelfPatternCount: number;
  memorySize: number;
}

/**
 * Full system state snapshot
 */
export interface ECAPSystemState {
  metrics: ECAPMetrics;
  homeostasis: HomeostasisState;
  entanglement: EntanglementStats;
  immune: ImmuneStats;
}

/**
 * Configuration for ECAP engine
 */
export interface ECAPConfig {
  humanDim: number;
  aiDim: number;
  patternDim: number;
  targetCorrelation: number;
  targetLoad: number;
  memoryCapacity: number;
  selfThreshold: number;
  decayRate: number;
}

/**
 * Default configuration
 */
export const DEFAULT_ECAP_CONFIG: ECAPConfig = {
  humanDim: 128,
  aiDim: 256,
  patternDim: 64,
  targetCorrelation: 0.85,
  targetLoad: 0.6,
  memoryCapacity: 1000,
  selfThreshold: 0.7,
  decayRate: 0.995,
};

/**
 * Cognitive state feature labels for visualization
 */
export const COGNITIVE_FEATURES = [
  'attention_focus',
  'response_latency',
  'error_rate',
  'backtrack_frequency',
  'exploration_ratio',
  'action_diversity',
  'session_duration',
  'interaction_depth',
] as const;

/**
 * Edge types in ECAP causal graph
 */
export type ECAPEdgeType = 'ADAPTATION' | 'CORRELATION' | 'IMMUNE_MATCH' | 'HOMEOSTASIS';

/**
 * Node types in ECAP visualization
 */
export type ECAPNodeType = 'HUMAN' | 'AI' | 'PATTERN' | 'CONTROLLER';

/**
 * Visualization node for ECAP system
 */
export interface ECAPNode {
  id: string;
  type: ECAPNodeType;
  label: string;
  value: number;
  metadata?: Record<string, unknown>;
}

/**
 * Visualization edge for ECAP system
 */
export interface ECAPEdge {
  source: string;
  target: string;
  type: ECAPEdgeType;
  weight: number;
  label?: string;
}
