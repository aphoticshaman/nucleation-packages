/**
 * Causal Graph Types
 *
 * Types for force-directed causal topology visualization with
 * information-theoretic edge weighting (Transfer Entropy).
 */

// Semantic hierarchy levels
export type NodeLevel = 'MICRO' | 'MESO' | 'MACRO';

// Node classifications
export type NodeType = 'ACTOR' | 'EVENT' | 'RESOURCE' | 'HYPOTHESIS';

// Edge relationship types
export type EdgeType = 'INFLUENCE' | 'CAUSALITY' | 'CORRELATION';

/**
 * Context Layers - Tiered access model for intelligence depth
 *
 * SURFACE: Public geopolitical data (all users)
 * DEEP: Financial/crypto/economic (enterprise+)
 * DARK: Cyber/intel/classified (enterprise+ with vetting)
 */
export enum ContextLayer {
  SURFACE = 'SURFACE',
  DEEP = 'DEEP',
  DARK = 'DARK',
}

export const ContextLayerLabels: Record<ContextLayer, string> = {
  [ContextLayer.SURFACE]: 'Geopolitical',
  [ContextLayer.DEEP]: 'Economic/Crypto',
  [ContextLayer.DARK]: 'Cyber/Intel',
};

export const ContextLayerColors: Record<ContextLayer, string> = {
  [ContextLayer.SURFACE]: '#3b82f6', // Blue
  [ContextLayer.DEEP]: '#f59e0b', // Amber
  [ContextLayer.DARK]: '#ef4444', // Red
};

/**
 * Causal Node - Entity in the causal graph
 */
export interface CausalNode {
  id: string;
  label: string;
  type: NodeType;
  level: NodeLevel;

  // Dempster-Shafer belief mass (0-1)
  // Represents confidence/certainty in this node's state
  beliefMass: number;

  // Activity/pulse intensity (0-1)
  // Higher = more recent/active changes
  activity: number;

  // Physics state (set by simulation)
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;

  // Optional metadata
  entityId?: string; // Link to nations/threat_actors/etc
  entityType?: string;
}

/**
 * Causal Edge - Directed relationship between nodes
 */
export interface CausalEdge {
  source: string; // Node ID
  target: string; // Node ID

  // Transfer Entropy (0-1)
  // Information flow magnitude from source to target
  // TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
  transferEntropy: number;

  // Time lag in days
  // How long effects take to propagate
  lag: number;

  // Relationship type
  type: EdgeType;
}

/**
 * Regime state probabilities for Markov-Switching model
 */
export interface RegimeState {
  timestamp: string;
  stable: number; // Probability of stable regime
  volatile: number; // Probability of volatile regime
  crisis: number; // Probability of crisis regime
  transitionProbability: number; // P(regime change)
}

/**
 * CIC Theory Metrics - Causal Information Coherence
 */
export interface CICMetrics {
  // Integrated Information (Φ) - System irreducibility
  // Higher = more integrated/coherent system
  phi: number;

  // Conditional Entropy H(T|X)
  // Lower = better predictive compression
  entropy: number;

  // Multi-scale Causal Power (C_multi)
  // Effective causation across scales
  causalMulti: number;

  // Variational Free Energy F[T]
  // Surprise minimization objective
  freeEnergy: number;
}

/**
 * Intelligence packet for streaming feed
 */
export interface IntelPacket {
  id: string;
  timestamp: string;
  context: ContextLayer;
  category: string;
  header: string;
  summary: string;
  body: string;
  coherence: number; // 0-1 quality/confidence score
  source: string;
}

/**
 * Entropy source for external data feeds
 */
export interface EntropySource {
  id: string;
  source: 'CoinGecko' | 'OpenMeteo' | 'System' | 'Internal';
  value: number; // Normalized 0-1
  label: string;
  delta: number; // Change since last reading
}
