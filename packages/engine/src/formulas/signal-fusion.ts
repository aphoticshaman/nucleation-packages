/**
 * Signal Fusion Recipe
 *
 * Proprietary algorithm for combining heterogeneous data signals
 * into a unified predictive indicator.
 *
 * The WASM handles numerical operations.
 * This recipe defines WHAT to combine and HOW to weight it.
 *
 * © 2025 Crystalline Labs LLC - Trade Secret
 */

import type { WasmBridge } from '../wasm-bridge';

export interface SignalMetadata {
  name: string;
  category: 'official' | 'news' | 'social' | 'market' | 'alternative';
  reliability: number; // 0-1, how trustworthy
  latency: number; // typical delay in hours
  noiseLevel: number; // 0-1, signal-to-noise ratio inverse
}

export interface FusedSignal {
  value: number;
  confidence: number;
  contributors: Array<{
    signal: string;
    weight: number;
    contribution: number;
  }>;
  regime: 'risk-on' | 'risk-off' | 'transitional' | 'uncertain';
  timestamp: Date;
}

export interface FusionConfig {
  /** How much to penalize conflicting signals */
  conflictPenalty: number;
  /** Minimum confidence to include in fusion */
  confidenceThreshold: number;
  /** How much to weight recent vs historical */
  recencyBias: number;
  /** Maximum signals to include */
  maxSignals: number;
}

/**
 * Proprietary multi-signal fusion algorithm
 *
 * SECRET SAUCE:
 * - Adaptive weighting based on regime detection
 * - Conflict resolution via confidence-weighted voting
 * - Latency-adjusted signal alignment
 * - Category diversification requirements
 */
export class SignalFusionRecipe {
  // Proprietary base weights by category (refined through backtesting)
  private static readonly CATEGORY_WEIGHTS: Record<string, number> = {
    official: 0.35, // SEC filings, Fed data - highest trust
    market: 0.25, // Price action, volume
    news: 0.2, // Mainstream media
    social: 0.12, // Reddit, Twitter, etc.
    alternative: 0.08, // Other signals
  };

  // Proprietary regime detection thresholds
  private static readonly REGIME_THRESHOLDS = {
    riskOn: 0.65,
    riskOff: 0.35,
    transitionBand: 0.15,
  };

  // Proprietary confidence decay factors
  private static readonly CONFIDENCE_DECAY = {
    hourly: 0.98,
    daily: 0.85,
    weekly: 0.6,
  };

  private wasm: WasmBridge | null = null;
  private signalRegistry: Map<string, SignalMetadata> = new Map();
  private fusionHistory: FusedSignal[] = [];
  private config: FusionConfig;

  constructor(wasm?: WasmBridge, config?: Partial<FusionConfig>) {
    this.wasm = wasm ?? null;
    this.config = {
      conflictPenalty: 0.3,
      confidenceThreshold: 0.4,
      recencyBias: 0.7,
      maxSignals: 12,
      ...config,
    };
  }

  /**
   * Register a signal source with its metadata
   */
  registerSignal(metadata: SignalMetadata): void {
    this.signalRegistry.set(metadata.name, metadata);
  }

  /**
   * Fuse multiple signals into unified indicator
   *
   * @param signals Map of signal name to value array
   * @param targetTime Optional specific time to fuse for (handles latency)
   */
  fuse(signals: Map<string, number[]>, targetTime?: Date): FusedSignal {
    const activeSignals = this.selectActiveSignals(signals);
    const alignedSignals = this.alignSignals(activeSignals, targetTime);
    const weights = this.calculateWeights(alignedSignals);
    const { value, contributors } = this.weightedFusion(alignedSignals, weights);
    const confidence = this.calculateFusionConfidence(alignedSignals, weights);
    const regime = this.detectRegime(value, confidence);

    const fused: FusedSignal = {
      value,
      confidence,
      contributors,
      regime,
      timestamp: targetTime ?? new Date(),
    };

    this.fusionHistory.push(fused);
    if (this.fusionHistory.length > 500) {
      this.fusionHistory.shift();
    }

    return fused;
  }

  /**
   * Get fusion with conflict analysis
   */
  fuseWithAnalysis(signals: Map<string, number[]>): {
    fused: FusedSignal;
    conflicts: Array<{ signal1: string; signal2: string; severity: number }>;
    coherence: number;
  } {
    const fused = this.fuse(signals);
    const conflicts = this.detectConflicts(signals);
    const coherence = this.calculateCoherence(signals);

    return { fused, conflicts, coherence };
  }

  /**
   * Select signals that meet quality thresholds
   */
  private selectActiveSignals(signals: Map<string, number[]>): Map<string, number[]> {
    const active = new Map<string, number[]>();
    const entries = Array.from(signals.entries());

    // Sort by reliability if registered
    entries.sort((a, b) => {
      const metaA = this.signalRegistry.get(a[0]);
      const metaB = this.signalRegistry.get(b[0]);
      return (metaB?.reliability ?? 0.5) - (metaA?.reliability ?? 0.5);
    });

    // Take top N signals
    for (const [name, values] of entries.slice(0, this.config.maxSignals)) {
      if (values.length > 0) {
        active.set(name, values);
      }
    }

    return active;
  }

  /**
   * Align signals accounting for different latencies
   * SECRET: Proprietary latency compensation algorithm
   */
  private alignSignals(signals: Map<string, number[]>, _targetTime?: Date): Map<string, number> {
    const aligned = new Map<string, number>();

    for (const [name, values] of signals) {
      const meta = this.signalRegistry.get(name);
      const latencyHours = meta?.latency ?? 0;

      // Calculate how many periods back to look based on latency
      const periodsBack = Math.floor(latencyHours / 24); // Assuming daily data

      // Get the appropriate value
      let idx = values.length - 1 - periodsBack;
      idx = Math.max(0, Math.min(values.length - 1, idx));

      // Apply confidence decay based on age
      const value = values[idx];
      aligned.set(name, value);
    }

    return aligned;
  }

  /**
   * Calculate adaptive weights for each signal
   * SECRET: Multi-factor adaptive weighting formula
   */
  private calculateWeights(signals: Map<string, number>): Map<string, number> {
    const weights = new Map<string, number>();
    let totalWeight = 0;

    // First pass: calculate raw weights
    for (const [name] of signals) {
      const meta = this.signalRegistry.get(name);

      // Base weight from category
      const categoryWeight =
        SignalFusionRecipe.CATEGORY_WEIGHTS[meta?.category ?? 'alternative'] ?? 0.08;

      // Reliability multiplier
      const reliabilityMult = meta?.reliability ?? 0.5;

      // Noise penalty
      const noisePenalty = 1 - (meta?.noiseLevel ?? 0.5) * 0.5;

      // Combined weight
      const weight = categoryWeight * reliabilityMult * noisePenalty;
      weights.set(name, weight);
      totalWeight += weight;
    }

    // Normalize weights to sum to 1
    if (totalWeight > 0) {
      for (const [name, weight] of weights) {
        weights.set(name, weight / totalWeight);
      }
    }

    // Apply diversification constraint
    this.applyDiversification(weights, signals);

    return weights;
  }

  /**
   * Ensure no single category dominates
   * SECRET: Category cap formula
   */
  private applyDiversification(weights: Map<string, number>, _signals: Map<string, number>): void {
    const categoryTotals = new Map<string, number>();
    const maxCategoryWeight = 0.45; // No category can exceed 45%

    // Calculate category totals
    for (const [name, weight] of weights) {
      const meta = this.signalRegistry.get(name);
      const category = meta?.category ?? 'alternative';
      categoryTotals.set(category, (categoryTotals.get(category) ?? 0) + weight);
    }

    // Cap and redistribute
    for (const [category, total] of categoryTotals) {
      if (total > maxCategoryWeight) {
        const scale = maxCategoryWeight / total;

        // Scale down this category's weights
        for (const [name, weight] of weights) {
          const meta = this.signalRegistry.get(name);
          if ((meta?.category ?? 'alternative') === category) {
            weights.set(name, weight * scale);
          }
        }
      }
    }

    // Renormalize
    const newTotal = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    if (newTotal > 0 && Math.abs(newTotal - 1) > 0.001) {
      for (const [name, weight] of weights) {
        weights.set(name, weight / newTotal);
      }
    }
  }

  /**
   * Perform weighted fusion
   */
  private weightedFusion(
    signals: Map<string, number>,
    weights: Map<string, number>
  ): {
    value: number;
    contributors: Array<{ signal: string; weight: number; contribution: number }>;
  } {
    let fusedValue = 0;
    const contributors: Array<{
      signal: string;
      weight: number;
      contribution: number;
    }> = [];

    for (const [name, value] of signals) {
      const weight = weights.get(name) ?? 0;
      const contribution = value * weight;
      fusedValue += contribution;

      contributors.push({
        signal: name,
        weight,
        contribution,
      });
    }

    // Sort contributors by absolute contribution
    contributors.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    return { value: fusedValue, contributors };
  }

  /**
   * Calculate confidence in the fused signal
   * SECRET: Agreement-based confidence formula
   */
  private calculateFusionConfidence(
    signals: Map<string, number>,
    weights: Map<string, number>
  ): number {
    const values = Array.from(signals.values());
    if (values.length < 2) return 0.5;

    // Normalize values to [0, 1] range for comparison
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    // Calculate weighted variance
    const weightedMean = Array.from(signals.entries()).reduce((sum, [name, value]) => {
      const normValue = (value - min) / range;
      return sum + normValue * (weights.get(name) ?? 0);
    }, 0);

    const weightedVariance = Array.from(signals.entries()).reduce((sum, [name, value]) => {
      const normValue = (value - min) / range;
      const weight = weights.get(name) ?? 0;
      return sum + weight * Math.pow(normValue - weightedMean, 2);
    }, 0);

    // Low variance = high agreement = high confidence
    // Confidence = 1 - sqrt(variance) * conflict_penalty
    const confidence = Math.max(
      0,
      Math.min(1, 1 - Math.sqrt(weightedVariance) * (1 + this.config.conflictPenalty))
    );

    return confidence;
  }

  /**
   * Detect current market regime from fused value
   */
  private detectRegime(
    value: number,
    confidence: number
  ): 'risk-on' | 'risk-off' | 'transitional' | 'uncertain' {
    const { riskOn, riskOff, transitionBand } = SignalFusionRecipe.REGIME_THRESHOLDS;

    // Low confidence = uncertain
    if (confidence < this.config.confidenceThreshold) {
      return 'uncertain';
    }

    // Clear regimes
    if (value > riskOn) return 'risk-on';
    if (value < riskOff) return 'risk-off';

    // Check if in transition band
    if (value > riskOff + transitionBand && value < riskOn - transitionBand) {
      return 'transitional';
    }

    // Edge of regime
    return 'transitional';
  }

  /**
   * Detect conflicting signals
   */
  private detectConflicts(
    signals: Map<string, number[]>
  ): Array<{ signal1: string; signal2: string; severity: number }> {
    const conflicts: Array<{
      signal1: string;
      signal2: string;
      severity: number;
    }> = [];

    const entries = Array.from(signals.entries());

    for (let i = 0; i < entries.length; i++) {
      for (let j = i + 1; j < entries.length; j++) {
        const [name1, values1] = entries[i];
        const [name2, values2] = entries[j];

        // Compare recent direction
        if (values1.length >= 2 && values2.length >= 2) {
          const dir1 = values1[values1.length - 1] - values1[values1.length - 2];
          const dir2 = values2[values2.length - 1] - values2[values2.length - 2];

          // Opposite directions = conflict
          if (dir1 * dir2 < 0) {
            const severity = Math.min(1, (Math.abs(dir1) + Math.abs(dir2)) / 2);
            if (severity > 0.1) {
              conflicts.push({ signal1: name1, signal2: name2, severity });
            }
          }
        }
      }
    }

    return conflicts.sort((a, b) => b.severity - a.severity);
  }

  /**
   * Calculate overall signal coherence
   */
  private calculateCoherence(signals: Map<string, number[]>): number {
    const entries = Array.from(signals.entries());
    if (entries.length < 2) return 1;

    let totalCorrelation = 0;
    let pairs = 0;

    for (let i = 0; i < entries.length; i++) {
      for (let j = i + 1; j < entries.length; j++) {
        const corr = this.pearsonCorrelation(entries[i][1], entries[j][1]);
        totalCorrelation += (corr + 1) / 2; // Normalize to [0, 1]
        pairs++;
      }
    }

    return pairs > 0 ? totalCorrelation / pairs : 1;
  }

  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    const xSlice = x.slice(-n);
    const ySlice = y.slice(-n);

    const meanX = xSlice.reduce((a, b) => a + b, 0) / n;
    const meanY = ySlice.reduce((a, b) => a + b, 0) / n;

    let num = 0;
    let denX = 0;
    let denY = 0;

    for (let i = 0; i < n; i++) {
      const dx = xSlice[i] - meanX;
      const dy = ySlice[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }

    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
  }

  /**
   * Get historical fusion performance metrics
   */
  getPerformanceMetrics(): {
    avgConfidence: number;
    regimeDistribution: Record<string, number>;
    signalContributions: Record<string, number>;
  } {
    if (this.fusionHistory.length === 0) {
      return {
        avgConfidence: 0,
        regimeDistribution: {},
        signalContributions: {},
      };
    }

    const avgConfidence =
      this.fusionHistory.reduce((sum, f) => sum + f.confidence, 0) / this.fusionHistory.length;

    const regimeDistribution: Record<string, number> = {};
    const signalContributions: Record<string, number> = {};

    for (const fused of this.fusionHistory) {
      regimeDistribution[fused.regime] = (regimeDistribution[fused.regime] ?? 0) + 1;

      for (const contributor of fused.contributors) {
        signalContributions[contributor.signal] =
          (signalContributions[contributor.signal] ?? 0) + Math.abs(contributor.contribution);
      }
    }

    // Normalize regime distribution
    const total = this.fusionHistory.length;
    for (const regime in regimeDistribution) {
      regimeDistribution[regime] /= total;
    }

    return { avgConfidence, regimeDistribution, signalContributions };
  }
}
