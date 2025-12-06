/**
 * ECAP - Entangled Co-Adaptive Protocol
 * TypeScript implementation for browser/web usage
 *
 * This is a lightweight client-side implementation for:
 * - Real-time visualization of human-AI co-adaptation
 * - User interaction telemetry processing
 * - Adaptation parameter preview
 *
 * For full training/inference, use the Python implementation.
 */

import type {
  CognitiveState,
  AIState,
  InteractionResult,
  ECAPMetrics,
  ECAPConfig,
  PatternRecognitionResult,
  HomeostasisState,
  EntanglementStats,
  ImmuneStats,
  ECAPSystemState,
} from './types';
import { DEFAULT_ECAP_CONFIG } from './types';

/**
 * Normalize vector to unit norm (quantum state preparation)
 */
function normalizeVector(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  if (norm < 1e-10) return vec.map(() => 0);
  return vec.map((v) => v / norm);
}

/**
 * Compute dot product of two vectors
 */
function dotProduct(a: number[], b: number[]): number {
  const minLen = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < minLen; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Sigmoid activation
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

/**
 * Create SHA-256 hash (simplified for browser)
 */
function createPatternId(features: number[]): string {
  // Simple hash based on quantized features
  const quantized = features.slice(0, 16).map((f) => Math.round(f * 100));
  return quantized.join('_').slice(0, 16);
}

/**
 * Lightweight immune memory for browser
 */
class ClientImmuneMemory {
  private selfPatterns: Map<string, number> = new Map();
  private nonSelfPatterns: Set<string> = new Set();
  private stats = {
    totalInteractions: 0,
    selfClassifications: 0,
    nonSelfClassifications: 0,
    patternRetrievals: 0,
  };

  constructor(
    private readonly selfThreshold: number = 0.7,
    private readonly maxPatterns: number = 500
  ) {}

  addInteraction(
    humanState: CognitiveState,
    aiState: AIState,
    reward: number
  ): string {
    const patternId = createPatternId(humanState.featureVector);
    this.stats.totalInteractions++;

    if (reward >= this.selfThreshold) {
      const existing = this.selfPatterns.get(patternId) ?? 0;
      this.selfPatterns.set(patternId, Math.min(1, existing + 0.1 * reward));
      this.stats.selfClassifications++;
      this.nonSelfPatterns.delete(patternId);
    } else {
      this.nonSelfPatterns.add(patternId);
      this.stats.nonSelfClassifications++;

      if (this.selfPatterns.has(patternId)) {
        const current = this.selfPatterns.get(patternId)!;
        if (current * 0.5 < 0.1) {
          this.selfPatterns.delete(patternId);
        } else {
          this.selfPatterns.set(patternId, current * 0.5);
        }
      }
    }

    // Enforce capacity limit
    if (this.selfPatterns.size > this.maxPatterns) {
      // Remove lowest confidence patterns
      const sorted = [...this.selfPatterns.entries()].sort((a, b) => a[1] - b[1]);
      for (let i = 0; i < sorted.length - this.maxPatterns; i++) {
        this.selfPatterns.delete(sorted[i][0]);
      }
    }

    return patternId;
  }

  checkPattern(humanState: CognitiveState): PatternRecognitionResult {
    const patternId = createPatternId(humanState.featureVector);
    this.stats.patternRetrievals++;

    if (this.nonSelfPatterns.has(patternId)) {
      return { isSelf: false, confidence: 0 };
    }

    if (this.selfPatterns.has(patternId)) {
      return {
        isSelf: true,
        confidence: this.selfPatterns.get(patternId)!,
        matchedPatternId: patternId,
      };
    }

    return { isSelf: false, confidence: 0.5 }; // Unknown
  }

  decay(rate: number = 0.995): void {
    for (const [id, conf] of this.selfPatterns.entries()) {
      const newConf = conf * rate;
      if (newConf < 0.01) {
        this.selfPatterns.delete(id);
      } else {
        this.selfPatterns.set(id, newConf);
      }
    }
  }

  getStats(): ImmuneStats {
    return {
      ...this.stats,
      selfPatternCount: this.selfPatterns.size,
      nonSelfPatternCount: this.nonSelfPatterns.size,
      memorySize: this.selfPatterns.size + this.nonSelfPatterns.size,
    };
  }
}

/**
 * Entanglement correlation monitor
 */
class ClientEntanglementMonitor {
  private correlationHistory: number[] = [];
  private readonly maxHistory: number;

  constructor(maxHistory: number = 50) {
    this.maxHistory = maxHistory;
  }

  computeCorrelation(humanFeatures: number[], aiEmbedding: number[]): number {
    // Normalize to quantum states
    const psiH = normalizeVector(humanFeatures);
    const psiA = normalizeVector(aiEmbedding);

    // Compute fidelity (squared overlap)
    const fidelity = Math.pow(Math.abs(dotProduct(psiH, psiA)), 2);

    // Dimension correction
    const dimCorrection = 1 - 1 / Math.sqrt(psiH.length + 1);
    const correlation = fidelity * dimCorrection;

    // Store in history
    this.correlationHistory.push(correlation);
    if (this.correlationHistory.length > this.maxHistory) {
      this.correlationHistory.shift();
    }

    return correlation;
  }

  getTrend(): number {
    if (this.correlationHistory.length < 5) return 0;

    const recent = this.correlationHistory.slice(-10);
    const n = recent.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = recent.reduce((a, b) => a + b, 0);
    const sumXY = recent.reduce((sum, y, i) => sum + i * y, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope;
  }

  getStats(): EntanglementStats {
    if (this.correlationHistory.length === 0) {
      return { avgCorrelation: 0, stdCorrelation: 0, trend: 0, historyLength: 0 };
    }

    const avg =
      this.correlationHistory.reduce((a, b) => a + b, 0) /
      this.correlationHistory.length;
    const variance =
      this.correlationHistory.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) /
      this.correlationHistory.length;

    return {
      avgCorrelation: avg,
      stdCorrelation: Math.sqrt(variance),
      trend: this.getTrend(),
      historyLength: this.correlationHistory.length,
    };
  }
}

/**
 * PID homeostasis controller
 */
class ClientHomeostasisController {
  private integralError = 0;
  private prevError = 0;
  private prevLoad = 0.5;
  private adaptationRate = 1;

  constructor(
    private readonly targetCorrelation: number = 0.85,
    private readonly targetLoad: number = 0.6,
    private readonly Kp: number = 0.8,
    private readonly Ki: number = 0.2,
    private readonly Kd: number = 0.1
  ) {}

  update(currentCorrelation: number, currentLoad: number, deltaTime: number = 1): number {
    const correlationError = this.targetCorrelation - currentCorrelation;
    const loadError = currentLoad - this.targetLoad;
    const loadDerivative = (currentLoad - this.prevLoad) / Math.max(deltaTime, 0.001);
    this.prevLoad = currentLoad;

    // Composite error (prioritize avoiding overload)
    const error = 0.3 * correlationError - 0.7 * loadError;

    // PID
    const P = this.Kp * error;
    this.integralError = Math.max(-2, Math.min(2, this.integralError + error * deltaTime));
    const I = this.Ki * this.integralError;
    const D = this.Kd * (error - this.prevError) / Math.max(deltaTime, 0.001);
    this.prevError = error;

    // Anticipatory control
    const anticipation = -0.5 * Math.tanh(loadDerivative * 2);

    // Control signal
    const control = P + I + D + anticipation;
    const rateChange = 0.1 * Math.tanh(control);
    this.adaptationRate = Math.max(0.01, Math.min(1, this.adaptationRate * (1 + rateChange)));

    return this.adaptationRate;
  }

  getState(): HomeostasisState {
    return {
      currentRate: this.adaptationRate,
      targetCorrelation: this.targetCorrelation,
      targetLoad: this.targetLoad,
      integralError: this.integralError,
    };
  }

  reset(): void {
    this.integralError = 0;
    this.prevError = 0;
    this.prevLoad = 0.5;
    this.adaptationRate = 1;
  }
}

/**
 * Main ECAP Engine for client-side usage
 */
export class ECAPEngine {
  private readonly config: ECAPConfig;
  private readonly immuneMemory: ClientImmuneMemory;
  private readonly entanglementMonitor: ClientEntanglementMonitor;
  private readonly homeostasis: ClientHomeostasisController;

  private generation = 0;
  private cumulativeReward = 0;
  private interactionCount = 0;

  private metrics: ECAPMetrics = {
    avgCorrelation: 0,
    avgLoad: 0,
    avgReward: 0,
    adaptationCount: 0,
    immuneHits: 0,
    selfPatternRatio: 0,
    generation: 0,
    cumulativeReward: 0,
    interactionCount: 0,
  };

  constructor(config: Partial<ECAPConfig> = {}) {
    this.config = { ...DEFAULT_ECAP_CONFIG, ...config };

    this.immuneMemory = new ClientImmuneMemory(
      this.config.selfThreshold,
      this.config.memoryCapacity
    );
    this.entanglementMonitor = new ClientEntanglementMonitor(50);
    this.homeostasis = new ClientHomeostasisController(
      this.config.targetCorrelation,
      this.config.targetLoad
    );
  }

  /**
   * Process a human-AI interaction
   */
  processInteraction(
    humanState: CognitiveState,
    aiState: AIState,
    externalReward?: number
  ): InteractionResult {
    this.interactionCount++;

    // Check immune memory
    const patternResult = this.immuneMemory.checkPattern(humanState);
    if (patternResult.isSelf) {
      this.metrics.immuneHits++;
    }

    // Compute correlation
    const aiEmbedding = new Array(this.config.aiDim).fill(aiState.embeddingMean);
    const correlation = this.entanglementMonitor.computeCorrelation(
      humanState.featureVector,
      aiEmbedding
    );

    // Update homeostasis
    const adaptationRate = this.homeostasis.update(
      correlation,
      humanState.cognitiveLoad
    );

    // Compute adaptation parameters (simplified)
    const regulation = 0.5 + 0.5 * patternResult.confidence;
    const learningRate = sigmoid(correlation - 0.5) * adaptationRate * regulation;
    const gradientScale = humanState.featureVector
      .slice(0, this.config.aiDim)
      .map((f) => sigmoid(f) * regulation);
    const attentionMask = humanState.attentionPattern.map((a) => sigmoid(a * 2));

    // Compute reward
    const reward = externalReward ?? correlation * (1 - humanState.cognitiveLoad);

    // Store in memory
    this.immuneMemory.addInteraction(humanState, aiState, reward);

    // Update metrics (EMA)
    const alpha = 0.1;
    this.metrics.avgCorrelation =
      (1 - alpha) * this.metrics.avgCorrelation + alpha * correlation;
    this.metrics.avgLoad =
      (1 - alpha) * this.metrics.avgLoad + alpha * humanState.cognitiveLoad;
    this.metrics.avgReward = (1 - alpha) * this.metrics.avgReward + alpha * reward;
    this.metrics.adaptationCount++;

    const immuneStats = this.immuneMemory.getStats();
    if (immuneStats.totalInteractions > 0) {
      this.metrics.selfPatternRatio =
        immuneStats.selfClassifications / immuneStats.totalInteractions;
    }

    this.cumulativeReward += reward;
    this.generation++;
    this.metrics.generation = this.generation;
    this.metrics.cumulativeReward = this.cumulativeReward;
    this.metrics.interactionCount = this.interactionCount;

    return {
      learningRate,
      gradientScale,
      attentionMask,
      regulation,
      correlation,
      adaptationRate,
      reward,
      isSelfPattern: patternResult.isSelf,
      immuneConfidence: patternResult.confidence,
    };
  }

  /**
   * Apply periodic memory decay
   */
  decayMemories(): void {
    this.immuneMemory.decay(this.config.decayRate);
  }

  /**
   * Get current system state
   */
  getSystemState(): ECAPSystemState {
    return {
      metrics: { ...this.metrics },
      homeostasis: this.homeostasis.getState(),
      entanglement: this.entanglementMonitor.getStats(),
      immune: this.immuneMemory.getStats(),
    };
  }

  /**
   * Get current metrics
   */
  getMetrics(): ECAPMetrics {
    return { ...this.metrics };
  }

  /**
   * Reset engine state (keeps learned patterns)
   */
  reset(): void {
    this.generation = 0;
    this.cumulativeReward = 0;
    this.interactionCount = 0;
    this.homeostasis.reset();
    this.metrics = {
      avgCorrelation: 0,
      avgLoad: 0,
      avgReward: 0,
      adaptationCount: 0,
      immuneHits: 0,
      selfPatternRatio: 0,
      generation: 0,
      cumulativeReward: 0,
      interactionCount: 0,
    };
  }
}

/**
 * Create cognitive state from user interaction telemetry
 */
export function createCognitiveStateFromTelemetry(telemetry: {
  keystrokeLatencies?: number[];
  mouseVelocities?: number[];
  scrollDepths?: number[];
  clickPatterns?: number[];
  sessionDuration?: number;
  errorCount?: number;
  backtrackCount?: number;
}): CognitiveState {
  // Compute features from telemetry
  const features: number[] = [];

  // Keystroke dynamics → attention/cognitive load proxy
  if (telemetry.keystrokeLatencies?.length) {
    const avgLatency = telemetry.keystrokeLatencies.reduce((a, b) => a + b, 0) /
      telemetry.keystrokeLatencies.length;
    const latencyVar = telemetry.keystrokeLatencies.reduce(
      (sum, l) => sum + Math.pow(l - avgLatency, 2),
      0
    ) / telemetry.keystrokeLatencies.length;
    features.push(avgLatency / 1000, Math.sqrt(latencyVar) / 500);
  } else {
    features.push(0.3, 0.1);
  }

  // Mouse dynamics
  if (telemetry.mouseVelocities?.length) {
    const avgVel = telemetry.mouseVelocities.reduce((a, b) => a + b, 0) /
      telemetry.mouseVelocities.length;
    features.push(avgVel / 1000);
  } else {
    features.push(0.5);
  }

  // Scroll behavior
  if (telemetry.scrollDepths?.length) {
    const maxScroll = Math.max(...telemetry.scrollDepths);
    features.push(maxScroll);
  } else {
    features.push(0.5);
  }

  // Pad to feature dimension
  while (features.length < 128) {
    features.push(Math.random() * 0.1);
  }

  // Attention pattern from click distribution
  const attention = telemetry.clickPatterns?.length
    ? telemetry.clickPatterns.slice(0, 10)
    : new Array(10).fill(0.1);

  // Normalize attention
  const attentionSum = attention.reduce((a, b) => a + b, 0) || 1;
  const normalizedAttention = attention.map((a) => a / attentionSum);

  // Cognitive load estimation
  const errorRate = (telemetry.errorCount ?? 0) / Math.max(1, telemetry.sessionDuration ?? 60);
  const backtrackRate = (telemetry.backtrackCount ?? 0) / Math.max(1, telemetry.sessionDuration ?? 60);
  const cognitiveLoad = Math.min(1, errorRate * 10 + backtrackRate * 5 + 0.2);

  // Entropy from action diversity
  const entropy = features.slice(0, 10).reduce(
    (sum, f) => sum - (f > 0 ? f * Math.log(f + 1e-10) : 0),
    0
  );

  return {
    featureVector: features,
    attentionPattern: normalizedAttention,
    cognitiveLoad,
    entropy: Math.max(0.1, Math.min(3, entropy)),
    timestamp: Date.now(),
  };
}

/**
 * Create AI state from model metadata
 */
export function createAIStateFromModel(model: {
  version?: string;
  embeddingMean?: number;
  embeddingStd?: number;
  generation?: number;
}): AIState {
  return {
    paramsHash: model.version ?? `model_${Date.now()}`,
    embeddingMean: model.embeddingMean ?? 0,
    embeddingStd: model.embeddingStd ?? 1,
    adaptationRate: 1,
    generation: model.generation ?? 0,
  };
}
