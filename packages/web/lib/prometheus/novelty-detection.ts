/**
 * Novelty Detection Engine
 *
 * Identifies "Unknown Knowns" - knowledge that exists implicitly
 * in the latent space but hasn't been explicitly surfaced.
 *
 * Uses information-theoretic measures to detect:
 * 1. Surprising patterns (high information content)
 * 2. Structural anomalies (deviation from expected manifold)
 * 3. Cross-domain isomorphisms (hidden connections)
 */

import {
  klDivergenceGaussian,
  fisherRaoDistance,
  surpriseGaussian,
  type DistributionParams,
} from '../physics/information-geometry';

export type NoveltyType =
  | 'PATTERN_ANOMALY'      // Unexpected statistical pattern
  | 'STRUCTURAL_GAP'       // Missing connection in knowledge graph
  | 'CROSS_DOMAIN_ISO'     // Isomorphism between unrelated fields
  | 'TEMPORAL_BREAK'       // Regime shift in time series
  | 'CAUSAL_LOOP'          // Unexpected feedback mechanism
  | 'EMERGENT_PROPERTY';   // Macro behavior not in micro rules

export interface NoveltySignal {
  id: string;
  type: NoveltyType;
  timestamp: string;
  description: string;
  confidence: number;       // 0-1: How confident are we this is real novelty?
  significance: number;     // 0-1: How important if true?
  evidence: string[];
  suggestedAction: string;
  sourceData: Record<string, unknown>;
}

export interface KnowledgeQuadrant {
  knownKnowns: string[];      // Established facts
  knownUnknowns: string[];    // Acknowledged gaps
  unknownUnknowns: string[];  // Discovered surprises
  unknownKnowns: string[];    // Implicit knowledge surfaced
}

export interface NoveltyConfig {
  surpriseThreshold: number;      // z-score threshold for anomaly
  klThreshold: number;            // KL divergence threshold
  fisherThreshold: number;        // Fisher distance threshold
  minConfidence: number;          // Minimum confidence to report
  lookbackWindow: number;         // Time steps for baseline
}

const DEFAULT_CONFIG: NoveltyConfig = {
  surpriseThreshold: 2.5,
  klThreshold: 0.5,
  fisherThreshold: 1.0,
  minConfidence: 0.6,
  lookbackWindow: 100,
};

/**
 * Detect pattern anomalies in time series data
 */
export function detectPatternAnomaly(
  observations: number[],
  config: Partial<NoveltyConfig> = {}
): NoveltySignal[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const signals: NoveltySignal[] = [];

  if (observations.length < cfg.lookbackWindow + 1) {
    return signals;
  }

  // Compute baseline distribution
  const baseline = observations.slice(0, cfg.lookbackWindow);
  const mean = baseline.reduce((a, b) => a + b, 0) / baseline.length;
  const variance = baseline.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / baseline.length;

  const baselineParams: DistributionParams = { mean, variance };

  // Scan for anomalies
  for (let i = cfg.lookbackWindow; i < observations.length; i++) {
    const value = observations[i];
    const surprise = surpriseGaussian(value, baselineParams);
    const zScore = Math.abs(value - mean) / Math.sqrt(variance);

    if (zScore > cfg.surpriseThreshold) {
      const confidence = Math.min(1, zScore / 5);
      const significance = Math.min(1, surprise / 10);

      signals.push({
        id: `anomaly_${i}`,
        type: 'PATTERN_ANOMALY',
        timestamp: new Date(Date.now() - (observations.length - i) * 3600000).toISOString(),
        description: `Observation ${value.toFixed(2)} is ${zScore.toFixed(1)}σ from baseline`,
        confidence,
        significance,
        evidence: [
          `Z-score: ${zScore.toFixed(2)}`,
          `Surprise: ${surprise.toFixed(2)} bits`,
          `Baseline mean: ${mean.toFixed(2)}`,
          `Baseline std: ${Math.sqrt(variance).toFixed(2)}`,
        ],
        suggestedAction: 'Investigate root cause of deviation',
        sourceData: { index: i, value, zScore, surprise },
      });
    }
  }

  return signals;
}

/**
 * Detect temporal regime breaks using distribution shift
 */
export function detectTemporalBreak(
  observations: number[],
  windowSize: number = 50,
  config: Partial<NoveltyConfig> = {}
): NoveltySignal[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const signals: NoveltySignal[] = [];

  if (observations.length < windowSize * 2) {
    return signals;
  }

  // Sliding window comparison
  for (let i = windowSize; i < observations.length - windowSize; i++) {
    const before = observations.slice(i - windowSize, i);
    const after = observations.slice(i, i + windowSize);

    const meanBefore = before.reduce((a, b) => a + b, 0) / before.length;
    const varBefore = before.reduce((a, b) => a + Math.pow(b - meanBefore, 2), 0) / before.length;

    const meanAfter = after.reduce((a, b) => a + b, 0) / after.length;
    const varAfter = after.reduce((a, b) => a + Math.pow(b - meanAfter, 2), 0) / after.length;

    const distBefore: DistributionParams = { mean: meanBefore, variance: Math.max(0.01, varBefore) };
    const distAfter: DistributionParams = { mean: meanAfter, variance: Math.max(0.01, varAfter) };

    const kl = klDivergenceGaussian(distAfter, distBefore);
    const fisher = fisherRaoDistance(distAfter, distBefore);

    if (kl > cfg.klThreshold || fisher > cfg.fisherThreshold) {
      const confidence = Math.min(1, Math.max(kl / cfg.klThreshold, fisher / cfg.fisherThreshold) / 2);

      if (confidence >= cfg.minConfidence) {
        signals.push({
          id: `regime_break_${i}`,
          type: 'TEMPORAL_BREAK',
          timestamp: new Date(Date.now() - (observations.length - i) * 3600000).toISOString(),
          description: `Regime shift detected: distribution changed significantly`,
          confidence,
          significance: Math.min(1, kl),
          evidence: [
            `KL divergence: ${kl.toFixed(3)}`,
            `Fisher distance: ${fisher.toFixed(3)}`,
            `Mean shift: ${meanBefore.toFixed(2)} → ${meanAfter.toFixed(2)}`,
            `Variance shift: ${varBefore.toFixed(2)} → ${varAfter.toFixed(2)}`,
          ],
          suggestedAction: 'Analyze events around breakpoint for causal factors',
          sourceData: { index: i, kl, fisher, meanBefore, meanAfter },
        });

        // Skip ahead to avoid duplicate detections
        i += windowSize / 2;
      }
    }
  }

  return signals;
}

/**
 * Detect cross-domain isomorphisms
 *
 * Finds structural similarities between different data streams
 * that might indicate hidden causal connections.
 */
export function detectCrossDomainIso(
  series1: { name: string; data: number[] },
  series2: { name: string; data: number[] },
  lagRange: number = 20,
  config: Partial<NoveltyConfig> = {}
): NoveltySignal[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const signals: NoveltySignal[] = [];

  const minLength = Math.min(series1.data.length, series2.data.length);
  if (minLength < lagRange * 2) {
    return signals;
  }

  // Compute cross-correlation at different lags
  let maxCorr = 0;
  let bestLag = 0;

  for (let lag = -lagRange; lag <= lagRange; lag++) {
    const corr = computeCorrelation(series1.data, series2.data, lag);
    if (Math.abs(corr) > Math.abs(maxCorr)) {
      maxCorr = corr;
      bestLag = lag;
    }
  }

  // Significant correlation suggests isomorphism
  if (Math.abs(maxCorr) > 0.7) {
    const confidence = Math.abs(maxCorr);
    const significance = Math.min(1, Math.abs(maxCorr) * 1.2);

    signals.push({
      id: `iso_${series1.name}_${series2.name}`,
      type: 'CROSS_DOMAIN_ISO',
      timestamp: new Date().toISOString(),
      description: `Strong correlation (${maxCorr.toFixed(2)}) between ${series1.name} and ${series2.name} at lag ${bestLag}`,
      confidence,
      significance,
      evidence: [
        `Correlation: ${maxCorr.toFixed(3)}`,
        `Optimal lag: ${bestLag} time steps`,
        `${series1.name} ${bestLag > 0 ? 'leads' : 'lags'} ${series2.name}`,
      ],
      suggestedAction: bestLag !== 0
        ? `Investigate if ${bestLag > 0 ? series1.name : series2.name} causes ${bestLag > 0 ? series2.name : series1.name}`
        : 'Look for common cause driving both series',
      sourceData: { series1: series1.name, series2: series2.name, correlation: maxCorr, lag: bestLag },
    });
  }

  return signals;
}

/**
 * Detect structural gaps in knowledge graph
 *
 * Finds missing edges that should exist based on graph structure.
 */
export function detectStructuralGap(
  nodes: Array<{ id: string; attributes: Record<string, number> }>,
  edges: Array<{ source: string; target: string; weight: number }>,
  config: Partial<NoveltyConfig> = {}
): NoveltySignal[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const signals: NoveltySignal[] = [];

  // Build adjacency and compute node similarity
  const edgeSet = new Set(edges.map(e => `${e.source}-${e.target}`));

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const node1 = nodes[i];
      const node2 = nodes[j];

      // Check if edge exists
      if (edgeSet.has(`${node1.id}-${node2.id}`) || edgeSet.has(`${node2.id}-${node1.id}`)) {
        continue;
      }

      // Compute attribute similarity
      const similarity = computeAttributeSimilarity(node1.attributes, node2.attributes);

      // Check for common neighbors (triadic closure)
      const neighbors1 = new Set(
        edges.filter(e => e.source === node1.id || e.target === node1.id)
          .map(e => e.source === node1.id ? e.target : e.source)
      );
      const neighbors2 = new Set(
        edges.filter(e => e.source === node2.id || e.target === node2.id)
          .map(e => e.source === node2.id ? e.target : e.source)
      );

      const commonNeighbors = [...neighbors1].filter(n => neighbors2.has(n));
      const triadicScore = commonNeighbors.length / Math.sqrt(neighbors1.size * neighbors2.size || 1);

      // High similarity + many common neighbors but no edge = structural gap
      if (similarity > 0.7 && triadicScore > 0.3) {
        const confidence = (similarity + triadicScore) / 2;

        if (confidence >= cfg.minConfidence) {
          signals.push({
            id: `gap_${node1.id}_${node2.id}`,
            type: 'STRUCTURAL_GAP',
            timestamp: new Date().toISOString(),
            description: `Missing edge between similar nodes: ${node1.id} and ${node2.id}`,
            confidence,
            significance: triadicScore,
            evidence: [
              `Attribute similarity: ${similarity.toFixed(2)}`,
              `Common neighbors: ${commonNeighbors.length}`,
              `Triadic closure score: ${triadicScore.toFixed(2)}`,
            ],
            suggestedAction: 'Investigate potential relationship between these entities',
            sourceData: { node1: node1.id, node2: node2.id, similarity, commonNeighbors },
          });
        }
      }
    }
  }

  return signals;
}

/**
 * Aggregate novelty signals into knowledge quadrants
 */
export function buildKnowledgeQuadrant(
  signals: NoveltySignal[],
  existingKnowledge: string[]
): KnowledgeQuadrant {
  const quadrant: KnowledgeQuadrant = {
    knownKnowns: [...existingKnowledge],
    knownUnknowns: [],
    unknownUnknowns: [],
    unknownKnowns: [],
  };

  for (const signal of signals) {
    switch (signal.type) {
      case 'PATTERN_ANOMALY':
      case 'TEMPORAL_BREAK':
        // These reveal unknown unknowns - things we didn't know we didn't know
        quadrant.unknownUnknowns.push(signal.description);
        break;

      case 'CROSS_DOMAIN_ISO':
        // These surface unknown knowns - implicit knowledge made explicit
        quadrant.unknownKnowns.push(signal.description);
        break;

      case 'STRUCTURAL_GAP':
        // These identify known unknowns - gaps we can now see
        quadrant.knownUnknowns.push(signal.description);
        break;

      case 'CAUSAL_LOOP':
      case 'EMERGENT_PROPERTY':
        // Complex phenomena - depends on confidence
        if (signal.confidence > 0.8) {
          quadrant.unknownKnowns.push(signal.description);
        } else {
          quadrant.unknownUnknowns.push(signal.description);
        }
        break;
    }
  }

  return quadrant;
}

/**
 * Compute priority score for investigating novelty signals
 */
export function prioritizeNovelties(signals: NoveltySignal[]): NoveltySignal[] {
  return signals
    .map(signal => ({
      ...signal,
      // Priority = confidence × significance × type weight
      priority: signal.confidence * signal.significance * getTypeWeight(signal.type),
    }))
    .sort((a, b) => (b as any).priority - (a as any).priority);
}

function getTypeWeight(type: NoveltyType): number {
  const weights: Record<NoveltyType, number> = {
    'CAUSAL_LOOP': 1.5,
    'CROSS_DOMAIN_ISO': 1.4,
    'EMERGENT_PROPERTY': 1.3,
    'TEMPORAL_BREAK': 1.2,
    'STRUCTURAL_GAP': 1.1,
    'PATTERN_ANOMALY': 1.0,
  };
  return weights[type];
}

// Helper functions

function computeCorrelation(series1: number[], series2: number[], lag: number): number {
  const n = Math.min(series1.length, series2.length) - Math.abs(lag);
  if (n < 10) return 0;

  let s1: number[], s2: number[];
  if (lag >= 0) {
    s1 = series1.slice(0, n);
    s2 = series2.slice(lag, lag + n);
  } else {
    s1 = series1.slice(-lag, -lag + n);
    s2 = series2.slice(0, n);
  }

  const mean1 = s1.reduce((a, b) => a + b, 0) / n;
  const mean2 = s2.reduce((a, b) => a + b, 0) / n;

  let cov = 0, var1 = 0, var2 = 0;
  for (let i = 0; i < n; i++) {
    const d1 = s1[i] - mean1;
    const d2 = s2[i] - mean2;
    cov += d1 * d2;
    var1 += d1 * d1;
    var2 += d2 * d2;
  }

  return cov / Math.sqrt(var1 * var2 || 1);
}

function computeAttributeSimilarity(
  attrs1: Record<string, number>,
  attrs2: Record<string, number>
): number {
  const keys = new Set([...Object.keys(attrs1), ...Object.keys(attrs2)]);
  if (keys.size === 0) return 0;

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (const key of keys) {
    const v1 = attrs1[key] || 0;
    const v2 = attrs2[key] || 0;
    dotProduct += v1 * v2;
    norm1 += v1 * v1;
    norm2 += v2 * v2;
  }

  return dotProduct / Math.sqrt(norm1 * norm2 || 1);
}
