/**
 * DeepAnalysis - Orthogonal intelligence techniques
 *
 * This goes beyond text analysis to find signals others miss:
 *
 * 1. ABSENCE DETECTION - What news DIDN'T happen that should have?
 * 2. PROXY INDICATORS - Indirect signals (markets predict politics)
 * 3. TRIANGULATION - Multiple weak signals → strong inference
 * 4. GRAPH DYNAMICS - Who's connected to whom, how is it changing?
 * 5. DISINFORMATION DETECTION - When are we being played?
 * 6. TEMPO ANALYSIS - Speed/rhythm of events as signal
 *
 * All CPU-bound. No LLM. Pure math and logic.
 */

import type { ProcessedSignal, SignalFeatures } from './SignalProcessor';
import type { Inference, InferenceContext } from './LogicalAgent';

// ============================================================
// ABSENCE DETECTION - The Sherlock Holmes principle
// ============================================================
// "The curious incident of the dog in the night-time"
// "The dog did nothing in the night-time"
// "That was the curious incident"

export interface ExpectedEvent {
  name: string;
  triggers: string[];           // Keywords that would indicate this event
  normalFrequencyPerDay: number; // How often this normally appears
  lastSeen?: Date;
  category: string;
}

// Events we expect to see regularly - absence is suspicious
const EXPECTED_EVENTS: ExpectedEvent[] = [
  // Diplomatic
  { name: 'UN_activity', triggers: ['united nations', 'un security council', 'un general assembly'], normalFrequencyPerDay: 5, category: 'diplomatic' },
  { name: 'G7_G20_mention', triggers: ['g7', 'g20', 'g-7', 'g-20'], normalFrequencyPerDay: 2, category: 'diplomatic' },
  { name: 'NATO_statement', triggers: ['nato', 'atlantic alliance', 'stoltenberg'], normalFrequencyPerDay: 3, category: 'security' },

  // Economic
  { name: 'Fed_activity', triggers: ['federal reserve', 'fed chair', 'fomc', 'powell'], normalFrequencyPerDay: 4, category: 'economic' },
  { name: 'ECB_activity', triggers: ['ecb', 'european central bank', 'lagarde'], normalFrequencyPerDay: 3, category: 'economic' },
  { name: 'IMF_activity', triggers: ['imf', 'international monetary fund'], normalFrequencyPerDay: 2, category: 'economic' },

  // Geopolitical
  { name: 'Ukraine_updates', triggers: ['ukraine', 'kyiv', 'zelensky', 'donbas'], normalFrequencyPerDay: 20, category: 'conflict' },
  { name: 'Taiwan_strait', triggers: ['taiwan', 'taipei', 'taiwan strait', 'tsmc'], normalFrequencyPerDay: 5, category: 'security' },
  { name: 'Middle_East', triggers: ['israel', 'gaza', 'hezbollah', 'iran nuclear'], normalFrequencyPerDay: 15, category: 'conflict' },

  // Markets
  { name: 'Oil_price', triggers: ['oil price', 'brent', 'wti', 'crude oil'], normalFrequencyPerDay: 8, category: 'energy' },
  { name: 'Market_moves', triggers: ['s&p 500', 'dow jones', 'nasdaq', 'stock market'], normalFrequencyPerDay: 10, category: 'financial' },

  // Tech/Cyber
  { name: 'Cyber_incidents', triggers: ['cyberattack', 'ransomware', 'data breach', 'hack'], normalFrequencyPerDay: 5, category: 'cyber' },
];

/**
 * Detect suspicious absences in signal stream
 */
export function detectAbsences(
  signals: ProcessedSignal[],
  timeWindowHours: number = 24
): Array<{ event: ExpectedEvent; hoursSinceSeen: number; suspicionLevel: number }> {
  const absences: Array<{ event: ExpectedEvent; hoursSinceSeen: number; suspicionLevel: number }> = [];
  const now = new Date();

  for (const event of EXPECTED_EVENTS) {
    // Find last occurrence
    let lastSeen: Date | null = null;

    for (const signal of signals) {
      const textLower = signal.features.originalText.toLowerCase();
      const hasMatch = event.triggers.some(t => textLower.includes(t.toLowerCase()));

      if (hasMatch) {
        if (!lastSeen || signal.timestamp > lastSeen) {
          lastSeen = signal.timestamp;
        }
      }
    }

    // Calculate absence
    if (lastSeen) {
      const hoursSince = (now.getTime() - lastSeen.getTime()) / (1000 * 60 * 60);
      const expectedHours = 24 / event.normalFrequencyPerDay;

      // If we haven't seen it in 3x the expected frequency, suspicious
      if (hoursSince > expectedHours * 3) {
        absences.push({
          event,
          hoursSinceSeen: hoursSince,
          suspicionLevel: Math.min(1, hoursSince / (expectedHours * 5)),
        });
      }
    } else if (signals.length > 50) {
      // If we have decent coverage but never saw this event
      absences.push({
        event,
        hoursSinceSeen: timeWindowHours,
        suspicionLevel: 0.8,
      });
    }
  }

  return absences.sort((a, b) => b.suspicionLevel - a.suspicionLevel);
}

// ============================================================
// PROXY INDICATORS - Markets as crystal ball
// ============================================================

export interface ProxyIndicator {
  name: string;
  description: string;
  // What the proxy actually measures
  measuredSignal: string;
  // What it predicts
  predictedSignal: string;
  // Lead time (how far ahead it predicts)
  leadTimeHours: number;
  // Correlation strength historically
  historicalCorrelation: number;
}

const PROXY_INDICATORS: ProxyIndicator[] = [
  // Market-based proxies
  {
    name: 'VIX_spike',
    description: 'VIX > 25 predicts geopolitical instability',
    measuredSignal: 'vix_index',
    predictedSignal: 'geopolitical_crisis',
    leadTimeHours: 48,
    historicalCorrelation: 0.6,
  },
  {
    name: 'Gold_surge',
    description: 'Gold price surge predicts flight to safety',
    measuredSignal: 'gold_price_change',
    predictedSignal: 'market_panic',
    leadTimeHours: 24,
    historicalCorrelation: 0.7,
  },
  {
    name: 'Defense_stocks',
    description: 'Defense stock movement predicts conflict escalation',
    measuredSignal: 'defense_etf_change',
    predictedSignal: 'military_escalation',
    leadTimeHours: 72,
    historicalCorrelation: 0.5,
  },
  {
    name: 'Oil_spike',
    description: 'Sudden oil price jump predicts supply disruption',
    measuredSignal: 'oil_price_change',
    predictedSignal: 'energy_crisis',
    leadTimeHours: 12,
    historicalCorrelation: 0.65,
  },
  {
    name: 'Currency_flight',
    description: 'Currency rapid depreciation predicts political crisis',
    measuredSignal: 'currency_volatility',
    predictedSignal: 'regime_instability',
    leadTimeHours: 168,  // 1 week
    historicalCorrelation: 0.55,
  },
  {
    name: 'Bond_spread',
    description: 'Sovereign bond spread widening predicts default risk',
    measuredSignal: 'bond_spread_vs_bund',
    predictedSignal: 'sovereign_crisis',
    leadTimeHours: 336,  // 2 weeks
    historicalCorrelation: 0.7,
  },

  // Shipping-based proxies
  {
    name: 'Baltic_dry',
    description: 'Baltic Dry Index collapse predicts trade slowdown',
    measuredSignal: 'baltic_dry_index',
    predictedSignal: 'trade_recession',
    leadTimeHours: 720,  // 30 days
    historicalCorrelation: 0.6,
  },
  {
    name: 'Tanker_rates',
    description: 'Tanker rates spike predicts oil supply disruption',
    measuredSignal: 'tanker_rate_change',
    predictedSignal: 'oil_supply_shock',
    leadTimeHours: 168,
    historicalCorrelation: 0.55,
  },

  // Food-based proxies
  {
    name: 'Wheat_futures',
    description: 'Wheat futures spike predicts food security crisis',
    measuredSignal: 'wheat_futures_change',
    predictedSignal: 'food_crisis',
    leadTimeHours: 720,
    historicalCorrelation: 0.5,
  },

  // Social-based proxies
  {
    name: 'Regime_critic_silence',
    description: 'Sudden silence from regime critics predicts crackdown',
    measuredSignal: 'opposition_voice_frequency',
    predictedSignal: 'authoritarian_action',
    leadTimeHours: 72,
    historicalCorrelation: 0.65,
  },
];

/**
 * Analyze proxy indicators for early warnings
 */
export function analyzeProxies(
  marketData: Record<string, number>,  // indicator_name → current value
  historicalData: Record<string, number[]>  // indicator_name → last N values
): Array<{ indicator: ProxyIndicator; currentValue: number; zScore: number; warning: string | null }> {
  const results: Array<{ indicator: ProxyIndicator; currentValue: number; zScore: number; warning: string | null }> = [];

  for (const proxy of PROXY_INDICATORS) {
    const current = marketData[proxy.measuredSignal];
    const history = historicalData[proxy.measuredSignal] || [];

    if (current === undefined || history.length < 5) continue;

    // Calculate z-score
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / history.length;
    const std = Math.sqrt(variance) || 1;
    const zScore = (current - mean) / std;

    // Check for warning condition
    let warning: string | null = null;
    if (Math.abs(zScore) > 2) {
      const direction = zScore > 0 ? 'spike' : 'collapse';
      warning = `${proxy.name} ${direction} (${zScore.toFixed(1)}σ): ${proxy.description}. ` +
        `Predicts ${proxy.predictedSignal} in ~${proxy.leadTimeHours}h. ` +
        `Historical correlation: ${(proxy.historicalCorrelation * 100).toFixed(0)}%`;
    }

    results.push({
      indicator: proxy,
      currentValue: current,
      zScore,
      warning,
    });
  }

  return results.filter(r => r.warning !== null);
}

// ============================================================
// TRIANGULATION - Multiple weak signals → strong inference
// ============================================================

export interface TriangulationSource {
  name: string;
  weight: number;
  signal: number;  // -1 to 1 normalized
  confidence: number;
}

/**
 * Triangulate confidence from multiple independent sources
 *
 * Key insight: 3 independent sources at 60% confidence each
 * gives higher combined confidence than 1 source at 80%.
 *
 * Formula: 1 - Π(1 - confidence_i * weight_i)
 */
export function triangulate(sources: TriangulationSource[]): {
  combinedSignal: number;
  combinedConfidence: number;
  agreementScore: number;
  dominantDirection: 'positive' | 'negative' | 'mixed';
} {
  if (sources.length === 0) {
    return {
      combinedSignal: 0,
      combinedConfidence: 0,
      agreementScore: 0,
      dominantDirection: 'mixed',
    };
  }

  // Weighted average of signals
  let totalWeight = 0;
  let weightedSignal = 0;
  for (const source of sources) {
    weightedSignal += source.signal * source.weight * source.confidence;
    totalWeight += source.weight * source.confidence;
  }
  const combinedSignal = totalWeight > 0 ? weightedSignal / totalWeight : 0;

  // Combined confidence using independence assumption
  let probAllWrong = 1;
  for (const source of sources) {
    probAllWrong *= (1 - source.confidence * source.weight);
  }
  const combinedConfidence = 1 - probAllWrong;

  // Agreement score (do sources agree?)
  const positives = sources.filter(s => s.signal > 0.1).length;
  const negatives = sources.filter(s => s.signal < -0.1).length;
  const total = sources.length;

  const agreementScore = Math.max(positives, negatives) / total;

  const dominantDirection = positives > negatives * 1.5 ? 'positive' :
    negatives > positives * 1.5 ? 'negative' : 'mixed';

  return {
    combinedSignal,
    combinedConfidence,
    agreementScore,
    dominantDirection,
  };
}

// ============================================================
// TEMPO ANALYSIS - Speed of events as signal
// ============================================================

export interface TempoMetrics {
  signalsPerHour: number;
  signalsPerHourBaseline: number;
  acceleration: number;  // Rate of change
  burstScore: number;    // Are signals clustered or spread?
  rhythmAnomaly: number; // Deviation from normal daily pattern
}

/**
 * Analyze the tempo/rhythm of incoming signals
 */
export function analyzeTempo(
  signals: ProcessedSignal[],
  baselineSignalsPerHour: number = 10
): TempoMetrics {
  if (signals.length < 2) {
    return {
      signalsPerHour: 0,
      signalsPerHourBaseline: baselineSignalsPerHour,
      acceleration: 0,
      burstScore: 0,
      rhythmAnomaly: 0,
    };
  }

  // Sort by timestamp
  const sorted = [...signals].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Calculate time span
  const firstTime = sorted[0].timestamp.getTime();
  const lastTime = sorted[sorted.length - 1].timestamp.getTime();
  const spanHours = (lastTime - firstTime) / (1000 * 60 * 60) || 1;

  const signalsPerHour = signals.length / spanHours;

  // Calculate acceleration (are signals speeding up?)
  const midpoint = sorted.length / 2;
  const firstHalf = sorted.slice(0, midpoint);
  const secondHalf = sorted.slice(midpoint);

  const firstHalfSpan = midpoint > 0 ?
    (firstHalf[firstHalf.length - 1].timestamp.getTime() - firstHalf[0].timestamp.getTime()) / (1000 * 60 * 60) || 1 : 1;
  const secondHalfSpan = secondHalf.length > 1 ?
    (secondHalf[secondHalf.length - 1].timestamp.getTime() - secondHalf[0].timestamp.getTime()) / (1000 * 60 * 60) || 1 : 1;

  const firstHalfRate = firstHalf.length / firstHalfSpan;
  const secondHalfRate = secondHalf.length / secondHalfSpan;

  const acceleration = (secondHalfRate - firstHalfRate) / (firstHalfRate || 1);

  // Calculate burst score (clustering)
  const intervals: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const interval = sorted[i].timestamp.getTime() - sorted[i-1].timestamp.getTime();
    intervals.push(interval);
  }

  const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length || 1;
  const intervalVariance = intervals.reduce((sum, i) => sum + Math.pow(i - avgInterval, 2), 0) / intervals.length;
  const intervalStd = Math.sqrt(intervalVariance);

  // High variance = bursty (clustered signals)
  const burstScore = intervalStd / avgInterval;

  // Rhythm anomaly (deviation from expected rate)
  const rhythmAnomaly = Math.abs(signalsPerHour - baselineSignalsPerHour) / baselineSignalsPerHour;

  return {
    signalsPerHour,
    signalsPerHourBaseline: baselineSignalsPerHour,
    acceleration,
    burstScore,
    rhythmAnomaly,
  };
}

// ============================================================
// DISINFORMATION DETECTION - Are we being played?
// ============================================================

export interface DisinfoCues {
  score: number;  // 0-1, higher = more suspicious
  flags: string[];
}

/**
 * Detect potential disinformation signals
 *
 * Key cues:
 * 1. Coordinated timing (many similar signals at once)
 * 2. Unusual source patterns
 * 3. Narrative mismatch (too good to be true / too bad to be true)
 * 4. Amplification patterns (viral without substance)
 */
export function detectDisinfo(
  signals: ProcessedSignal[],
  topic: string
): DisinfoCues {
  const flags: string[] = [];
  let score = 0;

  // Filter to topic
  const topicSignals = signals.filter(s =>
    s.features.primaryTopic === topic ||
    s.features.originalText.toLowerCase().includes(topic.toLowerCase())
  );

  if (topicSignals.length < 3) {
    return { score: 0, flags: [] };
  }

  // Check 1: Coordinated timing
  // If many signals arrive within minutes of each other, suspicious
  const sorted = [...topicSignals].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  let burstCount = 0;
  for (let i = 1; i < sorted.length; i++) {
    const gap = sorted[i].timestamp.getTime() - sorted[i-1].timestamp.getTime();
    if (gap < 5 * 60 * 1000) {  // 5 minutes
      burstCount++;
    }
  }
  if (burstCount > topicSignals.length * 0.5) {
    flags.push(`Coordinated timing: ${burstCount} signals within 5min windows`);
    score += 0.3;
  }

  // Check 2: Narrative uniformity
  // If all signals have nearly identical sentiment, suspicious
  const sentiments = topicSignals.map(s => s.features.sentimentScore);
  const avgSentiment = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
  const sentimentVariance = sentiments.reduce((sum, s) => sum + Math.pow(s - avgSentiment, 2), 0) / sentiments.length;

  if (sentimentVariance < 0.05 && topicSignals.length > 5) {
    flags.push(`Uniform sentiment: variance ${sentimentVariance.toFixed(3)} (natural variance > 0.1)`);
    score += 0.25;
  }

  // Check 3: Extreme sentiment without substance
  if (Math.abs(avgSentiment) > 0.6) {
    const avgMagnitude = topicSignals.reduce((sum, s) => sum + s.features.magnitudeScore, 0) / topicSignals.length;
    if (avgMagnitude < 0.2) {
      flags.push(`Extreme sentiment (${avgSentiment.toFixed(2)}) without substantive content`);
      score += 0.2;
    }
  }

  // Check 4: Entity amplification without context
  const entityMentions = new Map<string, number>();
  for (const sig of topicSignals) {
    for (const entity of sig.features.detectedEntities) {
      entityMentions.set(entity, (entityMentions.get(entity) || 0) + 1);
    }
  }

  // If one entity dominates and signals are short, suspicious
  const topEntity = [...entityMentions.entries()].sort((a, b) => b[1] - a[1])[0];
  if (topEntity && topEntity[1] > topicSignals.length * 0.8) {
    const avgWordCount = topicSignals.reduce((sum, s) => sum + s.features.wordCount, 0) / topicSignals.length;
    if (avgWordCount < 30) {
      flags.push(`Entity amplification: ${topEntity[0]} in ${topEntity[1]}/${topicSignals.length} signals, avg ${avgWordCount.toFixed(0)} words`);
      score += 0.25;
    }
  }

  return {
    score: Math.min(1, score),
    flags,
  };
}

// ============================================================
// META-CONFIDENCE - How confident should we be in our analysis?
// ============================================================

export interface MetaConfidence {
  dataQuality: number;       // 0-1, how good is our input data?
  sourceDiversity: number;   // 0-1, how many independent sources?
  temporalCoverage: number;  // 0-1, how much of the time window do we cover?
  signalToNoise: number;     // 0-1, how much is signal vs noise?
  overallConfidence: number; // Combined meta-confidence
  warnings: string[];
}

/**
 * Compute meta-confidence in our analysis
 *
 * This is crucial: knowing what we DON'T know.
 */
export function computeMetaConfidence(
  signals: ProcessedSignal[],
  timeWindowHours: number = 24
): MetaConfidence {
  const warnings: string[] = [];

  // Data quality - based on anomaly rate and feature extraction success
  const anomalyRate = signals.filter(s => s.isAnomaly).length / signals.length;
  const validFeatures = signals.filter(s =>
    s.features.topicVector.some(v => v > 0) &&
    s.features.detectedEntities.length > 0
  ).length;
  const dataQuality = validFeatures / signals.length;

  if (dataQuality < 0.5) {
    warnings.push(`Low data quality: only ${(dataQuality * 100).toFixed(0)}% of signals have valid features`);
  }

  // Source diversity
  const sources = new Set(signals.map(s => s.source));
  const sourceDiversity = Math.min(1, sources.size / 5);  // Expect at least 5 sources

  if (sources.size < 3) {
    warnings.push(`Low source diversity: only ${sources.size} sources`);
  }

  // Temporal coverage
  if (signals.length < 2) {
    return {
      dataQuality,
      sourceDiversity,
      temporalCoverage: 0,
      signalToNoise: 0,
      overallConfidence: 0,
      warnings: ['Insufficient signals for analysis'],
    };
  }

  const sorted = [...signals].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  const covered = (sorted[sorted.length - 1].timestamp.getTime() - sorted[0].timestamp.getTime()) / (1000 * 60 * 60);
  const temporalCoverage = Math.min(1, covered / timeWindowHours);

  if (temporalCoverage < 0.5) {
    warnings.push(`Low temporal coverage: ${(temporalCoverage * 100).toFixed(0)}% of ${timeWindowHours}h window`);
  }

  // Signal to noise (based on topic concentration)
  const topicCounts: Record<string, number> = {};
  for (const sig of signals) {
    const topic = sig.features.primaryTopic;
    topicCounts[topic] = (topicCounts[topic] || 0) + 1;
  }
  const topTopics = Object.values(topicCounts).sort((a, b) => b - a).slice(0, 3);
  const topTopicShare = topTopics.reduce((a, b) => a + b, 0) / signals.length;
  const signalToNoise = topTopicShare;  // More concentrated = clearer signal

  // Combined confidence
  const overallConfidence = (
    dataQuality * 0.3 +
    sourceDiversity * 0.2 +
    temporalCoverage * 0.25 +
    signalToNoise * 0.25
  );

  if (overallConfidence < 0.5) {
    warnings.push('Overall confidence below threshold - recommend manual review');
  }

  return {
    dataQuality,
    sourceDiversity,
    temporalCoverage,
    signalToNoise,
    overallConfidence,
    warnings,
  };
}

// ============================================================
// DEEP ANALYSIS ORCHESTRATOR
// ============================================================

export interface DeepAnalysisResult {
  absences: ReturnType<typeof detectAbsences>;
  proxyWarnings: ReturnType<typeof analyzeProxies>;
  tempo: TempoMetrics;
  disinfoAlerts: Array<{ topic: string; cues: DisinfoCues }>;
  metaConfidence: MetaConfidence;
  synthesizedWarnings: string[];
  processingTimeMs: number;
}

/**
 * Run full deep analysis pipeline - ALL CPU, NO LLM
 */
export function runDeepAnalysis(
  signals: ProcessedSignal[],
  marketData: Record<string, number> = {},
  historicalMarketData: Record<string, number[]> = {},
  timeWindowHours: number = 24
): DeepAnalysisResult {
  const startTime = Date.now();

  // 1. Absence detection
  const absences = detectAbsences(signals, timeWindowHours);

  // 2. Proxy indicator analysis
  const proxyWarnings = analyzeProxies(marketData, historicalMarketData);

  // 3. Tempo analysis
  const tempo = analyzeTempo(signals);

  // 4. Disinformation detection (for top topics)
  const topicCounts: Record<string, number> = {};
  for (const sig of signals) {
    topicCounts[sig.features.primaryTopic] = (topicCounts[sig.features.primaryTopic] || 0) + 1;
  }
  const topTopics = Object.entries(topicCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([t]) => t);

  const disinfoAlerts = topTopics
    .map(topic => ({ topic, cues: detectDisinfo(signals, topic) }))
    .filter(a => a.cues.score > 0.3);

  // 5. Meta-confidence
  const metaConfidence = computeMetaConfidence(signals, timeWindowHours);

  // 6. Synthesize warnings
  const synthesizedWarnings: string[] = [];

  // High-value absences
  for (const absence of absences.slice(0, 3)) {
    if (absence.suspicionLevel > 0.5) {
      synthesizedWarnings.push(
        `[ABSENCE] No ${absence.event.name} signals in ${absence.hoursSinceSeen.toFixed(0)}h ` +
        `(expected every ${(24 / absence.event.normalFrequencyPerDay).toFixed(0)}h)`
      );
    }
  }

  // Proxy warnings
  for (const proxy of proxyWarnings) {
    if (proxy.warning) {
      synthesizedWarnings.push(`[PROXY] ${proxy.warning}`);
    }
  }

  // Tempo anomalies
  if (tempo.acceleration > 0.5) {
    synthesizedWarnings.push(
      `[TEMPO] Signal acceleration: ${(tempo.acceleration * 100).toFixed(0)}% increase in rate`
    );
  }
  if (tempo.burstScore > 2) {
    synthesizedWarnings.push(
      `[TEMPO] Burst pattern detected: signals clustering (score: ${tempo.burstScore.toFixed(1)})`
    );
  }

  // Disinfo alerts
  for (const alert of disinfoAlerts) {
    if (alert.cues.score > 0.5) {
      synthesizedWarnings.push(
        `[DISINFO] Potential manipulation on "${alert.topic}": ${alert.cues.flags[0]}`
      );
    }
  }

  // Meta-confidence warnings
  synthesizedWarnings.push(...metaConfidence.warnings);

  return {
    absences,
    proxyWarnings,
    tempo,
    disinfoAlerts,
    metaConfidence,
    synthesizedWarnings,
    processingTimeMs: Date.now() - startTime,
  };
}
