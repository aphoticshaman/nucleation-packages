/**
 * LogicalAgent - Pure CPU inference engine
 *
 * This is NOT an LLM. It's a deterministic logic engine that:
 * 1. Applies formal rules to data
 * 2. Performs syllogistic reasoning
 * 3. Detects causal chains
 * 4. Identifies cascade risks
 *
 * All inference is CPU-bound, multithreadable, and predictable.
 *
 * Philosophy: If you can write the rule in code, you don't need an LLM.
 * LLMs are for ambiguity. Logic handles certainty.
 */

import type { ProcessedSignal, SignalFeatures, BaselineStats } from './SignalProcessor';

// ============================================================
// TYPES
// ============================================================

export interface Fact {
  subject: string;           // e.g., "RUS", "oil_price", "EUR_USD"
  predicate: string;         // e.g., "has_high_risk", "increased_by", "allied_with"
  object?: string | number;  // e.g., "0.8", "CHN", "15%"
  confidence: number;        // 0-1
  source: string;            // Where this fact came from
  timestamp: Date;
}

export interface Rule {
  id: string;
  name: string;
  description: string;
  // Conditions as function for flexibility
  condition: (facts: Fact[], context: InferenceContext) => boolean;
  // What to infer if condition matches
  inference: (facts: Fact[], context: InferenceContext) => Inference[];
  // Priority (higher = evaluate first)
  priority: number;
  // Rule category
  category: 'cascade' | 'correlation' | 'temporal' | 'causal' | 'geopolitical';
}

export interface Inference {
  type: 'risk_elevation' | 'cascade_warning' | 'correlation' | 'trend' | 'action_needed';
  subject: string;
  conclusion: string;
  confidence: number;
  reasoning: string[];  // Chain of logic that led here
  affectedEntities: string[];
  severity: 'low' | 'moderate' | 'high' | 'critical';
}

export interface InferenceContext {
  currentDate: Date;
  baseline: BaselineStats;
  recentSignals: ProcessedSignal[];
  nationRisks: Map<string, number>;      // ISO code → risk score
  alliances: Map<string, string[]>;       // ISO code → allied nations
  tradeLinks: Map<string, string[]>;      // ISO code → major trade partners
  energyDependencies: Map<string, string[]>; // ISO code → energy sources
}

// ============================================================
// KNOWLEDGE BASE - Static facts about world structure
// ============================================================

export const ALLIANCE_GRAPH: Record<string, string[]> = {
  // NATO
  USA: ['GBR', 'FRA', 'DEU', 'CAN', 'ITA', 'POL', 'TUR', 'NLD', 'BEL', 'ESP'],
  GBR: ['USA', 'FRA', 'DEU', 'CAN', 'AUS', 'NZL', 'JPN'],
  FRA: ['USA', 'GBR', 'DEU', 'ITA', 'ESP', 'BEL', 'NLD'],
  DEU: ['USA', 'FRA', 'GBR', 'ITA', 'POL', 'NLD', 'BEL'],

  // Eastern
  RUS: ['BLR', 'KAZ', 'ARM', 'SYR', 'IRN'],
  CHN: ['RUS', 'PRK', 'PAK', 'IRN', 'MMR'],

  // Middle East
  ISR: ['USA', 'GBR', 'DEU', 'EGY', 'JOR', 'SAU', 'ARE'],
  SAU: ['USA', 'ARE', 'EGY', 'JOR', 'BHR', 'KWT'],
  IRN: ['RUS', 'CHN', 'SYR', 'IRQ', 'LBN'],

  // Asia-Pacific
  JPN: ['USA', 'AUS', 'KOR', 'IND', 'PHL', 'TWN'],
  KOR: ['USA', 'JPN', 'AUS'],
  IND: ['USA', 'JPN', 'AUS', 'FRA', 'GBR'],
  AUS: ['USA', 'GBR', 'NZL', 'JPN', 'IND'],
};

export const TRADE_DEPENDENCIES: Record<string, string[]> = {
  DEU: ['CHN', 'USA', 'FRA', 'NLD', 'POL', 'ITA', 'AUT', 'CHE'],
  USA: ['CHN', 'CAN', 'MEX', 'JPN', 'DEU', 'KOR', 'GBR', 'IND'],
  CHN: ['USA', 'JPN', 'KOR', 'DEU', 'VNM', 'TWN', 'AUS', 'BRA'],
  JPN: ['CHN', 'USA', 'KOR', 'TWN', 'THA', 'DEU', 'AUS', 'VNM'],
  GBR: ['USA', 'DEU', 'NLD', 'FRA', 'CHN', 'IRL', 'BEL', 'ITA'],
};

export const ENERGY_DEPENDENCIES: Record<string, string[]> = {
  // Who depends on whom for energy
  DEU: ['RUS', 'NOR', 'USA', 'NLD', 'QAT'],
  JPN: ['SAU', 'ARE', 'QAT', 'AUS', 'USA', 'RUS'],
  KOR: ['SAU', 'IRQ', 'KWT', 'ARE', 'USA', 'QAT'],
  IND: ['IRQ', 'SAU', 'ARE', 'USA', 'NGA', 'KWT'],
  CHN: ['SAU', 'RUS', 'IRQ', 'ARE', 'KWT', 'BRA', 'ANG'],
  POL: ['RUS', 'SAU', 'USA', 'NOR', 'KAZ'],
  TUR: ['RUS', 'IRN', 'IRQ', 'AZE', 'USA'],
};

export const CONFLICT_PAIRS: Array<[string, string]> = [
  ['RUS', 'UKR'],
  ['ISR', 'PSE'],
  ['ISR', 'IRN'],
  ['ISR', 'LBN'],
  ['SAU', 'IRN'],
  ['CHN', 'TWN'],
  ['PRK', 'KOR'],
  ['IND', 'PAK'],
  ['CHN', 'IND'],
  ['ARM', 'AZE'],
  ['ETH', 'ERI'],
  ['SDN', 'SSD'],
];

// ============================================================
// RULE DEFINITIONS - Pure logic, no ML
// ============================================================

const RULES: Rule[] = [
  // Rule 1: Alliance Cascade
  {
    id: 'alliance-cascade',
    name: 'Alliance Cascade Risk',
    description: 'If ally has high risk, monitor for cascade',
    priority: 100,
    category: 'cascade',
    condition: (facts, ctx) => {
      // Find any nation with high risk
      for (const [nation, risk] of ctx.nationRisks) {
        if (risk > 0.7) {
          const allies = ctx.alliances.get(nation) || [];
          if (allies.length > 0) return true;
        }
      }
      return false;
    },
    inference: (facts, ctx) => {
      const inferences: Inference[] = [];

      for (const [nation, risk] of ctx.nationRisks) {
        if (risk > 0.7) {
          const allies = ctx.alliances.get(nation) || [];
          for (const ally of allies) {
            const allyRisk = ctx.nationRisks.get(ally) || 0;
            if (allyRisk < 0.5) {  // Ally not yet elevated
              inferences.push({
                type: 'cascade_warning',
                subject: ally,
                conclusion: `${ally} may face cascade pressure from ${nation} instability`,
                confidence: risk * 0.6,  // Damped by alliance distance
                reasoning: [
                  `${nation} has risk score ${(risk * 100).toFixed(0)}%`,
                  `${nation} and ${ally} are allied`,
                  `Alliance stress may propagate to ${ally}`,
                ],
                affectedEntities: [nation, ally],
                severity: risk > 0.85 ? 'high' : 'moderate',
              });
            }
          }
        }
      }

      return inferences;
    },
  },

  // Rule 2: Energy Dependency Risk
  {
    id: 'energy-dependency',
    name: 'Energy Supply Risk',
    description: 'If energy supplier has risk, dependent nations face energy security risk',
    priority: 90,
    category: 'causal',
    condition: (facts, ctx) => {
      // Check if any major energy supplier has high risk
      const energySuppliers = new Set(['RUS', 'SAU', 'IRN', 'IRQ', 'ARE', 'QAT', 'USA', 'NOR']);
      for (const supplier of energySuppliers) {
        const risk = ctx.nationRisks.get(supplier) || 0;
        if (risk > 0.5) return true;
      }
      return false;
    },
    inference: (facts, ctx) => {
      const inferences: Inference[] = [];
      const energySuppliers = new Set(['RUS', 'SAU', 'IRN', 'IRQ', 'ARE', 'QAT', 'USA', 'NOR']);

      for (const supplier of energySuppliers) {
        const supplierRisk = ctx.nationRisks.get(supplier) || 0;
        if (supplierRisk > 0.5) {
          // Find who depends on this supplier
          for (const [dependent, sources] of ctx.energyDependencies) {
            if (sources.includes(supplier)) {
              inferences.push({
                type: 'risk_elevation',
                subject: dependent,
                conclusion: `${dependent} faces energy security risk due to ${supplier} instability`,
                confidence: supplierRisk * 0.7,
                reasoning: [
                  `${supplier} is a key energy source for ${dependent}`,
                  `${supplier} has elevated risk score: ${(supplierRisk * 100).toFixed(0)}%`,
                  `Energy supply disruption possible`,
                ],
                affectedEntities: [supplier, dependent],
                severity: supplierRisk > 0.7 ? 'high' : 'moderate',
              });
            }
          }
        }
      }

      return inferences;
    },
  },

  // Rule 3: Conflict Escalation Detection
  {
    id: 'conflict-escalation',
    name: 'Conflict Escalation Warning',
    description: 'Detect when known conflict pairs show elevated signals',
    priority: 95,
    category: 'geopolitical',
    condition: (facts, ctx) => {
      for (const [a, b] of CONFLICT_PAIRS) {
        const riskA = ctx.nationRisks.get(a) || 0;
        const riskB = ctx.nationRisks.get(b) || 0;
        if (riskA > 0.5 && riskB > 0.5) return true;
      }
      return false;
    },
    inference: (facts, ctx) => {
      const inferences: Inference[] = [];

      for (const [a, b] of CONFLICT_PAIRS) {
        const riskA = ctx.nationRisks.get(a) || 0;
        const riskB = ctx.nationRisks.get(b) || 0;

        if (riskA > 0.5 && riskB > 0.5) {
          const combinedRisk = (riskA + riskB) / 2;
          inferences.push({
            type: 'cascade_warning',
            subject: `${a}-${b}`,
            conclusion: `Elevated tension in ${a}-${b} conflict axis`,
            confidence: combinedRisk,
            reasoning: [
              `${a} and ${b} are known conflict pair`,
              `${a} risk: ${(riskA * 100).toFixed(0)}%`,
              `${b} risk: ${(riskB * 100).toFixed(0)}%`,
              `Dual elevation suggests active escalation`,
            ],
            affectedEntities: [a, b],
            severity: combinedRisk > 0.7 ? 'critical' : 'high',
          });
        }
      }

      return inferences;
    },
  },

  // Rule 4: Trade Cascade
  {
    id: 'trade-cascade',
    name: 'Trade Disruption Cascade',
    description: 'Major trade partner instability affects economic risk',
    priority: 80,
    category: 'cascade',
    condition: (facts, ctx) => {
      for (const [nation, risk] of ctx.nationRisks) {
        if (risk > 0.6) {
          const partners = ctx.tradeLinks.get(nation) || [];
          if (partners.length >= 3) return true;  // Major trade hub
        }
      }
      return false;
    },
    inference: (facts, ctx) => {
      const inferences: Inference[] = [];

      for (const [nation, risk] of ctx.nationRisks) {
        if (risk > 0.6) {
          const partners = ctx.tradeLinks.get(nation) || [];
          if (partners.length >= 3) {
            for (const partner of partners.slice(0, 3)) {  // Top 3 partners
              inferences.push({
                type: 'risk_elevation',
                subject: partner,
                conclusion: `${partner} faces trade disruption risk from ${nation} instability`,
                confidence: risk * 0.5,
                reasoning: [
                  `${nation} is major trade partner of ${partner}`,
                  `${nation} instability may disrupt bilateral trade`,
                  `Supply chain and export exposure to be monitored`,
                ],
                affectedEntities: [nation, partner],
                severity: 'moderate',
              });
            }
          }
        }
      }

      return inferences;
    },
  },

  // Rule 5: Sentiment Trend Detection
  {
    id: 'sentiment-trend',
    name: 'Negative Sentiment Trend',
    description: 'Detect sustained negative sentiment shift',
    priority: 70,
    category: 'temporal',
    condition: (facts, ctx) => {
      // Look for consistent negative sentiment in recent signals
      if (ctx.recentSignals.length < 5) return false;

      const recentSentiments = ctx.recentSignals
        .slice(-10)
        .map(s => s.features.sentimentScore);

      const avgSentiment = recentSentiments.reduce((a, b) => a + b, 0) / recentSentiments.length;
      return avgSentiment < -0.2;  // Sustained negative
    },
    inference: (facts, ctx) => {
      const recentSentiments = ctx.recentSignals
        .slice(-10)
        .map(s => s.features.sentimentScore);
      const avgSentiment = recentSentiments.reduce((a, b) => a + b, 0) / recentSentiments.length;

      // Find which topics are driving negativity
      const negativeSignals = ctx.recentSignals.filter(s => s.features.sentimentScore < -0.2);
      const topicCounts: Record<string, number> = {};
      for (const sig of negativeSignals) {
        const topic = sig.features.primaryTopic;
        topicCounts[topic] = (topicCounts[topic] || 0) + 1;
      }

      const topTopic = Object.entries(topicCounts)
        .sort((a, b) => b[1] - a[1])[0];

      return [{
        type: 'trend',
        subject: 'global_sentiment',
        conclusion: `Sustained negative sentiment trend detected (avg: ${avgSentiment.toFixed(2)})`,
        confidence: Math.min(1, Math.abs(avgSentiment)),
        reasoning: [
          `Average sentiment over last ${recentSentiments.length} signals: ${avgSentiment.toFixed(2)}`,
          `Primary negative topic: ${topTopic?.[0] || 'mixed'}`,
          `${negativeSignals.length} of ${ctx.recentSignals.length} signals negative`,
        ],
        affectedEntities: [topTopic?.[0] || 'general'],
        severity: avgSentiment < -0.4 ? 'high' : 'moderate',
      }];
    },
  },

  // Rule 6: Multi-Entity Concentration
  {
    id: 'entity-concentration',
    name: 'Entity Mention Spike',
    description: 'Detect when specific entity is mentioned abnormally often',
    priority: 75,
    category: 'correlation',
    condition: (facts, ctx) => {
      if (ctx.recentSignals.length < 5) return false;

      // Count entity mentions
      const entityCounts: Record<string, number> = {};
      for (const sig of ctx.recentSignals) {
        for (const entity of sig.features.detectedEntities) {
          entityCounts[entity] = (entityCounts[entity] || 0) + 1;
        }
      }

      // Check if any entity appears in > 50% of signals
      const threshold = ctx.recentSignals.length * 0.5;
      return Object.values(entityCounts).some(count => count > threshold);
    },
    inference: (facts, ctx) => {
      const entityCounts: Record<string, number> = {};
      for (const sig of ctx.recentSignals) {
        for (const entity of sig.features.detectedEntities) {
          entityCounts[entity] = (entityCounts[entity] || 0) + 1;
        }
      }

      const threshold = ctx.recentSignals.length * 0.5;
      const concentratedEntities = Object.entries(entityCounts)
        .filter(([_, count]) => count > threshold)
        .sort((a, b) => b[1] - a[1]);

      return concentratedEntities.map(([entity, count]) => ({
        type: 'correlation' as const,
        subject: entity,
        conclusion: `${entity} mentioned in ${((count / ctx.recentSignals.length) * 100).toFixed(0)}% of recent signals`,
        confidence: count / ctx.recentSignals.length,
        reasoning: [
          `${entity} appears in ${count} of ${ctx.recentSignals.length} signals`,
          `Above 50% threshold for significance`,
          `Warrants focused monitoring`,
        ],
        affectedEntities: [entity],
        severity: count / ctx.recentSignals.length > 0.7 ? 'high' : 'moderate',
      }));
    },
  },

  // Rule 7: Urgency Spike
  {
    id: 'urgency-spike',
    name: 'Breaking News Surge',
    description: 'Detect surge in urgent/breaking signals',
    priority: 85,
    category: 'temporal',
    condition: (facts, ctx) => {
      if (ctx.recentSignals.length < 3) return false;

      const urgentSignals = ctx.recentSignals.filter(s => s.features.urgencyScore > 0.5);
      return urgentSignals.length >= 3;
    },
    inference: (facts, ctx) => {
      const urgentSignals = ctx.recentSignals.filter(s => s.features.urgencyScore > 0.5);

      // Find common entities across urgent signals
      const entityCounts: Record<string, number> = {};
      for (const sig of urgentSignals) {
        for (const entity of sig.features.detectedEntities) {
          entityCounts[entity] = (entityCounts[entity] || 0) + 1;
        }
      }

      const topEntities = Object.entries(entityCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([e]) => e);

      return [{
        type: 'action_needed',
        subject: 'breaking_news',
        conclusion: `${urgentSignals.length} breaking/urgent signals detected`,
        confidence: Math.min(1, urgentSignals.length / 5),
        reasoning: [
          `${urgentSignals.length} signals with urgency score > 0.5`,
          `Key entities: ${topEntities.join(', ') || 'various'}`,
          `Recommend immediate review`,
        ],
        affectedEntities: topEntities,
        severity: urgentSignals.length >= 5 ? 'critical' : 'high',
      }];
    },
  },
];

// ============================================================
// INFERENCE ENGINE
// ============================================================

export interface InferenceResult {
  inferences: Inference[];
  factsUsed: number;
  rulesEvaluated: number;
  processingTimeMs: number;
}

/**
 * Run the logical inference engine
 *
 * This is pure CPU computation - no LLM calls.
 * Can process thousands of facts in milliseconds.
 */
export function runInference(
  signals: ProcessedSignal[],
  nationRisks: Record<string, number>,
  baseline: BaselineStats
): InferenceResult {
  const startTime = Date.now();

  // Build context
  const ctx: InferenceContext = {
    currentDate: new Date(),
    baseline,
    recentSignals: signals,
    nationRisks: new Map(Object.entries(nationRisks)),
    alliances: new Map(Object.entries(ALLIANCE_GRAPH)),
    tradeLinks: new Map(Object.entries(TRADE_DEPENDENCIES)),
    energyDependencies: new Map(Object.entries(ENERGY_DEPENDENCIES)),
  };

  // Extract facts from signals
  const facts: Fact[] = [];
  for (const signal of signals) {
    for (const entity of signal.features.detectedEntities) {
      facts.push({
        subject: entity,
        predicate: 'mentioned_in_signal',
        object: signal.id,
        confidence: 1,
        source: signal.source,
        timestamp: signal.timestamp,
      });
    }

    facts.push({
      subject: signal.features.primaryTopic,
      predicate: 'signal_topic',
      object: signal.features.sentimentScore,
      confidence: 1,
      source: signal.source,
      timestamp: signal.timestamp,
    });
  }

  // Add nation risks as facts
  for (const [nation, risk] of Object.entries(nationRisks)) {
    facts.push({
      subject: nation,
      predicate: 'has_risk_score',
      object: risk,
      confidence: 1,
      source: 'nation_data',
      timestamp: new Date(),
    });
  }

  // Sort rules by priority
  const sortedRules = [...RULES].sort((a, b) => b.priority - a.priority);

  // Evaluate rules
  const allInferences: Inference[] = [];
  let rulesEvaluated = 0;

  for (const rule of sortedRules) {
    rulesEvaluated++;
    try {
      if (rule.condition(facts, ctx)) {
        const newInferences = rule.inference(facts, ctx);
        allInferences.push(...newInferences);
      }
    } catch (e) {
      console.error(`Rule ${rule.id} failed:`, e);
    }
  }

  // Deduplicate inferences by subject
  const uniqueInferences = deduplicateInferences(allInferences);

  return {
    inferences: uniqueInferences,
    factsUsed: facts.length,
    rulesEvaluated,
    processingTimeMs: Date.now() - startTime,
  };
}

/**
 * Deduplicate inferences, keeping highest confidence
 */
function deduplicateInferences(inferences: Inference[]): Inference[] {
  const bySubject = new Map<string, Inference>();

  for (const inf of inferences) {
    const key = `${inf.type}:${inf.subject}`;
    const existing = bySubject.get(key);

    if (!existing || inf.confidence > existing.confidence) {
      bySubject.set(key, inf);
    }
  }

  return Array.from(bySubject.values())
    .sort((a, b) => b.confidence - a.confidence);
}

// ============================================================
// BRIEFING GENERATION FROM INFERENCES
// ============================================================

export interface LogicalBriefing {
  summary: string;
  keyFindings: string[];
  riskAlerts: Array<{
    entity: string;
    level: string;
    reason: string;
  }>;
  cascadeWarnings: string[];
  trends: string[];
  actionItems: string[];
  confidence: number;
  dataPoints: number;
}

/**
 * Generate briefing from inferences - NO LLM
 *
 * This turns structured inferences into readable prose
 * using templates and logic, not language models.
 */
export function generateLogicalBriefing(
  result: InferenceResult,
  nationRisks: Record<string, number>
): LogicalBriefing {
  const inferences = result.inferences;

  // Categorize inferences
  const riskElevations = inferences.filter(i => i.type === 'risk_elevation');
  const cascadeWarnings = inferences.filter(i => i.type === 'cascade_warning');
  const correlations = inferences.filter(i => i.type === 'correlation');
  const trends = inferences.filter(i => i.type === 'trend');
  const actions = inferences.filter(i => i.type === 'action_needed');

  // Find highest risk nations
  const sortedRisks = Object.entries(nationRisks)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  // Build summary
  const criticalCount = inferences.filter(i => i.severity === 'critical').length;
  const highCount = inferences.filter(i => i.severity === 'high').length;

  let summaryLevel = 'STABLE';
  if (criticalCount > 0) summaryLevel = 'CRITICAL';
  else if (highCount > 2) summaryLevel = 'ELEVATED';
  else if (highCount > 0) summaryLevel = 'MODERATE';

  const summary = `[LOGICAL ANALYSIS] Status: ${summaryLevel}. ` +
    `Processed ${result.factsUsed} facts through ${result.rulesEvaluated} rules. ` +
    `${inferences.length} inferences generated. ` +
    `${criticalCount} critical, ${highCount} high severity findings. ` +
    `Top risk: ${sortedRisks[0]?.[0] || 'None'} (${((sortedRisks[0]?.[1] || 0) * 100).toFixed(0)}%).`;

  // Key findings (top 5 by confidence)
  const keyFindings = inferences
    .slice(0, 5)
    .map(i => `[${i.severity.toUpperCase()}] ${i.conclusion}`);

  // Risk alerts
  const riskAlerts = riskElevations.map(i => ({
    entity: i.subject,
    level: i.severity,
    reason: i.reasoning[0] || i.conclusion,
  }));

  // Cascade warnings as strings
  const cascadeStrings = cascadeWarnings.map(i =>
    `${i.subject}: ${i.conclusion} (${(i.confidence * 100).toFixed(0)}% confidence)`
  );

  // Trends as strings
  const trendStrings = trends.map(i => i.conclusion);

  // Action items
  const actionItems = actions.map(i => i.conclusion);

  // Add inferred action items
  if (criticalCount > 0) {
    actionItems.push('IMMEDIATE: Review critical findings');
  }
  if (cascadeWarnings.length > 2) {
    actionItems.push('MONITOR: Multiple cascade pathways detected');
  }

  // Overall confidence (average of inference confidences)
  const avgConfidence = inferences.length > 0
    ? inferences.reduce((s, i) => s + i.confidence, 0) / inferences.length
    : 0.5;

  return {
    summary,
    keyFindings,
    riskAlerts,
    cascadeWarnings: cascadeStrings,
    trends: trendStrings,
    actionItems,
    confidence: avgConfidence,
    dataPoints: result.factsUsed,
  };
}

// ============================================================
// STAGGERED WAVE COORDINATOR
// ============================================================

export type WaveType =
  | 'gdelt_fetch'        // Minute 0-5
  | 'rss_fetch'          // Minute 5-10
  | 'economic_fetch'     // Minute 10-15
  | 'featurize'          // Minute 15-20
  | 'cluster'            // Minute 20-25
  | 'inference'          // Minute 25-30
  | 'anomaly_llm'        // Minute 30-40 (only if needed)
  | 'aggregate'          // Minute 40-50
  | 'publish';           // Minute 50-60

export interface WaveSchedule {
  wave: WaveType;
  minuteStart: number;
  minuteEnd: number;
  usesLLM: boolean;
  description: string;
}

export const WAVE_SCHEDULE: WaveSchedule[] = [
  { wave: 'gdelt_fetch', minuteStart: 0, minuteEnd: 5, usesLLM: false, description: 'Fetch GDELT signals' },
  { wave: 'rss_fetch', minuteStart: 5, minuteEnd: 10, usesLLM: false, description: 'Fetch RSS feeds' },
  { wave: 'economic_fetch', minuteStart: 10, minuteEnd: 15, usesLLM: false, description: 'Fetch economic indicators' },
  { wave: 'featurize', minuteStart: 15, minuteEnd: 20, usesLLM: false, description: 'CPU featurization' },
  { wave: 'cluster', minuteStart: 20, minuteEnd: 25, usesLLM: false, description: 'Rust WASM clustering' },
  { wave: 'inference', minuteStart: 25, minuteEnd: 30, usesLLM: false, description: 'Logical inference engine' },
  { wave: 'anomaly_llm', minuteStart: 30, minuteEnd: 40, usesLLM: true, description: 'LLM for anomalies only' },
  { wave: 'aggregate', minuteStart: 40, minuteEnd: 50, usesLLM: false, description: 'Aggregate all results' },
  { wave: 'publish', minuteStart: 50, minuteEnd: 60, usesLLM: false, description: 'Push to cache/website' },
];

/**
 * Get current wave based on minute of hour
 */
export function getCurrentWave(date: Date = new Date()): WaveSchedule | null {
  const minute = date.getMinutes();

  for (const schedule of WAVE_SCHEDULE) {
    if (minute >= schedule.minuteStart && minute < schedule.minuteEnd) {
      return schedule;
    }
  }

  return null;
}

/**
 * Check if we should run a specific wave
 */
export function shouldRunWave(wave: WaveType, date: Date = new Date()): boolean {
  const current = getCurrentWave(date);
  return current?.wave === wave;
}
