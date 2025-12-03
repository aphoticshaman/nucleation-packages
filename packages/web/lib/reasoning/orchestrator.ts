/**
 * LATTICE REASONING ORCHESTRATOR
 *
 * Routes queries to appropriate reasoning engines and combines results.
 * This is the "sleeping brain" that will wake up as data flows through it.
 *
 * Architecture:
 * - Deductive: Logic rules, constraints, implications
 * - Inductive: Pattern learning from examples
 * - Abductive: Best explanation selection
 * - Analogical: Historical case matching
 *
 * Security: Users NEVER interact with underlying LLMs directly.
 * All queries go through this orchestrator which sanitizes, reasons, then narrates.
 */

import { createClient } from '@supabase/supabase-js';
import {
  generateContextEmbedding,
  featureSimilarity,
  EMBEDDING_DIMENSIONS,
} from '@/lib/embeddings';

// Types for reasoning system
export interface ReasoningQuery {
  intent: 'analyze' | 'predict' | 'explain' | 'compare' | 'recommend';
  domain: string;
  context: Record<string, unknown>;
  userTier: 'consumer' | 'pro' | 'enterprise';
  userId: string;
  sessionId: string;
}

export interface ReasoningResult {
  conclusion: string;
  confidence: number;
  reasoning_trace: ReasoningStep[];
  causal_factors: CausalFactor[];
  analogies: HistoricalCase[];
  uncertainty: UncertaintyBounds;
  metadata: {
    engines_used: string[];
    compute_time_ms: number;
    data_freshness: string;
  };
}

export interface ReasoningStep {
  engine: 'deductive' | 'inductive' | 'abductive' | 'analogical';
  input: string;
  output: string;
  confidence: number;
  timestamp: number;
}

export interface CausalFactor {
  factor: string;
  contribution: number; // -1 to 1
  evidence: string[];
}

export interface HistoricalCase {
  id: string;
  description: string;
  similarity: number;
  outcome: string;
  lessons: string[];
}

export interface UncertaintyBounds {
  lower: number;
  upper: number;
  distribution: 'normal' | 'beta' | 'uniform';
  sources: string[];
}

// Deductive reasoning rules (expandable)
const DEDUCTIVE_RULES = [
  {
    id: 'high_risk_transition',
    condition: (ctx: Record<string, unknown>) =>
      (ctx.transition_risk as number) > 0.7 && (ctx.basin_strength as number) < 0.3,
    conclusion: 'Critical transition likely within 30 days',
    confidence: 0.85,
  },
  {
    id: 'stable_regime',
    condition: (ctx: Record<string, unknown>) =>
      (ctx.basin_strength as number) > 0.7 && (ctx.transition_risk as number) < 0.2,
    conclusion: 'Regime stability high, low change probability',
    confidence: 0.9,
  },
  {
    id: 'conflicting_signals',
    condition: (ctx: Record<string, unknown>) =>
      Math.abs((ctx.basin_strength as number) - (1 - (ctx.transition_risk as number))) > 0.3,
    conclusion: 'Conflicting indicators suggest uncertainty',
    confidence: 0.6,
  },
  {
    id: 'cascade_risk',
    condition: (ctx: Record<string, unknown>) => (ctx.connected_high_risk_nations as number) > 3,
    conclusion: 'Cascade/contagion risk elevated due to network effects',
    confidence: 0.75,
  },
];

// Abductive hypothesis templates
const ABDUCTIVE_HYPOTHESES = [
  {
    id: 'internal_pressure',
    indicators: ['domestic_unrest', 'economic_stress', 'leadership_instability'],
    explanation: 'Internal pressures driving instability',
  },
  {
    id: 'external_pressure',
    indicators: ['sanctions', 'military_threat', 'diplomatic_isolation'],
    explanation: 'External pressures forcing regime adaptation',
  },
  {
    id: 'structural_shift',
    indicators: ['demographic_change', 'technology_disruption', 'resource_depletion'],
    explanation: 'Long-term structural factors reshaping landscape',
  },
  {
    id: 'shock_event',
    indicators: ['sudden_leader_change', 'natural_disaster', 'market_crash'],
    explanation: 'Exogenous shock disrupting equilibrium',
  },
];

/**
 * Main reasoning orchestrator class
 */
export class ReasoningOrchestrator {
  private supabase;
  private learningEnabled: boolean;

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
    this.learningEnabled = true;
  }

  /**
   * Main entry point for reasoning
   */
  async reason(query: ReasoningQuery): Promise<ReasoningResult> {
    const startTime = Date.now();
    const trace: ReasoningStep[] = [];
    const enginesUsed: string[] = [];

    // 1. DEDUCTIVE: Apply logical rules
    const deductiveResults = this.applyDeductiveRules(query.context);
    if (deductiveResults.length > 0) {
      enginesUsed.push('deductive');
      trace.push(
        ...deductiveResults.map((r) => ({
          engine: 'deductive' as const,
          input: JSON.stringify(query.context),
          output: r.conclusion,
          confidence: r.confidence,
          timestamp: Date.now(),
        }))
      );
    }

    // 2. ABDUCTIVE: Find best explanation
    const abductiveResult = this.findBestExplanation(query.context);
    if (abductiveResult) {
      enginesUsed.push('abductive');
      trace.push({
        engine: 'abductive',
        input: JSON.stringify(query.context),
        output: abductiveResult.explanation,
        confidence: abductiveResult.confidence,
        timestamp: Date.now(),
      });
    }

    // 3. ANALOGICAL: Find historical matches
    const analogies = await this.findAnalogies(query);
    if (analogies.length > 0) {
      enginesUsed.push('analogical');
      trace.push({
        engine: 'analogical',
        input: query.domain,
        output: `Found ${analogies.length} historical analogies`,
        confidence: analogies[0]?.similarity || 0,
        timestamp: Date.now(),
      });
    }

    // 4. Combine results using Dempster-Shafer-style fusion
    const fusedConclusion = this.fuseConclusions(deductiveResults, abductiveResult, analogies);

    // 5. Compute causal factors
    const causalFactors = this.extractCausalFactors(query.context, trace);

    // 6. Compute uncertainty bounds
    const uncertainty = this.computeUncertainty(trace);

    const result: ReasoningResult = {
      conclusion: fusedConclusion.text,
      confidence: fusedConclusion.confidence,
      reasoning_trace: trace,
      causal_factors: causalFactors,
      analogies,
      uncertainty,
      metadata: {
        engines_used: enginesUsed,
        compute_time_ms: Date.now() - startTime,
        data_freshness: new Date().toISOString(),
      },
    };

    // 7. Log for learning (async, don't wait)
    if (this.learningEnabled) {
      void this.logForLearning(query, result);
    }

    return result;
  }

  /**
   * Apply deductive rules to context
   */
  private applyDeductiveRules(context: Record<string, unknown>) {
    const results: Array<{ conclusion: string; confidence: number; ruleId: string }> = [];

    for (const rule of DEDUCTIVE_RULES) {
      try {
        if (rule.condition(context)) {
          results.push({
            conclusion: rule.conclusion,
            confidence: rule.confidence,
            ruleId: rule.id,
          });
        }
      } catch {
        // Rule didn't apply (missing data)
        continue;
      }
    }

    return results;
  }

  /**
   * Find best explanation for observed signals
   */
  private findBestExplanation(context: Record<string, unknown>) {
    let bestHypothesis: { explanation: string; confidence: number } | null = null;
    let bestScore = 0;

    for (const hypothesis of ABDUCTIVE_HYPOTHESES) {
      // Count how many indicators are present/elevated
      let matchCount = 0;
      for (const indicator of hypothesis.indicators) {
        const value = context[indicator];
        if (value !== undefined && (value as number) > 0.5) {
          matchCount++;
        }
      }

      const score = matchCount / hypothesis.indicators.length;
      if (score > bestScore && score > 0.3) {
        bestScore = score;
        bestHypothesis = {
          explanation: hypothesis.explanation,
          confidence: score,
        };
      }
    }

    return bestHypothesis;
  }

  /**
   * Find historical analogies using vector similarity search
   */
  private async findAnalogies(query: ReasoningQuery): Promise<HistoricalCase[]> {
    try {
      // Extract numeric features from context for feature-based matching
      const numericFeatures: Record<string, number> = {};
      for (const [key, value] of Object.entries(query.context)) {
        if (typeof value === 'number') {
          numericFeatures[key] = value;
        }
      }

      // Try vector similarity search first (if embeddings available)
      const embeddingResult = await generateContextEmbedding(query.domain, query.context);

      if (embeddingResult) {
        // Use pgvector similarity search
        const { data: cases, error } = await this.supabase.rpc('find_similar_cases', {
          query_embedding: `[${embeddingResult.embedding.join(',')}]`,
          query_domain: query.domain,
          match_threshold: 0.5,
          match_count: 5,
        });

        if (!error && cases && cases.length > 0) {
          return cases.map(
            (c: {
              case_id: string;
              description: string;
              similarity: number;
              outcome: string;
              lessons: string[];
            }) => ({
              id: c.case_id,
              description: c.description,
              similarity: c.similarity,
              outcome: c.outcome,
              lessons: c.lessons || [],
            })
          );
        }
      }

      // Fallback: Query historical cases from database with feature matching
      const { data: allCases, error: fetchError } = await this.supabase
        .from('historical_cases')
        .select('case_id, description, outcome, lessons, features, domain')
        .eq('verified', true)
        .or(`domain.eq.${query.domain},domain.is.null`)
        .limit(20);

      if (fetchError || !allCases || allCases.length === 0) {
        // Final fallback: return hardcoded examples
        return this.getHardcodedAnalogies(query.domain);
      }

      // Compute feature similarity for each case
      const scoredCases = allCases
        .map((c) => {
          const caseFeatures =
            typeof c.features === 'object' ? (c.features as Record<string, number>) : {};
          const similarity = featureSimilarity(numericFeatures, caseFeatures);
          return {
            id: c.case_id,
            description: c.description,
            similarity,
            outcome: c.outcome,
            lessons: c.lessons || [],
          };
        })
        .filter((c) => c.similarity > 0.3)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5);

      return scoredCases.length > 0 ? scoredCases : this.getHardcodedAnalogies(query.domain);
    } catch (error) {
      console.error('Error finding analogies:', error);
      return this.getHardcodedAnalogies(query.domain);
    }
  }

  /**
   * Hardcoded fallback analogies when database unavailable
   */
  private getHardcodedAnalogies(domain: string): HistoricalCase[] {
    const analogyDatabase: Record<string, HistoricalCase[]> = {
      conflict: [
        {
          id: 'crimea_2014',
          description: '2014 Crimea annexation preceded by similar indicators',
          similarity: 0.72,
          outcome: 'Rapid territorial change, prolonged sanctions regime',
          lessons: [
            'Speed of action exceeded predictions',
            'Economic interdependence did not deter',
          ],
        },
      ],
      economic: [
        {
          id: 'asian_crisis_1997',
          description: '1997 Asian Financial Crisis showed similar contagion patterns',
          similarity: 0.68,
          outcome: 'Regional cascade, IMF intervention, 2-year recovery',
          lessons: [
            'Currency pegs vulnerable to speculation',
            'Regional interdependence amplified shock',
          ],
        },
      ],
      political: [
        {
          id: 'arab_spring_2011',
          description: '2011 Arab Spring showed similar domestic pressure indicators',
          similarity: 0.65,
          outcome: 'Multiple regime transitions, varied outcomes by country',
          lessons: ['Social media accelerated coordination', 'Military stance determined outcomes'],
        },
      ],
    };

    return analogyDatabase[domain] || [];
  }

  /**
   * Fuse conclusions from multiple engines
   */
  private fuseConclusions(
    deductive: Array<{ conclusion: string; confidence: number }>,
    abductive: { explanation: string; confidence: number } | null,
    analogies: HistoricalCase[]
  ) {
    const conclusions: Array<{ text: string; weight: number }> = [];

    // Weight by engine reliability
    for (const d of deductive) {
      conclusions.push({ text: d.conclusion, weight: d.confidence * 0.9 }); // High weight for logic
    }

    if (abductive) {
      conclusions.push({ text: abductive.explanation, weight: abductive.confidence * 0.7 });
    }

    if (analogies.length > 0) {
      conclusions.push({
        text: `Historical parallel: ${analogies[0].description}`,
        weight: analogies[0].similarity * 0.6,
      });
    }

    if (conclusions.length === 0) {
      return { text: 'Insufficient data for conclusion', confidence: 0.2 };
    }

    // Combine weighted conclusions
    const totalWeight = conclusions.reduce((sum, c) => sum + c.weight, 0);
    const avgConfidence = totalWeight / conclusions.length;

    // Select primary conclusion (highest weight)
    conclusions.sort((a, b) => b.weight - a.weight);

    return {
      text: conclusions[0].text,
      confidence: Math.min(avgConfidence, 0.95), // Cap at 95%
    };
  }

  /**
   * Extract causal factors from reasoning trace
   */
  private extractCausalFactors(
    context: Record<string, unknown>,
    _trace: ReasoningStep[]
  ): CausalFactor[] {
    const factors: CausalFactor[] = [];

    // Extract top contributing factors from context
    const numericFactors = Object.entries(context)
      .filter(([_, v]) => typeof v === 'number')
      .map(([k, v]) => ({ factor: k, value: v as number }))
      .sort((a, b) => Math.abs(b.value - 0.5) - Math.abs(a.value - 0.5))
      .slice(0, 5);

    for (const f of numericFactors) {
      factors.push({
        factor: f.factor,
        contribution: (f.value - 0.5) * 2, // Normalize to -1 to 1
        evidence: [`Current value: ${(f.value * 100).toFixed(0)}%`],
      });
    }

    return factors;
  }

  /**
   * Compute uncertainty bounds
   */
  private computeUncertainty(trace: ReasoningStep[]): UncertaintyBounds {
    if (trace.length === 0) {
      return {
        lower: 0,
        upper: 1,
        distribution: 'uniform',
        sources: ['No reasoning data'],
      };
    }

    const confidences = trace.map((t) => t.confidence);
    const mean = confidences.reduce((a, b) => a + b, 0) / confidences.length;
    const variance =
      confidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / confidences.length;
    const std = Math.sqrt(variance);

    return {
      lower: Math.max(0, mean - 2 * std),
      upper: Math.min(1, mean + 2 * std),
      distribution: 'normal',
      sources: trace.map((t) => t.engine),
    };
  }

  /**
   * Log interaction for future training
   * CRITICAL: This is anonymized and compliant
   */
  private async logForLearning(query: ReasoningQuery, result: ReasoningResult) {
    try {
      // Anonymize user data
      const anonymizedLog = {
        // Hash user ID - can't be reversed
        user_hash: await this.hashUserId(query.userId),
        session_hash: await this.hashUserId(query.sessionId),

        // Keep domain and intent for learning
        domain: query.domain,
        intent: query.intent,
        user_tier: query.userTier,

        // Context without PII
        context_keys: Object.keys(query.context),
        context_summary: this.summarizeContext(query.context),

        // Result metadata (not the actual content to users)
        result_confidence: result.confidence,
        engines_used: result.metadata.engines_used,
        compute_time_ms: result.metadata.compute_time_ms,

        // Reasoning trace for learning
        trace_summary: result.reasoning_trace.map((t) => ({
          engine: t.engine,
          confidence: t.confidence,
        })),

        timestamp: new Date().toISOString(),
      };

      // Store in learning_logs table
      await this.supabase.from('learning_logs').insert(anonymizedLog);
    } catch (error) {
      // Silent fail - don't break user experience for logging
      console.error('Learning log failed:', error);
    }
  }

  /**
   * One-way hash for anonymization
   */
  private async hashUserId(userId: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(userId + process.env.ANONYMIZATION_SALT);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('')
      .slice(0, 16);
  }

  /**
   * Summarize context without exposing raw values
   */
  private summarizeContext(context: Record<string, unknown>): Record<string, string> {
    const summary: Record<string, string> = {};

    for (const [key, value] of Object.entries(context)) {
      if (typeof value === 'number') {
        // Bucket into ranges
        if (value < 0.25) summary[key] = 'low';
        else if (value < 0.5) summary[key] = 'moderate';
        else if (value < 0.75) summary[key] = 'elevated';
        else summary[key] = 'high';
      } else if (typeof value === 'boolean') {
        summary[key] = value ? 'true' : 'false';
      } else if (typeof value === 'string') {
        // Don't log actual strings - could be PII
        summary[key] = `string_len_${(value as string).length}`;
      }
    }

    return summary;
  }
}

/**
 * Singleton instance
 */
let orchestratorInstance: ReasoningOrchestrator | null = null;

export function getReasoningOrchestrator(): ReasoningOrchestrator {
  if (!orchestratorInstance) {
    orchestratorInstance = new ReasoningOrchestrator(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );
  }
  return orchestratorInstance;
}
