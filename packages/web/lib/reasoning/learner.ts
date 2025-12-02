/**
 * LATTICE LEARNING COLLECTOR
 *
 * Captures all interactions for future model training.
 * Fully anonymized, compliant, and transparent.
 *
 * What we collect:
 * - Reasoning traces (how we arrived at conclusions)
 * - User feedback signals (implicit and explicit)
 * - API data that flows through (signals, metrics)
 * - LLM interactions (prompts and responses, anonymized)
 *
 * What we DON'T collect:
 * - PII (names, emails, IPs - all hashed)
 * - Raw user queries (summarized only)
 * - Anything that could identify individuals
 *
 * This data will be used to:
 * 1. Fine-tune domain-specific models
 * 2. Improve reasoning rules
 * 3. Build historical case library
 * 4. Train the "Great Attractor" model
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Learning event types
export type LearningEventType =
  | 'reasoning_trace'
  | 'llm_interaction'
  | 'user_feedback'
  | 'signal_observation'
  | 'prediction_outcome'
  | 'api_call';

export interface LearningEvent {
  type: LearningEventType;
  timestamp: string;
  session_hash: string;
  user_tier: string;
  domain: string;
  data: AnonymizedData;
  metadata: EventMetadata;
}

export interface AnonymizedData {
  // Numeric features (safe to store)
  numeric_features: Record<string, number>;
  // Categorical features (bucketed)
  categorical_features: Record<string, string>;
  // Reasoning output (our conclusions, not user data)
  reasoning_output?: {
    conclusion_type: string;
    confidence: number;
    engines_used: string[];
  };
  // LLM interaction (sanitized)
  llm_interaction?: {
    prompt_template: string; // Which template, not the actual prompt
    response_length: number;
    latency_ms: number;
    model: string;
  };
}

export interface EventMetadata {
  source: string;
  version: string;
  environment: 'development' | 'staging' | 'production';
}

/**
 * Learning Collector - the "always-on" data capture system
 */
export class LearningCollector {
  private supabase: SupabaseClient;
  private buffer: LearningEvent[] = [];
  private flushInterval: ReturnType<typeof setInterval> | null = null;
  private readonly BUFFER_SIZE = 100;
  private readonly FLUSH_INTERVAL_MS = 30000; // 30 seconds

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
    this.startAutoFlush();
  }

  /**
   * Log a reasoning trace for learning
   */
  async logReasoningTrace(
    sessionHash: string,
    userTier: string,
    domain: string,
    trace: {
      engines: string[];
      confidence: number;
      conclusionType: string;
      inputFeatures: Record<string, number>;
    }
  ) {
    const event: LearningEvent = {
      type: 'reasoning_trace',
      timestamp: new Date().toISOString(),
      session_hash: sessionHash,
      user_tier: userTier,
      domain,
      data: {
        numeric_features: trace.inputFeatures,
        categorical_features: {
          conclusion_type: trace.conclusionType,
        },
        reasoning_output: {
          conclusion_type: trace.conclusionType,
          confidence: trace.confidence,
          engines_used: trace.engines,
        },
      },
      metadata: this.getMetadata(),
    };

    this.addToBuffer(event);
  }

  /**
   * Log an LLM interaction (what we send/receive from Anthropic)
   * CRITICAL: We do NOT log the actual prompt content
   */
  async logLLMInteraction(
    sessionHash: string,
    userTier: string,
    domain: string,
    interaction: {
      promptTemplate: string; // e.g., "intel_briefing_v1"
      inputTokens: number;
      outputTokens: number;
      latencyMs: number;
      model: string;
      success: boolean;
    }
  ) {
    const event: LearningEvent = {
      type: 'llm_interaction',
      timestamp: new Date().toISOString(),
      session_hash: sessionHash,
      user_tier: userTier,
      domain,
      data: {
        numeric_features: {
          input_tokens: interaction.inputTokens,
          output_tokens: interaction.outputTokens,
          latency_ms: interaction.latencyMs,
          success: interaction.success ? 1 : 0,
        },
        categorical_features: {
          prompt_template: interaction.promptTemplate,
          model: interaction.model,
        },
        llm_interaction: {
          prompt_template: interaction.promptTemplate,
          response_length: interaction.outputTokens,
          latency_ms: interaction.latencyMs,
          model: interaction.model,
        },
      },
      metadata: this.getMetadata(),
    };

    this.addToBuffer(event);
  }

  /**
   * Log a signal observation (data from APIs)
   */
  async logSignalObservation(domain: string, signals: Record<string, number>, source: string) {
    const event: LearningEvent = {
      type: 'signal_observation',
      timestamp: new Date().toISOString(),
      session_hash: 'system', // Not user-specific
      user_tier: 'system',
      domain,
      data: {
        numeric_features: signals,
        categorical_features: {
          source,
        },
      },
      metadata: this.getMetadata(),
    };

    this.addToBuffer(event);
  }

  /**
   * Log a prediction outcome (ground truth when we get it)
   * This is GOLD for training - when our predictions can be verified
   */
  async logPredictionOutcome(
    domain: string,
    prediction: {
      predicted_state: string;
      predicted_confidence: number;
      actual_state: string;
      lead_time_days: number;
    }
  ) {
    const event: LearningEvent = {
      type: 'prediction_outcome',
      timestamp: new Date().toISOString(),
      session_hash: 'system',
      user_tier: 'system',
      domain,
      data: {
        numeric_features: {
          predicted_confidence: prediction.predicted_confidence,
          lead_time_days: prediction.lead_time_days,
          correct: prediction.predicted_state === prediction.actual_state ? 1 : 0,
        },
        categorical_features: {
          predicted_state: prediction.predicted_state,
          actual_state: prediction.actual_state,
        },
      },
      metadata: this.getMetadata(),
    };

    // Prediction outcomes are critical - flush immediately
    this.addToBuffer(event);
    await this.flush();
  }

  /**
   * Log implicit user feedback (engagement signals)
   */
  async logUserFeedback(
    sessionHash: string,
    userTier: string,
    feedback: {
      action: 'view' | 'expand' | 'save' | 'share' | 'dismiss';
      section: string;
      dwell_time_ms: number;
    }
  ) {
    const event: LearningEvent = {
      type: 'user_feedback',
      timestamp: new Date().toISOString(),
      session_hash: sessionHash,
      user_tier: userTier,
      domain: feedback.section,
      data: {
        numeric_features: {
          dwell_time_ms: feedback.dwell_time_ms,
        },
        categorical_features: {
          action: feedback.action,
          section: feedback.section,
        },
      },
      metadata: this.getMetadata(),
    };

    this.addToBuffer(event);
  }

  /**
   * Add event to buffer, flush if full
   */
  private addToBuffer(event: LearningEvent) {
    this.buffer.push(event);

    if (this.buffer.length >= this.BUFFER_SIZE) {
      void this.flush();
    }
  }

  /**
   * Flush buffer to database
   */
  async flush() {
    if (this.buffer.length === 0) return;

    const eventsToFlush = [...this.buffer];
    this.buffer = [];

    try {
      const { error } = await this.supabase.from('learning_events').insert(eventsToFlush);

      if (error) {
        console.error('Failed to flush learning events:', error);
        // Re-add to buffer on failure (up to limit)
        this.buffer = [...eventsToFlush.slice(-50), ...this.buffer].slice(-this.BUFFER_SIZE);
      }
    } catch (err) {
      console.error('Learning collector flush error:', err);
    }
  }

  /**
   * Start auto-flush timer
   */
  private startAutoFlush() {
    if (typeof window !== 'undefined') {
      // Client-side: use requestIdleCallback if available
      this.flushInterval = setInterval(() => {
        void this.flush();
      }, this.FLUSH_INTERVAL_MS);
    }
  }

  /**
   * Get metadata for events
   */
  private getMetadata(): EventMetadata {
    return {
      source: 'lattice-web',
      version: '1.0.0',
      environment:
        (process.env.NODE_ENV as 'development' | 'staging' | 'production') || 'development',
    };
  }

  /**
   * Cleanup on shutdown
   */
  async shutdown() {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    await this.flush();
  }
}

/**
 * Training data exporter
 * Exports collected data in formats suitable for fine-tuning
 */
export class TrainingDataExporter {
  private supabase: SupabaseClient;

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
  }

  /**
   * Export reasoning traces as training examples
   * Format: (input_features, conclusion, confidence)
   */
  async exportReasoningTraces(options: {
    startDate: string;
    endDate: string;
    minConfidence: number;
    domain?: string;
  }): Promise<TrainingExample[]> {
    let query = this.supabase
      .from('learning_events')
      .select('*')
      .eq('type', 'reasoning_trace')
      .gte('timestamp', options.startDate)
      .lte('timestamp', options.endDate);

    if (options.domain) {
      query = query.eq('domain', options.domain);
    }

    const { data, error } = await query;

    if (error || !data) {
      console.error('Export error:', error);
      return [];
    }

    return data
      .filter((e) => e.data?.reasoning_output?.confidence >= options.minConfidence)
      .map((e) => ({
        input: e.data.numeric_features,
        output: {
          conclusion_type: e.data.reasoning_output.conclusion_type,
          confidence: e.data.reasoning_output.confidence,
        },
        metadata: {
          domain: e.domain,
          timestamp: e.timestamp,
        },
      }));
  }

  /**
   * Export prediction outcomes for model evaluation
   */
  async exportPredictionOutcomes(options: {
    startDate: string;
    endDate: string;
    domain?: string;
  }): Promise<PredictionOutcome[]> {
    let query = this.supabase
      .from('learning_events')
      .select('*')
      .eq('type', 'prediction_outcome')
      .gte('timestamp', options.startDate)
      .lte('timestamp', options.endDate);

    if (options.domain) {
      query = query.eq('domain', options.domain);
    }

    const { data, error } = await query;

    if (error || !data) {
      console.error('Export error:', error);
      return [];
    }

    return data.map((e) => ({
      predicted: e.data.categorical_features.predicted_state,
      actual: e.data.categorical_features.actual_state,
      confidence: e.data.numeric_features.predicted_confidence,
      correct: e.data.numeric_features.correct === 1,
      lead_time_days: e.data.numeric_features.lead_time_days,
      domain: e.domain,
      timestamp: e.timestamp,
    }));
  }

  /**
   * Compute training metrics
   */
  async computeTrainingMetrics(): Promise<TrainingMetrics> {
    const { data: counts } = await this.supabase
      .from('learning_events')
      .select('type')
      .limit(100000);

    const typeCounts: Record<string, number> = {};
    for (const row of counts || []) {
      typeCounts[row.type] = (typeCounts[row.type] || 0) + 1;
    }

    const { data: outcomes } = await this.supabase
      .from('learning_events')
      .select('data')
      .eq('type', 'prediction_outcome');

    const correctPredictions = (outcomes || []).filter(
      (o) => o.data?.numeric_features?.correct === 1
    ).length;

    return {
      total_events: Object.values(typeCounts).reduce((a, b) => a + b, 0),
      events_by_type: typeCounts,
      prediction_accuracy: outcomes?.length ? correctPredictions / outcomes.length : 0,
      data_freshness: new Date().toISOString(),
    };
  }
}

// Types for exports
interface TrainingExample {
  input: Record<string, number>;
  output: {
    conclusion_type: string;
    confidence: number;
  };
  metadata: {
    domain: string;
    timestamp: string;
  };
}

interface PredictionOutcome {
  predicted: string;
  actual: string;
  confidence: number;
  correct: boolean;
  lead_time_days: number;
  domain: string;
  timestamp: string;
}

interface TrainingMetrics {
  total_events: number;
  events_by_type: Record<string, number>;
  prediction_accuracy: number;
  data_freshness: string;
}

/**
 * Singleton instances
 */
let collectorInstance: LearningCollector | null = null;
let exporterInstance: TrainingDataExporter | null = null;

export function getLearningCollector(): LearningCollector {
  if (!collectorInstance) {
    collectorInstance = new LearningCollector(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );
  }
  return collectorInstance;
}

export function getTrainingDataExporter(): TrainingDataExporter {
  if (!exporterInstance) {
    exporterInstance = new TrainingDataExporter(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }
  return exporterInstance;
}
