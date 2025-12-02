/**
 * LatticeForge Usage Metering System
 *
 * Token-based usage tracking inspired by Anthropic's Claude API.
 * Tracks, bills, and enforces usage limits across all API operations.
 *
 * Token Types:
 * - signal_tokens: Raw signals fetched from sources
 * - fusion_tokens: Fusion operations (weighted by complexity)
 * - analysis_tokens: Formula/algorithm executions
 * - storage_tokens: Data retention beyond default
 *
 * © 2025 Crystalline Labs LLC
 */

export interface UsageTokens {
  /** Tokens from fetching raw signals */
  signal_tokens: number;
  /** Tokens from fusion operations */
  fusion_tokens: number;
  /** Tokens from analysis/formula execution */
  analysis_tokens: number;
  /** Tokens from data storage */
  storage_tokens: number;
  /** Total tokens */
  total_tokens: number;
}

export interface UsageRecord {
  id: string;
  client_id: string;
  api_key_id: string;
  timestamp: Date;
  operation: string;
  tokens: UsageTokens;
  metadata: Record<string, unknown>;
  billing_period: string;
}

export interface UsageLimits {
  /** Max signal tokens per period */
  max_signal_tokens: number;
  /** Max fusion tokens per period */
  max_fusion_tokens: number;
  /** Max analysis tokens per period */
  max_analysis_tokens: number;
  /** Max storage tokens (total, not per period) */
  max_storage_tokens: number;
  /** Max total tokens per period */
  max_total_tokens: number;
  /** Billing period in seconds */
  period_seconds: number;
}

export interface UsageSummary {
  client_id: string;
  period_start: Date;
  period_end: Date;
  tokens: UsageTokens;
  limits: UsageLimits;
  percentage_used: number;
  estimated_cost: number;
}

export interface TokenPricing {
  signal_per_1k: number;
  fusion_per_1k: number;
  analysis_per_1k: number;
  storage_per_1k_per_day: number;
}

/**
 * Operation token costs (proprietary pricing model)
 */
const TOKEN_COSTS: Record<string, Partial<UsageTokens>> = {
  // Signal fetch operations
  'source.fetch': { signal_tokens: 10 },
  'source.fetch.bulk': { signal_tokens: 50 },
  'source.stream.start': { signal_tokens: 5 },
  'source.stream.tick': { signal_tokens: 1 },

  // Fusion operations
  'fusion.simple': { fusion_tokens: 20 },
  'fusion.weighted': { fusion_tokens: 50 },
  'fusion.multi_source': { fusion_tokens: 100 },
  'fusion.real_time': { fusion_tokens: 200 },

  // Analysis operations (proprietary formulas)
  'analysis.phase_transition': { analysis_tokens: 150 },
  'analysis.cascade_predictor': { analysis_tokens: 200 },
  'analysis.sentiment_harmonic': { analysis_tokens: 175 },
  'analysis.anomaly_fingerprint': { analysis_tokens: 250 },
  'analysis.quantum_optimize': { analysis_tokens: 500 },
  'analysis.custom': { analysis_tokens: 100 },

  // Storage operations
  'storage.write': { storage_tokens: 5 },
  'storage.read': { storage_tokens: 1 },
  'storage.retention.extend': { storage_tokens: 10 },
};

/**
 * Tier-based limits (monthly)
 */
const TIER_LIMITS: Record<string, UsageLimits> = {
  free: {
    max_signal_tokens: 10_000,
    max_fusion_tokens: 5_000,
    max_analysis_tokens: 1_000,
    max_storage_tokens: 50_000,
    max_total_tokens: 15_000,
    period_seconds: 30 * 24 * 60 * 60, // Monthly
  },
  pro: {
    max_signal_tokens: 500_000,
    max_fusion_tokens: 250_000,
    max_analysis_tokens: 100_000,
    max_storage_tokens: 5_000_000,
    max_total_tokens: 750_000,
    period_seconds: 30 * 24 * 60 * 60,
  },
  enterprise: {
    max_signal_tokens: 10_000_000,
    max_fusion_tokens: 5_000_000,
    max_analysis_tokens: 2_000_000,
    max_storage_tokens: 100_000_000,
    max_total_tokens: 15_000_000,
    period_seconds: 30 * 24 * 60 * 60,
  },
  government: {
    max_signal_tokens: Infinity,
    max_fusion_tokens: Infinity,
    max_analysis_tokens: Infinity,
    max_storage_tokens: Infinity,
    max_total_tokens: Infinity,
    period_seconds: 30 * 24 * 60 * 60,
  },
};

/**
 * Pricing per 1000 tokens (USD)
 */
const DEFAULT_PRICING: TokenPricing = {
  signal_per_1k: 0.001, // $0.001 per 1K signal tokens
  fusion_per_1k: 0.005, // $0.005 per 1K fusion tokens
  analysis_per_1k: 0.01, // $0.01 per 1K analysis tokens
  storage_per_1k_per_day: 0.0001, // $0.0001 per 1K storage tokens/day
};

/**
 * Usage Meter - tracks and enforces usage limits
 */
export class UsageMeter {
  private records: UsageRecord[] = [];
  private clientUsage: Map<string, Map<string, UsageTokens>> = new Map();
  private pricing: TokenPricing;

  constructor(pricing: TokenPricing = DEFAULT_PRICING) {
    this.pricing = pricing;
  }

  /**
   * Record token usage for an operation
   */
  record(
    clientId: string,
    apiKeyId: string,
    operation: string,
    additionalTokens?: Partial<UsageTokens>,
    metadata?: Record<string, unknown>
  ): UsageRecord {
    // Get base tokens for operation
    const baseTokens = TOKEN_COSTS[operation] ?? {};

    // Merge with additional tokens
    const tokens: UsageTokens = {
      signal_tokens: (baseTokens.signal_tokens ?? 0) + (additionalTokens?.signal_tokens ?? 0),
      fusion_tokens: (baseTokens.fusion_tokens ?? 0) + (additionalTokens?.fusion_tokens ?? 0),
      analysis_tokens: (baseTokens.analysis_tokens ?? 0) + (additionalTokens?.analysis_tokens ?? 0),
      storage_tokens: (baseTokens.storage_tokens ?? 0) + (additionalTokens?.storage_tokens ?? 0),
      total_tokens: 0,
    };

    tokens.total_tokens =
      tokens.signal_tokens +
      tokens.fusion_tokens +
      tokens.analysis_tokens +
      tokens.storage_tokens;

    const now = new Date();
    const billingPeriod = this.getBillingPeriod(now);

    const record: UsageRecord = {
      id: `usage_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`,
      client_id: clientId,
      api_key_id: apiKeyId,
      timestamp: now,
      operation,
      tokens,
      metadata: metadata ?? {},
      billing_period: billingPeriod,
    };

    // Store record
    this.records.push(record);
    if (this.records.length > 100000) {
      this.records = this.records.slice(-50000);
    }

    // Update client usage
    this.updateClientUsage(clientId, billingPeriod, tokens);

    return record;
  }

  /**
   * Check if operation is within limits
   */
  checkLimits(
    clientId: string,
    tier: string,
    operation: string,
    additionalTokens?: Partial<UsageTokens>
  ): {
    allowed: boolean;
    reason?: string;
    current: UsageTokens;
    limits: UsageLimits;
    headroom: UsageTokens;
  } {
    const limits = TIER_LIMITS[tier] ?? TIER_LIMITS.free;
    const billingPeriod = this.getBillingPeriod(new Date());
    const current = this.getClientPeriodUsage(clientId, billingPeriod);

    // Estimate tokens for this operation
    const baseTokens = TOKEN_COSTS[operation] ?? {};
    const operationTokens: UsageTokens = {
      signal_tokens: (baseTokens.signal_tokens ?? 0) + (additionalTokens?.signal_tokens ?? 0),
      fusion_tokens: (baseTokens.fusion_tokens ?? 0) + (additionalTokens?.fusion_tokens ?? 0),
      analysis_tokens: (baseTokens.analysis_tokens ?? 0) + (additionalTokens?.analysis_tokens ?? 0),
      storage_tokens: (baseTokens.storage_tokens ?? 0) + (additionalTokens?.storage_tokens ?? 0),
      total_tokens: 0,
    };
    operationTokens.total_tokens =
      operationTokens.signal_tokens +
      operationTokens.fusion_tokens +
      operationTokens.analysis_tokens +
      operationTokens.storage_tokens;

    // Calculate headroom
    const headroom: UsageTokens = {
      signal_tokens: limits.max_signal_tokens - current.signal_tokens,
      fusion_tokens: limits.max_fusion_tokens - current.fusion_tokens,
      analysis_tokens: limits.max_analysis_tokens - current.analysis_tokens,
      storage_tokens: limits.max_storage_tokens - current.storage_tokens,
      total_tokens: limits.max_total_tokens - current.total_tokens,
    };

    // Check each limit
    let allowed = true;
    let reason: string | undefined;

    if (current.signal_tokens + operationTokens.signal_tokens > limits.max_signal_tokens) {
      allowed = false;
      reason = `Signal token limit exceeded (${current.signal_tokens}/${limits.max_signal_tokens})`;
    } else if (current.fusion_tokens + operationTokens.fusion_tokens > limits.max_fusion_tokens) {
      allowed = false;
      reason = `Fusion token limit exceeded (${current.fusion_tokens}/${limits.max_fusion_tokens})`;
    } else if (current.analysis_tokens + operationTokens.analysis_tokens > limits.max_analysis_tokens) {
      allowed = false;
      reason = `Analysis token limit exceeded (${current.analysis_tokens}/${limits.max_analysis_tokens})`;
    } else if (current.total_tokens + operationTokens.total_tokens > limits.max_total_tokens) {
      allowed = false;
      reason = `Total token limit exceeded (${current.total_tokens}/${limits.max_total_tokens})`;
    }

    return { allowed, reason, current, limits, headroom };
  }

  /**
   * Get usage summary for a client
   */
  getSummary(clientId: string, tier: string): UsageSummary {
    const now = new Date();
    const billingPeriod = this.getBillingPeriod(now);
    const limits = TIER_LIMITS[tier] ?? TIER_LIMITS.free;
    const tokens = this.getClientPeriodUsage(clientId, billingPeriod);

    // Calculate period boundaries
    const periodStart = new Date(billingPeriod);
    const periodEnd = new Date(periodStart.getTime() + limits.period_seconds * 1000);

    // Calculate percentage used
    const percentageUsed = limits.max_total_tokens > 0
      ? (tokens.total_tokens / limits.max_total_tokens) * 100
      : 0;

    // Estimate cost
    const estimatedCost = this.calculateCost(tokens);

    return {
      client_id: clientId,
      period_start: periodStart,
      period_end: periodEnd,
      tokens,
      limits,
      percentage_used: Math.min(100, percentageUsed),
      estimated_cost: estimatedCost,
    };
  }

  /**
   * Get detailed usage history for a client
   */
  getHistory(
    clientId: string,
    options: {
      startDate?: Date;
      endDate?: Date;
      operation?: string;
      limit?: number;
    } = {}
  ): UsageRecord[] {
    let filtered = this.records.filter((r) => r.client_id === clientId);

    if (options.startDate) {
      filtered = filtered.filter((r) => r.timestamp >= options.startDate!);
    }

    if (options.endDate) {
      filtered = filtered.filter((r) => r.timestamp <= options.endDate!);
    }

    if (options.operation) {
      filtered = filtered.filter((r) => r.operation === options.operation);
    }

    // Sort by timestamp descending
    filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    if (options.limit) {
      filtered = filtered.slice(0, options.limit);
    }

    return filtered;
  }

  /**
   * Calculate cost for tokens
   */
  calculateCost(tokens: UsageTokens): number {
    return (
      (tokens.signal_tokens / 1000) * this.pricing.signal_per_1k +
      (tokens.fusion_tokens / 1000) * this.pricing.fusion_per_1k +
      (tokens.analysis_tokens / 1000) * this.pricing.analysis_per_1k +
      (tokens.storage_tokens / 1000) * this.pricing.storage_per_1k_per_day
    );
  }

  /**
   * Get token cost for an operation
   */
  getOperationCost(operation: string): Partial<UsageTokens> {
    return TOKEN_COSTS[operation] ?? {};
  }

  /**
   * Get all available operations and their costs
   */
  getOperationCatalog(): Record<string, Partial<UsageTokens>> {
    return { ...TOKEN_COSTS };
  }

  /**
   * Get limits for a tier
   */
  getTierLimits(tier: string): UsageLimits {
    return TIER_LIMITS[tier] ?? TIER_LIMITS.free;
  }

  /**
   * Export usage data for billing/audit
   */
  exportUsage(
    clientId: string,
    billingPeriod: string
  ): {
    records: UsageRecord[];
    summary: {
      tokens: UsageTokens;
      cost: number;
      operations: Record<string, number>;
    };
  } {
    const records = this.records.filter(
      (r) => r.client_id === clientId && r.billing_period === billingPeriod
    );

    const tokens: UsageTokens = {
      signal_tokens: 0,
      fusion_tokens: 0,
      analysis_tokens: 0,
      storage_tokens: 0,
      total_tokens: 0,
    };

    const operations: Record<string, number> = {};

    for (const record of records) {
      tokens.signal_tokens += record.tokens.signal_tokens;
      tokens.fusion_tokens += record.tokens.fusion_tokens;
      tokens.analysis_tokens += record.tokens.analysis_tokens;
      tokens.storage_tokens += record.tokens.storage_tokens;
      tokens.total_tokens += record.tokens.total_tokens;

      operations[record.operation] = (operations[record.operation] ?? 0) + 1;
    }

    return {
      records,
      summary: {
        tokens,
        cost: this.calculateCost(tokens),
        operations,
      },
    };
  }

  // Helper methods
  private getBillingPeriod(date: Date): string {
    // Monthly billing periods (first of month)
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-01`;
  }

  private updateClientUsage(
    clientId: string,
    billingPeriod: string,
    tokens: UsageTokens
  ): void {
    if (!this.clientUsage.has(clientId)) {
      this.clientUsage.set(clientId, new Map());
    }

    const clientPeriods = this.clientUsage.get(clientId)!;

    if (!clientPeriods.has(billingPeriod)) {
      clientPeriods.set(billingPeriod, {
        signal_tokens: 0,
        fusion_tokens: 0,
        analysis_tokens: 0,
        storage_tokens: 0,
        total_tokens: 0,
      });
    }

    const current = clientPeriods.get(billingPeriod)!;
    current.signal_tokens += tokens.signal_tokens;
    current.fusion_tokens += tokens.fusion_tokens;
    current.analysis_tokens += tokens.analysis_tokens;
    current.storage_tokens += tokens.storage_tokens;
    current.total_tokens += tokens.total_tokens;
  }

  private getClientPeriodUsage(clientId: string, billingPeriod: string): UsageTokens {
    const clientPeriods = this.clientUsage.get(clientId);
    if (!clientPeriods) {
      return {
        signal_tokens: 0,
        fusion_tokens: 0,
        analysis_tokens: 0,
        storage_tokens: 0,
        total_tokens: 0,
      };
    }

    return clientPeriods.get(billingPeriod) ?? {
      signal_tokens: 0,
      fusion_tokens: 0,
      analysis_tokens: 0,
      storage_tokens: 0,
      total_tokens: 0,
    };
  }
}

// Singleton instance
export const usageMeter = new UsageMeter();

// Export types
export { TOKEN_COSTS, TIER_LIMITS, DEFAULT_PRICING };
