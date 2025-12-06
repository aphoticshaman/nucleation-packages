/**
 * Alert Pipeline V2 - Anti-Complaint Engineering Spec Section 2 Implementation
 *
 * Precision-First Alert Architecture:
 * - 3-stage gauntlet: Contextual Relevance → Temporal Deduplication → Confidence Threshold
 * - Default: SUPPRESS. Burden of proof on ALERT, not IGNORE.
 * - Hard caps: 5 alerts/hour, 20 alerts/day
 * - Mandatory cooldown between same-category alerts
 * - User-calibrated relevance with active learning
 */

import { createClient } from '@supabase/supabase-js';

// =============================================================================
// TYPES
// =============================================================================

export interface CrownJewel {
  id: string;
  type: 'asset' | 'ip' | 'domain' | 'executive' | 'facility' | 'brand';
  value: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  metadata?: Record<string, unknown>;
}

export interface ThreatActorOfInterest {
  id: string;
  name: string;
  aliases: string[];
  type: 'apt' | 'nation_state' | 'criminal' | 'hacktivist' | 'insider';
  relevance: 'direct_threat' | 'sector_threat' | 'monitoring';
}

export interface NoiseSource {
  id: string;
  pattern: string;
  type: 'regex' | 'keyword' | 'domain' | 'source_id';
  reason: string;
  addedAt: Date;
  falsePositiveCount: number;
}

export interface UserAlertProfile {
  userId: string;
  orgId: string;

  // Crown jewels - what the user cares about most
  crownJewels: CrownJewel[];

  // Threat actors of interest
  threatActors: ThreatActorOfInterest[];

  // Known noise sources
  noiseSources: NoiseSource[];

  // Confidence thresholds
  notificationThreshold: number; // Default 0.7, floor 0.5
  logOnlyThreshold: number;      // Below notification, above this = log only

  // Rate preferences
  maxAlertsPerHour: number;      // Default 5, max 10
  maxAlertsPerDay: number;       // Default 20, max 50

  // Cooldown preferences (seconds)
  sameCategoryCooldown: number;  // Default 300

  // Learning state
  dismissalPatterns: DismissalPattern[];
  forwardPatterns: ForwardPattern[];
}

export interface DismissalPattern {
  alertType: string;
  avgDismissalTimeMs: number;
  count: number;
  lastOccurrence: Date;
}

export interface ForwardPattern {
  alertType: string;
  forwardCount: number;
  lastForwarded: Date;
}

export interface RawAlert {
  id: string;
  timestamp: Date;
  category: AlertCategory;
  subcategory?: string;

  // Content
  title: string;
  description: string;
  rawContent: string;

  // Source information
  sourceId: string;
  sourceName: string;
  sourceType: 'osint' | 'threat_feed' | 'internal' | 'partner';
  sourceUrl?: string;

  // Initial scoring
  rawConfidence: number;
  indicators: string[];
  entities: string[];

  // Metadata
  language: string;
  region?: string;
}

export type AlertCategory =
  | 'security'
  | 'geopolitical'
  | 'economic'
  | 'military'
  | 'humanitarian'
  | 'cyber'
  | 'supply_chain'
  | 'regulatory';

export interface EnrichedAlert extends RawAlert {
  // Contextual enrichment
  relevanceScore: number;
  matchedCrownJewels: CrownJewel[];
  matchedThreatActors: ThreatActorOfInterest[];

  // Corroboration
  corroborationScore: number;
  corroboratingSourceCount: number;
  corroboratingSources: string[];

  // Historical context
  previousOccurrences: number;
  lastSeenAt?: Date;
  historicalOutcome?: 'confirmed' | 'false_positive' | 'unknown';

  // Confidence with uncertainty
  adjustedConfidence: number;
  confidenceInterval: [number, number]; // [lower, upper]

  // Recommended action
  recommendedAction: RecommendedAction;
  actionRationale: string;

  // Pipeline metadata
  pipelineStages: PipelineStageResult[];
  finalDecision: 'notify' | 'log_only' | 'suppress';
  suppressionReason?: string;
}

export interface RecommendedAction {
  action: 'investigate' | 'escalate' | 'monitor' | 'block' | 'inform_stakeholder' | 'no_action';
  priority: 'immediate' | 'urgent' | 'normal' | 'low';
  suggestedSteps: string[];
  estimatedEffortMinutes: number;
}

export interface PipelineStageResult {
  stage: 'contextual_relevance' | 'temporal_deduplication' | 'confidence_threshold';
  passed: boolean;
  score: number;
  reason: string;
  executionTimeMs: number;
}

// =============================================================================
// PIPELINE STAGES
// =============================================================================

/**
 * Stage 1: Contextual Relevance Filter
 *
 * Does this alert match the user's actual attack surface?
 * - Checks against crown jewels
 * - Matches threat actors of interest
 * - Filters known noise sources
 */
export class ContextualRelevanceFilter {
  constructor(private profile: UserAlertProfile) {}

  async filter(alert: RawAlert): Promise<PipelineStageResult & { enrichment: Partial<EnrichedAlert> }> {
    const startTime = Date.now();

    // Check noise sources first (fast rejection)
    const matchedNoise = this.checkNoiseSources(alert);
    if (matchedNoise) {
      return {
        stage: 'contextual_relevance',
        passed: false,
        score: 0,
        reason: `Matched noise source: ${matchedNoise.reason}`,
        executionTimeMs: Date.now() - startTime,
        enrichment: {},
      };
    }

    // Check crown jewels
    const matchedJewels = this.matchCrownJewels(alert);

    // Check threat actors
    const matchedActors = this.matchThreatActors(alert);

    // Calculate relevance score
    let relevanceScore = 0;

    // Crown jewel matches
    for (const jewel of matchedJewels) {
      switch (jewel.priority) {
        case 'critical': relevanceScore += 0.4; break;
        case 'high': relevanceScore += 0.3; break;
        case 'medium': relevanceScore += 0.2; break;
        case 'low': relevanceScore += 0.1; break;
      }
    }

    // Threat actor matches
    for (const actor of matchedActors) {
      switch (actor.relevance) {
        case 'direct_threat': relevanceScore += 0.4; break;
        case 'sector_threat': relevanceScore += 0.2; break;
        case 'monitoring': relevanceScore += 0.1; break;
      }
    }

    // Cap at 1.0
    relevanceScore = Math.min(1.0, relevanceScore);

    // If no matches but high raw confidence, still give a base score
    if (relevanceScore === 0 && alert.rawConfidence >= 0.8) {
      relevanceScore = 0.3; // Base relevance for high-confidence generic alerts
    }

    const passed = relevanceScore >= 0.2;

    return {
      stage: 'contextual_relevance',
      passed,
      score: relevanceScore,
      reason: passed
        ? `Matched ${matchedJewels.length} crown jewels, ${matchedActors.length} threat actors`
        : 'No relevant matches to attack surface',
      executionTimeMs: Date.now() - startTime,
      enrichment: {
        relevanceScore,
        matchedCrownJewels: matchedJewels,
        matchedThreatActors: matchedActors,
      },
    };
  }

  private checkNoiseSources(alert: RawAlert): NoiseSource | null {
    const contentToCheck = `${alert.title} ${alert.description} ${alert.rawContent}`.toLowerCase();

    for (const noise of this.profile.noiseSources) {
      switch (noise.type) {
        case 'keyword':
          if (contentToCheck.includes(noise.pattern.toLowerCase())) {
            return noise;
          }
          break;
        case 'regex':
          try {
            const regex = new RegExp(noise.pattern, 'i');
            if (regex.test(contentToCheck)) {
              return noise;
            }
          } catch {
            // Invalid regex, skip
          }
          break;
        case 'domain':
          if (alert.sourceUrl?.includes(noise.pattern)) {
            return noise;
          }
          break;
        case 'source_id':
          if (alert.sourceId === noise.pattern) {
            return noise;
          }
          break;
      }
    }
    return null;
  }

  private matchCrownJewels(alert: RawAlert): CrownJewel[] {
    const matched: CrownJewel[] = [];
    const contentToCheck = `${alert.title} ${alert.description} ${alert.indicators.join(' ')} ${alert.entities.join(' ')}`.toLowerCase();

    for (const jewel of this.profile.crownJewels) {
      const valueLower = jewel.value.toLowerCase();
      if (contentToCheck.includes(valueLower)) {
        matched.push(jewel);
      }
    }
    return matched;
  }

  private matchThreatActors(alert: RawAlert): ThreatActorOfInterest[] {
    const matched: ThreatActorOfInterest[] = [];
    const contentToCheck = `${alert.title} ${alert.description} ${alert.entities.join(' ')}`.toLowerCase();

    for (const actor of this.profile.threatActors) {
      const namesToCheck = [actor.name, ...actor.aliases].map(n => n.toLowerCase());
      if (namesToCheck.some(name => contentToCheck.includes(name))) {
        matched.push(actor);
      }
    }
    return matched;
  }
}

/**
 * Stage 2: Temporal Deduplication Filter
 *
 * Is this meaningfully different from alerts in the last 72 hours?
 * - Content similarity check
 * - Indicator overlap detection
 * - Cooldown enforcement
 */
export class TemporalDeduplicationFilter {
  private recentAlerts: Map<string, { alert: RawAlert; timestamp: Date }[]> = new Map();
  private readonly DEDUP_WINDOW_MS = 72 * 60 * 60 * 1000; // 72 hours

  constructor(private profile: UserAlertProfile) {}

  async filter(alert: RawAlert, category: string): Promise<PipelineStageResult> {
    const startTime = Date.now();

    // Check cooldown for same category
    const cooldownViolation = this.checkCooldown(alert.category);
    if (cooldownViolation) {
      return {
        stage: 'temporal_deduplication',
        passed: false,
        score: 0,
        reason: `Cooldown active: ${cooldownViolation.remainingSeconds}s remaining for ${alert.category}`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // Check content similarity
    const similarityResult = this.checkSimilarity(alert);

    if (similarityResult.isDuplicate) {
      return {
        stage: 'temporal_deduplication',
        passed: false,
        score: similarityResult.maxSimilarity,
        reason: `Duplicate of alert from ${similarityResult.duplicateAge} ago (${(similarityResult.maxSimilarity * 100).toFixed(0)}% similar)`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    // Store this alert for future dedup checks
    this.storeAlert(alert);

    return {
      stage: 'temporal_deduplication',
      passed: true,
      score: 1 - similarityResult.maxSimilarity,
      reason: 'Alert is sufficiently novel',
      executionTimeMs: Date.now() - startTime,
    };
  }

  private checkCooldown(category: AlertCategory): { remainingSeconds: number } | null {
    const categoryAlerts = this.recentAlerts.get(category) || [];
    const now = Date.now();
    const cooldownMs = this.profile.sameCategoryCooldown * 1000;

    for (const { timestamp } of categoryAlerts) {
      const elapsed = now - timestamp.getTime();
      if (elapsed < cooldownMs) {
        return { remainingSeconds: Math.ceil((cooldownMs - elapsed) / 1000) };
      }
    }
    return null;
  }

  private checkSimilarity(alert: RawAlert): {
    isDuplicate: boolean;
    maxSimilarity: number;
    duplicateAge?: string;
  } {
    const now = Date.now();
    const cutoff = now - this.DEDUP_WINDOW_MS;
    let maxSimilarity = 0;
    let duplicateAge = '';

    // Check all categories (similar alerts might be in different categories)
    for (const [, alerts] of this.recentAlerts) {
      for (const { alert: recentAlert, timestamp } of alerts) {
        if (timestamp.getTime() < cutoff) continue;

        const similarity = this.computeSimilarity(alert, recentAlert);
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          const ageMs = now - timestamp.getTime();
          const hours = Math.floor(ageMs / (1000 * 60 * 60));
          duplicateAge = hours > 0 ? `${hours}h` : `${Math.floor(ageMs / (1000 * 60))}m`;
        }
      }
    }

    return {
      isDuplicate: maxSimilarity >= 0.8, // 80% similarity threshold
      maxSimilarity,
      duplicateAge,
    };
  }

  private computeSimilarity(a: RawAlert, b: RawAlert): number {
    // Indicator overlap (weighted heavily)
    const indicatorOverlap = this.jaccardSimilarity(
      new Set(a.indicators),
      new Set(b.indicators)
    );

    // Entity overlap
    const entityOverlap = this.jaccardSimilarity(
      new Set(a.entities),
      new Set(b.entities)
    );

    // Title similarity (simple word overlap)
    const titleSimilarity = this.jaccardSimilarity(
      new Set(a.title.toLowerCase().split(/\s+/)),
      new Set(b.title.toLowerCase().split(/\s+/))
    );

    // Weighted combination
    return indicatorOverlap * 0.5 + entityOverlap * 0.3 + titleSimilarity * 0.2;
  }

  private jaccardSimilarity<T>(setA: Set<T>, setB: Set<T>): number {
    if (setA.size === 0 && setB.size === 0) return 0;
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    return intersection.size / union.size;
  }

  private storeAlert(alert: RawAlert): void {
    const categoryAlerts = this.recentAlerts.get(alert.category) || [];
    categoryAlerts.push({ alert, timestamp: new Date() });

    // Cleanup old alerts
    const cutoff = Date.now() - this.DEDUP_WINDOW_MS;
    const filtered = categoryAlerts.filter(a => a.timestamp.getTime() >= cutoff);
    this.recentAlerts.set(alert.category, filtered);
  }
}

/**
 * Stage 3: Confidence Threshold Filter
 *
 * Does the evidence exceed user-calibrated threshold?
 * - Applies user's notification threshold
 * - Computes confidence intervals
 * - Accounts for source reliability
 */
export class ConfidenceThresholdFilter {
  constructor(private profile: UserAlertProfile) {}

  async filter(
    alert: RawAlert,
    enrichment: Partial<EnrichedAlert>
  ): Promise<PipelineStageResult & { enrichment: Partial<EnrichedAlert> }> {
    const startTime = Date.now();

    // Compute adjusted confidence
    const sourceReliability = this.getSourceReliability(alert.sourceType);
    const corroborationBoost = (enrichment.corroborationScore || 0) * 0.1;
    const relevanceBoost = (enrichment.relevanceScore || 0) * 0.1;

    let adjustedConfidence = alert.rawConfidence * sourceReliability + corroborationBoost + relevanceBoost;
    adjustedConfidence = Math.min(1.0, Math.max(0, adjustedConfidence));

    // Compute confidence interval
    const uncertainty = this.computeUncertainty(alert, enrichment);
    const confidenceInterval: [number, number] = [
      Math.max(0, adjustedConfidence - uncertainty),
      Math.min(1, adjustedConfidence + uncertainty),
    ];

    // Check thresholds
    const meetsNotificationThreshold = adjustedConfidence >= this.profile.notificationThreshold;
    const meetsLogThreshold = adjustedConfidence >= this.profile.logOnlyThreshold;

    let decision: 'notify' | 'log_only' | 'suppress';
    let reason: string;

    if (meetsNotificationThreshold) {
      decision = 'notify';
      reason = `Confidence ${(adjustedConfidence * 100).toFixed(0)}% exceeds notification threshold ${(this.profile.notificationThreshold * 100).toFixed(0)}%`;
    } else if (meetsLogThreshold) {
      decision = 'log_only';
      reason = `Confidence ${(adjustedConfidence * 100).toFixed(0)}% below notification but above log threshold`;
    } else {
      decision = 'suppress';
      reason = `Confidence ${(adjustedConfidence * 100).toFixed(0)}% below minimum threshold`;
    }

    // Generate recommended action
    const recommendedAction = this.generateRecommendedAction(alert, enrichment, adjustedConfidence);

    return {
      stage: 'confidence_threshold',
      passed: meetsNotificationThreshold,
      score: adjustedConfidence,
      reason,
      executionTimeMs: Date.now() - startTime,
      enrichment: {
        adjustedConfidence,
        confidenceInterval,
        recommendedAction,
        actionRationale: this.generateActionRationale(alert, enrichment, recommendedAction),
        finalDecision: decision,
      },
    };
  }

  private getSourceReliability(sourceType: string): number {
    switch (sourceType) {
      case 'internal': return 1.0;
      case 'partner': return 0.95;
      case 'threat_feed': return 0.85;
      case 'osint': return 0.7;
      default: return 0.5;
    }
  }

  private computeUncertainty(alert: RawAlert, enrichment: Partial<EnrichedAlert>): number {
    let uncertainty = 0.1; // Base uncertainty

    // More sources = less uncertainty
    const sourceCount = enrichment.corroboratingSourceCount || 1;
    uncertainty -= Math.min(0.05, sourceCount * 0.01);

    // Historical confirmation reduces uncertainty
    if (enrichment.historicalOutcome === 'confirmed') {
      uncertainty -= 0.03;
    } else if (enrichment.historicalOutcome === 'false_positive') {
      uncertainty += 0.05;
    }

    // OSINT has higher uncertainty
    if (alert.sourceType === 'osint') {
      uncertainty += 0.05;
    }

    return Math.max(0.05, Math.min(0.2, uncertainty));
  }

  private generateRecommendedAction(
    alert: RawAlert,
    enrichment: Partial<EnrichedAlert>,
    confidence: number
  ): RecommendedAction {
    const hasCriticalJewel = enrichment.matchedCrownJewels?.some(j => j.priority === 'critical');
    const hasDirectThreat = enrichment.matchedThreatActors?.some(a => a.relevance === 'direct_threat');

    // Determine priority
    let priority: RecommendedAction['priority'];
    if (hasCriticalJewel || hasDirectThreat) {
      priority = confidence >= 0.85 ? 'immediate' : 'urgent';
    } else if (confidence >= 0.8) {
      priority = 'urgent';
    } else if (confidence >= 0.7) {
      priority = 'normal';
    } else {
      priority = 'low';
    }

    // Determine action
    let action: RecommendedAction['action'];
    let steps: string[];
    let effort: number;

    if (alert.category === 'security' || alert.category === 'cyber') {
      if (hasCriticalJewel) {
        action = 'escalate';
        steps = [
          'Notify security team lead immediately',
          'Verify indicator presence in environment',
          'Initiate incident response if confirmed',
          'Document timeline and affected assets',
        ];
        effort = 60;
      } else {
        action = 'investigate';
        steps = [
          'Search logs for related indicators',
          'Check if indicators match known-good activity',
          'Assess potential impact if true positive',
          'Update threat model if new TTP observed',
        ];
        effort = 30;
      }
    } else if (alert.category === 'geopolitical' || alert.category === 'military') {
      action = hasDirectThreat ? 'inform_stakeholder' : 'monitor';
      steps = hasDirectThreat
        ? [
          'Brief relevant business units',
          'Update risk assessment for affected regions',
          'Consider operational adjustments if warranted',
        ]
        : [
          'Add to daily intelligence brief',
          'Set up monitoring for escalation indicators',
          'Review related historical events',
        ];
      effort = hasDirectThreat ? 45 : 15;
    } else {
      action = 'monitor';
      steps = [
        'Review alert details for relevance',
        'Add to tracking if pattern emerges',
        'No immediate action required',
      ];
      effort = 10;
    }

    return { action, priority, suggestedSteps: steps, estimatedEffortMinutes: effort };
  }

  private generateActionRationale(
    alert: RawAlert,
    enrichment: Partial<EnrichedAlert>,
    action: RecommendedAction
  ): string {
    const parts: string[] = [];

    if (enrichment.matchedCrownJewels?.length) {
      const jewels = enrichment.matchedCrownJewels.map(j => j.value).join(', ');
      parts.push(`Matches critical assets: ${jewels}`);
    }

    if (enrichment.matchedThreatActors?.length) {
      const actors = enrichment.matchedThreatActors.map(a => a.name).join(', ');
      parts.push(`Associated threat actors: ${actors}`);
    }

    if (enrichment.corroboratingSourceCount && enrichment.corroboratingSourceCount > 1) {
      parts.push(`Corroborated by ${enrichment.corroboratingSourceCount} independent sources`);
    }

    if (enrichment.previousOccurrences) {
      parts.push(`Seen ${enrichment.previousOccurrences} times previously`);
    }

    parts.push(`Recommended ${action.action} with ${action.priority} priority`);

    return parts.join('. ');
  }
}

// =============================================================================
// RATE LIMITER
// =============================================================================

export class AlertRateLimiter {
  private hourlyCount: Map<string, { count: number; resetAt: Date }> = new Map();
  private dailyCount: Map<string, { count: number; resetAt: Date }> = new Map();

  constructor(private profile: UserAlertProfile) {}

  canSendAlert(userId: string): { allowed: boolean; reason?: string } {
    const now = new Date();

    // Check hourly limit
    const hourly = this.hourlyCount.get(userId);
    if (hourly) {
      if (now < hourly.resetAt) {
        if (hourly.count >= this.profile.maxAlertsPerHour) {
          return { allowed: false, reason: `Hourly limit (${this.profile.maxAlertsPerHour}) reached` };
        }
      } else {
        this.hourlyCount.set(userId, { count: 0, resetAt: new Date(now.getTime() + 60 * 60 * 1000) });
      }
    } else {
      this.hourlyCount.set(userId, { count: 0, resetAt: new Date(now.getTime() + 60 * 60 * 1000) });
    }

    // Check daily limit
    const daily = this.dailyCount.get(userId);
    if (daily) {
      if (now < daily.resetAt) {
        if (daily.count >= this.profile.maxAlertsPerDay) {
          return { allowed: false, reason: `Daily limit (${this.profile.maxAlertsPerDay}) reached` };
        }
      } else {
        this.dailyCount.set(userId, { count: 0, resetAt: new Date(now.getTime() + 24 * 60 * 60 * 1000) });
      }
    } else {
      this.dailyCount.set(userId, { count: 0, resetAt: new Date(now.getTime() + 24 * 60 * 60 * 1000) });
    }

    return { allowed: true };
  }

  recordAlert(userId: string): void {
    const hourly = this.hourlyCount.get(userId);
    if (hourly) hourly.count++;

    const daily = this.dailyCount.get(userId);
    if (daily) daily.count++;
  }
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

export class AlertPipelineV2 {
  private contextualFilter: ContextualRelevanceFilter;
  private deduplicationFilter: TemporalDeduplicationFilter;
  private confidenceFilter: ConfidenceThresholdFilter;
  private rateLimiter: AlertRateLimiter;

  // Hard caps per spec
  private static readonly HARD_MAX_ALERTS_PER_HOUR = 10;
  private static readonly HARD_MAX_ALERTS_PER_DAY = 50;
  private static readonly HARD_MIN_CONFIDENCE = 0.5;

  constructor(private profile: UserAlertProfile) {
    // Enforce hard limits
    this.profile.maxAlertsPerHour = Math.min(
      this.profile.maxAlertsPerHour,
      AlertPipelineV2.HARD_MAX_ALERTS_PER_HOUR
    );
    this.profile.maxAlertsPerDay = Math.min(
      this.profile.maxAlertsPerDay,
      AlertPipelineV2.HARD_MAX_ALERTS_PER_DAY
    );
    this.profile.notificationThreshold = Math.max(
      this.profile.notificationThreshold,
      AlertPipelineV2.HARD_MIN_CONFIDENCE
    );

    this.contextualFilter = new ContextualRelevanceFilter(this.profile);
    this.deduplicationFilter = new TemporalDeduplicationFilter(this.profile);
    this.confidenceFilter = new ConfidenceThresholdFilter(this.profile);
    this.rateLimiter = new AlertRateLimiter(this.profile);
  }

  /**
   * Process a raw alert through the 3-stage gauntlet.
   *
   * Default: SUPPRESS. Alert must prove it deserves attention.
   */
  async process(alert: RawAlert): Promise<EnrichedAlert> {
    const pipelineStages: PipelineStageResult[] = [];
    let enrichment: Partial<EnrichedAlert> = {};

    // Stage 1: Contextual Relevance
    const relevanceResult = await this.contextualFilter.filter(alert);
    pipelineStages.push(relevanceResult);
    enrichment = { ...enrichment, ...relevanceResult.enrichment };

    if (!relevanceResult.passed) {
      return this.finalizeAlert(alert, enrichment, pipelineStages, 'suppress', relevanceResult.reason);
    }

    // Stage 2: Temporal Deduplication
    const dedupResult = await this.deduplicationFilter.filter(alert, alert.category);
    pipelineStages.push(dedupResult);

    if (!dedupResult.passed) {
      return this.finalizeAlert(alert, enrichment, pipelineStages, 'suppress', dedupResult.reason);
    }

    // Stage 3: Confidence Threshold
    const confidenceResult = await this.confidenceFilter.filter(alert, enrichment);
    pipelineStages.push(confidenceResult);
    enrichment = { ...enrichment, ...confidenceResult.enrichment };

    const decision = enrichment.finalDecision || 'suppress';

    // Check rate limits for notification
    if (decision === 'notify') {
      const rateCheck = this.rateLimiter.canSendAlert(this.profile.userId);
      if (!rateCheck.allowed) {
        return this.finalizeAlert(alert, enrichment, pipelineStages, 'log_only', rateCheck.reason);
      }
      this.rateLimiter.recordAlert(this.profile.userId);
    }

    return this.finalizeAlert(alert, enrichment, pipelineStages, decision);
  }

  private finalizeAlert(
    alert: RawAlert,
    enrichment: Partial<EnrichedAlert>,
    stages: PipelineStageResult[],
    decision: 'notify' | 'log_only' | 'suppress',
    suppressionReason?: string
  ): EnrichedAlert {
    return {
      ...alert,
      relevanceScore: enrichment.relevanceScore || 0,
      matchedCrownJewels: enrichment.matchedCrownJewels || [],
      matchedThreatActors: enrichment.matchedThreatActors || [],
      corroborationScore: enrichment.corroborationScore || 0,
      corroboratingSourceCount: enrichment.corroboratingSourceCount || 1,
      corroboratingSources: enrichment.corroboratingSources || [alert.sourceName],
      previousOccurrences: enrichment.previousOccurrences || 0,
      adjustedConfidence: enrichment.adjustedConfidence || alert.rawConfidence,
      confidenceInterval: enrichment.confidenceInterval || [
        alert.rawConfidence - 0.1,
        alert.rawConfidence + 0.1,
      ],
      recommendedAction: enrichment.recommendedAction || {
        action: 'no_action',
        priority: 'low',
        suggestedSteps: [],
        estimatedEffortMinutes: 0,
      },
      actionRationale: enrichment.actionRationale || '',
      pipelineStages: stages,
      finalDecision: decision,
      suppressionReason,
    };
  }

  /**
   * Learn from user feedback on alerts.
   */
  async recordFeedback(
    alertId: string,
    feedback: 'dismissed' | 'forwarded' | 'confirmed' | 'false_positive',
    dismissalTimeMs?: number
  ): Promise<void> {
    // Update dismissal patterns for learning
    if (feedback === 'dismissed' && dismissalTimeMs !== undefined) {
      // Quick dismissal (<5s) indicates likely noise
      if (dismissalTimeMs < 5000) {
        // Would update profile.dismissalPatterns here
        // This data feeds back into relevance scoring
      }
    }

    // Forward patterns indicate high-value alerts
    if (feedback === 'forwarded') {
      // Would update profile.forwardPatterns here
    }
  }
}

// =============================================================================
// FACTORY
// =============================================================================

export function createDefaultAlertProfile(userId: string, orgId: string): UserAlertProfile {
  return {
    userId,
    orgId,
    crownJewels: [],
    threatActors: [],
    noiseSources: [],
    notificationThreshold: 0.7, // Per spec default
    logOnlyThreshold: 0.5,
    maxAlertsPerHour: 5, // Per spec default
    maxAlertsPerDay: 20, // Per spec default
    sameCategoryCooldown: 300, // 5 minutes per spec
    dismissalPatterns: [],
    forwardPatterns: [],
  };
}
