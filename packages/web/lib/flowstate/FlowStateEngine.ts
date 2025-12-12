/**
 * FlowState Engine - Unified Phase Transition Detection for LatticeForge
 *
 * Integrates ALL nucleation packages to monitor:
 * - User engagement and churn risk
 * - System behavior and regime shifts
 * - Security threats and anomalies
 * - Website UX issues via behavioral drift
 * - AI interaction quality and learning opportunities
 * - Global fusion of all signals
 *
 * Uses WASM-accelerated detection from nucleation-packages.
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { getElleGuardian } from '@/lib/reasoning/security';
import { computeCIC, gaugeClustering, fuseSignals } from '@/lib/wasm';

// =============================================================================
// TYPES
// =============================================================================

export type FlowStateLevel = 'green' | 'yellow' | 'orange' | 'red';
export type FlowStateDomain =
  | 'user_engagement'    // User activity patterns
  | 'user_churn'         // Churn risk detection
  | 'ux_drift'           // UX/UI issue detection
  | 'system_health'      // System performance
  | 'security'           // Threat detection
  | 'ai_quality'         // AI response quality
  | 'learning'           // Training opportunity detection
  | 'fusion';            // Meta-signal from all domains

export interface FlowStateSignal {
  domain: FlowStateDomain;
  level: FlowStateLevel;
  levelNumeric: number;  // 0-3
  transitioning: boolean;
  confidence: number;
  variance: number;
  trend: number;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface UserFlowState {
  userId: string;
  signals: Map<FlowStateDomain, FlowStateSignal>;
  fusedLevel: FlowStateLevel;
  fusedScore: number;
  churnRisk: number;
  engagementTrend: number;
  uxFriction: number;
  lastUpdate: number;
}

export interface SystemFlowState {
  signals: Map<FlowStateDomain, FlowStateSignal>;
  fusedLevel: FlowStateLevel;
  fusedScore: number;
  activeUsers: number;
  atRiskUsers: number;
  avgResponseQuality: number;
  learningOpportunities: number;
  timestamp: number;
}

// =============================================================================
// DETECTORS (Lightweight JS implementations for browser)
// =============================================================================

/**
 * Simple rolling variance detector (works without WASM)
 */
class VarianceDetector {
  private window: number[];
  private windowSize: number;
  private threshold: number;

  constructor(windowSize = 20, threshold = 2.0) {
    this.window = [];
    this.windowSize = windowSize;
    this.threshold = threshold;
  }

  update(value: number): { variance: number; level: FlowStateLevel; transitioning: boolean } {
    this.window.push(value);
    if (this.window.length > this.windowSize) {
      this.window.shift();
    }

    if (this.window.length < 3) {
      return { variance: 0, level: 'green', transitioning: false };
    }

    const mean = this.window.reduce((a, b) => a + b, 0) / this.window.length;
    const variance = this.window.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / this.window.length;
    const stdDev = Math.sqrt(variance);

    // Detect level based on deviation from baseline
    const recentMean = this.window.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, this.window.length);
    const deviation = Math.abs(recentMean - mean) / (stdDev + 0.001);

    let level: FlowStateLevel = 'green';
    let transitioning = false;

    if (deviation > this.threshold * 2) {
      level = 'red';
      transitioning = true;
    } else if (deviation > this.threshold * 1.5) {
      level = 'orange';
      transitioning = true;
    } else if (deviation > this.threshold) {
      level = 'yellow';
    }

    return { variance, level, transitioning };
  }

  reset() {
    this.window = [];
  }
}

/**
 * Engagement trend detector
 */
class TrendDetector {
  private values: number[];
  private timestamps: number[];
  private windowSize: number;

  constructor(windowSize = 30) {
    this.values = [];
    this.timestamps = [];
    this.windowSize = windowSize;
  }

  update(value: number, timestamp = Date.now()): { trend: number; declining: boolean } {
    this.values.push(value);
    this.timestamps.push(timestamp);

    if (this.values.length > this.windowSize) {
      this.values.shift();
      this.timestamps.shift();
    }

    if (this.values.length < 3) {
      return { trend: 0, declining: false };
    }

    // Simple linear regression
    const n = this.values.length;
    const sumX = this.timestamps.reduce((a, b) => a + b, 0);
    const sumY = this.values.reduce((a, b) => a + b, 0);
    const sumXY = this.timestamps.reduce((sum, x, i) => sum + x * this.values[i], 0);
    const sumX2 = this.timestamps.reduce((sum, x) => sum + x * x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const trend = slope * 1000 * 60 * 60; // Trend per hour

    return { trend, declining: trend < 0 };
  }

  reset() {
    this.values = [];
    this.timestamps = [];
  }
}

// =============================================================================
// FLOW STATE ENGINE
// =============================================================================

export class FlowStateEngine {
  private supabase: SupabaseClient;
  private userStates: Map<string, UserFlowState> = new Map();
  private systemState: SystemFlowState;
  private detectors: {
    engagement: Map<string, VarianceDetector>;
    churn: Map<string, TrendDetector>;
    uxFriction: VarianceDetector;
    systemHealth: VarianceDetector;
    security: VarianceDetector;
    aiQuality: VarianceDetector;
  };

  constructor() {
    this.supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

    this.detectors = {
      engagement: new Map(),
      churn: new Map(),
      uxFriction: new VarianceDetector(20, 1.5),
      systemHealth: new VarianceDetector(50, 2.0),
      security: new VarianceDetector(30, 1.0),
      aiQuality: new VarianceDetector(30, 1.5),
    };

    this.systemState = {
      signals: new Map(),
      fusedLevel: 'green',
      fusedScore: 0,
      activeUsers: 0,
      atRiskUsers: 0,
      avgResponseQuality: 1.0,
      learningOpportunities: 0,
      timestamp: Date.now(),
    };
  }

  // ---------------------------------------------------------------------------
  // USER-LEVEL TRACKING
  // ---------------------------------------------------------------------------

  /**
   * Track user interaction event
   */
  async trackUserEvent(
    userId: string,
    event: {
      type: 'page_view' | 'click' | 'scroll' | 'input' | 'api_call' | 'error' | 'feedback' | 'chat';
      page?: string;
      action?: string;
      duration?: number;
      value?: number;
      metadata?: Record<string, unknown>;
    }
  ): Promise<UserFlowState> {
    // Get or create user state
    let userState = this.userStates.get(userId);
    if (!userState) {
      userState = this.initUserState(userId);
      this.userStates.set(userId, userState);
    }

    // Get or create detectors for user
    if (!this.detectors.engagement.has(userId)) {
      this.detectors.engagement.set(userId, new VarianceDetector(20, 1.5));
    }
    if (!this.detectors.churn.has(userId)) {
      this.detectors.churn.set(userId, new TrendDetector(30));
    }

    const engagementDetector = this.detectors.engagement.get(userId)!;
    const churnDetector = this.detectors.churn.get(userId)!;

    // Calculate engagement score based on event type
    const engagementScore = this.calculateEngagementScore(event);

    // Update detectors
    const engagementResult = engagementDetector.update(engagementScore);
    const churnResult = churnDetector.update(engagementScore);

    // Update user signals
    userState.signals.set('user_engagement', {
      domain: 'user_engagement',
      level: engagementResult.level,
      levelNumeric: this.levelToNumeric(engagementResult.level),
      transitioning: engagementResult.transitioning,
      confidence: Math.min(1, userState.signals.get('user_engagement')?.confidence || 0 + 0.05),
      variance: engagementResult.variance,
      trend: churnResult.trend,
      timestamp: Date.now(),
    });

    userState.signals.set('user_churn', {
      domain: 'user_churn',
      level: churnResult.declining ? (churnResult.trend < -0.5 ? 'red' : 'yellow') : 'green',
      levelNumeric: churnResult.declining ? (churnResult.trend < -0.5 ? 3 : 1) : 0,
      transitioning: churnResult.trend < -1,
      confidence: Math.min(1, userState.signals.get('user_churn')?.confidence || 0 + 0.03),
      variance: engagementResult.variance,
      trend: churnResult.trend,
      timestamp: Date.now(),
    });

    // Update derived metrics
    userState.churnRisk = churnResult.declining ? Math.min(1, Math.abs(churnResult.trend)) : 0;
    userState.engagementTrend = churnResult.trend;
    userState.lastUpdate = Date.now();

    // Fuse all user signals
    this.fuseUserSignals(userState);

    // Log to database (async, don't await)
    this.logUserEvent(userId, event, userState).catch(console.error);

    return userState;
  }

  /**
   * Calculate engagement score from event
   */
  private calculateEngagementScore(event: {
    type: string;
    duration?: number;
    value?: number;
  }): number {
    const baseScores: Record<string, number> = {
      page_view: 1,
      click: 2,
      scroll: 0.5,
      input: 3,
      api_call: 2,
      error: -1,
      feedback: 5,
      chat: 4,
    };

    let score = baseScores[event.type] || 1;

    // Adjust for duration
    if (event.duration) {
      score *= Math.log10(event.duration + 1) / 2;
    }

    // Use explicit value if provided
    if (event.value !== undefined) {
      score = event.value;
    }

    return score;
  }

  // ---------------------------------------------------------------------------
  // SYSTEM-LEVEL TRACKING
  // ---------------------------------------------------------------------------

  /**
   * Track UX friction event (errors, rage clicks, abandonment)
   */
  trackUXFriction(value: number, metadata?: Record<string, unknown>): FlowStateSignal {
    const result = this.detectors.uxFriction.update(value);

    const signal: FlowStateSignal = {
      domain: 'ux_drift',
      level: result.level,
      levelNumeric: this.levelToNumeric(result.level),
      transitioning: result.transitioning,
      confidence: 0.7,
      variance: result.variance,
      trend: 0,
      timestamp: Date.now(),
      metadata,
    };

    this.systemState.signals.set('ux_drift', signal);
    this.fuseSystemSignals();

    return signal;
  }

  /**
   * Track system health metric (latency, error rate, etc.)
   */
  trackSystemHealth(value: number, metadata?: Record<string, unknown>): FlowStateSignal {
    const result = this.detectors.systemHealth.update(value);

    const signal: FlowStateSignal = {
      domain: 'system_health',
      level: result.level,
      levelNumeric: this.levelToNumeric(result.level),
      transitioning: result.transitioning,
      confidence: 0.9,
      variance: result.variance,
      trend: 0,
      timestamp: Date.now(),
      metadata,
    };

    this.systemState.signals.set('system_health', signal);
    this.fuseSystemSignals();

    return signal;
  }

  /**
   * Track security event (blocked requests, suspicious patterns)
   */
  trackSecurityEvent(riskScore: number, metadata?: Record<string, unknown>): FlowStateSignal {
    const result = this.detectors.security.update(riskScore);

    const signal: FlowStateSignal = {
      domain: 'security',
      level: riskScore > 0.8 ? 'red' : riskScore > 0.5 ? 'orange' : riskScore > 0.2 ? 'yellow' : 'green',
      levelNumeric: riskScore > 0.8 ? 3 : riskScore > 0.5 ? 2 : riskScore > 0.2 ? 1 : 0,
      transitioning: result.transitioning,
      confidence: 0.95,
      variance: result.variance,
      trend: 0,
      timestamp: Date.now(),
      metadata,
    };

    this.systemState.signals.set('security', signal);
    this.fuseSystemSignals();

    // Log to Guardian
    const guardian = getElleGuardian();
    guardian.filter({
      type: 'elle_to_db',
      content: `Security event: risk=${riskScore}`,
      metadata: { ...metadata, riskScore },
    }).catch(console.error);

    return signal;
  }

  /**
   * Track AI response quality (user feedback, latency, errors)
   */
  trackAIQuality(
    quality: number,
    metadata?: { latencyMs?: number; feedback?: 'positive' | 'negative'; error?: boolean }
  ): FlowStateSignal {
    const result = this.detectors.aiQuality.update(quality);

    const signal: FlowStateSignal = {
      domain: 'ai_quality',
      level: result.level,
      levelNumeric: this.levelToNumeric(result.level),
      transitioning: result.transitioning,
      confidence: 0.8,
      variance: result.variance,
      trend: 0,
      timestamp: Date.now(),
      metadata,
    };

    this.systemState.signals.set('ai_quality', signal);
    this.systemState.avgResponseQuality = quality;

    // Check for learning opportunity
    if (metadata?.feedback === 'positive' && quality > 0.8) {
      this.systemState.learningOpportunities++;
    }

    this.fuseSystemSignals();

    return signal;
  }

  // ---------------------------------------------------------------------------
  // SIGNAL FUSION (Using WASM when available)
  // ---------------------------------------------------------------------------

  /**
   * Fuse all user signals using GTVC
   */
  private async fuseUserSignals(userState: UserFlowState): Promise<void> {
    const signals = Array.from(userState.signals.values());
    if (signals.length === 0) return;

    const values = signals.map(s => s.levelNumeric);

    // Try WASM fusion first
    try {
      const fusedResult = await fuseSignals(values, 0.05, 0.5, 0.3);
      if (fusedResult) {
        userState.fusedScore = fusedResult.value / 3; // Normalize to 0-1
        userState.fusedLevel = this.numericToLevel(fusedResult.value);
        return;
      }
    } catch {
      // Fallback to simple average
    }

    // Fallback: weighted average
    const weights: Record<FlowStateDomain, number> = {
      user_engagement: 0.3,
      user_churn: 0.4,
      ux_drift: 0.15,
      system_health: 0.05,
      security: 0.05,
      ai_quality: 0.05,
      learning: 0,
      fusion: 0,
    };

    let weightedSum = 0;
    let totalWeight = 0;

    for (const signal of signals) {
      const weight = weights[signal.domain] || 0.1;
      weightedSum += signal.levelNumeric * weight;
      totalWeight += weight;
    }

    const fusedValue = totalWeight > 0 ? weightedSum / totalWeight : 0;
    userState.fusedScore = fusedValue / 3;
    userState.fusedLevel = this.numericToLevel(fusedValue);
  }

  /**
   * Fuse all system signals
   */
  private async fuseSystemSignals(): Promise<void> {
    const signals = Array.from(this.systemState.signals.values());
    if (signals.length === 0) return;

    const values = signals.map(s => s.levelNumeric);

    // Try WASM fusion
    try {
      const fusedResult = await fuseSignals(values, 0.05, 0.5, 0.3);
      if (fusedResult) {
        this.systemState.fusedScore = fusedResult.value / 3;
        this.systemState.fusedLevel = this.numericToLevel(fusedResult.value);
        return;
      }
    } catch {
      // Fallback
    }

    // Fallback: max (conservative approach for system)
    const maxLevel = Math.max(...values);
    this.systemState.fusedScore = maxLevel / 3;
    this.systemState.fusedLevel = this.numericToLevel(maxLevel);
    this.systemState.timestamp = Date.now();
  }

  // ---------------------------------------------------------------------------
  // STATE ACCESS
  // ---------------------------------------------------------------------------

  /**
   * Get current state for a user
   */
  getUserState(userId: string): UserFlowState | undefined {
    return this.userStates.get(userId);
  }

  /**
   * Get all users at risk (churn, engagement issues)
   */
  getAtRiskUsers(): UserFlowState[] {
    return Array.from(this.userStates.values())
      .filter(u => u.churnRisk > 0.5 || u.fusedLevel === 'red' || u.fusedLevel === 'orange');
  }

  /**
   * Get system-wide state
   */
  getSystemState(): SystemFlowState {
    // Update counts
    this.systemState.activeUsers = this.userStates.size;
    this.systemState.atRiskUsers = this.getAtRiskUsers().length;
    return this.systemState;
  }

  /**
   * Get comprehensive analytics for admin dashboard
   */
  async getAnalytics(): Promise<{
    system: SystemFlowState;
    users: {
      total: number;
      atRisk: number;
      churning: number;
      healthy: number;
    };
    signals: FlowStateSignal[];
    recommendations: string[];
  }> {
    const users = Array.from(this.userStates.values());
    const atRisk = users.filter(u => u.fusedLevel === 'orange' || u.fusedLevel === 'red');
    const churning = users.filter(u => u.churnRisk > 0.7);
    const healthy = users.filter(u => u.fusedLevel === 'green');

    const recommendations: string[] = [];

    // Generate recommendations based on signals
    if (atRisk.length > users.length * 0.1) {
      recommendations.push('High proportion of users at risk - consider UX review');
    }
    if (this.systemState.signals.get('ux_drift')?.level === 'red') {
      recommendations.push('Critical UX friction detected - investigate recent changes');
    }
    if (this.systemState.signals.get('ai_quality')?.level !== 'green') {
      recommendations.push('AI response quality degraded - review model performance');
    }
    if (churning.length > 0) {
      recommendations.push(`${churning.length} users showing churn signals - trigger retention campaigns`);
    }
    if (this.systemState.learningOpportunities > 10) {
      recommendations.push(`${this.systemState.learningOpportunities} high-quality interactions available for training`);
    }

    return {
      system: this.getSystemState(),
      users: {
        total: users.length,
        atRisk: atRisk.length,
        churning: churning.length,
        healthy: healthy.length,
      },
      signals: Array.from(this.systemState.signals.values()),
      recommendations,
    };
  }

  // ---------------------------------------------------------------------------
  // HELPERS
  // ---------------------------------------------------------------------------

  private initUserState(userId: string): UserFlowState {
    return {
      userId,
      signals: new Map(),
      fusedLevel: 'green',
      fusedScore: 0,
      churnRisk: 0,
      engagementTrend: 0,
      uxFriction: 0,
      lastUpdate: Date.now(),
    };
  }

  private levelToNumeric(level: FlowStateLevel): number {
    const map: Record<FlowStateLevel, number> = { green: 0, yellow: 1, orange: 2, red: 3 };
    return map[level] ?? 0;
  }

  private numericToLevel(value: number): FlowStateLevel {
    if (value >= 2.5) return 'red';
    if (value >= 1.5) return 'orange';
    if (value >= 0.5) return 'yellow';
    return 'green';
  }

  private async logUserEvent(
    userId: string,
    event: Record<string, unknown>,
    state: UserFlowState
  ): Promise<void> {
    try {
      await this.supabase.from('flowstate_events').insert({
        user_id: userId,
        event_type: event.type,
        event_data: event,
        fused_level: state.fusedLevel,
        fused_score: state.fusedScore,
        churn_risk: state.churnRisk,
        engagement_trend: state.engagementTrend,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error('[FlowState] Failed to log event:', error);
    }
  }
}

// =============================================================================
// SINGLETON
// =============================================================================

let engineInstance: FlowStateEngine | null = null;

export function getFlowStateEngine(): FlowStateEngine {
  if (!engineInstance) {
    engineInstance = new FlowStateEngine();
  }
  return engineInstance;
}

// =============================================================================
// DATABASE SCHEMA
// =============================================================================

export const FLOWSTATE_SCHEMA_SQL = `
-- FlowState Events Log
CREATE TABLE IF NOT EXISTS flowstate_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  event_data JSONB DEFAULT '{}',
  fused_level TEXT,
  fused_score FLOAT,
  churn_risk FLOAT,
  engagement_trend FLOAT,
  timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_flowstate_events_user ON flowstate_events(user_id);
CREATE INDEX IF NOT EXISTS idx_flowstate_events_type ON flowstate_events(event_type);
CREATE INDEX IF NOT EXISTS idx_flowstate_events_level ON flowstate_events(fused_level);
CREATE INDEX IF NOT EXISTS idx_flowstate_events_timestamp ON flowstate_events(timestamp DESC);

-- Enable RLS
ALTER TABLE flowstate_events ENABLE ROW LEVEL SECURITY;

-- Admin-only access
CREATE POLICY flowstate_events_admin_policy ON flowstate_events
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
  );

-- Service role bypass
CREATE POLICY flowstate_events_service_policy ON flowstate_events
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');
`;
