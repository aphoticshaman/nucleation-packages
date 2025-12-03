/**
 * LatticeForge Tiered Alerting System
 *
 * Feature-gated alerts via email (Resend), push (web), and in-app.
 * Respects tier limits to stay within budget.
 *
 * Tier limits:
 * - Free (Explorer): In-app only, no email
 * - Starter (Analyst): 10 emails/day, basic alerts
 * - Pro (Strategist): 50 emails/day, priority alerts
 * - Enterprise (Architect): Unlimited, custom rules
 */

import { UserTier } from '@/lib/config/powerUser';

// ============================================
// Types
// ============================================

export type AlertChannel = 'email' | 'push' | 'in-app' | 'sms' | 'webhook';
export type AlertPriority = 'low' | 'medium' | 'high' | 'critical';
export type AlertCategory =
  | 'security'
  | 'geopolitical'
  | 'economic'
  | 'military'
  | 'humanitarian'
  | 'watchlist'
  | 'system';

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;

  // Trigger conditions
  conditions: AlertCondition[];
  conditionLogic: 'AND' | 'OR';

  // Actions
  channels: AlertChannel[];
  priority: AlertPriority;
  cooldownMinutes: number; // Don't re-alert for same trigger within this window

  // Targeting
  entityIds?: string[];
  regions?: string[];
  categories?: AlertCategory[];

  // Metadata
  createdAt: string;
  lastTriggered?: string;
  triggerCount: number;
}

export interface AlertCondition {
  type: 'keyword' | 'entity' | 'severity' | 'region' | 'category' | 'custom';
  operator: 'contains' | 'equals' | 'greater_than' | 'less_than' | 'regex';
  value: string | number;
  field?: string; // For custom conditions
}

export interface Alert {
  id: string;
  ruleId: string;
  title: string;
  summary: string;
  details: string;
  priority: AlertPriority;
  category: AlertCategory;
  entityIds: string[];
  region?: string;
  sourceUrl?: string;
  createdAt: string;
  readAt?: string;
  dismissed?: boolean;
}

export interface AlertDelivery {
  alertId: string;
  channel: AlertChannel;
  status: 'pending' | 'sent' | 'failed' | 'blocked';
  sentAt?: string;
  error?: string;
}

// ============================================
// Tier Limits
// ============================================

export interface TierAlertLimits {
  emailsPerDay: number;
  pushPerDay: number;
  maxRules: number;
  allowedChannels: AlertChannel[];
  allowedPriorities: AlertPriority[];
  customConditions: boolean;
  webhookIntegration: boolean;
  smsAlerts: boolean;
}

export const TIER_ALERT_LIMITS: Record<UserTier, TierAlertLimits> = {
  explorer: {
    emailsPerDay: 0,
    pushPerDay: 5,
    maxRules: 3,
    allowedChannels: ['in-app'],
    allowedPriorities: ['high', 'critical'],
    customConditions: false,
    webhookIntegration: false,
    smsAlerts: false,
  },
  analyst: {
    emailsPerDay: 10,
    pushPerDay: 25,
    maxRules: 10,
    allowedChannels: ['in-app', 'email', 'push'],
    allowedPriorities: ['medium', 'high', 'critical'],
    customConditions: false,
    webhookIntegration: false,
    smsAlerts: false,
  },
  strategist: {
    emailsPerDay: 50,
    pushPerDay: 100,
    maxRules: 50,
    allowedChannels: ['in-app', 'email', 'push', 'webhook'],
    allowedPriorities: ['low', 'medium', 'high', 'critical'],
    customConditions: true,
    webhookIntegration: true,
    smsAlerts: false,
  },
  architect: {
    emailsPerDay: -1, // Unlimited
    pushPerDay: -1,
    maxRules: -1,
    allowedChannels: ['in-app', 'email', 'push', 'webhook', 'sms'],
    allowedPriorities: ['low', 'medium', 'high', 'critical'],
    customConditions: true,
    webhookIntegration: true,
    smsAlerts: true,
  },
};

// ============================================
// Default Alert Rules (Templates)
// ============================================

export const DEFAULT_ALERT_RULES: Omit<AlertRule, 'id' | 'createdAt' | 'triggerCount'>[] = [
  {
    name: 'Critical Security Events',
    description: 'Major security incidents affecting your watchlist regions or entities',
    enabled: true,
    conditions: [
      { type: 'category', operator: 'equals', value: 'security' },
      { type: 'severity', operator: 'greater_than', value: 4 },
    ],
    conditionLogic: 'AND',
    channels: ['in-app', 'email', 'push'],
    priority: 'critical',
    cooldownMinutes: 60,
    categories: ['security'],
  },
  {
    name: 'Watchlist Entity Activity',
    description: 'Any significant activity involving entities on your watchlist',
    enabled: true,
    conditions: [{ type: 'entity', operator: 'contains', value: 'watchlist' }],
    conditionLogic: 'OR',
    channels: ['in-app', 'email'],
    priority: 'high',
    cooldownMinutes: 120,
  },
  {
    name: 'Military Escalation',
    description: 'Detected military buildups, movements, or confrontations',
    enabled: true,
    conditions: [
      { type: 'category', operator: 'equals', value: 'military' },
      { type: 'keyword', operator: 'regex', value: 'escalat|mobiliz|deploy|strike' },
    ],
    conditionLogic: 'AND',
    channels: ['in-app', 'push'],
    priority: 'high',
    cooldownMinutes: 180,
    categories: ['military'],
  },
  {
    name: 'Economic Shocks',
    description: 'Major economic events: sanctions, market crashes, currency crises',
    enabled: false,
    conditions: [
      { type: 'category', operator: 'equals', value: 'economic' },
      { type: 'keyword', operator: 'regex', value: 'sanction|crash|crisis|default|embargo' },
    ],
    conditionLogic: 'AND',
    channels: ['in-app', 'email'],
    priority: 'medium',
    cooldownMinutes: 240,
    categories: ['economic'],
  },
  {
    name: 'Regional Focus',
    description: 'All significant events in your focus regions',
    enabled: false,
    conditions: [{ type: 'region', operator: 'contains', value: 'focus' }],
    conditionLogic: 'OR',
    channels: ['in-app'],
    priority: 'medium',
    cooldownMinutes: 60,
  },
];

// ============================================
// Alert Manager Class
// ============================================

interface AlertUsage {
  emailsSentToday: number;
  pushSentToday: number;
  lastReset: string; // ISO date string
}

export class AlertManager {
  private tier: UserTier;
  private limits: TierAlertLimits;
  private rules: AlertRule[];
  private alerts: Alert[];
  private usage: AlertUsage;
  private userId: string;

  constructor(userId: string, tier: UserTier) {
    this.userId = userId;
    this.tier = tier;
    this.limits = TIER_ALERT_LIMITS[tier];
    this.rules = [];
    this.alerts = [];
    this.usage = {
      emailsSentToday: 0,
      pushSentToday: 0,
      lastReset: new Date().toISOString().split('T')[0],
    };
  }

  /**
   * Check if user can use a specific channel
   */
  canUseChannel(channel: AlertChannel): boolean {
    return this.limits.allowedChannels.includes(channel);
  }

  /**
   * Check if user has quota remaining for a channel
   */
  hasQuota(channel: AlertChannel): { available: boolean; remaining: number; limit: number } {
    this.checkDailyReset();

    switch (channel) {
      case 'email':
        const emailLimit = this.limits.emailsPerDay;
        return {
          available: emailLimit === -1 || this.usage.emailsSentToday < emailLimit,
          remaining: emailLimit === -1 ? -1 : emailLimit - this.usage.emailsSentToday,
          limit: emailLimit,
        };

      case 'push':
        const pushLimit = this.limits.pushPerDay;
        return {
          available: pushLimit === -1 || this.usage.pushSentToday < pushLimit,
          remaining: pushLimit === -1 ? -1 : pushLimit - this.usage.pushSentToday,
          limit: pushLimit,
        };

      case 'in-app':
        return { available: true, remaining: -1, limit: -1 };

      default:
        return { available: this.canUseChannel(channel), remaining: -1, limit: -1 };
    }
  }

  /**
   * Reset daily counters if needed
   */
  private checkDailyReset(): void {
    const today = new Date().toISOString().split('T')[0];
    if (this.usage.lastReset !== today) {
      this.usage = {
        emailsSentToday: 0,
        pushSentToday: 0,
        lastReset: today,
      };
    }
  }

  /**
   * Add a new alert rule
   */
  addRule(rule: Omit<AlertRule, 'id' | 'createdAt' | 'triggerCount'>): AlertRule | { error: string } {
    // Check rule limit
    if (this.limits.maxRules !== -1 && this.rules.length >= this.limits.maxRules) {
      return { error: `Rule limit reached (${this.limits.maxRules}). Upgrade to add more rules.` };
    }

    // Check custom conditions
    if (!this.limits.customConditions && rule.conditions.some((c) => c.type === 'custom')) {
      return { error: 'Custom conditions require Strategist tier or higher.' };
    }

    // Check channels
    const invalidChannels = rule.channels.filter((c) => !this.canUseChannel(c));
    if (invalidChannels.length > 0) {
      return { error: `Channels not available on your plan: ${invalidChannels.join(', ')}` };
    }

    // Check priority
    if (!this.limits.allowedPriorities.includes(rule.priority)) {
      return { error: `Priority level '${rule.priority}' not available on your plan.` };
    }

    const newRule: AlertRule = {
      ...rule,
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      triggerCount: 0,
    };

    this.rules.push(newRule);
    return newRule;
  }

  /**
   * Process an event and check against rules
   */
  async processEvent(event: {
    title: string;
    content: string;
    category: AlertCategory;
    severity: number;
    entityIds: string[];
    region?: string;
    sourceUrl?: string;
  }): Promise<Alert[]> {
    const triggeredAlerts: Alert[] = [];

    for (const rule of this.rules) {
      if (!rule.enabled) continue;

      // Check cooldown
      if (rule.lastTriggered) {
        const lastTriggered = new Date(rule.lastTriggered).getTime();
        const cooldownMs = rule.cooldownMinutes * 60 * 1000;
        if (Date.now() - lastTriggered < cooldownMs) {
          continue;
        }
      }

      // Evaluate conditions
      if (this.evaluateConditions(rule, event)) {
        const alert = await this.createAndDeliverAlert(rule, event);
        if (alert) {
          triggeredAlerts.push(alert);
        }
      }
    }

    return triggeredAlerts;
  }

  /**
   * Evaluate rule conditions against event
   */
  private evaluateConditions(
    rule: AlertRule,
    event: { content: string; category: AlertCategory; severity: number; entityIds: string[]; region?: string }
  ): boolean {
    const results = rule.conditions.map((condition) => {
      switch (condition.type) {
        case 'category':
          return this.evaluateCondition(event.category, condition);

        case 'severity':
          return this.evaluateCondition(event.severity, condition);

        case 'keyword':
          return this.evaluateCondition(event.content, condition);

        case 'entity':
          return event.entityIds.some((id) => this.evaluateCondition(id, condition));

        case 'region':
          return event.region ? this.evaluateCondition(event.region, condition) : false;

        default:
          return false;
      }
    });

    return rule.conditionLogic === 'AND' ? results.every(Boolean) : results.some(Boolean);
  }

  /**
   * Evaluate a single condition
   */
  private evaluateCondition(value: string | number, condition: AlertCondition): boolean {
    switch (condition.operator) {
      case 'equals':
        return value === condition.value;

      case 'contains':
        return String(value).toLowerCase().includes(String(condition.value).toLowerCase());

      case 'greater_than':
        return Number(value) > Number(condition.value);

      case 'less_than':
        return Number(value) < Number(condition.value);

      case 'regex':
        try {
          return new RegExp(String(condition.value), 'i').test(String(value));
        } catch {
          return false;
        }

      default:
        return false;
    }
  }

  /**
   * Create alert and deliver via configured channels
   */
  private async createAndDeliverAlert(
    rule: AlertRule,
    event: {
      title: string;
      content: string;
      category: AlertCategory;
      entityIds: string[];
      region?: string;
      sourceUrl?: string;
    }
  ): Promise<Alert | null> {
    const alert: Alert = {
      id: crypto.randomUUID(),
      ruleId: rule.id,
      title: event.title,
      summary: event.content.slice(0, 200) + (event.content.length > 200 ? '...' : ''),
      details: event.content,
      priority: rule.priority,
      category: event.category,
      entityIds: event.entityIds,
      region: event.region,
      sourceUrl: event.sourceUrl,
      createdAt: new Date().toISOString(),
    };

    // Update rule
    rule.lastTriggered = alert.createdAt;
    rule.triggerCount++;

    // Deliver via each channel
    for (const channel of rule.channels) {
      await this.deliverAlert(alert, channel);
    }

    this.alerts.push(alert);
    return alert;
  }

  /**
   * Deliver alert via specific channel
   */
  private async deliverAlert(alert: Alert, channel: AlertChannel): Promise<AlertDelivery> {
    const quota = this.hasQuota(channel);

    if (!quota.available) {
      return {
        alertId: alert.id,
        channel,
        status: 'blocked',
        error: `Daily ${channel} quota exceeded (${quota.limit})`,
      };
    }

    try {
      switch (channel) {
        case 'email':
          await this.sendEmail(alert);
          this.usage.emailsSentToday++;
          break;

        case 'push':
          await this.sendPush(alert);
          this.usage.pushSentToday++;
          break;

        case 'in-app':
          // In-app alerts are just stored, no delivery needed
          break;

        case 'webhook':
          await this.sendWebhook(alert);
          break;

        case 'sms':
          await this.sendSms(alert);
          break;
      }

      return {
        alertId: alert.id,
        channel,
        status: 'sent',
        sentAt: new Date().toISOString(),
      };
    } catch (error) {
      return {
        alertId: alert.id,
        channel,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Send email via Resend
   */
  private async sendEmail(alert: Alert): Promise<void> {
    const resendApiKey = process.env.RESEND_API_KEY;
    if (!resendApiKey) {
      throw new Error('Email service not configured');
    }

    // This would be the actual Resend API call
    // For now, just log (actual implementation in API route)
    console.log(`[ALERT EMAIL] ${alert.priority.toUpperCase()}: ${alert.title}`);

    // In production, this would call /api/alerts/send-email
  }

  /**
   * Send push notification
   */
  private async sendPush(alert: Alert): Promise<void> {
    // Web Push API or service worker notification
    if ('Notification' in globalThis && Notification.permission === 'granted') {
      new Notification(`LatticeForge: ${alert.title}`, {
        body: alert.summary,
        tag: alert.id,
        icon: '/icons/alert-icon.png',
      });
    }
  }

  /**
   * Send webhook
   */
  private async sendWebhook(alert: Alert): Promise<void> {
    // Would fetch user's configured webhook URL and POST
    console.log(`[ALERT WEBHOOK] ${alert.priority.toUpperCase()}: ${alert.title}`);
  }

  /**
   * Send SMS (Enterprise only)
   */
  private async sendSms(alert: Alert): Promise<void> {
    // Would integrate with Twilio or similar
    console.log(`[ALERT SMS] ${alert.priority.toUpperCase()}: ${alert.title}`);
  }

  /**
   * Get user's alert history
   */
  getAlerts(options?: { unreadOnly?: boolean; limit?: number }): Alert[] {
    let result = [...this.alerts];

    if (options?.unreadOnly) {
      result = result.filter((a) => !a.readAt);
    }

    result.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    if (options?.limit) {
      result = result.slice(0, options.limit);
    }

    return result;
  }

  /**
   * Mark alert as read
   */
  markRead(alertId: string): void {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.readAt = new Date().toISOString();
    }
  }

  /**
   * Get usage stats
   */
  getUsageStats(): {
    tier: UserTier;
    limits: TierAlertLimits;
    usage: AlertUsage;
    rulesUsed: number;
    unreadAlerts: number;
  } {
    this.checkDailyReset();
    return {
      tier: this.tier,
      limits: this.limits,
      usage: this.usage,
      rulesUsed: this.rules.length,
      unreadAlerts: this.alerts.filter((a) => !a.readAt).length,
    };
  }
}

// ============================================
// Upgrade Prompts
// ============================================

export function getUpgradePrompt(
  currentTier: UserTier,
  attemptedFeature: string
): { message: string; upgradeTo: UserTier } | null {
  const prompts: Record<string, { message: string; upgradeTo: UserTier }> = {
    'email-alerts': {
      message: 'Email alerts are available on Analyst tier and above. Upgrade to get real-time email notifications.',
      upgradeTo: 'analyst',
    },
    'more-rules': {
      message: 'Need more alert rules? Upgrade to create up to 50 custom rules.',
      upgradeTo: 'strategist',
    },
    'custom-conditions': {
      message: 'Custom alert conditions require Strategist tier. Build complex rules tailored to your needs.',
      upgradeTo: 'strategist',
    },
    'webhook-integration': {
      message: 'Webhook integration is available for Strategist tier and above. Connect to your existing tools.',
      upgradeTo: 'strategist',
    },
    'sms-alerts': {
      message: 'SMS alerts are an Enterprise feature. Contact us for Architect tier access.',
      upgradeTo: 'architect',
    },
    'unlimited-alerts': {
      message: 'Hit your daily limit? Upgrade to Architect for unlimited alerts.',
      upgradeTo: 'architect',
    },
  };

  return prompts[attemptedFeature] || null;
}
