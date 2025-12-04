/**
 * PagerDuty Integration Library
 *
 * Handles event triggering and incident management for PagerDuty integration.
 * Uses PagerDuty's Events API v2.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface PagerDutyConfig {
  routingKey: string; // Integration key from PagerDuty service
}

export interface PagerDutyEvent {
  routing_key: string;
  event_action: 'trigger' | 'acknowledge' | 'resolve';
  dedup_key?: string;
  payload: {
    summary: string;
    source: string;
    severity: 'critical' | 'error' | 'warning' | 'info';
    timestamp?: string;
    component?: string;
    group?: string;
    class?: string;
    custom_details?: Record<string, unknown>;
  };
  images?: Array<{
    src: string;
    href?: string;
    alt?: string;
  }>;
  links?: Array<{
    href: string;
    text: string;
  }>;
  client?: string;
  client_url?: string;
}

export interface PagerDutyResponse {
  status: string;
  message: string;
  dedup_key: string;
}

export interface AlertPayload {
  type: 'breaking_news' | 'risk_change' | 'daily_digest' | 'custom';
  severity: 'low' | 'moderate' | 'elevated' | 'high' | 'critical';
  title: string;
  summary: string;
  category?: string;
  region?: string;
  details?: Record<string, string>;
  link?: string;
}

// =============================================================================
// CONFIG
// =============================================================================

export function getPagerDutyConfig(): PagerDutyConfig | null {
  const routingKey = process.env.PAGERDUTY_ROUTING_KEY;

  if (!routingKey) {
    return null;
  }

  return { routingKey };
}

export function isPagerDutyConfigured(): boolean {
  return getPagerDutyConfig() !== null;
}

// =============================================================================
// EVENT API
// =============================================================================

const PAGERDUTY_EVENTS_URL = 'https://events.pagerduty.com/v2/enqueue';

/**
 * Send an event to PagerDuty
 */
export async function sendPagerDutyEvent(
  routingKey: string,
  event: Omit<PagerDutyEvent, 'routing_key'>
): Promise<{ ok: boolean; dedupKey?: string; error?: string }> {
  try {
    const response = await fetch(PAGERDUTY_EVENTS_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...event,
        routing_key: routingKey,
        client: 'LatticeForge',
        client_url: 'https://latticeforge.com',
      }),
    });

    if (response.ok) {
      const data = (await response.json()) as PagerDutyResponse;
      return { ok: true, dedupKey: data.dedup_key };
    }

    const text = await response.text();
    return { ok: false, error: text };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Trigger an incident
 */
export async function triggerIncident(
  routingKey: string,
  summary: string,
  severity: PagerDutyEvent['payload']['severity'],
  options?: {
    dedupKey?: string;
    component?: string;
    group?: string;
    customDetails?: Record<string, unknown>;
    link?: string;
  }
): Promise<{ ok: boolean; dedupKey?: string; error?: string }> {
  const links = options?.link
    ? [{ href: options.link, text: 'View in LatticeForge' }]
    : undefined;

  return sendPagerDutyEvent(routingKey, {
    event_action: 'trigger',
    dedup_key: options?.dedupKey,
    payload: {
      summary,
      source: 'LatticeForge Intel',
      severity,
      timestamp: new Date().toISOString(),
      component: options?.component,
      group: options?.group,
      custom_details: options?.customDetails,
    },
    links,
  });
}

/**
 * Acknowledge an incident
 */
export async function acknowledgeIncident(
  routingKey: string,
  dedupKey: string
): Promise<{ ok: boolean; error?: string }> {
  return sendPagerDutyEvent(routingKey, {
    event_action: 'acknowledge',
    dedup_key: dedupKey,
    payload: {
      summary: 'Incident acknowledged',
      source: 'LatticeForge Intel',
      severity: 'info',
    },
  });
}

/**
 * Resolve an incident
 */
export async function resolveIncident(
  routingKey: string,
  dedupKey: string
): Promise<{ ok: boolean; error?: string }> {
  return sendPagerDutyEvent(routingKey, {
    event_action: 'resolve',
    dedup_key: dedupKey,
    payload: {
      summary: 'Incident resolved',
      source: 'LatticeForge Intel',
      severity: 'info',
    },
  });
}

// =============================================================================
// ALERT FORMATTING
// =============================================================================

const SEVERITY_MAP: Record<AlertPayload['severity'], PagerDutyEvent['payload']['severity']> = {
  low: 'info',
  moderate: 'info',
  elevated: 'warning',
  high: 'error',
  critical: 'critical',
};

/**
 * Format an alert for PagerDuty
 */
export function formatAlertForPagerDuty(
  alert: AlertPayload,
  routingKey: string
): Omit<PagerDutyEvent, 'routing_key'> {
  const pdSeverity = SEVERITY_MAP[alert.severity];

  const customDetails: Record<string, unknown> = {
    type: alert.type,
    original_severity: alert.severity,
  };

  if (alert.category) {
    customDetails.category = alert.category;
  }
  if (alert.region) {
    customDetails.region = alert.region;
  }
  if (alert.details) {
    Object.assign(customDetails, alert.details);
  }

  const links = alert.link
    ? [{ href: alert.link, text: 'View in LatticeForge' }]
    : undefined;

  return {
    event_action: 'trigger',
    dedup_key: `latticeforge-${alert.type}-${Date.now()}`,
    payload: {
      summary: `[${alert.severity.toUpperCase()}] ${alert.title}: ${alert.summary}`,
      source: 'LatticeForge Intel',
      severity: pdSeverity,
      timestamp: new Date().toISOString(),
      component: alert.category,
      group: alert.region,
      custom_details: customDetails,
    },
    links,
  };
}

/**
 * Create a test event for PagerDuty
 */
export function createTestEvent(): Omit<PagerDutyEvent, 'routing_key'> {
  return {
    event_action: 'trigger',
    dedup_key: `latticeforge-test-${Date.now()}`,
    payload: {
      summary: '[TEST] LatticeForge Integration Test - This is a test alert',
      source: 'LatticeForge Intel',
      severity: 'info',
      timestamp: new Date().toISOString(),
      custom_details: {
        message: 'Your PagerDuty integration is working correctly.',
        test: true,
      },
    },
    links: [
      {
        href: 'https://latticeforge.com/app/settings',
        text: 'Manage Settings',
      },
    ],
  };
}
