/**
 * Microsoft Teams Integration Library
 *
 * Handles webhook message sending for Microsoft Teams integration.
 * Uses Teams Incoming Webhook connector.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface TeamsConfig {
  webhookUrl: string;
}

export interface TeamsMessageCard {
  '@type': 'MessageCard';
  '@context': 'http://schema.org/extensions';
  themeColor: string;
  summary: string;
  sections: TeamsSection[];
  potentialAction?: TeamsAction[];
}

export interface TeamsSection {
  activityTitle?: string;
  activitySubtitle?: string;
  activityImage?: string;
  facts?: Array<{
    name: string;
    value: string;
  }>;
  markdown?: boolean;
  text?: string;
}

export interface TeamsAction {
  '@type': 'OpenUri' | 'HttpPOST' | 'ActionCard';
  name: string;
  targets?: Array<{
    os: 'default' | 'iOS' | 'android' | 'windows';
    uri: string;
  }>;
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

export function getTeamsConfig(): TeamsConfig | null {
  const webhookUrl = process.env.TEAMS_WEBHOOK_URL;

  if (!webhookUrl) {
    return null;
  }

  return { webhookUrl };
}

export function isTeamsConfigured(): boolean {
  return getTeamsConfig() !== null;
}

// =============================================================================
// MESSAGING
// =============================================================================

/**
 * Send a message card to Microsoft Teams via webhook
 */
export async function sendTeamsMessage(
  webhookUrl: string,
  card: TeamsMessageCard
): Promise<{ ok: boolean; error?: string }> {
  try {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(card),
    });

    // Teams returns "1" on success
    if (response.ok) {
      const text = await response.text();
      if (text === '1') {
        return { ok: true };
      }
      return { ok: false, error: text };
    }

    const text = await response.text();
    return { ok: false, error: text };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

// =============================================================================
// ALERT FORMATTING
// =============================================================================

const SEVERITY_COLORS: Record<AlertPayload['severity'], string> = {
  low: '36a64f',      // Green
  moderate: '2196f3', // Blue
  elevated: 'ffeb3b', // Yellow
  high: 'ff9800',     // Orange
  critical: 'f44336', // Red
};

const SEVERITY_EMOJI: Record<AlertPayload['severity'], string> = {
  low: '✅',
  moderate: 'ℹ️',
  elevated: '⚠️',
  high: '🚨',
  critical: '🔥',
};

/**
 * Format an alert into Teams MessageCard
 */
export function formatAlertForTeams(alert: AlertPayload): TeamsMessageCard {
  const emoji = SEVERITY_EMOJI[alert.severity];
  const color = SEVERITY_COLORS[alert.severity];

  const facts: TeamsSection['facts'] = [];

  if (alert.category) {
    facts.push({ name: 'Category', value: alert.category });
  }
  if (alert.region) {
    facts.push({ name: 'Region', value: alert.region });
  }
  facts.push({ name: 'Severity', value: `${emoji} ${alert.severity.toUpperCase()}` });
  facts.push({ name: 'Time', value: new Date().toLocaleString() });

  if (alert.details) {
    for (const [key, value] of Object.entries(alert.details)) {
      facts.push({ name: key, value });
    }
  }

  const sections: TeamsSection[] = [
    {
      activityTitle: `${emoji} ${alert.title}`,
      activitySubtitle: 'LatticeForge Intel',
      activityImage: 'https://latticeforge.com/images/brand/monogram.png',
      facts,
      markdown: true,
    },
    {
      text: alert.summary,
      markdown: true,
    },
  ];

  const potentialAction: TeamsAction[] = [];
  if (alert.link) {
    potentialAction.push({
      '@type': 'OpenUri',
      name: 'View in LatticeForge',
      targets: [{ os: 'default', uri: alert.link }],
    });
  }

  return {
    '@type': 'MessageCard',
    '@context': 'http://schema.org/extensions',
    themeColor: color,
    summary: `${emoji} ${alert.title}`,
    sections,
    potentialAction: potentialAction.length > 0 ? potentialAction : undefined,
  };
}

/**
 * Format daily digest for Teams
 */
export function formatDailyDigestForTeams(
  date: string,
  summaries: Array<{ category: string; summary: string; severity: AlertPayload['severity'] }>
): TeamsMessageCard {
  const sections: TeamsSection[] = [
    {
      activityTitle: `📰 Daily Intel Digest - ${date}`,
      activitySubtitle: 'LatticeForge',
      activityImage: 'https://latticeforge.com/images/brand/monogram.png',
      markdown: true,
    },
  ];

  for (const item of summaries) {
    const emoji = SEVERITY_EMOJI[item.severity];
    sections.push({
      activityTitle: `${emoji} ${item.category}`,
      text: item.summary,
      markdown: true,
    });
  }

  return {
    '@type': 'MessageCard',
    '@context': 'http://schema.org/extensions',
    themeColor: '3b82f6',
    summary: `Daily Intel Digest - ${date}`,
    sections,
    potentialAction: [
      {
        '@type': 'OpenUri',
        name: 'View Dashboard',
        targets: [{ os: 'default', uri: 'https://latticeforge.com/app' }],
      },
    ],
  };
}

/**
 * Create a test message for Teams
 */
export function createTestMessage(): TeamsMessageCard {
  return {
    '@type': 'MessageCard',
    '@context': 'http://schema.org/extensions',
    themeColor: '36a64f',
    summary: 'LatticeForge Integration Test',
    sections: [
      {
        activityTitle: '✅ LatticeForge Integration Test',
        activitySubtitle: 'Integration working correctly',
        activityImage: 'https://latticeforge.com/images/brand/monogram.png',
        text: 'Your Microsoft Teams integration is working correctly. You will receive alerts in this channel based on your configured preferences.',
        markdown: true,
        facts: [
          { name: 'Status', value: 'Connected' },
          { name: 'Time', value: new Date().toLocaleString() },
        ],
      },
    ],
    potentialAction: [
      {
        '@type': 'OpenUri',
        name: 'Manage Settings',
        targets: [{ os: 'default', uri: 'https://latticeforge.com/app/settings' }],
      },
    ],
  };
}
