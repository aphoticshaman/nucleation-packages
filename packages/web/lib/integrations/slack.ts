/**
 * Slack Integration Library
 *
 * Handles OAuth, message sending, and channel management for Slack integration.
 * Uses Slack's Web API and Incoming Webhooks.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface SlackConfig {
  clientId: string;
  clientSecret: string;
  signingSecret: string;
}

export interface SlackOAuthResponse {
  ok: boolean;
  access_token: string;
  token_type: string;
  scope: string;
  bot_user_id: string;
  app_id: string;
  team: {
    id: string;
    name: string;
  };
  incoming_webhook?: {
    channel: string;
    channel_id: string;
    configuration_url: string;
    url: string;
  };
  error?: string;
}

export interface SlackChannel {
  id: string;
  name: string;
  is_private: boolean;
  is_member: boolean;
}

export interface SlackMessage {
  channel: string;
  text: string;
  blocks?: SlackBlock[];
  attachments?: SlackAttachment[];
  unfurl_links?: boolean;
  unfurl_media?: boolean;
}

export interface SlackBlock {
  type: 'section' | 'header' | 'divider' | 'context' | 'actions';
  text?: {
    type: 'plain_text' | 'mrkdwn';
    text: string;
    emoji?: boolean;
  };
  fields?: Array<{
    type: 'plain_text' | 'mrkdwn';
    text: string;
  }>;
  accessory?: {
    type: 'button' | 'image';
    text?: { type: 'plain_text'; text: string };
    url?: string;
    action_id?: string;
    image_url?: string;
    alt_text?: string;
  };
  elements?: Array<{
    type: 'plain_text' | 'mrkdwn' | 'image';
    text?: string;
    image_url?: string;
    alt_text?: string;
  }>;
}

export interface SlackAttachment {
  color: string;
  fallback: string;
  title?: string;
  title_link?: string;
  text?: string;
  fields?: Array<{
    title: string;
    value: string;
    short?: boolean;
  }>;
  footer?: string;
  ts?: number;
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

export function getSlackConfig(): SlackConfig | null {
  const clientId = process.env.SLACK_CLIENT_ID;
  const clientSecret = process.env.SLACK_CLIENT_SECRET;
  const signingSecret = process.env.SLACK_SIGNING_SECRET;

  if (!clientId || !clientSecret || !signingSecret) {
    return null;
  }

  return { clientId, clientSecret, signingSecret };
}

export function isSlackConfigured(): boolean {
  return getSlackConfig() !== null;
}

// =============================================================================
// OAUTH
// =============================================================================

/**
 * Generate Slack OAuth URL for user authorization
 */
export function getSlackAuthUrl(state: string): string {
  const config = getSlackConfig();
  if (!config) throw new Error('Slack not configured');

  const baseUrl = process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

  const redirectUri = `${baseUrl}/api/integrations/slack/callback`;

  // Scopes needed:
  // - chat:write - Send messages
  // - channels:read - List public channels
  // - groups:read - List private channels bot is in
  // - incoming-webhook - Get webhook URL during install
  const scopes = [
    'chat:write',
    'channels:read',
    'groups:read',
    'incoming-webhook',
  ].join(',');

  const params = new URLSearchParams({
    client_id: config.clientId,
    scope: scopes,
    redirect_uri: redirectUri,
    state,
  });

  return `https://slack.com/oauth/v2/authorize?${params.toString()}`;
}

/**
 * Exchange OAuth code for access token
 */
export async function exchangeSlackCode(code: string): Promise<SlackOAuthResponse> {
  const config = getSlackConfig();
  if (!config) throw new Error('Slack not configured');

  const baseUrl = process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

  const redirectUri = `${baseUrl}/api/integrations/slack/callback`;

  const response = await fetch('https://slack.com/api/oauth.v2.access', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: config.clientId,
      client_secret: config.clientSecret,
      code,
      redirect_uri: redirectUri,
    }),
  });

  const data = await response.json();
  return data as SlackOAuthResponse;
}

// =============================================================================
// CHANNELS
// =============================================================================

/**
 * List channels the bot can post to
 */
export async function listSlackChannels(accessToken: string): Promise<SlackChannel[]> {
  const channels: SlackChannel[] = [];

  // Get public channels
  const publicResponse = await fetch('https://slack.com/api/conversations.list?types=public_channel&limit=200', {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  const publicData = await publicResponse.json();

  if (publicData.ok && publicData.channels) {
    for (const ch of publicData.channels) {
      channels.push({
        id: ch.id,
        name: ch.name,
        is_private: false,
        is_member: ch.is_member || false,
      });
    }
  }

  // Get private channels bot is in
  const privateResponse = await fetch('https://slack.com/api/conversations.list?types=private_channel&limit=200', {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  const privateData = await privateResponse.json();

  if (privateData.ok && privateData.channels) {
    for (const ch of privateData.channels) {
      channels.push({
        id: ch.id,
        name: ch.name,
        is_private: true,
        is_member: true, // Bot only sees private channels it's in
      });
    }
  }

  return channels;
}

// =============================================================================
// MESSAGING
// =============================================================================

/**
 * Send a message to Slack
 */
export async function sendSlackMessage(
  accessToken: string,
  message: SlackMessage
): Promise<{ ok: boolean; error?: string; ts?: string }> {
  const response = await fetch('https://slack.com/api/chat.postMessage', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify(message),
  });

  return response.json();
}

/**
 * Send via incoming webhook (simpler, doesn't need token refresh)
 */
export async function sendSlackWebhook(
  webhookUrl: string,
  message: Omit<SlackMessage, 'channel'>
): Promise<{ ok: boolean; error?: string }> {
  try {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(message),
    });

    if (response.ok) {
      return { ok: true };
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
  low: '#36a64f',      // Green
  moderate: '#2196f3', // Blue
  elevated: '#ffeb3b', // Yellow
  high: '#ff9800',     // Orange
  critical: '#f44336', // Red
};

const SEVERITY_EMOJI: Record<AlertPayload['severity'], string> = {
  low: ':white_check_mark:',
  moderate: ':information_source:',
  elevated: ':warning:',
  high: ':rotating_light:',
  critical: ':fire:',
};

/**
 * Format an alert into Slack blocks
 */
export function formatAlertForSlack(alert: AlertPayload): SlackMessage {
  const emoji = SEVERITY_EMOJI[alert.severity];
  const color = SEVERITY_COLORS[alert.severity];

  const blocks: SlackBlock[] = [
    {
      type: 'header',
      text: {
        type: 'plain_text',
        text: `${alert.title}`,
        emoji: true,
      },
    },
    {
      type: 'section',
      text: {
        type: 'mrkdwn',
        text: alert.summary,
      },
    },
  ];

  // Add details fields if present
  if (alert.details && Object.keys(alert.details).length > 0) {
    blocks.push({
      type: 'section',
      fields: Object.entries(alert.details).map(([key, value]) => ({
        type: 'mrkdwn' as const,
        text: `*${key}:*\n${value}`,
      })),
    });
  }

  // Add context (category, region, severity)
  const contextElements: Array<{ type: 'mrkdwn'; text: string }> = [];
  if (alert.category) {
    contextElements.push({ type: 'mrkdwn', text: `*Category:* ${alert.category}` });
  }
  if (alert.region) {
    contextElements.push({ type: 'mrkdwn', text: `*Region:* ${alert.region}` });
  }
  contextElements.push({ type: 'mrkdwn', text: `*Severity:* ${emoji} ${alert.severity.toUpperCase()}` });

  blocks.push({
    type: 'context',
    elements: contextElements,
  });

  // Add link button if present
  if (alert.link) {
    blocks.push({
      type: 'section',
      text: {
        type: 'mrkdwn',
        text: ' ',
      },
      accessory: {
        type: 'button',
        text: {
          type: 'plain_text',
          text: 'View in LatticeForge',
        },
        url: alert.link,
        action_id: 'view_alert',
      },
    });
  }

  return {
    channel: '', // Will be set by caller
    text: `${emoji} ${alert.title}: ${alert.summary}`, // Fallback text
    blocks,
    attachments: [
      {
        color,
        fallback: `${alert.title}: ${alert.summary}`,
        footer: 'LatticeForge Intel',
        ts: Math.floor(Date.now() / 1000),
      },
    ],
    unfurl_links: false,
    unfurl_media: false,
  };
}

/**
 * Format daily digest for Slack
 */
export function formatDailyDigestForSlack(
  date: string,
  summaries: Array<{ category: string; summary: string; severity: AlertPayload['severity'] }>
): SlackMessage {
  const blocks: SlackBlock[] = [
    {
      type: 'header',
      text: {
        type: 'plain_text',
        text: `:newspaper: Daily Intel Digest - ${date}`,
        emoji: true,
      },
    },
    {
      type: 'divider',
    },
  ];

  for (const item of summaries) {
    const emoji = SEVERITY_EMOJI[item.severity];
    blocks.push({
      type: 'section',
      text: {
        type: 'mrkdwn',
        text: `*${item.category}* ${emoji}\n${item.summary}`,
      },
    });
  }

  blocks.push({
    type: 'divider',
  });

  blocks.push({
    type: 'context',
    elements: [
      {
        type: 'mrkdwn',
        text: 'Generated by LatticeForge | <https://latticeforge.com/app|View Dashboard>',
      },
    ],
  });

  return {
    channel: '',
    text: `Daily Intel Digest - ${date}`,
    blocks,
    unfurl_links: false,
    unfurl_media: false,
  };
}

/**
 * Send a test message to verify integration
 */
export function createTestMessage(): SlackMessage {
  return {
    channel: '',
    text: ':white_check_mark: LatticeForge integration test successful!',
    blocks: [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: ':white_check_mark: *LatticeForge Integration Test*\n\nYour Slack integration is working correctly. You will receive alerts in this channel based on your configured preferences.',
        },
      },
      {
        type: 'context',
        elements: [
          {
            type: 'mrkdwn',
            text: `Sent at ${new Date().toISOString()} | <https://latticeforge.com/app/settings|Manage Settings>`,
          },
        ],
      },
    ],
  };
}
