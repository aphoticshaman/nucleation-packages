/**
 * Discord Integration Library
 *
 * Handles OAuth, message sending, and channel management for Discord integration.
 * Uses Discord's Webhook API for sending alerts.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface DiscordConfig {
  clientId: string;
  clientSecret: string;
  botToken: string;
}

export interface DiscordOAuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token: string;
  scope: string;
  guild?: {
    id: string;
    name: string;
    icon: string | null;
  };
  webhook?: {
    id: string;
    token: string;
    name: string;
    channel_id: string;
    guild_id: string;
    url: string;
  };
  error?: string;
  error_description?: string;
}

export interface DiscordChannel {
  id: string;
  name: string;
  type: number; // 0 = text, 2 = voice, etc.
  guild_id: string;
  position: number;
}

export interface DiscordEmbed {
  title?: string;
  description?: string;
  url?: string;
  color?: number;
  timestamp?: string;
  footer?: {
    text: string;
    icon_url?: string;
  };
  author?: {
    name: string;
    url?: string;
    icon_url?: string;
  };
  fields?: Array<{
    name: string;
    value: string;
    inline?: boolean;
  }>;
  thumbnail?: { url: string };
  image?: { url: string };
}

export interface DiscordMessage {
  content?: string;
  embeds?: DiscordEmbed[];
  username?: string;
  avatar_url?: string;
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

export function getDiscordConfig(): DiscordConfig | null {
  const clientId = process.env.DISCORD_CLIENT_ID;
  const clientSecret = process.env.DISCORD_CLIENT_SECRET;
  const botToken = process.env.DISCORD_BOT_TOKEN;

  if (!clientId || !clientSecret) {
    return null;
  }

  return { clientId, clientSecret, botToken: botToken || '' };
}

export function isDiscordConfigured(): boolean {
  return getDiscordConfig() !== null;
}

// =============================================================================
// OAUTH
// =============================================================================

/**
 * Generate Discord OAuth URL for user authorization
 */
export function getDiscordAuthUrl(state: string): string {
  const config = getDiscordConfig();
  if (!config) throw new Error('Discord not configured');

  const baseUrl = process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

  const redirectUri = `${baseUrl}/api/integrations/discord/callback`;

  // Scopes needed:
  // - webhook.incoming - Create incoming webhook
  // - guilds - Access user's guilds
  const scopes = ['webhook.incoming', 'guilds'].join(' ');

  const params = new URLSearchParams({
    client_id: config.clientId,
    permissions: '536870912', // Send Messages + Embed Links
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: scopes,
    state,
  });

  return `https://discord.com/api/oauth2/authorize?${params.toString()}`;
}

/**
 * Exchange OAuth code for access token and webhook
 */
export async function exchangeDiscordCode(code: string): Promise<DiscordOAuthResponse> {
  const config = getDiscordConfig();
  if (!config) throw new Error('Discord not configured');

  const baseUrl = process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

  const redirectUri = `${baseUrl}/api/integrations/discord/callback`;

  const response = await fetch('https://discord.com/api/oauth2/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: config.clientId,
      client_secret: config.clientSecret,
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri,
    }),
  });

  const data = await response.json();
  return data as DiscordOAuthResponse;
}

// =============================================================================
// MESSAGING
// =============================================================================

/**
 * Send a message via Discord webhook
 */
export async function sendDiscordWebhook(
  webhookUrl: string,
  message: DiscordMessage
): Promise<{ ok: boolean; error?: string }> {
  try {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...message,
        username: message.username || 'LatticeForge',
        avatar_url: message.avatar_url || 'https://latticeforge.com/images/brand/monogram.png',
      }),
    });

    if (response.ok || response.status === 204) {
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

const SEVERITY_COLORS: Record<AlertPayload['severity'], number> = {
  low: 0x36a64f,      // Green
  moderate: 0x2196f3, // Blue
  elevated: 0xffeb3b, // Yellow
  high: 0xff9800,     // Orange
  critical: 0xf44336, // Red
};

const SEVERITY_EMOJI: Record<AlertPayload['severity'], string> = {
  low: '✅',
  moderate: 'ℹ️',
  elevated: '⚠️',
  high: '🚨',
  critical: '🔥',
};

/**
 * Format an alert into Discord embed
 */
export function formatAlertForDiscord(alert: AlertPayload): DiscordMessage {
  const emoji = SEVERITY_EMOJI[alert.severity];
  const color = SEVERITY_COLORS[alert.severity];

  const fields: DiscordEmbed['fields'] = [];

  if (alert.category) {
    fields.push({ name: 'Category', value: alert.category, inline: true });
  }
  if (alert.region) {
    fields.push({ name: 'Region', value: alert.region, inline: true });
  }
  fields.push({ name: 'Severity', value: `${emoji} ${alert.severity.toUpperCase()}`, inline: true });

  if (alert.details) {
    for (const [key, value] of Object.entries(alert.details)) {
      fields.push({ name: key, value, inline: true });
    }
  }

  const embed: DiscordEmbed = {
    title: `${emoji} ${alert.title}`,
    description: alert.summary,
    color,
    timestamp: new Date().toISOString(),
    footer: {
      text: 'LatticeForge Intel',
      icon_url: 'https://latticeforge.com/images/brand/monogram.png',
    },
    fields,
  };

  if (alert.link) {
    embed.url = alert.link;
  }

  return {
    embeds: [embed],
  };
}

/**
 * Format daily digest for Discord
 */
export function formatDailyDigestForDiscord(
  date: string,
  summaries: Array<{ category: string; summary: string; severity: AlertPayload['severity'] }>
): DiscordMessage {
  const fields: DiscordEmbed['fields'] = summaries.map((item) => ({
    name: `${SEVERITY_EMOJI[item.severity]} ${item.category}`,
    value: item.summary,
    inline: false,
  }));

  return {
    embeds: [
      {
        title: `📰 Daily Intel Digest - ${date}`,
        color: 0x3b82f6,
        fields,
        footer: {
          text: 'LatticeForge',
          icon_url: 'https://latticeforge.com/images/brand/monogram.png',
        },
        timestamp: new Date().toISOString(),
      },
    ],
  };
}

/**
 * Create a test message for Discord
 */
export function createTestMessage(): DiscordMessage {
  return {
    embeds: [
      {
        title: '✅ LatticeForge Integration Test',
        description: 'Your Discord integration is working correctly. You will receive alerts in this channel based on your configured preferences.',
        color: 0x36a64f,
        footer: {
          text: `Sent at ${new Date().toISOString()}`,
        },
      },
    ],
  };
}
