/**
 * Custom Webhook Integration Library
 *
 * Handles sending alerts to custom HTTP endpoints (Zapier, IFTTT, etc.)
 */

// =============================================================================
// TYPES
// =============================================================================

export interface WebhookConfig {
  url: string;
  secret?: string;
  headers?: Record<string, string>;
}

export interface WebhookPayload {
  event: string;
  timestamp: string;
  data: AlertPayload;
  signature?: string;
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
// SIGNATURE
// =============================================================================

/**
 * Generate HMAC signature for webhook payload
 */
export async function generateSignature(
  payload: string,
  secret: string
): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );

  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(payload));
  const hashArray = Array.from(new Uint8Array(signature));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Verify webhook signature
 */
export async function verifySignature(
  payload: string,
  signature: string,
  secret: string
): Promise<boolean> {
  const expected = await generateSignature(payload, secret);
  return signature === expected;
}

// =============================================================================
// SENDING
// =============================================================================

/**
 * Send alert to custom webhook
 */
export async function sendWebhook(
  config: WebhookConfig,
  alert: AlertPayload
): Promise<{ ok: boolean; statusCode?: number; error?: string }> {
  try {
    const payload: WebhookPayload = {
      event: `latticeforge.alert.${alert.type}`,
      timestamp: new Date().toISOString(),
      data: alert,
    };

    const bodyString = JSON.stringify(payload);

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'User-Agent': 'LatticeForge-Webhook/1.0',
      ...config.headers,
    };

    // Add signature if secret is configured
    if (config.secret) {
      const signature = await generateSignature(bodyString, config.secret);
      headers['X-LatticeForge-Signature'] = `sha256=${signature}`;
    }

    const response = await fetch(config.url, {
      method: 'POST',
      headers,
      body: bodyString,
    });

    if (response.ok) {
      return { ok: true, statusCode: response.status };
    }

    const text = await response.text();
    return { ok: false, statusCode: response.status, error: text };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Send test webhook
 */
export async function sendTestWebhook(
  config: WebhookConfig
): Promise<{ ok: boolean; statusCode?: number; error?: string }> {
  const testAlert: AlertPayload = {
    type: 'custom',
    severity: 'low',
    title: 'LatticeForge Integration Test',
    summary: 'Your webhook integration is working correctly. You will receive alerts at this endpoint based on your configured preferences.',
    category: 'Test',
    region: 'Global',
    details: {
      test: 'true',
      timestamp: new Date().toISOString(),
    },
    link: 'https://latticeforge.com/app/settings',
  };

  return sendWebhook(config, testAlert);
}

// =============================================================================
// ZAPIER TEMPLATES
// =============================================================================

export interface ZapierTemplate {
  id: string;
  name: string;
  description: string;
  triggerApp: string;
  actionApp: string;
  zapierUrl: string;
}

export const ZAPIER_TEMPLATES: ZapierTemplate[] = [
  {
    id: 'slack-alert',
    name: 'Send to Slack',
    description: 'Post LatticeForge alerts to a Slack channel',
    triggerApp: 'LatticeForge',
    actionApp: 'Slack',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-slack',
  },
  {
    id: 'email-alert',
    name: 'Send Email',
    description: 'Get email notifications for critical alerts',
    triggerApp: 'LatticeForge',
    actionApp: 'Gmail',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-email',
  },
  {
    id: 'sheets-log',
    name: 'Log to Sheets',
    description: 'Create a log of all alerts in Google Sheets',
    triggerApp: 'LatticeForge',
    actionApp: 'Google Sheets',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-sheets',
  },
  {
    id: 'notion-page',
    name: 'Create Notion Page',
    description: 'Create a Notion page for each critical alert',
    triggerApp: 'LatticeForge',
    actionApp: 'Notion',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-notion',
  },
  {
    id: 'airtable-record',
    name: 'Add to Airtable',
    description: 'Create Airtable records from alerts',
    triggerApp: 'LatticeForge',
    actionApp: 'Airtable',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-airtable',
  },
  {
    id: 'trello-card',
    name: 'Create Trello Card',
    description: 'Create Trello cards for action items',
    triggerApp: 'LatticeForge',
    actionApp: 'Trello',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-trello',
  },
  {
    id: 'jira-ticket',
    name: 'Create Jira Ticket',
    description: 'Create Jira tickets for investigation',
    triggerApp: 'LatticeForge',
    actionApp: 'Jira',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-jira',
  },
  {
    id: 'sms-twilio',
    name: 'Send SMS',
    description: 'Get SMS alerts via Twilio',
    triggerApp: 'LatticeForge',
    actionApp: 'Twilio',
    zapierUrl: 'https://zapier.com/app/editor/template/latticeforge-sms',
  },
];

// =============================================================================
// IFTTT SUPPORT
// =============================================================================

export interface IFTTTConfig {
  webhookKey: string;
  eventName: string;
}

/**
 * Send alert to IFTTT Webhooks
 */
export async function sendIFTTT(
  config: IFTTTConfig,
  alert: AlertPayload
): Promise<{ ok: boolean; error?: string }> {
  const url = `https://maker.ifttt.com/trigger/${config.eventName}/with/key/${config.webhookKey}`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        value1: `[${alert.severity.toUpperCase()}] ${alert.title}`,
        value2: alert.summary,
        value3: alert.link || 'https://latticeforge.com/app',
      }),
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
