import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

/**
 * Alert Email Delivery API
 *
 * Sends alert emails via Resend.
 * Rate-limited by user tier.
 *
 * Requires:
 * - RESEND_API_KEY env var
 * - User authentication
 * - Tier-based quota check
 */

export const runtime = 'edge';

interface EmailRequest {
  alertId: string;
  to: string;
  subject: string;
  preview: string;
  title: string;
  summary: string;
  details: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  sourceUrl?: string;
}

// Tier email limits (daily)
const TIER_LIMITS: Record<string, number> = {
  explorer: 0,
  analyst: 10,
  strategist: 50,
  architect: -1, // Unlimited
};

export async function POST(req: NextRequest) {
  try {
    // Check auth
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Get user tier
    const { data: profile } = await supabase
      .from('profiles')
      .select('tier, email_alerts_sent_today, email_alerts_last_reset')
      .eq('id', user.id)
      .single();

    const tier = profile?.tier || 'explorer';
    const limit = TIER_LIMITS[tier] ?? 0;

    // Check if free tier
    if (limit === 0) {
      return NextResponse.json(
        {
          error: 'Email alerts not available on free plan',
          upgrade: true,
          upgradeTo: 'analyst',
        },
        { status: 403 }
      );
    }

    // Check daily quota
    const today = new Date().toISOString().split('T')[0];
    let sentToday = profile?.email_alerts_sent_today || 0;
    const lastReset = profile?.email_alerts_last_reset;

    // Reset if new day
    if (lastReset !== today) {
      sentToday = 0;
      await supabase
        .from('profiles')
        .update({
          email_alerts_sent_today: 0,
          email_alerts_last_reset: today,
        })
        .eq('id', user.id);
    }

    // Check quota (skip for unlimited)
    if (limit !== -1 && sentToday >= limit) {
      return NextResponse.json(
        {
          error: `Daily email quota exceeded (${limit}/day)`,
          upgrade: true,
          upgradeTo: tier === 'analyst' ? 'strategist' : 'architect',
          remaining: 0,
          limit,
        },
        { status: 429 }
      );
    }

    // Parse request
    const body: EmailRequest = await req.json();

    // Validate
    if (!body.to || !body.subject || !body.title) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // Send via Resend
    const resendKey = process.env.RESEND_API_KEY;
    if (!resendKey) {
      console.error('RESEND_API_KEY not configured');
      return NextResponse.json({ error: 'Email service not configured' }, { status: 500 });
    }

    const emailResponse = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${resendKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'LatticeForge Alerts <alerts@latticeforge.ai>',
        to: [body.to],
        subject: `[${body.priority.toUpperCase()}] ${body.subject}`,
        html: generateEmailHtml(body),
        text: generateEmailText(body),
        tags: [
          { name: 'category', value: body.category },
          { name: 'priority', value: body.priority },
          { name: 'alert_id', value: body.alertId },
        ],
      }),
    });

    if (!emailResponse.ok) {
      const errorData = await emailResponse.json();
      console.error('Resend error:', errorData);
      return NextResponse.json({ error: 'Failed to send email' }, { status: 500 });
    }

    const result = await emailResponse.json();

    // Update quota
    await supabase
      .from('profiles')
      .update({
        email_alerts_sent_today: sentToday + 1,
      })
      .eq('id', user.id);

    // Log for audit
    await supabase.from('alert_logs').insert({
      user_id: user.id,
      alert_id: body.alertId,
      channel: 'email',
      status: 'sent',
      metadata: {
        resend_id: result.id,
        priority: body.priority,
        category: body.category,
      },
    });

    return NextResponse.json({
      success: true,
      messageId: result.id,
      remaining: limit === -1 ? -1 : limit - sentToday - 1,
      limit,
    });
  } catch (error) {
    console.error('Alert email error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

/**
 * Generate HTML email
 */
function generateEmailHtml(data: EmailRequest): string {
  const priorityColors: Record<string, string> = {
    low: '#6B7280',
    medium: '#F59E0B',
    high: '#EF4444',
    critical: '#DC2626',
  };

  const priorityBg: Record<string, string> = {
    low: '#F3F4F6',
    medium: '#FEF3C7',
    high: '#FEE2E2',
    critical: '#FEE2E2',
  };

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${data.subject}</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #111827; color: #E5E7EB;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #111827; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color: #1F2937; border-radius: 8px; overflow: hidden;">

          <!-- Header -->
          <tr>
            <td style="padding: 24px; background-color: ${priorityBg[data.priority]};">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td>
                    <span style="display: inline-block; padding: 4px 12px; background-color: ${priorityColors[data.priority]}; color: white; font-size: 12px; font-weight: 600; border-radius: 4px; text-transform: uppercase;">
                      ${data.priority}
                    </span>
                    <span style="margin-left: 12px; color: #6B7280; font-size: 14px;">
                      ${data.category}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style="padding-top: 12px;">
                    <h1 style="margin: 0; font-size: 20px; font-weight: 600; color: #111827;">
                      ${data.title}
                    </h1>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Content -->
          <tr>
            <td style="padding: 24px;">
              <p style="margin: 0 0 16px 0; font-size: 16px; line-height: 1.6; color: #D1D5DB;">
                ${data.summary}
              </p>

              ${
                data.details && data.details !== data.summary
                  ? `
              <div style="margin-top: 20px; padding: 16px; background-color: #374151; border-radius: 6px;">
                <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #9CA3AF;">
                  ${data.details.slice(0, 500)}${data.details.length > 500 ? '...' : ''}
                </p>
              </div>
              `
                  : ''
              }

              ${
                data.sourceUrl
                  ? `
              <p style="margin-top: 20px;">
                <a href="${data.sourceUrl}" style="color: #60A5FA; text-decoration: none; font-size: 14px;">
                  View full details &rarr;
                </a>
              </p>
              `
                  : ''
              }
            </td>
          </tr>

          <!-- CTA -->
          <tr>
            <td style="padding: 0 24px 24px;">
              <a href="https://latticeforge.ai/dashboard?alert=${data.alertId}"
                 style="display: inline-block; padding: 12px 24px; background-color: #3B82F6; color: white; text-decoration: none; font-weight: 500; border-radius: 6px; font-size: 14px;">
                Open in LatticeForge
              </a>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding: 24px; background-color: #111827; border-top: 1px solid #374151;">
              <p style="margin: 0; font-size: 12px; color: #6B7280;">
                You're receiving this because you have alert notifications enabled.
                <a href="https://latticeforge.ai/settings/alerts" style="color: #60A5FA; text-decoration: none;">
                  Manage preferences
                </a>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
  `.trim();
}

/**
 * Generate plain text email
 */
function generateEmailText(data: EmailRequest): string {
  return `
[${data.priority.toUpperCase()}] ${data.title}

Category: ${data.category}

${data.summary}

${data.details && data.details !== data.summary ? `Details:\n${data.details.slice(0, 500)}${data.details.length > 500 ? '...' : ''}\n` : ''}
${data.sourceUrl ? `Source: ${data.sourceUrl}\n` : ''}
---
View in LatticeForge: https://latticeforge.ai/dashboard?alert=${data.alertId}

Manage alert preferences: https://latticeforge.ai/settings/alerts
  `.trim();
}
