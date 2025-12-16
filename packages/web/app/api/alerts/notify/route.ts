import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { mapUserTierToPricing, TIER_CAPABILITIES } from '@/lib/doctrine/types';

export const runtime = 'edge';

/**
 * POST /api/alerts/notify - Send alert notifications
 * Requires: Analyst tier or higher
 *
 * Stub implementation for alert notification delivery.
 * Full implementation would integrate with:
 * - Email (SendGrid, Resend, etc.)
 * - Push notifications (Firebase, OneSignal)
 * - Slack/Teams webhooks
 * - SMS (Twilio)
 */
export async function POST(req: Request) {
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].api_access) {
    return NextResponse.json(
      { error: 'Alert notifications require Analyst tier or higher' },
      { status: 403 }
    );
  }

  try {
    const body = await req.json();
    const {
      alert_id,
      channels = ['email'],
      recipients,
      message,
    } = body as {
      alert_id: string;
      channels: ('email' | 'push' | 'slack' | 'sms')[];
      recipients: string[];
      message?: string;
    };

    if (!alert_id || !recipients || recipients.length === 0) {
      return NextResponse.json(
        { error: 'alert_id and recipients are required' },
        { status: 400 }
      );
    }

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    // Log notification request
    const notificationRecord = {
      type: 'alert_notification',
      timestamp: new Date().toISOString(),
      session_hash: 'alert_notify',
      user_tier: userTier,
      domain: 'system',
      data: {
        alert_id,
        channels,
        recipient_count: recipients.length,
        status: 'queued',
      },
      metadata: {
        note: 'Alert notification stub - full implementation requires notification service',
      },
    };

    await supabase.from('learning_events').insert(notificationRecord);

    // Stub response for each channel
    const results = channels.map((channel) => ({
      channel,
      status: 'queued',
      message: `${channel} notification queued for ${recipients.length} recipient(s)`,
      note: `Full ${channel} delivery requires service integration`,
    }));

    return NextResponse.json({
      notification_id: `notify-${Date.now()}`,
      alert_id,
      results,
      message: 'Notifications queued',
      implementation_notes: {
        email: 'Integrate SendGrid or Resend',
        push: 'Integrate Firebase Cloud Messaging',
        slack: 'Use Slack Incoming Webhooks',
        sms: 'Integrate Twilio',
      },
    });
  } catch (error) {
    console.error('Alert notification error:', error);
    return NextResponse.json(
      { error: 'Failed to send notification' },
      { status: 500 }
    );
  }
}
