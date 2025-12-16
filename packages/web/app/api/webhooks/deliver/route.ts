import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { mapUserTierToPricing, TIER_CAPABILITIES } from '@/lib/doctrine/types';

export const runtime = 'edge';

/**
 * POST /api/webhooks/deliver - Queue webhook delivery
 * Requires: Integrated tier or higher (webhook_access)
 *
 * This is a stub for the webhook delivery system.
 * Full implementation would use a queue (e.g., Upstash QStash) for reliable delivery.
 */
export async function POST(req: Request) {
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].webhook_access) {
    return NextResponse.json(
      { error: 'Webhook delivery requires Integrated tier or higher' },
      { status: 403 }
    );
  }

  try {
    const body = await req.json();
    const { webhook_url, event_type, payload } = body as {
      webhook_url: string;
      event_type: string;
      payload: Record<string, unknown>;
    };

    if (!webhook_url || !event_type) {
      return NextResponse.json(
        { error: 'webhook_url and event_type are required' },
        { status: 400 }
      );
    }

    // Stub: Log the webhook request
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );

    const webhookRecord = {
      type: 'webhook_queued',
      timestamp: new Date().toISOString(),
      session_hash: 'webhook_delivery',
      user_tier: userTier,
      domain: 'system',
      data: {
        webhook_url,
        event_type,
        payload_size: JSON.stringify(payload || {}).length,
        status: 'queued',
      },
      metadata: {
        note: 'Webhook delivery stub - full implementation requires queue service',
      },
    };

    await supabase.from('learning_events').insert(webhookRecord);

    return NextResponse.json({
      status: 'queued',
      webhook_id: `wh-${Date.now()}`,
      message: 'Webhook queued for delivery',
      note: 'Full webhook delivery requires queue service integration (e.g., QStash)',
    });
  } catch (error) {
    console.error('Webhook delivery error:', error);
    return NextResponse.json(
      { error: 'Failed to queue webhook' },
      { status: 500 }
    );
  }
}
