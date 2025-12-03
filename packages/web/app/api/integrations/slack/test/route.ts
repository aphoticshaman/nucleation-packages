import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { sendSlackMessage, sendSlackWebhook, createTestMessage } from '@/lib/integrations/slack';

/**
 * POST /api/integrations/slack/test
 *
 * Sends a test message to the configured Slack channel to verify the integration.
 */
export async function POST() {
  try {
    // Get current user
    const cookieStore = await cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          get(name: string) {
            return cookieStore.get(name)?.value;
          },
          set(name: string, value: string, options: Record<string, unknown>) {
            try {
              cookieStore.set({ name, value, ...options });
            } catch {
              // Server component - can't set cookies
            }
          },
          remove(name: string, options: Record<string, unknown>) {
            try {
              cookieStore.set({ name, value: '', ...options });
            } catch {
              // Server component - can't set cookies
            }
          },
        },
      }
    );

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
    }

    // Get user's organization
    const { data: profile } = await supabase
      .from('profiles')
      .select('organization_id')
      .eq('id', user.id)
      .single();

    if (!profile?.organization_id) {
      return NextResponse.json({ error: 'No organization found' }, { status: 400 });
    }

    // Get Slack integration
    const { data: integration, error: integrationError } = await supabase
      .from('integrations')
      .select('*')
      .eq('organization_id', profile.organization_id)
      .eq('provider', 'slack')
      .single();

    if (integrationError || !integration) {
      return NextResponse.json(
        { error: 'Slack integration not found. Please connect Slack first.' },
        { status: 404 }
      );
    }

    if (!integration.enabled) {
      return NextResponse.json(
        { error: 'Slack integration is disabled' },
        { status: 400 }
      );
    }

    const config = integration.provider_config as {
      access_token?: string;
      webhook?: { url?: string; channel_id?: string };
    };
    const alertConfig = integration.alert_config as {
      channel_id?: string;
    };

    // Create test message
    const testMessage = createTestMessage();

    // Try webhook first (simpler, more reliable)
    if (config.webhook?.url) {
      const result = await sendSlackWebhook(config.webhook.url, testMessage);

      if (result.ok) {
        // Update last_sync_at
        await supabase
          .from('integrations')
          .update({ last_sync_at: new Date().toISOString() })
          .eq('id', integration.id);

        return NextResponse.json({
          success: true,
          method: 'webhook',
          message: 'Test message sent successfully',
        });
      }

      // Fall through to try API if webhook fails
      console.warn('Webhook failed, trying API:', result.error);
    }

    // Try API method
    if (config.access_token) {
      const channelId = alertConfig?.channel_id || config.webhook?.channel_id;

      if (!channelId) {
        return NextResponse.json(
          { error: 'No channel configured. Please select a channel in settings.' },
          { status: 400 }
        );
      }

      const result = await sendSlackMessage(config.access_token, {
        ...testMessage,
        channel: channelId,
      });

      if (result.ok) {
        // Update last_sync_at
        await supabase
          .from('integrations')
          .update({ last_sync_at: new Date().toISOString() })
          .eq('id', integration.id);

        return NextResponse.json({
          success: true,
          method: 'api',
          message: 'Test message sent successfully',
        });
      }

      return NextResponse.json(
        { error: result.error || 'Failed to send message' },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { error: 'No valid Slack credentials found' },
      { status: 500 }
    );
  } catch (error) {
    console.error('Slack test error:', error);
    return NextResponse.json(
      { error: 'Failed to send test message' },
      { status: 500 }
    );
  }
}
