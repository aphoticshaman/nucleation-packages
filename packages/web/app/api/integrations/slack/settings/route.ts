import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

/**
 * PATCH /api/integrations/slack/settings
 *
 * Updates Slack integration settings (channel, alert preferences, etc.)
 */
export async function PATCH(req: Request) {
  try {
    const body = await req.json();
    const {
      enabled,
      channelId,
      channelName,
      severities,
      categories,
      dailyDigest,
      digestTime,
    } = body;

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

    // Get existing integration
    const { data: integration, error: fetchError } = await supabase
      .from('integrations')
      .select('*')
      .eq('organization_id', profile.organization_id)
      .eq('provider', 'slack')
      .single();

    if (fetchError || !integration) {
      return NextResponse.json(
        { error: 'Slack integration not found' },
        { status: 404 }
      );
    }

    // Build update object
    const updates: Record<string, unknown> = {};

    if (typeof enabled === 'boolean') {
      updates.enabled = enabled;
    }

    // Update alert_config
    const currentAlertConfig = (integration.alert_config as Record<string, unknown>) || {};
    const newAlertConfig = { ...currentAlertConfig };

    if (channelId !== undefined) {
      newAlertConfig.channel_id = channelId;
    }
    if (channelName !== undefined) {
      newAlertConfig.channel_name = channelName;
    }
    if (severities !== undefined) {
      newAlertConfig.severities = severities;
    }
    if (categories !== undefined) {
      newAlertConfig.categories = categories;
    }
    if (typeof dailyDigest === 'boolean') {
      newAlertConfig.daily_digest = dailyDigest;
    }
    if (digestTime !== undefined) {
      newAlertConfig.digest_time = digestTime;
    }

    updates.alert_config = newAlertConfig;
    updates.updated_at = new Date().toISOString();

    // Update integration
    const { error: updateError } = await supabase
      .from('integrations')
      .update(updates)
      .eq('id', integration.id);

    if (updateError) {
      console.error('Failed to update Slack settings:', updateError);
      return NextResponse.json(
        { error: 'Failed to update settings' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Settings updated successfully',
    });
  } catch (error) {
    console.error('Slack settings error:', error);
    return NextResponse.json(
      { error: 'Failed to update settings' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/integrations/slack/settings
 *
 * Disconnects Slack integration
 */
export async function DELETE() {
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

    // Delete integration
    const { error: deleteError } = await supabase
      .from('integrations')
      .delete()
      .eq('organization_id', profile.organization_id)
      .eq('provider', 'slack');

    if (deleteError) {
      console.error('Failed to delete Slack integration:', deleteError);
      return NextResponse.json(
        { error: 'Failed to disconnect Slack' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Slack disconnected successfully',
    });
  } catch (error) {
    console.error('Slack disconnect error:', error);
    return NextResponse.json(
      { error: 'Failed to disconnect Slack' },
      { status: 500 }
    );
  }
}
