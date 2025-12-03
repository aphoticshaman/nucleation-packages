import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { listSlackChannels } from '@/lib/integrations/slack';

/**
 * GET /api/integrations/slack/channels
 *
 * Lists Slack channels the bot can post to.
 */
export async function GET() {
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
      .select('provider_config')
      .eq('organization_id', profile.organization_id)
      .eq('provider', 'slack')
      .single();

    if (integrationError || !integration) {
      return NextResponse.json(
        { error: 'Slack integration not found' },
        { status: 404 }
      );
    }

    const config = integration.provider_config as { access_token?: string };

    if (!config.access_token) {
      return NextResponse.json(
        { error: 'No access token found' },
        { status: 500 }
      );
    }

    // Fetch channels
    const channels = await listSlackChannels(config.access_token);

    return NextResponse.json({
      channels: channels.map((ch) => ({
        id: ch.id,
        name: ch.name,
        isPrivate: ch.is_private,
        isMember: ch.is_member,
      })),
    });
  } catch (error) {
    console.error('Slack channels error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch channels' },
      { status: 500 }
    );
  }
}
