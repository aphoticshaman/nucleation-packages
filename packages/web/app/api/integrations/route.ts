import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { isSlackConfigured } from '@/lib/integrations/slack';

/**
 * GET /api/integrations
 *
 * Returns all integrations for the user's organization and available providers.
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
      return NextResponse.json({
        integrations: [],
        availableProviders: getAvailableProviders(),
      });
    }

    // Get all integrations for org
    const { data: integrations, error } = await supabase
      .from('integrations')
      .select('id, provider, enabled, alert_config, last_sync_at, created_at')
      .eq('organization_id', profile.organization_id);

    if (error) {
      console.error('Failed to fetch integrations:', error);
      return NextResponse.json(
        { error: 'Failed to fetch integrations' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      integrations: integrations || [],
      availableProviders: getAvailableProviders(),
    });
  } catch (error) {
    console.error('Integrations fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch integrations' },
      { status: 500 }
    );
  }
}

function getAvailableProviders() {
  return [
    {
      id: 'slack',
      name: 'Slack',
      description: 'Receive alerts and daily digests in Slack channels',
      icon: 'slack',
      configured: isSlackConfigured(),
      comingSoon: false,
    },
    {
      id: 'teams',
      name: 'Microsoft Teams',
      description: 'Get alerts and briefings in Teams channels',
      icon: 'teams',
      configured: false,
      comingSoon: true,
    },
    {
      id: 'webhook',
      name: 'Custom Webhook',
      description: 'Send alerts to any HTTP endpoint',
      icon: 'webhook',
      configured: true, // Webhooks are always "configured"
      comingSoon: true,
    },
    {
      id: 'email',
      name: 'Email Digest',
      description: 'Daily email summaries of key intelligence',
      icon: 'email',
      configured: true,
      comingSoon: true,
    },
  ];
}
