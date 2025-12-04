import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { isSlackConfigured } from '@/lib/integrations/slack';
import { isDiscordConfigured } from '@/lib/integrations/discord';
import { isPagerDutyConfigured } from '@/lib/integrations/pagerduty';
import { isTeamsConfigured } from '@/lib/integrations/teams';

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
      category: 'messaging',
    },
    {
      id: 'discord',
      name: 'Discord',
      description: 'Get alerts in Discord servers via webhooks',
      icon: 'discord',
      configured: isDiscordConfigured(),
      comingSoon: false,
      category: 'messaging',
    },
    {
      id: 'teams',
      name: 'Microsoft Teams',
      description: 'Get alerts and briefings in Teams channels',
      icon: 'teams',
      configured: isTeamsConfigured(),
      comingSoon: false,
      category: 'messaging',
    },
    {
      id: 'pagerduty',
      name: 'PagerDuty',
      description: 'Trigger incidents for critical alerts',
      icon: 'pagerduty',
      configured: isPagerDutyConfigured(),
      comingSoon: false,
      category: 'incident',
    },
    {
      id: 'webhook',
      name: 'Custom Webhook',
      description: 'Send alerts to any HTTP endpoint (Zapier, IFTTT, etc.)',
      icon: 'webhook',
      configured: true,
      comingSoon: false,
      category: 'automation',
    },
    {
      id: 'email',
      name: 'Email Digest',
      description: 'Daily email summaries of key intelligence',
      icon: 'email',
      configured: true,
      comingSoon: false,
      category: 'messaging',
    },
    {
      id: 'opsgenie',
      name: 'Opsgenie',
      description: 'Alert routing and on-call management',
      icon: 'opsgenie',
      configured: false,
      comingSoon: true,
      category: 'incident',
    },
    {
      id: 'jira',
      name: 'Jira',
      description: 'Create tickets from alerts automatically',
      icon: 'jira',
      configured: false,
      comingSoon: true,
      category: 'ticketing',
    },
    {
      id: 'servicenow',
      name: 'ServiceNow',
      description: 'Enterprise IT service management',
      icon: 'servicenow',
      configured: false,
      comingSoon: true,
      category: 'ticketing',
    },
    {
      id: 'splunk',
      name: 'Splunk',
      description: 'Forward alerts to Splunk for analysis',
      icon: 'splunk',
      configured: false,
      comingSoon: true,
      category: 'siem',
    },
    {
      id: 'datadog',
      name: 'Datadog',
      description: 'Send events to Datadog monitoring',
      icon: 'datadog',
      configured: false,
      comingSoon: true,
      category: 'monitoring',
    },
  ];
}
