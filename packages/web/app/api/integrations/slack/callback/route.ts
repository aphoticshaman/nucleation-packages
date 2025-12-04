import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { exchangeSlackCode, isSlackConfigured } from '@/lib/integrations/slack';

/**
 * GET /api/integrations/slack/callback
 *
 * Handles the OAuth callback from Slack. Exchanges the authorization code
 * for an access token and saves the integration to the database.
 */
export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const code = url.searchParams.get('code');
    const state = url.searchParams.get('state');
    const error = url.searchParams.get('error');

    // Base URL for redirects
    const baseUrl = process.env.VERCEL_URL
      ? `https://${process.env.VERCEL_URL}`
      : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

    // Handle user denial
    if (error) {
      console.log('Slack OAuth denied by user:', error);
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=denied`
      );
    }

    // Validate required params
    if (!code || !state) {
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=missing_params`
      );
    }

    // Check Slack configuration
    if (!isSlackConfigured()) {
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=not_configured`
      );
    }

    // Verify state token from cookie
    const cookieStore = await cookies();
    const storedState = cookieStore.get('slack_oauth_state')?.value;

    if (!storedState || storedState !== state) {
      console.error('Slack OAuth state mismatch', { storedState, receivedState: state });
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=invalid_state`
      );
    }

    // Parse state to get org and user info
    let stateData: { orgId: string; userId: string; timestamp: number };
    try {
      stateData = JSON.parse(Buffer.from(state, 'base64url').toString());
    } catch {
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=invalid_state`
      );
    }

    // Check state isn't too old (10 minutes)
    if (Date.now() - stateData.timestamp > 10 * 60 * 1000) {
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=expired_state`
      );
    }

    // Exchange code for access token
    const tokenResponse = await exchangeSlackCode(code);

    if (!tokenResponse.ok || tokenResponse.error) {
      console.error('Slack token exchange failed:', tokenResponse.error);
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=token_exchange`
      );
    }

    // Set up Supabase client
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

    // Verify user is still authenticated and has access to the org
    const { data: { user } } = await supabase.auth.getUser();

    if (!user || user.id !== stateData.userId) {
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=unauthorized`
      );
    }

    // Prepare integration data
    const integrationData = {
      organization_id: stateData.orgId,
      provider: 'slack' as const,
      provider_config: {
        access_token: tokenResponse.access_token,
        token_type: tokenResponse.token_type,
        scope: tokenResponse.scope,
        bot_user_id: tokenResponse.bot_user_id,
        app_id: tokenResponse.app_id,
        team: tokenResponse.team,
        webhook: tokenResponse.incoming_webhook,
      },
      alert_config: {
        // Default alert configuration
        enabled: true,
        channel_id: tokenResponse.incoming_webhook?.channel_id || null,
        channel_name: tokenResponse.incoming_webhook?.channel || null,
        severities: ['high', 'critical'], // Only high/critical by default
        categories: [], // All categories
        daily_digest: false,
        digest_time: '09:00',
      },
      enabled: true,
      created_by: user.id,
    };

    // Upsert integration (update if exists, insert if not)
    const { error: upsertError } = await supabase
      .from('integrations')
      .upsert(
        integrationData,
        { onConflict: 'organization_id,provider' }
      );

    if (upsertError) {
      console.error('Failed to save Slack integration:', upsertError);
      return NextResponse.redirect(
        `${baseUrl}/app/settings?tab=integrations&slack=error&reason=save_failed`
      );
    }

    // Clear the state cookie
    const response = NextResponse.redirect(
      `${baseUrl}/app/settings?tab=integrations&slack=success`
    );
    response.cookies.delete('slack_oauth_state');

    return response;
  } catch (error) {
    console.error('Slack callback error:', error);
    const baseUrl = process.env.VERCEL_URL
      ? `https://${process.env.VERCEL_URL}`
      : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

    return NextResponse.redirect(
      `${baseUrl}/app/settings?tab=integrations&slack=error&reason=unknown`
    );
  }
}
