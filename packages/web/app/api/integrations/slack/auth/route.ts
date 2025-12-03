import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { getSlackAuthUrl, isSlackConfigured } from '@/lib/integrations/slack';

/**
 * GET /api/integrations/slack/auth
 *
 * Initiates Slack OAuth flow by redirecting user to Slack's authorization page.
 * Generates a state token to prevent CSRF attacks.
 */
export async function GET() {
  try {
    // Check if Slack is configured
    if (!isSlackConfigured()) {
      return NextResponse.json(
        { error: 'Slack integration not configured' },
        { status: 503 }
      );
    }

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
      return NextResponse.json(
        { error: 'No organization found. Please complete account setup first.' },
        { status: 400 }
      );
    }

    // Generate state token (includes org ID for verification in callback)
    const stateData = {
      orgId: profile.organization_id,
      userId: user.id,
      timestamp: Date.now(),
      nonce: crypto.randomUUID(),
    };
    const state = Buffer.from(JSON.stringify(stateData)).toString('base64url');

    // Store state in a short-lived cookie for verification
    const response = NextResponse.redirect(getSlackAuthUrl(state));
    response.cookies.set('slack_oauth_state', state, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 10, // 10 minutes
      path: '/',
    });

    return response;
  } catch (error) {
    console.error('Slack auth error:', error);
    return NextResponse.json(
      { error: 'Failed to initiate Slack authorization' },
      { status: 500 }
    );
  }
}
