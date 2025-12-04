import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get('code');
  const redirect = requestUrl.searchParams.get('redirect') || '/app';
  const error = requestUrl.searchParams.get('error');
  const errorDescription = requestUrl.searchParams.get('error_description');

  // Handle OAuth errors
  if (error) {
    console.error('OAuth error:', error, errorDescription);
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(errorDescription || error)}`, requestUrl.origin)
    );
  }

  if (code) {
    const cookieStore = await cookies();

    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll(cookiesToSet) {
            try {
              cookiesToSet.forEach(({ name, value, options }) => {
                cookieStore.set(name, value, options);
              });
            } catch {
              // The `setAll` method was called from a Server Component.
              // This can be ignored if you have middleware refreshing sessions.
            }
          },
        },
      }
    );

    const { error: exchangeError } = await supabase.auth.exchangeCodeForSession(code);

    if (exchangeError) {
      console.error('Code exchange error:', exchangeError.message);
      return NextResponse.redirect(
        new URL(`/login?error=${encodeURIComponent(exchangeError.message)}`, requestUrl.origin)
      );
    }

    // Ensure profile exists and record sign in
    let userRole = 'consumer';
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        // Check if profile exists, create if not
        const { data: profile } = await supabase
          .from('profiles')
          .select('id, role')
          .eq('id', user.id)
          .single();

        if (!profile) {
          await supabase.from('profiles').insert({
            id: user.id,
            email: user.email,
            full_name: user.user_metadata?.full_name || user.user_metadata?.name || null,
            avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture || null,
            role: 'consumer',
            is_active: true,
          });
        } else {
          // Get user role for redirect
          userRole = profile.role || 'consumer';
        }

        // Record sign in with device info
        const rememberMe = requestUrl.searchParams.get('remember') === 'true';
        const userAgent = request.headers.get('user-agent') || '';
        const forwardedFor = request.headers.get('x-forwarded-for');
        const realIp = request.headers.get('x-real-ip');
        const ipAddress = forwardedFor?.split(',')[0] || realIp || null;

        try {
          await supabase.rpc('record_sign_in', {
            p_user_id: user.id,
            p_device_info: {
              user_agent: userAgent,
              platform: userAgent.includes('Mobile') ? 'mobile' : 'desktop',
            },
            p_ip_address: ipAddress,
            p_remember_me: rememberMe,
          });
        } catch (signInErr) {
          console.error('Failed to record sign in:', signInErr);
          // Don't block login if recording fails
        }
      }
    } catch (err) {
      console.error('Profile ensure error:', err);
      // Continue anyway - profile will be created by trigger or next request
    }

    // Admin users go to admin panel, others go to app (or their intended redirect)
    const finalRedirect = userRole === 'admin' ? '/admin' : redirect;
    return NextResponse.redirect(new URL(finalRedirect, requestUrl.origin));
  }

  // Redirect to the intended destination
  return NextResponse.redirect(new URL(redirect, requestUrl.origin));
}
