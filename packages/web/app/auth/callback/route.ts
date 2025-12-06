import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';

// Cookie domain for cross-subdomain auth - MUST match lib/auth.ts and middleware.ts
const COOKIE_DOMAIN = '.latticeforge.ai';

// More robust production detection: check VERCEL_ENV OR hostname
function isProductionEnvironment(hostname: string): boolean {
  // VERCEL_ENV is 'production' on production deployments
  if (process.env.VERCEL_ENV === 'production') return true;
  // Fallback: check if hostname is on latticeforge.ai
  if (hostname.endsWith('latticeforge.ai')) return true;
  return false;
}

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

    // Create response upfront so we can set cookies on it
    const response = NextResponse.redirect(new URL(redirect, requestUrl.origin));

    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll(cookiesToSet) {
            // Use hostname-based production check (more reliable than just VERCEL_ENV)
            const isProduction = isProductionEnvironment(requestUrl.hostname);

            // Set cookies on the response with explicit options
            cookiesToSet.forEach(({ name, value, options }) => {
              // Ensure proper cookie options for auth cookies
              // CRITICAL: domain must match lib/auth.ts and middleware.ts
              const cookieOptions: Parameters<typeof response.cookies.set>[2] = {
                path: options?.path || '/',
                sameSite: (options?.sameSite as 'lax' | 'strict' | 'none') || 'lax',
                secure: isProduction,
                httpOnly: options?.httpOnly ?? true,
                maxAge: options?.maxAge,
                ...(isProduction && { domain: COOKIE_DOMAIN }),
              };

              // Set on response (this is what gets sent to browser)
              response.cookies.set(name, value, cookieOptions);

              // Also try to set on cookieStore for downstream RSC
              try {
                cookieStore.set(name, value, cookieOptions);
              } catch {
                // Server Component context - ignore
              }
            });
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

    // Update redirect URL based on role (admin goes to admin panel)
    if (userRole === 'admin') {
      // Use hostname-based production check (more reliable)
      const isProduction = isProductionEnvironment(requestUrl.hostname);

      // Create new response with updated redirect, copy cookies WITH DOMAIN
      const adminResponse = NextResponse.redirect(new URL('/admin', requestUrl.origin));
      response.cookies.getAll().forEach(cookie => {
        adminResponse.cookies.set(cookie.name, cookie.value, {
          path: '/',
          sameSite: 'lax',
          secure: isProduction,
          httpOnly: true,
          ...(isProduction && { domain: COOKIE_DOMAIN }),
        });
      });
      return adminResponse;
    }

    return response;
  }

  // Redirect to the intended destination
  return NextResponse.redirect(new URL(redirect, requestUrl.origin));
}
