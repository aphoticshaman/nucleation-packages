import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextResponse, type NextRequest } from 'next/server';

// Cookie domain for cross-subdomain auth
const COOKIE_DOMAIN = '.latticeforge.ai';

export async function GET(request: NextRequest) {
  const cookieStore = await cookies();

  // Check if we're on latticeforge.ai (not localhost)
  const isProduction = request.nextUrl.hostname.includes('latticeforge.ai');

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
            const cookieOptions = {
              name,
              value,
              ...options,
              ...(isProduction && { domain: COOKIE_DOMAIN }),
            };
            cookieStore.set(cookieOptions);
          } catch {
            // Might fail in some contexts
          }
        },
        remove(name: string, options: Record<string, unknown>) {
          try {
            const cookieOptions = {
              name,
              value: '',
              ...options,
              ...(isProduction && { domain: COOKIE_DOMAIN }),
            };
            cookieStore.set(cookieOptions);
          } catch {
            // Might fail in some contexts
          }
        },
      },
    }
  );

  // Get user ID before signing out (for session tracking)
  const { data: { user } } = await supabase.auth.getUser();

  // Record sign out in database if user exists
  if (user) {
    try {
      await supabase.rpc('record_sign_out', { p_user_id: user.id });
    } catch (err) {
      // Don't block sign out if recording fails
      console.error('Failed to record sign out:', err);
    }
  }

  // Sign out from Supabase - this invalidates the session
  await supabase.auth.signOut();

  // Build redirect response
  const redirectUrl = new URL('/login', request.url);
  redirectUrl.searchParams.set('signedOut', 'true');
  const response = NextResponse.redirect(redirectUrl);

  // Explicitly clear all Supabase auth cookies
  // Supabase uses cookies like: sb-<project-ref>-auth-token, sb-<project-ref>-auth-token.0, etc.
  const allCookies = cookieStore.getAll();
  for (const cookie of allCookies) {
    if (cookie.name.includes('sb-') || cookie.name.includes('supabase')) {
      // Clear cookie on both the response and with the domain
      response.cookies.set({
        name: cookie.name,
        value: '',
        expires: new Date(0),
        path: '/',
        ...(isProduction && { domain: COOKIE_DOMAIN }),
      });
    }
  }

  return response;
}

// Also support POST for security best practices
export async function POST(request: NextRequest) {
  return GET(request);
}
