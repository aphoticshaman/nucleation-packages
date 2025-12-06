import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

// Use Node.js runtime instead of Edge (Supabase SSR uses Node.js APIs)
export const runtime = 'nodejs';

// Cookie domain for cross-subdomain auth (auth.latticeforge.ai ↔ latticeforge.ai)
const COOKIE_DOMAIN = '.latticeforge.ai';

// More robust production detection: check VERCEL_ENV OR hostname
function isProductionEnvironment(hostname: string): boolean {
  // VERCEL_ENV is 'production' on production deployments
  if (process.env.VERCEL_ENV === 'production') return true;
  // Fallback: check if hostname is on latticeforge.ai
  if (hostname.endsWith('latticeforge.ai')) return true;
  return false;
}

export async function middleware(request: NextRequest) {
  // Create ONE response object and reuse it for all cookie operations
  // (Creating new responses on each set() loses previous cookies)
  const supabaseResponse = NextResponse.next({
    request,
  });

  // Use hostname-based production check (more reliable than just VERCEL_ENV)
  const isProduction = isProductionEnvironment(request.nextUrl.hostname);

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          // First set on request (for downstream middleware/RSC)
          cookiesToSet.forEach(({ name, value }) => {
            request.cookies.set(name, value);
          });

          // Then set on response (for browser)
          cookiesToSet.forEach(({ name, value, options }) => {
            const cookieOptions = {
              ...options,
              ...(isProduction && { domain: COOKIE_DOMAIN }),
            };
            supabaseResponse.cookies.set(name, value, cookieOptions);
          });
        },
      },
    }
  );

  // Refresh session if expired
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  const path = request.nextUrl.pathname;

  // DEBUG: Log auth state
  console.log('[MIDDLEWARE] Path:', path);
  console.log('[MIDDLEWARE] Hostname:', request.nextUrl.hostname);
  console.log('[MIDDLEWARE] isProduction:', isProduction);
  console.log('[MIDDLEWARE] User exists:', !!user);
  console.log('[MIDDLEWARE] User error:', userError?.message || 'none');
  if (user) {
    console.log('[MIDDLEWARE] User email:', user.email);
  }
  console.log('[MIDDLEWARE] Cookies received:', request.cookies.getAll().map(c => c.name));

  // Protected routes
  const protectedPaths = ['/admin', '/dashboard', '/app'];
  const isProtected = protectedPaths.some((p) => path.startsWith(p)) || path === '/';

  // Helper to create redirect with cookies preserved
  const redirectWithCookies = (url: URL) => {
    const redirectResponse = NextResponse.redirect(url);
    // Copy all cookies from supabaseResponse to the redirect
    supabaseResponse.cookies.getAll().forEach((cookie) => {
      redirectResponse.cookies.set(cookie.name, cookie.value);
    });
    return redirectResponse;
  };

  if (isProtected && !user) {
    const url = request.nextUrl.clone();
    url.pathname = '/login';
    url.searchParams.set('redirect', path === '/' ? '/app' : path);
    return redirectWithCookies(url);
  }

  // Redirect logged-in users from home to appropriate dashboard
  if (path === '/' && user) {
    // Get user role to redirect to correct dashboard
    let role = 'consumer';
    try {
      const { data: profile } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', user.id)
        .single();
      role = profile?.role || 'consumer';
    } catch {
      console.warn('Failed to fetch user profile for home redirect');
    }

    const url = request.nextUrl.clone();
    switch (role) {
      case 'admin':
        url.pathname = '/admin';
        break;
      case 'enterprise':
        url.pathname = '/dashboard';
        break;
      default:
        url.pathname = '/app';
    }
    return redirectWithCookies(url);
  }

  // Auth pages - redirect if already logged in
  const authPaths = ['/login', '/signup'];
  const isAuthPage = authPaths.some((p) => path.startsWith(p));

  if (isAuthPage && user) {
    // Get user role to redirect to correct dashboard
    // Wrapped in try-catch to handle cases where profiles table doesn't exist yet
    let role = 'consumer';
    try {
      const { data: profile } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', user.id)
        .single();
      role = profile?.role || 'consumer';
    } catch {
      // If profiles query fails (table doesn't exist, RLS issue, etc), default to consumer
      console.warn('Failed to fetch user profile, defaulting to consumer role');
    }

    const url = request.nextUrl.clone();
    switch (role) {
      case 'admin':
        url.pathname = '/admin';
        break;
      case 'enterprise':
        url.pathname = '/dashboard';
        break;
      default:
        url.pathname = '/app';
    }
    return redirectWithCookies(url);
  }

  return supabaseResponse;
}

export const config = {
  matcher: [
    // Home page
    '/',
    // Protected routes
    '/admin/:path*',
    '/dashboard/:path*',
    '/app/:path*',
    // Auth routes (redirect if already logged in)
    '/login',
    '/signup',
    // NOTE: /auth/callback is intentionally NOT included
    // The callback route handles its own cookie setting and
    // running middleware there can interfere with the auth flow
  ],
};
