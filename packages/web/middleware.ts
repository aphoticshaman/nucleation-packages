import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

// Use Node.js runtime instead of Edge (Supabase SSR uses Node.js APIs)
export const runtime = 'nodejs';

export async function middleware(request: NextRequest) {
  let supabaseResponse = NextResponse.next({
    request,
  });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return request.cookies.get(name)?.value;
        },
        set(name: string, value: string, options: Record<string, unknown>) {
          request.cookies.set(name, value);
          supabaseResponse = NextResponse.next({
            request,
          });
          supabaseResponse.cookies.set(name, value, options);
        },
        remove(name: string, options: Record<string, unknown>) {
          request.cookies.set(name, '');
          supabaseResponse = NextResponse.next({
            request,
          });
          supabaseResponse.cookies.set(name, '', options);
        },
      },
    }
  );

  // Refresh session if expired
  const { data: { user } } = await supabase.auth.getUser();

  const path = request.nextUrl.pathname;

  // Protected routes (including home page)
  const protectedPaths = ['/admin', '/dashboard', '/app'];
  const isProtected = protectedPaths.some((p) => path.startsWith(p)) || path === '/';

  if (isProtected && !user) {
    const url = request.nextUrl.clone();
    url.pathname = '/login';
    url.searchParams.set('redirect', path);
    return NextResponse.redirect(url);
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
    return NextResponse.redirect(url);
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
    // Auth routes
    '/login',
    '/signup',
    // Auth callback - CRITICAL: Must be included for session cookies to be set
    '/auth/callback',
  ],
};
