import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  let supabaseResponse = NextResponse.next({
    request,
  });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          );
          supabaseResponse = NextResponse.next({
            request,
          });
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          );
        },
      },
    }
  );

  // Refresh session if expired
  const { data: { user } } = await supabase.auth.getUser();

  const path = request.nextUrl.pathname;

  // Protected routes
  const protectedPaths = ['/admin', '/dashboard', '/app'];
  const isProtected = protectedPaths.some((p) => path.startsWith(p));

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
    const { data: profile } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    const url = request.nextUrl.clone();
    switch (profile?.role) {
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
    // Protected routes
    '/admin/:path*',
    '/dashboard/:path*',
    '/app/:path*',
    // Auth routes
    '/login',
    '/signup',
  ],
};
