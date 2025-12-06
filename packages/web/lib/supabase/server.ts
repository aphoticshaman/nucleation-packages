import { createServerClient, type CookieOptions } from '@supabase/ssr';
import { cookies, headers } from 'next/headers';

// Cookie domain for cross-subdomain auth (auth.latticeforge.ai ↔ latticeforge.ai)
const COOKIE_DOMAIN = '.latticeforge.ai';

// Check if we're in production by examining VERCEL_ENV or hostname
async function isProductionEnvironment(): Promise<boolean> {
  // VERCEL_ENV check
  if (process.env.VERCEL_ENV === 'production') return true;

  // Fallback: check hostname from request headers
  try {
    const headerStore = await headers();
    const host = headerStore.get('host') || headerStore.get('x-forwarded-host') || '';
    if (host.endsWith('latticeforge.ai')) return true;
  } catch {
    // Headers not available in this context
  }

  return false;
}

/**
 * Creates a Supabase client for server-side use (API routes, server components)
 * This client reads/writes auth cookies for session management
 *
 * Uses the newer getAll/setAll pattern recommended for Next.js App Router
 */
export async function createClient() {
  const cookieStore = await cookies();

  // Check production with hostname fallback
  const isProduction = await isProductionEnvironment();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet: { name: string; value: string; options: CookieOptions }[]) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              const cookieOptions = {
                name,
                value,
                ...options,
                ...(isProduction && { domain: COOKIE_DOMAIN }),
              };
              cookieStore.set(cookieOptions);
            });
          } catch {
            // Server component - can't set cookies during render
          }
        },
      },
    }
  );
}

/**
 * Creates a Supabase admin client with service role key
 * Use only for server-side operations that need elevated privileges
 */
export function createAdminClient() {
  const { createClient } = require('@supabase/supabase-js');

  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}
