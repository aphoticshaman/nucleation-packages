import { createServerClient, type CookieOptions } from '@supabase/ssr';
import { cookies } from 'next/headers';

// Cookie domain for cross-subdomain auth (auth.latticeforge.ai ↔ latticeforge.ai)
const COOKIE_DOMAIN = '.latticeforge.ai';

/**
 * Creates a Supabase client for server-side use (API routes, server components)
 * This client reads/writes auth cookies for session management
 *
 * Uses the newer getAll/setAll pattern recommended for Next.js App Router
 */
export async function createClient() {
  const cookieStore = await cookies();

  // Only set cross-subdomain cookie domain on Vercel production
  const isProduction = process.env.VERCEL_ENV === 'production';

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
