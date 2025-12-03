import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export type UserRole = 'admin' | 'enterprise' | 'consumer' | 'support';
export type UserTier = 'free' | 'starter' | 'pro' | 'enterprise_tier';

export interface UserProfile {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  role: UserRole;
  tier: UserTier;
  organization_id: string | null;
  is_active: boolean;
  last_seen_at: string | null;
}

// Cookie domain for cross-subdomain auth (auth.latticeforge.ai ↔ latticeforge.ai)
const COOKIE_DOMAIN = '.latticeforge.ai';

// Create Supabase client for server components
export async function createClient() {
  const cookieStore = await cookies();

  // Check if we're on latticeforge.ai (not localhost) by looking at the Supabase URL
  // In production, NEXT_PUBLIC_SUPABASE_URL will contain latticeforge
  const isProduction =
    process.env.NEXT_PUBLIC_SUPABASE_URL?.includes('latticeforge') ||
    process.env.VERCEL_ENV === 'production';

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value;
        },
        set(name: string, value: string, options: Record<string, unknown>) {
          try {
            // Set cookie with cross-subdomain domain only in production
            const cookieOptions = {
              name,
              value,
              ...options,
              ...(isProduction && { domain: COOKIE_DOMAIN }),
            };
            cookieStore.set(cookieOptions);
          } catch {
            // Called from Server Component - can't set cookies
          }
        },
        remove(name: string, options: Record<string, unknown>) {
          try {
            // Remove cookie with cross-subdomain domain only in production
            const cookieOptions = {
              name,
              value: '',
              ...options,
              ...(isProduction && { domain: COOKIE_DOMAIN }),
            };
            cookieStore.set(cookieOptions);
          } catch {
            // Called from Server Component - can't set cookies
          }
        },
      },
    }
  );
}

// Get current user with profile
// Auto-creates a profile if one doesn't exist (for OAuth users)
export async function getUser(): Promise<UserProfile | null> {
  const supabase = await createClient();

  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError) {
    console.error('Auth error in getUser:', authError);
    return null;
  }

  if (!user) return null;

  // Try to get existing profile - but don't fail if profiles table has issues
  try {
    const { data: profile, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', user.id)
      .single();

    // If profile exists, return it
    if (profile) {
      return profile as UserProfile;
    }

    // If profile doesn't exist (new OAuth user), try to create one
    if (error?.code === 'PGRST116') {
      // PGRST116 = "JSON object requested, multiple (or no) rows returned"
      const newProfile: Omit<UserProfile, 'last_seen_at'> & { last_seen_at: string } = {
        id: user.id,
        email: user.email || '',
        full_name: user.user_metadata?.full_name || user.user_metadata?.name || null,
        avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture || null,
        role: 'consumer',
        tier: 'free',
        organization_id: null,
        is_active: true,
        last_seen_at: new Date().toISOString(),
      };

      const { data: createdProfile, error: insertError } = await supabase
        .from('profiles')
        .insert(newProfile)
        .select()
        .single();

      if (!insertError && createdProfile) {
        return createdProfile as UserProfile;
      }

      // Insert failed - fall through to return minimal profile
      console.error('Failed to create user profile:', insertError);
    } else if (error) {
      // Some other error (RLS, table missing, etc)
      console.error('Error fetching user profile:', error);
    }
  } catch (e) {
    console.error('Exception in profile fetch:', e);
  }

  // ALWAYS return a minimal profile if we have a valid Supabase user
  // This prevents redirect loops when profiles table has issues
  return {
    id: user.id,
    email: user.email || '',
    full_name: user.user_metadata?.full_name || user.user_metadata?.name || null,
    avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture || null,
    role: 'consumer',
    tier: 'free',
    organization_id: null,
    is_active: true,
    last_seen_at: null,
  };
}

// Require authentication
export async function requireAuth(): Promise<UserProfile> {
  const user = await getUser();
  if (!user) {
    redirect('/login');
  }
  return user;
}

// Require specific role(s)
export async function requireRole(allowedRoles: UserRole[]): Promise<UserProfile> {
  const user = await requireAuth();

  if (!allowedRoles.includes(user.role)) {
    // Redirect to appropriate dashboard based on role
    switch (user.role) {
      case 'admin':
        redirect('/admin');
        break;
      case 'enterprise':
        redirect('/dashboard');
        break;
      case 'consumer':
        redirect('/app');
        break;
      default:
        redirect('/');
    }
  }

  return user;
}

// Role check helpers
export async function requireAdmin(): Promise<UserProfile> {
  return requireRole(['admin']);
}

export async function requireEnterprise(): Promise<UserProfile> {
  return requireRole(['admin', 'enterprise']);
}

export async function requireConsumer(): Promise<UserProfile> {
  return requireRole(['admin', 'consumer']);
}

// Log user activity
export async function logActivity(action: string, details: Record<string, unknown> = {}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (user) {
    await supabase.from('user_activity').insert({
      user_id: user.id,
      action,
      details,
    });
  }
}
