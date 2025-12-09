import { createServerClient } from '@supabase/ssr';
import { createClient as createSupabaseClient } from '@supabase/supabase-js';
import { cookies, headers } from 'next/headers';
import { redirect } from 'next/navigation';

export type UserRole = 'admin' | 'enterprise' | 'consumer' | 'support';

// Service role client for admin operations (bypasses RLS)
function createServiceRoleClient() {
  return createSupabaseClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}
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
  onboarding_completed_at: string | null;
}

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

// Create Supabase client for server components
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
        setAll(cookiesToSet) {
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
            // Called from Server Component - can't set cookies
          }
        },
      },
    }
  );
}

// Map organization plan to user tier
function planToTier(plan: string | null, role: UserRole): UserTier {
  // Admins always get enterprise tier
  if (role === 'admin') return 'enterprise_tier';

  switch (plan) {
    case 'enterprise':
      return 'enterprise_tier';
    case 'pro':
      return 'pro';
    case 'starter':
      return 'starter';
    default:
      return 'free';
  }
}

// Database profile shape (what Supabase actually returns)
interface DBProfile {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  role: UserRole;
  organization_id: string | null;
  is_active: boolean;
  last_seen_at: string | null;
  metadata: Record<string, unknown> | null;
  organizations?: { plan: string } | null;
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

  // Try to get existing profile with organization for plan/tier
  const { data: profile, error } = await supabase
    .from('profiles')
    .select('*, organizations(plan)')
    .eq('id', user.id)
    .single();

  // If profile exists, return it with derived tier
  if (profile) {
    const dbProfile = profile as DBProfile;

    // Update last_seen_at using service role (bypasses RLS, runs in background)
    // Using service role because user client RLS can fail in async context
    const serviceClient = createServiceRoleClient();
    serviceClient
      .from('profiles')
      .update({ last_seen_at: new Date().toISOString() })
      .eq('id', user.id)
      .then(({ error }) => {
        if (error) {
          console.error('[AUTH] Failed to update last_seen_at:', error.message);
        }
      });

    // Derive tier from org plan or role
    const tier = planToTier(dbProfile.organizations?.plan || null, dbProfile.role);

    // Check onboarding status from metadata
    const onboardingCompleted = dbProfile.metadata?.onboarding_completed_at as string | null;

    return {
      id: dbProfile.id,
      email: dbProfile.email,
      full_name: dbProfile.full_name,
      avatar_url: dbProfile.avatar_url,
      role: dbProfile.role,
      tier,
      organization_id: dbProfile.organization_id,
      is_active: dbProfile.is_active,
      last_seen_at: dbProfile.last_seen_at,
      onboarding_completed_at: onboardingCompleted,
    };
  }

  // If profile doesn't exist (new OAuth user), try to create one
  if (error?.code === 'PGRST116') {
    // PGRST116 = "JSON object requested, multiple (or no) rows returned"
    const newProfile = {
      id: user.id,
      email: user.email || '',
      full_name: user.user_metadata?.full_name || user.user_metadata?.name || null,
      avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture || null,
      role: 'consumer' as UserRole,
      organization_id: null,
      is_active: true,
      last_seen_at: new Date().toISOString(),
      metadata: {},
    };

    const { data: createdProfile, error: insertError } = await supabase
      .from('profiles')
      .insert(newProfile)
      .select('*, organizations(plan)')
      .single();

    if (!insertError && createdProfile) {
      const dbProfile = createdProfile as DBProfile;
      return {
        id: dbProfile.id,
        email: dbProfile.email,
        full_name: dbProfile.full_name,
        avatar_url: dbProfile.avatar_url,
        role: dbProfile.role,
        tier: 'free',
        organization_id: dbProfile.organization_id,
        is_active: dbProfile.is_active,
        last_seen_at: dbProfile.last_seen_at,
        onboarding_completed_at: null,
      };
    }

    console.error('Failed to create user profile:', insertError);
  } else if (error) {
    // Log the actual error so we can debug RLS issues
    console.error('Profile fetch error:', error.code, error.message, 'for user:', user.id);
  }

  // Only fall back to minimal profile if we truly couldn't get/create one
  // This should be rare - indicates RLS or table issues
  console.error('CRITICAL: Falling back to minimal profile for user:', user.id, user.email);
  console.error('This usually means RLS is blocking profile access. Check profiles table policies.');

  // Return minimal profile - user will need DB fix for proper role/tier
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
    onboarding_completed_at: null,
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
