import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export type UserRole = 'admin' | 'enterprise' | 'consumer' | 'support';

export interface UserProfile {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  role: UserRole;
  organization_id: string | null;
  is_active: boolean;
  last_seen_at: string | null;
}

// Create Supabase client for server components
export async function createClient() {
  const cookieStore = await cookies();

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
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            );
          } catch {
            // Called from Server Component
          }
        },
      },
    }
  );
}

// Get current user with profile
export async function getUser(): Promise<UserProfile | null> {
  const supabase = await createClient();

  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  const { data: profile } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', user.id)
    .single();

  return profile as UserProfile | null;
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
      case 'enterprise':
        redirect('/dashboard');
      case 'consumer':
        redirect('/app');
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
  const { data: { user } } = await supabase.auth.getUser();

  if (user) {
    await supabase.from('user_activity').insert({
      user_id: user.id,
      action,
      details,
    });
  }
}
