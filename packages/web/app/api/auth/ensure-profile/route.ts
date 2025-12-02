import { createClient } from '@supabase/supabase-js';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

// Create admin client with service role (bypasses RLS)
function createAdminClient() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}

// Create user client for auth checks
async function createUserClient() {
  const cookieStore = await cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value;
        },
        set() {},
        remove() {},
      },
    }
  );
}

export async function POST() {
  try {
    // Get the authenticated user
    const userClient = await createUserClient();
    const { data: { user }, error: authError } = await userClient.auth.getUser();

    if (authError || !user) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    // Use admin client to check/create profile (bypasses RLS)
    const adminClient = createAdminClient();

    // Check if profile exists
    const { data: existingProfile } = await adminClient
      .from('profiles')
      .select('id')
      .eq('id', user.id)
      .single();

    if (existingProfile) {
      return NextResponse.json({ status: 'exists', profile: existingProfile });
    }

    // Create profile
    const newProfile = {
      id: user.id,
      email: user.email || '',
      full_name: user.user_metadata?.full_name || user.user_metadata?.name || null,
      avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture || null,
      role: 'consumer',
      organization_id: null,
      is_active: true,
      last_seen_at: new Date().toISOString(),
    };

    const { data: createdProfile, error: insertError } = await adminClient
      .from('profiles')
      .insert(newProfile)
      .select()
      .single();

    if (insertError) {
      console.error('Failed to create profile:', insertError);
      return NextResponse.json(
        { error: 'Failed to create profile', details: insertError.message },
        { status: 500 }
      );
    }

    return NextResponse.json({ status: 'created', profile: createdProfile });
  } catch (error) {
    console.error('Ensure profile error:', error);
    return NextResponse.json(
      { error: 'Internal error' },
      { status: 500 }
    );
  }
}
