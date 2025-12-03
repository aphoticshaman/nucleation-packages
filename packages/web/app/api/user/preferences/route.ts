import { createClient, getUser } from '@/lib/auth';
import { NextRequest, NextResponse } from 'next/server';

/**
 * GET /api/user/preferences
 * Fetch current user's preferences from profile
 */
export async function GET() {
  const user = await getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const supabase = await createClient();

  const { data, error } = await supabase
    .from('user_preferences')
    .select('*')
    .eq('user_id', user.id)
    .single();

  if (error && error.code !== 'PGRST116') {
    console.error('Error fetching preferences:', error);
    return NextResponse.json({ error: 'Failed to fetch preferences' }, { status: 500 });
  }

  return NextResponse.json({
    preferences: data || null,
    onboardingCompleted: !!user.onboarding_completed_at,
  });
}

/**
 * POST /api/user/preferences
 * Save user preferences (from onboarding or settings)
 */
export async function POST(request: NextRequest) {
  const user = await getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = await request.json();
  const { userId, onboardingConfig, onboardingCompletedAt } = body;

  // Verify the userId matches the authenticated user
  if (userId && userId !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const supabase = await createClient();

  // If this is from onboarding completion, update the profile
  if (onboardingCompletedAt) {
    const { error: profileError } = await supabase
      .from('profiles')
      .update({ onboarding_completed_at: onboardingCompletedAt })
      .eq('id', user.id);

    if (profileError) {
      console.error('Error updating profile onboarding status:', profileError);
      // Don't fail completely - preferences can still be saved
    }
  }

  // Upsert user preferences
  if (onboardingConfig) {
    const { error: prefsError } = await supabase
      .from('user_preferences')
      .upsert({
        user_id: user.id,
        preferences: onboardingConfig,
        updated_at: new Date().toISOString(),
      }, {
        onConflict: 'user_id',
      });

    if (prefsError) {
      console.error('Error saving preferences:', prefsError);
      // Try to create the preferences if upsert failed (table might not have row yet)
      const { error: insertError } = await supabase
        .from('user_preferences')
        .insert({
          user_id: user.id,
          preferences: onboardingConfig,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });

      if (insertError) {
        console.error('Error inserting preferences:', insertError);
        return NextResponse.json({ error: 'Failed to save preferences' }, { status: 500 });
      }
    }
  }

  return NextResponse.json({ success: true });
}

/**
 * PATCH /api/user/preferences
 * Partial update of user preferences
 */
export async function PATCH(request: NextRequest) {
  const user = await getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const updates = await request.json();
  const supabase = await createClient();

  // Get existing preferences
  const { data: existing } = await supabase
    .from('user_preferences')
    .select('preferences')
    .eq('user_id', user.id)
    .single();

  const mergedPreferences = {
    ...(existing?.preferences || {}),
    ...updates,
  };

  const { error } = await supabase
    .from('user_preferences')
    .upsert({
      user_id: user.id,
      preferences: mergedPreferences,
      updated_at: new Date().toISOString(),
    }, {
      onConflict: 'user_id',
    });

  if (error) {
    console.error('Error updating preferences:', error);
    return NextResponse.json({ error: 'Failed to update preferences' }, { status: 500 });
  }

  return NextResponse.json({ success: true, preferences: mergedPreferences });
}
