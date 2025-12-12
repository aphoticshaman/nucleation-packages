/**
 * Study Book Feedback API
 *
 * Allows admin to rate and provide feedback on Elle responses.
 * High ratings automatically create training examples.
 *
 * ADMIN ONLY - requires admin role
 */

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { getLearningEngine } from '@/lib/study/learning';

export const runtime = 'edge';

// =============================================================================
// AUTH CHECK
// =============================================================================

async function checkAdminAuth(): Promise<{ userId: string; email: string } | null> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey || !anonKey) {
    console.error('[Feedback] Missing Supabase config');
    return null;
  }

  const cookieStore = await cookies();
  const authClient = createServerClient(supabaseUrl, anonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll() {},
    },
  });

  const { data: { user }, error: authError } = await authClient.auth.getUser();

  if (authError || !user) {
    return null;
  }

  const serviceClient = createClient(supabaseUrl, supabaseKey);
  const { data: profile } = await serviceClient
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single();

  if (!profile || profile.role !== 'admin') {
    return null;
  }

  return { userId: user.id, email: user.email || '' };
}

// =============================================================================
// POST - Submit feedback
// =============================================================================

interface FeedbackRequest {
  messageId: string;
  rating?: number;         // 1-5 stars
  feedback?: string;       // Free-form feedback
  flagForTraining?: boolean;
  flagAsBad?: boolean;
  wasEdited?: boolean;
  taskCompleted?: boolean;
}

export async function POST(request: NextRequest) {
  try {
    const auth = await checkAdminAuth();
    if (!auth) {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const body = await request.json() as FeedbackRequest;

    if (!body.messageId) {
      return NextResponse.json({ error: 'messageId is required' }, { status: 400 });
    }

    // Validate rating if provided
    if (body.rating !== undefined && (body.rating < 1 || body.rating > 5)) {
      return NextResponse.json({ error: 'Rating must be between 1 and 5' }, { status: 400 });
    }

    const learningEngine = getLearningEngine();

    await learningEngine.updateInteraction(body.messageId, {
      user_rating: body.rating,
      user_feedback: body.feedback,
      flagged_for_training: body.flagForTraining,
      flagged_as_bad: body.flagAsBad,
      was_edited: body.wasEdited,
      task_completed: body.taskCompleted,
    });

    console.log(`[Feedback] ${auth.email} rated message ${body.messageId}: ${body.rating || 'n/a'} stars`);

    return NextResponse.json({
      success: true,
      message: 'Feedback recorded',
      trainingQueued: body.rating && body.rating >= 4,
    });
  } catch (error) {
    console.error('[Feedback] Error:', error);
    return NextResponse.json(
      { error: 'Failed to record feedback', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

// =============================================================================
// GET - Get learning stats
// =============================================================================

export async function GET(request: NextRequest) {
  try {
    const auth = await checkAdminAuth();
    if (!auth) {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 });
    }

    const { searchParams } = new URL(request.url);
    const includeRecommendations = searchParams.get('recommendations') === 'true';

    const learningEngine = getLearningEngine();
    const stats = await learningEngine.getStats(auth.userId);

    let recommendations: string[] = [];
    if (includeRecommendations) {
      recommendations = await learningEngine.getImprovementRecommendations();
    }

    return NextResponse.json({
      success: true,
      stats,
      recommendations: includeRecommendations ? recommendations : undefined,
    });
  } catch (error) {
    console.error('[Feedback] Stats error:', error);
    return NextResponse.json(
      { error: 'Failed to get stats' },
      { status: 500 }
    );
  }
}
