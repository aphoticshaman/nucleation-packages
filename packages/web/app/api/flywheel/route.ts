import { NextResponse } from 'next/server';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

// Lazy initialization for service role client (for database operations)
let supabase: SupabaseClient | null = null;

function getSupabase() {
  if (!supabase) {
    supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }
  return supabase;
}

// Server-side auth verification - requires admin role for sensitive operations
async function verifyAdminAuth(): Promise<{ isAdmin: boolean; userId?: string; error?: string }> {
  try {
    const cookieStore = await cookies();
    const authClient = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll() {
            // Read-only for this check
          },
        },
      }
    );

    const { data: { user }, error: authError } = await authClient.auth.getUser();
    if (authError || !user) {
      return { isAdmin: false, error: 'Authentication required' };
    }

    // Check admin role in profiles table using service client (bypasses RLS)
    const db = getSupabase();
    const { data: profile, error: profileError } = await db
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (profileError || !profile) {
      return { isAdmin: false, userId: user.id, error: 'Profile not found' };
    }

    return { isAdmin: profile.role === 'admin', userId: user.id };
  } catch {
    return { isAdmin: false, error: 'Auth check failed' };
  }
}

// Verify cron secret for scheduled jobs
function verifyCronAuth(request: Request): boolean {
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;
  const isVercelCron = request.headers.get('x-vercel-cron') === '1';

  if (isVercelCron) return true;
  if (cronSecret && authHeader === `Bearer ${cronSecret}`) return true;
  return false;
}

interface PredictionInput {
  domain: string;
  type: string;
  content: Record<string, unknown>;
  confidence: number;
  timeframe_hours: number;
  source_example_ids?: string[];
}

interface ScoreInput {
  prediction_id: string;
  accuracy: number;
  notes?: string;
}

// GET: Flywheel stats and pending predictions
// SECURITY: Requires admin role or cron secret
export async function GET(request: Request) {
  // Verify authorization - cron jobs or admin users only
  const isCron = verifyCronAuth(request);
  if (!isCron) {
    const auth = await verifyAdminAuth();
    if (!auth.isAdmin) {
      return NextResponse.json(
        { error: auth.error || 'Admin access required' },
        { status: 403 }
      );
    }
  }

  const { searchParams } = new URL(request.url);
  const mode = searchParams.get('mode') || 'stats';

  try {
    const db = getSupabase();

    if (mode === 'stats') {
      // Get flywheel statistics
      const { data: stats, error: statsError } = await db
        .from('flywheel_stats')
        .select('*');

      if (statsError) throw statsError;

      // Get overall counts
      const { count: totalExamples } = await db
        .from('training_examples')
        .select('*', { count: 'exact', head: true });

      const { count: pendingPredictions } = await db
        .from('predictions')
        .select('*', { count: 'exact', head: true })
        .eq('outcome_observed', false);

      const { count: scoredPredictions } = await db
        .from('predictions')
        .select('*', { count: 'exact', head: true })
        .eq('outcome_observed', true);

      return NextResponse.json({
        mode: 'stats',
        total_examples: totalExamples,
        pending_predictions: pendingPredictions,
        scored_predictions: scoredPredictions,
        domain_stats: stats,
        timestamp: new Date().toISOString(),
      });
    }

    if (mode === 'pending') {
      // Get predictions awaiting scoring
      const { data, error } = await db
        .from('predictions')
        .select('*')
        .eq('outcome_observed', false)
        .lt('expires_at', new Date().toISOString())
        .order('created_at', { ascending: true })
        .limit(50);

      if (error) throw error;

      return NextResponse.json({
        mode: 'pending',
        predictions: data,
        count: data?.length || 0,
        timestamp: new Date().toISOString(),
      });
    }

    if (mode === 'weighted-sample') {
      const size = parseInt(searchParams.get('size') || '100');

      // Get weighted sample of training data
      const { data, error } = await db
        .rpc('get_weighted_training_sample', { sample_size: size });

      if (error) throw error;

      return NextResponse.json({
        mode: 'weighted-sample',
        sample: data,
        size: data?.length || 0,
        timestamp: new Date().toISOString(),
      });
    }

    if (mode === 'top-performers') {
      // Get highest-weighted training examples
      const { data, error } = await db
        .from('training_examples')
        .select('id, instruction, input, output, domain, selection_weight, prediction_count')
        .order('selection_weight', { ascending: false })
        .limit(20);

      if (error) throw error;

      return NextResponse.json({
        mode: 'top-performers',
        examples: data,
        timestamp: new Date().toISOString(),
      });
    }

    return NextResponse.json({ error: 'Invalid mode' }, { status: 400 });
  } catch (error) {
    console.error('Flywheel GET error:', error);
    return NextResponse.json({ error: 'Failed to get flywheel data' }, { status: 500 });
  }
}

// POST: Record prediction or score outcome
// SECURITY: Requires admin role or cron secret for ALL actions
export async function POST(request: Request) {
  // Verify authorization - cron jobs or admin users only
  const isCron = verifyCronAuth(request);
  if (!isCron) {
    const auth = await verifyAdminAuth();
    if (!auth.isAdmin) {
      return NextResponse.json(
        { error: auth.error || 'Admin access required' },
        { status: 403 }
      );
    }
  }

  try {
    const body = await request.json();
    const db = getSupabase();

    // Record a new prediction
    if (body.action === 'predict') {
      const pred = body.prediction as PredictionInput;

      const { data, error } = await db
        .rpc('record_prediction', {
          p_domain: pred.domain,
          p_type: pred.type,
          p_content: pred.content,
          p_confidence: pred.confidence,
          p_timeframe_hours: pred.timeframe_hours,
          p_source_examples: pred.source_example_ids || [],
        });

      if (error) throw error;

      return NextResponse.json({
        success: true,
        prediction_id: data,
        expires_at: new Date(Date.now() + pred.timeframe_hours * 60 * 60 * 1000).toISOString(),
      });
    }

    // Score a prediction
    if (body.action === 'score') {
      const score = body.score as ScoreInput;

      const { error } = await db
        .rpc('score_prediction', {
          p_prediction_id: score.prediction_id,
          p_accuracy: score.accuracy,
          p_notes: score.notes || null,
        });

      if (error) throw error;

      return NextResponse.json({
        success: true,
        message: 'Prediction scored and weights updated',
      });
    }

    // Batch score multiple predictions
    if (body.action === 'batch-score') {
      const scores = body.scores as ScoreInput[];
      const results = [];

      for (const score of scores) {
        try {
          await db.rpc('score_prediction', {
            p_prediction_id: score.prediction_id,
            p_accuracy: score.accuracy,
            p_notes: score.notes || null,
          });
          results.push({ id: score.prediction_id, success: true });
        } catch (e) {
          results.push({ id: score.prediction_id, success: false, error: String(e) });
        }
      }

      return NextResponse.json({
        success: true,
        results,
        scored: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length,
      });
    }

    // Export weighted training data
    if (body.action === 'export-weighted') {
      const size = body.size || 1000;
      const format = body.format || 'alpaca';

      const { data, error } = await db
        .rpc('get_weighted_training_sample', { sample_size: size });

      if (error) throw error;

      if (format === 'alpaca') {
        return NextResponse.json(data);
      }

      // ChatML format
      const chatml = (data || []).map((d: { instruction: string; input: string; output: string }) => ({
        messages: [
          { role: 'system', content: 'You are a multi-domain intelligence analyst.' },
          { role: 'user', content: `${d.instruction}\n\n${d.input}` },
          { role: 'assistant', content: d.output },
        ],
      }));

      return NextResponse.json(chatml);
    }

    // Cleanup expired predictions
    if (body.action === 'cleanup') {
      const { data, error } = await db.rpc('cleanup_expired_predictions');

      if (error) throw error;

      return NextResponse.json({
        success: true,
        deleted: data,
      });
    }

    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    console.error('Flywheel POST error:', error);
    return NextResponse.json({ error: 'Failed to process flywheel action' }, { status: 500 });
  }
}
