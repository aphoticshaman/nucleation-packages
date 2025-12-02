import { NextResponse } from 'next/server';
import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Lazy initialization
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

interface CascadePattern {
  trigger_domain: string;
  effect_domain: string;
  co_occurrences: number;
  avg_lag_hours: number;
  median_lag_hours: number;
  cascade_strength?: number;
}

interface CascadePrediction {
  trigger_domain: string;
  trigger_count: number;
  likely_effect_domain: string;
  probability: number;
  expected_lag_hours: number;
}

interface KnownPattern {
  name: string;
  trigger_domains: string[];
  effect_domains: string[];
  typical_lag_hours: number;
  description: string;
  historical_examples: string[];
}

// GET: Analyze cascade patterns
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const mode = searchParams.get('mode') || 'matrix';
  const domain = searchParams.get('domain');
  const hours = parseInt(searchParams.get('hours') || '24');

  try {
    const db = getSupabase();

    if (mode === 'matrix') {
      // Return full cascade matrix
      const { data, error } = await db
        .from('domain_cascade_matrix')
        .select('*')
        .order('co_occurrences', { ascending: false });

      if (error) throw error;

      return NextResponse.json({
        mode: 'matrix',
        cascades: data as CascadePattern[],
        timestamp: new Date().toISOString(),
      });
    }

    if (mode === 'predict') {
      // Predict cascades based on recent events
      const { data: recentEvents, error: recentError } = await db
        .from('training_examples')
        .select('domain')
        .gte('created_at', new Date(Date.now() - hours * 60 * 60 * 1000).toISOString());

      if (recentError) throw recentError;

      // Count events by domain
      const domainCounts: Record<string, number> = {};
      for (const event of recentEvents || []) {
        domainCounts[event.domain] = (domainCounts[event.domain] || 0) + 1;
      }

      // Get cascade matrix
      const { data: matrix, error: matrixError } = await db
        .from('domain_cascade_matrix')
        .select('*')
        .in('trigger_domain', Object.keys(domainCounts));

      if (matrixError) throw matrixError;

      // Calculate predictions
      const predictions: CascadePrediction[] = [];

      for (const [triggerDomain, count] of Object.entries(domainCounts)) {
        const domainCascades = (matrix || []).filter(
          (c: CascadePattern) => c.trigger_domain === triggerDomain
        );

        const totalCoOccurrences = domainCascades.reduce(
          (sum: number, c: CascadePattern) => sum + c.co_occurrences, 0
        );

        for (const cascade of domainCascades) {
          predictions.push({
            trigger_domain: triggerDomain,
            trigger_count: count,
            likely_effect_domain: cascade.effect_domain,
            probability: cascade.co_occurrences / (totalCoOccurrences || 1),
            expected_lag_hours: cascade.avg_lag_hours,
          });
        }
      }

      // Sort by likelihood
      predictions.sort((a, b) =>
        (b.trigger_count * b.probability) - (a.trigger_count * a.probability)
      );

      return NextResponse.json({
        mode: 'predict',
        hours_analyzed: hours,
        recent_events: domainCounts,
        predictions: predictions.slice(0, 20),
        timestamp: new Date().toISOString(),
      });
    }

    if (mode === 'domain' && domain) {
      // Get cascades for specific domain
      const { data: asSource, error: sourceError } = await db
        .from('domain_cascade_matrix')
        .select('*')
        .eq('trigger_domain', domain)
        .order('co_occurrences', { ascending: false });

      const { data: asTarget, error: targetError } = await db
        .from('domain_cascade_matrix')
        .select('*')
        .eq('effect_domain', domain)
        .order('co_occurrences', { ascending: false });

      if (sourceError || targetError) throw sourceError || targetError;

      return NextResponse.json({
        mode: 'domain',
        domain,
        triggers_cascades_to: asSource,
        receives_cascades_from: asTarget,
        timestamp: new Date().toISOString(),
      });
    }

    if (mode === 'known') {
      // Return known cascade patterns
      const { data, error } = await db
        .from('known_cascade_patterns')
        .select('*');

      if (error) throw error;

      return NextResponse.json({
        mode: 'known',
        patterns: data as KnownPattern[],
        timestamp: new Date().toISOString(),
      });
    }

    return NextResponse.json({ error: 'Invalid mode' }, { status: 400 });
  } catch (error) {
    console.error('Cascade analysis error:', error);
    return NextResponse.json(
      { error: 'Failed to analyze cascades' },
      { status: 500 }
    );
  }
}

// POST: Refresh cascade matrix (call periodically or on-demand)
export async function POST(request: Request) {
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    const isVercelCron = request.headers.get('x-vercel-cron') === '1';
    if (!isVercelCron) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  try {
    const db = getSupabase();

    // Refresh materialized view
    await db.rpc('refresh_cascade_matrix');

    // Get stats
    const { count } = await db
      .from('domain_cascade_matrix')
      .select('*', { count: 'exact', head: true });

    return NextResponse.json({
      success: true,
      relationships_detected: count,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Cascade refresh error:', error);
    return NextResponse.json(
      { error: 'Failed to refresh cascade matrix' },
      { status: 500 }
    );
  }
}
