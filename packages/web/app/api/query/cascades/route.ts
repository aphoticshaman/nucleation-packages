import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { mapUserTierToPricing, TIER_CAPABILITIES } from '@/lib/doctrine/types';

export const runtime = 'edge';

/**
 * Query cascade analysis results
 * GET /api/query/cascades - Returns cascade matrix and recent predictions
 */
export async function GET(req: Request) {
  // Check tier access - requires Operational or higher
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].api_access) {
    return NextResponse.json(
      { error: 'Cascade API requires Operational tier or higher' },
      { status: 403 }
    );
  }

  const url = new URL(req.url);
  const mode = url.searchParams.get('mode') || 'summary';

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  try {
    if (mode === 'matrix') {
      // Get full cascade matrix
      const { data, error } = await supabase
        .from('domain_cascade_matrix')
        .select('*')
        .order('co_occurrences', { ascending: false })
        .limit(100);

      if (error) {
        // Table might not exist yet
        return NextResponse.json({
          matrix: [],
          note: 'Cascade matrix not yet computed'
        });
      }

      return NextResponse.json({ matrix: data });
    }

    if (mode === 'recent') {
      // Get recent cascade-related events
      const { data, error } = await supabase
        .from('learning_events')
        .select('timestamp, domain, data')
        .eq('type', 'signal_observation')
        .gte('timestamp', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString())
        .order('timestamp', { ascending: false })
        .limit(50);

      if (error) {
        return NextResponse.json({ error: error.message }, { status: 500 });
      }

      // Group by domain to identify potential cascades
      const domainEvents: Record<string, number> = {};
      data?.forEach((e) => {
        if (e.domain && e.domain !== 'global') {
          domainEvents[e.domain] = (domainEvents[e.domain] || 0) + 1;
        }
      });

      // Find domains with high activity (potential cascade triggers)
      const activeDoamins = Object.entries(domainEvents)
        .filter(([_, count]) => count >= 3)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

      return NextResponse.json({
        activeDomains: activeDoamins.map(([domain, count]) => ({ domain, eventCount: count })),
        totalEvents: data?.length || 0,
        timeWindow: '24h',
      });
    }

    // Default: summary view
    const { data: matrixData } = await supabase
      .from('domain_cascade_matrix')
      .select('trigger_domain, effect_domain, co_occurrences')
      .order('co_occurrences', { ascending: false })
      .limit(10);

    const { data: recentData } = await supabase
      .from('learning_events')
      .select('timestamp')
      .eq('type', 'signal_observation')
      .order('timestamp', { ascending: false })
      .limit(1);

    return NextResponse.json({
      topCascades: matrixData || [],
      lastUpdate: recentData?.[0]?.timestamp || null,
      hasData: (matrixData?.length || 0) > 0,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
