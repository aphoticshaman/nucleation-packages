import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { mapUserTierToPricing, TIER_CAPABILITIES } from '@/lib/doctrine/types';

export const runtime = 'edge';

/**
 * Query recent high-risk signals as alerts
 * GET /api/query/alerts - Returns recent alerts based on risk signals
 */
export async function GET(req: Request) {
  // Check tier access - requires Operational or higher
  const userTier = req.headers.get('x-user-tier') || 'free';
  const pricingTier = mapUserTierToPricing(userTier);

  if (!TIER_CAPABILITIES[pricingTier].api_access) {
    return NextResponse.json(
      { error: 'Alerts API requires Operational tier or higher' },
      { status: 403 }
    );
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  try {
    // Get recent high-risk events from learning_events
    const { data: riskEvents, error } = await supabase
      .from('learning_events')
      .select('timestamp, domain, data')
      .eq('type', 'signal_observation')
      .gte('timestamp', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString())
      .order('timestamp', { ascending: false })
      .limit(100);

    if (error) {
      console.error('Risk events query error:', error);
      return NextResponse.json({ alerts: [] });
    }

    // Filter for high-risk signals and format as alerts
    const alerts = riskEvents
      ?.filter((e) => {
        const risk = e.data?.numeric_features?.risk_score ||
                     e.data?.numeric_features?.gdelt_tone_risk || 0;
        return risk > 0.6;
      })
      .slice(0, 20)
      .map((e, i) => {
        const riskScore = e.data?.numeric_features?.risk_score ||
                          e.data?.numeric_features?.gdelt_tone_risk || 0;
        return {
          id: `risk-${i}-${e.timestamp}`,
          type: riskScore > 0.8 ? 'critical' : 'warning',
          category: 'risk' as const,
          title: `Elevated Risk: ${e.domain || 'Global'}`,
          message: e.data?.summary ||
                   e.data?.title ||
                   `Risk score: ${(riskScore * 100).toFixed(0)}%`,
          nation: e.domain !== 'global' ? e.domain : undefined,
          timestamp: e.timestamp,
          read: false,
        };
      }) || [];

    return NextResponse.json({ alerts });
  } catch (error) {
    console.error('Alerts query error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error', alerts: [] },
      { status: 500 }
    );
  }
}
