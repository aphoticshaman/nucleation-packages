import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';

/**
 * Query stored signal data (GDELT, USGS, Sentiment)
 * GET /api/query/signals?source=gdelt&limit=50
 * GET /api/query/signals?source=usgs&limit=20
 * GET /api/query/signals?source=sentiment&limit=10
 * GET /api/query/signals?source=freshness (returns last update times for all sources)
 */
export async function GET(req: Request) {
  const url = new URL(req.url);
  const source = url.searchParams.get('source') || 'all';
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 200);

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  try {
    // Special case: get freshness data for all sources
    if (source === 'freshness') {
      const sources = ['gdelt_ingest', 'usgs_ingest', 'sentiment_ingest'];
      const freshness: Record<string, { last_update: string | null; count_24h: number }> = {};

      for (const src of sources) {
        const { data, error } = await supabase
          .from('learning_events')
          .select('timestamp')
          .eq('session_hash', src)
          .order('timestamp', { ascending: false })
          .limit(1);

        const { count } = await supabase
          .from('learning_events')
          .select('*', { count: 'exact', head: true })
          .eq('session_hash', src)
          .gte('timestamp', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString());

        freshness[src.replace('_ingest', '')] = {
          last_update: data?.[0]?.timestamp || null,
          count_24h: count || 0,
        };
      }

      return NextResponse.json({ freshness });
    }

    // Map source name to session_hash
    const sessionHashMap: Record<string, string> = {
      gdelt: 'gdelt_ingest',
      usgs: 'usgs_ingest',
      sentiment: 'sentiment_ingest',
    };

    const sessionHash = sessionHashMap[source];
    if (!sessionHash && source !== 'all') {
      return NextResponse.json({ error: 'Invalid source' }, { status: 400 });
    }

    let query = supabase
      .from('learning_events')
      .select('timestamp, session_hash, domain, data, metadata')
      .eq('type', 'signal_observation')
      .order('timestamp', { ascending: false })
      .limit(limit);

    if (sessionHash) {
      query = query.eq('session_hash', sessionHash);
    } else {
      // For 'all', get from all ingest sources
      query = query.in('session_hash', ['gdelt_ingest', 'usgs_ingest', 'sentiment_ingest']);
    }

    const { data, error } = await query;

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    // Transform data for frontend consumption
    const signals = data?.map((row) => ({
      timestamp: row.timestamp,
      source: row.session_hash?.replace('_ingest', ''),
      domain: row.domain,
      ...row.data,
    }));

    return NextResponse.json({ signals, count: signals?.length || 0 });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
