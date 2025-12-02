import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';

// Free sentiment/market data APIs
const FEAR_GREED_API = 'https://api.alternative.me/fng/';
const COINGECKO_GLOBAL = 'https://api.coingecko.com/api/v3/global';

interface FearGreedResponse {
  data: Array<{
    value: string;
    value_classification: string;
    timestamp: string;
  }>;
}

interface CoinGeckoGlobal {
  data: {
    active_cryptocurrencies: number;
    markets: number;
    total_market_cap: { usd: number };
    total_volume: { usd: number };
    market_cap_percentage: { btc: number; eth: number };
    market_cap_change_percentage_24h_usd: number;
  };
}

/**
 * Ingest market sentiment data
 * Fear & Greed Index + CoinGecko global metrics
 */
export async function GET(req: Request) {
  const startTime = Date.now();

  // Verify cron secret
  const authHeader = req.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const results = {
    timestamp: new Date().toISOString(),
    sources_fetched: 0,
    signals_stored: 0,
    errors: [] as string[],
  };

  const numericFeatures: Record<string, number> = {};
  const categoricalFeatures: Record<string, unknown> = { sources: [] as string[] };

  try {
    // 1. Fetch Fear & Greed Index
    try {
      const fgResponse = await fetch(`${FEAR_GREED_API}?limit=7`, {
        headers: { 'User-Agent': 'LatticeAI/1.0' },
      });

      if (fgResponse.ok) {
        const fgData: FearGreedResponse = await fgResponse.json();

        if (fgData.data && fgData.data.length > 0) {
          const latest = fgData.data[0];
          numericFeatures['fear_greed_index'] = parseInt(latest.value, 10);
          categoricalFeatures['fear_greed_class'] = latest.value_classification;

          // Calculate 7-day trend
          if (fgData.data.length >= 7) {
            const values = fgData.data.slice(0, 7).map((d) => parseInt(d.value, 10));
            const avg = values.reduce((a, b) => a + b, 0) / values.length;
            const trend = values[0] - values[values.length - 1]; // + means improving sentiment
            numericFeatures['fear_greed_7d_avg'] = avg;
            numericFeatures['fear_greed_7d_trend'] = trend;
          }

          (categoricalFeatures.sources as string[]).push('fear_greed');
          results.sources_fetched++;
        }
      } else {
        results.errors.push(`Fear & Greed: ${fgResponse.status}`);
      }
    } catch (fgError) {
      results.errors.push(
        `Fear & Greed: ${fgError instanceof Error ? fgError.message : 'unknown'}`
      );
    }

    // 2. Fetch CoinGecko global metrics
    try {
      const cgResponse = await fetch(COINGECKO_GLOBAL, {
        headers: { 'User-Agent': 'LatticeAI/1.0' },
      });

      if (cgResponse.ok) {
        const cgData: CoinGeckoGlobal = await cgResponse.json();

        if (cgData.data) {
          numericFeatures['crypto_market_cap_usd'] = cgData.data.total_market_cap.usd;
          numericFeatures['crypto_volume_24h_usd'] = cgData.data.total_volume.usd;
          numericFeatures['crypto_market_cap_change_24h'] =
            cgData.data.market_cap_change_percentage_24h_usd;
          numericFeatures['btc_dominance'] = cgData.data.market_cap_percentage.btc;
          numericFeatures['eth_dominance'] = cgData.data.market_cap_percentage.eth;
          numericFeatures['active_cryptocurrencies'] = cgData.data.active_cryptocurrencies;
          numericFeatures['active_markets'] = cgData.data.markets;

          (categoricalFeatures.sources as string[]).push('coingecko');
          results.sources_fetched++;
        }
      } else {
        results.errors.push(`CoinGecko: ${cgResponse.status}`);
      }
    } catch (cgError) {
      results.errors.push(`CoinGecko: ${cgError instanceof Error ? cgError.message : 'unknown'}`);
    }

    // Store in learning_events
    if (Object.keys(numericFeatures).length > 0) {
      const { error: insertError } = await supabase.from('learning_events').insert({
        type: 'signal_observation',
        timestamp: new Date().toISOString(),
        session_hash: 'sentiment_ingest',
        user_tier: 'system',
        domain: 'market_sentiment',
        data: {
          numeric_features: numericFeatures,
          categorical_features: categoricalFeatures,
        },
        metadata: {
          source: 'sentiment_cron',
          version: '1.0.0',
          environment: process.env.NODE_ENV || 'production',
        },
      });

      if (insertError) {
        results.errors.push(`DB insert: ${insertError.message}`);
      } else {
        results.signals_stored = Object.keys(numericFeatures).length;
      }
    }

    return NextResponse.json({
      ...results,
      latency_ms: Date.now() - startTime,
      features: Object.keys(numericFeatures),
    });
  } catch (error) {
    console.error('Sentiment ingestion error:', error);
    return NextResponse.json(
      {
        error: 'Ingestion failed',
        details: error instanceof Error ? error.message : 'unknown',
        latency_ms: Date.now() - startTime,
      },
      { status: 500 }
    );
  }
}
