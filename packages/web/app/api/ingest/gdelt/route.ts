import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Vercel Edge Runtime
export const runtime = 'edge';

// GDELT API endpoints
const GDELT_DOC_API = 'https://api.gdeltproject.org/api/v2/doc/doc';
const _GDELT_GEO_API = 'https://api.gdeltproject.org/api/v2/geo/geo'; // Reserved for geo queries

interface GDELTArticle {
  url: string;
  title: string;
  seendate: string;
  domain: string;
  language: string;
  sourcecountry: string;
  tone: number;
  themes: string[];
}

interface GDELTResponse {
  articles?: GDELTArticle[];
}

// Risk-relevant search themes
const RISK_THEMES = [
  'MILITARY',
  'PROTEST',
  'TERROR',
  'CRISISLEX_CRISISLEXREC',
  'ECON_BANKRUPTCY',
  'ECON_PRICECHANGE',
  'ENV_CLIMATECHANGE',
  'HEALTH_PANDEMIC',
  'LEADER',
  'ARREST',
  'COUP',
  'SANCTION',
];

/**
 * Ingest GDELT data for risk monitoring
 * Called by Vercel Cron or manually
 */
export async function GET(req: Request) {
  const startTime = Date.now();

  // Verify cron secret or admin access
  const authHeader = req.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );

  const results: Record<string, unknown> = {
    timestamp: new Date().toISOString(),
    themes_processed: 0,
    articles_found: 0,
    signals_stored: 0,
    errors: [] as string[],
  };

  // Interface for storing article text data
  interface ArticleRecord {
    title: string;
    url: string;
    domain: string;
    country: string;
    tone: number;
    theme: string;
    timestamp: string;
  }

  try {
    // Fetch recent articles for each risk theme
    const allSignals: Record<string, number> = {};
    const toneByCountry: Record<string, number[]> = {};
    const articleRecords: ArticleRecord[] = [];

    for (const theme of RISK_THEMES) {
      try {
        // GDELT Doc API query - last 24 hours, sorted by tone
        // Increased from 50 to 100 for better signal coverage
        const params = new URLSearchParams({
          query: `theme:${theme}`,
          mode: 'artlist',
          maxrecords: '100',
          format: 'json',
          timespan: '24h',
          sort: 'toneasc', // Most negative first (risk signals)
        });

        const response = await fetch(`${GDELT_DOC_API}?${params}`, {
          headers: { 'User-Agent': 'LatticeAI/1.0' },
        });

        if (!response.ok) {
          results.errors = [...(results.errors as string[]), `GDELT ${theme}: ${response.status}`];
          continue;
        }

        const data: GDELTResponse = await response.json();

        if (data.articles && data.articles.length > 0) {
          results.articles_found = (results.articles_found as number) + data.articles.length;

          // Aggregate tone by country AND collect individual articles
          for (const article of data.articles) {
            const country = article.sourcecountry || 'UNKNOWN';
            if (!toneByCountry[country]) {
              toneByCountry[country] = [];
            }
            toneByCountry[country].push(article.tone);

            // Store articles with negative tone (risk signals) for text analysis
            // Only store most relevant articles to avoid overwhelming storage
            if (article.tone < 0 && article.title && articleRecords.length < 200) {
              articleRecords.push({
                title: article.title,
                url: article.url,
                domain: article.domain || 'unknown',
                country: country,
                tone: article.tone,
                theme: theme,
                timestamp: article.seendate || new Date().toISOString(),
              });
            }
          }

          // Count theme occurrences as signal strength
          allSignals[`gdelt_${theme.toLowerCase()}_count`] = data.articles.length;

          // Average tone for theme (more negative = higher risk)
          const avgTone =
            data.articles.reduce((sum, a) => sum + a.tone, 0) / data.articles.length;
          allSignals[`gdelt_${theme.toLowerCase()}_tone`] = avgTone;
        }

        results.themes_processed = (results.themes_processed as number) + 1;

        // Rate limit: GDELT allows 60 req/min
        await new Promise((r) => setTimeout(r, 1100));
      } catch (themeError) {
        results.errors = [
          ...(results.errors as string[]),
          `Theme ${theme}: ${themeError instanceof Error ? themeError.message : 'unknown'}`,
        ];
      }
    }

    // Compute country-level risk scores from tone
    const countryRiskScores: Record<string, number> = {};
    for (const [country, tones] of Object.entries(toneByCountry)) {
      if (tones.length >= 3) {
        // Need at least 3 articles
        const avgTone = tones.reduce((a, b) => a + b, 0) / tones.length;
        // Normalize: GDELT tone ranges roughly -100 to +100
        // Convert to 0-1 risk score (more negative = higher risk)
        countryRiskScores[country] = Math.max(0, Math.min(1, (50 - avgTone) / 100));
      }
    }

    // Store signals in learning_events
    if (Object.keys(allSignals).length > 0) {
      const { error: insertError } = await supabase.from('learning_events').insert({
        type: 'signal_observation',
        timestamp: new Date().toISOString(),
        session_hash: 'gdelt_ingest',
        user_tier: 'system',
        domain: 'global',
        data: {
          numeric_features: allSignals,
          categorical_features: {
            source: 'gdelt',
            themes_queried: RISK_THEMES,
          },
        },
        metadata: {
          source: 'gdelt_cron',
          version: '1.0.0',
          environment: process.env.NODE_ENV || 'production',
        },
      });

      if (insertError) {
        results.errors = [...(results.errors as string[]), `DB insert: ${insertError.message}`];
      } else {
        results.signals_stored = Object.keys(allSignals).length;
      }
    }

    // Store individual article records with text data for LogicalAgent
    if (articleRecords.length > 0) {
      // Batch insert articles - these have actual text for inference
      const articleInserts = articleRecords.map((article) => ({
        type: 'signal_observation' as const,
        timestamp: article.timestamp,
        session_hash: 'gdelt_ingest',
        user_tier: 'system',
        domain: article.country,
        data: {
          // These fields are what SignalProcessor/LogicalAgent expect
          title: article.title,
          text: article.title, // Use title as text for now
          summary: `${article.theme} signal from ${article.domain}: ${article.title}`,
          url: article.url,
          source: 'gdelt',
          // Also include numeric features for signal processing
          numeric_features: {
            tone: article.tone,
            risk_score: Math.max(0, Math.min(1, (50 - article.tone) / 100)),
          },
          categorical_features: {
            theme: article.theme,
            country: article.country,
            domain: article.domain,
          },
        },
        metadata: {
          source: 'gdelt_cron',
          version: '1.1.0', // Version bump for text storage
          article_url: article.url,
          environment: process.env.NODE_ENV || 'production',
        },
      }));

      // Insert in batches of 50 to avoid timeouts
      const batchSize = 50;
      let articlesStored = 0;
      for (let i = 0; i < articleInserts.length; i += batchSize) {
        const batch = articleInserts.slice(i, i + batchSize);
        const { error: articleError } = await supabase.from('learning_events').insert(batch);
        if (articleError) {
          results.errors = [
            ...(results.errors as string[]),
            `Article batch ${i / batchSize}: ${articleError.message}`,
          ];
        } else {
          articlesStored += batch.length;
        }
      }
      results.articles_stored = articlesStored;
    }

    // Store country-level tone data
    if (Object.keys(countryRiskScores).length > 0) {
      const countryInserts = Object.entries(countryRiskScores).map(([country, risk]) => ({
        type: 'signal_observation' as const,
        timestamp: new Date().toISOString(),
        session_hash: 'gdelt_ingest',
        user_tier: 'system',
        domain: country,
        data: {
          numeric_features: {
            gdelt_tone_risk: risk,
            article_count: toneByCountry[country]?.length || 0,
          },
          categorical_features: {
            source: 'gdelt',
            country_code: country,
          },
        },
        metadata: {
          source: 'gdelt_cron',
          version: '1.0.0',
          environment: process.env.NODE_ENV || 'production',
        },
      }));

      const { error: countryError } = await supabase.from('learning_events').insert(countryInserts);

      if (countryError) {
        results.errors = [
          ...(results.errors as string[]),
          `Country insert: ${countryError.message}`,
        ];
      }
    }

    results.latency_ms = Date.now() - startTime;
    results.country_signals = Object.keys(countryRiskScores).length;

    return NextResponse.json(results);
  } catch (error) {
    console.error('GDELT ingestion error:', error);
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
