/**
 * GDELT Real-Time Poller
 *
 * GDELT API specs:
 * - Rate limit: 60 requests/minute (generous!)
 * - Update frequency: Every 15 minutes
 * - No API key required
 *
 * Strategy:
 * - Poll every 15-20 minutes (match GDELT update cycle)
 * - Query multiple themes per cycle (up to 60)
 * - Store in Supabase for briefing generation
 */

import { createClient } from '@supabase/supabase-js';

const GDELT_DOC_API = 'https://api.gdeltproject.org/api/v2/doc/doc';
const GDELT_GEO_API = 'https://api.gdeltproject.org/api/v2/geo/geo';

// Risk-relevant GDELT themes (can query all 60 per cycle)
const THEMES = {
  // Security & Conflict
  security: ['MILITARY', 'TERROR', 'KILL', 'WOUND', 'ARREST', 'PROTEST', 'RIOT'],
  // Political
  political: ['LEADER', 'ELECTION', 'LEGISLATION', 'COUP', 'REBELLION'],
  // Economic
  economic: ['ECON_BANKRUPTCY', 'ECON_PRICECHANGE', 'SANCTION', 'TRADE'],
  // Crisis
  crisis: ['CRISISLEX_CRISISLEXREC', 'EMERGENCY', 'EVACUATION', 'DISASTER'],
  // Health
  health: ['HEALTH_PANDEMIC', 'HEALTH_DISEASE', 'MEDICAL'],
  // Environment
  environment: ['ENV_CLIMATECHANGE', 'ENV_DISASTER', 'WATER_SECURITY'],
  // Cyber
  cyber: ['CYBER_ATTACK', 'HACK', 'DATA_BREACH'],
};

interface GDELTArticle {
  url: string;
  title: string;
  seendate: string;
  domain: string;
  language: string;
  sourcecountry: string;
  tone: number;
  themes?: string[];
}

interface GDELTSignal {
  theme: string;
  country: string;
  articleCount: number;
  avgTone: number;
  riskScore: number;
  headlines: string[];
  timestamp: string;
}

export async function pollGDELT(
  themes: string[] = Object.values(THEMES).flat(),
  timespan: string = '15min'
): Promise<GDELTSignal[]> {
  const signals: GDELTSignal[] = [];
  const countryTones: Record<string, Record<string, number[]>> = {};

  // Process themes in batches to respect rate limits
  // 60 req/min = 1 req/sec is safe
  for (const theme of themes) {
    try {
      const params = new URLSearchParams({
        query: `theme:${theme}`,
        mode: 'artlist',
        maxrecords: '75',
        format: 'json',
        timespan: timespan,
        sort: 'toneasc', // Most negative first
      });

      const response = await fetch(`${GDELT_DOC_API}?${params}`, {
        headers: { 'User-Agent': 'LatticeForge/2.0' },
      });

      if (!response.ok) {
        console.warn(`GDELT ${theme}: ${response.status}`);
        continue;
      }

      const data = await response.json();
      const articles: GDELTArticle[] = data.articles || [];

      // Aggregate by country
      for (const article of articles) {
        const country = article.sourcecountry || 'UNKNOWN';
        if (!countryTones[country]) {
          countryTones[country] = {};
        }
        if (!countryTones[country][theme]) {
          countryTones[country][theme] = [];
        }
        countryTones[country][theme].push(article.tone);
      }

      // Create signal for this theme
      if (articles.length > 0) {
        const avgTone = articles.reduce((sum, a) => sum + a.tone, 0) / articles.length;
        // Convert tone to risk: more negative = higher risk
        // GDELT tone ranges roughly -100 to +100
        const riskScore = Math.max(0, Math.min(1, (50 - avgTone) / 100));

        signals.push({
          theme,
          country: 'GLOBAL',
          articleCount: articles.length,
          avgTone,
          riskScore,
          headlines: articles.slice(0, 5).map(a => a.title),
          timestamp: new Date().toISOString(),
        });
      }

      // Rate limit: 1 request per second to be safe
      await new Promise(r => setTimeout(r, 1000));

    } catch (error) {
      console.error(`GDELT theme ${theme} error:`, error);
    }
  }

  // Generate country-level signals
  for (const [country, themes] of Object.entries(countryTones)) {
    for (const [theme, tones] of Object.entries(themes)) {
      if (tones.length >= 3) { // Require at least 3 articles
        const avgTone = tones.reduce((a, b) => a + b, 0) / tones.length;
        const riskScore = Math.max(0, Math.min(1, (50 - avgTone) / 100));

        signals.push({
          theme,
          country,
          articleCount: tones.length,
          avgTone,
          riskScore,
          headlines: [],
          timestamp: new Date().toISOString(),
        });
      }
    }
  }

  return signals;
}

/**
 * Store GDELT signals in Supabase
 */
export async function storeSignals(
  supabase: ReturnType<typeof createClient>,
  signals: GDELTSignal[]
): Promise<{ stored: number; errors: number }> {
  let stored = 0;
  let errors = 0;

  // Batch insert into learning_events
  const events = signals.map(signal => ({
    type: 'signal_observation',
    timestamp: signal.timestamp,
    session_hash: 'gdelt_realtime',
    user_tier: 'system',
    domain: signal.country,
    data: {
      numeric_features: {
        [`gdelt_${signal.theme.toLowerCase()}_count`]: signal.articleCount,
        [`gdelt_${signal.theme.toLowerCase()}_tone`]: signal.avgTone,
        gdelt_tone_risk: signal.riskScore,
      },
      categorical_features: {
        source: 'gdelt_realtime',
        theme: signal.theme,
        country: signal.country,
      },
    },
    metadata: {
      headlines: signal.headlines,
      source: 'gdelt_realtime_v2',
    },
  }));

  // Insert in batches of 100
  for (let i = 0; i < events.length; i += 100) {
    const batch = events.slice(i, i + 100);
    const { error } = await supabase.from('learning_events').insert(batch);

    if (error) {
      console.error('Batch insert error:', error);
      errors += batch.length;
    } else {
      stored += batch.length;
    }
  }

  return { stored, errors };
}

/**
 * Get aggregated signals for briefing generation
 */
export async function getAggregatedSignals(
  supabase: ReturnType<typeof createClient>,
  hoursBack: number = 24
): Promise<{
  byCountry: Record<string, { risk: number; articleCount: number; themes: string[] }>;
  byTheme: Record<string, { risk: number; articleCount: number; countries: string[] }>;
  overall: { totalArticles: number; avgRisk: number; hotspots: string[] };
}> {
  const cutoff = new Date(Date.now() - hoursBack * 60 * 60 * 1000).toISOString();

  const { data: signals } = await supabase
    .from('learning_events')
    .select('domain, data')
    .eq('session_hash', 'gdelt_realtime')
    .gte('timestamp', cutoff);

  const byCountry: Record<string, { risk: number; articleCount: number; themes: string[] }> = {};
  const byTheme: Record<string, { risk: number; articleCount: number; countries: string[] }> = {};

  let totalArticles = 0;
  let totalRisk = 0;
  let riskCount = 0;

  for (const signal of signals || []) {
    const country = signal.domain;
    const data = signal.data as {
      numeric_features?: { gdelt_tone_risk?: number };
      categorical_features?: { theme?: string };
    };

    const risk = data?.numeric_features?.gdelt_tone_risk || 0;
    const theme = data?.categorical_features?.theme || 'unknown';

    // By country
    if (!byCountry[country]) {
      byCountry[country] = { risk: 0, articleCount: 0, themes: [] };
    }
    byCountry[country].risk = Math.max(byCountry[country].risk, risk);
    byCountry[country].articleCount++;
    if (!byCountry[country].themes.includes(theme)) {
      byCountry[country].themes.push(theme);
    }

    // By theme
    if (!byTheme[theme]) {
      byTheme[theme] = { risk: 0, articleCount: 0, countries: [] };
    }
    byTheme[theme].risk = Math.max(byTheme[theme].risk, risk);
    byTheme[theme].articleCount++;
    if (!byTheme[theme].countries.includes(country)) {
      byTheme[theme].countries.push(country);
    }

    totalArticles++;
    totalRisk += risk;
    riskCount++;
  }

  // Find hotspots (high risk countries)
  const hotspots = Object.entries(byCountry)
    .filter(([_, data]) => data.risk > 0.7)
    .sort((a, b) => b[1].risk - a[1].risk)
    .slice(0, 10)
    .map(([country]) => country);

  return {
    byCountry,
    byTheme,
    overall: {
      totalArticles,
      avgRisk: riskCount > 0 ? totalRisk / riskCount : 0,
      hotspots,
    },
  };
}

/**
 * Query GDELT for a specific historical date range
 *
 * GDELT data availability:
 * - v2 API: 2015-02-18 to present
 * - Updates: Every 15 minutes
 * - Historical queries: Use startdatetime/enddatetime params
 */
export async function queryHistoricalGDELT(
  startDate: string, // ISO date: '2022-01-01'
  endDate: string,   // ISO date: '2022-12-31'
  themes: string[] = ['MILITARY', 'TERROR', 'ELECTION', 'PROTEST'],
  maxRecords: number = 250
): Promise<{
  signals: GDELTSignal[];
  summary: {
    totalArticles: number;
    dateRange: { start: string; end: string };
    topCountries: Array<{ country: string; count: number; avgTone: number }>;
    topThemes: Array<{ theme: string; count: number; avgTone: number }>;
  };
}> {
  const signals: GDELTSignal[] = [];
  const countryStats: Record<string, { count: number; tones: number[] }> = {};
  const themeStats: Record<string, { count: number; tones: number[] }> = {};

  // Convert dates to GDELT format (YYYYMMDDHHMMSS)
  const startDT = startDate.replace(/-/g, '') + '000000';
  const endDT = endDate.replace(/-/g, '') + '235959';

  for (const theme of themes) {
    try {
      const params = new URLSearchParams({
        query: `theme:${theme}`,
        mode: 'artlist',
        maxrecords: String(maxRecords),
        format: 'json',
        startdatetime: startDT,
        enddatetime: endDT,
        sort: 'datedesc',
      });

      const response = await fetch(`${GDELT_DOC_API}?${params}`, {
        headers: { 'User-Agent': 'LatticeForge/2.0 HistoricalAnalysis' },
      });

      if (!response.ok) {
        console.warn(`GDELT historical ${theme}: ${response.status}`);
        continue;
      }

      const data = await response.json();
      const articles: GDELTArticle[] = data.articles || [];

      // Aggregate stats
      for (const article of articles) {
        const country = article.sourcecountry || 'UNKNOWN';

        // Country stats
        if (!countryStats[country]) {
          countryStats[country] = { count: 0, tones: [] };
        }
        countryStats[country].count++;
        countryStats[country].tones.push(article.tone);

        // Theme stats
        if (!themeStats[theme]) {
          themeStats[theme] = { count: 0, tones: [] };
        }
        themeStats[theme].count++;
        themeStats[theme].tones.push(article.tone);
      }

      // Create signal
      if (articles.length > 0) {
        const avgTone = articles.reduce((sum, a) => sum + a.tone, 0) / articles.length;
        const riskScore = Math.max(0, Math.min(1, (50 - avgTone) / 100));

        signals.push({
          theme,
          country: 'GLOBAL',
          articleCount: articles.length,
          avgTone,
          riskScore,
          headlines: articles.slice(0, 10).map(a => a.title),
          timestamp: new Date().toISOString(),
        });
      }

      // Rate limit
      await new Promise(r => setTimeout(r, 1000));

    } catch (error) {
      console.error(`GDELT historical ${theme} error:`, error);
    }
  }

  // Build summary
  const topCountries = Object.entries(countryStats)
    .map(([country, stats]) => ({
      country,
      count: stats.count,
      avgTone: stats.tones.reduce((a, b) => a + b, 0) / stats.tones.length,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20);

  const topThemes = Object.entries(themeStats)
    .map(([theme, stats]) => ({
      theme,
      count: stats.count,
      avgTone: stats.tones.reduce((a, b) => a + b, 0) / stats.tones.length,
    }))
    .sort((a, b) => b.count - a.count);

  return {
    signals,
    summary: {
      totalArticles: Object.values(countryStats).reduce((sum, s) => sum + s.count, 0),
      dateRange: { start: startDate, end: endDate },
      topCountries,
      topThemes,
    },
  };
}

/**
 * Get GDELT timeline data for trend analysis
 *
 * Queries GDELT's timeline API for event counts over time
 */
export async function getGDELTTimeline(
  query: string = 'conflict OR protest OR election',
  startDate: string,
  endDate: string,
  resolution: 'day' | 'week' | 'month' = 'day'
): Promise<Array<{ date: string; count: number; tone: number }>> {
  try {
    const startDT = startDate.replace(/-/g, '') + '000000';
    const endDT = endDate.replace(/-/g, '') + '235959';

    const params = new URLSearchParams({
      query,
      mode: 'timelinevol',
      format: 'json',
      startdatetime: startDT,
      enddatetime: endDT,
      timeres: resolution,
    });

    const response = await fetch(`${GDELT_DOC_API}?${params}`, {
      headers: { 'User-Agent': 'LatticeForge/2.0 Timeline' },
    });

    if (!response.ok) {
      throw new Error(`GDELT timeline: ${response.status}`);
    }

    const data = await response.json();
    const timeline = data.timeline || [];

    return timeline.map((point: { date: string; value: number; tone?: number }) => ({
      date: point.date,
      count: point.value,
      tone: point.tone || 0,
    }));

  } catch (error) {
    console.error('GDELT timeline error:', error);
    return [];
  }
}

// Export for use in cron routes
export { THEMES };
