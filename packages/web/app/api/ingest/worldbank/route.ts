import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const runtime = 'edge';

// World Bank API - completely free, no auth needed
const WB_API = 'https://api.worldbank.org/v2';

// Key economic indicators for risk assessment
const INDICATORS = [
  { id: 'NY.GDP.MKTP.KD.ZG', name: 'gdp_growth' }, // GDP growth %
  { id: 'FP.CPI.TOTL.ZG', name: 'inflation' }, // Inflation %
  { id: 'SL.UEM.TOTL.ZS', name: 'unemployment' }, // Unemployment %
  { id: 'GC.DOD.TOTL.GD.ZS', name: 'debt_to_gdp' }, // Debt to GDP %
  { id: 'BN.CAB.XOKA.GD.ZS', name: 'current_account' }, // Current account % GDP
  { id: 'FI.RES.TOTL.MO', name: 'reserves_months' }, // Reserves in months of imports
  { id: 'NE.EXP.GNFS.ZS', name: 'exports_gdp' }, // Exports % GDP
  { id: 'NE.IMP.GNFS.ZS', name: 'imports_gdp' }, // Imports % GDP
];

// Focus countries (can expand)
const COUNTRIES = [
  'USA',
  'CHN',
  'RUS',
  'DEU',
  'GBR',
  'FRA',
  'JPN',
  'IND',
  'BRA',
  'ZAF',
  'SAU',
  'IRN',
  'TUR',
  'POL',
  'UKR',
  'KOR',
  'IDN',
  'MEX',
  'ARG',
  'EGY',
];

interface WBDataPoint {
  indicator: { id: string; value: string };
  country: { id: string; value: string };
  date: string;
  value: number | null;
}

/**
 * Ingest World Bank economic data
 * Updates country_signals table with latest macro data
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
    countries_processed: 0,
    indicators_fetched: 0,
    records_upserted: 0,
    errors: [] as string[],
  };

  try {
    const upsertBatch: Array<{
      country_code: string;
      country_name: string;
      indicator: string;
      value: number;
      year: number;
      source: string;
      metadata: Record<string, unknown>;
    }> = [];

    // Fetch each indicator for all countries
    for (const indicator of INDICATORS) {
      try {
        const countriesParam = COUNTRIES.join(';');
        const url = `${WB_API}/country/${countriesParam}/indicator/${indicator.id}?format=json&per_page=500&date=2020:2024&source=2`;

        const response = await fetch(url, {
          headers: { 'User-Agent': 'LatticeAI/1.0' },
        });

        if (!response.ok) {
          results.errors.push(`WB ${indicator.name}: ${response.status}`);
          continue;
        }

        const data = await response.json();

        // World Bank returns [metadata, data] array
        if (!Array.isArray(data) || data.length < 2 || !Array.isArray(data[1])) {
          results.errors.push(`WB ${indicator.name}: invalid response format`);
          continue;
        }

        const points: WBDataPoint[] = data[1];

        for (const point of points) {
          if (point.value !== null) {
            upsertBatch.push({
              country_code: point.country.id,
              country_name: point.country.value,
              indicator: indicator.name,
              value: point.value,
              year: parseInt(point.date, 10),
              source: 'worldbank',
              metadata: {
                original_indicator: indicator.id,
                fetched_at: new Date().toISOString(),
              },
            });
          }
        }

        results.indicators_fetched++;

        // Rate limit: be nice to free API
        await new Promise((r) => setTimeout(r, 500));
      } catch (indicatorError) {
        results.errors.push(
          `Indicator ${indicator.name}: ${indicatorError instanceof Error ? indicatorError.message : 'unknown'}`
        );
      }
    }

    // Upsert to country_signals table
    if (upsertBatch.length > 0) {
      const { error: upsertError } = await supabase.from('country_signals').upsert(upsertBatch, {
        onConflict: 'country_code,indicator,year',
        ignoreDuplicates: false,
      });

      if (upsertError) {
        results.errors.push(`DB upsert: ${upsertError.message}`);
      } else {
        results.records_upserted = upsertBatch.length;
      }
    }

    // Also log to learning_events for training
    const latestByCountry: Record<string, Record<string, number>> = {};
    for (const record of upsertBatch) {
      if (!latestByCountry[record.country_code]) {
        latestByCountry[record.country_code] = {};
      }
      // Keep latest year for each indicator
      const key = `${record.indicator}_${record.year}`;
      latestByCountry[record.country_code][key] = record.value;
    }

    // Log aggregated signals
    const { error: logError } = await supabase.from('learning_events').insert({
      type: 'signal_observation',
      timestamp: new Date().toISOString(),
      session_hash: 'worldbank_ingest',
      user_tier: 'system',
      domain: 'economic',
      data: {
        numeric_features: {
          countries_updated: Object.keys(latestByCountry).length,
          total_records: upsertBatch.length,
        },
        categorical_features: {
          source: 'worldbank',
          indicators: INDICATORS.map((i) => i.name),
          countries: COUNTRIES,
        },
      },
      metadata: {
        source: 'worldbank_cron',
        version: '1.0.0',
        environment: process.env.NODE_ENV || 'production',
      },
    });

    if (logError) {
      results.errors.push(`Learning log: ${logError.message}`);
    }

    results.countries_processed = Object.keys(latestByCountry).length;

    return NextResponse.json({
      ...results,
      latency_ms: Date.now() - startTime,
    });
  } catch (error) {
    console.error('World Bank ingestion error:', error);
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
