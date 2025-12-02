// LatticeForge Data Ingestion - Supabase Edge Function
// Pulls data from free sources: World Bank, FRED, CIA Factbook
// Deploy: supabase functions deploy ingest

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// World Bank indicator codes
const WORLD_BANK_INDICATORS = {
  gdp: 'NY.GDP.MKTP.CD',
  gdp_growth: 'NY.GDP.MKTP.KD.ZG',
  gdp_per_capita: 'NY.GDP.PCAP.CD',
  inflation: 'FP.CPI.TOTL.ZG',
  unemployment: 'SL.UEM.TOTL.ZS',
  debt_to_gdp: 'GC.DOD.TOTL.GD.ZS',
  current_account: 'BN.CAB.XOKA.GD.ZS',
  reserves: 'FI.RES.TOTL.CD',
  population: 'SP.POP.TOTL',
  fdi_inflow: 'BX.KLT.DINV.WD.GD.ZS',
};

// ISO country codes for major economies
const PRIORITY_COUNTRIES = [
  'USA',
  'CHN',
  'JPN',
  'DEU',
  'GBR',
  'IND',
  'FRA',
  'ITA',
  'BRA',
  'CAN',
  'RUS',
  'KOR',
  'AUS',
  'ESP',
  'MEX',
  'IDN',
  'NLD',
  'SAU',
  'TUR',
  'CHE',
  'POL',
  'SWE',
  'BEL',
  'ARG',
  'NOR',
  'AUT',
  'IRN',
  'THA',
  'ARE',
  'NGA',
  'ISR',
  'ZAF',
  'SGP',
  'HKG',
  'MYS',
  'PHL',
  'DNK',
  'COL',
  'PAK',
  'CHL',
  'FIN',
  'EGY',
  'BGD',
  'VNM',
  'CZE',
  'PRT',
  'ROU',
  'NZL',
  'PER',
  'IRL',
];

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const url = new URL(req.url);
    const source = url.searchParams.get('source') || 'all';
    const countries = url.searchParams.get('countries')?.split(',') || PRIORITY_COUNTRIES;

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const results: Record<string, unknown> = {};

    if (source === 'all' || source === 'worldbank') {
      results.worldbank = await ingestWorldBank(supabase, countries);
    }

    if (source === 'all' || source === 'factbook') {
      results.factbook = await ingestCIAFactbook(supabase, countries);
    }

    if (source === 'all' || source === 'fred') {
      results.fred = await ingestFRED(supabase);
    }

    return jsonResponse({
      status: 'success',
      timestamp: new Date().toISOString(),
      results,
    });
  } catch (error) {
    console.error('Ingestion error:', error);
    return jsonResponse({ error: error.message }, 500);
  }
});

// ============================================
// WORLD BANK API
// ============================================

async function ingestWorldBank(supabase: any, countries: string[]) {
  const results = { countries_updated: 0, indicators_fetched: 0, errors: [] as string[] };

  for (const [name, code] of Object.entries(WORLD_BANK_INDICATORS)) {
    try {
      // Fetch latest data for all countries
      const response = await fetch(
        `https://api.worldbank.org/v2/country/${countries.join(';')}/indicator/${code}?format=json&per_page=500&mrnev=1`
      );

      if (!response.ok) continue;

      const data = await response.json();
      if (!data[1]) continue;

      for (const record of data[1]) {
        if (record.value === null) continue;

        await supabase.from('country_signals').upsert(
          {
            country_code: record.countryiso3code,
            country_name: record.country.value,
            indicator: name,
            value: record.value,
            year: parseInt(record.date),
            source: 'worldbank',
            updated_at: new Date().toISOString(),
          },
          { onConflict: 'country_code,indicator,year' }
        );

        results.countries_updated++;
      }
      results.indicators_fetched++;

      // Small delay to be nice to the API
      await new Promise((r) => setTimeout(r, 100));
    } catch (error) {
      results.errors.push(`${name}: ${error.message}`);
    }
  }

  return results;
}

// ============================================
// CIA FACTBOOK (GitHub JSON)
// ============================================

async function ingestCIAFactbook(supabase: any, countries: string[]) {
  const results = { countries_updated: 0, errors: [] as string[] };

  // Country code to factbook path mapping (subset)
  const factbookPaths: Record<string, string> = {
    USA: 'north-america/us',
    CHN: 'east-n-southeast-asia/ch',
    JPN: 'east-n-southeast-asia/ja',
    DEU: 'europe/gm',
    GBR: 'europe/uk',
    IND: 'south-asia/in',
    FRA: 'europe/fr',
    ITA: 'europe/it',
    BRA: 'south-america/br',
    CAN: 'north-america/ca',
    RUS: 'central-asia/rs',
    KOR: 'east-n-southeast-asia/ks',
    AUS: 'australia-oceania/as',
    MEX: 'north-america/mx',
  };

  for (const code of countries) {
    const path = factbookPaths[code];
    if (!path) continue;

    try {
      const response = await fetch(
        `https://raw.githubusercontent.com/factbook/factbook.json/master/${path}.json`
      );

      if (!response.ok) continue;

      const data = await response.json();
      const economy = data.Economy || {};

      // Extract key economic indicators
      const indicators = extractFactbookIndicators(economy);

      for (const [indicator, value] of Object.entries(indicators)) {
        if (value === null) continue;

        await supabase.from('country_signals').upsert(
          {
            country_code: code,
            country_name:
              data.Government?.['Country name']?.['conventional short form']?.text || code,
            indicator: `factbook_${indicator}`,
            value: value as number,
            year: new Date().getFullYear(),
            source: 'cia_factbook',
            updated_at: new Date().toISOString(),
          },
          { onConflict: 'country_code,indicator,year' }
        );
      }

      results.countries_updated++;
      await new Promise((r) => setTimeout(r, 50));
    } catch (error) {
      results.errors.push(`${code}: ${error.message}`);
    }
  }

  return results;
}

function extractFactbookIndicators(economy: any): Record<string, number | null> {
  const extractNumber = (obj: any): number | null => {
    if (!obj) return null;
    const text = obj.text || obj.annual_values?.[0]?.value || '';
    const match = text.toString().match(/[\d,.]+/);
    if (!match) return null;
    return parseFloat(match[0].replace(/,/g, ''));
  };

  return {
    gdp_ppp: extractNumber(economy['Real GDP (purchasing power parity)']),
    gdp_growth_rate: extractNumber(economy['Real GDP growth rate']),
    inflation_rate: extractNumber(economy['Inflation rate (consumer prices)']),
    unemployment_rate: extractNumber(economy['Unemployment rate']),
    public_debt: extractNumber(economy['Public debt']),
    budget_surplus: extractNumber(economy['Budget surplus (+) or deficit (-)']),
    exports: extractNumber(economy['Exports']),
    imports: extractNumber(economy['Imports']),
    reserves: extractNumber(economy['Reserves of foreign exchange and gold']),
    external_debt: extractNumber(economy['Debt - external']),
  };
}

// ============================================
// FRED API (US Economic Data)
// ============================================

async function ingestFRED(supabase: any) {
  const results = { series_updated: 0, errors: [] as string[] };

  const fredApiKey = Deno.env.get('FRED_API_KEY');
  if (!fredApiKey) {
    results.errors.push('FRED_API_KEY not configured');
    return results;
  }

  // Key FRED series
  const fredSeries = {
    DFF: 'fed_funds_rate',
    T10Y2Y: 'yield_curve_10y2y',
    VIXCLS: 'vix',
    DCOILWTICO: 'oil_wti',
    DEXUSEU: 'usd_eur',
    SP500: 'sp500',
    UNRATE: 'us_unemployment',
    CPIAUCSL: 'us_cpi',
    M2SL: 'us_m2_money_supply',
    FEDFUNDS: 'us_fed_funds',
  };

  for (const [seriesId, name] of Object.entries(fredSeries)) {
    try {
      const response = await fetch(
        `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=${fredApiKey}&file_type=json&limit=1&sort_order=desc`
      );

      if (!response.ok) continue;

      const data = await response.json();
      if (!data.observations?.[0]) continue;

      const obs = data.observations[0];
      const value = parseFloat(obs.value);
      if (isNaN(value)) continue;

      await supabase.from('country_signals').upsert(
        {
          country_code: 'USA',
          country_name: 'United States',
          indicator: name,
          value: value,
          year: new Date(obs.date).getFullYear(),
          source: 'fred',
          metadata: { date: obs.date, series_id: seriesId },
          updated_at: new Date().toISOString(),
        },
        { onConflict: 'country_code,indicator,year' }
      );

      results.series_updated++;
      await new Promise((r) => setTimeout(r, 100));
    } catch (error) {
      results.errors.push(`${name}: ${error.message}`);
    }
  }

  return results;
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}
