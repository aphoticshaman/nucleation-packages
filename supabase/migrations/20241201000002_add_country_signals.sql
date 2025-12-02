-- Country signals table for storing macro data from free sources
-- Sources: World Bank, CIA Factbook, FRED, IMF

CREATE TABLE IF NOT EXISTS country_signals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  country_code TEXT NOT NULL,        -- ISO 3166-1 alpha-3 (USA, CHN, DEU)
  country_name TEXT NOT NULL,
  indicator TEXT NOT NULL,           -- gdp, inflation, unemployment, etc.
  value NUMERIC NOT NULL,
  year INTEGER NOT NULL,
  source TEXT NOT NULL,              -- worldbank, cia_factbook, fred, imf
  metadata JSONB DEFAULT '{}',
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(country_code, indicator, year)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_country_signals_country ON country_signals(country_code);
CREATE INDEX IF NOT EXISTS idx_country_signals_indicator ON country_signals(indicator);
CREATE INDEX IF NOT EXISTS idx_country_signals_year ON country_signals(year DESC);
CREATE INDEX IF NOT EXISTS idx_country_signals_source ON country_signals(source);
CREATE INDEX IF NOT EXISTS idx_country_signals_updated ON country_signals(updated_at DESC);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_country_signals_country_indicator ON country_signals(country_code, indicator, year DESC);

-- Enable RLS
ALTER TABLE country_signals ENABLE ROW LEVEL SECURITY;

-- Policy: authenticated users can read all country signals (it's public data)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE tablename = 'country_signals' AND policyname = 'Anyone can view country signals'
  ) THEN
    CREATE POLICY "Anyone can view country signals"
      ON country_signals FOR SELECT
      USING (true);
  END IF;
END $$;

-- Policy: only service role can insert/update (Edge Functions)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE tablename = 'country_signals' AND policyname = 'Service role can manage country signals'
  ) THEN
    CREATE POLICY "Service role can manage country signals"
      ON country_signals FOR ALL
      USING (auth.role() = 'service_role');
  END IF;
END $$;

-- Materialized view for latest values per country/indicator
CREATE MATERIALIZED VIEW IF NOT EXISTS country_signals_latest AS
SELECT DISTINCT ON (country_code, indicator)
  id,
  country_code,
  country_name,
  indicator,
  value,
  year,
  source,
  updated_at
FROM country_signals
ORDER BY country_code, indicator, year DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_country_signals_latest_pk
  ON country_signals_latest(country_code, indicator);

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_country_signals_latest()
RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY country_signals_latest;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public;

-- Useful views for analysis
CREATE OR REPLACE VIEW country_risk_score AS
SELECT
  country_code,
  country_name,
  MAX(CASE WHEN indicator = 'debt_to_gdp' THEN value END) as debt_to_gdp,
  MAX(CASE WHEN indicator = 'current_account' THEN value END) as current_account,
  MAX(CASE WHEN indicator = 'inflation' THEN value END) as inflation,
  MAX(CASE WHEN indicator = 'gdp_growth' THEN value END) as gdp_growth,
  MAX(CASE WHEN indicator = 'unemployment' THEN value END) as unemployment,
  MAX(CASE WHEN indicator = 'reserves' THEN value END) as reserves,
  MAX(updated_at) as last_updated
FROM country_signals_latest
GROUP BY country_code, country_name;
