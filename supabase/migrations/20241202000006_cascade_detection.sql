-- Cross-Domain Cascade Detection
-- Finds patterns where events in one domain precede events in another

-- Add cascade tracking columns to training_examples
ALTER TABLE training_examples
ADD COLUMN IF NOT EXISTS cascade_source_id UUID REFERENCES training_examples(id),
ADD COLUMN IF NOT EXISTS cascade_lag_hours INTEGER;

-- Index for cascade queries
CREATE INDEX IF NOT EXISTS idx_training_examples_domain_date
ON training_examples(domain, created_at DESC);

-- Materialized view for domain co-occurrence within time windows
CREATE MATERIALIZED VIEW IF NOT EXISTS domain_cascade_matrix AS
WITH time_windows AS (
  SELECT
    a.id as trigger_id,
    a.domain as trigger_domain,
    a.created_at as trigger_time,
    b.id as effect_id,
    b.domain as effect_domain,
    b.created_at as effect_time,
    EXTRACT(EPOCH FROM (b.created_at - a.created_at)) / 3600 as lag_hours
  FROM training_examples a
  JOIN training_examples b
    ON b.created_at BETWEEN a.created_at AND a.created_at + interval '7 days'
    AND b.domain != a.domain
    AND b.id != a.id
  WHERE a.created_at > NOW() - interval '90 days'
)
SELECT
  trigger_domain,
  effect_domain,
  COUNT(*) as co_occurrences,
  AVG(lag_hours) as avg_lag_hours,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lag_hours) as median_lag_hours,
  MIN(lag_hours) as min_lag_hours,
  MAX(lag_hours) as max_lag_hours,
  STDDEV(lag_hours) as stddev_lag_hours
FROM time_windows
GROUP BY trigger_domain, effect_domain
HAVING COUNT(*) >= 3;  -- Minimum observations for statistical relevance

-- Index the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_cascade_matrix_domains
ON domain_cascade_matrix(trigger_domain, effect_domain);

-- Function to refresh cascade matrix (call periodically)
CREATE OR REPLACE FUNCTION refresh_cascade_matrix()
RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY domain_cascade_matrix;
END;
$$ LANGUAGE plpgsql;

-- View for strongest cascade relationships
CREATE OR REPLACE VIEW strongest_cascades AS
SELECT
  trigger_domain,
  effect_domain,
  co_occurrences,
  avg_lag_hours,
  median_lag_hours,
  -- Cascade strength score: more co-occurrences + tighter time window = stronger
  (co_occurrences::float / NULLIF(avg_lag_hours, 0)) *
    (1 / NULLIF(stddev_lag_hours, 0)) as cascade_strength
FROM domain_cascade_matrix
WHERE co_occurrences >= 5
ORDER BY cascade_strength DESC NULLS LAST;

-- Function to detect live cascades (what might happen next)
CREATE OR REPLACE FUNCTION predict_cascades(recent_hours INTEGER DEFAULT 24)
RETURNS TABLE (
  trigger_domain TEXT,
  trigger_count BIGINT,
  likely_effect_domain TEXT,
  probability FLOAT,
  expected_lag_hours FLOAT
) AS $$
BEGIN
  RETURN QUERY
  WITH recent_events AS (
    SELECT domain, COUNT(*) as event_count
    FROM training_examples
    WHERE created_at > NOW() - (recent_hours || ' hours')::interval
    GROUP BY domain
  )
  SELECT
    r.domain as trigger_domain,
    r.event_count as trigger_count,
    c.effect_domain as likely_effect_domain,
    -- Probability based on historical co-occurrence rate
    (c.co_occurrences::float / (
      SELECT SUM(co_occurrences)
      FROM domain_cascade_matrix
      WHERE trigger_domain = r.domain
    )) as probability,
    c.avg_lag_hours as expected_lag_hours
  FROM recent_events r
  JOIN domain_cascade_matrix c ON c.trigger_domain = r.domain
  WHERE c.co_occurrences >= 5
  ORDER BY r.event_count DESC, probability DESC;
END;
$$ LANGUAGE plpgsql;

-- Specific cascade patterns we care about (known geopolitical cascades)
CREATE TABLE IF NOT EXISTS known_cascade_patterns (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  trigger_domains TEXT[] NOT NULL,
  effect_domains TEXT[] NOT NULL,
  typical_lag_hours INTEGER,
  description TEXT,
  historical_examples TEXT[],
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed with known patterns
INSERT INTO known_cascade_patterns (name, trigger_domains, effect_domains, typical_lag_hours, description, historical_examples)
VALUES
  ('Energy Crisis → Financial Stress',
   ARRAY['energy'],
   ARRAY['financial', 'geopolitical'],
   72,
   'Oil price spikes cascade to inflation fears and political instability',
   ARRAY['1973 Oil Crisis', '2008 Oil Spike', '2022 Russia-Ukraine']),

  ('Cyber Attack → Defense Posture',
   ARRAY['cyber'],
   ARRAY['defense', 'geopolitical'],
   24,
   'Major cyber incidents trigger military/diplomatic responses',
   ARRAY['SolarWinds 2020', 'Colonial Pipeline 2021']),

  ('Supply Chain → Manufacturing → Employment',
   ARRAY['supply_chain'],
   ARRAY['manufacturing', 'employment'],
   168,
   'Supply disruptions cascade through production to labor markets',
   ARRAY['COVID Supply Shock 2020', 'Suez Canal 2021']),

  ('Health Crisis → Economic → Political',
   ARRAY['health'],
   ARRAY['financial', 'employment', 'geopolitical'],
   336,
   'Pandemics cascade through economy to political instability',
   ARRAY['COVID-19 2020', 'SARS 2003']),

  ('Climate Event → Agriculture → Geopolitical',
   ARRAY['climate'],
   ARRAY['agriculture', 'geopolitical'],
   720,
   'Droughts/floods cause food insecurity and migration/conflict',
   ARRAY['Syrian Drought 2006-2011', 'Arab Spring Food Prices 2010'])
ON CONFLICT DO NOTHING;
