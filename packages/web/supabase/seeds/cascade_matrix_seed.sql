-- Seed data for domain_cascade_matrix
-- Represents historical cross-domain event propagation patterns

INSERT INTO domain_cascade_matrix (trigger_domain, effect_domain, co_occurrences, avg_lag_hours, correlation_strength, last_observed)
VALUES
  -- Economic → Political cascades
  ('economic', 'political', 847, 72.5, 0.78, NOW() - INTERVAL '2 hours'),
  ('economic', 'social', 623, 48.0, 0.65, NOW() - INTERVAL '4 hours'),
  ('economic', 'security', 412, 96.0, 0.52, NOW() - INTERVAL '12 hours'),

  -- Security → Multiple domains
  ('security', 'political', 756, 24.0, 0.82, NOW() - INTERVAL '1 hour'),
  ('security', 'economic', 534, 48.0, 0.61, NOW() - INTERVAL '6 hours'),
  ('security', 'humanitarian', 489, 12.0, 0.74, NOW() - INTERVAL '3 hours'),
  ('security', 'migration', 367, 168.0, 0.58, NOW() - INTERVAL '24 hours'),

  -- Political → Other domains
  ('political', 'economic', 698, 36.0, 0.71, NOW() - INTERVAL '5 hours'),
  ('political', 'diplomatic', 582, 6.0, 0.85, NOW() - INTERVAL '30 minutes'),
  ('political', 'social', 445, 24.0, 0.54, NOW() - INTERVAL '8 hours'),

  -- Natural disasters → Cascades
  ('seismic', 'humanitarian', 234, 2.0, 0.91, NOW() - INTERVAL '48 hours'),
  ('seismic', 'economic', 189, 72.0, 0.67, NOW() - INTERVAL '72 hours'),
  ('seismic', 'infrastructure', 312, 1.0, 0.94, NOW() - INTERVAL '48 hours'),

  -- Energy → Economic
  ('energy', 'economic', 567, 12.0, 0.79, NOW() - INTERVAL '6 hours'),
  ('energy', 'political', 345, 48.0, 0.62, NOW() - INTERVAL '18 hours'),
  ('energy', 'industrial', 478, 6.0, 0.83, NOW() - INTERVAL '4 hours'),

  -- Trade/Sanctions cascades
  ('trade', 'economic', 612, 24.0, 0.76, NOW() - INTERVAL '12 hours'),
  ('trade', 'diplomatic', 389, 12.0, 0.68, NOW() - INTERVAL '8 hours'),
  ('sanctions', 'economic', 534, 48.0, 0.81, NOW() - INTERVAL '24 hours'),
  ('sanctions', 'political', 423, 24.0, 0.72, NOW() - INTERVAL '16 hours'),

  -- Cyber cascades
  ('cyber', 'economic', 289, 6.0, 0.58, NOW() - INTERVAL '36 hours'),
  ('cyber', 'infrastructure', 356, 0.5, 0.87, NOW() - INTERVAL '12 hours'),
  ('cyber', 'security', 412, 2.0, 0.79, NOW() - INTERVAL '8 hours'),

  -- Social unrest cascades
  ('social', 'political', 534, 48.0, 0.69, NOW() - INTERVAL '10 hours'),
  ('social', 'security', 378, 72.0, 0.55, NOW() - INTERVAL '20 hours'),
  ('social', 'economic', 267, 96.0, 0.47, NOW() - INTERVAL '32 hours'),

  -- Migration cascades
  ('migration', 'political', 312, 168.0, 0.63, NOW() - INTERVAL '48 hours'),
  ('migration', 'social', 289, 120.0, 0.58, NOW() - INTERVAL '36 hours'),
  ('migration', 'economic', 234, 240.0, 0.51, NOW() - INTERVAL '72 hours')

ON CONFLICT (trigger_domain, effect_domain)
DO UPDATE SET
  co_occurrences = EXCLUDED.co_occurrences,
  avg_lag_hours = EXCLUDED.avg_lag_hours,
  correlation_strength = EXCLUDED.correlation_strength,
  last_observed = EXCLUDED.last_observed,
  updated_at = NOW();

-- Also seed some country_signals for World Bank demo
INSERT INTO country_signals (country_code, country_name, indicator, value, year, source, metadata)
VALUES
  -- USA
  ('USA', 'United States', 'gdp_growth', 2.5, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('USA', 'United States', 'inflation', 3.4, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('USA', 'United States', 'unemployment', 3.7, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('USA', 'United States', 'debt_to_gdp', 123.4, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- China
  ('CHN', 'China', 'gdp_growth', 5.2, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('CHN', 'China', 'inflation', 0.2, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('CHN', 'China', 'unemployment', 5.1, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('CHN', 'China', 'debt_to_gdp', 83.6, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- Germany
  ('DEU', 'Germany', 'gdp_growth', 0.3, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('DEU', 'Germany', 'inflation', 2.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('DEU', 'Germany', 'unemployment', 5.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('DEU', 'Germany', 'debt_to_gdp', 66.3, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- Russia
  ('RUS', 'Russia', 'gdp_growth', 3.6, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('RUS', 'Russia', 'inflation', 7.4, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('RUS', 'Russia', 'unemployment', 2.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('RUS', 'Russia', 'debt_to_gdp', 18.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- Ukraine
  ('UKR', 'Ukraine', 'gdp_growth', 5.3, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('UKR', 'Ukraine', 'inflation', 5.8, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('UKR', 'Ukraine', 'unemployment', 14.5, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('UKR', 'Ukraine', 'debt_to_gdp', 84.4, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- Japan
  ('JPN', 'Japan', 'gdp_growth', 1.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('JPN', 'Japan', 'inflation', 2.7, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('JPN', 'Japan', 'unemployment', 2.5, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('JPN', 'Japan', 'debt_to_gdp', 255.2, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- India
  ('IND', 'India', 'gdp_growth', 6.8, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('IND', 'India', 'inflation', 5.4, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('IND', 'India', 'unemployment', 7.2, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('IND', 'India', 'debt_to_gdp', 81.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),

  -- Brazil
  ('BRA', 'Brazil', 'gdp_growth', 2.9, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('BRA', 'Brazil', 'inflation', 4.5, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('BRA', 'Brazil', 'unemployment', 7.8, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}'),
  ('BRA', 'Brazil', 'debt_to_gdp', 74.4, 2024, 'worldbank', '{"fetched_at": "2024-12-16"}')

ON CONFLICT (country_code, indicator, year)
DO UPDATE SET
  value = EXCLUDED.value,
  updated_at = NOW();
