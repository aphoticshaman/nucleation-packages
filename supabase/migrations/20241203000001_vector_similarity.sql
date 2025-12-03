-- ============================================================
-- VECTOR SIMILARITY SEARCH FOR HISTORICAL CASES
-- Enables pgvector for efficient similarity queries
-- ============================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column for semantic search
ALTER TABLE historical_cases
ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_historical_cases_embedding
ON historical_cases USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Function to find similar historical cases using vector similarity
CREATE OR REPLACE FUNCTION find_similar_cases(
    query_embedding vector(1536),
    query_domain TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    case_id TEXT,
    domain TEXT,
    title TEXT,
    description TEXT,
    outcome TEXT,
    lessons TEXT[],
    severity TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        hc.id,
        hc.case_id,
        hc.domain,
        hc.title,
        hc.description,
        hc.outcome,
        hc.lessons,
        hc.severity,
        1 - (hc.embedding <=> query_embedding) as similarity
    FROM historical_cases hc
    WHERE
        hc.embedding IS NOT NULL
        AND (query_domain IS NULL OR hc.domain = query_domain)
        AND 1 - (hc.embedding <=> query_embedding) > match_threshold
    ORDER BY hc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to find cases by feature similarity (fallback without embeddings)
CREATE OR REPLACE FUNCTION find_cases_by_features(
    query_features JSONB,
    query_domain TEXT DEFAULT NULL,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    case_id TEXT,
    domain TEXT,
    title TEXT,
    description TEXT,
    outcome TEXT,
    lessons TEXT[],
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        hc.id,
        hc.case_id,
        hc.domain,
        hc.title,
        hc.description,
        hc.outcome,
        hc.lessons,
        -- Simple Jaccard-like similarity on feature keys
        (
            SELECT COUNT(*)::FLOAT / GREATEST(
                jsonb_object_keys(hc.features)::INT,
                jsonb_object_keys(query_features)::INT,
                1
            )
            FROM jsonb_object_keys(hc.features) k1
            WHERE EXISTS (SELECT 1 FROM jsonb_object_keys(query_features) k2 WHERE k1 = k2)
        ) as similarity
    FROM historical_cases hc
    WHERE
        (query_domain IS NULL OR hc.domain = query_domain)
        AND hc.verified = TRUE
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Seed some initial historical cases for testing
INSERT INTO historical_cases (case_id, domain, title, description, start_date, end_date, features, outcome, lessons, severity, verified) VALUES
('crimea_2014', 'conflict', '2014 Crimea Annexation',
 'Russia''s annexation of Crimea following political instability in Ukraine. Preceded by similar indicators including military buildup, political rhetoric escalation, and exploitation of ethnic tensions.',
 '2014-02-20', '2014-03-18',
 '{"military_buildup": 0.9, "ethnic_tension": 0.8, "political_instability": 0.85, "external_pressure": 0.7, "economic_stress": 0.6}'::jsonb,
 'Rapid territorial change, prolonged sanctions regime, frozen conflict',
 ARRAY['Speed of action exceeded predictions', 'Economic interdependence did not deter', 'Hybrid warfare tactics effective', 'Information warfare preceded kinetic action'],
 'major', true),

('asian_crisis_1997', 'economic', '1997 Asian Financial Crisis',
 'Currency and financial crisis that began in Thailand and spread across East Asia. Demonstrated contagion patterns in interconnected financial systems.',
 '1997-07-02', '1998-12-31',
 '{"currency_peg_stress": 0.85, "capital_flight": 0.9, "debt_ratio": 0.8, "trade_deficit": 0.7, "regional_contagion": 0.95}'::jsonb,
 'Regional cascade, IMF intervention, 2-year recovery, structural reforms',
 ARRAY['Currency pegs vulnerable to speculation', 'Regional interdependence amplified shock', 'IMF conditionality controversial', 'Moral hazard concerns emerged'],
 'major', true),

('arab_spring_2011', 'political', '2011 Arab Spring',
 'Wave of pro-democracy protests and uprisings across the Arab world, beginning in Tunisia. Social media played unprecedented role in coordination.',
 '2010-12-17', '2012-12-31',
 '{"domestic_unrest": 0.95, "youth_unemployment": 0.85, "social_media_penetration": 0.7, "regime_legitimacy": 0.3, "economic_grievance": 0.8}'::jsonb,
 'Multiple regime transitions, varied outcomes by country, some civil wars',
 ARRAY['Social media accelerated coordination', 'Military stance determined outcomes', 'Economic factors underlay political demands', 'External intervention shaped outcomes'],
 'catastrophic', true),

('lehman_2008', 'economic', '2008 Lehman Brothers Collapse',
 'Bankruptcy of Lehman Brothers triggered global financial crisis. Demonstrated systemic risk in interconnected financial institutions.',
 '2008-09-15', '2009-06-30',
 '{"leverage_ratio": 0.95, "counterparty_risk": 0.9, "liquidity_stress": 0.85, "regulatory_gap": 0.8, "housing_bubble": 0.9}'::jsonb,
 'Global recession, massive government interventions, regulatory overhaul',
 ARRAY['Too big to fail created moral hazard', 'Interconnectedness amplified shock', 'Regulatory arbitrage enabled buildup', 'Central bank action critical'],
 'catastrophic', true),

('covid_2020', 'health', 'COVID-19 Pandemic',
 'Global pandemic caused by SARS-CoV-2 virus. Unprecedented disruption to global economy and society.',
 '2020-01-01', '2023-05-05',
 '{"transmission_rate": 0.9, "healthcare_capacity": 0.4, "supply_chain_disruption": 0.85, "travel_restriction": 0.95, "economic_shock": 0.9}'::jsonb,
 'Global recession, accelerated digitalization, supply chain restructuring',
 ARRAY['Early warning systems inadequate', 'Global coordination challenges', 'Economic vs health tradeoffs', 'Vaccine development accelerated'],
 'catastrophic', true),

('suez_2021', 'economic', '2021 Suez Canal Blockage',
 'Ever Given container ship blocked Suez Canal for 6 days, disrupting global trade.',
 '2021-03-23', '2021-03-29',
 '{"chokepoint_dependency": 0.95, "just_in_time_vulnerability": 0.8, "trade_volume": 0.85, "alternative_routes": 0.3}'::jsonb,
 'Short-term trade disruption, $9.6B daily trade impact, supply chain review',
 ARRAY['Single points of failure in global trade', 'Just-in-time vulnerable to shocks', 'Alternative routes costly', 'Insurance implications significant'],
 'moderate', true)
ON CONFLICT (case_id) DO NOTHING;

COMMENT ON FUNCTION find_similar_cases IS 'Find historical cases similar to a query using vector embeddings';
COMMENT ON FUNCTION find_cases_by_features IS 'Find historical cases by feature similarity (fallback without embeddings)';
