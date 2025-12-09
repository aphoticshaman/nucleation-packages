-- =============================================================================
-- FIX SUPABASE SECURITY LINTER ERRORS
-- =============================================================================
-- Fixes:
-- 1. RLS not enabled on briefing_cache
-- 2. Functions without search_path set
-- 3. Views with SECURITY DEFINER (converted to SECURITY INVOKER where safe)
-- =============================================================================

-- ============================================================
-- 1. ENABLE RLS ON briefing_cache
-- ============================================================
-- This table caches briefing data - should be readable by authenticated users
-- but only writable by service role
ALTER TABLE IF EXISTS public.briefing_cache ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read cache (it's public intel data)
DROP POLICY IF EXISTS "Anyone can read briefing cache" ON public.briefing_cache;
CREATE POLICY "Anyone can read briefing cache" ON public.briefing_cache
  FOR SELECT
  USING (true);

-- Only service role can insert/update (cron jobs)
DROP POLICY IF EXISTS "Service role can manage cache" ON public.briefing_cache;
CREATE POLICY "Service role can manage cache" ON public.briefing_cache
  FOR ALL
  USING (auth.role() = 'service_role')
  WITH CHECK (auth.role() = 'service_role');

-- ============================================================
-- 2. FIX FUNCTIONS WITH MUTABLE SEARCH_PATH
-- ============================================================
-- These functions need explicit search_path to prevent injection attacks

-- Fix reset_daily_email_quota
CREATE OR REPLACE FUNCTION public.reset_daily_email_quota()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  UPDATE public.profiles
  SET
    email_alerts_sent_today = 0,
    email_alerts_last_reset = NOW()
  WHERE email_alerts_last_reset < CURRENT_DATE
     OR email_alerts_last_reset IS NULL;
END;
$$;

-- Fix handle_new_user_email_prefs
CREATE OR REPLACE FUNCTION public.handle_new_user_email_prefs()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  INSERT INTO public.email_export_preferences (user_id, frequency, enabled)
  VALUES (NEW.id, 'daily', true)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$;

-- Fix get_trial_end_date
CREATE OR REPLACE FUNCTION public.get_trial_end_date()
RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
STABLE
SET search_path = public, pg_temp
AS $$
BEGIN
  RETURN NOW() + INTERVAL '14 days';
END;
$$;

-- Fix record_nation_snapshot (if it exists)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'record_nation_snapshot') THEN
    EXECUTE $func$
      CREATE OR REPLACE FUNCTION public.record_nation_snapshot(
        p_nation_code TEXT,
        p_basin_strength NUMERIC,
        p_transition_risk NUMERIC,
        p_regime INTEGER
      )
      RETURNS void
      LANGUAGE plpgsql
      SECURITY DEFINER
      SET search_path = public, pg_temp
      AS $inner$
      BEGIN
        INSERT INTO public.nation_snapshots (nation_code, basin_strength, transition_risk, regime, recorded_at)
        VALUES (p_nation_code, p_basin_strength, p_transition_risk, p_regime, NOW());
      END;
      $inner$
    $func$;
  END IF;
END $$;

-- ============================================================
-- 3. FIX VIEWS WITH SECURITY DEFINER
-- ============================================================
-- These views should use SECURITY INVOKER (default) to respect RLS
-- We recreate them without SECURITY DEFINER

-- Note: Dropping and recreating views is safe - they don't store data

-- Fix nation_trends view
DROP VIEW IF EXISTS public.nation_trends CASCADE;
CREATE VIEW public.nation_trends AS
SELECT
  n.code,
  n.name,
  n.basin_strength,
  n.transition_risk,
  n.regime,
  n.updated_at
FROM public.nations n
ORDER BY n.transition_risk DESC;

-- Fix edges_geojson view
DROP VIEW IF EXISTS public.edges_geojson CASCADE;
CREATE VIEW public.edges_geojson AS
SELECT
  e.id,
  e.source_nation,
  e.target_nation,
  e.edge_type,
  e.weight,
  e.created_at
FROM public.edges e;

-- Fix active_nations view
DROP VIEW IF EXISTS public.active_nations CASCADE;
CREATE VIEW public.active_nations AS
SELECT
  n.code,
  n.name,
  n.basin_strength,
  n.transition_risk,
  n.regime,
  n.updated_at
FROM public.nations n
WHERE n.updated_at > NOW() - INTERVAL '7 days';

-- Fix training_data_stats view
DROP VIEW IF EXISTS public.training_data_stats CASCADE;
CREATE VIEW public.training_data_stats AS
SELECT
  COUNT(*) as total_examples,
  COUNT(DISTINCT category) as categories,
  MAX(created_at) as last_updated
FROM public.training_data;

-- Fix feedback_stats view
DROP VIEW IF EXISTS public.feedback_stats CASCADE;
CREATE VIEW public.feedback_stats AS
SELECT
  status,
  COUNT(*) as count,
  AVG(admin_priority::int) as avg_priority
FROM public.feedback
GROUP BY status;

-- Fix nations_at_risk view
DROP VIEW IF EXISTS public.nations_at_risk CASCADE;
CREATE VIEW public.nations_at_risk AS
SELECT
  n.code,
  n.name,
  n.basin_strength,
  n.transition_risk,
  n.regime
FROM public.nations n
WHERE n.transition_risk > 0.6
ORDER BY n.transition_risk DESC;

-- Fix disputed_nations view
DROP VIEW IF EXISTS public.disputed_nations CASCADE;
CREATE VIEW public.disputed_nations AS
SELECT
  n.code,
  n.name,
  n.basin_strength,
  n.transition_risk
FROM public.nations n
WHERE n.regime = 2 OR n.transition_risk > 0.7;

-- Fix exportable_training_data view
DROP VIEW IF EXISTS public.exportable_training_data CASCADE;
CREATE VIEW public.exportable_training_data AS
SELECT
  id,
  prompt,
  completion,
  category,
  quality_score,
  created_at
FROM public.training_data
WHERE quality_score >= 0.7;

-- Fix country_risk_score view
DROP VIEW IF EXISTS public.country_risk_score CASCADE;
CREATE VIEW public.country_risk_score AS
SELECT
  n.code,
  n.name,
  n.transition_risk as risk_score,
  CASE
    WHEN n.transition_risk > 0.7 THEN 'critical'
    WHEN n.transition_risk > 0.5 THEN 'elevated'
    WHEN n.transition_risk > 0.3 THEN 'moderate'
    ELSE 'low'
  END as risk_level
FROM public.nations n;

-- Fix nations_geojson view
DROP VIEW IF EXISTS public.nations_geojson CASCADE;
CREATE VIEW public.nations_geojson AS
SELECT
  n.code,
  n.name,
  n.basin_strength,
  n.transition_risk,
  n.regime,
  ST_AsGeoJSON(n.geometry)::jsonb as geometry
FROM public.nations n
WHERE n.geometry IS NOT NULL;

-- ============================================================
-- 4. REVOKE PUBLIC ACCESS FROM MATERIALIZED VIEWS
-- ============================================================
-- These contain aggregated data and should only be accessible to authenticated users

-- Revoke anon access from materialized views
REVOKE SELECT ON public.usage_summary FROM anon;
REVOKE SELECT ON public.country_signals_latest FROM anon;

-- Grant only to authenticated users
GRANT SELECT ON public.usage_summary TO authenticated;
GRANT SELECT ON public.country_signals_latest TO authenticated;

-- ============================================================
-- DONE
-- ============================================================
-- Note: spatial_ref_sys is a PostGIS system table - don't enable RLS on it
-- Note: postgis extension in public schema is OK for now (moving it is complex)
