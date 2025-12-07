-- Complete Security Linter Fix
-- Migration: 20241206000001_security_linter_complete_fix
--
-- Fixes:
-- 1. SECURITY DEFINER views -> SECURITY INVOKER (explicit)
-- 2. Function search_path issues
-- 3. RLS on spatial_ref_sys
-- 4. Materialized views API access
--
-- NOTE: All view creations wrapped in DO blocks to skip gracefully if tables don't exist

-- =============================================
-- PART 1: Fix SECURITY DEFINER views with explicit security_invoker
-- Each wrapped in exception handler to skip if dependent tables missing
-- =============================================

-- admin_consumer_overview
DO $$
BEGIN
  DROP VIEW IF EXISTS admin_consumer_overview CASCADE;
  CREATE VIEW admin_consumer_overview
  WITH (security_invoker = true)
  AS
  SELECT
    p.id,
    p.email,
    p.full_name,
    p.last_seen_at,
    p.is_active
  FROM profiles p
  WHERE p.role = 'consumer';
  RAISE NOTICE 'Created view: admin_consumer_overview';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped admin_consumer_overview: %', SQLERRM;
END $$;

-- admin_enterprise_overview
DO $$
BEGIN
  DROP VIEW IF EXISTS admin_enterprise_overview CASCADE;
  CREATE VIEW admin_enterprise_overview
  WITH (security_invoker = true)
  AS
  SELECT
    o.id,
    o.name,
    o.slug,
    o.created_at,
    o.updated_at
  FROM organizations o;
  RAISE NOTICE 'Created view: admin_enterprise_overview';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped admin_enterprise_overview: %', SQLERRM;
END $$;

-- admin_dashboard_stats
DO $$
BEGIN
  DROP VIEW IF EXISTS admin_dashboard_stats CASCADE;
  CREATE VIEW admin_dashboard_stats
  WITH (security_invoker = true)
  AS
  SELECT
    (SELECT COUNT(*) FROM profiles WHERE role = 'consumer') as total_consumers,
    (SELECT COUNT(*) FROM organizations) as total_enterprise;
  RAISE NOTICE 'Created view: admin_dashboard_stats';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped admin_dashboard_stats: %', SQLERRM;
END $$;

-- admin_trial_invites
DO $$
BEGIN
  DROP VIEW IF EXISTS admin_trial_invites CASCADE;
  CREATE VIEW admin_trial_invites
  WITH (security_invoker = true)
  AS
  SELECT * FROM trial_invites;
  RAISE NOTICE 'Created view: admin_trial_invites';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped admin_trial_invites: %', SQLERRM;
END $$;

-- admin_active_sessions (skip if user_sessions doesn't exist)
DO $$
BEGIN
  DROP VIEW IF EXISTS admin_active_sessions CASCADE;
  CREATE VIEW admin_active_sessions
  WITH (security_invoker = true)
  AS
  SELECT
    s.id,
    s.user_id,
    p.email,
    p.full_name
  FROM user_sessions s
  JOIN profiles p ON s.user_id = p.id;
  RAISE NOTICE 'Created view: admin_active_sessions';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped admin_active_sessions: %', SQLERRM;
END $$;

-- exportable_training_data
DO $$
BEGIN
  DROP VIEW IF EXISTS exportable_training_data CASCADE;
  CREATE VIEW exportable_training_data
  WITH (security_invoker = true)
  AS
  SELECT
    id,
    instruction,
    input,
    output,
    category,
    source,
    created_at
  FROM training_examples
  WHERE is_exported = false AND quality_score >= 0.7;
  RAISE NOTICE 'Created view: exportable_training_data';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped exportable_training_data: %', SQLERRM;
END $$;

-- training_data_stats
DO $$
BEGIN
  DROP VIEW IF EXISTS training_data_stats CASCADE;
  CREATE VIEW training_data_stats
  WITH (security_invoker = true)
  AS
  SELECT
    COUNT(*) as total_examples,
    COUNT(*) FILTER (WHERE is_exported = true) as exported_count,
    COUNT(*) FILTER (WHERE quality_score >= 0.7) as high_quality_count,
    AVG(quality_score) as avg_quality_score
  FROM training_examples;
  RAISE NOTICE 'Created view: training_data_stats';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped training_data_stats: %', SQLERRM;
END $$;

-- country_risk_score
DO $$
BEGIN
  DROP VIEW IF EXISTS country_risk_score CASCADE;
  CREATE VIEW country_risk_score
  WITH (security_invoker = true)
  AS
  SELECT
    n.iso_a3,
    n.name,
    COALESCE(nr.overall_risk, 0.5) as risk_score,
    nr.updated_at
  FROM nations n
  LEFT JOIN nation_risk nr ON n.iso_a3 = nr.iso_a3;
  RAISE NOTICE 'Created view: country_risk_score';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped country_risk_score: %', SQLERRM;
END $$;

-- nations_at_risk
DO $$
BEGIN
  DROP VIEW IF EXISTS nations_at_risk CASCADE;
  CREATE VIEW nations_at_risk
  WITH (security_invoker = true)
  AS
  SELECT
    n.iso_a3,
    n.name,
    nr.overall_risk,
    nr.political_risk,
    nr.economic_risk,
    nr.social_risk
  FROM nations n
  JOIN nation_risk nr ON n.iso_a3 = nr.iso_a3
  WHERE nr.overall_risk > 0.7;
  RAISE NOTICE 'Created view: nations_at_risk';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped nations_at_risk: %', SQLERRM;
END $$;

-- disputed_nations
DO $$
BEGIN
  DROP VIEW IF EXISTS disputed_nations CASCADE;
  CREATE VIEW disputed_nations
  WITH (security_invoker = true)
  AS
  SELECT * FROM nations WHERE status = 'disputed' OR disputed = true;
  RAISE NOTICE 'Created view: disputed_nations';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped disputed_nations: %', SQLERRM;
END $$;

-- active_nations
DO $$
BEGIN
  DROP VIEW IF EXISTS active_nations CASCADE;
  CREATE VIEW active_nations
  WITH (security_invoker = true)
  AS
  SELECT * FROM nations WHERE is_active = true;
  RAISE NOTICE 'Created view: active_nations';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped active_nations: %', SQLERRM;
END $$;

-- nations_geojson
DO $$
BEGIN
  DROP VIEW IF EXISTS nations_geojson CASCADE;
  CREATE VIEW nations_geojson
  WITH (security_invoker = true)
  AS
  SELECT
    iso_a3,
    name,
    ST_AsGeoJSON(geometry)::jsonb as geometry
  FROM nations
  WHERE geometry IS NOT NULL;
  RAISE NOTICE 'Created view: nations_geojson';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped nations_geojson: %', SQLERRM;
END $$;

-- edges_geojson
DO $$
BEGIN
  DROP VIEW IF EXISTS edges_geojson CASCADE;
  CREATE VIEW edges_geojson
  WITH (security_invoker = true)
  AS
  SELECT
    id,
    source_iso,
    target_iso,
    relationship_type,
    ST_AsGeoJSON(geometry)::jsonb as geometry
  FROM nation_edges
  WHERE geometry IS NOT NULL;
  RAISE NOTICE 'Created view: edges_geojson';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped edges_geojson: %', SQLERRM;
END $$;

-- =============================================
-- PART 2: Fix spatial_ref_sys RLS
-- =============================================

DO $$
BEGIN
  ALTER TABLE public.spatial_ref_sys ENABLE ROW LEVEL SECURITY;
  DROP POLICY IF EXISTS "Allow read access to spatial_ref_sys" ON public.spatial_ref_sys;
  CREATE POLICY "Allow read access to spatial_ref_sys" ON public.spatial_ref_sys
    FOR SELECT USING (true);
  RAISE NOTICE 'RLS enabled on spatial_ref_sys';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Could not modify spatial_ref_sys: %', SQLERRM;
END $$;

-- =============================================
-- PART 3: Fix function search_path
-- =============================================

DO $$
BEGIN
  ALTER FUNCTION public.cleanup_old_learning_events() SET search_path = '';
  RAISE NOTICE 'Fixed search_path: cleanup_old_learning_events';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped cleanup_old_learning_events: %', SQLERRM;
END $$;

DO $$
BEGIN
  ALTER FUNCTION public.update_updated_at() SET search_path = '';
  RAISE NOTICE 'Fixed search_path: update_updated_at';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped update_updated_at: %', SQLERRM;
END $$;

DO $$
BEGIN
  ALTER FUNCTION public.compute_training_metrics() SET search_path = '';
  RAISE NOTICE 'Fixed search_path: compute_training_metrics';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped compute_training_metrics: %', SQLERRM;
END $$;

DO $$
BEGIN
  ALTER FUNCTION public.export_training_batch(timestamptz, timestamptz, float) SET search_path = '';
  RAISE NOTICE 'Fixed search_path: export_training_batch';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped export_training_batch: %', SQLERRM;
END $$;

-- =============================================
-- PART 4: Restrict materialized views from anon
-- =============================================

DO $$
BEGIN
  REVOKE ALL ON public.usage_summary FROM anon;
  GRANT SELECT ON public.usage_summary TO authenticated;
  RAISE NOTICE 'Fixed permissions: usage_summary';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped usage_summary: %', SQLERRM;
END $$;

DO $$
BEGIN
  REVOKE ALL ON public.country_signals_latest FROM anon;
  GRANT SELECT ON public.country_signals_latest TO authenticated;
  RAISE NOTICE 'Fixed permissions: country_signals_latest';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Skipped country_signals_latest: %', SQLERRM;
END $$;

-- =============================================
-- PART 5: Grant permissions on views (if they exist)
-- =============================================

DO $$
BEGIN
  GRANT SELECT ON admin_consumer_overview TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON admin_enterprise_overview TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON admin_dashboard_stats TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON admin_trial_invites TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON admin_active_sessions TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON exportable_training_data TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON training_data_stats TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON country_risk_score TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON nations_at_risk TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON disputed_nations TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON active_nations TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON nations_geojson TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$
BEGIN
  GRANT SELECT ON edges_geojson TO authenticated;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

RAISE NOTICE 'Security linter fix migration complete';
