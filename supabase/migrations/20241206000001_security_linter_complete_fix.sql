-- Complete Security Linter Fix
-- Migration: 20241206000001_security_linter_complete_fix
--
-- Fixes:
-- 1. SECURITY DEFINER views -> SECURITY INVOKER (explicit)
-- 2. Function search_path issues
-- 3. RLS on spatial_ref_sys
-- 4. Materialized views API access

-- =============================================
-- PART 1: Fix SECURITY DEFINER views with explicit security_invoker
-- PostgreSQL 15+ requires explicit security_invoker = true
-- =============================================

-- admin_consumer_overview
DROP VIEW IF EXISTS admin_consumer_overview CASCADE;
CREATE VIEW admin_consumer_overview
WITH (security_invoker = true)
AS
SELECT
  p.id,
  p.email,
  p.full_name,
  p.last_seen_at,
  p.is_active,
  COALESCE((SELECT COUNT(*) FROM user_activity WHERE user_id = p.id), 0) as simulation_count,
  COALESCE((SELECT COUNT(*) FROM user_activity WHERE user_id = p.id AND created_at > NOW() - INTERVAL '7 days'), 0) as actions_7d
FROM profiles p
WHERE p.role = 'consumer';

-- admin_enterprise_overview
DROP VIEW IF EXISTS admin_enterprise_overview CASCADE;
CREATE VIEW admin_enterprise_overview
WITH (security_invoker = true)
AS
SELECT
  o.id,
  o.name,
  o.slug,
  COALESCE(o.subscription_plan, 'free') as plan,
  COALESCE(o.subscription_status, 'inactive') as status,
  (SELECT COUNT(*) FROM profiles WHERE organization_id = o.id) as team_size,
  o.updated_at as last_activity
FROM organizations o;

-- admin_dashboard_stats
DROP VIEW IF EXISTS admin_dashboard_stats CASCADE;
CREATE VIEW admin_dashboard_stats
WITH (security_invoker = true)
AS
SELECT
  (SELECT COUNT(*) FROM profiles WHERE role = 'consumer') as total_consumers,
  (SELECT COUNT(*) FROM organizations) as total_enterprise,
  (SELECT COUNT(*) FROM organizations WHERE subscription_status = 'active') as active_orgs,
  (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '24 hours') as active_24h,
  (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '7 days') as active_7d;

-- admin_trial_invites
DROP VIEW IF EXISTS admin_trial_invites CASCADE;
CREATE VIEW admin_trial_invites
WITH (security_invoker = true)
AS
SELECT * FROM trial_invites;

-- admin_active_sessions
DROP VIEW IF EXISTS admin_active_sessions CASCADE;
CREATE VIEW admin_active_sessions
WITH (security_invoker = true)
AS
SELECT
  s.id,
  s.user_id,
  p.email,
  p.full_name,
  s.created_at as session_start,
  s.last_activity_at,
  s.ip_address,
  s.user_agent
FROM user_sessions s
JOIN profiles p ON s.user_id = p.id
WHERE s.is_active = true;

-- exportable_training_data
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

-- training_data_stats
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

-- country_risk_score
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

-- nations_at_risk
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

-- disputed_nations
DROP VIEW IF EXISTS disputed_nations CASCADE;
CREATE VIEW disputed_nations
WITH (security_invoker = true)
AS
SELECT * FROM nations WHERE status = 'disputed' OR disputed = true;

-- active_nations
DROP VIEW IF EXISTS active_nations CASCADE;
CREATE VIEW active_nations
WITH (security_invoker = true)
AS
SELECT * FROM nations WHERE is_active = true;

-- nations_geojson
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

-- edges_geojson
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

-- =============================================
-- PART 2: Fix spatial_ref_sys RLS
-- This is a PostGIS system table - we can enable RLS but allow all reads
-- =============================================

-- Enable RLS on spatial_ref_sys (PostGIS reference table)
DO $$
BEGIN
  ALTER TABLE IF EXISTS public.spatial_ref_sys ENABLE ROW LEVEL SECURITY;

  -- Allow everyone to read (it's reference data)
  DROP POLICY IF EXISTS "Allow read access to spatial_ref_sys" ON public.spatial_ref_sys;
  CREATE POLICY "Allow read access to spatial_ref_sys" ON public.spatial_ref_sys
    FOR SELECT USING (true);

  RAISE NOTICE 'RLS enabled on spatial_ref_sys';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Could not modify spatial_ref_sys: %', SQLERRM;
END $$;

-- =============================================
-- PART 3: Fix function search_path
-- Set empty search_path for security
-- =============================================

DO $$
BEGIN
  -- Fix cleanup_old_learning_events
  ALTER FUNCTION public.cleanup_old_learning_events() SET search_path = '';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'Function cleanup_old_learning_events does not exist, skipping';
END $$;

DO $$
BEGIN
  -- Fix update_updated_at
  ALTER FUNCTION public.update_updated_at() SET search_path = '';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'Function update_updated_at does not exist, skipping';
END $$;

DO $$
BEGIN
  -- Fix compute_training_metrics
  ALTER FUNCTION public.compute_training_metrics() SET search_path = '';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'Function compute_training_metrics does not exist, skipping';
END $$;

-- Fix export_training_batch (with signature)
DO $$
BEGIN
  ALTER FUNCTION public.export_training_batch(timestamptz, timestamptz, float)
    SET search_path = '';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Could not alter export_training_batch: %', SQLERRM;
END $$;

-- =============================================
-- PART 4: Restrict materialized views from anon
-- =============================================

-- Revoke anon access from materialized views
REVOKE ALL ON public.usage_summary FROM anon;
REVOKE ALL ON public.country_signals_latest FROM anon;

-- Keep authenticated access
GRANT SELECT ON public.usage_summary TO authenticated;
GRANT SELECT ON public.country_signals_latest TO authenticated;

-- =============================================
-- PART 5: Grant permissions on views
-- =============================================

GRANT SELECT ON admin_consumer_overview TO authenticated;
GRANT SELECT ON admin_enterprise_overview TO authenticated;
GRANT SELECT ON admin_dashboard_stats TO authenticated;
GRANT SELECT ON admin_trial_invites TO authenticated;
GRANT SELECT ON admin_active_sessions TO authenticated;
GRANT SELECT ON exportable_training_data TO authenticated;
GRANT SELECT ON training_data_stats TO authenticated;
GRANT SELECT ON country_risk_score TO authenticated;
GRANT SELECT ON nations_at_risk TO authenticated;
GRANT SELECT ON disputed_nations TO authenticated;
GRANT SELECT ON active_nations TO authenticated;
GRANT SELECT ON nations_geojson TO authenticated;
GRANT SELECT ON edges_geojson TO authenticated;

-- =============================================
-- Verification
-- =============================================

DO $$
DECLARE
  definer_count INT;
  no_search_path_count INT;
BEGIN
  -- Check for remaining SECURITY DEFINER views
  SELECT COUNT(*) INTO definer_count
  FROM pg_views
  WHERE schemaname = 'public'
  AND viewname IN (
    'admin_consumer_overview', 'admin_enterprise_overview', 'admin_dashboard_stats',
    'admin_trial_invites', 'admin_active_sessions', 'exportable_training_data',
    'training_data_stats', 'country_risk_score', 'nations_at_risk',
    'disputed_nations', 'active_nations', 'nations_geojson', 'edges_geojson'
  );

  RAISE NOTICE 'Views created: %', definer_count;

  -- Check for functions without search_path
  SELECT COUNT(*) INTO no_search_path_count
  FROM pg_proc p
  JOIN pg_namespace n ON p.pronamespace = n.oid
  WHERE n.nspname = 'public'
  AND p.proname IN ('cleanup_old_learning_events', 'update_updated_at',
                    'compute_training_metrics', 'export_training_batch')
  AND (
    p.proconfig IS NULL
    OR NOT EXISTS (
      SELECT 1 FROM unnest(p.proconfig) AS cfg
      WHERE cfg LIKE 'search_path=%'
    )
  );

  RAISE NOTICE 'Functions still needing search_path fix: %', no_search_path_count;
END $$;
