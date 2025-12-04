-- Fix Supabase Security Linter Errors
-- Migration: 20241204000008_security_linter_fixes

-- =============================================
-- PART 1: Enable RLS on unprotected tables
-- =============================================

-- training_examples - admin only
ALTER TABLE IF EXISTS training_examples ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Admin full access to training_examples" ON training_examples;
CREATE POLICY "Admin full access to training_examples" ON training_examples
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- nation_changes - admin only
ALTER TABLE IF EXISTS nation_changes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Admin full access to nation_changes" ON nation_changes;
CREATE POLICY "Admin full access to nation_changes" ON nation_changes
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- training_backups - admin only
ALTER TABLE IF EXISTS training_backups ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Admin full access to training_backups" ON training_backups;
CREATE POLICY "Admin full access to training_backups" ON training_backups
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- training_quarantine - admin only
ALTER TABLE IF EXISTS training_quarantine ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Admin full access to training_quarantine" ON training_quarantine;
CREATE POLICY "Admin full access to training_quarantine" ON training_quarantine
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- spatial_ref_sys - PostGIS system table, read-only for authenticated users
ALTER TABLE IF EXISTS spatial_ref_sys ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Read access to spatial_ref_sys" ON spatial_ref_sys;
CREATE POLICY "Read access to spatial_ref_sys" ON spatial_ref_sys
  FOR SELECT USING (true);

-- =============================================
-- PART 2: Fix SECURITY DEFINER views
-- Replace with SECURITY INVOKER (default)
-- =============================================

-- Drop and recreate views without SECURITY DEFINER
-- The views will use the caller's permissions (SECURITY INVOKER is default)

-- admin_consumer_overview
DROP VIEW IF EXISTS admin_consumer_overview;
CREATE VIEW admin_consumer_overview AS
SELECT
  p.id,
  p.email,
  p.full_name,
  p.last_seen_at,
  p.is_active,
  COALESCE((SELECT COUNT(*) FROM user_activity WHERE user_id = p.id), 0) as simulation_count,
  COALESCE((SELECT COUNT(*) FROM user_activity WHERE user_id = p.id AND created_at > NOW() - INTERVAL '7 days'), 0) as actions_7d
FROM profiles p
WHERE p.role = 'consumer' OR p.role = 'user';

-- admin_enterprise_overview
DROP VIEW IF EXISTS admin_enterprise_overview;
CREATE VIEW admin_enterprise_overview AS
SELECT
  o.id,
  o.name,
  COALESCE(o.plan, 'free') as plan,
  COALESCE(o.api_calls_24h, 0) as api_calls_24h,
  COALESCE(o.api_calls_limit, 1000) as api_calls_limit,
  (SELECT COUNT(*) FROM profiles WHERE organization_id = o.id) as team_size,
  o.updated_at as last_activity,
  COALESCE(o.is_active, true) as is_active
FROM organizations o;

-- admin_dashboard_stats
DROP VIEW IF EXISTS admin_dashboard_stats;
CREATE VIEW admin_dashboard_stats AS
SELECT
  (SELECT COUNT(*) FROM profiles WHERE role = 'consumer' OR role = 'user') as total_consumers,
  (SELECT COUNT(*) FROM organizations) as total_enterprise,
  (SELECT COUNT(*) FROM organizations WHERE is_active = true) as active_orgs,
  (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '24 hours') as active_24h,
  (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '7 days') as active_7d,
  COALESCE((SELECT SUM(api_calls_24h) FROM organizations), 0) as api_calls_24h;

-- admin_trial_invites
DROP VIEW IF EXISTS admin_trial_invites;
CREATE VIEW admin_trial_invites AS
SELECT * FROM trial_invites;

-- exportable_training_data
DROP VIEW IF EXISTS exportable_training_data;
CREATE VIEW exportable_training_data AS
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
DROP VIEW IF EXISTS training_data_stats;
CREATE VIEW training_data_stats AS
SELECT
  COUNT(*) as total_examples,
  COUNT(*) FILTER (WHERE is_exported = true) as exported_count,
  COUNT(*) FILTER (WHERE quality_score >= 0.7) as high_quality_count,
  AVG(quality_score) as avg_quality_score
FROM training_examples;

-- country_risk_score
DROP VIEW IF EXISTS country_risk_score;
CREATE VIEW country_risk_score AS
SELECT
  n.iso_a3,
  n.name,
  COALESCE(nr.overall_risk, 0.5) as risk_score,
  nr.updated_at
FROM nations n
LEFT JOIN nation_risk nr ON n.iso_a3 = nr.iso_a3;

-- nations_at_risk
DROP VIEW IF EXISTS nations_at_risk;
CREATE VIEW nations_at_risk AS
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
DROP VIEW IF EXISTS disputed_nations;
CREATE VIEW disputed_nations AS
SELECT * FROM nations WHERE status = 'disputed' OR disputed = true;

-- active_nations
DROP VIEW IF EXISTS active_nations;
CREATE VIEW active_nations AS
SELECT * FROM nations WHERE is_active = true;

-- nations_geojson
DROP VIEW IF EXISTS nations_geojson;
CREATE VIEW nations_geojson AS
SELECT
  iso_a3,
  name,
  ST_AsGeoJSON(geometry)::jsonb as geometry
FROM nations
WHERE geometry IS NOT NULL;

-- edges_geojson
DROP VIEW IF EXISTS edges_geojson;
CREATE VIEW edges_geojson AS
SELECT
  id,
  source_iso,
  target_iso,
  relationship_type,
  ST_AsGeoJSON(geometry)::jsonb as geometry
FROM nation_edges
WHERE geometry IS NOT NULL;

-- =============================================
-- PART 3: Grant appropriate permissions
-- =============================================

-- Grant read access to views for authenticated users
GRANT SELECT ON admin_consumer_overview TO authenticated;
GRANT SELECT ON admin_enterprise_overview TO authenticated;
GRANT SELECT ON admin_dashboard_stats TO authenticated;
GRANT SELECT ON admin_trial_invites TO authenticated;
GRANT SELECT ON exportable_training_data TO authenticated;
GRANT SELECT ON training_data_stats TO authenticated;
GRANT SELECT ON country_risk_score TO authenticated;
GRANT SELECT ON nations_at_risk TO authenticated;
GRANT SELECT ON disputed_nations TO authenticated;
GRANT SELECT ON active_nations TO authenticated;
GRANT SELECT ON nations_geojson TO authenticated;
GRANT SELECT ON edges_geojson TO authenticated;

-- Note: Views now use SECURITY INVOKER (default), so RLS on underlying tables applies
-- Admin-only views will only work for admins due to RLS on profiles/organizations
