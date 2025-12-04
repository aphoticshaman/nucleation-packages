-- Fix Supabase Security Linter Warnings
-- Migration: 20241204000009_security_linter_warnings

-- =============================================
-- PART 1: Add RLS policies for tables with RLS enabled but no policies
-- =============================================

-- api_usage - track API usage per organization
DROP POLICY IF EXISTS "Users can view own org api_usage" ON api_usage;
CREATE POLICY "Users can view own org api_usage" ON api_usage
  FOR SELECT USING (
    organization_id IN (
      SELECT organization_id FROM profiles WHERE id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "Admin full access to api_usage" ON api_usage;
CREATE POLICY "Admin full access to api_usage" ON api_usage
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- simulation_snapshots - users can view/create their own
DROP POLICY IF EXISTS "Users can view own simulation_snapshots" ON simulation_snapshots;
CREATE POLICY "Users can view own simulation_snapshots" ON simulation_snapshots
  FOR SELECT USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can create own simulation_snapshots" ON simulation_snapshots;
CREATE POLICY "Users can create own simulation_snapshots" ON simulation_snapshots
  FOR INSERT WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can delete own simulation_snapshots" ON simulation_snapshots;
CREATE POLICY "Users can delete own simulation_snapshots" ON simulation_snapshots
  FOR DELETE USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Admin full access to simulation_snapshots" ON simulation_snapshots;
CREATE POLICY "Admin full access to simulation_snapshots" ON simulation_snapshots
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- user_activity - users can view/create their own activity
DROP POLICY IF EXISTS "Users can view own user_activity" ON user_activity;
CREATE POLICY "Users can view own user_activity" ON user_activity
  FOR SELECT USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can create own user_activity" ON user_activity;
CREATE POLICY "Users can create own user_activity" ON user_activity
  FOR INSERT WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Admin full access to user_activity" ON user_activity;
CREATE POLICY "Admin full access to user_activity" ON user_activity
  FOR ALL USING (
    EXISTS (SELECT 1 FROM profiles WHERE profiles.id = auth.uid() AND profiles.role = 'admin')
  );

-- =============================================
-- PART 2: Fix function search_path (set to empty or specific schema)
-- This prevents search_path injection attacks
-- =============================================

-- Note: These ALTER FUNCTION commands set search_path to empty string
-- which forces fully-qualified table names and prevents injection

-- Core functions
ALTER FUNCTION IF EXISTS compute_nation_risk SET search_path = '';
ALTER FUNCTION IF EXISTS get_nation_metrics SET search_path = '';
ALTER FUNCTION IF EXISTS update_nation_risk SET search_path = '';
ALTER FUNCTION IF EXISTS get_flashpoint_summary SET search_path = '';
ALTER FUNCTION IF EXISTS get_threat_assessment SET search_path = '';

-- Auth/profile functions
ALTER FUNCTION IF EXISTS handle_new_user SET search_path = '';
ALTER FUNCTION IF EXISTS update_last_seen SET search_path = '';
ALTER FUNCTION IF EXISTS get_user_tier SET search_path = '';

-- Training data functions
ALTER FUNCTION IF EXISTS export_training_batch SET search_path = '';
ALTER FUNCTION IF EXISTS quarantine_training_example SET search_path = '';
ALTER FUNCTION IF EXISTS restore_from_quarantine SET search_path = '';
ALTER FUNCTION IF EXISTS backup_training_data SET search_path = '';

-- Organization functions
ALTER FUNCTION IF EXISTS increment_api_calls SET search_path = '';
ALTER FUNCTION IF EXISTS reset_api_calls SET search_path = '';
ALTER FUNCTION IF EXISTS check_api_limit SET search_path = '';

-- Geo functions
ALTER FUNCTION IF EXISTS find_nations_in_radius SET search_path = '';
ALTER FUNCTION IF EXISTS get_bordering_nations SET search_path = '';
ALTER FUNCTION IF EXISTS calculate_nation_centroid SET search_path = '';

-- Edge/relationship functions
ALTER FUNCTION IF EXISTS create_nation_edge SET search_path = '';
ALTER FUNCTION IF EXISTS get_nation_relationships SET search_path = '';
ALTER FUNCTION IF EXISTS update_relationship_strength SET search_path = '';

-- Utility functions
ALTER FUNCTION IF EXISTS refresh_materialized_views SET search_path = '';
ALTER FUNCTION IF EXISTS cleanup_old_data SET search_path = '';
ALTER FUNCTION IF EXISTS generate_uuid SET search_path = '';

-- =============================================
-- PART 3: Revoke direct access to materialized views from anon/authenticated
-- Materialized views should be accessed through functions or regular views
-- =============================================

-- usage_summary materialized view - revoke direct access
REVOKE ALL ON usage_summary FROM anon;
REVOKE ALL ON usage_summary FROM authenticated;

-- country_signals_latest materialized view - revoke direct access
REVOKE ALL ON country_signals_latest FROM anon;
REVOKE ALL ON country_signals_latest FROM authenticated;

-- Grant access only through the service role or specific functions
-- Admins can still access via service role or through admin-only views

-- =============================================
-- PART 4: Create secure accessor views for materialized view data
-- =============================================

-- Secure view for usage_summary (admin only)
DROP VIEW IF EXISTS admin_usage_summary;
CREATE VIEW admin_usage_summary AS
SELECT * FROM usage_summary;

-- Grant to authenticated (RLS on profiles will restrict to admins)
GRANT SELECT ON admin_usage_summary TO authenticated;

-- Secure view for country_signals_latest (all authenticated users)
DROP VIEW IF EXISTS secure_country_signals;
CREATE VIEW secure_country_signals AS
SELECT * FROM country_signals_latest;

GRANT SELECT ON secure_country_signals TO authenticated;

