-- Security Fixes Migration
-- Addresses Supabase linter warnings for RLS and SECURITY DEFINER views

-- ============================================
-- 1. Enable RLS on tables that are missing it
-- ============================================

-- training_examples - CRITICAL: Contains all our training data
ALTER TABLE IF EXISTS public.training_examples ENABLE ROW LEVEL SECURITY;

-- training_backups - Contains backup data
ALTER TABLE IF EXISTS public.training_backups ENABLE ROW LEVEL SECURITY;

-- training_quarantine - Contains quarantined data
ALTER TABLE IF EXISTS public.training_quarantine ENABLE ROW LEVEL SECURITY;

-- nation_changes - Historical nation change records
ALTER TABLE IF EXISTS public.nation_changes ENABLE ROW LEVEL SECURITY;

-- predictions - Flywheel predictions
ALTER TABLE IF EXISTS public.predictions ENABLE ROW LEVEL SECURITY;

-- known_cascade_patterns - Cascade detection patterns
ALTER TABLE IF EXISTS public.known_cascade_patterns ENABLE ROW LEVEL SECURITY;

-- Note: spatial_ref_sys is a PostGIS system table and should be excluded from public schema
-- We'll handle it separately below

-- ============================================
-- 2. Create RLS policies for training_examples
-- ============================================

-- Drop existing policies if any
DROP POLICY IF EXISTS "training_examples_read_authenticated" ON public.training_examples;
DROP POLICY IF EXISTS "training_examples_insert_service" ON public.training_examples;
DROP POLICY IF EXISTS "training_examples_update_service" ON public.training_examples;
DROP POLICY IF EXISTS "training_examples_delete_service" ON public.training_examples;

-- Authenticated users can read training examples (for the flywheel to work)
CREATE POLICY "training_examples_read_authenticated"
ON public.training_examples FOR SELECT
TO authenticated
USING (true);

-- Only service role can insert (API routes use service role)
CREATE POLICY "training_examples_insert_service"
ON public.training_examples FOR INSERT
TO service_role
WITH CHECK (true);

-- Only service role can update
CREATE POLICY "training_examples_update_service"
ON public.training_examples FOR UPDATE
TO service_role
USING (true)
WITH CHECK (true);

-- Only service role can delete
CREATE POLICY "training_examples_delete_service"
ON public.training_examples FOR DELETE
TO service_role
USING (true);

-- ============================================
-- 3. Create RLS policies for training_backups
-- ============================================

DROP POLICY IF EXISTS "training_backups_read_authenticated" ON public.training_backups;
DROP POLICY IF EXISTS "training_backups_insert_service" ON public.training_backups;
DROP POLICY IF EXISTS "training_backups_update_service" ON public.training_backups;
DROP POLICY IF EXISTS "training_backups_delete_service" ON public.training_backups;

-- Only authenticated users can read backups
CREATE POLICY "training_backups_read_authenticated"
ON public.training_backups FOR SELECT
TO authenticated
USING (true);

-- Only service role can manage backups
CREATE POLICY "training_backups_insert_service"
ON public.training_backups FOR INSERT
TO service_role
WITH CHECK (true);

CREATE POLICY "training_backups_update_service"
ON public.training_backups FOR UPDATE
TO service_role
USING (true);

CREATE POLICY "training_backups_delete_service"
ON public.training_backups FOR DELETE
TO service_role
USING (true);

-- ============================================
-- 4. Create RLS policies for training_quarantine
-- ============================================

DROP POLICY IF EXISTS "training_quarantine_read_authenticated" ON public.training_quarantine;
DROP POLICY IF EXISTS "training_quarantine_manage_service" ON public.training_quarantine;

CREATE POLICY "training_quarantine_read_authenticated"
ON public.training_quarantine FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "training_quarantine_manage_service"
ON public.training_quarantine FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ============================================
-- 5. Create RLS policies for nation_changes
-- ============================================

DROP POLICY IF EXISTS "nation_changes_read_authenticated" ON public.nation_changes;
DROP POLICY IF EXISTS "nation_changes_manage_service" ON public.nation_changes;

CREATE POLICY "nation_changes_read_authenticated"
ON public.nation_changes FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "nation_changes_manage_service"
ON public.nation_changes FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ============================================
-- 6. Create RLS policies for predictions
-- ============================================

DROP POLICY IF EXISTS "predictions_read_authenticated" ON public.predictions;
DROP POLICY IF EXISTS "predictions_manage_service" ON public.predictions;

CREATE POLICY "predictions_read_authenticated"
ON public.predictions FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "predictions_manage_service"
ON public.predictions FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ============================================
-- 7. Create RLS policies for known_cascade_patterns
-- ============================================

DROP POLICY IF EXISTS "cascade_patterns_read" ON public.known_cascade_patterns;
DROP POLICY IF EXISTS "cascade_patterns_manage_service" ON public.known_cascade_patterns;

CREATE POLICY "cascade_patterns_read"
ON public.known_cascade_patterns FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "cascade_patterns_manage_service"
ON public.known_cascade_patterns FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ============================================
-- 8. Fix SECURITY DEFINER views
-- These views need to be recreated without SECURITY DEFINER
-- or with proper security invoker settings
-- ============================================

-- Recreate views as SECURITY INVOKER (the default, safer option)
-- This means the view will respect the RLS policies of the querying user

-- admin_consumer_overview
DROP VIEW IF EXISTS public.admin_consumer_overview;
CREATE VIEW public.admin_consumer_overview
WITH (security_invoker = true)
AS
SELECT
  p.id,
  p.email,
  p.full_name,
  p.skill_level,
  p.created_at,
  p.updated_at,
  COALESCE(
    (SELECT COUNT(*) FROM learning_events le WHERE le.user_id = p.id),
    0
  ) as event_count
FROM profiles p
WHERE p.role = 'user';

-- admin_enterprise_overview
DROP VIEW IF EXISTS public.admin_enterprise_overview;
CREATE VIEW public.admin_enterprise_overview
WITH (security_invoker = true)
AS
SELECT
  p.id,
  p.email,
  p.full_name,
  p.company,
  p.created_at,
  p.updated_at,
  c.tier,
  c.api_enabled,
  (SELECT COUNT(*) FROM api_keys ak WHERE ak.client_id = c.id AND ak.revoked = false) as active_keys
FROM profiles p
LEFT JOIN clients c ON c.user_id = p.id
WHERE p.role = 'enterprise';

-- admin_dashboard_stats
DROP VIEW IF EXISTS public.admin_dashboard_stats;
CREATE VIEW public.admin_dashboard_stats
WITH (security_invoker = true)
AS
SELECT
  (SELECT COUNT(*) FROM profiles WHERE role = 'user') as consumer_count,
  (SELECT COUNT(*) FROM profiles WHERE role = 'enterprise') as enterprise_count,
  (SELECT COUNT(*) FROM profiles WHERE role = 'admin') as admin_count,
  (SELECT COUNT(*) FROM training_examples) as training_examples_count,
  (SELECT COUNT(*) FROM learning_events) as learning_events_count,
  (SELECT COUNT(*) FROM nations WHERE status = 'active') as active_nations_count,
  (SELECT AVG(quality_score) FROM training_examples) as avg_training_quality;

-- exportable_training_data
DROP VIEW IF EXISTS public.exportable_training_data;
CREATE VIEW public.exportable_training_data
WITH (security_invoker = true)
AS
SELECT
  id,
  domain,
  input,
  output,
  quality_score,
  weight,
  metadata,
  created_at
FROM training_examples
WHERE quality_score >= 0.5
ORDER BY weight DESC, quality_score DESC;

-- nations_at_risk
DROP VIEW IF EXISTS public.nations_at_risk;
CREATE VIEW public.nations_at_risk
WITH (security_invoker = true)
AS
SELECT
  code,
  name,
  region,
  transition_risk,
  basin_strength,
  last_event,
  status
FROM nations
WHERE transition_risk > 0.6
  AND status = 'active'
ORDER BY transition_risk DESC;

-- admin_trial_invites - Check if exists first
DROP VIEW IF EXISTS public.admin_trial_invites;
-- Recreate only if the underlying table exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trial_invites') THEN
    EXECUTE '
      CREATE VIEW public.admin_trial_invites
      WITH (security_invoker = true)
      AS
      SELECT * FROM trial_invites
      ORDER BY created_at DESC
    ';
  END IF;
END $$;

-- training_data_stats
DROP VIEW IF EXISTS public.training_data_stats;
CREATE VIEW public.training_data_stats
WITH (security_invoker = true)
AS
SELECT
  domain,
  COUNT(*) as example_count,
  AVG(quality_score) as avg_quality,
  AVG(weight) as avg_weight,
  MAX(created_at) as last_created
FROM training_examples
GROUP BY domain
ORDER BY example_count DESC;

-- country_risk_score
DROP VIEW IF EXISTS public.country_risk_score;
CREATE VIEW public.country_risk_score
WITH (security_invoker = true)
AS
SELECT
  n.code,
  n.name,
  n.region,
  n.transition_risk,
  n.basin_strength,
  n.status,
  COALESCE(
    (SELECT COUNT(*) FROM learning_events le WHERE le.nation_code = n.code),
    0
  ) as event_count,
  CASE
    WHEN n.transition_risk > 0.8 THEN 'CRITICAL'
    WHEN n.transition_risk > 0.6 THEN 'HIGH'
    WHEN n.transition_risk > 0.4 THEN 'MODERATE'
    WHEN n.transition_risk > 0.2 THEN 'LOW'
    ELSE 'STABLE'
  END as risk_level
FROM nations n
WHERE n.status = 'active';

-- nations_geojson
DROP VIEW IF EXISTS public.nations_geojson;
CREATE VIEW public.nations_geojson
WITH (security_invoker = true)
AS
SELECT
  code,
  name,
  region,
  transition_risk,
  basin_strength,
  position,
  velocity,
  status,
  last_event
FROM nations
WHERE status IN ('active', 'disputed');

-- edges_geojson
DROP VIEW IF EXISTS public.edges_geojson;
CREATE VIEW public.edges_geojson
WITH (security_invoker = true)
AS
SELECT
  ie.id,
  ie.source_code,
  ie.target_code,
  ie.influence_type,
  ie.strength,
  ie.created_at,
  sn.name as source_name,
  tn.name as target_name
FROM influence_edges ie
LEFT JOIN nations sn ON sn.code = ie.source_code
LEFT JOIN nations tn ON tn.code = ie.target_code;

-- ============================================
-- 9. Handle spatial_ref_sys (PostGIS table)
-- This is a system table that shouldn't be in public schema
-- We can either move it or revoke public access
-- ============================================

-- Revoke direct access to spatial_ref_sys from anon/authenticated
-- (service_role will still have access for PostGIS functions)
REVOKE ALL ON public.spatial_ref_sys FROM anon;
REVOKE ALL ON public.spatial_ref_sys FROM authenticated;

-- Grant only necessary SELECT to authenticated for spatial queries
GRANT SELECT ON public.spatial_ref_sys TO authenticated;

-- ============================================
-- 10. Grant view access to authenticated users
-- ============================================

GRANT SELECT ON public.admin_consumer_overview TO authenticated;
GRANT SELECT ON public.admin_enterprise_overview TO authenticated;
GRANT SELECT ON public.admin_dashboard_stats TO authenticated;
GRANT SELECT ON public.exportable_training_data TO authenticated;
GRANT SELECT ON public.nations_at_risk TO authenticated;
GRANT SELECT ON public.training_data_stats TO authenticated;
GRANT SELECT ON public.country_risk_score TO authenticated;
GRANT SELECT ON public.nations_geojson TO authenticated;
GRANT SELECT ON public.edges_geojson TO authenticated;

COMMENT ON TABLE public.training_examples IS 'Training data for model fine-tuning. RLS enabled - read by authenticated, write by service_role only.';
