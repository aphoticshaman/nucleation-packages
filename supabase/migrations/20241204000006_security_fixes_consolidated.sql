-- ============================================
-- CONSOLIDATED SECURITY FIXES
-- Run this to fix all RLS and SECURITY DEFINER issues
-- ============================================

-- 1. Enable RLS on all tables that need it
ALTER TABLE IF EXISTS public.training_examples ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.known_cascade_patterns ENABLE ROW LEVEL SECURITY;

-- 2. RLS Policies for training_examples
DROP POLICY IF EXISTS "training_examples_read_authenticated" ON public.training_examples;
DROP POLICY IF EXISTS "training_examples_insert_service" ON public.training_examples;
DROP POLICY IF EXISTS "training_examples_update_service" ON public.training_examples;
DROP POLICY IF EXISTS "training_examples_delete_service" ON public.training_examples;

CREATE POLICY "training_examples_read_authenticated" ON public.training_examples
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "training_examples_insert_service" ON public.training_examples
  FOR INSERT TO service_role WITH CHECK (true);

CREATE POLICY "training_examples_update_service" ON public.training_examples
  FOR UPDATE TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "training_examples_delete_service" ON public.training_examples
  FOR DELETE TO service_role USING (true);

-- 3. RLS Policies for predictions
DROP POLICY IF EXISTS "predictions_read_authenticated" ON public.predictions;
DROP POLICY IF EXISTS "predictions_manage_service" ON public.predictions;

CREATE POLICY "predictions_read_authenticated" ON public.predictions
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "predictions_manage_service" ON public.predictions
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- 4. RLS Policies for known_cascade_patterns
DROP POLICY IF EXISTS "cascade_patterns_read" ON public.known_cascade_patterns;
DROP POLICY IF EXISTS "cascade_patterns_manage_service" ON public.known_cascade_patterns;

CREATE POLICY "cascade_patterns_read" ON public.known_cascade_patterns
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "cascade_patterns_manage_service" ON public.known_cascade_patterns
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- 5. Recreate views with SECURITY INVOKER (not DEFINER)

-- admin_consumer_overview
DROP VIEW IF EXISTS public.admin_consumer_overview;
CREATE VIEW public.admin_consumer_overview WITH (security_invoker = true) AS
SELECT
  p.id,
  p.email,
  p.full_name,
  p.skill_level,
  p.created_at,
  p.updated_at,
  COALESCE((SELECT COUNT(*) FROM learning_events le WHERE le.user_id = p.id), 0) as event_count
FROM profiles p
WHERE p.role = 'user';
GRANT SELECT ON public.admin_consumer_overview TO authenticated;

-- admin_enterprise_overview
DROP VIEW IF EXISTS public.admin_enterprise_overview;
CREATE VIEW public.admin_enterprise_overview WITH (security_invoker = true) AS
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
GRANT SELECT ON public.admin_enterprise_overview TO authenticated;

-- admin_dashboard_stats
DROP VIEW IF EXISTS public.admin_dashboard_stats;
CREATE VIEW public.admin_dashboard_stats WITH (security_invoker = true) AS
SELECT
  (SELECT COUNT(*) FROM profiles WHERE role = 'user') as consumer_count,
  (SELECT COUNT(*) FROM profiles WHERE role = 'enterprise') as enterprise_count,
  (SELECT COUNT(*) FROM profiles WHERE role = 'admin') as admin_count,
  COALESCE((SELECT COUNT(*) FROM training_examples), 0) as training_examples_count,
  COALESCE((SELECT COUNT(*) FROM learning_events), 0) as learning_events_count,
  COALESCE((SELECT COUNT(*) FROM nations WHERE status = 'active'), 0) as active_nations_count,
  COALESCE((SELECT AVG(quality_score) FROM training_examples), 0) as avg_training_quality;
GRANT SELECT ON public.admin_dashboard_stats TO authenticated;

-- exportable_training_data
DROP VIEW IF EXISTS public.exportable_training_data;
CREATE VIEW public.exportable_training_data WITH (security_invoker = true) AS
SELECT id, domain, input, output, quality_score, weight, metadata, created_at
FROM training_examples
WHERE quality_score >= 0.5
ORDER BY weight DESC, quality_score DESC;
GRANT SELECT ON public.exportable_training_data TO authenticated;

-- nations_at_risk
DROP VIEW IF EXISTS public.nations_at_risk;
CREATE VIEW public.nations_at_risk WITH (security_invoker = true) AS
SELECT code, name, region, transition_risk, basin_strength, last_event, status
FROM nations
WHERE transition_risk > 0.6 AND status = 'active'
ORDER BY transition_risk DESC;
GRANT SELECT ON public.nations_at_risk TO authenticated;

-- admin_trial_invites
DROP VIEW IF EXISTS public.admin_trial_invites;
CREATE VIEW public.admin_trial_invites WITH (security_invoker = true) AS
SELECT * FROM trial_invites ORDER BY created_at DESC;
GRANT SELECT ON public.admin_trial_invites TO authenticated;

-- training_data_stats
DROP VIEW IF EXISTS public.training_data_stats;
CREATE VIEW public.training_data_stats WITH (security_invoker = true) AS
SELECT domain, COUNT(*) as example_count, AVG(quality_score) as avg_quality,
       AVG(weight) as avg_weight, MAX(created_at) as last_created
FROM training_examples
GROUP BY domain
ORDER BY example_count DESC;
GRANT SELECT ON public.training_data_stats TO authenticated;

-- country_risk_score
DROP VIEW IF EXISTS public.country_risk_score;
CREATE VIEW public.country_risk_score WITH (security_invoker = true) AS
SELECT
  n.code, n.name, n.region, n.transition_risk, n.basin_strength, n.status,
  COALESCE((SELECT COUNT(*) FROM learning_events le WHERE le.nation_code = n.code), 0) as event_count,
  CASE
    WHEN n.transition_risk > 0.8 THEN 'CRITICAL'
    WHEN n.transition_risk > 0.6 THEN 'HIGH'
    WHEN n.transition_risk > 0.4 THEN 'MODERATE'
    WHEN n.transition_risk > 0.2 THEN 'LOW'
    ELSE 'STABLE'
  END as risk_level
FROM nations n
WHERE n.status = 'active';
GRANT SELECT ON public.country_risk_score TO authenticated;

-- nations_geojson
DROP VIEW IF EXISTS public.nations_geojson;
CREATE VIEW public.nations_geojson WITH (security_invoker = true) AS
SELECT code, name, region, transition_risk, basin_strength, position, velocity, status, last_event
FROM nations
WHERE status IN ('active', 'disputed');
GRANT SELECT ON public.nations_geojson TO authenticated;

-- edges_geojson
DROP VIEW IF EXISTS public.edges_geojson;
CREATE VIEW public.edges_geojson WITH (security_invoker = true) AS
SELECT ie.id, ie.source_code, ie.target_code, ie.influence_type, ie.strength, ie.created_at,
       sn.name as source_name, tn.name as target_name
FROM influence_edges ie
LEFT JOIN nations sn ON sn.code = ie.source_code
LEFT JOIN nations tn ON tn.code = ie.target_code;
GRANT SELECT ON public.edges_geojson TO authenticated;

-- 6. Fix spatial_ref_sys permissions if PostGIS is installed
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'spatial_ref_sys') THEN
    REVOKE ALL ON public.spatial_ref_sys FROM anon;
    REVOKE ALL ON public.spatial_ref_sys FROM authenticated;
    GRANT SELECT ON public.spatial_ref_sys TO authenticated;
  END IF;
END $$;

-- 7. Add documentation comments
COMMENT ON TABLE public.training_examples IS 'Training data for model fine-tuning. RLS enabled - read by authenticated, write by service_role only.';
