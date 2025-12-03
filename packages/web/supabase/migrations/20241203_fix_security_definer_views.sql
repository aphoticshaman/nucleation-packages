-- ============================================
-- FIX SUPABASE SECURITY LINTER ERRORS
-- ============================================
-- Run this in Supabase SQL Editor
--
-- Issues being fixed:
-- 1. SECURITY DEFINER views (10 views) - bypasses RLS
-- 2. RLS disabled tables (2 tables)
--
-- IMPORTANT: Review each change before applying
-- Some views may intentionally need SECURITY DEFINER for admin functions
-- ============================================

-- ============================================
-- PART 1: FIX SECURITY DEFINER VIEWS
-- ============================================
-- SECURITY DEFINER means the view runs with the OWNER's permissions
-- SECURITY INVOKER (default) means it runs with the QUERYING user's permissions
-- For RLS to work properly, views should use SECURITY INVOKER

-- Option A: Drop and recreate views as SECURITY INVOKER
-- Option B: Keep SECURITY DEFINER but add explicit permission checks
--
-- We'll use Option B for admin views (they need elevated access)
-- We'll use Option A for public data views

-- 1. Admin views - KEEP SECURITY DEFINER but add role checks inside
-- These legitimately need elevated permissions to aggregate data

-- admin_dashboard_stats - Keep SECURITY DEFINER, add internal auth check
DROP VIEW IF EXISTS public.admin_dashboard_stats CASCADE;
CREATE OR REPLACE VIEW public.admin_dashboard_stats
WITH (security_invoker = false) -- SECURITY DEFINER
AS
SELECT
    CASE
        WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN
            (SELECT COUNT(*) FROM auth.users)
        ELSE NULL
    END as total_users,
    CASE
        WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN
            (SELECT COUNT(*) FROM auth.users WHERE created_at > NOW() - INTERVAL '30 days')
        ELSE NULL
    END as users_last_30_days,
    CASE
        WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN
            (SELECT COUNT(*) FROM auth.users WHERE last_sign_in_at > NOW() - INTERVAL '7 days')
        ELSE NULL
    END as active_users_7_days;

COMMENT ON VIEW public.admin_dashboard_stats IS
'Admin-only dashboard statistics. SECURITY DEFINER with internal role check.';

-- admin_consumer_overview - Keep SECURITY DEFINER, add internal auth check
DROP VIEW IF EXISTS public.admin_consumer_overview CASCADE;
CREATE OR REPLACE VIEW public.admin_consumer_overview
WITH (security_invoker = false)
AS
SELECT
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN id ELSE NULL END as id,
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN email ELSE NULL END as email,
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN created_at ELSE NULL END as created_at
FROM auth.users
WHERE auth.jwt() ->> 'role' IN ('admin', 'service_role');

COMMENT ON VIEW public.admin_consumer_overview IS
'Admin-only consumer overview. SECURITY DEFINER with internal role check.';

-- admin_enterprise_overview - Keep SECURITY DEFINER, add internal auth check
DROP VIEW IF EXISTS public.admin_enterprise_overview CASCADE;
CREATE OR REPLACE VIEW public.admin_enterprise_overview
WITH (security_invoker = false)
AS
SELECT
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN id ELSE NULL END as id,
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN email ELSE NULL END as email,
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN raw_user_meta_data ELSE NULL END as metadata
FROM auth.users
WHERE auth.jwt() ->> 'role' IN ('admin', 'service_role')
AND raw_user_meta_data ->> 'tier' IN ('strategist', 'architect', 'enterprise');

COMMENT ON VIEW public.admin_enterprise_overview IS
'Admin-only enterprise user overview. SECURITY DEFINER with internal role check.';

-- admin_trial_invites - Keep SECURITY DEFINER, add internal auth check
DROP VIEW IF EXISTS public.admin_trial_invites CASCADE;
CREATE OR REPLACE VIEW public.admin_trial_invites
WITH (security_invoker = false)
AS
SELECT
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN id ELSE NULL END as id,
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN email ELSE NULL END as email,
    CASE WHEN auth.jwt() ->> 'role' IN ('admin', 'service_role') THEN created_at ELSE NULL END as created_at
FROM auth.users
WHERE auth.jwt() ->> 'role' IN ('admin', 'service_role')
AND raw_user_meta_data ->> 'trial' = 'true';

COMMENT ON VIEW public.admin_trial_invites IS
'Admin-only trial invites. SECURITY DEFINER with internal role check.';

-- 2. Public data views - Convert to SECURITY INVOKER (respects RLS)

-- nations_geojson - Public geographic data, should use INVOKER
DROP VIEW IF EXISTS public.nations_geojson CASCADE;
CREATE OR REPLACE VIEW public.nations_geojson
WITH (security_invoker = true) -- SECURITY INVOKER
AS
SELECT
    id,
    name,
    iso_code,
    ST_AsGeoJSON(geometry)::json as geometry
FROM public.nations
WHERE geometry IS NOT NULL;

COMMENT ON VIEW public.nations_geojson IS
'Public GeoJSON view of nations. Uses SECURITY INVOKER to respect RLS.';

-- edges_geojson - Geographic edges, should use INVOKER
DROP VIEW IF EXISTS public.edges_geojson CASCADE;
CREATE OR REPLACE VIEW public.edges_geojson
WITH (security_invoker = true)
AS
SELECT
    id,
    source_nation,
    target_nation,
    edge_type,
    weight,
    ST_AsGeoJSON(geometry)::json as geometry
FROM public.edges
WHERE geometry IS NOT NULL;

COMMENT ON VIEW public.edges_geojson IS
'Public GeoJSON view of edges. Uses SECURITY INVOKER to respect RLS.';

-- nations_at_risk - Risk scores, should use INVOKER
DROP VIEW IF EXISTS public.nations_at_risk CASCADE;
CREATE OR REPLACE VIEW public.nations_at_risk
WITH (security_invoker = true)
AS
SELECT
    n.id,
    n.name,
    n.iso_code,
    r.risk_score,
    r.risk_factors,
    r.updated_at
FROM public.nations n
LEFT JOIN public.risk_scores r ON n.id = r.nation_id
WHERE r.risk_score > 0.5
ORDER BY r.risk_score DESC;

COMMENT ON VIEW public.nations_at_risk IS
'Nations with elevated risk scores. Uses SECURITY INVOKER to respect RLS.';

-- country_risk_score - Risk aggregation, should use INVOKER
DROP VIEW IF EXISTS public.country_risk_score CASCADE;
CREATE OR REPLACE VIEW public.country_risk_score
WITH (security_invoker = true)
AS
SELECT
    n.id,
    n.name,
    n.iso_code,
    COALESCE(r.risk_score, 0) as risk_score,
    r.risk_factors,
    r.confidence,
    r.updated_at
FROM public.nations n
LEFT JOIN public.risk_scores r ON n.id = r.nation_id;

COMMENT ON VIEW public.country_risk_score IS
'Country risk scores. Uses SECURITY INVOKER to respect RLS.';

-- 3. Training data views - Convert to INVOKER and add RLS

-- training_data_stats - Should use INVOKER
DROP VIEW IF EXISTS public.training_data_stats CASCADE;
CREATE OR REPLACE VIEW public.training_data_stats
WITH (security_invoker = true)
AS
SELECT
    COUNT(*) as total_examples,
    COUNT(*) FILTER (WHERE validated = true) as validated_examples,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as recent_examples
FROM public.training_examples;

COMMENT ON VIEW public.training_data_stats IS
'Training data statistics. Uses SECURITY INVOKER to respect RLS.';

-- exportable_training_data - Should use INVOKER
DROP VIEW IF EXISTS public.exportable_training_data CASCADE;
CREATE OR REPLACE VIEW public.exportable_training_data
WITH (security_invoker = true)
AS
SELECT
    id,
    input_text,
    output_text,
    category,
    quality_score,
    validated,
    created_at
FROM public.training_examples
WHERE validated = true
AND quality_score >= 0.7;

COMMENT ON VIEW public.exportable_training_data IS
'Validated training data for export. Uses SECURITY INVOKER to respect RLS.';


-- ============================================
-- PART 2: ENABLE RLS ON TABLES
-- ============================================

-- spatial_ref_sys - PostGIS system table
-- This is a PostGIS system table with coordinate reference definitions
-- Option 1: Move to a non-public schema
-- Option 2: Enable RLS and allow public read

-- Enable RLS on spatial_ref_sys (PostGIS reference table)
ALTER TABLE public.spatial_ref_sys ENABLE ROW LEVEL SECURITY;

-- Allow public read access (it's reference data)
DROP POLICY IF EXISTS "Allow public read of spatial_ref_sys" ON public.spatial_ref_sys;
CREATE POLICY "Allow public read of spatial_ref_sys"
ON public.spatial_ref_sys
FOR SELECT
TO public
USING (true);

-- Prevent modifications except by service role
DROP POLICY IF EXISTS "Only service role can modify spatial_ref_sys" ON public.spatial_ref_sys;
CREATE POLICY "Only service role can modify spatial_ref_sys"
ON public.spatial_ref_sys
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

COMMENT ON TABLE public.spatial_ref_sys IS
'PostGIS spatial reference system table. RLS enabled, public read-only.';

-- training_examples - Enable RLS with proper policies
ALTER TABLE public.training_examples ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read validated examples
DROP POLICY IF EXISTS "Users can read validated training examples" ON public.training_examples;
CREATE POLICY "Users can read validated training examples"
ON public.training_examples
FOR SELECT
TO authenticated
USING (validated = true);

-- Allow admins to read all examples
DROP POLICY IF EXISTS "Admins can read all training examples" ON public.training_examples;
CREATE POLICY "Admins can read all training examples"
ON public.training_examples
FOR SELECT
TO authenticated
USING (
    auth.jwt() ->> 'role' = 'admin'
    OR auth.jwt() ->> 'role' = 'service_role'
);

-- Allow admins to insert/update
DROP POLICY IF EXISTS "Admins can modify training examples" ON public.training_examples;
CREATE POLICY "Admins can modify training examples"
ON public.training_examples
FOR ALL
TO authenticated
USING (
    auth.jwt() ->> 'role' = 'admin'
    OR auth.jwt() ->> 'role' = 'service_role'
)
WITH CHECK (
    auth.jwt() ->> 'role' = 'admin'
    OR auth.jwt() ->> 'role' = 'service_role'
);

COMMENT ON TABLE public.training_examples IS
'ML training examples. RLS enabled - authenticated read validated, admin full access.';


-- ============================================
-- PART 3: VERIFY FIXES
-- ============================================

-- Run this query to verify no more security_definer issues
-- (Should return empty or only intentional SECURITY DEFINER views)
/*
SELECT
    schemaname,
    viewname,
    viewowner,
    CASE
        WHEN definition LIKE '%security_invoker = false%' THEN 'SECURITY_DEFINER'
        ELSE 'SECURITY_INVOKER'
    END as security_mode
FROM pg_views
WHERE schemaname = 'public'
ORDER BY viewname;
*/

-- Run this query to verify RLS is enabled
/*
SELECT
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('spatial_ref_sys', 'training_examples')
ORDER BY tablename;
*/


-- ============================================
-- ROLLBACK SCRIPT (if needed)
-- ============================================
/*
-- To rollback, run:
ALTER TABLE public.spatial_ref_sys DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_examples DISABLE ROW LEVEL SECURITY;

-- Then recreate original views
-- (You'd need the original view definitions)
*/
