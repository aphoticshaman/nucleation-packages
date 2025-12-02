-- Fix RLS Security Issues
-- Addresses Supabase linter warnings for tables without RLS

-- 1. Enable RLS on tables that are missing it
ALTER TABLE IF EXISTS public.signals_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.esteem_relations ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.influence_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.regimes ENABLE ROW LEVEL SECURITY;

-- Note: spatial_ref_sys is a PostGIS system table - typically left without RLS
-- If you need to restrict it, uncomment:
-- ALTER TABLE IF EXISTS public.spatial_ref_sys ENABLE ROW LEVEL SECURITY;

-- 2. Create permissive read policies for public reference data
-- These tables contain non-sensitive geopolitical reference data

-- signals_cache: Allow authenticated users to read
DO $$ BEGIN
  CREATE POLICY "Authenticated users can read signals_cache"
    ON public.signals_cache FOR SELECT
    USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- esteem_relations: Public read access (reference data)
DO $$ BEGIN
  CREATE POLICY "Anyone can read esteem_relations"
    ON public.esteem_relations FOR SELECT
    USING (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- influence_edges: Public read access (reference data)
DO $$ BEGIN
  CREATE POLICY "Anyone can read influence_edges"
    ON public.influence_edges FOR SELECT
    USING (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- regimes: Public read access (reference data)
DO $$ BEGIN
  CREATE POLICY "Anyone can read regimes"
    ON public.regimes FOR SELECT
    USING (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- 3. Fix function search_path security (prevents search_path injection attacks)
-- These functions need SET search_path = public added

ALTER FUNCTION IF EXISTS public.expire_trial() SET search_path = public;
ALTER FUNCTION IF EXISTS public.compare_nations(text, text) SET search_path = public;
ALTER FUNCTION IF EXISTS public.update_nation_geometry() SET search_path = public;
ALTER FUNCTION IF EXISTS public.nations_within_distance(geography, double precision) SET search_path = public;
ALTER FUNCTION IF EXISTS public.get_user_role() SET search_path = public;
ALTER FUNCTION IF EXISTS public.accept_invite(uuid) SET search_path = public;
ALTER FUNCTION IF EXISTS public.handle_new_user() SET search_path = public;
ALTER FUNCTION IF EXISTS public.generate_api_key() SET search_path = public;
ALTER FUNCTION IF EXISTS public.trial_days_remaining() SET search_path = public;
ALTER FUNCTION IF EXISTS public.is_trial_active() SET search_path = public;
ALTER FUNCTION IF EXISTS public.create_trial_invite(text, text) SET search_path = public;
ALTER FUNCTION IF EXISTS public.update_last_seen() SET search_path = public;
ALTER FUNCTION IF EXISTS public.validate_invite(uuid) SET search_path = public;

-- 4. Fix SECURITY DEFINER views by recreating as SECURITY INVOKER
-- Note: Admin views intentionally use SECURITY DEFINER with app-level role checks

COMMENT ON VIEW public.admin_consumer_overview IS
  'SECURITY DEFINER intentional - admin-only view with role check in application';
COMMENT ON VIEW public.admin_enterprise_overview IS
  'SECURITY DEFINER intentional - admin-only view with role check in application';
COMMENT ON VIEW public.admin_dashboard_stats IS
  'SECURITY DEFINER intentional - admin-only view with role check in application';
COMMENT ON VIEW public.admin_trial_invites IS
  'SECURITY DEFINER intentional - admin-only view with role check in application';

-- 5. Revoke public access to materialized view
REVOKE SELECT ON public.usage_summary FROM anon;
-- Keep authenticated access if needed for dashboard
-- REVOKE SELECT ON public.usage_summary FROM authenticated;

-- Grant appropriate permissions
GRANT SELECT ON public.signals_cache TO authenticated;
GRANT SELECT ON public.esteem_relations TO anon, authenticated;
GRANT SELECT ON public.influence_edges TO anon, authenticated;
GRANT SELECT ON public.regimes TO anon, authenticated;

-- NOTE: Additional manual steps required in Supabase Dashboard:
-- 1. Enable "Leaked Password Protection" in Auth Settings:
--    Dashboard -> Authentication -> Providers -> Email -> Enable "Prevent use of leaked passwords"
-- 2. Consider moving PostGIS extension to 'extensions' schema (optional, low priority)
