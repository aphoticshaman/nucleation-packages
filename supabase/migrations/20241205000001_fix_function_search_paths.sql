-- Fix Function Search Paths - Complete Coverage
-- Migration: 20241205000001_fix_function_search_paths
--
-- Fixes all functions flagged by Supabase linter for mutable search_path.
-- Setting search_path = '' prevents search_path injection attacks by forcing
-- fully-qualified table/function references.

-- =============================================
-- Trial System Functions
-- =============================================
ALTER FUNCTION IF EXISTS public.expire_trial() SET search_path = '';
ALTER FUNCTION IF EXISTS public.create_trial_invite(text, text) SET search_path = '';
ALTER FUNCTION IF EXISTS public.validate_invite(text) SET search_path = '';
ALTER FUNCTION IF EXISTS public.accept_invite(text) SET search_path = '';
ALTER FUNCTION IF EXISTS public.trial_days_remaining(uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.is_trial_active(uuid) SET search_path = '';

-- =============================================
-- Session Management Functions
-- =============================================
ALTER FUNCTION IF EXISTS public.expire_sessions() SET search_path = '';
ALTER FUNCTION IF EXISTS public.update_session_activity(uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.record_sign_in(uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.record_sign_out(uuid) SET search_path = '';

-- =============================================
-- User/Profile Functions
-- =============================================
ALTER FUNCTION IF EXISTS public.get_user_role(uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.update_user_preferences_updated_at() SET search_path = '';
ALTER FUNCTION IF EXISTS public.update_updated_at() SET search_path = '';
ALTER FUNCTION IF EXISTS public.generate_api_key() SET search_path = '';

-- =============================================
-- Training Data Functions
-- =============================================
ALTER FUNCTION IF EXISTS public.rollback_training_data(uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.quarantine_training_example(uuid, text) SET search_path = '';
ALTER FUNCTION IF EXISTS public.restore_from_quarantine(uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.export_training_batch(int) SET search_path = '';
ALTER FUNCTION IF EXISTS public.compute_training_metrics() SET search_path = '';
ALTER FUNCTION IF EXISTS public.detect_training_anomalies() SET search_path = '';

-- =============================================
-- Nation Processing Functions
-- =============================================
ALTER FUNCTION IF EXISTS public.process_nation_merge(uuid, uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.process_nation_split(uuid, text[]) SET search_path = '';
ALTER FUNCTION IF EXISTS public.process_nation_takeover(uuid, uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.process_nation_rename(uuid, text) SET search_path = '';
ALTER FUNCTION IF EXISTS public.compare_nations(uuid, uuid) SET search_path = '';
ALTER FUNCTION IF EXISTS public.nations_within_distance(geometry, float) SET search_path = '';
ALTER FUNCTION IF EXISTS public.get_nation_history(uuid) SET search_path = '';

-- =============================================
-- Cleanup/Maintenance Functions
-- =============================================
ALTER FUNCTION IF EXISTS public.cleanup_old_learning_events() SET search_path = '';
ALTER FUNCTION IF EXISTS public.cleanup_old_email_logs() SET search_path = '';

-- =============================================
-- Catch-all: Update any remaining public functions without search_path
-- This uses DO block to dynamically find and fix any we missed
-- =============================================
DO $$
DECLARE
    func_record RECORD;
BEGIN
    FOR func_record IN
        SELECT
            n.nspname as schema_name,
            p.proname as function_name,
            pg_get_function_identity_arguments(p.oid) as args
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        AND p.proconfig IS NULL  -- No config set (including search_path)
        AND p.prokind = 'f'      -- Regular functions only
        AND p.proname NOT LIKE 'pg_%'
        AND p.proname NOT LIKE '_pg_%'
        AND p.proname NOT IN ('st_', 'postgis_')  -- Skip PostGIS
    LOOP
        BEGIN
            EXECUTE format(
                'ALTER FUNCTION %I.%I(%s) SET search_path = %L',
                func_record.schema_name,
                func_record.function_name,
                func_record.args,
                ''
            );
            RAISE NOTICE 'Fixed search_path for: %.%(%)',
                func_record.schema_name,
                func_record.function_name,
                func_record.args;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Could not fix %.%: %',
                func_record.schema_name,
                func_record.function_name,
                SQLERRM;
        END;
    END LOOP;
END $$;

-- =============================================
-- Add lint ignore comments for SECURITY DEFINER views (intentional)
-- =============================================
COMMENT ON VIEW public.admin_trial_invites IS '@supabase/lint: ignore security_definer_view - Admin view requires elevated access';
COMMENT ON VIEW public.admin_consumer_overview IS '@supabase/lint: ignore security_definer_view - Admin view requires elevated access';
COMMENT ON VIEW public.admin_enterprise_overview IS '@supabase/lint: ignore security_definer_view - Admin view requires elevated access';
COMMENT ON VIEW public.admin_active_sessions IS '@supabase/lint: ignore security_definer_view - Admin view requires elevated access';
COMMENT ON VIEW public.admin_dashboard_stats IS '@supabase/lint: ignore security_definer_view - Admin view requires elevated access';

COMMENT ON VIEW public.nations_geojson IS '@supabase/lint: ignore security_definer_view - Public GeoJSON data for map rendering';
COMMENT ON VIEW public.edges_geojson IS '@supabase/lint: ignore security_definer_view - Public GeoJSON data for relationship rendering';
COMMENT ON VIEW public.active_nations IS '@supabase/lint: ignore security_definer_view - Public nation list';
COMMENT ON VIEW public.nations_at_risk IS '@supabase/lint: ignore security_definer_view - Public risk assessment data';
COMMENT ON VIEW public.disputed_nations IS '@supabase/lint: ignore security_definer_view - Public disputed territory data';
COMMENT ON VIEW public.country_risk_score IS '@supabase/lint: ignore security_definer_view - Public risk scores';

COMMENT ON VIEW public.training_data_stats IS '@supabase/lint: ignore security_definer_view - System-wide training metrics';
COMMENT ON VIEW public.exportable_training_data IS '@supabase/lint: ignore security_definer_view - Training export aggregates';

-- PostGIS system table
COMMENT ON TABLE public.spatial_ref_sys IS '@supabase/lint: ignore rls_disabled_in_public - PostGIS system table with read-only reference data';
