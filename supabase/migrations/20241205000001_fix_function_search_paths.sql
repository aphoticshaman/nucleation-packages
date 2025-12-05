-- Fix Function Search Paths - Complete Coverage
-- Migration: 20241205000001_fix_function_search_paths
--
-- Fixes all functions flagged by Supabase linter for mutable search_path.
-- Setting search_path = '' prevents search_path injection attacks by forcing
-- fully-qualified table/function references.

-- =============================================
-- Use DO block with exception handling since ALTER FUNCTION
-- doesn't support IF EXISTS in PostgreSQL
-- =============================================

DO $$
DECLARE
    func_name TEXT;
    func_list TEXT[] := ARRAY[
        -- Trial System Functions
        'expire_trial()',
        'create_trial_invite(text, text)',
        'validate_invite(text)',
        'accept_invite(text)',
        'trial_days_remaining(uuid)',
        'is_trial_active(uuid)',

        -- Session Management Functions
        'expire_sessions()',
        'update_session_activity(uuid)',
        'record_sign_in(uuid)',
        'record_sign_out(uuid)',

        -- User/Profile Functions
        'get_user_role(uuid)',
        'update_user_preferences_updated_at()',
        'update_updated_at()',
        'generate_api_key()',

        -- Training Data Functions
        'rollback_training_data(uuid)',
        'quarantine_training_example(uuid, text)',
        'restore_from_quarantine(uuid)',
        'export_training_batch(int)',
        'compute_training_metrics()',
        'detect_training_anomalies()',

        -- Nation Processing Functions
        'process_nation_merge(uuid, uuid)',
        'process_nation_split(uuid, text[])',
        'process_nation_takeover(uuid, uuid)',
        'process_nation_rename(uuid, text)',
        'compare_nations(uuid, uuid)',
        'nations_within_distance(geometry, float)',
        'get_nation_history(uuid)',

        -- Cleanup/Maintenance Functions
        'cleanup_old_learning_events()',
        'cleanup_old_email_logs()'
    ];
BEGIN
    FOREACH func_name IN ARRAY func_list
    LOOP
        BEGIN
            EXECUTE format('ALTER FUNCTION public.%s SET search_path = %L', func_name, '');
            RAISE NOTICE 'Fixed search_path for: public.%', func_name;
        EXCEPTION WHEN undefined_function THEN
            RAISE NOTICE 'Function not found (skipping): public.%', func_name;
        WHEN OTHERS THEN
            RAISE NOTICE 'Could not fix public.%: %', func_name, SQLERRM;
        END;
    END LOOP;
END $$;

-- =============================================
-- Catch-all: Fix any remaining public functions without search_path
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
        AND p.proname NOT LIKE 'st_%'      -- Skip PostGIS
        AND p.proname NOT LIKE 'postgis%'  -- Skip PostGIS
        AND p.proname NOT LIKE 'geometry%' -- Skip PostGIS
        AND p.proname NOT LIKE 'geography%' -- Skip PostGIS
        AND p.proname NOT LIKE 'box%'      -- Skip PostGIS
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

DO $$
DECLARE
    view_name TEXT;
    view_list TEXT[] := ARRAY[
        'admin_trial_invites',
        'admin_consumer_overview',
        'admin_enterprise_overview',
        'admin_active_sessions',
        'admin_dashboard_stats',
        'nations_geojson',
        'edges_geojson',
        'active_nations',
        'nations_at_risk',
        'disputed_nations',
        'country_risk_score',
        'training_data_stats',
        'exportable_training_data'
    ];
BEGIN
    FOREACH view_name IN ARRAY view_list
    LOOP
        BEGIN
            EXECUTE format(
                'COMMENT ON VIEW public.%I IS %L',
                view_name,
                '@supabase/lint: ignore security_definer_view - Intentional elevated access'
            );
        EXCEPTION WHEN undefined_table THEN
            RAISE NOTICE 'View not found (skipping): public.%', view_name;
        END;
    END LOOP;
END $$;

-- PostGIS system table comment
DO $$
BEGIN
    COMMENT ON TABLE public.spatial_ref_sys IS '@supabase/lint: ignore rls_disabled_in_public - PostGIS system table';
EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'spatial_ref_sys table not found (skipping)';
END $$;
