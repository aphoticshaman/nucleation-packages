-- Fix Function Search Paths - Query Actual Signatures
-- Migration: 20241205000002_fix_function_search_paths_v2
--
-- This migration queries pg_proc to get exact function signatures
-- and fixes all functions that don't have search_path set.

-- =============================================
-- Fix ALL public functions without search_path set
-- Handles both regular functions and trigger functions
-- =============================================

DO $$
DECLARE
    func_record RECORD;
    fixed_count INT := 0;
    skipped_count INT := 0;
BEGIN
    RAISE NOTICE 'Starting search_path fix for all public functions...';

    FOR func_record IN
        SELECT
            p.oid,
            n.nspname as schema_name,
            p.proname as function_name,
            pg_get_function_identity_arguments(p.oid) as args,
            p.prokind,
            p.proconfig
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        -- Include functions that have no config OR config without search_path
        AND (
            p.proconfig IS NULL
            OR NOT EXISTS (
                SELECT 1 FROM unnest(p.proconfig) AS cfg
                WHERE cfg LIKE 'search_path=%'
            )
        )
        -- Skip PostGIS and system functions
        AND p.proname NOT LIKE 'st_%'
        AND p.proname NOT LIKE 'postgis%'
        AND p.proname NOT LIKE 'geometry%'
        AND p.proname NOT LIKE 'geography%'
        AND p.proname NOT LIKE 'box%'
        AND p.proname NOT LIKE 'path%'
        AND p.proname NOT LIKE 'point%'
        AND p.proname NOT LIKE 'polygon%'
        AND p.proname NOT LIKE 'line%'
        AND p.proname NOT LIKE 'circle%'
        AND p.proname NOT LIKE 'lseg%'
        AND p.proname NOT LIKE '_st_%'
        AND p.proname NOT LIKE '_postgis%'
        AND p.proname NOT LIKE 'pg_%'
        AND p.proname NOT LIKE '_pg_%'
        AND p.proname NOT LIKE 'spheroid%'
        AND p.proname NOT LIKE 'bytea%'
        AND p.proname NOT LIKE 'text%'
        AND p.proname NOT LIKE 'float%'
        AND p.proname NOT LIKE 'int%'
        -- Skip aggregate functions
        AND p.prokind IN ('f', 'p')  -- functions and procedures only
    LOOP
        BEGIN
            EXECUTE format(
                'ALTER FUNCTION %I.%I(%s) SET search_path = %L',
                func_record.schema_name,
                func_record.function_name,
                func_record.args,
                ''
            );
            fixed_count := fixed_count + 1;
            RAISE NOTICE 'Fixed: %.%(%)',
                func_record.schema_name,
                func_record.function_name,
                func_record.args;
        EXCEPTION WHEN OTHERS THEN
            skipped_count := skipped_count + 1;
            RAISE NOTICE 'Skipped %.% - %',
                func_record.schema_name,
                func_record.function_name,
                SQLERRM;
        END;
    END LOOP;

    RAISE NOTICE 'Complete. Fixed: %, Skipped: %', fixed_count, skipped_count;
END $$;

-- =============================================
-- Verify the fix worked
-- =============================================

DO $$
DECLARE
    remaining INT;
BEGIN
    SELECT COUNT(*) INTO remaining
    FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'public'
    AND p.prokind IN ('f', 'p')
    AND (
        p.proconfig IS NULL
        OR NOT EXISTS (
            SELECT 1 FROM unnest(p.proconfig) AS cfg
            WHERE cfg LIKE 'search_path=%'
        )
    )
    AND p.proname NOT LIKE 'st_%'
    AND p.proname NOT LIKE 'postgis%'
    AND p.proname NOT LIKE 'geometry%'
    AND p.proname NOT LIKE 'geography%';

    RAISE NOTICE 'Remaining functions without search_path: %', remaining;
END $$;
