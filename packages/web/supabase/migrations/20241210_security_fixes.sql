-- ============================================================================
-- Security Fixes - Remaining Linter Issues
-- ============================================================================
-- Fixes:
-- 1. SECURITY DEFINER views → SECURITY INVOKER
-- 2. Functions missing search_path
-- 3. Materialized views exposed to API → revoke anon access
-- 4. Tables with RLS but no policies
-- ============================================================================

-- ============================================================================
-- PART 1: Fix SECURITY DEFINER views
-- ============================================================================
-- Recreate views with SECURITY INVOKER (default, respects caller's RLS)

-- Drop and recreate training_data_stats
DROP VIEW IF EXISTS public.training_data_stats;
CREATE VIEW public.training_data_stats
WITH (security_invoker = true)
AS
SELECT
    COUNT(*) as total_examples,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as examples_last_7d,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as examples_last_30d
FROM public.training_examples;

-- Drop and recreate exportable_training_data
DROP VIEW IF EXISTS public.exportable_training_data;
CREATE VIEW public.exportable_training_data
WITH (security_invoker = true)
AS
SELECT
    id,
    created_at
FROM public.training_examples;

-- ============================================================================
-- PART 2: Fix function search_path
-- ============================================================================

-- Fix record_nation_snapshot
CREATE OR REPLACE FUNCTION public.record_nation_snapshot()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO nation_history (
        nation_code,
        position,
        velocity,
        basin_strength,
        transition_risk,
        regime,
        recorded_at
    ) VALUES (
        NEW.code,
        NEW.position,
        NEW.velocity,
        NEW.basin_strength,
        NEW.transition_risk,
        NEW.regime,
        NOW()
    );
    RETURN NEW;
END;
$$;

-- Fix sync_profile_to_client (we just created this)
CREATE OR REPLACE FUNCTION public.sync_profile_to_client()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    -- On INSERT: Create corresponding client with same id
    IF TG_OP = 'INSERT' THEN
        INSERT INTO public.clients (id, name, email, created_at)
        VALUES (
            NEW.id,
            COALESCE(NEW.full_name, split_part(NEW.email, '@', 1)),
            NEW.email,
            NEW.created_at
        )
        ON CONFLICT (email) DO UPDATE SET
            name = EXCLUDED.name;
        RETURN NEW;
    END IF;

    -- On UPDATE: Sync changes to client
    IF TG_OP = 'UPDATE' THEN
        UPDATE public.clients
        SET
            name = COALESCE(NEW.full_name, split_part(NEW.email, '@', 1)),
            email = NEW.email,
            updated_at = NOW()
        WHERE id = NEW.id;
        RETURN NEW;
    END IF;

    -- On DELETE: Delete corresponding client
    IF TG_OP = 'DELETE' THEN
        DELETE FROM public.clients WHERE id = OLD.id;
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$;

-- ============================================================================
-- PART 3: Revoke anon access to materialized views
-- ============================================================================

REVOKE SELECT ON public.usage_summary FROM anon;
REVOKE SELECT ON public.country_signals_latest FROM anon;

-- ============================================================================
-- PART 4: Add RLS policies for tables with RLS enabled but no policies
-- ============================================================================

-- api_usage: Only accessible via service role (internal metrics)
CREATE POLICY "Service role access to api_usage" ON public.api_usage
    FOR ALL USING ((select auth.role()) = 'service_role');

-- simulation_snapshots: Users can view snapshots of their own simulations
CREATE POLICY "Users can view own simulation snapshots" ON public.simulation_snapshots
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.simulations s
            WHERE s.id = simulation_snapshots.simulation_id
            AND s.user_id = (select auth.uid())
        )
    );

CREATE POLICY "Service role can manage simulation snapshots" ON public.simulation_snapshots
    FOR ALL USING ((select auth.role()) = 'service_role');

-- user_activity: Users can view their own activity, admins can view all
CREATE POLICY "Users can view own activity" ON public.user_activity
    FOR SELECT USING ((select auth.uid()) = user_id);

CREATE POLICY "Admins can view all activity" ON public.user_activity
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM public.profiles WHERE id = (select auth.uid()) AND role = 'admin')
    );

CREATE POLICY "Service role can manage activity" ON public.user_activity
    FOR ALL USING ((select auth.role()) = 'service_role');
