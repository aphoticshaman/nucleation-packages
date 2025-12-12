-- Fix remaining function search_path warnings
-- These functions have specific signatures that need to be dropped and recreated

-- =============================================================================
-- create_training_item_from_evaluation
-- =============================================================================
DROP FUNCTION IF EXISTS public.create_training_item_from_evaluation(UUID);
CREATE OR REPLACE FUNCTION public.create_training_item_from_evaluation(p_eval_id UUID)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO public.training_items (source_id, source_type)
    VALUES (p_eval_id, 'evaluation')
    RETURNING id INTO v_id;
    RETURN v_id;
EXCEPTION WHEN undefined_table THEN
    -- Table might not exist yet
    RETURN NULL;
END;
$$;

-- =============================================================================
-- compute_audit_hash
-- =============================================================================
DROP FUNCTION IF EXISTS public.compute_audit_hash() CASCADE;
CREATE OR REPLACE FUNCTION public.compute_audit_hash()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    NEW.hash = encode(sha256(concat(
        NEW.id::text,
        COALESCE(NEW.action, ''),
        NEW.created_at::text
    )::bytea), 'hex');
    RETURN NEW;
END;
$$;

-- =============================================================================
-- log_training_action
-- =============================================================================
DROP FUNCTION IF EXISTS public.log_training_action() CASCADE;
CREATE OR REPLACE FUNCTION public.log_training_action()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    INSERT INTO public.training_audit_log (action, item_id, performed_by)
    VALUES (TG_OP, COALESCE(NEW.id, OLD.id), (SELECT auth.uid()))
    ON CONFLICT DO NOTHING;
    RETURN COALESCE(NEW, OLD);
EXCEPTION WHEN undefined_table THEN
    -- Table might not exist
    RETURN COALESCE(NEW, OLD);
END;
$$;

-- =============================================================================
-- RLS POLICIES FOR TABLES WITH RLS BUT NO POLICIES
-- =============================================================================

-- api_usage - service role only
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'api_usage' AND schemaname = 'public') THEN
        DROP POLICY IF EXISTS "api_usage_service" ON public.api_usage;
        CREATE POLICY "api_usage_service" ON public.api_usage
            FOR ALL USING ((SELECT auth.jwt()->>'role') = 'service_role');
    END IF;
END $$;

-- simulation_snapshots - user owns their own
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'simulation_snapshots' AND schemaname = 'public') THEN
        DROP POLICY IF EXISTS "simulation_snapshots_user" ON public.simulation_snapshots;
        CREATE POLICY "simulation_snapshots_user" ON public.simulation_snapshots
            FOR ALL USING (user_id = (SELECT auth.uid()));
    END IF;
END $$;

-- user_activity - user owns their own
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'user_activity' AND schemaname = 'public') THEN
        DROP POLICY IF EXISTS "user_activity_user" ON public.user_activity;
        CREATE POLICY "user_activity_user" ON public.user_activity
            FOR ALL USING (user_id = (SELECT auth.uid()));
    END IF;
END $$;

-- =============================================================================
-- NOTE: PostGIS extension_in_public warning cannot be fixed
-- PostGIS must remain in public schema for Supabase compatibility
-- =============================================================================
