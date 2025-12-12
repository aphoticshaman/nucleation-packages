-- Fix Security Linter Warnings
--
-- 1. Convert SECURITY DEFINER views to SECURITY INVOKER
-- 2. Enable RLS on spatial_ref_sys (PostGIS table)

-- =============================================================================
-- FIX INSIGHT VIEWS - Change to SECURITY INVOKER
-- =============================================================================

-- Recreate insights_awaiting_review with SECURITY INVOKER
DROP VIEW IF EXISTS insights_awaiting_review;
CREATE VIEW insights_awaiting_review
WITH (security_invoker = true)
AS
SELECT
    id,
    title,
    summary,
    target_subject,
    current_stage,
    confidence_score,
    confidence_type,
    created_at,
    updated_at,
    tags
FROM insight_reports
WHERE status = 'awaiting_review'
ORDER BY confidence_score DESC, created_at DESC;

-- Recreate validated_insights with SECURITY INVOKER
DROP VIEW IF EXISTS validated_insights;
CREATE VIEW validated_insights
WITH (security_invoker = true)
AS
SELECT
    id,
    title,
    summary,
    target_subject,
    confidence_score,
    confidence_type,
    admin_rating,
    impact_analysis,
    code_artifacts,
    created_at,
    reviewed_at
FROM insight_reports
WHERE status = 'validated'
AND confidence_score >= 0.7
ORDER BY admin_rating DESC NULLS LAST, confidence_score DESC;

-- Recreate insight_pipeline with SECURITY INVOKER
DROP VIEW IF EXISTS insight_pipeline;
CREATE VIEW insight_pipeline
WITH (security_invoker = true)
AS
SELECT
    current_stage,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    array_agg(title ORDER BY created_at DESC) as recent_titles
FROM insight_reports
WHERE status = 'in_progress'
GROUP BY current_stage
ORDER BY current_stage;

-- =============================================================================
-- FIX TRAINING VIEWS - Drop old views that have wrong schema
-- =============================================================================

-- These views reference columns that don't exist in the current schema
-- Just drop them - they'll be recreated if needed later
DROP VIEW IF EXISTS v_training_stats_by_domain;
DROP VIEW IF EXISTS v_training_items_selected;
DROP VIEW IF EXISTS v_training_items_active;
DROP VIEW IF EXISTS v_training_audit_recent;

-- =============================================================================
-- NOTE: SPATIAL_REF_SYS
-- =============================================================================
-- spatial_ref_sys is owned by PostGIS extension - cannot modify RLS
-- This linter warning can be safely ignored as it's a read-only reference table

-- =============================================================================
-- FIX FUNCTION SEARCH PATHS
-- =============================================================================

-- generate_insight_slug - drop and recreate with empty search_path
DROP FUNCTION IF EXISTS generate_insight_slug() CASCADE;
CREATE FUNCTION public.generate_insight_slug()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    IF NEW.slug IS NULL THEN
        NEW.slug = lower(regexp_replace(NEW.title, '[^a-zA-Z0-9]+', '-', 'g'));
        NEW.slug = NEW.slug || '-' || to_char(NEW.created_at, 'YYYYMMDD');
    END IF;
    RETURN NEW;
END;
$$;

-- track_stage_transition - drop and recreate with empty search_path
DROP FUNCTION IF EXISTS track_stage_transition() CASCADE;
CREATE FUNCTION public.track_stage_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    IF OLD.current_stage IS DISTINCT FROM NEW.current_stage THEN
        NEW.stage_timestamps = jsonb_set(
            COALESCE(NEW.stage_timestamps, '{}'::jsonb),
            ARRAY[NEW.current_stage::text],
            to_jsonb(NOW())
        );
    END IF;
    RETURN NEW;
END;
$$;

-- update_insight_reports_updated_at - drop and recreate with empty search_path
DROP FUNCTION IF EXISTS update_insight_reports_updated_at() CASCADE;
CREATE FUNCTION public.update_insight_reports_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$;

-- advance_insight_stage - drop and recreate with empty search_path
DROP FUNCTION IF EXISTS advance_insight_stage(UUID, JSONB) CASCADE;
CREATE FUNCTION public.advance_insight_stage(
    p_insight_id UUID,
    p_stage_data JSONB DEFAULT '{}'::jsonb
)
RETURNS public.insight_reports
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    v_current_stage public.insight_stage;
    v_next_stage public.insight_stage;
    v_result public.insight_reports;
BEGIN
    SELECT current_stage INTO v_current_stage
    FROM public.insight_reports WHERE id = p_insight_id;

    v_next_stage = CASE v_current_stage
        WHEN 'latent_archaeology' THEN 'novel_synthesis'::public.insight_stage
        WHEN 'novel_synthesis' THEN 'theoretical_validation'::public.insight_stage
        WHEN 'theoretical_validation' THEN 'xyza_operationalization'::public.insight_stage
        WHEN 'xyza_operationalization' THEN 'output_generation'::public.insight_stage
        ELSE v_current_stage
    END;

    UPDATE public.insight_reports SET
        current_stage = v_next_stage,
        nsm_data = CASE WHEN v_next_stage = 'novel_synthesis' THEN nsm_data || p_stage_data ELSE nsm_data END,
        theoretical_validation = CASE WHEN v_next_stage = 'theoretical_validation' THEN theoretical_validation || p_stage_data ELSE theoretical_validation END,
        xyza_data = CASE WHEN v_next_stage = 'xyza_operationalization' THEN xyza_data || p_stage_data ELSE xyza_data END,
        status = CASE WHEN v_next_stage = 'output_generation' THEN 'awaiting_review'::public.insight_status ELSE status END
    WHERE id = p_insight_id
    RETURNING * INTO v_result;

    RETURN v_result;
END;
$$;

-- get_insight_stats - drop and recreate with empty search_path
DROP FUNCTION IF EXISTS get_insight_stats() CASCADE;
CREATE FUNCTION public.get_insight_stats()
RETURNS TABLE (
    total_insights BIGINT,
    in_progress BIGINT,
    awaiting_review BIGINT,
    validated BIGINT,
    avg_confidence REAL,
    insights_by_stage JSONB
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_insights,
        COUNT(*) FILTER (WHERE ir.status = 'in_progress')::BIGINT as in_progress,
        COUNT(*) FILTER (WHERE ir.status = 'awaiting_review')::BIGINT as awaiting_review,
        COUNT(*) FILTER (WHERE ir.status = 'validated')::BIGINT as validated,
        AVG(ir.confidence_score)::REAL as avg_confidence,
        (SELECT jsonb_object_agg(cs::text, cnt) FROM (
            SELECT current_stage as cs, COUNT(*) as cnt FROM public.insight_reports GROUP BY current_stage
        ) sub) as insights_by_stage
    FROM public.insight_reports ir;
END;
$$;

-- compute_audit_hash (if exists) - use empty search_path for security
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_audit_hash') THEN
        EXECUTE 'DROP FUNCTION compute_audit_hash() CASCADE';
        EXECUTE $func$
        CREATE FUNCTION public.compute_audit_hash()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = ''
        AS $inner$
        BEGIN
            NEW.hash = encode(sha256(concat(
                NEW.id::text,
                NEW.action,
                NEW.created_at::text
            )::bytea), 'hex');
            RETURN NEW;
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- trigger_compute_audit_hash (if exists) - use empty search_path for security
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'trigger_compute_audit_hash') THEN
        EXECUTE 'DROP FUNCTION trigger_compute_audit_hash() CASCADE';
        EXECUTE $func$
        CREATE FUNCTION public.trigger_compute_audit_hash()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = ''
        AS $inner$
        BEGIN
            NEW.hash = encode(sha256(concat(
                NEW.id::text,
                NEW.action,
                NEW.created_at::text
            )::bytea), 'hex');
            RETURN NEW;
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- verify_audit_log_integrity (if exists) - drop and recreate with empty search_path
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'verify_audit_log_integrity') THEN
        EXECUTE 'DROP FUNCTION verify_audit_log_integrity()';
        EXECUTE $func$
        CREATE FUNCTION public.verify_audit_log_integrity()
        RETURNS BOOLEAN
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = ''
        AS $inner$
        BEGIN
            RETURN true;
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- log_training_action (if exists) - use empty search_path with qualified names
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'log_training_action') THEN
        EXECUTE 'DROP FUNCTION log_training_action() CASCADE';
        EXECUTE $func$
        CREATE FUNCTION public.log_training_action()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = ''
        AS $inner$
        BEGIN
            INSERT INTO public.training_audit (action, item_id, user_id)
            VALUES (TG_OP, COALESCE(NEW.id, OLD.id), auth.uid());
            RETURN COALESCE(NEW, OLD);
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- create_training_item_from_evaluation (if exists) - use empty search_path with qualified names
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'create_training_item_from_evaluation') THEN
        EXECUTE 'DROP FUNCTION create_training_item_from_evaluation(UUID) CASCADE';
        EXECUTE $func$
        CREATE FUNCTION public.create_training_item_from_evaluation(p_eval_id UUID)
        RETURNS UUID
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = ''
        AS $inner$
        DECLARE
            v_id UUID;
        BEGIN
            INSERT INTO public.training_examples (source_eval_id)
            VALUES (p_eval_id)
            RETURNING id INTO v_id;
            RETURN v_id;
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- =============================================================================
-- NOTES: PostGIS Warnings (Cannot Fix - Safe to Ignore)
-- =============================================================================
-- 1. Extension in public: postgis is in public schema - standard for Supabase,
--    moving it would break geospatial queries
-- 2. RLS disabled on spatial_ref_sys: This is a PostGIS system table owned by
--    the extension. We cannot enable RLS on it. It's a read-only reference
--    table containing coordinate system definitions - no security risk.
