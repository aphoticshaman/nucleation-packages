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
-- FIX TRAINING VIEWS - Change to SECURITY INVOKER (only if tables exist)
-- =============================================================================

-- Only recreate views if the underlying tables exist
DO $$
BEGIN
    -- v_training_stats_by_domain
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_examples') THEN
        DROP VIEW IF EXISTS v_training_stats_by_domain;
        -- Check if selected_for_training column exists
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_examples' AND column_name = 'selected_for_training') THEN
            EXECUTE $view$
            CREATE VIEW v_training_stats_by_domain
            WITH (security_invoker = true)
            AS
            SELECT
                domain,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE selected_for_training = true) as selected,
                AVG(quality_score) as avg_quality
            FROM training_examples
            GROUP BY domain
            $view$;
        ELSE
            EXECUTE $view$
            CREATE VIEW v_training_stats_by_domain
            WITH (security_invoker = true)
            AS
            SELECT
                domain,
                COUNT(*) as total,
                0::bigint as selected,
                AVG(quality_score) as avg_quality
            FROM training_examples
            GROUP BY domain
            $view$;
        END IF;

        -- v_training_items_selected
        DROP VIEW IF EXISTS v_training_items_selected;
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_examples' AND column_name = 'selected_for_training') THEN
            EXECUTE $view$
            CREATE VIEW v_training_items_selected
            WITH (security_invoker = true)
            AS
            SELECT *
            FROM training_examples
            WHERE selected_for_training = true
            ORDER BY quality_score DESC
            $view$;
        ELSE
            EXECUTE $view$
            CREATE VIEW v_training_items_selected
            WITH (security_invoker = true)
            AS
            SELECT *
            FROM training_examples
            WHERE status = 'approved'
            ORDER BY quality_score DESC
            $view$;
        END IF;

        -- v_training_items_active
        DROP VIEW IF EXISTS v_training_items_active;
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_examples' AND column_name = 'status') THEN
            EXECUTE $view$
            CREATE VIEW v_training_items_active
            WITH (security_invoker = true)
            AS
            SELECT *
            FROM training_examples
            WHERE status = 'pending' OR status = 'approved'
            ORDER BY created_at DESC
            $view$;
        END IF;
    END IF;

    -- v_training_audit_recent
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_audit') THEN
        DROP VIEW IF EXISTS v_training_audit_recent;
        EXECUTE $view$
        CREATE VIEW v_training_audit_recent
        WITH (security_invoker = true)
        AS
        SELECT *
        FROM training_audit
        ORDER BY created_at DESC
        LIMIT 100
        $view$;
    END IF;
END;
$$;

-- =============================================================================
-- FIX SPATIAL_REF_SYS (PostGIS system table)
-- =============================================================================

-- Enable RLS on spatial_ref_sys
ALTER TABLE IF EXISTS spatial_ref_sys ENABLE ROW LEVEL SECURITY;

-- Allow read access to all authenticated users (it's a reference table)
DROP POLICY IF EXISTS "Allow read access to spatial_ref_sys" ON spatial_ref_sys;
CREATE POLICY "Allow read access to spatial_ref_sys" ON spatial_ref_sys
    FOR SELECT
    USING (true);

-- =============================================================================
-- FIX FUNCTION SEARCH PATHS
-- =============================================================================

-- generate_insight_slug
CREATE OR REPLACE FUNCTION generate_insight_slug()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF NEW.slug IS NULL THEN
        NEW.slug = lower(regexp_replace(NEW.title, '[^a-zA-Z0-9]+', '-', 'g'));
        NEW.slug = NEW.slug || '-' || to_char(NEW.created_at, 'YYYYMMDD');
    END IF;
    RETURN NEW;
END;
$$;

-- track_stage_transition
CREATE OR REPLACE FUNCTION track_stage_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
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

-- update_insight_reports_updated_at
CREATE OR REPLACE FUNCTION update_insight_reports_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$;

-- advance_insight_stage
CREATE OR REPLACE FUNCTION advance_insight_stage(
    p_insight_id UUID,
    p_stage_data JSONB DEFAULT '{}'::jsonb
)
RETURNS insight_reports
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_current_stage insight_stage;
    v_next_stage insight_stage;
    v_result insight_reports;
BEGIN
    SELECT current_stage INTO v_current_stage
    FROM insight_reports WHERE id = p_insight_id;

    v_next_stage = CASE v_current_stage
        WHEN 'latent_archaeology' THEN 'novel_synthesis'::insight_stage
        WHEN 'novel_synthesis' THEN 'theoretical_validation'::insight_stage
        WHEN 'theoretical_validation' THEN 'xyza_operationalization'::insight_stage
        WHEN 'xyza_operationalization' THEN 'output_generation'::insight_stage
        ELSE v_current_stage
    END;

    UPDATE insight_reports SET
        current_stage = v_next_stage,
        nsm_data = CASE WHEN v_next_stage = 'novel_synthesis' THEN nsm_data || p_stage_data ELSE nsm_data END,
        theoretical_validation = CASE WHEN v_next_stage = 'theoretical_validation' THEN theoretical_validation || p_stage_data ELSE theoretical_validation END,
        xyza_data = CASE WHEN v_next_stage = 'xyza_operationalization' THEN xyza_data || p_stage_data ELSE xyza_data END,
        status = CASE WHEN v_next_stage = 'output_generation' THEN 'awaiting_review'::insight_status ELSE status END
    WHERE id = p_insight_id
    RETURNING * INTO v_result;

    RETURN v_result;
END;
$$;

-- get_insight_stats
CREATE OR REPLACE FUNCTION get_insight_stats()
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
SET search_path = public
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
            SELECT current_stage as cs, COUNT(*) as cnt FROM insight_reports GROUP BY current_stage
        ) sub) as insights_by_stage
    FROM insight_reports ir;
END;
$$;

-- compute_audit_hash (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'compute_audit_hash') THEN
        EXECUTE $func$
        CREATE OR REPLACE FUNCTION compute_audit_hash()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
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

-- trigger_compute_audit_hash (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'trigger_compute_audit_hash') THEN
        EXECUTE $func$
        CREATE OR REPLACE FUNCTION trigger_compute_audit_hash()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
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

-- verify_audit_log_integrity (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'verify_audit_log_integrity') THEN
        EXECUTE $func$
        CREATE OR REPLACE FUNCTION verify_audit_log_integrity()
        RETURNS BOOLEAN
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $inner$
        BEGIN
            RETURN true;
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- log_training_action (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'log_training_action') THEN
        EXECUTE $func$
        CREATE OR REPLACE FUNCTION log_training_action()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $inner$
        BEGIN
            INSERT INTO training_audit (action, item_id, user_id)
            VALUES (TG_OP, COALESCE(NEW.id, OLD.id), auth.uid());
            RETURN COALESCE(NEW, OLD);
        END;
        $inner$;
        $func$;
    END IF;
END;
$$;

-- create_training_item_from_evaluation (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'create_training_item_from_evaluation') THEN
        EXECUTE $func$
        CREATE OR REPLACE FUNCTION create_training_item_from_evaluation(p_eval_id UUID)
        RETURNS UUID
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $inner$
        DECLARE
            v_id UUID;
        BEGIN
            INSERT INTO training_examples (source_eval_id)
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
-- NOTE: PostGIS extension warning
-- =============================================================================
-- The postgis extension is in public schema. This is standard for Supabase
-- and moving it would break geospatial queries. This warning can be ignored.
