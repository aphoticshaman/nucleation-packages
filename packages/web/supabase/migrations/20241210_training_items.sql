-- ═══════════════════════════════════════════════════════════════════════════════
-- TRAINING ITEMS: Granular, selectable training data for Elle and Guardian
-- ═══════════════════════════════════════════════════════════════════════════════
--
-- Design principles:
-- 1. Every training item is individually selectable
-- 2. Items can be marked for inclusion/exclusion in training exports
-- 3. All decisions are immutably logged (append-only audit)
-- 4. Supports both Elle (LLM) and Guardian (symbolic) training
-- 5. Exportable in various formats for different training pipelines
--
-- ═══════════════════════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────────────────────────
-- TRAINING ITEMS TABLE
-- Stores individual training items with metadata
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source identification
    source_type VARCHAR(50) NOT NULL, -- 'evaluation', 'disagreement', 'proposal', 'correction', 'synthetic', 'human_authored'
    source_id UUID, -- Reference to source record (evaluation, proposal, etc.)

    -- Target system
    target_system VARCHAR(20) NOT NULL CHECK (target_system IN ('elle', 'guardian', 'both')),

    -- Content
    domain VARCHAR(50), -- 'political', 'economic', 'security', etc.
    input_text TEXT NOT NULL, -- The input/prompt
    expected_output TEXT, -- Ground truth or expected response
    actual_output TEXT, -- What the system actually produced (if applicable)

    -- Training metadata
    training_type VARCHAR(50) NOT NULL, -- 'positive', 'negative', 'correction', 'reinforcement'
    quality_score DECIMAL(3,2), -- 0.00-1.00 quality rating
    difficulty_level VARCHAR(20), -- 'easy', 'medium', 'hard', 'edge_case'
    tags TEXT[], -- Flexible tagging for filtering

    -- Selection state
    selected_for_export BOOLEAN DEFAULT FALSE,
    export_priority INTEGER DEFAULT 0, -- Higher = export first

    -- Lifecycle
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'exported', 'archived')),

    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES auth.users(id),
    reviewed_at TIMESTAMPTZ,
    reviewed_by UUID REFERENCES auth.users(id),
    exported_at TIMESTAMPTZ,
    exported_by UUID REFERENCES auth.users(id),

    -- Soft delete (we never hard delete training items for auditability)
    deleted_at TIMESTAMPTZ,
    deleted_by UUID REFERENCES auth.users(id),
    deletion_reason TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_training_items_source ON training_items(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_training_items_target ON training_items(target_system);
CREATE INDEX IF NOT EXISTS idx_training_items_domain ON training_items(domain);
CREATE INDEX IF NOT EXISTS idx_training_items_status ON training_items(status);
CREATE INDEX IF NOT EXISTS idx_training_items_selected ON training_items(selected_for_export) WHERE selected_for_export = TRUE;
CREATE INDEX IF NOT EXISTS idx_training_items_tags ON training_items USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_training_items_created ON training_items(created_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────────
-- TRAINING EXPORT BATCHES
-- Groups of training items exported together
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_export_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Export metadata
    export_name VARCHAR(255) NOT NULL,
    target_system VARCHAR(20) NOT NULL CHECK (target_system IN ('elle', 'guardian', 'both')),
    export_format VARCHAR(50) NOT NULL, -- 'jsonl', 'parquet', 'csv', 'alpaca', 'sharegpt'

    -- Statistics
    total_items INTEGER NOT NULL DEFAULT 0,
    elle_items INTEGER NOT NULL DEFAULT 0,
    guardian_items INTEGER NOT NULL DEFAULT 0,

    -- Filters used
    filters_applied JSONB, -- Record of what filters were used to select items

    -- File info
    file_url TEXT, -- URL to exported file (if stored)
    file_hash TEXT, -- SHA256 of exported file for integrity
    file_size_bytes BIGINT,

    -- Lifecycle
    status VARCHAR(20) DEFAULT 'created' CHECK (status IN ('created', 'exporting', 'completed', 'failed')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES auth.users(id) NOT NULL,
    completed_at TIMESTAMPTZ,

    -- Notes
    notes TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────────
-- TRAINING ITEM BATCH JUNCTION
-- Links items to export batches (many-to-many)
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_item_exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_item_id UUID NOT NULL REFERENCES training_items(id),
    export_batch_id UUID NOT NULL REFERENCES training_export_batches(id),
    exported_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure each item only appears once per batch
    UNIQUE(training_item_id, export_batch_id)
);

CREATE INDEX IF NOT EXISTS idx_item_exports_batch ON training_item_exports(export_batch_id);
CREATE INDEX IF NOT EXISTS idx_item_exports_item ON training_item_exports(training_item_id);

-- ─────────────────────────────────────────────────────────────────────────────────
-- IMMUTABLE TRAINING AUDIT LOG
-- Append-only log of ALL training-related decisions
-- This table uses INSERT-only policy - no updates or deletes allowed
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What happened
    action VARCHAR(100) NOT NULL, -- 'item_created', 'item_selected', 'item_deselected', 'item_approved', 'item_rejected', 'batch_exported', etc.

    -- Who did it
    performed_by UUID REFERENCES auth.users(id) NOT NULL,
    performed_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,

    -- What it affected
    entity_type VARCHAR(50) NOT NULL, -- 'training_item', 'export_batch', 'bulk_selection'
    entity_id UUID, -- ID of affected entity
    entity_ids UUID[], -- For bulk operations

    -- State change
    previous_state JSONB, -- State before action (for items: {selected, status, etc.})
    new_state JSONB, -- State after action

    -- Context
    reason TEXT, -- Human-provided reason for the action
    metadata JSONB, -- Additional context (filters used, export format, etc.)

    -- Immutability enforcement
    -- This hash chains entries for tamper detection
    previous_hash TEXT, -- Hash of previous entry
    entry_hash TEXT -- Hash of this entry (computed on insert)
);

-- Index for querying audit history
CREATE INDEX IF NOT EXISTS idx_training_audit_performed ON training_audit_log(performed_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_audit_action ON training_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_training_audit_entity ON training_audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_training_audit_user ON training_audit_log(performed_by);

-- ─────────────────────────────────────────────────────────────────────────────────
-- FUNCTION: Compute hash for audit entry (for chain integrity)
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION compute_audit_hash(
    p_action TEXT,
    p_performed_by UUID,
    p_performed_at TIMESTAMPTZ,
    p_entity_type TEXT,
    p_entity_id UUID,
    p_previous_hash TEXT
) RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN encode(
        sha256(
            (COALESCE(p_action, '') ||
             COALESCE(p_performed_by::TEXT, '') ||
             COALESCE(p_performed_at::TEXT, '') ||
             COALESCE(p_entity_type, '') ||
             COALESCE(p_entity_id::TEXT, '') ||
             COALESCE(p_previous_hash, 'GENESIS'))::bytea
        ),
        'hex'
    );
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────────
-- TRIGGER: Auto-compute hash on audit log insert
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION trigger_compute_audit_hash()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_previous_hash TEXT;
BEGIN
    -- Get hash of most recent entry
    SELECT entry_hash INTO v_previous_hash
    FROM training_audit_log
    ORDER BY performed_at DESC, id DESC
    LIMIT 1;

    -- Set previous hash reference
    NEW.previous_hash := COALESCE(v_previous_hash, 'GENESIS');

    -- Compute this entry's hash
    NEW.entry_hash := compute_audit_hash(
        NEW.action,
        NEW.performed_by,
        NEW.performed_at,
        NEW.entity_type,
        NEW.entity_id,
        NEW.previous_hash
    );

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_compute_audit_hash ON training_audit_log;
CREATE TRIGGER trg_compute_audit_hash
    BEFORE INSERT ON training_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION trigger_compute_audit_hash();

-- ─────────────────────────────────────────────────────────────────────────────────
-- FUNCTION: Log training action (with immutability)
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION log_training_action(
    p_action TEXT,
    p_user_id UUID,
    p_entity_type TEXT,
    p_entity_id UUID DEFAULT NULL,
    p_entity_ids UUID[] DEFAULT NULL,
    p_previous_state JSONB DEFAULT NULL,
    p_new_state JSONB DEFAULT NULL,
    p_reason TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_log_id UUID;
BEGIN
    INSERT INTO training_audit_log (
        action,
        performed_by,
        entity_type,
        entity_id,
        entity_ids,
        previous_state,
        new_state,
        reason,
        metadata
    ) VALUES (
        p_action,
        p_user_id,
        p_entity_type,
        p_entity_id,
        p_entity_ids,
        p_previous_state,
        p_new_state,
        p_reason,
        p_metadata
    )
    RETURNING id INTO v_log_id;

    RETURN v_log_id;
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────────
-- FUNCTION: Verify audit log integrity (detect tampering)
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION verify_audit_log_integrity()
RETURNS TABLE (
    is_valid BOOLEAN,
    invalid_entries UUID[],
    total_entries BIGINT,
    verified_entries BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_entry RECORD;
    v_expected_hash TEXT;
    v_invalid UUID[];
    v_total BIGINT;
    v_verified BIGINT := 0;
BEGIN
    v_invalid := ARRAY[]::UUID[];

    SELECT COUNT(*) INTO v_total FROM training_audit_log;

    FOR v_entry IN
        SELECT * FROM training_audit_log
        ORDER BY performed_at ASC, id ASC
    LOOP
        v_expected_hash := compute_audit_hash(
            v_entry.action,
            v_entry.performed_by,
            v_entry.performed_at,
            v_entry.entity_type,
            v_entry.entity_id,
            v_entry.previous_hash
        );

        IF v_entry.entry_hash = v_expected_hash THEN
            v_verified := v_verified + 1;
        ELSE
            v_invalid := array_append(v_invalid, v_entry.id);
        END IF;
    END LOOP;

    RETURN QUERY SELECT
        (array_length(v_invalid, 1) IS NULL OR array_length(v_invalid, 1) = 0),
        v_invalid,
        v_total,
        v_verified;
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────────
-- VIEWS
-- ─────────────────────────────────────────────────────────────────────────────────

-- Active (non-deleted) training items
CREATE OR REPLACE VIEW v_training_items_active AS
SELECT
    ti.*,
    (SELECT COUNT(*) FROM training_item_exports tie WHERE tie.training_item_id = ti.id) as export_count
FROM training_items ti
WHERE ti.deleted_at IS NULL;

-- Selected items ready for export
CREATE OR REPLACE VIEW v_training_items_selected AS
SELECT * FROM v_training_items_active
WHERE selected_for_export = TRUE
  AND status IN ('pending', 'approved')
ORDER BY export_priority DESC, created_at ASC;

-- Training items by domain with stats
CREATE OR REPLACE VIEW v_training_stats_by_domain AS
SELECT
    domain,
    target_system,
    COUNT(*) as total_items,
    COUNT(*) FILTER (WHERE selected_for_export = TRUE) as selected_items,
    COUNT(*) FILTER (WHERE status = 'approved') as approved_items,
    COUNT(*) FILTER (WHERE status = 'exported') as exported_items,
    AVG(quality_score) as avg_quality_score
FROM v_training_items_active
GROUP BY domain, target_system
ORDER BY domain, target_system;

-- Recent training audit log with user info
CREATE OR REPLACE VIEW v_training_audit_recent AS
SELECT
    tal.*,
    p.email as performed_by_email,
    p.full_name as performed_by_name
FROM training_audit_log tal
LEFT JOIN profiles p ON p.id = tal.performed_by
ORDER BY tal.performed_at DESC
LIMIT 100;

-- ─────────────────────────────────────────────────────────────────────────────────
-- RLS POLICIES (Admin only for training management)
-- ─────────────────────────────────────────────────────────────────────────────────
ALTER TABLE training_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_export_batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_item_exports ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_audit_log ENABLE ROW LEVEL SECURITY;

-- Training items: Admin read/write
DROP POLICY IF EXISTS training_items_admin ON training_items;
CREATE POLICY training_items_admin ON training_items
    FOR ALL
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE id = auth.uid()
            AND role = 'admin'
        )
    );

-- Export batches: Admin read/write
DROP POLICY IF EXISTS training_batches_admin ON training_export_batches;
CREATE POLICY training_batches_admin ON training_export_batches
    FOR ALL
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE id = auth.uid()
            AND role = 'admin'
        )
    );

-- Item exports junction: Admin read/write
DROP POLICY IF EXISTS training_item_exports_admin ON training_item_exports;
CREATE POLICY training_item_exports_admin ON training_item_exports
    FOR ALL
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE id = auth.uid()
            AND role = 'admin'
        )
    );

-- Audit log: Admin INSERT only (read for all admins, no update/delete)
DROP POLICY IF EXISTS training_audit_insert ON training_audit_log;
CREATE POLICY training_audit_insert ON training_audit_log
    FOR INSERT
    TO authenticated
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE id = auth.uid()
            AND role = 'admin'
        )
    );

DROP POLICY IF EXISTS training_audit_select ON training_audit_log;
CREATE POLICY training_audit_select ON training_audit_log
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE id = auth.uid()
            AND role = 'admin'
        )
    );

-- IMPORTANT: No UPDATE or DELETE policies on audit log = immutable

-- ─────────────────────────────────────────────────────────────────────────────────
-- HELPER FUNCTION: Create training item from evaluation disagreement
-- ─────────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION create_training_item_from_evaluation(
    p_evaluation_id UUID,
    p_user_id UUID
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_eval RECORD;
    v_item_id UUID;
    v_target VARCHAR(20);
    v_training_type VARCHAR(50);
BEGIN
    -- Get evaluation
    SELECT * INTO v_eval FROM guardian_evaluations WHERE id = p_evaluation_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Evaluation not found: %', p_evaluation_id;
    END IF;

    -- Determine target based on ground truth
    IF v_eval.ground_truth = 'elle_correct' THEN
        v_target := 'guardian';
        v_training_type := 'correction';
    ELSIF v_eval.ground_truth = 'guardian_correct' THEN
        v_target := 'elle';
        v_training_type := 'correction';
    ELSIF v_eval.ground_truth = 'both_correct' THEN
        v_target := 'both';
        v_training_type := 'reinforcement';
    ELSE
        v_target := 'both';
        v_training_type := 'negative';
    END IF;

    -- Create training item
    INSERT INTO training_items (
        source_type,
        source_id,
        target_system,
        domain,
        input_text,
        expected_output,
        actual_output,
        training_type,
        tags,
        created_by
    ) VALUES (
        'evaluation',
        p_evaluation_id,
        v_target,
        v_eval.domain,
        v_eval.input_summary,
        CASE
            WHEN v_eval.ground_truth IN ('elle_correct', 'both_correct') THEN v_eval.elle_reasoning
            ELSE v_eval.guardian_reasoning
        END,
        CASE
            WHEN v_eval.ground_truth IN ('guardian_correct', 'both_wrong') THEN v_eval.elle_reasoning
            ELSE NULL
        END,
        v_training_type,
        ARRAY[v_eval.domain, v_eval.ground_truth, 'auto_generated'],
        p_user_id
    )
    RETURNING id INTO v_item_id;

    -- Log the action
    PERFORM log_training_action(
        'item_created_from_evaluation',
        p_user_id,
        'training_item',
        v_item_id,
        NULL,
        NULL,
        jsonb_build_object('source_evaluation', p_evaluation_id, 'target', v_target),
        'Auto-generated from evaluation review',
        jsonb_build_object('ground_truth', v_eval.ground_truth)
    );

    RETURN v_item_id;
END;
$$;

-- Grant execute to authenticated users (RLS will still enforce admin-only)
GRANT EXECUTE ON FUNCTION log_training_action TO authenticated;
GRANT EXECUTE ON FUNCTION verify_audit_log_integrity TO authenticated;
GRANT EXECUTE ON FUNCTION create_training_item_from_evaluation TO authenticated;
