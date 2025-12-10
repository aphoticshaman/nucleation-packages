-- ═══════════════════════════════════════════════════════════════════════════════
-- GUARDIAN GOVERNANCE SYSTEM
-- Self-improving rule-based validation with versioning, metrics, and rollback
-- ═══════════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════════
-- 1. GUARDIAN RULES - Versioned rule storage
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS guardian_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version INT NOT NULL,

    -- Rule identification
    rule_name TEXT NOT NULL,
    domain TEXT NOT NULL,  -- 'political', 'economic', 'security', etc.
    rule_type TEXT NOT NULL DEFAULT 'validation',  -- 'validation', 'threshold', 'format', 'semantic'

    -- Rule content (JSON for flexibility)
    rule_config JSONB NOT NULL,
    -- Example: {"threshold": 0.7, "operator": ">=", "field": "confidence", "action": "reject"}

    -- Metadata
    description TEXT,
    rationale TEXT,  -- Why this rule exists

    -- Versioning
    is_active BOOLEAN NOT NULL DEFAULT false,  -- Only one active version per rule_name
    previous_version_id UUID REFERENCES guardian_rules(id),

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by UUID REFERENCES auth.users(id),
    activated_at TIMESTAMPTZ,
    deactivated_at TIMESTAMPTZ,

    -- Ensure unique active version per rule
    CONSTRAINT unique_active_rule UNIQUE (rule_name, is_active)
        DEFERRABLE INITIALLY DEFERRED
);

-- Index for fast lookups
CREATE INDEX idx_guardian_rules_active ON guardian_rules(rule_name, is_active) WHERE is_active = true;
CREATE INDEX idx_guardian_rules_domain ON guardian_rules(domain, is_active);
CREATE INDEX idx_guardian_rules_version ON guardian_rules(rule_name, version DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- 2. GUARDIAN EVALUATIONS - Decision log for Elle vs Guardian
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS guardian_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Evaluation context
    session_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    input_hash TEXT NOT NULL,  -- Blake3 hash of input for deduplication

    -- Elle's output
    elle_output JSONB NOT NULL,
    elle_confidence FLOAT,

    -- Guardian's decision
    guardian_decision TEXT NOT NULL,  -- 'accept', 'reject', 'modify'
    guardian_confidence FLOAT NOT NULL,
    rules_triggered TEXT[],  -- Array of rule_names that fired
    modifications_made JSONB,  -- If modified, what changed

    -- Ground truth (filled later by human review or automated check)
    ground_truth TEXT,  -- 'elle_correct', 'guardian_correct', 'both_wrong', 'both_correct'
    ground_truth_notes TEXT,
    reviewed_at TIMESTAMPTZ,
    reviewed_by UUID REFERENCES auth.users(id),

    -- Computed metrics (filled by trigger/cron)
    was_hallucination BOOLEAN,
    factualness_score FLOAT,  -- 0-1, verified against sources
    informativeness_score FLOAT,  -- Shannon entropy normalized

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for analytics
CREATE INDEX idx_guardian_evals_domain ON guardian_evaluations(domain, created_at DESC);
CREATE INDEX idx_guardian_evals_decision ON guardian_evaluations(guardian_decision, created_at DESC);
CREATE INDEX idx_guardian_evals_truth ON guardian_evaluations(ground_truth) WHERE ground_truth IS NOT NULL;
CREATE INDEX idx_guardian_evals_hallucination ON guardian_evaluations(was_hallucination) WHERE was_hallucination = true;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 3. GUARDIAN METRICS - Aggregated performance metrics
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS guardian_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Time bucket
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    granularity TEXT NOT NULL DEFAULT 'daily',  -- 'hourly', 'daily', 'weekly'

    -- Scope
    domain TEXT,  -- NULL for global, or specific domain
    rule_version INT,  -- Which ruleset version was active

    -- Volume metrics
    total_evaluations INT NOT NULL DEFAULT 0,
    accepts INT NOT NULL DEFAULT 0,
    rejects INT NOT NULL DEFAULT 0,
    modifications INT NOT NULL DEFAULT 0,

    -- Quality metrics (0-100 scale)
    accuracy_pct FLOAT,  -- (correct_guardian + correct_both) / reviewed
    hallucination_rate_pct FLOAT,  -- hallucinations / total
    false_positive_rate_pct FLOAT,  -- wrong_rejects / total_rejects
    false_negative_rate_pct FLOAT,  -- wrong_accepts / total_accepts

    -- Content quality (0-100 scale)
    avg_factualness FLOAT,
    avg_informativeness FLOAT,
    avg_depth FLOAT,
    avg_breadth FLOAT,

    -- Confidence calibration
    avg_guardian_confidence FLOAT,
    avg_elle_confidence FLOAT,
    confidence_correlation FLOAT,  -- How well confidence predicts correctness

    -- Computed at
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT unique_metric_period UNIQUE (period_start, granularity, domain)
);

CREATE INDEX idx_guardian_metrics_period ON guardian_metrics(period_start DESC, granularity);
CREATE INDEX idx_guardian_metrics_domain ON guardian_metrics(domain, period_start DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- 4. GUARDIAN RULE PROPOSALS - Auto-generated rule suggestions
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS guardian_rule_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Proposal content
    proposed_rule_name TEXT NOT NULL,
    proposed_domain TEXT NOT NULL,
    proposed_config JSONB NOT NULL,

    -- Source of proposal
    source TEXT NOT NULL,  -- 'elle_analysis', 'disagreement_pattern', 'human_suggestion'
    source_evaluation_ids UUID[],  -- Which evaluations triggered this

    -- Confidence and impact
    confidence_score FLOAT NOT NULL,  -- 0-1, how confident are we this is good
    predicted_accuracy_delta FLOAT,  -- Expected change in accuracy
    predicted_hallucination_delta FLOAT,  -- Expected change in hallucination rate

    -- Reasoning
    rationale TEXT NOT NULL,
    supporting_evidence JSONB,  -- Examples that support this rule

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'accepted', 'rejected', 'testing'

    -- Review
    reviewed_at TIMESTAMPTZ,
    reviewed_by UUID REFERENCES auth.users(id),
    review_notes TEXT,

    -- If accepted, which rule was created
    created_rule_id UUID REFERENCES guardian_rules(id),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_guardian_proposals_status ON guardian_rule_proposals(status, confidence_score DESC);
CREATE INDEX idx_guardian_proposals_domain ON guardian_rule_proposals(proposed_domain, status);

-- ═══════════════════════════════════════════════════════════════════════════════
-- 5. GUARDIAN AUDIT LOG - Track all changes for compliance
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS guardian_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What changed
    action TEXT NOT NULL,  -- 'rule_activated', 'rule_deactivated', 'rollback', 'proposal_accepted'
    entity_type TEXT NOT NULL,  -- 'rule', 'proposal', 'evaluation'
    entity_id UUID NOT NULL,

    -- Change details
    old_value JSONB,
    new_value JSONB,

    -- Who and when
    performed_by UUID REFERENCES auth.users(id),
    performed_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Context
    reason TEXT,
    ip_address INET
);

CREATE INDEX idx_guardian_audit_entity ON guardian_audit_log(entity_type, entity_id);
CREATE INDEX idx_guardian_audit_time ON guardian_audit_log(performed_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- 6. VIEWS FOR DASHBOARD
-- ═══════════════════════════════════════════════════════════════════════════════

-- Current active rules
CREATE OR REPLACE VIEW v_guardian_active_rules AS
SELECT
    r.id,
    r.rule_name,
    r.domain,
    r.rule_type,
    r.rule_config,
    r.description,
    r.version,
    r.activated_at,
    r.created_by
FROM guardian_rules r
WHERE r.is_active = true
ORDER BY r.domain, r.rule_name;

-- Latest metrics by domain
CREATE OR REPLACE VIEW v_guardian_latest_metrics AS
SELECT DISTINCT ON (domain)
    domain,
    period_start,
    period_end,
    total_evaluations,
    accuracy_pct,
    hallucination_rate_pct,
    false_positive_rate_pct,
    false_negative_rate_pct,
    avg_factualness,
    avg_informativeness
FROM guardian_metrics
WHERE granularity = 'daily'
ORDER BY domain, period_start DESC;

-- Disagreements needing review
CREATE OR REPLACE VIEW v_guardian_disagreements AS
SELECT
    e.id,
    e.domain,
    e.elle_output,
    e.guardian_decision,
    e.guardian_confidence,
    e.rules_triggered,
    e.created_at
FROM guardian_evaluations e
WHERE e.ground_truth IS NULL
  AND e.guardian_decision = 'reject'
  AND e.guardian_confidence < 0.8  -- Low confidence rejects need review
ORDER BY e.created_at DESC
LIMIT 100;

-- Pending proposals by confidence
CREATE OR REPLACE VIEW v_guardian_pending_proposals AS
SELECT
    p.id,
    p.proposed_rule_name,
    p.proposed_domain,
    p.proposed_config,
    p.confidence_score,
    p.predicted_accuracy_delta,
    p.rationale,
    p.created_at
FROM guardian_rule_proposals p
WHERE p.status = 'pending'
ORDER BY p.confidence_score DESC;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 7. FUNCTIONS FOR RULE MANAGEMENT
-- ═══════════════════════════════════════════════════════════════════════════════

-- Activate a new rule version (deactivates previous)
CREATE OR REPLACE FUNCTION activate_guardian_rule(
    p_rule_id UUID,
    p_user_id UUID DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    v_rule_name TEXT;
    v_old_rule_id UUID;
BEGIN
    -- Get the rule name
    SELECT rule_name INTO v_rule_name FROM guardian_rules WHERE id = p_rule_id;

    -- Find currently active rule with same name
    SELECT id INTO v_old_rule_id
    FROM guardian_rules
    WHERE rule_name = v_rule_name AND is_active = true;

    -- Deactivate old rule
    IF v_old_rule_id IS NOT NULL THEN
        UPDATE guardian_rules
        SET is_active = false, deactivated_at = now()
        WHERE id = v_old_rule_id;

        -- Log deactivation
        INSERT INTO guardian_audit_log (action, entity_type, entity_id, performed_by, reason)
        VALUES ('rule_deactivated', 'rule', v_old_rule_id, p_user_id, 'Replaced by newer version');
    END IF;

    -- Activate new rule
    UPDATE guardian_rules
    SET is_active = true, activated_at = now(), previous_version_id = v_old_rule_id
    WHERE id = p_rule_id;

    -- Log activation
    INSERT INTO guardian_audit_log (action, entity_type, entity_id, performed_by, reason)
    VALUES ('rule_activated', 'rule', p_rule_id, p_user_id, 'New version activated');
END;
$$ LANGUAGE plpgsql;

-- Rollback to previous rule version
CREATE OR REPLACE FUNCTION rollback_guardian_rule(
    p_rule_name TEXT,
    p_user_id UUID DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_current_id UUID;
    v_previous_id UUID;
BEGIN
    -- Find current active rule
    SELECT id, previous_version_id INTO v_current_id, v_previous_id
    FROM guardian_rules
    WHERE rule_name = p_rule_name AND is_active = true;

    IF v_previous_id IS NULL THEN
        RAISE EXCEPTION 'No previous version to rollback to for rule: %', p_rule_name;
    END IF;

    -- Deactivate current
    UPDATE guardian_rules
    SET is_active = false, deactivated_at = now()
    WHERE id = v_current_id;

    -- Reactivate previous
    UPDATE guardian_rules
    SET is_active = true, activated_at = now()
    WHERE id = v_previous_id;

    -- Log rollback
    INSERT INTO guardian_audit_log (action, entity_type, entity_id, performed_by, reason)
    VALUES ('rollback', 'rule', v_current_id, p_user_id,
            format('Rolled back to version %s', (SELECT version FROM guardian_rules WHERE id = v_previous_id)));

    RETURN v_previous_id;
END;
$$ LANGUAGE plpgsql;

-- Compute metrics for a time period
CREATE OR REPLACE FUNCTION compute_guardian_metrics(
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ,
    p_granularity TEXT DEFAULT 'daily'
) RETURNS VOID AS $$
BEGIN
    INSERT INTO guardian_metrics (
        period_start, period_end, granularity, domain,
        total_evaluations, accepts, rejects, modifications,
        accuracy_pct, hallucination_rate_pct,
        false_positive_rate_pct, false_negative_rate_pct,
        avg_factualness, avg_informativeness,
        avg_guardian_confidence, avg_elle_confidence
    )
    SELECT
        p_start,
        p_end,
        p_granularity,
        domain,
        COUNT(*) as total_evaluations,
        COUNT(*) FILTER (WHERE guardian_decision = 'accept') as accepts,
        COUNT(*) FILTER (WHERE guardian_decision = 'reject') as rejects,
        COUNT(*) FILTER (WHERE guardian_decision = 'modify') as modifications,
        -- Accuracy: correct decisions / reviewed
        (COUNT(*) FILTER (WHERE ground_truth IN ('guardian_correct', 'both_correct'))::FLOAT /
            NULLIF(COUNT(*) FILTER (WHERE ground_truth IS NOT NULL), 0)) * 100,
        -- Hallucination rate
        (COUNT(*) FILTER (WHERE was_hallucination = true)::FLOAT / NULLIF(COUNT(*), 0)) * 100,
        -- False positive rate (wrongly rejected)
        (COUNT(*) FILTER (WHERE ground_truth = 'elle_correct' AND guardian_decision = 'reject')::FLOAT /
            NULLIF(COUNT(*) FILTER (WHERE guardian_decision = 'reject'), 0)) * 100,
        -- False negative rate (wrongly accepted)
        (COUNT(*) FILTER (WHERE ground_truth = 'both_wrong' AND guardian_decision = 'accept')::FLOAT /
            NULLIF(COUNT(*) FILTER (WHERE guardian_decision = 'accept'), 0)) * 100,
        AVG(factualness_score) * 100,
        AVG(informativeness_score) * 100,
        AVG(guardian_confidence),
        AVG(elle_confidence)
    FROM guardian_evaluations
    WHERE created_at >= p_start AND created_at < p_end
    GROUP BY domain
    ON CONFLICT (period_start, granularity, domain)
    DO UPDATE SET
        total_evaluations = EXCLUDED.total_evaluations,
        accepts = EXCLUDED.accepts,
        rejects = EXCLUDED.rejects,
        modifications = EXCLUDED.modifications,
        accuracy_pct = EXCLUDED.accuracy_pct,
        hallucination_rate_pct = EXCLUDED.hallucination_rate_pct,
        false_positive_rate_pct = EXCLUDED.false_positive_rate_pct,
        false_negative_rate_pct = EXCLUDED.false_negative_rate_pct,
        avg_factualness = EXCLUDED.avg_factualness,
        avg_informativeness = EXCLUDED.avg_informativeness,
        avg_guardian_confidence = EXCLUDED.avg_guardian_confidence,
        avg_elle_confidence = EXCLUDED.avg_elle_confidence,
        computed_at = now();
END;
$$ LANGUAGE plpgsql;

-- ═══════════════════════════════════════════════════════════════════════════════
-- 8. RLS POLICIES
-- ═══════════════════════════════════════════════════════════════════════════════

ALTER TABLE guardian_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE guardian_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE guardian_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE guardian_rule_proposals ENABLE ROW LEVEL SECURITY;
ALTER TABLE guardian_audit_log ENABLE ROW LEVEL SECURITY;

-- Admin-only access for rules and proposals
CREATE POLICY guardian_rules_admin ON guardian_rules
    FOR ALL USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

CREATE POLICY guardian_proposals_admin ON guardian_rule_proposals
    FOR ALL USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

CREATE POLICY guardian_audit_admin ON guardian_audit_log
    FOR ALL USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

-- Read-only for metrics (all authenticated users)
CREATE POLICY guardian_metrics_read ON guardian_metrics
    FOR SELECT USING (auth.uid() IS NOT NULL);

-- Evaluations: admins can read all, service role can insert
CREATE POLICY guardian_evals_admin_read ON guardian_evaluations
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

CREATE POLICY guardian_evals_service_insert ON guardian_evaluations
    FOR INSERT WITH CHECK (true);  -- Service role only (used by API)

-- ═══════════════════════════════════════════════════════════════════════════════
-- 9. SEED DEFAULT RULES
-- ═══════════════════════════════════════════════════════════════════════════════

-- JSON format validation rule
INSERT INTO guardian_rules (rule_name, domain, rule_type, version, rule_config, description, rationale, is_active, activated_at)
VALUES (
    'json_format_valid',
    'global',
    'format',
    1,
    '{"check": "json_parse", "required_keys": ["political", "economic", "security", "summary", "nsm"]}',
    'Validate that output is valid JSON with required keys',
    'Ensures Elle outputs can be parsed and displayed correctly',
    true,
    now()
);

-- No hallucinated country codes
INSERT INTO guardian_rules (rule_name, domain, rule_type, version, rule_config, description, rationale, is_active, activated_at)
VALUES (
    'valid_country_codes',
    'global',
    'validation',
    1,
    '{"check": "country_code_exists", "source": "iso_3166_alpha3", "action": "reject"}',
    'Reject outputs containing non-existent country codes',
    'Prevents hallucinated geography from reaching users',
    true,
    now()
);

-- Confidence threshold
INSERT INTO guardian_rules (rule_name, domain, rule_type, version, rule_config, description, rationale, is_active, activated_at)
VALUES (
    'min_confidence',
    'global',
    'threshold',
    1,
    '{"field": "confidence", "operator": ">=", "threshold": 0.3, "action": "flag"}',
    'Flag low-confidence outputs for review',
    'Low confidence outputs may need human verification',
    true,
    now()
);

-- Political risk bounds
INSERT INTO guardian_rules (rule_name, domain, rule_type, version, rule_config, description, rationale, is_active, activated_at)
VALUES (
    'political_risk_bounds',
    'political',
    'threshold',
    1,
    '{"field": "risk_score", "min": 0, "max": 100, "action": "clamp"}',
    'Ensure political risk scores are within 0-100',
    'Prevents out-of-bounds risk scores from confusing users',
    true,
    now()
);

-- No future dates
INSERT INTO guardian_rules (rule_name, domain, rule_type, version, rule_config, description, rationale, is_active, activated_at)
VALUES (
    'no_future_dates',
    'global',
    'validation',
    1,
    '{"check": "date_not_future", "tolerance_hours": 24, "action": "reject"}',
    'Reject outputs referencing future events as facts',
    'Elle cannot predict the future with certainty',
    true,
    now()
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- 10. GRANTS
-- ═══════════════════════════════════════════════════════════════════════════════

GRANT SELECT ON v_guardian_active_rules TO authenticated;
GRANT SELECT ON v_guardian_latest_metrics TO authenticated;
GRANT SELECT ON v_guardian_disagreements TO authenticated;
GRANT SELECT ON v_guardian_pending_proposals TO authenticated;

GRANT EXECUTE ON FUNCTION activate_guardian_rule TO authenticated;
GRANT EXECUTE ON FUNCTION rollback_guardian_rule TO authenticated;
GRANT EXECUTE ON FUNCTION compute_guardian_metrics TO authenticated;
