-- ============================================================
-- LATTICE LEARNING INFRASTRUCTURE
-- Migration for data collection and model training pipeline
-- ============================================================

-- Learning events table (main data collection)
CREATE TABLE IF NOT EXISTS learning_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL CHECK (type IN (
        'reasoning_trace',
        'llm_interaction',
        'user_feedback',
        'signal_observation',
        'prediction_outcome',
        'api_call'
    )),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_hash TEXT NOT NULL,  -- Anonymized session ID
    user_tier TEXT NOT NULL DEFAULT 'consumer',
    domain TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Indexes for efficient querying
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for learning_events
CREATE INDEX IF NOT EXISTS idx_learning_events_type ON learning_events(type);
CREATE INDEX IF NOT EXISTS idx_learning_events_timestamp ON learning_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_learning_events_domain ON learning_events(domain);
CREATE INDEX IF NOT EXISTS idx_learning_events_session ON learning_events(session_hash);
CREATE INDEX IF NOT EXISTS idx_learning_events_type_timestamp ON learning_events(type, timestamp);

-- Security logs table (audit trail)
CREATE TABLE IF NOT EXISTS security_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id_hash TEXT NOT NULL,  -- Anonymized user ID
    event_type TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),

    -- For compliance reporting
    reviewed BOOLEAN DEFAULT FALSE,
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT
);

-- Indexes for security_logs
CREATE INDEX IF NOT EXISTS idx_security_logs_event_type ON security_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_security_logs_timestamp ON security_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_logs_severity ON security_logs(severity);
CREATE INDEX IF NOT EXISTS idx_security_logs_unreviewed ON security_logs(reviewed) WHERE reviewed = FALSE;

-- Reasoning traces table (detailed trace storage)
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_hash TEXT NOT NULL,
    query_intent TEXT NOT NULL,
    domain TEXT NOT NULL,
    input_features JSONB NOT NULL DEFAULT '{}',

    -- Trace steps
    steps JSONB NOT NULL DEFAULT '[]',

    -- Output
    conclusion TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    engines_used TEXT[] NOT NULL DEFAULT '{}',

    -- Metadata
    compute_time_ms INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- For training data export
    exported BOOLEAN DEFAULT FALSE,
    export_batch TEXT
);

-- Indexes for reasoning_traces
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_domain ON reasoning_traces(domain);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_confidence ON reasoning_traces(confidence);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_timestamp ON reasoning_traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_unexported ON reasoning_traces(exported) WHERE exported = FALSE;

-- Prediction outcomes table (ground truth for training)
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID,  -- Links to the original prediction
    domain TEXT NOT NULL,

    -- What we predicted
    predicted_state TEXT NOT NULL,
    predicted_confidence FLOAT NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,

    -- What actually happened
    actual_state TEXT NOT NULL,
    outcome_timestamp TIMESTAMPTZ NOT NULL,

    -- Analysis
    correct BOOLEAN NOT NULL,
    lead_time_days INTEGER NOT NULL,
    error_magnitude FLOAT,

    -- For model improvement
    analyzed BOOLEAN DEFAULT FALSE,
    analysis_notes TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for prediction_outcomes
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_domain ON prediction_outcomes(domain);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_correct ON prediction_outcomes(correct);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_timestamp ON prediction_outcomes(outcome_timestamp);

-- Historical cases table (for analogical reasoning)
CREATE TABLE IF NOT EXISTS historical_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id TEXT UNIQUE NOT NULL,  -- Human-readable identifier
    domain TEXT NOT NULL,

    -- Case description
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,

    -- Features for similarity matching
    features JSONB NOT NULL DEFAULT '{}',
    feature_vector FLOAT[] NOT NULL DEFAULT '{}',  -- For vector similarity

    -- Outcome and lessons
    outcome TEXT NOT NULL,
    lessons TEXT[] NOT NULL DEFAULT '{}',
    severity TEXT CHECK (severity IN ('minor', 'moderate', 'major', 'catastrophic')),

    -- Metadata
    sources TEXT[] NOT NULL DEFAULT '{}',
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for historical_cases
CREATE INDEX IF NOT EXISTS idx_historical_cases_domain ON historical_cases(domain);
CREATE INDEX IF NOT EXISTS idx_historical_cases_start_date ON historical_cases(start_date);
CREATE INDEX IF NOT EXISTS idx_historical_cases_verified ON historical_cases(verified);

-- Rate limiting table
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 1,

    UNIQUE(user_id, endpoint, window_start)
);

-- Index for rate_limits
CREATE INDEX IF NOT EXISTS idx_rate_limits_lookup ON rate_limits(user_id, endpoint, window_start);

-- Training batches table (tracks model training runs)
CREATE TABLE IF NOT EXISTS training_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_name TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('preparing', 'training', 'completed', 'failed')),

    -- Data range
    data_start TIMESTAMPTZ NOT NULL,
    data_end TIMESTAMPTZ NOT NULL,
    event_count INTEGER NOT NULL,

    -- Training config
    model_type TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',

    -- Results
    metrics JSONB DEFAULT '{}',
    model_path TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Notes
    notes TEXT
);

-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE learning_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE reasoning_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_outcomes ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limits ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_batches ENABLE ROW LEVEL SECURITY;

-- Service role has full access (for edge functions)
CREATE POLICY "Service role full access" ON learning_events
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON security_logs
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON reasoning_traces
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON prediction_outcomes
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON historical_cases
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON rate_limits
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON training_batches
    FOR ALL USING (auth.role() = 'service_role');

-- Admins can read security logs
CREATE POLICY "Admins can read security logs" ON security_logs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.role = 'admin'
        )
    );

-- Admins can read historical cases
CREATE POLICY "Admins can manage historical cases" ON historical_cases
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.role = 'admin'
        )
    );

-- ============================================================
-- FUNCTIONS FOR DATA MANAGEMENT
-- ============================================================

-- Function to clean up old learning events (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_learning_events()
RETURNS void AS $$
BEGIN
    -- Keep 90 days of data by default
    DELETE FROM learning_events
    WHERE timestamp < NOW() - INTERVAL '90 days'
    AND type NOT IN ('prediction_outcome'); -- Keep prediction outcomes forever

    -- Keep 1 year of security logs
    DELETE FROM security_logs
    WHERE timestamp < NOW() - INTERVAL '365 days';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to export training data
CREATE OR REPLACE FUNCTION export_training_batch(
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ,
    p_min_confidence FLOAT DEFAULT 0.6
)
RETURNS TABLE (
    trace_id UUID,
    input_features JSONB,
    conclusion TEXT,
    confidence FLOAT,
    domain TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        rt.id,
        rt.input_features,
        rt.conclusion,
        rt.confidence,
        rt.domain
    FROM reasoning_traces rt
    WHERE rt.timestamp BETWEEN p_start_date AND p_end_date
    AND rt.confidence >= p_min_confidence
    AND rt.exported = FALSE
    ORDER BY rt.timestamp;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to compute training metrics
CREATE OR REPLACE FUNCTION compute_training_metrics()
RETURNS TABLE (
    total_events BIGINT,
    events_by_type JSONB,
    prediction_accuracy FLOAT,
    avg_confidence FLOAT,
    data_freshness TIMESTAMPTZ
) AS $$
DECLARE
    v_type_counts JSONB;
    v_total BIGINT;
    v_accuracy FLOAT;
    v_confidence FLOAT;
BEGIN
    -- Count by type
    SELECT jsonb_object_agg(type, cnt)
    INTO v_type_counts
    FROM (
        SELECT type, COUNT(*) as cnt
        FROM learning_events
        GROUP BY type
    ) t;

    -- Total count
    SELECT COUNT(*) INTO v_total FROM learning_events;

    -- Prediction accuracy
    SELECT
        COALESCE(AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 0)
    INTO v_accuracy
    FROM prediction_outcomes;

    -- Average confidence
    SELECT COALESCE(AVG(confidence), 0)
    INTO v_confidence
    FROM reasoning_traces;

    RETURN QUERY SELECT
        v_total,
        v_type_counts,
        v_accuracy,
        v_confidence,
        NOW();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================
-- TRIGGERS
-- ============================================================

-- Auto-update updated_at for historical_cases
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER historical_cases_updated_at
    BEFORE UPDATE ON historical_cases
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- COMMENTS
-- ============================================================

COMMENT ON TABLE learning_events IS 'Main table for collecting all learning data. Anonymized and compliant.';
COMMENT ON TABLE security_logs IS 'Audit trail for security-relevant events.';
COMMENT ON TABLE reasoning_traces IS 'Detailed reasoning traces for model training.';
COMMENT ON TABLE prediction_outcomes IS 'Ground truth data when predictions can be verified.';
COMMENT ON TABLE historical_cases IS 'Historical cases for analogical reasoning.';
COMMENT ON TABLE rate_limits IS 'Rate limiting state per user/endpoint.';
COMMENT ON TABLE training_batches IS 'Tracks model training runs and results.';
