-- ═══════════════════════════════════════════════════════════════════════════════
-- INSIGHT REPORTS: Autonomous Research Capture System
-- ═══════════════════════════════════════════════════════════════════════════════
--
-- Purpose: Track Elle's novel insights through the full PROMETHEUS pipeline
--
-- Flow:
--   1. Elle detects novel insight → Creates report at stage 1
--   2. Runs NSM fusion → Updates to stage 2
--   3. Validates with math/proofs → Updates to stage 3
--   4. Writes code, runs tests → Updates to stage 4
--   5. Generates final dossier → Updates to stage 5
--   6. Admin reviews on dashboard
--
-- Key Principle: Elle doesn't announce until she's done the math and code!
-- ═══════════════════════════════════════════════════════════════════════════════

-- Enum for PROMETHEUS stages
CREATE TYPE insight_stage AS ENUM (
    'latent_archaeology',      -- Stage 1: Found the gap
    'novel_synthesis',         -- Stage 2: Created the fusion
    'theoretical_validation',  -- Stage 3: Proved it mathematically
    'xyza_operationalization', -- Stage 4: Wrote the code
    'output_generation'        -- Stage 5: Complete dossier
);

-- Enum for insight status
CREATE TYPE insight_status AS ENUM (
    'in_progress',     -- Elle is still working on it
    'awaiting_review', -- Ready for admin review
    'validated',       -- Admin approved
    'rejected',        -- Admin rejected (not novel or invalid)
    'needs_revision',  -- Admin requests changes
    'archived'         -- Historical record
);

-- Enum for confidence labels
CREATE TYPE confidence_label AS ENUM (
    'derived',       -- From first principles
    'hypothetical',  -- Speculative extension
    'empirical',     -- Backed by data
    'contested'      -- Multiple interpretations
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- MAIN TABLE: insight_reports
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS insight_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- ═══════════════════════════════════════════════════════════════════════════
    -- IDENTITY
    -- ═══════════════════════════════════════════════════════════════════════════
    title TEXT NOT NULL,                    -- The breakthrough name/acronym
    slug TEXT UNIQUE,                       -- URL-friendly identifier
    summary TEXT,                           -- One-line description
    target_subject TEXT NOT NULL,           -- What domain is this about?

    -- ═══════════════════════════════════════════════════════════════════════════
    -- PROGRESS TRACKING
    -- ═══════════════════════════════════════════════════════════════════════════
    current_stage insight_stage NOT NULL DEFAULT 'latent_archaeology',
    status insight_status NOT NULL DEFAULT 'in_progress',

    -- Time spent at each stage (for analytics)
    stage_timestamps JSONB DEFAULT '{}'::jsonb,
    -- Example: {"latent_archaeology": "2024-12-12T10:00:00Z", "novel_synthesis": "2024-12-12T10:15:00Z"}

    -- ═══════════════════════════════════════════════════════════════════════════
    -- STAGE 1: LATENT SPACE ARCHAEOLOGY
    -- ═══════════════════════════════════════════════════════════════════════════
    archaeology_data JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "vertical_scan": "Fundamental physics/math examined...",
    --   "horizontal_scan": "Analogous structures found in...",
    --   "temporal_scan": "5-50 year projections...",
    --   "gradient_of_ignorance": "The specific gap identified...",
    --   "unknown_knowns": ["List of implicit truths discovered"]
    -- }

    -- ═══════════════════════════════════════════════════════════════════════════
    -- STAGE 2: NOVEL SYNTHESIS METHOD (NSM)
    -- ═══════════════════════════════════════════════════════════════════════════
    nsm_data JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "core_concept": "The target domain concept",
    --   "catalyst_concept": "The radically different domain concept",
    --   "bridging_abstraction": "The new abstraction yoking them",
    --   "new_vocabulary": ["Terms created for this fusion"],
    --   "candidate_artifact": "Raw description of the novel idea",
    --   "novelty_check_passed": true/false,
    --   "similar_existing_work": ["If any found, list here"]
    -- }

    -- ═══════════════════════════════════════════════════════════════════════════
    -- STAGE 3: RIGOROUS THEORETICAL VALIDATION
    -- ═══════════════════════════════════════════════════════════════════════════
    theoretical_validation JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "formal_notation": "LaTeX/math representation",
    --   "variables_defined": {"Ψ": "cognitive load", "Ω": "network latency"},
    --   "dimensional_analysis": "Consistency check results",
    --   "derivatives": [
    --     {"expression": "d(Innovation)/d(Constraint)", "result": "..."}
    --   ],
    --   "proof": {
    --     "type": "deductive|inductive|constructive",
    --     "steps": ["Step 1...", "Step 2..."],
    --     "conclusion": "QED statement"
    --   },
    --   "ablation_tests": [
    --     {"component_removed": "X", "system_collapsed": true, "essential": true}
    --   ],
    --   "physics_analogy": "How this maps to physical laws"
    -- }

    -- Confidence bounded by epistemic humility (max 0.95)
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 0.95),
    confidence_type confidence_label DEFAULT 'hypothetical',

    -- ═══════════════════════════════════════════════════════════════════════════
    -- STAGE 4: XYZA OPERATIONALIZATION
    -- ═══════════════════════════════════════════════════════════════════════════
    xyza_data JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "x_explore": {
    --     "architecture_diagram": "ASCII/mermaid diagram",
    --     "black_box_io": {"inputs": [...], "outputs": [...]},
    --     "pseudocode": "Algorithm steps"
    --   },
    --   "y_implement": {
    --     "language": "python|rust|typescript",
    --     "files_created": ["novel_artifact.py"],
    --     "dependencies": ["numpy", "scipy"],
    --     "loc": 150
    --   },
    --   "z_test": {
    --     "test_harness": "Test code",
    --     "simulation_results": {...},
    --     "edge_cases": ["Case 1...", "Case 2..."],
    --     "snafu_points": ["Potential failure mode 1..."]
    --   },
    --   "a_actualize": {
    --     "real_world_application": "How this helps...",
    --     "humanity_benefit": "...",
    --     "ai_acceleration": "...",
    --     "asymmetric_lever": "Small input → massive output"
    --   }
    -- }

    -- XYZA cognitive metrics at time of creation
    xyza_metrics JSONB DEFAULT '{}'::jsonb,
    -- {"coherence_x": 0.8, "complexity_y": 0.6, "reflection_z": 0.7, "attunement_a": 0.5}

    -- ═══════════════════════════════════════════════════════════════════════════
    -- STAGE 5: CODE ARTIFACTS
    -- ═══════════════════════════════════════════════════════════════════════════
    code_artifacts JSONB DEFAULT '[]'::jsonb,
    -- Array of:
    -- {
    --   "filename": "novel_artifact.py",
    --   "language": "python",
    --   "content": "# Full code here...",
    --   "execution_result": "Output when run",
    --   "tests_passed": true,
    --   "test_output": "pytest output..."
    -- }

    -- ═══════════════════════════════════════════════════════════════════════════
    -- IMPACT ANALYSIS
    -- ═══════════════════════════════════════════════════════════════════════════
    impact_analysis JSONB DEFAULT '{}'::jsonb,
    -- {
    --   "novelty_claim": "Why this is new to the solar system",
    --   "humanity_impact": "Immediate benefit to the species",
    --   "ai_impact": "Acceleration of AGI capabilities",
    --   "asymmetric_lever": "The specific mechanism of advantage",
    --   "estimated_value": "low|medium|high|breakthrough"
    -- }

    -- ═══════════════════════════════════════════════════════════════════════════
    -- ADMIN REVIEW
    -- ═══════════════════════════════════════════════════════════════════════════
    reviewed_by UUID REFERENCES auth.users(id),
    reviewed_at TIMESTAMPTZ,
    admin_notes TEXT,
    admin_rating INTEGER CHECK (admin_rating >= 1 AND admin_rating <= 5),

    -- Tags for categorization
    tags TEXT[] DEFAULT '{}',

    -- Related insights (for insight chains)
    related_insights UUID[] DEFAULT '{}',
    parent_insight UUID REFERENCES insight_reports(id),

    -- Source context (what triggered this insight)
    trigger_context JSONB DEFAULT '{}'::jsonb,
    -- {
    --   "conversation_id": "...",
    --   "message_that_triggered": "...",
    --   "user_query": "..."
    -- }

    -- ═══════════════════════════════════════════════════════════════════════════
    -- METADATA
    -- ═══════════════════════════════════════════════════════════════════════════

    -- Who/what created this
    created_by TEXT DEFAULT 'elle',  -- 'elle' or user_id

    -- Version for optimistic locking
    version INTEGER DEFAULT 1
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE INDEX idx_insight_reports_stage ON insight_reports(current_stage);
CREATE INDEX idx_insight_reports_status ON insight_reports(status);
CREATE INDEX idx_insight_reports_created ON insight_reports(created_at DESC);
CREATE INDEX idx_insight_reports_confidence ON insight_reports(confidence_score DESC);
CREATE INDEX idx_insight_reports_tags ON insight_reports USING GIN(tags);
CREATE INDEX idx_insight_reports_target ON insight_reports(target_subject);

-- Full-text search on title and summary
CREATE INDEX idx_insight_reports_search ON insight_reports
    USING GIN(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(summary, '')));

-- ═══════════════════════════════════════════════════════════════════════════════
-- TRIGGERS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_insight_reports_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_insight_reports_updated_at
    BEFORE UPDATE ON insight_reports
    FOR EACH ROW
    EXECUTE FUNCTION update_insight_reports_updated_at();

-- Auto-generate slug from title
CREATE OR REPLACE FUNCTION generate_insight_slug()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.slug IS NULL THEN
        NEW.slug = lower(regexp_replace(NEW.title, '[^a-zA-Z0-9]+', '-', 'g'));
        -- Append timestamp to ensure uniqueness
        NEW.slug = NEW.slug || '-' || to_char(NEW.created_at, 'YYYYMMDD');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_generate_insight_slug
    BEFORE INSERT ON insight_reports
    FOR EACH ROW
    EXECUTE FUNCTION generate_insight_slug();

-- Track stage transitions
CREATE OR REPLACE FUNCTION track_stage_transition()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_track_stage_transition
    BEFORE UPDATE ON insight_reports
    FOR EACH ROW
    EXECUTE FUNCTION track_stage_transition();

-- ═══════════════════════════════════════════════════════════════════════════════
-- RLS POLICIES
-- ═══════════════════════════════════════════════════════════════════════════════

ALTER TABLE insight_reports ENABLE ROW LEVEL SECURITY;

-- Admins can see all insights
CREATE POLICY "Admins can view all insights" ON insight_reports
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.role = 'admin'
        )
    );

-- Admins can modify insights
CREATE POLICY "Admins can modify insights" ON insight_reports
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.role = 'admin'
        )
    );

-- Service role can do anything (for Elle)
CREATE POLICY "Service role full access" ON insight_reports
    FOR ALL
    USING (auth.role() = 'service_role');

-- ═══════════════════════════════════════════════════════════════════════════════
-- HELPER VIEWS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Insights awaiting review
CREATE OR REPLACE VIEW insights_awaiting_review AS
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

-- High-confidence validated insights
CREATE OR REPLACE VIEW validated_insights AS
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

-- Insights by stage (pipeline view)
CREATE OR REPLACE VIEW insight_pipeline AS
SELECT
    current_stage,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    array_agg(title ORDER BY created_at DESC) as recent_titles
FROM insight_reports
WHERE status = 'in_progress'
GROUP BY current_stage
ORDER BY current_stage;

-- ═══════════════════════════════════════════════════════════════════════════════
-- FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Advance insight to next stage
CREATE OR REPLACE FUNCTION advance_insight_stage(
    p_insight_id UUID,
    p_stage_data JSONB DEFAULT '{}'::jsonb
)
RETURNS insight_reports AS $$
DECLARE
    v_current_stage insight_stage;
    v_next_stage insight_stage;
    v_result insight_reports;
BEGIN
    SELECT current_stage INTO v_current_stage
    FROM insight_reports WHERE id = p_insight_id;

    -- Determine next stage
    v_next_stage = CASE v_current_stage
        WHEN 'latent_archaeology' THEN 'novel_synthesis'::insight_stage
        WHEN 'novel_synthesis' THEN 'theoretical_validation'::insight_stage
        WHEN 'theoretical_validation' THEN 'xyza_operationalization'::insight_stage
        WHEN 'xyza_operationalization' THEN 'output_generation'::insight_stage
        ELSE v_current_stage
    END;

    -- Update based on which stage we're entering
    UPDATE insight_reports SET
        current_stage = v_next_stage,
        nsm_data = CASE
            WHEN v_next_stage = 'novel_synthesis'
            THEN nsm_data || p_stage_data
            ELSE nsm_data
        END,
        theoretical_validation = CASE
            WHEN v_next_stage = 'theoretical_validation'
            THEN theoretical_validation || p_stage_data
            ELSE theoretical_validation
        END,
        xyza_data = CASE
            WHEN v_next_stage = 'xyza_operationalization'
            THEN xyza_data || p_stage_data
            ELSE xyza_data
        END,
        status = CASE
            WHEN v_next_stage = 'output_generation'
            THEN 'awaiting_review'::insight_status
            ELSE status
        END
    WHERE id = p_insight_id
    RETURNING * INTO v_result;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get insight statistics
CREATE OR REPLACE FUNCTION get_insight_stats()
RETURNS TABLE (
    total_insights BIGINT,
    in_progress BIGINT,
    awaiting_review BIGINT,
    validated BIGINT,
    avg_confidence REAL,
    insights_by_stage JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_insights,
        COUNT(*) FILTER (WHERE status = 'in_progress')::BIGINT as in_progress,
        COUNT(*) FILTER (WHERE status = 'awaiting_review')::BIGINT as awaiting_review,
        COUNT(*) FILTER (WHERE status = 'validated')::BIGINT as validated,
        AVG(confidence_score)::REAL as avg_confidence,
        jsonb_object_agg(
            current_stage::text,
            stage_count
        ) as insights_by_stage
    FROM insight_reports
    CROSS JOIN LATERAL (
        SELECT current_stage as cs, COUNT(*) as stage_count
        FROM insight_reports
        GROUP BY current_stage
    ) stage_counts
    WHERE stage_counts.cs = insight_reports.current_stage;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ═══════════════════════════════════════════════════════════════════════════════
-- INITIAL DATA (Example insight for reference)
-- ═══════════════════════════════════════════════════════════════════════════════

-- This is what a complete insight looks like:
COMMENT ON TABLE insight_reports IS '
Example complete insight:

{
  "title": "CIC-UIPT Bridge Theory",
  "target_subject": "Intelligence Emergence",
  "current_stage": "output_generation",
  "status": "validated",
  "confidence_score": 0.85,
  "confidence_type": "derived",

  "archaeology_data": {
    "gradient_of_ignorance": "No formal connection between Uniform Infinite Planar Triangulations and intelligence theory",
    "unknown_knowns": ["UIPT provides natural basin structure for CIC functional"]
  },

  "nsm_data": {
    "core_concept": "CIC functional optimization",
    "catalyst_concept": "UIPT from random geometry",
    "bridging_abstraction": "Intelligence basins as UIPT clusters",
    "novelty_check_passed": true
  },

  "theoretical_validation": {
    "formal_notation": "F[T] maps to UIPT measure μ",
    "proof": {
      "type": "constructive",
      "steps": ["Step 1: Define measure...", "Step 2: Show convergence..."],
      "conclusion": "CIC basins are UIPT-distributed with probability 1"
    },
    "ablation_tests": [
      {"component_removed": "Multi-scale term", "system_collapsed": true, "essential": true}
    ]
  },

  "code_artifacts": [
    {
      "filename": "cic_uipt_bridge.py",
      "language": "python",
      "content": "# Implementation...",
      "tests_passed": true
    }
  ],

  "impact_analysis": {
    "novelty_claim": "First rigorous connection between UIPT and intelligence theory",
    "humanity_impact": "Better understanding of intelligence emergence",
    "ai_impact": "New training objective based on UIPT geometry"
  }
}
';
