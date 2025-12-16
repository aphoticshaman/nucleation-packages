-- Doctrine Registry Tables
-- These tables support the Doctrine Registry feature for Stewardship tier customers
-- Enables transparent, auditable rule management for intelligence computation

-- ============================================
-- Doctrines Table
-- ============================================
CREATE TABLE IF NOT EXISTS doctrines (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN ('signal_interpretation', 'analytic_judgment', 'policy_logic', 'narrative')),
  description TEXT NOT NULL,
  rule_definition JSONB NOT NULL,
  rationale TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  effective_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deprecated_at TIMESTAMPTZ,
  created_by TEXT NOT NULL DEFAULT 'system',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Ensure unique name per version
  UNIQUE (name, version)
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_doctrines_category ON doctrines(category);
CREATE INDEX IF NOT EXISTS idx_doctrines_effective ON doctrines(effective_from);
CREATE INDEX IF NOT EXISTS idx_doctrines_deprecated ON doctrines(deprecated_at) WHERE deprecated_at IS NULL;

-- ============================================
-- Doctrine Proposals Table
-- ============================================
CREATE TABLE IF NOT EXISTS doctrine_proposals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doctrine_id UUID REFERENCES doctrines(id),
  proposed_changes JSONB NOT NULL,
  change_rationale TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending_review' CHECK (status IN ('pending_review', 'approved', 'rejected', 'implemented')),
  proposed_by TEXT NOT NULL,
  proposed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  reviewed_by TEXT,
  reviewed_at TIMESTAMPTZ,
  review_notes TEXT,
  implemented_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for finding pending proposals
CREATE INDEX IF NOT EXISTS idx_doctrine_proposals_status ON doctrine_proposals(status);
CREATE INDEX IF NOT EXISTS idx_doctrine_proposals_doctrine ON doctrine_proposals(doctrine_id);

-- ============================================
-- Doctrine Change Log Table
-- ============================================
CREATE TABLE IF NOT EXISTS doctrine_change_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doctrine_id UUID NOT NULL REFERENCES doctrines(id),
  version INTEGER NOT NULL,
  change_type TEXT NOT NULL CHECK (change_type IN ('created', 'updated', 'deprecated')),
  previous_definition JSONB,
  new_definition JSONB NOT NULL,
  change_rationale TEXT NOT NULL,
  changed_by TEXT NOT NULL,
  changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for audit queries
CREATE INDEX IF NOT EXISTS idx_doctrine_change_log_doctrine ON doctrine_change_log(doctrine_id);
CREATE INDEX IF NOT EXISTS idx_doctrine_change_log_changed ON doctrine_change_log(changed_at);

-- ============================================
-- Shadow Evaluations Table
-- ============================================
CREATE TABLE IF NOT EXISTS shadow_evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doctrine_id UUID NOT NULL REFERENCES doctrines(id),
  proposed_changes JSONB NOT NULL,
  evaluation_period_start TIMESTAMPTZ NOT NULL,
  evaluation_period_end TIMESTAMPTZ NOT NULL,
  total_events_evaluated INTEGER NOT NULL DEFAULT 0,
  divergence_count INTEGER NOT NULL DEFAULT 0,
  divergence_rate DECIMAL(5,4) NOT NULL DEFAULT 0,
  results_summary JSONB,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
  created_by TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

-- Index for finding recent evaluations
CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_doctrine ON shadow_evaluations(doctrine_id);
CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_status ON shadow_evaluations(status);

-- ============================================
-- Domain Cascade Matrix Table
-- For cascade analysis feature
-- ============================================
CREATE TABLE IF NOT EXISTS domain_cascade_matrix (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trigger_domain TEXT NOT NULL,
  effect_domain TEXT NOT NULL,
  co_occurrences INTEGER NOT NULL DEFAULT 0,
  avg_lag_hours DECIMAL(10,2),
  correlation_strength DECIMAL(5,4),
  last_observed TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE (trigger_domain, effect_domain)
);

-- Index for cascade queries
CREATE INDEX IF NOT EXISTS idx_cascade_matrix_trigger ON domain_cascade_matrix(trigger_domain);
CREATE INDEX IF NOT EXISTS idx_cascade_matrix_occurrences ON domain_cascade_matrix(co_occurrences DESC);

-- ============================================
-- Country Signals Table (for World Bank data)
-- ============================================
CREATE TABLE IF NOT EXISTS country_signals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  country_code TEXT NOT NULL,
  country_name TEXT NOT NULL,
  indicator TEXT NOT NULL,
  value DECIMAL(20,6) NOT NULL,
  year INTEGER NOT NULL,
  source TEXT NOT NULL DEFAULT 'worldbank',
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE (country_code, indicator, year)
);

-- Indexes for country signal queries
CREATE INDEX IF NOT EXISTS idx_country_signals_country ON country_signals(country_code);
CREATE INDEX IF NOT EXISTS idx_country_signals_indicator ON country_signals(indicator);
CREATE INDEX IF NOT EXISTS idx_country_signals_year ON country_signals(year DESC);

-- ============================================
-- RLS Policies
-- ============================================

-- Doctrines: Read for Integrated+, Write for admin
ALTER TABLE doctrines ENABLE ROW LEVEL SECURITY;

CREATE POLICY "doctrines_read_integrated" ON doctrines
  FOR SELECT
  USING (true); -- Tier check happens in API layer

CREATE POLICY "doctrines_write_admin" ON doctrines
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM profiles
      WHERE profiles.id = auth.uid()
      AND profiles.role = 'admin'
    )
  );

-- Doctrine Proposals: Read/Write for Stewardship
ALTER TABLE doctrine_proposals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "proposals_read_stewardship" ON doctrine_proposals
  FOR SELECT
  USING (true); -- Tier check happens in API layer

CREATE POLICY "proposals_write_stewardship" ON doctrine_proposals
  FOR INSERT
  WITH CHECK (true); -- Tier check happens in API layer

-- Shadow Evaluations: Read/Write for Stewardship
ALTER TABLE shadow_evaluations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "shadow_read_stewardship" ON shadow_evaluations
  FOR SELECT
  USING (true);

CREATE POLICY "shadow_write_stewardship" ON shadow_evaluations
  FOR INSERT
  WITH CHECK (true);

-- Change Log: Read only
ALTER TABLE doctrine_change_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "changelog_read_all" ON doctrine_change_log
  FOR SELECT
  USING (true);

-- Cascade Matrix: Read for Operational+
ALTER TABLE domain_cascade_matrix ENABLE ROW LEVEL SECURITY;

CREATE POLICY "cascade_read_operational" ON domain_cascade_matrix
  FOR SELECT
  USING (true);

-- Country Signals: Read for Operational+
ALTER TABLE country_signals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "country_signals_read_operational" ON country_signals
  FOR SELECT
  USING (true);

-- ============================================
-- Comments for documentation
-- ============================================
COMMENT ON TABLE doctrines IS 'Rule definitions governing intelligence computation';
COMMENT ON TABLE doctrine_proposals IS 'Proposed changes to doctrines pending review';
COMMENT ON TABLE doctrine_change_log IS 'Audit trail of all doctrine modifications';
COMMENT ON TABLE shadow_evaluations IS 'Results of shadow evaluation runs for proposed changes';
COMMENT ON TABLE domain_cascade_matrix IS 'Cross-domain event propagation patterns';
COMMENT ON TABLE country_signals IS 'Economic indicators from World Bank and other sources';
