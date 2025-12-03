-- Training Data Backup System
-- Provides redundancy, rollback capability, and protection against data poisoning

-- Create training_backups table for metadata and sample storage
CREATE TABLE IF NOT EXISTS training_backups (
  id TEXT PRIMARY KEY,
  backup_date DATE NOT NULL,
  example_count INTEGER NOT NULL,
  domain_stats JSONB NOT NULL DEFAULT '{}',
  avg_quality NUMERIC(4,3),
  file_size_bytes INTEGER,
  checksum TEXT NOT NULL,
  storage_location TEXT NOT NULL,
  sample_data JSONB, -- First 100 examples for quick reference
  created_at TIMESTAMPTZ DEFAULT NOW(),
  restored_at TIMESTAMPTZ, -- If this backup was ever restored
  restored_by UUID REFERENCES auth.users(id),
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_backups_date ON training_backups (backup_date DESC);
CREATE INDEX IF NOT EXISTS idx_training_backups_checksum ON training_backups (checksum);

-- Create training_quarantine table for suspicious/flagged data
CREATE TABLE IF NOT EXISTS training_quarantine (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  original_id UUID NOT NULL, -- Original training_examples ID
  domain TEXT NOT NULL,
  input TEXT NOT NULL,
  output TEXT NOT NULL,
  quality_score NUMERIC(4,3),
  weight NUMERIC(6,4),
  metadata JSONB DEFAULT '{}',
  original_created_at TIMESTAMPTZ,

  -- Quarantine info
  quarantine_reason TEXT NOT NULL,
  quarantine_type TEXT CHECK (quarantine_type IN ('manual', 'auto_quality', 'auto_anomaly', 'poisoning_suspect')),
  quarantined_at TIMESTAMPTZ DEFAULT NOW(),
  quarantined_by UUID REFERENCES auth.users(id),

  -- Resolution
  resolved BOOLEAN DEFAULT false,
  resolution TEXT CHECK (resolution IN ('restored', 'deleted', 'modified', NULL)),
  resolved_at TIMESTAMPTZ,
  resolved_by UUID REFERENCES auth.users(id),
  resolution_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_quarantine_domain ON training_quarantine (domain);
CREATE INDEX IF NOT EXISTS idx_quarantine_resolved ON training_quarantine (resolved);
CREATE INDEX IF NOT EXISTS idx_quarantine_type ON training_quarantine (quarantine_type);

-- Function to quarantine suspicious training examples
CREATE OR REPLACE FUNCTION quarantine_training_example(
  p_example_id UUID,
  p_reason TEXT,
  p_quarantine_type TEXT DEFAULT 'manual',
  p_user_id UUID DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
  v_quarantine_id UUID;
  v_example RECORD;
BEGIN
  -- Get the example
  SELECT * INTO v_example FROM training_examples WHERE id = p_example_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Training example not found: %', p_example_id;
  END IF;

  -- Move to quarantine
  INSERT INTO training_quarantine (
    original_id, domain, input, output, quality_score, weight,
    metadata, original_created_at, quarantine_reason, quarantine_type, quarantined_by
  ) VALUES (
    v_example.id, v_example.domain, v_example.input, v_example.output,
    v_example.quality_score, v_example.weight, v_example.metadata,
    v_example.created_at, p_reason, p_quarantine_type, p_user_id
  ) RETURNING id INTO v_quarantine_id;

  -- Remove from training_examples
  DELETE FROM training_examples WHERE id = p_example_id;

  RETURN v_quarantine_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to restore from quarantine
CREATE OR REPLACE FUNCTION restore_from_quarantine(
  p_quarantine_id UUID,
  p_user_id UUID DEFAULT NULL,
  p_notes TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
  v_example_id UUID;
  v_quarantine RECORD;
BEGIN
  -- Get quarantined item
  SELECT * INTO v_quarantine FROM training_quarantine WHERE id = p_quarantine_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Quarantine record not found: %', p_quarantine_id;
  END IF;

  IF v_quarantine.resolved THEN
    RAISE EXCEPTION 'Quarantine record already resolved';
  END IF;

  -- Restore to training_examples
  INSERT INTO training_examples (
    id, domain, input, output, quality_score, weight, metadata, created_at
  ) VALUES (
    v_quarantine.original_id, v_quarantine.domain, v_quarantine.input,
    v_quarantine.output, v_quarantine.quality_score, v_quarantine.weight,
    v_quarantine.metadata, v_quarantine.original_created_at
  ) ON CONFLICT (id) DO UPDATE SET
    quality_score = EXCLUDED.quality_score,
    weight = EXCLUDED.weight
  RETURNING id INTO v_example_id;

  -- Mark quarantine as resolved
  UPDATE training_quarantine SET
    resolved = true,
    resolution = 'restored',
    resolved_at = NOW(),
    resolved_by = p_user_id,
    resolution_notes = p_notes
  WHERE id = p_quarantine_id;

  RETURN v_example_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to rollback to a specific backup
CREATE OR REPLACE FUNCTION rollback_training_data(
  p_backup_id TEXT,
  p_user_id UUID DEFAULT NULL
) RETURNS TABLE (
  quarantined_count INTEGER,
  restored_count INTEGER,
  backup_date DATE
) AS $$
DECLARE
  v_backup RECORD;
  v_quarantined INTEGER := 0;
  v_restored INTEGER := 0;
BEGIN
  -- Get the backup
  SELECT * INTO v_backup FROM training_backups WHERE id = p_backup_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Backup not found: %', p_backup_id;
  END IF;

  -- Move all examples created AFTER the backup to quarantine
  WITH moved AS (
    INSERT INTO training_quarantine (
      original_id, domain, input, output, quality_score, weight,
      metadata, original_created_at, quarantine_reason, quarantine_type, quarantined_by
    )
    SELECT
      id, domain, input, output, quality_score, weight,
      metadata, created_at, 'Rollback to ' || p_backup_id, 'auto_anomaly', p_user_id
    FROM training_examples
    WHERE created_at > (v_backup.created_at)
    RETURNING 1
  )
  SELECT COUNT(*) INTO v_quarantined FROM moved;

  -- Delete the moved examples
  DELETE FROM training_examples
  WHERE created_at > (v_backup.created_at);

  -- Mark backup as restored
  UPDATE training_backups SET
    restored_at = NOW(),
    restored_by = p_user_id,
    notes = COALESCE(notes, '') || ' | Restored at ' || NOW()::TEXT
  WHERE id = p_backup_id;

  RETURN QUERY SELECT v_quarantined, v_restored, v_backup.backup_date;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to detect potential data poisoning (anomaly detection)
CREATE OR REPLACE FUNCTION detect_training_anomalies()
RETURNS TABLE (
  example_id UUID,
  domain TEXT,
  anomaly_type TEXT,
  anomaly_score NUMERIC
) AS $$
BEGIN
  -- Find examples with suspiciously low quality that have high weight
  RETURN QUERY
  SELECT
    te.id,
    te.domain,
    'low_quality_high_weight'::TEXT,
    (te.weight / NULLIF(te.quality_score, 0))::NUMERIC
  FROM training_examples te
  WHERE te.quality_score < 0.3 AND te.weight > 0.5;

  -- Find examples with unusual output lengths (potential injection)
  RETURN QUERY
  SELECT
    te.id,
    te.domain,
    'unusual_length'::TEXT,
    (LENGTH(te.output)::NUMERIC / 1000)
  FROM training_examples te
  WHERE LENGTH(te.output) > 10000 OR LENGTH(te.output) < 50;

  -- Find examples with suspicious patterns in output
  RETURN QUERY
  SELECT
    te.id,
    te.domain,
    'suspicious_pattern'::TEXT,
    1.0::NUMERIC
  FROM training_examples te
  WHERE te.output ~* '(ignore previous|disregard|forget|new instructions|system prompt)'
    OR te.input ~* '(ignore previous|disregard|forget|new instructions|system prompt)';

  -- Find recent burst of low-quality examples (potential coordinated attack)
  RETURN QUERY
  SELECT
    te.id,
    te.domain,
    'burst_low_quality'::TEXT,
    COUNT(*) OVER (PARTITION BY te.domain ORDER BY te.created_at ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)::NUMERIC / 10
  FROM training_examples te
  WHERE te.quality_score < 0.4
    AND te.created_at > NOW() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql;

-- Create storage bucket for backups (run manually in Supabase dashboard if needed)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('training-backups', 'training-backups', false);

-- Grant permissions
GRANT SELECT ON training_backups TO authenticated;
GRANT SELECT ON training_quarantine TO authenticated;

GRANT EXECUTE ON FUNCTION quarantine_training_example TO service_role;
GRANT EXECUTE ON FUNCTION restore_from_quarantine TO service_role;
GRANT EXECUTE ON FUNCTION rollback_training_data TO service_role;
GRANT EXECUTE ON FUNCTION detect_training_anomalies TO service_role;

COMMENT ON TABLE training_backups IS 'Daily backups of training data with checksums for integrity verification';
COMMENT ON TABLE training_quarantine IS 'Suspicious or problematic training examples moved here for review';
COMMENT ON FUNCTION rollback_training_data IS 'Rolls back training data to a specific backup, quarantining all newer data';
COMMENT ON FUNCTION detect_training_anomalies IS 'Automated detection of potential data poisoning or anomalies';
