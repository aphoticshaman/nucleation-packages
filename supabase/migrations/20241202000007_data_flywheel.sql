-- Data Flywheel: Active Learning System
-- Training examples that lead to accurate predictions get upweighted

-- Add flywheel columns to training_examples
ALTER TABLE training_examples
ADD COLUMN IF NOT EXISTS selection_weight FLOAT DEFAULT 1.0,
ADD COLUMN IF NOT EXISTS times_used_in_training INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS prediction_accuracy_sum FLOAT DEFAULT 0,
ADD COLUMN IF NOT EXISTS prediction_count INTEGER DEFAULT 0;

-- Predictions table: track what we predicted and what actually happened
CREATE TABLE IF NOT EXISTS predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  -- What we predicted
  domain TEXT NOT NULL,
  prediction_type TEXT NOT NULL,  -- 'risk_increase', 'cascade', 'event', etc.
  prediction_content JSONB NOT NULL,
  confidence FLOAT NOT NULL,
  predicted_timeframe_hours INTEGER,  -- How far out we predicted

  -- Source training examples that influenced this prediction
  source_example_ids UUID[] DEFAULT '{}',

  -- What actually happened
  outcome_observed BOOLEAN DEFAULT FALSE,
  outcome_accuracy FLOAT,  -- 0-1 how accurate was the prediction
  outcome_notes TEXT,
  outcome_recorded_at TIMESTAMPTZ,

  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ  -- When this prediction is no longer relevant
);

-- Index for finding predictions to score
CREATE INDEX IF NOT EXISTS idx_predictions_pending
ON predictions(outcome_observed, expires_at)
WHERE outcome_observed = FALSE;

CREATE INDEX IF NOT EXISTS idx_predictions_domain
ON predictions(domain, created_at DESC);

-- Function to record a prediction
CREATE OR REPLACE FUNCTION record_prediction(
  p_domain TEXT,
  p_type TEXT,
  p_content JSONB,
  p_confidence FLOAT,
  p_timeframe_hours INTEGER,
  p_source_examples UUID[]
) RETURNS UUID AS $$
DECLARE
  new_id UUID;
BEGIN
  INSERT INTO predictions (
    domain, prediction_type, prediction_content, confidence,
    predicted_timeframe_hours, source_example_ids, expires_at
  ) VALUES (
    p_domain, p_type, p_content, p_confidence,
    p_timeframe_hours, p_source_examples,
    NOW() + (p_timeframe_hours || ' hours')::interval
  )
  RETURNING id INTO new_id;

  RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Function to score a prediction and update training example weights
CREATE OR REPLACE FUNCTION score_prediction(
  p_prediction_id UUID,
  p_accuracy FLOAT,  -- 0-1
  p_notes TEXT DEFAULT NULL
) RETURNS void AS $$
DECLARE
  source_ids UUID[];
  example_id UUID;
  weight_delta FLOAT;
BEGIN
  -- Get source examples
  SELECT source_example_ids INTO source_ids
  FROM predictions
  WHERE id = p_prediction_id;

  -- Calculate weight adjustment
  -- Accuracy > 0.5 increases weight, < 0.5 decreases
  weight_delta := (p_accuracy - 0.5) * 0.2;  -- Max ±10% per prediction

  -- Update each source example
  FOREACH example_id IN ARRAY source_ids
  LOOP
    UPDATE training_examples
    SET
      selection_weight = GREATEST(0.1, LEAST(10.0, selection_weight + weight_delta)),
      prediction_accuracy_sum = prediction_accuracy_sum + p_accuracy,
      prediction_count = prediction_count + 1
    WHERE id = example_id;
  END LOOP;

  -- Mark prediction as scored
  UPDATE predictions
  SET
    outcome_observed = TRUE,
    outcome_accuracy = p_accuracy,
    outcome_notes = p_notes,
    outcome_recorded_at = NOW()
  WHERE id = p_prediction_id;
END;
$$ LANGUAGE plpgsql;

-- View for weighted training data export
CREATE OR REPLACE VIEW weighted_training_data AS
SELECT
  id,
  instruction,
  input,
  output,
  domain,
  selection_weight,
  CASE
    WHEN prediction_count > 0
    THEN prediction_accuracy_sum / prediction_count
    ELSE NULL
  END as avg_accuracy,
  prediction_count,
  -- Sampling probability proportional to weight
  selection_weight / SUM(selection_weight) OVER () as sample_probability
FROM training_examples
WHERE exported = FALSE
  AND confidence >= 0.7
ORDER BY selection_weight DESC;

-- View for flywheel stats
CREATE OR REPLACE VIEW flywheel_stats AS
SELECT
  domain,
  COUNT(*) as total_examples,
  AVG(selection_weight) as avg_weight,
  MAX(selection_weight) as max_weight,
  MIN(selection_weight) as min_weight,
  SUM(prediction_count) as total_predictions,
  AVG(CASE WHEN prediction_count > 0 THEN prediction_accuracy_sum / prediction_count END) as avg_accuracy
FROM training_examples
GROUP BY domain;

-- Function to get weighted sample for training
CREATE OR REPLACE FUNCTION get_weighted_training_sample(sample_size INTEGER)
RETURNS TABLE (
  instruction TEXT,
  input TEXT,
  output TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    t.instruction,
    t.input,
    t.output
  FROM training_examples t
  WHERE t.exported = FALSE
    AND t.confidence >= 0.7
  ORDER BY RANDOM() * t.selection_weight DESC
  LIMIT sample_size;
END;
$$ LANGUAGE plpgsql;

-- Auto-expire old predictions that were never scored
CREATE OR REPLACE FUNCTION cleanup_expired_predictions()
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  DELETE FROM predictions
  WHERE outcome_observed = FALSE
    AND expires_at < NOW() - interval '7 days'
  RETURNING COUNT(*) INTO deleted_count;

  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
