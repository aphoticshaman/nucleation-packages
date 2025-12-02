-- Auto-generated training examples from LLM analysis of news
CREATE TABLE IF NOT EXISTS training_examples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  instruction TEXT NOT NULL,
  input TEXT NOT NULL,
  output TEXT NOT NULL,
  domain TEXT NOT NULL, -- geopolitical, economic, health, cyber, climate, etc.
  source_type TEXT NOT NULL, -- gdelt, rss, worldbank, usgs, etc.
  source_url TEXT,
  source_date TIMESTAMPTZ,
  goldstein_scale DECIMAL(4,2), -- if from GDELT
  confidence DECIMAL(3,2) DEFAULT 0.8, -- LLM confidence in quality
  exported BOOLEAN DEFAULT FALSE, -- whether included in training export
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for export queries
CREATE INDEX IF NOT EXISTS idx_training_examples_exported ON training_examples(exported);
CREATE INDEX IF NOT EXISTS idx_training_examples_domain ON training_examples(domain);
CREATE INDEX IF NOT EXISTS idx_training_examples_created ON training_examples(created_at DESC);

-- Avoid duplicates based on input text
CREATE UNIQUE INDEX IF NOT EXISTS idx_training_examples_input_hash
  ON training_examples(md5(input));

-- View for export
CREATE OR REPLACE VIEW exportable_training_data AS
SELECT
  instruction,
  input,
  output
FROM training_examples
WHERE exported = FALSE
  AND confidence >= 0.7
ORDER BY created_at DESC;

-- Stats view
CREATE OR REPLACE VIEW training_data_stats AS
SELECT
  domain,
  source_type,
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE exported = FALSE) as pending_export,
  AVG(confidence) as avg_confidence,
  MAX(created_at) as latest
FROM training_examples
GROUP BY domain, source_type;
