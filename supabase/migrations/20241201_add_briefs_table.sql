-- Briefs table for storing AI-generated executive summaries
CREATE TABLE briefs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  content TEXT NOT NULL,
  summary TEXT NOT NULL,
  signals_snapshot JSONB NOT NULL,
  model TEXT NOT NULL DEFAULT 'claude-3-haiku-20240307',
  tokens_used INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast retrieval of latest briefs
CREATE INDEX idx_briefs_created ON briefs(created_at DESC);

-- Enable RLS
ALTER TABLE briefs ENABLE ROW LEVEL SECURITY;

-- Policy: authenticated users can read briefs
CREATE POLICY "Authenticated users can view briefs"
  ON briefs FOR SELECT
  USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- Policy: only service role can insert (Edge Function)
CREATE POLICY "Service role can insert briefs"
  ON briefs FOR INSERT
  WITH CHECK (auth.role() = 'service_role');

-- Optional: auto-cleanup old briefs (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_briefs()
RETURNS void AS $$
BEGIN
  DELETE FROM briefs WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public;
