-- Briefs table for storing AI-generated executive summaries
CREATE TABLE IF NOT EXISTS briefs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  content TEXT NOT NULL,
  summary TEXT NOT NULL,
  signals_snapshot JSONB NOT NULL,
  model TEXT NOT NULL DEFAULT 'claude-3-haiku-20240307',
  tokens_used INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast retrieval of latest briefs
CREATE INDEX IF NOT EXISTS idx_briefs_created ON briefs(created_at DESC);

-- Enable RLS
ALTER TABLE briefs ENABLE ROW LEVEL SECURITY;

-- Policy: authenticated users can read briefs
DO $$ BEGIN
  CREATE POLICY "Authenticated users can view briefs"
    ON briefs FOR SELECT
    USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Policy: only service role can insert (Edge Function)
DO $$ BEGIN
  CREATE POLICY "Service role can insert briefs"
    ON briefs FOR INSERT
    WITH CHECK (auth.role() = 'service_role');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Optional: auto-cleanup old briefs (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_briefs()
RETURNS void AS $$
BEGIN
  DELETE FROM briefs WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public;
