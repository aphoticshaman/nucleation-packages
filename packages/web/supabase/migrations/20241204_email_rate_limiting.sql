-- Email export rate limiting and audit log
-- Tracks email exports per user for tier-based rate limiting

CREATE TABLE IF NOT EXISTS email_export_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  recipient_email TEXT NOT NULL,
  package_type TEXT NOT NULL,
  sections_count INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for efficient rate limit queries (user + date)
CREATE INDEX IF NOT EXISTS idx_email_export_log_user_date
  ON email_export_log(user_id, created_at DESC);

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_email_export_log_created_at
  ON email_export_log(created_at DESC);

-- RLS policies
ALTER TABLE email_export_log ENABLE ROW LEVEL SECURITY;

-- Users can only see their own export logs
CREATE POLICY "Users can view own email exports"
  ON email_export_log FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

-- Users can insert their own export logs
CREATE POLICY "Users can create own email exports"
  ON email_export_log FOR INSERT
  TO authenticated
  WITH CHECK (user_id = auth.uid());

-- Service role can do everything
CREATE POLICY "Service role full access to email exports"
  ON email_export_log FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Clean up old logs (keep 30 days) - run via cron
CREATE OR REPLACE FUNCTION cleanup_old_email_logs()
RETURNS void AS $$
BEGIN
  DELETE FROM email_export_log
  WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Alert rate limiting table
CREATE TABLE IF NOT EXISTS alert_send_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  alert_type TEXT NOT NULL,
  destination TEXT NOT NULL, -- 'email', 'sms', 'webhook'
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for efficient rate limit queries
CREATE INDEX IF NOT EXISTS idx_alert_send_log_user_date
  ON alert_send_log(user_id, created_at DESC);

-- RLS policies
ALTER TABLE alert_send_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own alert logs"
  ON alert_send_log FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

CREATE POLICY "Users can create own alert logs"
  ON alert_send_log FOR INSERT
  TO authenticated
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Service role full access to alert logs"
  ON alert_send_log FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

COMMENT ON TABLE email_export_log IS 'Rate limiting log for email exports. Tier limits: free=3/day, starter=10/day, pro=50/day, enterprise=500/day';
COMMENT ON TABLE alert_send_log IS 'Rate limiting log for alert notifications. Tier limits: free=5/day, starter=20/day, pro=100/day, enterprise=1000/day';
