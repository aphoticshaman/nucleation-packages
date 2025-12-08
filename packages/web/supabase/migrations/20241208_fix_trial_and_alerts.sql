-- ============================================================================
-- Fix Trial System & Add Missing Alert Columns
-- Run this in Supabase SQL Editor
-- ============================================================================

-- 1. Update trial duration from 7 to 14 days
-- Change the constant in the get_trial_end_date function
CREATE OR REPLACE FUNCTION get_trial_end_date()
RETURNS TIMESTAMPTZ AS $$
BEGIN
  -- Changed from 7 to 14 days for trial period
  RETURN NOW() + INTERVAL '14 days';
END;
$$ LANGUAGE plpgsql;

-- 2. Add missing columns to profiles for email alert tracking
-- These are needed by /api/alerts/send-email
DO $$
BEGIN
  -- Add tier column if not exists (for rate limiting)
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'profiles' AND column_name = 'tier'
  ) THEN
    ALTER TABLE profiles ADD COLUMN tier VARCHAR(50) DEFAULT 'analyst';
  END IF;

  -- Add email alerts tracking columns
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'profiles' AND column_name = 'email_alerts_sent_today'
  ) THEN
    ALTER TABLE profiles ADD COLUMN email_alerts_sent_today INTEGER DEFAULT 0;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'profiles' AND column_name = 'email_alerts_last_reset'
  ) THEN
    ALTER TABLE profiles ADD COLUMN email_alerts_last_reset DATE DEFAULT CURRENT_DATE;
  END IF;
END $$;

-- 3. Create email_export_preferences table if not exists
CREATE TABLE IF NOT EXISTS email_export_preferences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  frequency VARCHAR(20) DEFAULT 'daily' CHECK (frequency IN ('on_demand', 'daily', 'weekly')),
  preferred_time TIME DEFAULT '08:00:00',  -- UTC
  include_global BOOLEAN DEFAULT TRUE,
  include_watchlist BOOLEAN DEFAULT TRUE,
  format VARCHAR(20) DEFAULT 'summary' CHECK (format IN ('summary', 'detailed')),
  enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id)
);

-- 4. Create alert_send_log table if not exists
CREATE TABLE IF NOT EXISTS alert_send_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  alert_type VARCHAR(50) NOT NULL,
  sent_at TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'::jsonb
);

-- 5. Create briefing_cache table if not exists (for storing LLM briefings)
CREATE TABLE IF NOT EXISTS briefing_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  preset VARCHAR(50) NOT NULL,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '15 minutes'
);

-- 6. Auto-create email preferences for new users
CREATE OR REPLACE FUNCTION handle_new_user_email_prefs()
RETURNS TRIGGER AS $$
BEGIN
  -- Create default email preferences for new user
  INSERT INTO email_export_preferences (user_id, frequency, enabled)
  VALUES (NEW.id, 'daily', TRUE)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop and recreate trigger to avoid duplicates
DROP TRIGGER IF EXISTS on_auth_user_created_email_prefs ON auth.users;
CREATE TRIGGER on_auth_user_created_email_prefs
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION handle_new_user_email_prefs();

-- 7. Function to reset daily email quota (called by cron)
CREATE OR REPLACE FUNCTION reset_daily_email_quota()
RETURNS void AS $$
BEGIN
  UPDATE profiles
  SET email_alerts_sent_today = 0,
      email_alerts_last_reset = CURRENT_DATE
  WHERE email_alerts_last_reset < CURRENT_DATE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 8. Extend existing trials to 14 days (for users currently in trial)
UPDATE profiles
SET trial_ends_at = created_at + INTERVAL '14 days'
WHERE plan = 'trial'
  AND trial_ends_at IS NOT NULL
  AND trial_ends_at > NOW();  -- Only extend active trials

-- 9. RLS policies for new tables
ALTER TABLE email_export_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_send_log ENABLE ROW LEVEL SECURITY;

-- Users can read/update their own preferences
DROP POLICY IF EXISTS "Users can view own email preferences" ON email_export_preferences;
CREATE POLICY "Users can view own email preferences" ON email_export_preferences
  FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own email preferences" ON email_export_preferences;
CREATE POLICY "Users can update own email preferences" ON email_export_preferences
  FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own email preferences" ON email_export_preferences;
CREATE POLICY "Users can insert own email preferences" ON email_export_preferences
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Service role can access all for cron jobs
DROP POLICY IF EXISTS "Service role full access to email preferences" ON email_export_preferences;
CREATE POLICY "Service role full access to email preferences" ON email_export_preferences
  FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Alert log - users can view their own
DROP POLICY IF EXISTS "Users can view own alert log" ON alert_send_log;
CREATE POLICY "Users can view own alert log" ON alert_send_log
  FOR SELECT USING (auth.uid() = user_id);

-- Service role full access for cron
DROP POLICY IF EXISTS "Service role full access to alert log" ON alert_send_log;
CREATE POLICY "Service role full access to alert log" ON alert_send_log
  FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- 10. Index for efficient cron queries
CREATE INDEX IF NOT EXISTS idx_email_prefs_frequency_enabled
  ON email_export_preferences(frequency, enabled)
  WHERE enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_alert_send_log_user_sent
  ON alert_send_log(user_id, sent_at DESC);

-- Done!
SELECT 'Migration complete: Trial extended to 14 days, email alert infrastructure ready' as status;
