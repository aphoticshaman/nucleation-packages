-- ============================================================================
-- User Preferences & Onboarding Schema
-- Run this in Supabase SQL Editor to enable persistent onboarding + preferences
-- ============================================================================

-- 1. Ensure profiles table has onboarding_completed_at column
-- ============================================================================
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'profiles' AND column_name = 'onboarding_completed_at'
  ) THEN
    ALTER TABLE profiles ADD COLUMN onboarding_completed_at TIMESTAMPTZ DEFAULT NULL;
    COMMENT ON COLUMN profiles.onboarding_completed_at IS 'When user completed onboarding wizard';
  END IF;
END $$;

-- 2. Create user_preferences table
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_preferences (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  -- Onboarding selections stored as JSONB for flexibility
  preferences JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,

  -- Ensure one preferences row per user
  CONSTRAINT user_preferences_user_id_key UNIQUE (user_id)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

-- 3. Enable RLS on user_preferences
-- ============================================================================
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- Users can only read their own preferences
DROP POLICY IF EXISTS "Users can view own preferences" ON user_preferences;
CREATE POLICY "Users can view own preferences" ON user_preferences
  FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own preferences
DROP POLICY IF EXISTS "Users can insert own preferences" ON user_preferences;
CREATE POLICY "Users can insert own preferences" ON user_preferences
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can update their own preferences
DROP POLICY IF EXISTS "Users can update own preferences" ON user_preferences;
CREATE POLICY "Users can update own preferences" ON user_preferences
  FOR UPDATE USING (auth.uid() = user_id);

-- Admins can read all preferences (for support/debugging)
DROP POLICY IF EXISTS "Admins can view all preferences" ON user_preferences;
CREATE POLICY "Admins can view all preferences" ON user_preferences
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM profiles
      WHERE profiles.id = auth.uid()
      AND profiles.role = 'admin'
    )
  );

-- 4. Function to auto-update updated_at timestamp
-- ============================================================================
CREATE OR REPLACE FUNCTION update_user_preferences_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to call function on update
DROP TRIGGER IF EXISTS user_preferences_updated_at ON user_preferences;
CREATE TRIGGER user_preferences_updated_at
  BEFORE UPDATE ON user_preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_user_preferences_updated_at();

-- 5. Grant permissions
-- ============================================================================
GRANT SELECT, INSERT, UPDATE ON user_preferences TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- 6. Add helpful comments
-- ============================================================================
COMMENT ON TABLE user_preferences IS 'Stores user preferences from onboarding and settings';
COMMENT ON COLUMN user_preferences.preferences IS 'JSONB containing: useCase, experienceLevel, regionsOfInterest, threatFocusAreas, contentDepth, etc.';

-- ============================================================================
-- SAMPLE PREFERENCES STRUCTURE (for reference):
-- {
--   "useCase": "researcher" | "analyst" | "journalist" | "investor" | "executive" | "government",
--   "experienceLevel": "beginner" | "intermediate" | "expert",
--   "regionsOfInterest": ["middle_east", "europe", "asia_pacific", ...],
--   "threatFocusAreas": ["military", "economic", "cyber", "political", ...],
--   "contentDepth": "summary" | "standard" | "detailed",
--   "notificationPreferences": {
--     "email": true,
--     "slack": false,
--     "frequency": "daily" | "weekly" | "realtime"
--   }
-- }
-- ============================================================================

-- 7. Verify the migration
-- ============================================================================
DO $$
BEGIN
  -- Check user_preferences table exists
  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'user_preferences') THEN
    RAISE EXCEPTION 'user_preferences table was not created';
  END IF;

  -- Check profiles.onboarding_completed_at exists
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'profiles' AND column_name = 'onboarding_completed_at'
  ) THEN
    RAISE EXCEPTION 'profiles.onboarding_completed_at column was not created';
  END IF;

  RAISE NOTICE 'Migration completed successfully!';
  RAISE NOTICE 'Tables: user_preferences (created/verified)';
  RAISE NOTICE 'Columns: profiles.onboarding_completed_at (created/verified)';
  RAISE NOTICE 'RLS policies applied to user_preferences';
END $$;
