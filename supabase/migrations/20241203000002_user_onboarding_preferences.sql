-- Add onboarding tracking to profiles and create user preferences table
-- Migration: 20241203000002_user_onboarding_preferences

-- 1. Add onboarding_completed_at to profiles table
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS onboarding_completed_at TIMESTAMPTZ DEFAULT NULL;

-- 2. Create user_preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  preferences JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT user_preferences_user_id_key UNIQUE (user_id)
);

-- 3. Enable RLS
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- 4. Create RLS policies for user_preferences
-- Users can only see their own preferences
CREATE POLICY "Users can view own preferences"
  ON user_preferences FOR SELECT
  USING (auth.uid() = user_id);

-- Users can insert their own preferences
CREATE POLICY "Users can insert own preferences"
  ON user_preferences FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Users can update their own preferences
CREATE POLICY "Users can update own preferences"
  ON user_preferences FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Users can delete their own preferences
CREATE POLICY "Users can delete own preferences"
  ON user_preferences FOR DELETE
  USING (auth.uid() = user_id);

-- 5. Create index for fast lookup
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id
  ON user_preferences(user_id);

-- 6. Add trigger to update updated_at
CREATE OR REPLACE FUNCTION update_user_preferences_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS user_preferences_updated_at ON user_preferences;
CREATE TRIGGER user_preferences_updated_at
  BEFORE UPDATE ON user_preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_user_preferences_updated_at();

-- 7. Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON user_preferences TO authenticated;

COMMENT ON TABLE user_preferences IS 'Stores user preferences and onboarding configuration';
COMMENT ON COLUMN user_preferences.preferences IS 'JSON object containing user preferences from onboarding wizard';
COMMENT ON COLUMN profiles.onboarding_completed_at IS 'Timestamp when user completed onboarding, NULL if not completed';
