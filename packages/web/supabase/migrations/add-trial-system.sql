-- LatticeForge Trial System Migration
-- Adds 7-day trial functionality for new users

-- ============================================
-- ADD TRIAL FIELDS TO PROFILES
-- ============================================

-- Add trial_ends_at column to profiles
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days');

-- Add plan column to profiles for individual user plans (separate from org)
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS plan VARCHAR(50) DEFAULT 'trial';

-- ============================================
-- UPDATE NEW USER TRIGGER
-- ============================================

-- Drop existing trigger first
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Update the function to set trial
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, email, full_name, role, plan, trial_ends_at)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
        COALESCE((NEW.raw_user_meta_data->>'role')::user_role, 'consumer'),
        'trial',
        NOW() + INTERVAL '7 days'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Recreate trigger
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to check if trial is active
CREATE OR REPLACE FUNCTION is_trial_active(user_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    trial_end TIMESTAMPTZ;
    user_plan VARCHAR(50);
BEGIN
    SELECT trial_ends_at, plan INTO trial_end, user_plan
    FROM profiles
    WHERE id = user_id;

    -- If they've upgraded, no longer on trial
    IF user_plan != 'trial' THEN
        RETURN FALSE;
    END IF;

    -- Check if trial hasn't expired
    RETURN trial_end IS NOT NULL AND trial_end > NOW();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get days remaining in trial
CREATE OR REPLACE FUNCTION trial_days_remaining(user_id UUID)
RETURNS INTEGER AS $$
DECLARE
    trial_end TIMESTAMPTZ;
BEGIN
    SELECT trial_ends_at INTO trial_end
    FROM profiles
    WHERE id = user_id;

    IF trial_end IS NULL OR trial_end < NOW() THEN
        RETURN 0;
    END IF;

    RETURN GREATEST(0, EXTRACT(DAY FROM (trial_end - NOW()))::INTEGER + 1);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to expire trial and downgrade to free
CREATE OR REPLACE FUNCTION expire_trial(user_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE profiles
    SET plan = 'free'
    WHERE id = user_id AND plan = 'trial';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- SCHEDULED JOB (requires pg_cron extension)
-- Run in Supabase Dashboard > SQL Editor if pg_cron is enabled
-- ============================================

-- To auto-expire trials, run this once pg_cron is available:
-- SELECT cron.schedule(
--     'expire-trials',
--     '0 0 * * *',  -- Run daily at midnight
--     $$UPDATE profiles SET plan = 'free' WHERE plan = 'trial' AND trial_ends_at < NOW()$$
-- );

-- ============================================
-- UPDATE EXISTING USERS (optional)
-- Uncomment to give existing users a trial
-- ============================================

-- UPDATE profiles
-- SET trial_ends_at = NOW() + INTERVAL '7 days', plan = 'trial'
-- WHERE plan IS NULL OR plan = 'free';
