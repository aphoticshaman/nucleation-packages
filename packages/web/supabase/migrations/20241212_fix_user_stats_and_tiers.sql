-- ============================================================================
-- Fix User Stats, Tiers, and Online/Subscriber Counts
-- ============================================================================
-- Issues Fixed:
--   1. Tier naming: standardize to 'free', 'pro', 'enterprise' (not starter)
--   2. Online count: using is_online + session activity, not just last_seen_at
--   3. Subscriber count: track paid users separately
--   4. Session heartbeat: proper 5-minute timeout for online status
--   5. Admin dashboard stats: accurate real-time counts
-- ============================================================================

-- ============================================================================
-- PART 1: Standardize Tiers (Free, Pro, Enterprise)
-- ============================================================================

-- Add tier column if not exists (the user management page expects this)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'profiles' AND column_name = 'tier'
    ) THEN
        ALTER TABLE profiles ADD COLUMN tier VARCHAR(20) DEFAULT 'free';
    END IF;
END$$;

-- Migrate any 'starter' tiers to 'pro'
UPDATE profiles SET tier = 'pro' WHERE tier = 'starter';
-- Migrate 'enterprise_tier' to 'enterprise'
UPDATE profiles SET tier = 'enterprise' WHERE tier = 'enterprise_tier';

-- Also update organizations table
UPDATE organizations SET plan = 'pro' WHERE plan = 'starter';

-- ============================================================================
-- PART 2: Enhanced Online Status Tracking
-- ============================================================================

-- Add online_timeout column for customizable timeout per user
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'profiles' AND column_name = 'online_timeout_minutes'
    ) THEN
        ALTER TABLE profiles ADD COLUMN online_timeout_minutes INTEGER DEFAULT 5;
    END IF;
END$$;

-- Create function to check if user is truly online
-- (has active session AND recent activity within timeout window)
CREATE OR REPLACE FUNCTION is_user_online(user_id_param UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_timeout_minutes INTEGER;
    v_last_active TIMESTAMPTZ;
    v_has_active_session BOOLEAN;
BEGIN
    -- Get user's timeout setting
    SELECT COALESCE(online_timeout_minutes, 5) INTO v_timeout_minutes
    FROM profiles WHERE id = user_id_param;

    -- Check for active session
    SELECT EXISTS(
        SELECT 1 FROM user_sessions
        WHERE user_id = user_id_param AND is_active = true
    ) INTO v_has_active_session;

    IF NOT v_has_active_session THEN
        RETURN FALSE;
    END IF;

    -- Check last activity within timeout
    SELECT last_active_at INTO v_last_active
    FROM user_sessions
    WHERE user_id = user_id_param AND is_active = true
    ORDER BY last_active_at DESC
    LIMIT 1;

    IF v_last_active IS NULL THEN
        RETURN FALSE;
    END IF;

    RETURN v_last_active > NOW() - (v_timeout_minutes || ' minutes')::INTERVAL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- ============================================================================
-- PART 3: Subscriber Tracking
-- ============================================================================

-- Add subscription tracking columns if not exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'profiles' AND column_name = 'stripe_customer_id'
    ) THEN
        ALTER TABLE profiles ADD COLUMN stripe_customer_id VARCHAR(255);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'profiles' AND column_name = 'subscription_status'
    ) THEN
        ALTER TABLE profiles ADD COLUMN subscription_status VARCHAR(50) DEFAULT 'none';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'profiles' AND column_name = 'subscription_period_end'
    ) THEN
        ALTER TABLE profiles ADD COLUMN subscription_period_end TIMESTAMPTZ;
    END IF;
END$$;

-- Create function to check if user is a paying subscriber
CREATE OR REPLACE FUNCTION is_subscriber(user_id_param UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS(
        SELECT 1 FROM profiles
        WHERE id = user_id_param
        AND tier IN ('pro', 'enterprise')
        AND subscription_status = 'active'
        AND (subscription_period_end IS NULL OR subscription_period_end > NOW())
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- ============================================================================
-- PART 4: Enhanced Admin Dashboard Stats View
-- ============================================================================

-- Drop and recreate the dashboard stats view with accurate counts
DROP VIEW IF EXISTS admin_dashboard_stats CASCADE;

CREATE VIEW admin_dashboard_stats AS
WITH online_users AS (
    -- Users with active sessions and recent activity (5 min default)
    SELECT DISTINCT p.id
    FROM profiles p
    JOIN user_sessions us ON us.user_id = p.id
    WHERE us.is_active = true
    AND us.last_active_at > NOW() - INTERVAL '5 minutes'
),
subscribers AS (
    -- Paid users (pro or enterprise with active subscription)
    SELECT id FROM profiles
    WHERE tier IN ('pro', 'enterprise')
    AND subscription_status = 'active'
    AND (subscription_period_end IS NULL OR subscription_period_end > NOW())
),
tier_counts AS (
    SELECT
        COUNT(*) FILTER (WHERE tier = 'free') as free_tier,
        COUNT(*) FILTER (WHERE tier = 'pro') as pro_tier,
        COUNT(*) FILTER (WHERE tier = 'enterprise') as enterprise_tier
    FROM profiles
)
SELECT
    -- Total users
    (SELECT COUNT(*) FROM profiles) as total_users,

    -- By role
    (SELECT COUNT(*) FROM profiles WHERE role = 'consumer') as total_consumers,
    (SELECT COUNT(*) FROM profiles WHERE role = 'enterprise') as total_enterprise_users,
    (SELECT COUNT(*) FROM profiles WHERE role = 'admin') as total_admins,

    -- By tier
    (SELECT free_tier FROM tier_counts) as free_tier_count,
    (SELECT pro_tier FROM tier_counts) as pro_tier_count,
    (SELECT enterprise_tier FROM tier_counts) as enterprise_tier_count,

    -- Online status (accurate real-time)
    (SELECT COUNT(*) FROM online_users) as users_online_now,

    -- Subscribers
    (SELECT COUNT(*) FROM subscribers) as total_subscribers,
    (SELECT COUNT(*) FROM subscribers s JOIN online_users o ON s.id = o.id) as subscribers_online,

    -- Activity metrics
    (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '24 hours') as active_24h,
    (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '7 days') as active_7d,
    (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '30 days') as active_30d,

    -- Organizations
    (SELECT COUNT(*) FROM organizations WHERE is_active = true) as active_orgs,

    -- API activity
    (SELECT COUNT(*) FROM api_usage WHERE created_at > NOW() - INTERVAL '24 hours') as api_calls_24h,

    -- Content
    (SELECT COUNT(*) FROM saved_simulations) as total_simulations,

    -- Sessions
    (SELECT COUNT(*) FROM user_sessions WHERE is_active = true) as active_sessions;

-- ============================================================================
-- PART 5: Real-time Stats Functions
-- ============================================================================

-- Function to get accurate online count
CREATE OR REPLACE FUNCTION get_online_count()
RETURNS TABLE(
    total_online INTEGER,
    consumers_online INTEGER,
    enterprise_online INTEGER,
    subscribers_online INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT p.id)::INTEGER as total_online,
        COUNT(DISTINCT p.id) FILTER (WHERE p.role = 'consumer')::INTEGER as consumers_online,
        COUNT(DISTINCT p.id) FILTER (WHERE p.role = 'enterprise')::INTEGER as enterprise_online,
        COUNT(DISTINCT p.id) FILTER (
            WHERE p.tier IN ('pro', 'enterprise')
            AND p.subscription_status = 'active'
        )::INTEGER as subscribers_online
    FROM profiles p
    JOIN user_sessions us ON us.user_id = p.id
    WHERE us.is_active = true
    AND us.last_active_at > NOW() - INTERVAL '5 minutes';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- Function to get subscriber stats
CREATE OR REPLACE FUNCTION get_subscriber_stats()
RETURNS TABLE(
    total_subscribers INTEGER,
    pro_subscribers INTEGER,
    enterprise_subscribers INTEGER,
    subscribers_online INTEGER,
    mrr_estimate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH subs AS (
        SELECT
            id,
            tier,
            CASE
                WHEN tier = 'pro' THEN 29.00  -- Example MRR
                WHEN tier = 'enterprise' THEN 99.00
                ELSE 0.00
            END as monthly_value
        FROM profiles
        WHERE tier IN ('pro', 'enterprise')
        AND subscription_status = 'active'
        AND (subscription_period_end IS NULL OR subscription_period_end > NOW())
    ),
    online AS (
        SELECT DISTINCT user_id FROM user_sessions
        WHERE is_active = true AND last_active_at > NOW() - INTERVAL '5 minutes'
    )
    SELECT
        COUNT(*)::INTEGER as total_subscribers,
        COUNT(*) FILTER (WHERE tier = 'pro')::INTEGER as pro_subscribers,
        COUNT(*) FILTER (WHERE tier = 'enterprise')::INTEGER as enterprise_subscribers,
        COUNT(o.user_id)::INTEGER as subscribers_online,
        COALESCE(SUM(monthly_value), 0)::NUMERIC as mrr_estimate
    FROM subs
    LEFT JOIN online o ON o.user_id = subs.id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;

-- ============================================================================
-- PART 6: Session Heartbeat and Cleanup
-- ============================================================================

-- Function to update session heartbeat (call from client every ~1 min)
CREATE OR REPLACE FUNCTION heartbeat(p_session_id UUID DEFAULT NULL)
RETURNS BOOLEAN AS $$
DECLARE
    v_user_id UUID;
BEGIN
    v_user_id := auth.uid();

    IF v_user_id IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Update session(s) last_active_at
    IF p_session_id IS NOT NULL THEN
        UPDATE user_sessions
        SET last_active_at = NOW()
        WHERE id = p_session_id AND user_id = v_user_id AND is_active = true;
    ELSE
        UPDATE user_sessions
        SET last_active_at = NOW()
        WHERE user_id = v_user_id AND is_active = true;
    END IF;

    -- Update profile last_seen_at
    UPDATE profiles
    SET last_seen_at = NOW()
    WHERE id = v_user_id;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Enhanced session expiration with proper cleanup
CREATE OR REPLACE FUNCTION expire_stale_sessions()
RETURNS TABLE(
    expired_count INTEGER,
    marked_offline INTEGER
) AS $$
DECLARE
    v_expired INTEGER := 0;
    v_offline INTEGER := 0;
BEGIN
    -- Expire sessions that have been inactive for longer than their expiry
    WITH expired AS (
        UPDATE user_sessions
        SET is_active = false,
            ended_at = NOW(),
            end_reason = 'expired'
        WHERE is_active = true
        AND (
            expires_at < NOW()
            OR last_active_at < NOW() - INTERVAL '30 minutes'  -- Force expire after 30 min inactive
        )
        RETURNING user_id
    )
    SELECT COUNT(DISTINCT user_id) INTO v_expired FROM expired;

    -- Mark users offline if they have no active sessions
    WITH marked AS (
        UPDATE profiles
        SET is_online = false
        WHERE is_online = true
        AND NOT EXISTS (
            SELECT 1 FROM user_sessions
            WHERE user_id = profiles.id AND is_active = true
        )
        RETURNING id
    )
    SELECT COUNT(*) INTO v_offline FROM marked;

    RETURN QUERY SELECT v_expired, v_offline;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- PART 7: Admin User Overview (Enhanced)
-- ============================================================================

-- Drop and recreate consumer overview with online status
DROP VIEW IF EXISTS admin_consumer_overview CASCADE;

CREATE VIEW admin_consumer_overview AS
SELECT
    p.id,
    p.email,
    p.full_name,
    p.tier,
    p.is_active,
    p.last_seen_at,
    p.created_at,
    p.subscription_status,
    -- Accurate online status
    EXISTS(
        SELECT 1 FROM user_sessions us
        WHERE us.user_id = p.id
        AND us.is_active = true
        AND us.last_active_at > NOW() - INTERVAL '5 minutes'
    ) as is_online_now,
    -- Activity counts
    COUNT(ss.id) as simulation_count,
    (SELECT COUNT(*) FROM user_activity ua WHERE ua.user_id = p.id AND ua.created_at > NOW() - INTERVAL '7 days') as actions_7d
FROM profiles p
LEFT JOIN saved_simulations ss ON ss.user_id = p.id
WHERE p.role = 'consumer'
GROUP BY p.id;

-- ============================================================================
-- PART 8: Indexes for Performance
-- ============================================================================

-- Index for online status checks
CREATE INDEX IF NOT EXISTS idx_user_sessions_online_check
ON user_sessions(user_id, is_active, last_active_at)
WHERE is_active = true;

-- Index for subscriber queries
CREATE INDEX IF NOT EXISTS idx_profiles_subscribers
ON profiles(tier, subscription_status)
WHERE tier IN ('pro', 'enterprise') AND subscription_status = 'active';

-- Index for activity queries
CREATE INDEX IF NOT EXISTS idx_profiles_last_seen
ON profiles(last_seen_at DESC);

-- ============================================================================
-- PART 9: Grant Permissions
-- ============================================================================

-- Allow authenticated users to call heartbeat
GRANT EXECUTE ON FUNCTION heartbeat TO authenticated;

-- Admin-only functions
GRANT EXECUTE ON FUNCTION get_online_count TO authenticated;
GRANT EXECUTE ON FUNCTION get_subscriber_stats TO authenticated;
GRANT EXECUTE ON FUNCTION expire_stale_sessions TO authenticated;
