-- Session Tracking & Email Export Preferences
-- Track user sign in/out, online status, and email export settings

-- ============================================
-- SESSION TRACKING
-- ============================================

-- Add session tracking columns to profiles
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS is_online BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS last_sign_in_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS last_sign_out_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS current_session_id UUID,
ADD COLUMN IF NOT EXISTS remember_me BOOLEAN DEFAULT false;

-- Sessions table to track individual user sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,

    -- Session info
    session_token VARCHAR(255) UNIQUE,
    device_info JSONB DEFAULT '{}',  -- user agent, device type, etc.
    ip_address INET,

    -- Status
    is_active BOOLEAN DEFAULT true,
    remember_me BOOLEAN DEFAULT false,

    -- Timestamps
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    -- End reason
    end_reason VARCHAR(50)  -- 'logout', 'expired', 'revoked', 'new_session'
);

-- Indexes for session queries
CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(user_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at) WHERE is_active = true;

-- RLS for sessions
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own sessions" ON user_sessions;
CREATE POLICY "Users can view own sessions"
    ON user_sessions FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Admins can view all sessions" ON user_sessions;
CREATE POLICY "Admins can view all sessions"
    ON user_sessions FOR SELECT
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

-- ============================================
-- EMAIL EXPORT PREFERENCES
-- ============================================

-- Email export preferences for intel packages
CREATE TABLE IF NOT EXISTS email_export_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL UNIQUE,

    -- Email settings
    email_enabled BOOLEAN DEFAULT false,
    email_address VARCHAR(255),  -- can be different from profile email

    -- Export format preferences
    include_text_body BOOLEAN DEFAULT true,
    include_pdf_attachment BOOLEAN DEFAULT false,
    include_json_attachment BOOLEAN DEFAULT false,
    include_markdown_attachment BOOLEAN DEFAULT false,

    -- Frequency
    frequency VARCHAR(20) DEFAULT 'on_demand',  -- on_demand, daily, weekly
    preferred_time TIME DEFAULT '08:00:00',  -- for scheduled emails
    timezone VARCHAR(50) DEFAULT 'America/New_York',

    -- Content preferences
    domains TEXT[] DEFAULT '{}',  -- which intel domains to include
    min_priority INTEGER DEFAULT 3,  -- 1-5, only include items >= this priority

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS for email preferences
ALTER TABLE email_export_preferences ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage own email prefs" ON email_export_preferences;
CREATE POLICY "Users can manage own email prefs"
    ON email_export_preferences FOR ALL
    USING (auth.uid() = user_id);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Record sign in
CREATE OR REPLACE FUNCTION record_sign_in(
    p_user_id UUID,
    p_session_token VARCHAR DEFAULT NULL,
    p_device_info JSONB DEFAULT '{}',
    p_ip_address INET DEFAULT NULL,
    p_remember_me BOOLEAN DEFAULT false
)
RETURNS UUID AS $$
DECLARE
    v_session_id UUID;
    v_expires_at TIMESTAMPTZ;
BEGIN
    -- Set expiry based on remember_me
    IF p_remember_me THEN
        v_expires_at := NOW() + INTERVAL '30 days';
    ELSE
        v_expires_at := NOW() + INTERVAL '24 hours';
    END IF;

    -- End any existing active sessions for this user (single session policy)
    UPDATE user_sessions
    SET is_active = false,
        ended_at = NOW(),
        end_reason = 'new_session'
    WHERE user_id = p_user_id AND is_active = true;

    -- Create new session
    INSERT INTO user_sessions (
        user_id, session_token, device_info, ip_address,
        remember_me, expires_at
    )
    VALUES (
        p_user_id, p_session_token, p_device_info, p_ip_address,
        p_remember_me, v_expires_at
    )
    RETURNING id INTO v_session_id;

    -- Update profile
    UPDATE profiles
    SET is_online = true,
        last_sign_in_at = NOW(),
        current_session_id = v_session_id,
        remember_me = p_remember_me,
        last_seen_at = NOW()
    WHERE id = p_user_id;

    -- Log activity
    INSERT INTO user_activity (user_id, action, details)
    VALUES (p_user_id, 'sign_in', jsonb_build_object(
        'session_id', v_session_id,
        'remember_me', p_remember_me,
        'device_info', p_device_info
    ));

    RETURN v_session_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Record sign out
CREATE OR REPLACE FUNCTION record_sign_out(p_user_id UUID)
RETURNS void AS $$
BEGIN
    -- End active sessions
    UPDATE user_sessions
    SET is_active = false,
        ended_at = NOW(),
        end_reason = 'logout'
    WHERE user_id = p_user_id AND is_active = true;

    -- Update profile
    UPDATE profiles
    SET is_online = false,
        last_sign_out_at = NOW(),
        current_session_id = NULL
    WHERE id = p_user_id;

    -- Log activity
    INSERT INTO user_activity (user_id, action, details)
    VALUES (p_user_id, 'sign_out', '{}');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Update session activity (heartbeat)
CREATE OR REPLACE FUNCTION update_session_activity(p_user_id UUID)
RETURNS void AS $$
BEGIN
    UPDATE user_sessions
    SET last_active_at = NOW()
    WHERE user_id = p_user_id AND is_active = true;

    UPDATE profiles
    SET last_seen_at = NOW()
    WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Expire old sessions (run periodically via cron/edge function)
CREATE OR REPLACE FUNCTION expire_sessions()
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    WITH expired AS (
        UPDATE user_sessions
        SET is_active = false,
            ended_at = NOW(),
            end_reason = 'expired'
        WHERE is_active = true
        AND expires_at < NOW()
        RETURNING user_id
    )
    SELECT COUNT(*) INTO v_count FROM expired;

    -- Update profiles for expired sessions
    UPDATE profiles
    SET is_online = false, current_session_id = NULL
    WHERE id IN (
        SELECT DISTINCT user_id FROM user_sessions
        WHERE end_reason = 'expired' AND ended_at > NOW() - INTERVAL '1 minute'
    )
    AND NOT EXISTS (
        SELECT 1 FROM user_sessions
        WHERE user_id = profiles.id AND is_active = true
    );

    RETURN v_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- ADMIN VIEWS
-- ============================================

-- Active sessions overview for admin
CREATE OR REPLACE VIEW admin_active_sessions AS
SELECT
    us.id as session_id,
    p.id as user_id,
    p.email,
    p.full_name,
    p.role,
    us.device_info,
    us.ip_address,
    us.started_at,
    us.last_active_at,
    us.expires_at,
    us.remember_me,
    EXTRACT(EPOCH FROM (NOW() - us.last_active_at))/60 as minutes_idle
FROM user_sessions us
JOIN profiles p ON p.id = us.user_id
WHERE us.is_active = true
ORDER BY us.last_active_at DESC;
