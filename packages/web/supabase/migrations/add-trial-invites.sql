-- LatticeForge Trial Invite System
-- Allows admins to create trial invite links for specific emails

-- ============================================
-- TRIAL INVITES TABLE
-- ============================================

CREATE TABLE trial_invites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Invite details
    email VARCHAR(255) NOT NULL,
    token VARCHAR(64) UNIQUE NOT NULL,  -- Random token for invite URL

    -- Customization
    trial_days INTEGER DEFAULT 7,  -- Can extend trial for special invites
    note TEXT,  -- Admin note about this invite

    -- Status tracking
    invited_by UUID REFERENCES profiles(id),
    status VARCHAR(50) DEFAULT 'pending',  -- pending, accepted, expired

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '30 days'),  -- Link expires in 30 days
    accepted_at TIMESTAMPTZ,
    accepted_by UUID REFERENCES profiles(id)
);

-- Index for token lookups
CREATE INDEX idx_trial_invites_token ON trial_invites(token);
CREATE INDEX idx_trial_invites_email ON trial_invites(email);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE trial_invites ENABLE ROW LEVEL SECURITY;

-- Admins can manage all invites
CREATE POLICY "Admins can manage trial invites"
    ON trial_invites FOR ALL
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

-- Public can read their own invite by token (handled via API with service key)

-- ============================================
-- FUNCTIONS
-- ============================================

-- Create a trial invite
CREATE OR REPLACE FUNCTION create_trial_invite(
    invite_email VARCHAR(255),
    admin_id UUID,
    custom_trial_days INTEGER DEFAULT 7,
    admin_note TEXT DEFAULT NULL
)
RETURNS TABLE(token VARCHAR(64), invite_url TEXT) AS $$
DECLARE
    new_token VARCHAR(64);
BEGIN
    -- Generate a random token
    new_token := encode(gen_random_bytes(32), 'hex');

    -- Insert the invite
    INSERT INTO trial_invites (email, token, trial_days, note, invited_by)
    VALUES (invite_email, new_token, custom_trial_days, admin_note, admin_id);

    -- Return the token and URL
    RETURN QUERY SELECT
        new_token,
        'https://latticeforge.ai/signup?invite=' || new_token;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Validate and consume an invite
CREATE OR REPLACE FUNCTION validate_invite(invite_token VARCHAR(64))
RETURNS TABLE(
    valid BOOLEAN,
    invite_email VARCHAR(255),
    invite_trial_days INTEGER,
    error_message TEXT
) AS $$
DECLARE
    invite_record trial_invites%ROWTYPE;
BEGIN
    -- Find the invite
    SELECT * INTO invite_record
    FROM trial_invites
    WHERE token = invite_token;

    -- Check if invite exists
    IF invite_record.id IS NULL THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(255), NULL::INTEGER, 'Invalid invite token';
        RETURN;
    END IF;

    -- Check if already used
    IF invite_record.status = 'accepted' THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(255), NULL::INTEGER, 'Invite has already been used';
        RETURN;
    END IF;

    -- Check if expired
    IF invite_record.expires_at < NOW() THEN
        UPDATE trial_invites SET status = 'expired' WHERE token = invite_token;
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(255), NULL::INTEGER, 'Invite has expired';
        RETURN;
    END IF;

    -- Valid invite
    RETURN QUERY SELECT TRUE, invite_record.email, invite_record.trial_days, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Mark invite as accepted (call after user signs up)
CREATE OR REPLACE FUNCTION accept_invite(invite_token VARCHAR(64), user_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE trial_invites
    SET
        status = 'accepted',
        accepted_at = NOW(),
        accepted_by = user_id
    WHERE token = invite_token AND status = 'pending';

    -- Also update the user's trial period if custom
    UPDATE profiles
    SET trial_ends_at = NOW() + (
        SELECT (trial_days || ' days')::INTERVAL
        FROM trial_invites
        WHERE token = invite_token
    )
    WHERE id = user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- ADMIN VIEW
-- ============================================

CREATE OR REPLACE VIEW admin_trial_invites AS
SELECT
    ti.id,
    ti.email,
    ti.token,
    ti.trial_days,
    ti.note,
    ti.status,
    ti.created_at,
    ti.expires_at,
    ti.accepted_at,
    p.full_name as invited_by_name,
    ap.full_name as accepted_by_name
FROM trial_invites ti
LEFT JOIN profiles p ON p.id = ti.invited_by
LEFT JOIN profiles ap ON ap.id = ti.accepted_by
ORDER BY ti.created_at DESC;
