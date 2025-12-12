-- User & IP Banning System
-- Allows admins to ban problematic users and IP addresses

-- =============================================================================
-- ADD BANNED COLUMNS TO PROFILES
-- =============================================================================
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_banned BOOLEAN DEFAULT false;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS ban_reason TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS banned_at TIMESTAMPTZ;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS banned_by UUID REFERENCES profiles(id);

-- Index for quickly filtering banned users
CREATE INDEX IF NOT EXISTS idx_profiles_is_banned ON profiles(is_banned) WHERE is_banned = true;

-- =============================================================================
-- BANNED IPS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS banned_ips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ip_address INET NOT NULL,
    ip_range CIDR,  -- Optional: for banning entire ranges (e.g., VPN providers)
    reason TEXT NOT NULL,

    -- Who banned and when
    banned_by UUID REFERENCES profiles(id),
    banned_at TIMESTAMPTZ DEFAULT NOW(),

    -- Optional expiry for temporary bans
    expires_at TIMESTAMPTZ,

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Unique constraint on IP address
CREATE UNIQUE INDEX IF NOT EXISTS idx_banned_ips_address ON banned_ips(ip_address) WHERE is_active = true;

-- Index for checking if an IP is banned
CREATE INDEX IF NOT EXISTS idx_banned_ips_active ON banned_ips(is_active, ip_address);

-- Index for CIDR range lookups
CREATE INDEX IF NOT EXISTS idx_banned_ips_range ON banned_ips USING gist (ip_range inet_ops) WHERE is_active = true AND ip_range IS NOT NULL;

-- =============================================================================
-- RLS FOR BANNED_IPS
-- =============================================================================
ALTER TABLE banned_ips ENABLE ROW LEVEL SECURITY;

-- Only admins can view/manage banned IPs
DROP POLICY IF EXISTS "banned_ips_admin_all" ON banned_ips;
CREATE POLICY "banned_ips_admin_all" ON banned_ips
    FOR ALL USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = (SELECT auth.uid()) AND role = 'admin')
    );

-- Service role access for middleware checks
DROP POLICY IF EXISTS "banned_ips_service" ON banned_ips;
CREATE POLICY "banned_ips_service" ON banned_ips
    FOR SELECT USING ((SELECT auth.jwt()->>'role') = 'service_role');

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Check if IP is banned (for middleware)
CREATE OR REPLACE FUNCTION public.is_ip_banned(check_ip INET)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM public.banned_ips
        WHERE is_active = true
        AND (expires_at IS NULL OR expires_at > NOW())
        AND (
            ip_address = check_ip
            OR (ip_range IS NOT NULL AND check_ip << ip_range)
        )
    );
END;
$$;

-- Check if user is banned
CREATE OR REPLACE FUNCTION public.is_user_banned(check_user_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM public.profiles
        WHERE id = check_user_id
        AND is_banned = true
    );
END;
$$;

-- Ban a user (admin only)
CREATE OR REPLACE FUNCTION public.ban_user(
    target_user_id UUID,
    reason TEXT DEFAULT 'Violation of terms of service'
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    admin_id UUID;
BEGIN
    -- Get current user
    admin_id := (SELECT auth.uid());

    -- Check if current user is admin
    IF NOT EXISTS (SELECT 1 FROM public.profiles WHERE id = admin_id AND role = 'admin') THEN
        RAISE EXCEPTION 'Only admins can ban users';
    END IF;

    -- Don't allow banning other admins
    IF EXISTS (SELECT 1 FROM public.profiles WHERE id = target_user_id AND role = 'admin') THEN
        RAISE EXCEPTION 'Cannot ban admin users';
    END IF;

    -- Ban the user
    UPDATE public.profiles
    SET
        is_banned = true,
        ban_reason = reason,
        banned_at = NOW(),
        banned_by = admin_id,
        is_active = false
    WHERE id = target_user_id;

    RETURN FOUND;
END;
$$;

-- Unban a user (admin only)
CREATE OR REPLACE FUNCTION public.unban_user(target_user_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    admin_id UUID;
BEGIN
    -- Get current user
    admin_id := (SELECT auth.uid());

    -- Check if current user is admin
    IF NOT EXISTS (SELECT 1 FROM public.profiles WHERE id = admin_id AND role = 'admin') THEN
        RAISE EXCEPTION 'Only admins can unban users';
    END IF;

    -- Unban the user
    UPDATE public.profiles
    SET
        is_banned = false,
        ban_reason = NULL,
        banned_at = NULL,
        banned_by = NULL,
        is_active = true
    WHERE id = target_user_id;

    RETURN FOUND;
END;
$$;

-- Ban an IP address (admin only)
CREATE OR REPLACE FUNCTION public.ban_ip(
    target_ip INET,
    reason TEXT DEFAULT 'Suspicious activity',
    duration_hours INTEGER DEFAULT NULL  -- NULL = permanent
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    admin_id UUID;
    ban_id UUID;
    expiry TIMESTAMPTZ;
BEGIN
    -- Get current user
    admin_id := (SELECT auth.uid());

    -- Check if current user is admin
    IF NOT EXISTS (SELECT 1 FROM public.profiles WHERE id = admin_id AND role = 'admin') THEN
        RAISE EXCEPTION 'Only admins can ban IPs';
    END IF;

    -- Calculate expiry if duration provided
    IF duration_hours IS NOT NULL THEN
        expiry := NOW() + (duration_hours || ' hours')::INTERVAL;
    END IF;

    -- Insert or update ban
    INSERT INTO public.banned_ips (ip_address, reason, banned_by, expires_at)
    VALUES (target_ip, reason, admin_id, expiry)
    ON CONFLICT (ip_address) WHERE is_active = true
    DO UPDATE SET
        reason = EXCLUDED.reason,
        banned_by = EXCLUDED.banned_by,
        expires_at = EXCLUDED.expires_at,
        updated_at = NOW()
    RETURNING id INTO ban_id;

    RETURN ban_id;
END;
$$;

-- Unban an IP address (admin only)
CREATE OR REPLACE FUNCTION public.unban_ip(target_ip INET)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    admin_id UUID;
BEGIN
    -- Get current user
    admin_id := (SELECT auth.uid());

    -- Check if current user is admin
    IF NOT EXISTS (SELECT 1 FROM public.profiles WHERE id = admin_id AND role = 'admin') THEN
        RAISE EXCEPTION 'Only admins can unban IPs';
    END IF;

    -- Deactivate the ban (soft delete)
    UPDATE public.banned_ips
    SET is_active = false, updated_at = NOW()
    WHERE ip_address = target_ip AND is_active = true;

    RETURN FOUND;
END;
$$;

-- =============================================================================
-- UPDATE ADMIN POLICIES FOR PROFILES
-- =============================================================================

-- Admins can update all profiles (for banning)
DROP POLICY IF EXISTS "Admins can update all profiles" ON profiles;
CREATE POLICY "Admins can update all profiles"
    ON profiles FOR UPDATE
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = (SELECT auth.uid()) AND role = 'admin')
    );

-- =============================================================================
-- AUDIT VIEW FOR BANNED USERS
-- =============================================================================
CREATE OR REPLACE VIEW admin_banned_users AS
SELECT
    p.id,
    p.email,
    p.full_name,
    p.role,
    p.ban_reason,
    p.banned_at,
    banner.email as banned_by_email,
    banner.full_name as banned_by_name
FROM profiles p
LEFT JOIN profiles banner ON p.banned_by = banner.id
WHERE p.is_banned = true
ORDER BY p.banned_at DESC;
