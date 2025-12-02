-- LatticeForge User Management Schema
-- Role-based access for Admin, Enterprise, Consumer dashboards

-- ============================================
-- USER ROLES & PROFILES
-- ============================================

-- User roles enum (idempotent - only create if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('admin', 'enterprise', 'consumer', 'support');
    END IF;
END$$;

-- Organizations (for enterprise customers) - must be created before profiles
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,

    -- Billing
    stripe_customer_id VARCHAR(255),
    plan VARCHAR(50) DEFAULT 'free',  -- free, starter, pro, enterprise
    plan_status VARCHAR(50) DEFAULT 'active',

    -- Limits
    api_calls_limit INTEGER DEFAULT 1000,
    api_calls_used INTEGER DEFAULT 0,
    team_seats_limit INTEGER DEFAULT 5,

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Metadata
    logo_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User profiles (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    role user_role NOT NULL DEFAULT 'consumer',

    -- Organization (for enterprise users)
    organization_id UUID REFERENCES organizations(id),

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_seen_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Keys (for enterprise)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_by UUID REFERENCES profiles(id),

    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash of the key
    key_prefix VARCHAR(8) NOT NULL, -- First 8 chars for display (lf_live_...)

    -- Permissions
    scopes TEXT[] DEFAULT '{"read", "write"}',

    -- Limits
    rate_limit INTEGER DEFAULT 100,  -- requests per minute

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Usage logs
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,

    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,

    -- Request metadata
    ip_address INET,
    user_agent TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for usage queries
CREATE INDEX IF NOT EXISTS idx_api_usage_org_time ON api_usage(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_key_time ON api_usage(api_key_id, created_at DESC);

-- ============================================
-- CONSUMER USER DATA
-- ============================================

-- Saved simulations (consumer)
CREATE TABLE IF NOT EXISTS saved_simulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Simulation state
    config JSONB NOT NULL,
    state JSONB NOT NULL,

    -- Sharing
    is_public BOOLEAN DEFAULT false,
    share_token VARCHAR(32) UNIQUE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User activity (for admin dashboard)
CREATE TABLE IF NOT EXISTS user_activity (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,

    action VARCHAR(100) NOT NULL,  -- login, simulation_run, api_call, etc.
    details JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_activity_time ON user_activity(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity(user_id, created_at DESC);

-- ============================================
-- ADMIN VIEWS
-- ============================================

-- Dashboard stats for admin
CREATE OR REPLACE VIEW admin_dashboard_stats AS
SELECT
    (SELECT COUNT(*) FROM profiles WHERE role = 'consumer') as total_consumers,
    (SELECT COUNT(*) FROM profiles WHERE role = 'enterprise') as total_enterprise,
    (SELECT COUNT(*) FROM organizations WHERE is_active = true) as active_orgs,
    (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '24 hours') as active_24h,
    (SELECT COUNT(*) FROM profiles WHERE last_seen_at > NOW() - INTERVAL '7 days') as active_7d,
    (SELECT COUNT(*) FROM api_usage WHERE created_at > NOW() - INTERVAL '24 hours') as api_calls_24h,
    (SELECT COUNT(*) FROM saved_simulations) as total_simulations;

-- Enterprise customers overview for admin
CREATE OR REPLACE VIEW admin_enterprise_overview AS
SELECT
    o.id,
    o.name,
    o.slug,
    o.plan,
    o.plan_status,
    o.api_calls_used,
    o.api_calls_limit,
    o.is_active,
    o.created_at,
    COUNT(DISTINCT p.id) as team_size,
    COUNT(DISTINCT ak.id) as api_key_count,
    MAX(p.last_seen_at) as last_activity,
    (SELECT COUNT(*) FROM api_usage au WHERE au.organization_id = o.id AND au.created_at > NOW() - INTERVAL '24 hours') as api_calls_24h
FROM organizations o
LEFT JOIN profiles p ON p.organization_id = o.id
LEFT JOIN api_keys ak ON ak.organization_id = o.id AND ak.is_active = true
GROUP BY o.id;

-- Consumer users overview for admin
CREATE OR REPLACE VIEW admin_consumer_overview AS
SELECT
    p.id,
    p.email,
    p.full_name,
    p.is_active,
    p.last_seen_at,
    p.created_at,
    COUNT(ss.id) as simulation_count,
    (SELECT COUNT(*) FROM user_activity ua WHERE ua.user_id = p.id AND ua.created_at > NOW() - INTERVAL '7 days') as actions_7d
FROM profiles p
LEFT JOIN saved_simulations ss ON ss.user_id = p.id
WHERE p.role = 'consumer'
GROUP BY p.id;

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_activity ENABLE ROW LEVEL SECURITY;

-- Profiles: users can read their own, admins can read all
DROP POLICY IF EXISTS "Users can view own profile" ON profiles;
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Admins can view all profiles" ON profiles;
CREATE POLICY "Admins can view all profiles"
    ON profiles FOR SELECT
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

DROP POLICY IF EXISTS "Users can update own profile" ON profiles;
CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

-- Organizations: members can view their org, admins can view all
DROP POLICY IF EXISTS "Org members can view their org" ON organizations;
CREATE POLICY "Org members can view their org"
    ON organizations FOR SELECT
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND organization_id = organizations.id)
    );

DROP POLICY IF EXISTS "Admins can view all orgs" ON organizations;
CREATE POLICY "Admins can view all orgs"
    ON organizations FOR SELECT
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
    );

-- API Keys: org members can manage their org's keys
DROP POLICY IF EXISTS "Org members can view their API keys" ON api_keys;
CREATE POLICY "Org members can view their API keys"
    ON api_keys FOR SELECT
    USING (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND organization_id = api_keys.organization_id)
    );

DROP POLICY IF EXISTS "Org members can create API keys" ON api_keys;
CREATE POLICY "Org members can create API keys"
    ON api_keys FOR INSERT
    WITH CHECK (
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND organization_id = api_keys.organization_id)
    );

-- Saved simulations: users own their simulations
DROP POLICY IF EXISTS "Users can CRUD own simulations" ON saved_simulations;
CREATE POLICY "Users can CRUD own simulations"
    ON saved_simulations FOR ALL
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Public simulations are viewable" ON saved_simulations;
CREATE POLICY "Public simulations are viewable"
    ON saved_simulations FOR SELECT
    USING (is_public = true);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Create profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, email, full_name, role)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
        COALESCE((NEW.raw_user_meta_data->>'role')::user_role, 'consumer')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- Update last_seen on activity
CREATE OR REPLACE FUNCTION update_last_seen()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE profiles SET last_seen_at = NOW() WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_user_activity ON user_activity;
CREATE TRIGGER on_user_activity
    AFTER INSERT ON user_activity
    FOR EACH ROW EXECUTE FUNCTION update_last_seen();

-- Get user role (for middleware)
CREATE OR REPLACE FUNCTION get_user_role(user_id UUID)
RETURNS user_role AS $$
    SELECT role FROM profiles WHERE id = user_id;
$$ LANGUAGE SQL SECURITY DEFINER;
