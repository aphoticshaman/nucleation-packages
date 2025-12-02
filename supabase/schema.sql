-- LatticeForge Database Schema
-- Run this in Supabase SQL Editor

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- ENUMS
-- ============================================

CREATE TYPE client_tier AS ENUM ('free', 'pro', 'enterprise', 'government');
CREATE TYPE api_key_status AS ENUM ('active', 'revoked', 'expired');
CREATE TYPE alert_severity AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE alert_type AS ENUM ('phase_change', 'anomaly', 'cascade', 'threshold');

-- ============================================
-- CLIENTS (Organizations/Users)
-- ============================================

CREATE TABLE clients (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  tier client_tier DEFAULT 'free',
  company TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- API KEYS
-- ============================================

CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
  key_hash TEXT NOT NULL, -- Store hash, not plaintext
  key_prefix TEXT NOT NULL, -- "lf_live_xxxx" for display
  name TEXT DEFAULT 'Default',
  status api_key_status DEFAULT 'active',
  last_used_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_client ON api_keys(client_id);

-- ============================================
-- USAGE TRACKING
-- ============================================

CREATE TABLE usage_records (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
  api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
  operation TEXT NOT NULL,
  signal_tokens INT DEFAULT 0,
  fusion_tokens INT DEFAULT 0,
  analysis_tokens INT DEFAULT 0,
  storage_tokens INT DEFAULT 0,
  total_tokens INT GENERATED ALWAYS AS (signal_tokens + fusion_tokens + analysis_tokens + storage_tokens) STORED,
  billing_period TEXT NOT NULL, -- "2025-12-01"
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_usage_client_period ON usage_records(client_id, billing_period);
CREATE INDEX idx_usage_created ON usage_records(created_at);

-- Monthly usage summary (materialized view for fast queries)
CREATE MATERIALIZED VIEW usage_summary AS
SELECT
  client_id,
  billing_period,
  SUM(signal_tokens) as total_signal_tokens,
  SUM(fusion_tokens) as total_fusion_tokens,
  SUM(analysis_tokens) as total_analysis_tokens,
  SUM(storage_tokens) as total_storage_tokens,
  SUM(total_tokens) as total_tokens,
  COUNT(*) as request_count
FROM usage_records
GROUP BY client_id, billing_period;

CREATE UNIQUE INDEX idx_usage_summary ON usage_summary(client_id, billing_period);

-- ============================================
-- ALERTS
-- ============================================

CREATE TABLE alerts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
  type alert_type NOT NULL,
  severity alert_severity NOT NULL,
  title TEXT NOT NULL,
  message TEXT,
  data JSONB DEFAULT '{}',
  read_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_client ON alerts(client_id, created_at DESC);
CREATE INDEX idx_alerts_unread ON alerts(client_id) WHERE read_at IS NULL;

-- ============================================
-- WEBHOOKS
-- ============================================

CREATE TABLE webhooks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  events TEXT[] NOT NULL, -- ['phase_change', 'anomaly']
  secret TEXT NOT NULL, -- For signature verification
  active BOOLEAN DEFAULT true,
  last_triggered_at TIMESTAMPTZ,
  failure_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_webhooks_client ON webhooks(client_id);

-- ============================================
-- CACHED SIGNALS (for fast reads)
-- ============================================

CREATE TABLE signals_cache (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source TEXT NOT NULL, -- 'SEC', 'FRED', 'CDC'
  signal_type TEXT NOT NULL, -- 'raw', 'fused', 'analysis'
  data JSONB NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_source ON signals_cache(source, signal_type);
CREATE INDEX idx_signals_expires ON signals_cache(expires_at);

-- Auto-delete expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
  DELETE FROM signals_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- TIER LIMITS (reference table)
-- ============================================

CREATE TABLE tier_limits (
  tier client_tier PRIMARY KEY,
  max_signal_tokens INT NOT NULL,
  max_fusion_tokens INT NOT NULL,
  max_analysis_tokens INT NOT NULL,
  max_total_tokens INT NOT NULL,
  max_api_keys INT NOT NULL,
  max_webhooks INT NOT NULL
);

INSERT INTO tier_limits VALUES
  ('free', 10000, 5000, 1000, 15000, 2, 1),
  ('pro', 500000, 250000, 100000, 750000, 10, 5),
  ('enterprise', 10000000, 5000000, 2000000, 15000000, 100, 50),
  ('government', 2147483647, 2147483647, 2147483647, 2147483647, 1000, 500);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhooks ENABLE ROW LEVEL SECURITY;

-- Clients: users can only see their own client record
CREATE POLICY "Users can view own client" ON clients
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own client" ON clients
  FOR UPDATE USING (auth.uid() = user_id);

-- API Keys: users can manage their own keys
CREATE POLICY "Users can view own api_keys" ON api_keys
  FOR SELECT USING (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

CREATE POLICY "Users can insert own api_keys" ON api_keys
  FOR INSERT WITH CHECK (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

CREATE POLICY "Users can update own api_keys" ON api_keys
  FOR UPDATE USING (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

-- Usage: users can view their own usage
CREATE POLICY "Users can view own usage" ON usage_records
  FOR SELECT USING (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

-- Alerts: users can view/update their own alerts
CREATE POLICY "Users can view own alerts" ON alerts
  FOR SELECT USING (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

CREATE POLICY "Users can update own alerts" ON alerts
  FOR UPDATE USING (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

-- Webhooks: users can manage their own webhooks
CREATE POLICY "Users can manage own webhooks" ON webhooks
  FOR ALL USING (
    client_id IN (SELECT id FROM clients WHERE user_id = auth.uid())
  );

-- Service role bypasses RLS for API operations
-- (Edge functions use service role)

-- ============================================
-- FUNCTIONS
-- ============================================

-- Generate API key (call from Edge Function)
CREATE OR REPLACE FUNCTION generate_api_key(p_client_id UUID, p_name TEXT DEFAULT 'Default')
RETURNS TABLE(key TEXT, key_id UUID) AS $$
DECLARE
  v_key TEXT;
  v_key_id UUID;
  v_hash TEXT;
  v_prefix TEXT;
BEGIN
  -- Generate random key
  v_key := 'lf_live_' || encode(gen_random_bytes(24), 'base64');
  v_key := replace(replace(replace(v_key, '+', ''), '/', ''), '=', '');

  -- Hash for storage
  v_hash := encode(digest(v_key, 'sha256'), 'hex');
  v_prefix := substring(v_key, 1, 12) || '...';

  -- Insert
  INSERT INTO api_keys (client_id, key_hash, key_prefix, name)
  VALUES (p_client_id, v_hash, v_prefix, p_name)
  RETURNING id INTO v_key_id;

  RETURN QUERY SELECT v_key, v_key_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Validate API key (call from Edge Function)
CREATE OR REPLACE FUNCTION validate_api_key(p_key TEXT)
RETURNS TABLE(
  client_id UUID,
  client_tier client_tier,
  key_id UUID
) AS $$
DECLARE
  v_hash TEXT;
BEGIN
  v_hash := encode(digest(p_key, 'sha256'), 'hex');

  RETURN QUERY
  SELECT
    c.id as client_id,
    c.tier as client_tier,
    ak.id as key_id
  FROM api_keys ak
  JOIN clients c ON c.id = ak.client_id
  WHERE ak.key_hash = v_hash
    AND ak.status = 'active'
    AND (ak.expires_at IS NULL OR ak.expires_at > NOW());

  -- Update last used
  UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = v_hash;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Check usage limits
CREATE OR REPLACE FUNCTION check_usage_limit(
  p_client_id UUID,
  p_operation TEXT,
  p_tokens INT
)
RETURNS TABLE(allowed BOOLEAN, reason TEXT, current_usage INT, max_usage INT) AS $$
DECLARE
  v_tier client_tier;
  v_limits RECORD;
  v_current INT;
  v_period TEXT;
BEGIN
  -- Get client tier
  SELECT tier INTO v_tier FROM clients WHERE id = p_client_id;

  -- Get limits
  SELECT * INTO v_limits FROM tier_limits WHERE tier = v_tier;

  -- Get current period usage
  v_period := to_char(NOW(), 'YYYY-MM-01');
  SELECT COALESCE(SUM(total_tokens), 0) INTO v_current
  FROM usage_records
  WHERE client_id = p_client_id AND billing_period = v_period;

  -- Check limit
  IF v_current + p_tokens > v_limits.max_total_tokens THEN
    RETURN QUERY SELECT false, 'Monthly token limit exceeded', v_current, v_limits.max_total_tokens;
  ELSE
    RETURN QUERY SELECT true, NULL::TEXT, v_current, v_limits.max_total_tokens;
  END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Record usage
CREATE OR REPLACE FUNCTION record_usage(
  p_client_id UUID,
  p_api_key_id UUID,
  p_operation TEXT,
  p_signal_tokens INT DEFAULT 0,
  p_fusion_tokens INT DEFAULT 0,
  p_analysis_tokens INT DEFAULT 0,
  p_storage_tokens INT DEFAULT 0,
  p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
  v_id UUID;
BEGIN
  INSERT INTO usage_records (
    client_id, api_key_id, operation, billing_period,
    signal_tokens, fusion_tokens, analysis_tokens, storage_tokens,
    metadata
  ) VALUES (
    p_client_id, p_api_key_id, p_operation, to_char(NOW(), 'YYYY-MM-01'),
    p_signal_tokens, p_fusion_tokens, p_analysis_tokens, p_storage_tokens,
    p_metadata
  )
  RETURNING id INTO v_id;

  RETURN v_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create client on signup (trigger)
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO clients (user_id, name, email)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'name', NEW.email),
    NEW.email
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX idx_clients_user ON clients(user_id);
CREATE INDEX idx_clients_email ON clients(email);

-- ============================================
-- REFRESH MATERIALIZED VIEW (run periodically)
-- ============================================

-- You can set up a cron job in Supabase to run this:
-- SELECT cron.schedule('refresh-usage-summary', '0 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY usage_summary');

COMMENT ON TABLE clients IS 'LatticeForge customer accounts';
COMMENT ON TABLE api_keys IS 'API keys for authentication';
COMMENT ON TABLE usage_records IS 'Token usage tracking for billing';
COMMENT ON TABLE alerts IS 'Signal alerts for customers';
COMMENT ON TABLE tier_limits IS 'Usage limits by subscription tier';
